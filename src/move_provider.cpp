#include "../include/move_provider.hpp"
#include "../include/evaluation.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <random>

using U64 = std::uint64_t;

int move_counter = 0;
int extra_moves = 0;

int q_depth = 0;        // current quiescence recursion depth
int q_max_depth = 0;    // deepest quiescence level reached

const int CHECKMATE_SCORE = 999999;
const int DRAW_SCORE = 0;

long long tt_probes = 0;
long long tt_hits = 0;
long long tt_exact_hits = 0;
long long tt_lower_hits = 0;
long long tt_upper_hits = 0;
long long tt_stores = 0;
long long tt_cutoffs = 0;

long long tt_static_eval_hits = 0;
long long tt_static_eval_stores = 0;

static constexpr int MAX_PLY = 128;
static constexpr int MAX_MOVES = 256;

// Victim/Attacker piece values for MVV-LVA
static constexpr int MVV_LVA_VALUES[7] = {
    100,   // PAWN
    300,   // KNIGHT
    300,   // BISHOP
    500,   // ROOK
    900,   // QUEEN
    0,     // KING
    0      // NONE (no victim)
};

static constexpr int SEE_VALUES[7] = {
    100,     // PAWN
    300,     // KNIGHT
    300,     // BISHOP
    500,     // ROOK
    900,     // QUEEN
    20000,   // KING
    0        // NONE
};

static int MVV_LVA_TABLE[7][7];

// Killer moves are quiet moves that previously caused beta cutoffs.
static chess::Move killer_moves[MAX_PLY][2];

// History heuristic: quiet move score indexed by from-square and to-square.
static int history_scores[64][64];

// we do this once instead of multiplying these values every time we evaluate a move
static void init_mvv_lva()
{
    for (int v = 0; v < 7; v++)
    {
        for (int a = 0; a < 7; a++)
        {
            // * by 10 to have bigger diff between capture scores
            MVV_LVA_TABLE[v][a] = MVV_LVA_VALUES[v] * 10 - MVV_LVA_VALUES[a];
        }
    }
}

namespace chessengine
{
    using namespace chess;

    struct SearchResult
    {
        int score;
        Move bestMove;
    };

    struct SearchStack
    {
        NNUEAccumulator accumulator;
        Move currentMove = Move(Move::NO_MOVE);
    };

    enum SCORE_TYPE
    {
        EXACT,
        LOWERBOUND, // fails high -- cut node
        UPPERBOUND  // fails low -- all nodes
    };

    struct TTEntry
    {
        U64 key = 0;                          // zobrist key of the position
        int depth = -1;                       // search depth
        int score = 0;                        // search eval
        SCORE_TYPE bound = EXACT;             // what kind of score this is
        Move bestMove = Move(Move::NO_MOVE);  // best move at this position

        int static_eval = 0;                  // white POV static evaluation
        bool has_static_eval = false;
    };

    // table is aprox 48MB+ now because TTEntry has static eval too
    std::vector<TTEntry> TT(1 << 21);

    // all 1s
    std::size_t mask = TT.size() - 1;

    static bool is_no_move(const Move& move)
    {
        return move.move() == Move::NO_MOVE;
    }

    static bool is_quiet_move(const Board& board, const Move& move)
    {
        return !board.isCapture(move) &&
               move.typeOf() == Move::NORMAL;
    }

    static bool is_draw_by_simple_rules(const Board& board)
    {
        /*
            Avoid board.isGameOver() here.

            board.isGameOver() can internally generate legal moves.
            In search we already generate legal moves immediately after this,
            so using board.isGameOver() causes duplicated work.
        */

        if (board.isHalfMoveDraw())
        {
            return true;
        }

        if (board.isRepetition(1))
        {
            return true;
        }

        if (board.isInsufficientMaterial())
        {
            return true;
        }

        return false;
    }

    static void clear_killer_moves()
    {
        for (int ply = 0; ply < MAX_PLY; ply++)
        {
            killer_moves[ply][0] = Move(Move::NO_MOVE);
            killer_moves[ply][1] = Move(Move::NO_MOVE);
        }
    }

    static void update_killer_and_history(
        const Board& board,
        const Move& move,
        int depth,
        int ply
    )
    {
        if (ply < 0 || ply >= MAX_PLY)
        {
            return;
        }

        if (!is_quiet_move(board, move))
        {
            return;
        }

        if (killer_moves[ply][0] != move)
        {
            killer_moves[ply][1] = killer_moves[ply][0];
            killer_moves[ply][0] = move;
        }

        int from = move.from().index();
        int to = move.to().index();

        int bonus = depth * depth;

        history_scores[from][to] += bonus;

        if (history_scores[from][to] > 1000000)
        {
            history_scores[from][to] = 1000000;
        }
    }

    static bool move_requires_full_nnue_refresh(
        const Board& board,
        const Move& move
    )
    {
        Piece moved_piece = board.at(move.from());

        if (moved_piece == Piece::NONE)
        {
            return true;
        }

        /*
            HalfKP feature indices depend on the king square.

            If the king moves, all HalfKP indices for that side change,
            so a small add/remove update is not enough.
            In that case we refresh the accumulator fully after makeMove().
        */
        if (moved_piece.type() == PieceType::KING)
        {
            return true;
        }

        /*
            Castling moves both the king and the rook.
            Since the king square changes, the affected accumulator must be rebuilt.
        */
        if (move.typeOf() == Move::CASTLING)
        {
            return true;
        }

        return false;
    }

    static void prepare_child_accumulator(
        Board& board,
        const Move& move,
        const NNUEAccumulator& parent_accumulator,
        NNUEAccumulator& child_accumulator
    )
    {
        bool needs_full_refresh =
            move_requires_full_nnue_refresh(board, move);

        if (!needs_full_refresh)
        {
            /*
                Try the fast path:

                child accumulator = parent accumulator
                remove moved piece old feature
                add moved piece new feature
                remove captured piece if needed

                This is the actual NNUE incremental update.
            */
            bool updated = updateNNUEAccumulatorAfterMove(
                board,
                move,
                parent_accumulator,
                child_accumulator
            );

            if (updated)
            {
                return;
            }
        }

        /*
            If we cannot safely update incrementally, mark it invalid.
            After board.makeMove(move), we will fully refresh it from the new board.
        */
        child_accumulator.valid = false;
    }

    static void refresh_child_accumulator_after_make_if_needed(
        Board& board,
        NNUEAccumulator& child_accumulator
    )
    {
        if (!child_accumulator.valid)
        {
            refreshNNUEAccumulator(board, child_accumulator);
        }
    }

    void store_position(
        const Board& board,
        int depth,
        int score,
        SCORE_TYPE bound,
        Move bestMove
    )
    {
        U64 key = board.hash();
        U64 index = key & mask; // this is done as the key is VERY big , but we only need the last 22 (mask size/entry size) bits

        TTEntry& entry = TT[index];

        // preserve static eval if this entry already belongs to the same position
        int old_static_eval = entry.static_eval;
        bool old_has_static_eval = entry.has_static_eval;
        bool same_position = entry.key == key;

        // we must replace the old value in case of a collison in the indexing of the table
        // odds are 0.00005%, but this will be very common, since lots of values will be stored in the table
        // !!!!!!!!!a better implementation for this that considers the age of the entry will be added at a later date, instead of replacing random values!!!!!!!!!!!!!!!

        // hash collisons can also happen, but the odds are negligible (1 / 2^64) for a 64 bit key

        // we will also replace the same postion searched if its depth is higher that the old one (we have a deeper evaluation)
        if (entry.key != key || entry.depth < depth)
        {
            entry.key = key;
            entry.depth = depth;
            entry.score = score;
            entry.bound = bound;
            entry.bestMove = bestMove;

            if (same_position)
            {
                entry.static_eval = old_static_eval;
                entry.has_static_eval = old_has_static_eval;
            }
            else
            {
                entry.static_eval = 0;
                entry.has_static_eval = false;
            }

            tt_stores++;
        }
    }

    void store_static_eval(
        const Board& board,
        int static_eval
    )
    {
        U64 key = board.hash();
        U64 index = key & mask;

        TTEntry& entry = TT[index];

        /*
            We store static eval only if:
            1. The slot is empty/uninitialized, or
            2. The slot already belongs to this exact position.

            This avoids replacing a useful deeper search entry just to cache
            a static eval for a colliding position.
        */
        if (entry.depth < 0 || entry.key == key)
        {
            entry.key = key;
            entry.static_eval = static_eval;
            entry.has_static_eval = true;
            tt_static_eval_stores++;
        }
    }

    TTEntry* query_table(const Board& board)
    {
        tt_probes++;

        U64 key = board.hash();
        U64 index = key & mask;

        TTEntry& entry = TT[index];

        if (entry.key == key)
        {
            tt_hits++;
            return &entry;
        }

        return nullptr;
    }

    static int get_static_eval_white_pov(
        Board& board,
        const NNUEAccumulator& accumulator,
        TTEntry* existing_entry
    )
    {
        if (existing_entry != nullptr && existing_entry->has_static_eval)
        {
            tt_static_eval_hits++;
            return existing_entry->static_eval;
        }

        int static_eval =
            evaluatePositionFromAccumulator(board, accumulator);

        store_static_eval(board, static_eval);

        return static_eval;
    }

    static Bitboard attacksToSquare(
        const Board& board,
        Square sq,
        Bitboard occupancy
    )
    {
        Bitboard attackers{};

        // add non-sliding attackers to the target square
        // we must add them for each color
        attackers |= attacks::pawn(Color::WHITE, sq)
                   & board.pieces(PieceType::PAWN, Color::BLACK);

        attackers |= attacks::pawn(Color::BLACK, sq)
                   & board.pieces(PieceType::PAWN, Color::WHITE);

        attackers |= attacks::knight(sq)
                   & (
                        board.pieces(PieceType::KNIGHT, Color::WHITE)
                      | board.pieces(PieceType::KNIGHT, Color::BLACK)
                     );

        attackers |= attacks::king(sq)
                   & (
                        board.pieces(PieceType::KING, Color::WHITE)
                      | board.pieces(PieceType::KING, Color::BLACK)
                     );

        // sliding pieces

        // diagonals
        const Bitboard bishops_queens =
            (
                board.pieces(PieceType::BISHOP, Color::WHITE)
              | board.pieces(PieceType::BISHOP, Color::BLACK)
              | board.pieces(PieceType::QUEEN, Color::WHITE)
              | board.pieces(PieceType::QUEEN, Color::BLACK)
            );

        // horiz/vert attacks
        const Bitboard rooks_queens =
            (
                board.pieces(PieceType::ROOK, Color::WHITE)
              | board.pieces(PieceType::ROOK, Color::BLACK)
              | board.pieces(PieceType::QUEEN, Color::WHITE)
              | board.pieces(PieceType::QUEEN, Color::BLACK)
            );

        attackers |= attacks::bishop(sq, occupancy) & bishops_queens;
        attackers |= attacks::rook(sq, occupancy) & rooks_queens;

        return attackers;
    }

    // returns a score of how good a capture is, based on the following exchanges that result from it
    // is used for ordering
    int SEE(Board& board, Square& toSq, Square& fromSq, Piece& target)
    {
        // gain is used to track how the advantage/disadvantage changes after each capture in the exchange
        int gain[32];
        int depth = 0;

        Bitboard occupancy = board.occ();

        gain[0] = SEE_VALUES[static_cast<int>(target.type())];

        // from square is removed from the occ board
        occupancy ^= Bitboard::fromSquare(fromSq);

        Color side;

        // get enemy color (reverse)
        if (board.sideToMove() == Color::WHITE)
        {
            side = Color::BLACK;
        }
        else
        {
            side = Color::WHITE;
        }

        while (true)
        {
            Bitboard att = attacksToSquare(board, toSq, occupancy);

            Square from{};
            PieceType piece_type = PieceType::NONE;
            bool found = false;

            const PieceType order[] = {
                // piece worth in order , we try to capture with the least valuable piece first
                PieceType::PAWN,
                PieceType::KNIGHT,
                PieceType::BISHOP,
                PieceType::ROOK,
                PieceType::QUEEN,
                PieceType::KING
            };

            for (PieceType candidate : order)
            {
                Bitboard bb = att & board.pieces(candidate, side);

                // we must check if the bitboard stil has 1s(pieces) left (unchecked)
                if (bb)
                {
                    from = bb.pop();
                    piece_type = candidate;
                    found = true;
                    break;
                }
            }

            // no more pieces
            if (!found)
            {
                break;
            }

            // add result of SEE from current capture to gain array
            ++depth;
            gain[depth] = SEE_VALUES[static_cast<int>(piece_type)] - gain[depth - 1];

            // if the best result until now is negative, there's no need to continue
            // example: queen takes pawn, no need to further evaulate
            if (std::max(-gain[depth - 1], gain[depth]) < 0)
            {
                break;
            }

            // removing the current attacker from the occupancy bitboard as we already proccesed it
            occupancy ^= Bitboard::fromSquare(from);

            // get enemy color (reverse)
            if (side == Color::WHITE)
            {
                side = Color::BLACK;
            }
            else
            {
                side = Color::WHITE;
            }

            // should theoretically be impossible, but just in case, a failsafe was added to avoid any "explosions"
            if (depth >= 30)
            {
                break;
            }
        }

        while (depth)
        {
            // we check gain for each capture and if continuing wiht capuring is not beneficial, we dont "do" the capture
            // we compare with negative gain for prev move beacuse perpective flips
            gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
            --depth;
        }

        return gain[0];
    }

    int MVVLVA(Board& board, Move& move)
    {
        if (!board.isCapture(move) || move.typeOf() == Move::CASTLING)
        {
            return 0;
        }

        Piece victim = board.at(move.to());
        Piece attacker = board.at(move.from());

        int v = static_cast<int>(victim.type());
        int a = static_cast<int>(attacker.type());

        // VALID RANGE = 0..6
        if (v < 0 || v > 6 || a < 0 || a > 6)
        {
            return 0;
        }

        return MVV_LVA_TABLE[v][a];
    }

    static int score_move(
        Board& board,
        Move& move,
        const Move& ttMove,
        int ply
    )
    {
        if (!is_no_move(ttMove) && move == ttMove)
        {
            return 10000000;
        }

        int score = 0;

        if (board.isCapture(move) && move.typeOf() != Move::CASTLING)
        {
            Piece victim = board.at(move.to());

            if (move.typeOf() == Move::ENPASSANT)
            {
                Color victim_color;

                if (board.sideToMove() == Color::WHITE)
                {
                    victim_color = Color::BLACK;
                }
                else
                {
                    victim_color = Color::WHITE;
                }

                victim = Piece(PieceType::PAWN, victim_color);
            }

            Square to = move.to();
            Square from = move.from();

            int see = SEE(board, to, from, victim);

            score = 1000000 + MVVLVA(board, move) + see;

            return score;
        }

        if (ply >= 0 && ply < MAX_PLY)
        {
            if (move == killer_moves[ply][0])
            {
                return 900000;
            }

            if (move == killer_moves[ply][1])
            {
                return 800000;
            }
        }

        int from = move.from().index();
        int to = move.to().index();

        score += history_scores[from][to];

        return score;
    }

    int quiesce(
        int alpha,
        int beta,
        Board& board,
        int color,
        SearchStack* stack,
        int ply
    )
    {
        q_depth++;

        if (q_depth > q_max_depth)
        {
            q_max_depth = q_depth; // for debugging
        }

        if (ply >= MAX_PLY - 1)
        {
            int static_eval =
                evaluatePositionFromAccumulator(board, stack[ply].accumulator);

            q_depth--;
            return color * static_eval;
        }

        if (is_draw_by_simple_rules(board))
        {
            q_depth--;
            return DRAW_SCORE;
        }

        Movelist rawMoves;
        movegen::legalmoves<movegen::MoveGenType::ALL>(rawMoves, board);

        // checkmate or stalemate
        if (rawMoves.empty())
        {
            if (board.inCheck())
            {
                q_depth--;
                return -CHECKMATE_SCORE;
            }
            else
            {
                q_depth--;
                return DRAW_SCORE; // fail soft
            }
        }

        int originalAlpha = alpha;

        TTEntry* entry = query_table(board);
        Move ttMove = Move(Move::NO_MOVE);

        if (entry != nullptr)
        {
            ttMove = entry->bestMove;

            if (entry->depth >= 0)
            {
                if (entry->bound == EXACT)
                {
                    tt_exact_hits++;
                    q_depth--;
                    return entry->score;
                }

                if (entry->bound == LOWERBOUND && entry->score >= beta)
                {
                    tt_lower_hits++;
                    tt_cutoffs++;
                    q_depth--;
                    return entry->score;
                }

                if (entry->bound == UPPERBOUND && entry->score <= alpha)
                {
                    tt_upper_hits++;
                    tt_cutoffs++;
                    q_depth--;
                    return entry->score;
                }
            }
        }

        // evaluation of the current position, without making any captures
        int static_eval_white_pov =
            get_static_eval_white_pov(board, stack[ply].accumulator, entry);

        int stand_pat = color * static_eval_white_pov;

        // failsafe
        if (q_depth > 24)
        {
            q_depth--;
            return stand_pat;
        }

        if (stand_pat >= beta)
        {
            store_position(board, 0, stand_pat, LOWERBOUND, Move(Move::NO_MOVE));
            q_depth--;
            return stand_pat;
        }

        if (stand_pat > alpha)
        {
            alpha = stand_pat;
        }

        struct ScoredMove
        {
            int score;
            Move move;
        };

        ScoredMove scored[MAX_MOVES];
        int count = 0;

        for (Move m : rawMoves)
        {
            int score = 0;
            bool givesCheck = false;
            bool losingCapture = false;

            if (board.givesCheck(m) != CheckType::NO_CHECK)
            {
                score += 800; // arbitrary value chosen for checks -- almost a full pawn
                givesCheck = true;
            }

            // avoid fake castle captures -- castling is king "captures" rook
            if (board.isCapture(m) && m.typeOf() != Move::CASTLING)
            {
                score += MVVLVA(board, m);

                // enpasssant capture does not 'attack' a square that is occupied by a piece
                Piece victim = board.at(m.to());

                if (m.typeOf() == Move::ENPASSANT)
                {
                    Color victim_color;

                    if (board.sideToMove() == Color::WHITE)
                    {
                        victim_color = Color::BLACK;
                    }
                    else
                    {
                        victim_color = Color::WHITE;
                    }

                    victim = Piece(PieceType::PAWN, victim_color);
                }

                Square to = m.to();
                Square from = m.from();

                int see = SEE(board, to, from, victim);

                score += see;

                if (see < 0)
                {
                    losingCapture = true;
                }
            }

            // skip losing captures if they do not give check
            if (losingCapture && !givesCheck)
            {
                continue;
            }

            // if it's a capture or check then evaluate further
            if ((board.isCapture(m) && m.typeOf() != Move::CASTLING) || givesCheck)
            {
                if (!is_no_move(ttMove) && m == ttMove)
                {
                    score += 10000000;
                }

                if (count < MAX_MOVES)
                {
                    scored[count++] = {score, m};
                }
            }
        }

        if (count == 0)
        {
            SCORE_TYPE bound_type;

            if (alpha <= originalAlpha)
            {
                bound_type = UPPERBOUND;
            }
            else
            {
                bound_type = EXACT;
            }

            store_position(board, 0, alpha, bound_type, Move(Move::NO_MOVE));

            q_depth--;
            return alpha;
        }

        // insertion sort (fastest for small lists -- faster than quicksor for example)
        for (int i = 1; i < count; i++)
        {
            ScoredMove key = scored[i];
            int j = i - 1;

            while (j >= 0 && scored[j].score < key.score)
            {
                scored[j + 1] = scored[j];
                j--;
            }

            scored[j + 1] = key;
        }

        Move bestMove = Move(Move::NO_MOVE);

        for (int i = 0; i < count; i++)
        {
            Move move = scored[i].move;

            prepare_child_accumulator(
                board,
                move,
                stack[ply].accumulator,
                stack[ply + 1].accumulator
            );

            stack[ply + 1].currentMove = move;

            board.makeMove(move);

            refresh_child_accumulator_after_make_if_needed(
                board,
                stack[ply + 1].accumulator
            );

            int score = -quiesce(
                -beta,
                -alpha,
                board,
                -color,
                stack,
                ply + 1
            );

            board.unmakeMove(move);

            if (score >= beta)
            {
                store_position(board, 0, score, LOWERBOUND, move);
                q_depth--;
                return score;
            }

            if (score > alpha)
            {
                alpha = score;
                bestMove = move;
            }
        }

        SCORE_TYPE bound_type;

        if (alpha <= originalAlpha)
        {
            bound_type = UPPERBOUND;
        }
        else
        {
            bound_type = EXACT;
        }

        store_position(board, 0, alpha, bound_type, bestMove);

        q_depth--;
        return alpha;
    }

    SearchResult negaMax(
        int depth,
        int alpha,
        int beta,
        Board& board,
        int color,
        SearchStack* stack,
        int ply
    )
    {
        if (ply >= MAX_PLY - 1)
        {
            int static_eval =
                evaluatePositionFromAccumulator(board, stack[ply].accumulator);

            return {color * static_eval, Move(Move::NO_MOVE)};
        }

        if (is_draw_by_simple_rules(board))
        {
            return {DRAW_SCORE, Move(Move::NO_MOVE)};
        }

        Movelist rawMoves;
        movegen::legalmoves<movegen::MoveGenType::ALL>(rawMoves, board);

        // checkmate or stalemate
        if (rawMoves.empty())
        {
            // checkmate
            if (board.inCheck())
            {
                // depth added so engine prefers faster mates
                return {-CHECKMATE_SCORE - depth, Move(Move::NO_MOVE)};
            }
            else
            {
                // Stalemate
                return {DRAW_SCORE, Move(Move::NO_MOVE)};
            }
        }

        if (depth == 0)
        {
            int qscore = quiesce(
                alpha,
                beta,
                board,
                color,
                stack,
                ply
            );

            return {qscore, Move(Move::NO_MOVE)};
        }

        int originalAlpha = alpha;

        TTEntry* entry = query_table(board);
        Move ttMove = Move(Move::NO_MOVE);

        if (entry != nullptr)
        {
            ttMove = entry->bestMove;

            if (entry->depth >= depth)
            {
                if (entry->bound == EXACT)
                {
                    tt_exact_hits++;
                    return {entry->score, entry->bestMove};
                }

                if (entry->bound == LOWERBOUND && entry->score >= beta)
                {
                    tt_lower_hits++;
                    tt_cutoffs++;
                    return {entry->score, entry->bestMove};
                }

                if (entry->bound == UPPERBOUND && entry->score <= alpha)
                {
                    tt_upper_hits++;
                    tt_cutoffs++;
                    return {entry->score, entry->bestMove};
                }
            }
        }

        struct ScoredMove
        {
            int score;
            Move move;
        }; // faster than vectors

        ScoredMove scored[MAX_MOVES];
        int count = 0;

        for (Move m : rawMoves)
        {
            int score = score_move(
                board,
                m,
                ttMove,
                ply
            );

            if (count < MAX_MOVES)
            {
                scored[count] = {score, m};
                count++;
            }
        }

        // insertion sort -- efficient because the list is small
        for (int i = 1; i < count; i++)
        {
            ScoredMove key = scored[i];
            int j = i - 1;

            while (j >= 0 && scored[j].score < key.score)
            {
                scored[j + 1] = scored[j];
                j--;
            }

            scored[j + 1] = key;
        }

        SearchResult best = {-10000000, Move(Move::NO_MOVE)};

        for (int i = 0; i < count; i++)
        {
            Move move = scored[i].move;
            move_counter++;

            prepare_child_accumulator(
                board,
                move,
                stack[ply].accumulator,
                stack[ply + 1].accumulator
            );

            stack[ply + 1].currentMove = move;

            board.makeMove(move);

            refresh_child_accumulator_after_make_if_needed(
                board,
                stack[ply + 1].accumulator
            );

            SearchResult child = negaMax(
                depth - 1,
                -beta,
                -alpha,
                board,
                -color,
                stack,
                ply + 1
            );

            board.unmakeMove(move);

            int score = -child.score;

            if (score > best.score)
            {
                best.score = score;
                best.bestMove = move;
            }

            if (score > alpha)
            {
                alpha = score;
            }

            if (alpha >= beta)
            {
                update_killer_and_history(
                    board,
                    move,
                    depth,
                    ply
                );

                store_position(board, depth, alpha, LOWERBOUND, move);
                return {alpha, move};
            }
        }

        SCORE_TYPE bound_type;

        if (best.score <= originalAlpha)
        {
            bound_type = UPPERBOUND;
        }
        else
        {
            bound_type = EXACT;
        }

        store_position(
            board,
            depth,
            best.score,
            bound_type,
            best.bestMove
        );

        return best;
    }

    Move findBestMove(int depth, Board& board)
    {
        int color = -1; // black to move

        if (board.sideToMove() == Color::WHITE)
        {
            color = 1;
        }

        move_counter = 0;
        extra_moves = 0;
        q_depth = 0;
        q_max_depth = 0;

        tt_probes = 0;
        tt_hits = 0;
        tt_exact_hits = 0;
        tt_lower_hits = 0;
        tt_upper_hits = 0;
        tt_stores = 0;
        tt_cutoffs = 0;
        tt_static_eval_hits = 0;
        tt_static_eval_stores = 0;

        int alpha = -10000000;
        int beta = +10000000;

        static bool initialize_mvv_lva_table = false;

        if (!initialize_mvv_lva_table)
        {
            init_mvv_lva();
            initialize_mvv_lva_table = true;
        }

        clear_killer_moves();

        /*
            Root accumulator.

            This is built once from the root board.
            After this, child positions try to update it incrementally.
        */
        SearchStack stack[MAX_PLY];

        refreshNNUEAccumulator(board, stack[0].accumulator);
        stack[0].currentMove = Move(Move::NO_MOVE);

        // ------------------------------------------------------
        // BENCHMARK START
        // ------------------------------------------------------
        auto start_time = std::chrono::steady_clock::now();
        // ------------------------------------------------------

        std::cout << evaluatePositionFromAccumulator(
            board,
            stack[0].accumulator
        ) << std::endl;

        SearchResult result = negaMax(
            depth,
            alpha,
            beta,
            board,
            color,
            stack,
            0
        );

        // ------------------------------------------------------
        // BENCHMARK END
        // ------------------------------------------------------
        auto end_time = std::chrono::steady_clock::now();

        long long ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time
            ).count();

        double seconds = ms / 1000.0;
        double nps = seconds > 0 ? move_counter / seconds : 0;
        // ------------------------------------------------------

        // heavily slows it down -- only use when testing

        // Fallback
        // if (result.bestMove == Move())
        // {
        //     Movelist moves;
        //     movegen::legalmoves<movegen::MoveGenType::ALL>(moves, board);

        //     if (!moves.empty())
        //     {
        //         std::cout << "WARNING: Search returned no best move, using fallback.\n";
        //         result.bestMove = moves[0];
        //     }
        //     else
        //     {
        //         std::cout << "No legal moves in findBestMove.\n";
        //     }
        // }

        // ------------------------------------------------------
        // PRINT BENCHMARK PER-MOVE
        // ------------------------------------------------------
        std::cout << "\n=========== ENGINE MOVE BENCHMARK ===========\n";
        std::cout << "Side to move: "
                  << (color == 1 ? "White" : "Black") << "\n";
        std::cout << "Search depth: " << depth << "\n";
        std::cout << "Best move: " << uci::moveToUci(result.bestMove) << "\n";
        std::cout << "Best score: " << result.score << "\n";
        std::cout << "Nodes searched: " << move_counter << "\n";
        std::cout << "Time: " << ms << " ms\n";
        std::cout << "NPS: " << static_cast<long long>(nps) << "\n";
        std::cout << "Max quiescence depth: " << q_max_depth << "\n";

        double tt_hit_rate =
            tt_probes > 0 ? (100.0 * tt_hits / tt_probes) : 0.0;

        double tt_static_eval_hit_rate =
            tt_static_eval_hits + tt_static_eval_stores > 0
                ? (100.0 * tt_static_eval_hits /
                   (tt_static_eval_hits + tt_static_eval_stores))
                : 0.0;

        std::cout << "TT probes: " << tt_probes << "\n";
        std::cout << "TT hits: " << tt_hits << "\n";
        std::cout << "TT hit rate: " << tt_hit_rate << "%\n";
        std::cout << "TT exact hits: " << tt_exact_hits << "\n";
        std::cout << "TT lower hits: " << tt_lower_hits << "\n";
        std::cout << "TT upper hits: " << tt_upper_hits << "\n";
        std::cout << "TT cutoffs: " << tt_cutoffs << "\n";
        std::cout << "TT stores: " << tt_stores << "\n";
        std::cout << "TT static eval hits: " << tt_static_eval_hits << "\n";
        std::cout << "TT static eval stores: " << tt_static_eval_stores << "\n";
        std::cout << "TT static eval hit rate: " << tt_static_eval_hit_rate << "%\n";

        std::cout << "==============================================\n\n";

        return result.bestMove;
    }

    Move getNextMove(Board& board)
    {
        Movelist moves;
        movegen::legalmoves<movegen::MoveGenType::ALL>(moves, board);

        if (moves.empty())
        {
            std::cout << "No legal moves available!\n";
            return Move(Move::NO_MOVE);
        }

        // int eval = evaluatePosition(board);
        // std::cout << "Evaluation (White - Black): " << eval << " centipawns\n";

        Move best = findBestMove(8, board);

        return best;
    }
}