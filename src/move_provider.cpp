#include "../include/move_provider.hpp"
#include "../include/evaluation.hpp"
#include <random>
#include <cctype>
#include <iostream>
#include <chrono>



int move_counter = 0;
int extra_moves = 0;

int q_depth = 0;        // current quiescence recursion depth
int q_max_depth = 0;    // deepest quiescence level reached

const int CHECKMATE_SCORE = 999999;
const int DRAW_SCORE = 0;



// Victim/Attacker piece values for MVV-LVA 

static constexpr int MVV_LVA_VALUES[7] = {
    100,   // PAWN
    300,   // KNIGHT
    300,   // BISHOP
    500,   // ROOK
    900,   // QUEEN
      0,   // KING
      0    // NONE (no victim)
};

static constexpr int SEE_VALUES[7] = {
    100,   // PAWN
    300,   // KNIGHT
    300,   // BISHOP
    500,   // ROOK
    900,   // QUEEN
  20000,   // KING
      0    // NONE
};

static int MVV_LVA_TABLE[7][7];




// we do this once instead of multiplying these values every time we evaluate a move
static void init_mvv_lva() 
{
    for (int v = 0; v < 7; v++)
    {
        for (int a = 0; a < 7; a++)
        {
            MVV_LVA_TABLE[v][a] = MVV_LVA_VALUES[v] * 10 - MVV_LVA_VALUES[a]; // * by 10 to have bigger diff between capture scores
        }
    }
}


namespace chessengine {
    using namespace chess;

    struct SearchResult
    {
        int score;
        Move bestMove;
    };
        

        static Bitboard attacksToSquare(const Board& board, Square sq, Bitboard occupancy)
        {
            Bitboard attackers{};

            attackers |= attacks::pawn(Color::WHITE, sq) & board.pieces(PieceType::PAWN, Color::BLACK);
            attackers |= attacks::pawn(Color::BLACK, sq) & board.pieces(PieceType::PAWN, Color::WHITE);

            attackers |= attacks::knight(sq) & (board.pieces(PieceType::KNIGHT, Color::WHITE) | board.pieces(PieceType::KNIGHT, Color::BLACK));

            attackers |= attacks::king(sq) & (board.pieces(PieceType::KING, Color::WHITE) | board.pieces(PieceType::KING, Color::BLACK));

            //diagonals
            const Bitboard bishops_queens =
                (
                    board.pieces(PieceType::BISHOP, Color::WHITE) | board.pieces(PieceType::BISHOP, Color::BLACK) |
                    board.pieces(PieceType::QUEEN,  Color::WHITE) | board.pieces(PieceType::QUEEN,  Color::BLACK)
                );

            // horiz/vert attacks
            const Bitboard rooks_queens =
                (
                    board.pieces(PieceType::ROOK,   Color::WHITE) | board.pieces(PieceType::ROOK,   Color::BLACK) |
                    board.pieces(PieceType::QUEEN,  Color::WHITE) | board.pieces(PieceType::QUEEN,  Color::BLACK)
                );

            attackers |= attacks::bishop(sq, occupancy) & bishops_queens;
            attackers |= attacks::rook(sq, occupancy)   & rooks_queens;

            return attackers;
        }

        int SEE(Board& board, Square& toSq, Square& fromSq, Piece& target)
        {

            int gain[32];
            int depth = 0;

            Bitboard occupancy = board.occ();

            gain[0] = SEE_VALUES[static_cast<int>(target.type())];

            occupancy ^= Bitboard::fromSquare(fromSq);

            Color side = (board.sideToMove() == Color::WHITE) ? Color::BLACK : Color::WHITE;

            while (true)
            {
                Bitboard att = attacksToSquare(board, toSq, occupancy);

                Square from{};
                PieceType pt = PieceType::NONE;
                bool found = false;

                const PieceType order[] = {
                    PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                    PieceType::ROOK, PieceType::QUEEN,  PieceType::KING
                };

                for (PieceType cand : order)
                {
                    Bitboard bb = att & board.pieces(cand, side);
                    if (bb)
                    {
                        from = bb.pop();
                        pt = cand;
                        found = true;
                        break;
                    }
                }

                if (!found) break;

                ++depth;
                gain[depth] = SEE_VALUES[static_cast<int>(pt)] - gain[depth - 1];

                if (std::max(-gain[depth - 1], gain[depth]) < 0) break;

                occupancy ^= Bitboard::fromSquare(from);
                side = (side == Color::WHITE) ? Color::BLACK : Color::WHITE;

                if (depth >= 30) break;
            }

            while (depth)
            {
                gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
                --depth;
            }

            return gain[0];
        }

        int MVVLVA(Board& board, Move& move)
        {
            if (!board.isCapture(move) || move.typeOf() == Move::CASTLING)
                return 0;

            Piece victim   = board.at(move.to());
            Piece attacker = board.at(move.from());

            int v = static_cast<int>(victim.type());
            int a = static_cast<int>(attacker.type());

            // VALID RANGE = 0..6
            if (v < 0 || v > 6 || a < 0 || a > 6)
                return 0;

            return MVV_LVA_TABLE[v][a];
        }


        int quiesce(int alpha, int beta, Board& board, int color)
        {

            
            q_depth++;
            if (q_depth > q_max_depth)
                q_max_depth = q_depth; // for debugging

            int stand_pat = color * evaluatePosition(board); // evaluation of the current position, without making any captures

            // WILL BE REMOVED IN FAVOUR OF EXTRA PRUNING AND STATIC EXCHANGE EVAL -- stops at 12 depth
            if (q_depth > 12) 
            {
                q_depth--;
                return stand_pat;
            }

            if (stand_pat >= beta) 
            {
                q_depth--;
                return stand_pat;
            }

            if (stand_pat > alpha)
                alpha = stand_pat;

            Movelist rawMoves;
            movegen::legalmoves<movegen::MoveGenType::ALL>(rawMoves, board);

            if (rawMoves.empty()) 
            {
                q_depth--;
                return stand_pat; // fail soft
            }

            struct ScoredMove { int score; Move move; };
            ScoredMove scored[256];
            int count = 0;

            for (Move m : rawMoves)
            {
                int score = 0;

                if (board.isCapture(m) && m.typeOf() != Move::CASTLING) // avoid fake castle captures -- castling is king "captures" rook
                {
                    score = MVVLVA(board, m);

                    // enpasssant capture does not 'attack' a square that is occupied by a piece
                    Piece victim = board.at(m.to());
                    if (m.typeOf() == Move::ENPASSANT)
                    {
                        victim = Piece(PieceType::PAWN, (board.sideToMove() == Color::WHITE) ? Color::BLACK : Color::WHITE);
                    }


                    Square to = m.to();
                    Square from = m.from();
                    int see = SEE(board, to, from, victim);

                    score = MVVLVA(board, m);

                    score += see;
                }

                board.makeMove(m);
                if (board.inCheck())
                {
                    score += 800; // arbitrary value chosen for checks -- almost a full pawn
                }
                board.unmakeMove(m);

                if ((board.isCapture(m) && m.typeOf() != Move::CASTLING) || board.inCheck()) // if it's a capture or check then evaluate further
                {
                    scored[count++] = { score, m };
                }
            }

            if (count == 0) 
            {
                q_depth--;
                return stand_pat;
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

            for (int i = 0; i < count; i++)
            {
                Move move = scored[i].move;

                board.makeMove(move);
                int score = -quiesce(-beta, -alpha, board, -color);
                board.unmakeMove(move);

                if (score >= beta) 
                {
                    q_depth--;
                    return score;
                }

                if (score > alpha)
                    alpha = score;
            }

            q_depth--;
            return alpha;
        }






        SearchResult negaMax(int depth, int alpha, int beta, Board& board, int color)
        {

            if (depth == 0)
            {   
                int qscore = quiesce(alpha, beta, board, color);
                return { qscore, Move() };
            }

            Movelist rawMoves;
            movegen::legalmoves<movegen::MoveGenType::ALL>(rawMoves, board);

            //checkmate or stalemate
            if (rawMoves.empty())
            {
                // checkmate
                if (board.inCheck())
                {
                    // depth added so engine prefers faster mates
                    return { -CHECKMATE_SCORE - depth, Move() };
                }
                else
                {
                    // Stalemate
                    return { DRAW_SCORE, Move() };
                }
            }

            if (board.isGameOver().second == GameResult::DRAW) // second in the pair is the result, first is reason
            {
                // 3 move repetition/50 move rule/insufficeint material
                return { DRAW_SCORE, Move() };
            }

            struct ScoredMove { int score; Move move; }; // faster than vectors
            ScoredMove scored[256];
            int count = 0;

            for (Move m : rawMoves)
            {
                int score = MVVLVA(board, m);

                if (board.isCapture(m) && m.typeOf() != Move::CASTLING)
                {
                    Piece victim = board.at(m.to());
                    if (m.typeOf() == Move::ENPASSANT)
                    {
                        victim = Piece(PieceType::PAWN, (board.sideToMove() == Color::WHITE) ? Color::BLACK : Color::WHITE);
                    }

                    Square to = m.to();
                    Square from = m.from();
                    int see = SEE(board, to, from, victim);

                    score += see;
                }

                scored[count] = { score, m };
                count++;
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

            SearchResult best = { -10000000, Move() };

            for (int i = 0; i < count; i++)
            {
                Move move = scored[i].move;
                move_counter++;

                board.makeMove(move);
                SearchResult child = negaMax(depth - 1, -beta, -alpha, board, -color);
                board.unmakeMove(move);


                int score = -child.score;

                // beta cutoff 
                if (score >= beta)
                    return { score, move };

                if (score > best.score)
                {
                    best.score = score;
                    best.bestMove = move;
                }

                if (score > alpha)
                    alpha = score;
            }

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

            int alpha = -10000000;
            int beta  = +10000000;

            static bool initialize_mvv_lva_table = false;
            if (!initialize_mvv_lva_table)
            {
                init_mvv_lva();
                initialize_mvv_lva_table = true;
            }

            // ------------------------------------------------------
            // BENCHMARK START
            // ------------------------------------------------------
            auto start_time = std::chrono::steady_clock::now();
            // ------------------------------------------------------

            SearchResult result = negaMax(depth, alpha, beta, board, color);

            // ------------------------------------------------------
            // BENCHMARK END
            // ------------------------------------------------------
            auto end_time = std::chrono::steady_clock::now();
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
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
            std::cout << "NPS: " << (long long)nps << "\n"; 
            std::cout << "Max quiescence depth: " << q_max_depth << "\n";
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
                return Move();
            }

        // int eval = evaluatePosition(board);
            //std::cout << "Evaluation (White - Black): " << eval << " centipawns\n";


            Move best = findBestMove(8, board);

            return best;
        }
}