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

static int MVV_LVA_TABLE[7][7];




// we do this once instead of multiplying these values every time we evaluate a move
static void init_mvv_lva() 
{
    for (int v = 0; v < 7; v++)
    {
        for (int a = 0; a < 7; a++)
        {
            MVV_LVA_TABLE[v][a] = MVV_LVA_VALUES[v] * 10 - MVV_LVA_VALUES[a];
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
                q_max_depth = q_depth;

            int stand_pat = color * evaluatePosition(board); // evaluation of the current position, without making any captures

            // WILL BE REMOVED IN FAVOUR OF EXTRA PRUNING AND STATIC EXCHANGE EVAL
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
                }

                board.makeMove(m);
                if (board.inCheck())
                {
                    score += 800; // arbitraty value chosen for checks -- almost a full pawn
                }
                board.unmakeMove(m);

                if (score > 0) // if it's a capture or check then evaluate further
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
                    return { -999999 - depth, Move() };
                }
                else
                {
                    // Stalemate
                    return { 0, Move() };
                }
            }

            struct ScoredMove { int score; Move move; }; // faster than vectors
            ScoredMove scored[256];
            int count = 0;

            for (Move m : rawMoves)
            {
                scored[count] = { MVVLVA(board, m), m };
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
            int color = (board.sideToMove() == Color::WHITE ? +1 : -1);

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


