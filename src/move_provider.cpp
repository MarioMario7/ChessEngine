#include "../include/move_provider.hpp"
#include "../include/evaluation.hpp"
#include <random>
#include <cctype>
#include <iostream>



int move_counter = 0;
int extra_moves = 0;

int q_depth = 0;        // current quiescence recursion depth
int q_max_depth = 0;    // deepest quiescence level reached

namespace chessengine {
    using namespace chess;

    struct SearchResult
    {
        int score;
        Move bestMove;
    };


        int quiesce(int alpha, int beta, Board& board, int color)
    {
        q_depth++;
        if (q_depth > q_max_depth)
            q_max_depth = q_depth;

        int stand_pat = color * evaluatePosition(board);

        if (stand_pat >= beta) {
            q_depth--;
            return stand_pat;
        }

        if (stand_pat > alpha)
            alpha = stand_pat;

        Movelist moves;
        movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, board);

        if (moves.empty()) {
            q_depth--;
            return stand_pat;
        }

        for (Move move : moves)
        {
            board.makeMove(move);
            int score = -quiesce(-beta, -alpha, board, -color);
            board.unmakeMove(move);

            if (score >= beta) {
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

        Movelist moves;
        movegen::legalmoves<movegen::MoveGenType::ALL>(moves, board);

        // No legal moves → checkmate or stalemate
        if (moves.empty())
        {
            int eval = color * evaluatePosition(board);
            return { eval, Move() };
        }

        SearchResult best = { -10000000, Move() };

        for (Move move : moves)
        {
            move_counter++;

            board.makeMove(move);

            SearchResult child = negaMax(depth - 1, -beta, -alpha, board, -color);

            board.unmakeMove(move);

            int score = -child.score;

            // Beta cutoff (fail-soft)
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

        SearchResult result = negaMax(depth, alpha, beta, board, color);

        std::cout << "We have searched " << move_counter << " positions\n";
        std::cout << "Max Quiescence Depth: " << q_max_depth << "\n";

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


        Move best = findBestMove(5, board);
       // std::cout << "Best move (depth 3): " << uci::moveToUci(best) << "\n";

        return best;
    }
}


