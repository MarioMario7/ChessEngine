#include "../include/move_provider.hpp"
#include "../include/evaluation.hpp"
#include <random>
#include <cctype>
#include <iostream>





namespace chessengine {
    using namespace chess;

    struct SearchResult
    {
        int score;
        Move bestMove;
    };

    SearchResult negaMax(int depth, Board& board, int color)
    {
        if (depth == 0)
            return { color * evaluatePosition(board), Move() };

        Movelist moves;
        movegen::legalmoves<movegen::MoveGenType::ALL>(moves, board);

        // if no moves: this side lost/stalemate possibly
        if (moves.empty())
            return { -1000000, Move() };

        SearchResult best = { -10000000, Move() };

        for (Move move : moves)
        {
            board.makeMove(move);

            // get opponent s response (we need to invert the evaluation because the color is switched )
            SearchResult child = negaMax(depth - 1, board, -color);

            board.unmakeMove(move);

            int score = -child.score;

            // in this verison, we only calculate the material so it will only be true for the first move checked
            // when we only check quiet moves or starting positions
            
            if (score > best.score)
            {
                best.score = score;
                best.bestMove = move;
            }
        }

        return best;
    }






    Move findBestMove( int depth, Board& board)
    {
        int color = (board.sideToMove() == Color::WHITE ? +1 : -1);
        SearchResult result = negaMax(depth, board, color);
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


