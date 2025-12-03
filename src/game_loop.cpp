#include "../include/game_loop.hpp"
#include "../include/move_provider.hpp"
#include <iostream>
#include <chrono>




namespace chessengine {
    using namespace chess;

    void playGame(Board& board) {
        while (true) {
            auto move = getNextMove(board);

            // maybe check if the move is valid before making it

            board.makeMove(move);

            std::cout <<  "Move is " << uci::moveToUci(move) << std::endl;

            auto [reason, result] = board.isGameOver();
            if (reason != GameResultReason::NONE) {
                std::cout << "Game over!\n";

                switch (reason) {
                    case GameResultReason::CHECKMATE: std::cout << "Reason: Checkmate\n"; break;
                    case GameResultReason::STALEMATE: std::cout << "Reason: Stalemate\n"; break;
                    case GameResultReason::THREEFOLD_REPETITION: std::cout << "Reason: Threefold repetition\n"; break;
                    case GameResultReason::FIFTY_MOVE_RULE: std::cout << "Reason: 50-move rule\n"; break;
                    case GameResultReason::INSUFFICIENT_MATERIAL: std::cout << "Reason: Insufficient material\n"; break;
                    default: std::cout << "Other reason\n"; break;
                }

                // this is possibly incorrect for now

                std::cout << "Result: ";
                if (result == GameResult::WIN) std::cout << "Black wins\n";
                else if (result == GameResult::LOSE) std::cout << "White wins\n";
                else if (result == GameResult::DRAW) std::cout << "Draw\n";

                std::cout << "Final FEN: " << board.getFen() << "\n";
                break;
            }
        }
    }
}
