#include "../include/uci_interface.hpp"
#include "../include/move_provider.hpp"
#include "../include/game_loop.hpp"

#include <iostream>
#include <sstream>
#include <thread>

namespace chessengine {
    using namespace chess;

    // Process one UCI command
    void handleUciCommand(const std::string& cmd, Board& board)
    {
        std::istringstream iss(cmd);
        std::string token;
        iss >> token;

        // -------- uci --------
        if (token == "uci")
        {
            std::cout << "id name RandomEngine\n";
            std::cout << "id author mario\n";
            std::cout << "uciok\n";
        }

        // -------- isready --------
        else if (token == "isready")
        {
            std::cout << "readyok\n";
        }

        // -------- ucinewgame --------
        else if (token == "ucinewgame")
        {
            board = Board(); // reset board
        }

        // -------- position --------
        else if (token == "position")
        {
            std::string sub;
            iss >> sub;

            // ------ position startpos [moves ...] ------
            if (sub == "startpos")
            {
                board = Board(); // standard start position
                std::string movesKeyword;
                if (iss >> movesKeyword && movesKeyword == "moves")
                {
                    std::string moveStr;
                    while (iss >> moveStr)
                    {
                        Move m = uci::uciToMove(board, moveStr);
                        board.makeMove(m);
                    }
                }
            }

            // ------ position fen <fenstring> [moves ...] ------
            else if (sub == "fen")
            {
                std::string fen;
                std::getline(iss >> std::ws, fen); // FIX: remove whitespace
                board.setFen(fen);

                std::string movesKeyword;
                if (iss >> movesKeyword && movesKeyword == "moves")
                {
                    std::string moveStr;
                    while (iss >> moveStr)
                    {
                        Move m = uci::uciToMove(board, moveStr);
                        board.makeMove(m);
                    }
                }
            }
        }

        // -------- go --------
        else if (token == "go")
        {
            Movelist moves;
            movegen::legalmoves<movegen::MoveGenType::ALL>(moves, board);

            // No legal moves
            if (moves.empty())
            {
                std::cout << "bestmove 0000\n";
                return;
            }

            // SEARCH FOR BEST MOVE (depth fixed for now)
            Move best = getNextMove(board);

            std::cout << "bestmove " << uci::moveToUci(best) << "\n";
        }

        // -------- quit --------
        else if (token == "quit")
        {
            exit(0);
        }
    }


    // UCI main loop
    void runUciLoop()
    {
        Board board;
        std::string line;

        while (std::getline(std::cin, line))
        {
            if (!line.empty())
                handleUciCommand(line, board);
        }
    }

}
