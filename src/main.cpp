


#include <iostream>
#include <chrono>
#include "../chess-library-master/include/chess.hpp"
#include "../include/game_loop.hpp"
#include "../include/uci_interface.hpp" 
#include "../include/move_provider.hpp" // remove after tests

using namespace chessengine;


uint64_t perft(chess::Board& board, int depth) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (depth == 1) {
        return moves.size();
    }

    uint64_t nodes = 0;

    for (int i = 0; i < moves.size(); i++) {
        const auto move = moves[i];
        board.makeMove(move);
        nodes += perft(board, depth - 1);
        board.unmakeMove(move);
    }

    return nodes;
}


int main() {

    #ifdef NDEBUG
    std::cout << "Release mode: YES\n";
    #else
        std::cout << "DEBUG MODE\n";
    #endif




    
    //           <<!!! To enable UCI conmmunciation , uncomment the runUciLoop command and comment out everything else !!!>>

    //chessengine::runUciLoop();




    // chess starting position

     Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
     board.setFen("r1b2rk1/ppqn1ppp/2pbpn2/3p4/3P4/2PBPNB1/PP1N1PPP/R2Q1RK1 b - - 8 9");
     std::cout << " FEN: " << board.getFen() << "\n\n";
     playGame(board);

    //Board board("4k3/8/4q3/4K3/8/8/8/8 w - - 0 1");
    //     std::cout << " FEN: " << board.getFen() << "\n\n";

    //     Move best = getNextMove(board);

    //     std::cout << " FEN: " << board.getFen() << "\n\n";
    //    // std::cout << "Engine played: " << uci::moveToUci(best) << "\n";
    //     std::cout << "Engine played: " << uci::moveToSan(board,best) << "\n";

    //     std::cout << " FEN: " << board.getFen() << "\n\n";


   // std::cout << "Initial FEN: " << board.getFen() << "\n\n";
 
  



    // benchmarkPerft(
    //     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    //     7
    // );




    return 0;


}
