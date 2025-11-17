#include "../include/evaluation.hpp"
#include <iostream>

namespace chessengine {
    using namespace chess;

    // Piece values in centipawns
    static constexpr int pawn_value   = 100;
    static constexpr int bishop_value = 300;
    static constexpr int knight_value = 300;
    static constexpr int rook_value   = 500;
    static constexpr int queen_value  = 900;

    int evaluatePosition(const Board& board)
    {
        return evaluateMaterial(board);
    }

    int evaluateMaterial(const Board& board)
    {
        Bitboard all = board.all();
        int black_value = 0;
        int white_value = 0;

        while (all)
        {
            Square square = all.pop();
            Piece pieceObject = board.at(square);
            std::string piece = std::string(pieceObject);

            // change later to switch with enums possibly
            if (piece == "p")
                black_value += pawn_value;
            else if (piece == "b")
                black_value += bishop_value;
            else if (piece == "n")
                black_value += knight_value;
            else if (piece == "r")
                black_value += rook_value;
            else if (piece == "q")
                black_value += queen_value;
            else if (piece == "P")
                white_value += pawn_value;
            else if (piece == "B")
                white_value += bishop_value;
            else if (piece == "N")
                white_value += knight_value;
            else if (piece == "R")
                white_value += rook_value;
            else if (piece == "Q")
                white_value += queen_value;
        }

       // std::cout << "White : " << white_value << std::endl;
       // std::cout << "Black : " << black_value << std::endl;

        
        return white_value - black_value;
    }

    

}
