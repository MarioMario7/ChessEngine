#include "../include/evaluation.hpp"
#include "../include/nnue_features.hpp"
#include "../include/nnue_evaluator.hpp"

#include <iostream>

namespace chessengine {
    using namespace chess;

    static constexpr int pawn_value   = 100;
    static constexpr int bishop_value = 300;
    static constexpr int knight_value = 300;
    static constexpr int rook_value   = 500;
    static constexpr int queen_value  = 900;

    static FloatHalfKPNNUE g_nnue;
    static bool g_nnue_initialized = false;

    int evaluatePosition(const Board& board)
    {
        if (!g_nnue_initialized)
        {
            g_nnue_initialized = true;

            if (g_nnue.load("halfkp_nnue_float.bin"))
            {
                std::cout << "NNUE evaluation enabled.\n";
            }
            else
            {
                std::cout << "NNUE file missing. Falling back to material evaluation.\n";
            }
        }

        if (g_nnue.is_loaded())
        {
            return g_nnue.evaluate_white_pov(board);
        }

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

        return white_value - black_value;
    }
}