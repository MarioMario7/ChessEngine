#include "../include/evaluation.hpp"
#include "../include/nnue_features.hpp"
#include "../include/nnue_evaluator.hpp"

#include <iostream>

namespace chessengine
{
    using namespace chess;

    static constexpr int pawn_value = 100;
    static constexpr int bishop_value = 300;
    static constexpr int knight_value = 300;
    static constexpr int rook_value = 500;
    static constexpr int queen_value = 900;

    static QuantizedHalfKPNNUE g_nnue;
    static bool g_nnue_initialized = false;

    static void initialize_nnue_if_needed()
    {
        if (g_nnue_initialized)
        {
            return;
        }

        g_nnue_initialized = true;

        if (g_nnue.load("halfkp_nnue_quantized.bin"))
        {
            std::cout << "Quantized NNUE evaluation enabled.\n";
        }
        else
        {
            std::cout << "Quantized NNUE file missing. Falling back to material evaluation.\n";
        }
    }

    bool isNNUEAvailable()
    {
        initialize_nnue_if_needed();

        return g_nnue.is_loaded();
    }

    bool refreshNNUEAccumulator(
        const Board& board,
        NNUEAccumulator& accumulator
    )
    {
        initialize_nnue_if_needed();

        if (!g_nnue.is_loaded())
        {
            accumulator.valid = false;
            return false;
        }

        g_nnue.refresh_accumulator(board, accumulator);

        return accumulator.valid;
    }

    bool updateNNUEAccumulatorAfterMove(
        const Board& board,
        const Move& move,
        const NNUEAccumulator& parent,
        NNUEAccumulator& child
    )
    {
        initialize_nnue_if_needed();

        if (!g_nnue.is_loaded() || !parent.valid)
        {
            child.valid = false;
            return false;
        }

        g_nnue.update_accumulator_after_move(
            board,
            move,
            parent,
            child
        );

        return child.valid;
    }

    int evaluatePositionFromAccumulator(
        const Board& board,
        const NNUEAccumulator& accumulator
    )
    {
        initialize_nnue_if_needed();

        if (g_nnue.is_loaded() && accumulator.valid)
        {
            return g_nnue.evaluate_white_pov_from_accumulator(
                board,
                accumulator
            );
        }

        return evaluateMaterial(board);
    }

    int evaluatePosition(const Board& board)
    {
        initialize_nnue_if_needed();

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
            {
                black_value += pawn_value;
            }
            else if (piece == "b")
            {
                black_value += bishop_value;
            }
            else if (piece == "n")
            {
                black_value += knight_value;
            }
            else if (piece == "r")
            {
                black_value += rook_value;
            }
            else if (piece == "q")
            {
                black_value += queen_value;
            }
            else if (piece == "P")
            {
                white_value += pawn_value;
            }
            else if (piece == "B")
            {
                white_value += bishop_value;
            }
            else if (piece == "N")
            {
                white_value += knight_value;
            }
            else if (piece == "R")
            {
                white_value += rook_value;
            }
            else if (piece == "Q")
            {
                white_value += queen_value;
            }
        }

        return white_value - black_value;
    }
}