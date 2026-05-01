#pragma once

#include "../chess-library-master/include/chess.hpp"
#include "nnue_evaluator.hpp"

namespace chessengine
{
    int evaluateMaterial(const chess::Board& board);

    int evaluatePosition(const chess::Board& board);

    bool refreshNNUEAccumulator(
        const chess::Board& board,
        NNUEAccumulator& accumulator
    );

    bool updateNNUEAccumulatorAfterMove(
        const chess::Board& board,
        const chess::Move& move,
        const NNUEAccumulator& parent,
        NNUEAccumulator& child
    );

    int evaluatePositionFromAccumulator(
        const chess::Board& board,
        const NNUEAccumulator& accumulator
    );

    bool isNNUEAvailable();
}