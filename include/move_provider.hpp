#pragma once
#include "../chess-library-master/include/chess.hpp"


namespace chessengine {
    using namespace chess;

    // Returns a move decided by the "player" (AI, random, or human).
    Move getNextMove(Board& board);
    Move findBestMove(int depth, Board& board);
}
