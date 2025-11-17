#pragma once
#include "chess.hpp"
#include <string>

namespace chessengine {
    using namespace chess;

    //(blocking) UCI command loop
    void runUciLoop();

    // respond to a single UCI command (debugging puroposes)
    void handleUciCommand(const std::string& cmd, Board& board);
}
