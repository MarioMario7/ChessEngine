#pragma once
#include "../chess-library-master/include/chess.hpp"


namespace chessengine {
    using namespace chess;

    // Runs a game until completion, using the move provider to select moves
    void playGame(Board& board);
}
