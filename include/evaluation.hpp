#pragma once
#include "../chess-library-master/include/chess.hpp"

namespace chessengine {

int evaluateMaterial(const chess::Board& board);
int evaluatePosition(const chess::Board& board);


}
