#pragma once

#include <vector>
#include "chess.hpp"

namespace chessengine {
    using namespace chess;

    struct HalfKP_Feature
    {
        int king_square;
        int piece_square;
        PieceType piece_type;
        Color piece_color;
        Color side;
    };

    std::vector<HalfKP_Feature> get_active_features_halfkp(const Board& board, Color side);

    std::vector<int> get_active_feature_indices_halfkp(const Board& board, Color side);

    int halfkp_index(const HalfKP_Feature& feature);

    void print_halfkp_features(const std::vector<HalfKP_Feature>& features);
}