#ifndef CHESSENGINE_NNUE_FEATURES_HPP
#define CHESSENGINE_NNUE_FEATURES_HPP

#include "../chess-library-master/include/chess.hpp"
#include <vector>

namespace chessengine
{
    struct HalfKP_Feature
    {
        int king_square;
        int piece_square;
        chess::PieceType piece_type;
        chess::Color piece_color;
        chess::Color side;
    };

    int piece_type_to_index(chess::PieceType piece_type);
    int color_to_index(chess::Color color);

    void add_features(const chess::Board& board, std::vector<HalfKP_Feature>& features, chess::PieceType piece_type, chess::Color piece_color, chess::Color side, int king_square);

    void print_halfkp_features(const std::vector<HalfKP_Feature>& features);

    std::vector<HalfKP_Feature> get_active_features_halfkp(const chess::Board& board, chess::Color side);

    int halfkp_index(const HalfKP_Feature& feature);

    std::vector<int> get_active_feature_indices_halfkp(const chess::Board& board, chess::Color side);
}

#endif