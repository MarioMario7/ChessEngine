#pragma once

#include "chess.hpp"

#include <array>
#include <vector>

namespace chessengine
{
    static constexpr int HALF_KP_INPUTS = 40960;
    static constexpr int HALF_KP_MAX_ACTIVE = 32;

    struct HalfKP_Feature
    {
        int king_square = -1;
        int piece_square = -1;
        chess::PieceType piece_type = chess::PieceType::NONE;
        chess::Color piece_color = chess::Color::WHITE;
        chess::Color side = chess::Color::WHITE;
    };

    int piece_type_to_index(chess::PieceType piece_type);

    int color_to_index(chess::Color color);

    int make_halfkp_index(
        int king_square,
        int piece_square,
        chess::PieceType piece_type,
        chess::Color piece_color
    );

    int get_king_square_halfkp(
        const chess::Board& board,
        chess::Color side
    );

    int get_active_feature_indices_halfkp_fast(
        const chess::Board& board,
        chess::Color side,
        std::array<int, HALF_KP_MAX_ACTIVE>& indices
    );

    std::vector<int> get_active_feature_indices_halfkp(
        const chess::Board& board,
        chess::Color side
    );

    int get_active_feature_indices_halfkp_fixed(
        const chess::Board& board,
        chess::Color side,
        std::array<int, HALF_KP_MAX_ACTIVE>& indices
    );

    void add_features(
        const chess::Board& board,
        std::vector<HalfKP_Feature>& features,
        chess::PieceType piece_type,
        chess::Color piece_color,
        chess::Color side,
        int king_square
    );

    void print_halfkp_features(
        const std::vector<HalfKP_Feature>& features
    );

    std::vector<HalfKP_Feature> get_active_features_halfkp(
        const chess::Board& board,
        chess::Color side
    );

    int halfkp_index(const HalfKP_Feature& feature);
}