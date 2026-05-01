#include "../../include/nnue_features.hpp"

#include <array>
#include <iostream>
#include <vector>

namespace chessengine
{
    using namespace chess;

    int piece_type_to_index(PieceType piece_type)
    {
        if (piece_type == PieceType::PAWN)
        {
            return 0;
        }

        if (piece_type == PieceType::KNIGHT)
        {
            return 1;
        }

        if (piece_type == PieceType::BISHOP)
        {
            return 2;
        }

        if (piece_type == PieceType::ROOK)
        {
            return 3;
        }

        if (piece_type == PieceType::QUEEN)
        {
            return 4;
        }

        return -1;
    }

    int color_to_index(Color color)
    {
        if (color == Color::WHITE)
        {
            return 0;
        }

        return 1;
    }

    int make_halfkp_index(
        int king_square,
        int piece_square,
        PieceType piece_type,
        Color piece_color
    )
    {
        int piece_type_index = piece_type_to_index(piece_type);

        if (piece_type_index == -1)
        {
            return -1;
        }

        int color_index = color_to_index(piece_color);

        int piece_code = color_index * 5 + piece_type_index;

        return king_square * 640
             + piece_square * 10
             + piece_code;
    }

    int get_king_square_halfkp(
        const Board& board,
        Color side
    )
    {
        Bitboard king_bitboard = board.pieces(PieceType::KING, side);

        if (!king_bitboard)
        {
            return -1;
        }

        Square king_square = king_bitboard.pop();

        return king_square.index();
    }

    int get_active_feature_indices_halfkp_fast(
        const Board& board,
        Color side,
        std::array<int, HALF_KP_MAX_ACTIVE>& indices
    )
    {
        indices.fill(-1);

        int count = 0;

        int king_square = get_king_square_halfkp(board, side);

        if (king_square == -1)
        {
            return 0;
        }

        for (Color piece_color : {Color::WHITE, Color::BLACK})
        {
            int color_index = color_to_index(piece_color);

            for (PieceType piece_type : {
                PieceType::PAWN,
                PieceType::KNIGHT,
                PieceType::BISHOP,
                PieceType::ROOK,
                PieceType::QUEEN
            })
            {
                int piece_type_index = piece_type_to_index(piece_type);

                if (piece_type_index == -1)
                {
                    continue;
                }

                int piece_code = color_index * 5 + piece_type_index;

                Bitboard pieces_bitboard = board.pieces(piece_type, piece_color);

                while (pieces_bitboard)
                {
                    if (count >= HALF_KP_MAX_ACTIVE)
                    {
                        return count;
                    }

                    Square piece_square = pieces_bitboard.pop();

                    int index = king_square * 640
                              + piece_square.index() * 10
                              + piece_code;

                    indices[count] = index;
                    count++;
                }
            }
        }

        return count;
    }

    std::vector<int> get_active_feature_indices_halfkp(
        const Board& board,
        Color side
    )
    {
        std::array<int, HALF_KP_MAX_ACTIVE> fixed_indices;

        int count = get_active_feature_indices_halfkp_fast(
            board,
            side,
            fixed_indices
        );

        std::vector<int> indices;
        indices.reserve(count);

        for (int i = 0; i < count; i++)
        {
            indices.push_back(fixed_indices[i]);
        }

        return indices;
    }

    int get_active_feature_indices_halfkp_fixed(
        const Board& board,
        Color side,
        std::array<int, HALF_KP_MAX_ACTIVE>& indices
    )
    {
        return get_active_feature_indices_halfkp_fast(
            board,
            side,
            indices
        );
    }

    void add_features(
        const Board& board,
        std::vector<HalfKP_Feature>& features,
        PieceType piece_type,
        Color piece_color,
        Color side,
        int king_square
    )
    {
        Bitboard pieces_bitboard = board.pieces(piece_type, piece_color);

        while (pieces_bitboard)
        {
            Square piece_square = pieces_bitboard.pop();

            HalfKP_Feature feature;

            feature.king_square = king_square;
            feature.piece_square = piece_square.index();
            feature.piece_type = piece_type;
            feature.piece_color = piece_color;
            feature.side = side;

            features.push_back(feature);
        }
    }

    void print_halfkp_features(
        const std::vector<HalfKP_Feature>& features
    )
    {
        std::cout << "Feature count: " << features.size() << '\n';

        for (const HalfKP_Feature& feature : features)
        {
            std::cout << "king=" << feature.king_square
                      << " piece=" << feature.piece_square
                      << " type=" << feature.piece_type
                      << " color=" << feature.piece_color
                      << " side=" << feature.side
                      << '\n';
        }
    }

    std::vector<HalfKP_Feature> get_active_features_halfkp(
        const Board& board,
        Color side
    )
    {
        std::vector<HalfKP_Feature> features;
        features.reserve(HALF_KP_MAX_ACTIVE);

        int king_square = get_king_square_halfkp(board, side);

        if (king_square == -1)
        {
            return features;
        }

        for (Color piece_color : {Color::WHITE, Color::BLACK})
        {
            for (PieceType piece_type : {
                PieceType::PAWN,
                PieceType::KNIGHT,
                PieceType::BISHOP,
                PieceType::ROOK,
                PieceType::QUEEN
            })
            {
                add_features(
                    board,
                    features,
                    piece_type,
                    piece_color,
                    side,
                    king_square
                );
            }
        }

        return features;
    }

    int halfkp_index(const HalfKP_Feature& feature)
    {
        return make_halfkp_index(
            feature.king_square,
            feature.piece_square,
            feature.piece_type,
            feature.piece_color
        );
    }
}