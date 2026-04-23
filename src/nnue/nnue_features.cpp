#include "../../include/evaluation.hpp"
#include "../../include/nnue_features.hpp"
#include <iostream>

namespace chessengine {
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

  

    void add_features(const Board& board, std::vector<HalfKP_Feature>& features ,PieceType piece_type, Color piece_color, Color side, int king_square)
    {
        Bitboard pieces_bitboard =  board.pieces(piece_type, piece_color);

        for (int bit = 0; bit < 64; bit++)
        {
            if (pieces_bitboard & (Bitboard(1) << bit))
            {
                HalfKP_Feature feature;
                feature.king_square = king_square;
                feature.piece_square = bit;
                feature.piece_type = piece_type;
                feature.piece_color = piece_color;
                feature.side = side;

                features.push_back(feature);
            }
        }
    }

    void print_halfkp_features(const std::vector<HalfKP_Feature>& features)
    {
        std::cout << "Feature count: " << features.size() << '\n';

        for (const HalfKP_Feature& feature : features)
        {
            std::cout << "king=" << feature.king_square
                    << " piece=" << feature.piece_square
                    << " type=" << (feature.piece_type)
                    << " color=" << (feature.piece_color)
                    << " side=" << (feature.side)
                    << '\n';
        }
    }

    std::vector<HalfKP_Feature> get_active_features_halfkp(const Board& board, Color side)
    {
        std::vector<HalfKP_Feature> features; 
        features.reserve(32);

        Bitboard king_bitboard = board.pieces(PieceType::KING, side);
        int king_square = -1;

        for (int bit = 0; bit <= 63; bit++)
        {
           if (king_bitboard & (Bitboard(1) << bit)) // we offset the one until we find the 1 in the king BB 
           {
                king_square = bit;
                break;
           }
        }

        if (king_square == -1)
        {
            return features; // no king exists for side, should never happen
        }


        for (Color color : {Color::WHITE, Color::BLACK})
        {
            for (PieceType piece_type : {PieceType::PAWN,PieceType::KNIGHT,PieceType::BISHOP,PieceType::ROOK,PieceType::QUEEN})
            {
                add_features(board, features , piece_type, color, side, king_square);
            }
        }

        return features;
    }

    int halfkp_index(const HalfKP_Feature& feature)
    {
        int piece_type_index = piece_type_to_index(feature.piece_type);
        int color_index = color_to_index(feature.piece_color);

        if (piece_type_index == -1)
        {
            return -1;
        }

        int piece_code = color_index * 5 + piece_type_index;

        /*
        
        White pawn   = 0
        White knight = 1
        White bishop = 2
        White rook   = 3
        White queen  = 4

        Black pawn   = 5
        Black knight = 6
        Black bishop = 7
        Black rook   = 8
        Black queen  = 9
        
        */

        return feature.king_square * 640 // 64 squares * 10 possible pieces == 640 features for one king square
            + feature.piece_square * 10 // 5 piece types (no kings) * 2 colors
            + piece_code;
    }

    std::vector<int> get_active_feature_indices_halfkp(const Board& board, Color side)
    {
        std::vector<int> indices;
        indices.reserve(32);

        std::vector<HalfKP_Feature> features = get_active_features_halfkp(board, side);

        for (const HalfKP_Feature& feature : features)
        {
            int index = halfkp_index(feature);

            if (index != -1)
            {
                indices.push_back(index);
            }
        }

        return indices;
    }
}
