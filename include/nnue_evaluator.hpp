#pragma once

#include "chess.hpp"
#include "nnue_features.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace chessengine
{
    struct NNUEAccumulator
    {
        std::array<std::int32_t, 256> white{};
        std::array<std::int32_t, 256> black{};
        bool valid = false;
    };

    class QuantizedHalfKPNNUE
    {
    public:
        bool load(const std::string& path);

        bool is_loaded() const;

        int evaluate_white_pov(const chess::Board& board) const;

        void refresh_accumulator(
            const chess::Board& board,
            NNUEAccumulator& accumulator
        ) const;

        void update_accumulator_after_move(
            const chess::Board& board,
            const chess::Move& move,
            const NNUEAccumulator& parent,
            NNUEAccumulator& child
        ) const;

        int evaluate_white_pov_from_accumulator(
            const chess::Board& board,
            const NNUEAccumulator& accumulator
        ) const;

    private:
        static constexpr int EXPECTED_INPUTS = 40960;
        static constexpr int EXPECTED_L1 = 256;
        static constexpr int EXPECTED_L2 = 32;
        static constexpr int EXPECTED_L3 = 32;

        static constexpr int CONCATENATED_L1 = EXPECTED_L1 * 2;

        static constexpr float OUTPUT_TO_CP = 600.0f;

        bool loaded_ = false;

        int inputs_ = 0;
        int l1_ = 0;
        int l2_ = 0;
        int l3_ = 0;
        int quant_scale_ = 0;

        std::vector<std::int16_t> feature_weights_;
        std::vector<std::int32_t> feature_bias_;

        std::vector<std::int16_t> hidden1_weights_;
        std::vector<std::int32_t> hidden1_bias_;

        std::vector<std::int16_t> hidden2_weights_;
        std::vector<std::int32_t> hidden2_bias_;

        std::vector<std::int16_t> output_weights_;
        std::vector<std::int32_t> output_bias_;

        std::int32_t divide_by_quant_scale(std::int64_t value) const;

        std::int16_t clamp_activation_value(std::int32_t value) const;

        std::int16_t clamp_scaled_sum_to_activation(std::int64_t value) const;

        void build_accumulator_fixed(
            const std::array<int, HALF_KP_MAX_ACTIVE>& features,
            int feature_count,
            std::array<std::int32_t, EXPECTED_L1>& accumulator
        ) const;

        void build_network_input_fixed(
            const std::array<std::int32_t, EXPECTED_L1>& white_accumulator,
            const std::array<std::int32_t, EXPECTED_L1>& black_accumulator,
            bool white_to_move,
            std::array<std::int16_t, CONCATENATED_L1>& input
        ) const;

        void run_dense_layer_512_to_32_fixed(
            const std::array<std::int16_t, CONCATENATED_L1>& input,
            std::array<std::int16_t, EXPECTED_L2>& output
        ) const;

        void run_dense_layer_32_to_32_fixed(
            const std::array<std::int16_t, EXPECTED_L2>& input,
            std::array<std::int16_t, EXPECTED_L3>& output
        ) const;

        std::int64_t run_output_layer_fixed(
            const std::array<std::int16_t, EXPECTED_L3>& input
        ) const;

        std::int64_t dot_product_i16_i16(
            const std::int16_t* input,
            const std::int16_t* weights,
            int size
        ) const;

        void add_feature_to_accumulator(
            std::array<std::int32_t, EXPECTED_L1>& accumulator,
            int feature
        ) const;

        void remove_feature_from_accumulator(
            std::array<std::int32_t, EXPECTED_L1>& accumulator,
            int feature
        ) const;

        void add_piece_to_both_accumulators(
            NNUEAccumulator& accumulator,
            int white_king_square,
            int black_king_square,
            int piece_square,
            chess::PieceType piece_type,
            chess::Color piece_color
        ) const;

        void remove_piece_from_both_accumulators(
            NNUEAccumulator& accumulator,
            int white_king_square,
            int black_king_square,
            int piece_square,
            chess::PieceType piece_type,
            chess::Color piece_color
        ) const;
    };
}