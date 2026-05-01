#pragma once

#include "evaluation.hpp"
#include "nnue_features.hpp"

#include <string>
#include <vector>

namespace chessengine
{
    class FloatHalfKPNNUE
    {
    public:
        bool load(const std::string& path);

        bool is_loaded() const;

        int evaluate_white_pov(const chess::Board& board) const;

    private:
        static constexpr int EXPECTED_INPUTS = 40960;
        static constexpr int EXPECTED_L1 = 256;
        static constexpr int EXPECTED_L2 = 32;
        static constexpr int EXPECTED_L3 = 32;

        static constexpr float OUTPUT_TO_CP = 600.0f;

        int inputs_ = 0;
        int l1_ = 0;
        int l2_ = 0;
        int l3_ = 0;

        bool loaded_ = false;

        std::vector<float> feature_weights_;
        std::vector<float> feature_bias_;

        std::vector<float> hidden1_weights_;
        std::vector<float> hidden1_bias_;

        std::vector<float> hidden2_weights_;
        std::vector<float> hidden2_bias_;

        std::vector<float> output_weights_;
        std::vector<float> output_bias_;

        void build_accumulator(
            const std::vector<int>& features,
            std::vector<float>& accumulator
        ) const;

        static float clipped_relu(float value);

        static void clipped_relu_vector(std::vector<float>& values);

        void run_dense_layer(
            const std::vector<float>& input,
            const std::vector<float>& weights,
            const std::vector<float>& bias,
            int input_size,
            int output_size,
            std::vector<float>& output
        ) const;

        float run_output_layer(const std::vector<float>& input) const;
    };
}