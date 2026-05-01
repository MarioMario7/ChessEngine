#include "../../include/nnue_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace chessengine
{
    static bool read_bytes(std::ifstream& file, void* dst, std::size_t bytes)
    {
        file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(file);
    }

    static bool read_int32(std::ifstream& file, int& value)
    {
        int temp = 0;

        if (!read_bytes(file, &temp, sizeof(temp)))
            return false;

        value = temp;
        return true;
    }

    template <typename T>
    static bool read_vector(std::ifstream& file, std::vector<T>& vector, std::size_t count)
    {
        vector.resize(count);
        return read_bytes(file, vector.data(), sizeof(T) * count);
    }

    bool FloatHalfKPNNUE::load(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);

        if (!file)
        {
            std::cout << "NNUE file not found: " << path << '\n';
            loaded_ = false;
            return false;
        }

        char magic[4];

        if (!read_bytes(file, magic, 4))
            return false;

        if (magic[0] != 'H' || magic[1] != 'K' || magic[2] != 'F' || magic[3] != '1')
        {
            std::cout << "Invalid NNUE file magic. Expected HKF1.\n";
            loaded_ = false;
            return false;
        }

        if (!read_int32(file, inputs_)) return false;
        if (!read_int32(file, l1_)) return false;
        if (!read_int32(file, l2_)) return false;
        if (!read_int32(file, l3_)) return false;

        if (inputs_ != EXPECTED_INPUTS ||
            l1_ != EXPECTED_L1 ||
            l2_ != EXPECTED_L2 ||
            l3_ != EXPECTED_L3)
        {
            std::cout << "NNUE dimensions do not match engine.\n";
            std::cout << "inputs=" << inputs_
                      << " l1=" << l1_
                      << " l2=" << l2_
                      << " l3=" << l3_
                      << '\n';

            loaded_ = false;
            return false;
        }

        if (!read_vector(file, feature_weights_, static_cast<std::size_t>(inputs_) * l1_))
            return false;

        if (!read_vector(file, feature_bias_, l1_))
            return false;

        if (!read_vector(file, hidden1_weights_, static_cast<std::size_t>(l2_) * (l1_ * 2)))
            return false;

        if (!read_vector(file, hidden1_bias_, l2_))
            return false;

        if (!read_vector(file, hidden2_weights_, static_cast<std::size_t>(l3_) * l2_))
            return false;

        if (!read_vector(file, hidden2_bias_, l3_))
            return false;

        if (!read_vector(file, output_weights_, l3_))
            return false;

        if (!read_vector(file, output_bias_, 1))
            return false;

        loaded_ = true;

        std::cout << "Loaded Float HalfKP NNUE: " << path << '\n';
        std::cout << "NNUE architecture: "
                  << inputs_ << " -> " << l1_
                  << " accumulators -> " << (l1_ * 2)
                  << " -> " << l2_
                  << " -> " << l3_
                  << " -> 1\n";

        return true;
    }

    bool FloatHalfKPNNUE::is_loaded() const
    {
        return loaded_;
    }

    float FloatHalfKPNNUE::clipped_relu(float value)
    {
        if (value < 0.0f)
            return 0.0f;

        if (value > 1.0f)
            return 1.0f;

        return value;
    }

    void FloatHalfKPNNUE::clipped_relu_vector(std::vector<float>& values)
    {
        for (float& value : values)
            value = clipped_relu(value);
    }

    void FloatHalfKPNNUE::build_accumulator(
        const std::vector<int>& features,
        std::vector<float>& accumulator
    ) const
    {
        accumulator.assign(l1_, 0.0f);

        for (int i = 0; i < l1_; i++)
            accumulator[i] = feature_bias_[i];

        for (int feature : features)
        {
            if (feature < 0 || feature >= inputs_)
                continue;

            std::size_t base = static_cast<std::size_t>(feature) * l1_;

            for (int i = 0; i < l1_; i++)
            {
                accumulator[i] += feature_weights_[base + i];
            }
        }

        clipped_relu_vector(accumulator);
    }

    void FloatHalfKPNNUE::run_dense_layer(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        const std::vector<float>& bias,
        int input_size,
        int output_size,
        std::vector<float>& output
    ) const
    {
        output.assign(output_size, 0.0f);

        for (int out = 0; out < output_size; out++)
        {
            float sum = bias[out];

            std::size_t weight_base = static_cast<std::size_t>(out) * input_size;

            for (int in = 0; in < input_size; in++)
            {
                sum += input[in] * weights[weight_base + in];
            }

            output[out] = clipped_relu(sum);
        }
    }

    float FloatHalfKPNNUE::run_output_layer(const std::vector<float>& input) const
    {
        float sum = output_bias_[0];

        for (int i = 0; i < l3_; i++)
        {
            sum += input[i] * output_weights_[i];
        }

        return sum;
    }

    int FloatHalfKPNNUE::evaluate_white_pov(const chess::Board& board) const
    {
        if (!loaded_)
            return 0;

        std::vector<int> white_features =
            get_active_feature_indices_halfkp(board, chess::Color::WHITE);

        std::vector<int> black_features =
            get_active_feature_indices_halfkp(board, chess::Color::BLACK);

        std::vector<float> white_accumulator;
        std::vector<float> black_accumulator;

        build_accumulator(white_features, white_accumulator);
        build_accumulator(black_features, black_accumulator);

        std::vector<float> input;
        input.resize(l1_ * 2);

        bool white_to_move = board.sideToMove() == chess::Color::WHITE;

        if (white_to_move)
        {
            for (int i = 0; i < l1_; i++)
            {
                input[i] = white_accumulator[i];
                input[l1_ + i] = black_accumulator[i];
            }
        }
        else
        {
            for (int i = 0; i < l1_; i++)
            {
                input[i] = black_accumulator[i];
                input[l1_ + i] = white_accumulator[i];
            }
        }

        clipped_relu_vector(input);

        std::vector<float> hidden1;
        std::vector<float> hidden2;

        run_dense_layer(
            input,
            hidden1_weights_,
            hidden1_bias_,
            l1_ * 2,
            l2_,
            hidden1
        );

        run_dense_layer(
            hidden1,
            hidden2_weights_,
            hidden2_bias_,
            l2_,
            l3_,
            hidden2
        );

        float side_to_move_score = run_output_layer(hidden2);

        float white_pov_score;

        if (white_to_move)
            white_pov_score = side_to_move_score;
        else
            white_pov_score = -side_to_move_score;

        int centipawns = static_cast<int>(std::round(white_pov_score * OUTPUT_TO_CP));

        return centipawns;
    }
}