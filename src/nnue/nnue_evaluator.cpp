#include "../../include/nnue_evaluator.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace chessengine
{
    using namespace chess;

    static bool read_bytes(std::ifstream& file, void* dst, std::size_t bytes)
    {
        file.read(
            reinterpret_cast<char*>(dst),
            static_cast<std::streamsize>(bytes)
        );

        return static_cast<bool>(file);
    }

    static bool read_int32(std::ifstream& file, int& value)
    {
        int temp = 0;

        if (!read_bytes(file, &temp, sizeof(temp)))
        {
            return false;
        }

        value = temp;
        return true;
    }

    template <typename T>
    static bool read_vector(
        std::ifstream& file,
        std::vector<T>& vector,
        std::size_t count
    )
    {
        vector.resize(count);

        if (count == 0)
        {
            return true;
        }

        return read_bytes(file, vector.data(), sizeof(T) * count);
    }

    bool QuantizedHalfKPNNUE::load(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);

        if (!file)
        {
            std::cout << "Quantized NNUE file not found: " << path << '\n';
            loaded_ = false;
            return false;
        }

        char magic[4];

        if (!read_bytes(file, magic, 4))
        {
            std::cout << "Could not read quantized NNUE file magic.\n";
            loaded_ = false;
            return false;
        }

        if (magic[0] != 'H' ||
            magic[1] != 'K' ||
            magic[2] != 'Q' ||
            magic[3] != '1')
        {
            std::cout << "Invalid quantized NNUE file magic. Expected HKQ1.\n";
            loaded_ = false;
            return false;
        }

        if (!read_int32(file, inputs_))
        {
            std::cout << "Could not read NNUE input count.\n";
            loaded_ = false;
            return false;
        }

        if (!read_int32(file, l1_))
        {
            std::cout << "Could not read NNUE L1 size.\n";
            loaded_ = false;
            return false;
        }

        if (!read_int32(file, l2_))
        {
            std::cout << "Could not read NNUE L2 size.\n";
            loaded_ = false;
            return false;
        }

        if (!read_int32(file, l3_))
        {
            std::cout << "Could not read NNUE L3 size.\n";
            loaded_ = false;
            return false;
        }

        if (!read_int32(file, quant_scale_))
        {
            std::cout << "Could not read NNUE quantization scale.\n";
            loaded_ = false;
            return false;
        }

        if (inputs_ != EXPECTED_INPUTS ||
            l1_ != EXPECTED_L1 ||
            l2_ != EXPECTED_L2 ||
            l3_ != EXPECTED_L3)
        {
            std::cout << "Quantized NNUE dimensions do not match engine.\n";
            std::cout << "inputs=" << inputs_
                      << " l1=" << l1_
                      << " l2=" << l2_
                      << " l3=" << l3_
                      << '\n';

            loaded_ = false;
            return false;
        }

        if (quant_scale_ <= 0 || quant_scale_ > 32767)
        {
            std::cout << "Invalid quantization scale: " << quant_scale_ << '\n';
            loaded_ = false;
            return false;
        }

        if (!read_vector(
                file,
                feature_weights_,
                static_cast<std::size_t>(inputs_) * l1_
            ))
        {
            std::cout << "Could not read quantized feature weights.\n";
            loaded_ = false;
            return false;
        }

        if (!read_vector(file, feature_bias_, l1_))
        {
            std::cout << "Could not read quantized feature bias.\n";
            loaded_ = false;
            return false;
        }

        if (!read_vector(
                file,
                hidden1_weights_,
                static_cast<std::size_t>(l2_) * (l1_ * 2)
            ))
        {
            std::cout << "Could not read quantized hidden1 weights.\n";
            loaded_ = false;
            return false;
        }

        if (!read_vector(file, hidden1_bias_, l2_))
        {
            std::cout << "Could not read quantized hidden1 bias.\n";
            loaded_ = false;
            return false;
        }

        if (!read_vector(
                file,
                hidden2_weights_,
                static_cast<std::size_t>(l3_) * l2_
            ))
        {
            std::cout << "Could not read quantized hidden2 weights.\n";
            loaded_ = false;
            return false;
        }

        if (!read_vector(file, hidden2_bias_, l3_))
        {
            std::cout << "Could not read quantized hidden2 bias.\n";
            loaded_ = false;
            return false;
        }

        if (!read_vector(file, output_weights_, l3_))
        {
            std::cout << "Could not read quantized output weights.\n";
            loaded_ = false;
            return false;
        }

        if (!read_vector(file, output_bias_, 1))
        {
            std::cout << "Could not read quantized output bias.\n";
            loaded_ = false;
            return false;
        }

        loaded_ = true;

        std::cout << "Loaded Quantized HalfKP NNUE: " << path << '\n';
        std::cout << "NNUE architecture: "
                  << inputs_ << " -> " << l1_
                  << " accumulators -> " << (l1_ * 2)
                  << " -> " << l2_
                  << " -> " << l3_
                  << " -> 1\n";
        std::cout << "NNUE quantization scale: " << quant_scale_ << '\n';

#if defined(__AVX2__)
        std::cout << "NNUE AVX2 dense and accumulator updates enabled.\n";
#else
        std::cout << "NNUE AVX2 not enabled, using scalar NNUE updates.\n";
#endif

        return true;
    }

    bool QuantizedHalfKPNNUE::is_loaded() const
    {
        return loaded_;
    }

    std::int32_t QuantizedHalfKPNNUE::divide_by_quant_scale(
        std::int64_t value
    ) const
    {
        std::int64_t scale = static_cast<std::int64_t>(quant_scale_);

        if (value >= 0)
        {
            return static_cast<std::int32_t>((value + scale / 2) / scale);
        }

        return static_cast<std::int32_t>(-((-value + scale / 2) / scale));
    }

    std::int16_t QuantizedHalfKPNNUE::clamp_activation_value(
        std::int32_t value
    ) const
    {
        if (value < 0)
        {
            return 0;
        }

        if (value > quant_scale_)
        {
            return static_cast<std::int16_t>(quant_scale_);
        }

        return static_cast<std::int16_t>(value);
    }

    std::int16_t QuantizedHalfKPNNUE::clamp_scaled_sum_to_activation(
        std::int64_t value
    ) const
    {
        std::int32_t divided = divide_by_quant_scale(value);

        return clamp_activation_value(divided);
    }

    std::int64_t QuantizedHalfKPNNUE::dot_product_i16_i16(
        const std::int16_t* input,
        const std::int16_t* weights,
        int size
    ) const
    {
#if defined(__AVX2__)
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        int i = 0;

        for (; i + 16 <= size; i += 16)
        {
            __m256i in_vec =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(input + i)
                );

            __m256i weight_vec =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weights + i)
                );

            /*
                _mm256_madd_epi16:
                Takes 16 pairs of int16 values.
                Multiplies them pairwise.
                Adds adjacent products.
                Produces 8 int32 partial sums.
            */
            __m256i pair_sums =
                _mm256_madd_epi16(in_vec, weight_vec);

            __m128i low128 =
                _mm256_castsi256_si128(pair_sums);

            __m128i high128 =
                _mm256_extracti128_si256(pair_sums, 1);

            __m256i low64 =
                _mm256_cvtepi32_epi64(low128);

            __m256i high64 =
                _mm256_cvtepi32_epi64(high128);

            acc0 = _mm256_add_epi64(acc0, low64);
            acc1 = _mm256_add_epi64(acc1, high64);
        }

        alignas(32) std::int64_t partial0[4];
        alignas(32) std::int64_t partial1[4];

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(partial0),
            acc0
        );

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(partial1),
            acc1
        );

        std::int64_t sum =
            partial0[0] + partial0[1] + partial0[2] + partial0[3] +
            partial1[0] + partial1[1] + partial1[2] + partial1[3];

        for (; i < size; i++)
        {
            sum += static_cast<std::int64_t>(input[i])
                 * static_cast<std::int64_t>(weights[i]);
        }

        return sum;
#else
        std::int64_t sum = 0;

        for (int i = 0; i < size; i++)
        {
            sum += static_cast<std::int64_t>(input[i])
                 * static_cast<std::int64_t>(weights[i]);
        }

        return sum;
#endif
    }

    void QuantizedHalfKPNNUE::build_accumulator_fixed(
        const std::array<int, HALF_KP_MAX_ACTIVE>& features,
        int feature_count,
        std::array<std::int32_t, EXPECTED_L1>& accumulator
    ) const
    {
        for (int i = 0; i < EXPECTED_L1; i++)
        {
            accumulator[i] = feature_bias_[i];
        }

        for (int feature_index = 0; feature_index < feature_count; feature_index++)
        {
            int feature = features[feature_index];

            if (feature < 0 || feature >= inputs_)
            {
                continue;
            }

            std::size_t base =
                static_cast<std::size_t>(feature) * EXPECTED_L1;

#if defined(__AVX2__)
            int i = 0;

            for (; i + 16 <= EXPECTED_L1; i += 16)
            {
                __m256i weights16 =
                    _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            feature_weights_.data() + base + i
                        )
                    );

                __m128i low128 =
                    _mm256_castsi256_si128(weights16);

                __m128i high128 =
                    _mm256_extracti128_si256(weights16, 1);

                __m256i low32 =
                    _mm256_cvtepi16_epi32(low128);

                __m256i high32 =
                    _mm256_cvtepi16_epi32(high128);

                __m256i acc0 =
                    _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            accumulator.data() + i
                        )
                    );

                __m256i acc1 =
                    _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            accumulator.data() + i + 8
                        )
                    );

                acc0 = _mm256_add_epi32(acc0, low32);
                acc1 = _mm256_add_epi32(acc1, high32);

                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(accumulator.data() + i),
                    acc0
                );

                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(accumulator.data() + i + 8),
                    acc1
                );
            }

            for (; i < EXPECTED_L1; i++)
            {
                accumulator[i] += feature_weights_[base + i];
            }
#else
            for (int i = 0; i < EXPECTED_L1; i++)
            {
                accumulator[i] += feature_weights_[base + i];
            }
#endif
        }

        /*
            Do not clip the raw accumulator.

            Raw accumulator stores:

                bias_q + sum(feature_weight_q)

            Clipping is applied only when building the network input.
        */
    }

    void QuantizedHalfKPNNUE::build_network_input_fixed(
        const std::array<std::int32_t, EXPECTED_L1>& white_accumulator,
        const std::array<std::int32_t, EXPECTED_L1>& black_accumulator,
        bool white_to_move,
        std::array<std::int16_t, CONCATENATED_L1>& input
    ) const
    {
        if (white_to_move)
        {
            for (int i = 0; i < EXPECTED_L1; i++)
            {
                input[i] = clamp_activation_value(white_accumulator[i]);
                input[EXPECTED_L1 + i] =
                    clamp_activation_value(black_accumulator[i]);
            }
        }
        else
        {
            for (int i = 0; i < EXPECTED_L1; i++)
            {
                input[i] = clamp_activation_value(black_accumulator[i]);
                input[EXPECTED_L1 + i] =
                    clamp_activation_value(white_accumulator[i]);
            }
        }
    }

    void QuantizedHalfKPNNUE::run_dense_layer_512_to_32_fixed(
        const std::array<std::int16_t, CONCATENATED_L1>& input,
        std::array<std::int16_t, EXPECTED_L2>& output
    ) const
    {
        for (int out = 0; out < EXPECTED_L2; out++)
        {
            std::int64_t sum =
                static_cast<std::int64_t>(hidden1_bias_[out])
              * static_cast<std::int64_t>(quant_scale_);

            std::size_t weight_base =
                static_cast<std::size_t>(out) * CONCATENATED_L1;

            sum += dot_product_i16_i16(
                input.data(),
                hidden1_weights_.data() + weight_base,
                CONCATENATED_L1
            );

            output[out] = clamp_scaled_sum_to_activation(sum);
        }
    }

    void QuantizedHalfKPNNUE::run_dense_layer_32_to_32_fixed(
        const std::array<std::int16_t, EXPECTED_L2>& input,
        std::array<std::int16_t, EXPECTED_L3>& output
    ) const
    {
        for (int out = 0; out < EXPECTED_L3; out++)
        {
            std::int64_t sum =
                static_cast<std::int64_t>(hidden2_bias_[out])
              * static_cast<std::int64_t>(quant_scale_);

            std::size_t weight_base =
                static_cast<std::size_t>(out) * EXPECTED_L2;

            sum += dot_product_i16_i16(
                input.data(),
                hidden2_weights_.data() + weight_base,
                EXPECTED_L2
            );

            output[out] = clamp_scaled_sum_to_activation(sum);
        }
    }

    std::int64_t QuantizedHalfKPNNUE::run_output_layer_fixed(
        const std::array<std::int16_t, EXPECTED_L3>& input
    ) const
    {
        std::int64_t sum =
            static_cast<std::int64_t>(output_bias_[0])
          * static_cast<std::int64_t>(quant_scale_);

        sum += dot_product_i16_i16(
            input.data(),
            output_weights_.data(),
            EXPECTED_L3
        );

        return sum;
    }

    void QuantizedHalfKPNNUE::refresh_accumulator(
        const Board& board,
        NNUEAccumulator& accumulator
    ) const
    {
        accumulator.valid = false;

        if (!loaded_)
        {
            return;
        }

        std::array<int, HALF_KP_MAX_ACTIVE> white_features;
        std::array<int, HALF_KP_MAX_ACTIVE> black_features;

        int white_feature_count =
            get_active_feature_indices_halfkp_fixed(
                board,
                Color::WHITE,
                white_features
            );

        int black_feature_count =
            get_active_feature_indices_halfkp_fixed(
                board,
                Color::BLACK,
                black_features
            );

        build_accumulator_fixed(
            white_features,
            white_feature_count,
            accumulator.white
        );

        build_accumulator_fixed(
            black_features,
            black_feature_count,
            accumulator.black
        );

        accumulator.valid = true;
    }

    void QuantizedHalfKPNNUE::add_feature_to_accumulator(
        std::array<std::int32_t, EXPECTED_L1>& accumulator,
        int feature
    ) const
    {
        if (feature < 0 || feature >= inputs_)
        {
            return;
        }

        std::size_t base =
            static_cast<std::size_t>(feature) * EXPECTED_L1;

#if defined(__AVX2__)
        int i = 0;

        for (; i + 16 <= EXPECTED_L1; i += 16)
        {
            __m256i weights16 =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        feature_weights_.data() + base + i
                    )
                );

            __m128i low128 =
                _mm256_castsi256_si128(weights16);

            __m128i high128 =
                _mm256_extracti128_si256(weights16, 1);

            __m256i low32 =
                _mm256_cvtepi16_epi32(low128);

            __m256i high32 =
                _mm256_cvtepi16_epi32(high128);

            __m256i acc0 =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        accumulator.data() + i
                    )
                );

            __m256i acc1 =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        accumulator.data() + i + 8
                    )
                );

            acc0 = _mm256_add_epi32(acc0, low32);
            acc1 = _mm256_add_epi32(acc1, high32);

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(accumulator.data() + i),
                acc0
            );

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(accumulator.data() + i + 8),
                acc1
            );
        }

        for (; i < EXPECTED_L1; i++)
        {
            accumulator[i] += feature_weights_[base + i];
        }
#else
        for (int i = 0; i < EXPECTED_L1; i++)
        {
            accumulator[i] += feature_weights_[base + i];
        }
#endif
    }

    void QuantizedHalfKPNNUE::remove_feature_from_accumulator(
        std::array<std::int32_t, EXPECTED_L1>& accumulator,
        int feature
    ) const
    {
        if (feature < 0 || feature >= inputs_)
        {
            return;
        }

        std::size_t base =
            static_cast<std::size_t>(feature) * EXPECTED_L1;

#if defined(__AVX2__)
        int i = 0;

        for (; i + 16 <= EXPECTED_L1; i += 16)
        {
            __m256i weights16 =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        feature_weights_.data() + base + i
                    )
                );

            __m128i low128 =
                _mm256_castsi256_si128(weights16);

            __m128i high128 =
                _mm256_extracti128_si256(weights16, 1);

            __m256i low32 =
                _mm256_cvtepi16_epi32(low128);

            __m256i high32 =
                _mm256_cvtepi16_epi32(high128);

            __m256i acc0 =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        accumulator.data() + i
                    )
                );

            __m256i acc1 =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        accumulator.data() + i + 8
                    )
                );

            acc0 = _mm256_sub_epi32(acc0, low32);
            acc1 = _mm256_sub_epi32(acc1, high32);

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(accumulator.data() + i),
                acc0
            );

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(accumulator.data() + i + 8),
                acc1
            );
        }

        for (; i < EXPECTED_L1; i++)
        {
            accumulator[i] -= feature_weights_[base + i];
        }
#else
        for (int i = 0; i < EXPECTED_L1; i++)
        {
            accumulator[i] -= feature_weights_[base + i];
        }
#endif
    }

    void QuantizedHalfKPNNUE::add_piece_to_both_accumulators(
        NNUEAccumulator& accumulator,
        int white_king_square,
        int black_king_square,
        int piece_square,
        PieceType piece_type,
        Color piece_color
    ) const
    {
        int white_feature = make_halfkp_index(
            white_king_square,
            piece_square,
            piece_type,
            piece_color
        );

        int black_feature = make_halfkp_index(
            black_king_square,
            piece_square,
            piece_type,
            piece_color
        );

        add_feature_to_accumulator(accumulator.white, white_feature);
        add_feature_to_accumulator(accumulator.black, black_feature);
    }

    void QuantizedHalfKPNNUE::remove_piece_from_both_accumulators(
        NNUEAccumulator& accumulator,
        int white_king_square,
        int black_king_square,
        int piece_square,
        PieceType piece_type,
        Color piece_color
    ) const
    {
        int white_feature = make_halfkp_index(
            white_king_square,
            piece_square,
            piece_type,
            piece_color
        );

        int black_feature = make_halfkp_index(
            black_king_square,
            piece_square,
            piece_type,
            piece_color
        );

        remove_feature_from_accumulator(accumulator.white, white_feature);
        remove_feature_from_accumulator(accumulator.black, black_feature);
    }

    void QuantizedHalfKPNNUE::update_accumulator_after_move(
        const Board& board,
        const Move& move,
        const NNUEAccumulator& parent,
        NNUEAccumulator& child
    ) const
    {
        child = parent;

        if (!loaded_ || !parent.valid)
        {
            child.valid = false;
            return;
        }

        Color moving_color = board.sideToMove();

        Piece moving_piece = board.at(move.from());

        PieceType moving_type = moving_piece.type();
        Color piece_color = moving_piece.color();

        if (moving_type == PieceType::KING || move.typeOf() == Move::CASTLING)
        {
            child.valid = false;
            return;
        }

        int white_king_square = board.kingSq(Color::WHITE).index();
        int black_king_square = board.kingSq(Color::BLACK).index();

        int from_square = move.from().index();
        int to_square = move.to().index();

        if (move.typeOf() == Move::PROMOTION)
        {
            remove_piece_from_both_accumulators(
                child,
                white_king_square,
                black_king_square,
                from_square,
                PieceType::PAWN,
                moving_color
            );

            add_piece_to_both_accumulators(
                child,
                white_king_square,
                black_king_square,
                to_square,
                move.promotionType(),
                moving_color
            );
        }
        else
        {
            remove_piece_from_both_accumulators(
                child,
                white_king_square,
                black_king_square,
                from_square,
                moving_type,
                piece_color
            );

            add_piece_to_both_accumulators(
                child,
                white_king_square,
                black_king_square,
                to_square,
                moving_type,
                piece_color
            );
        }

        if (move.typeOf() == Move::ENPASSANT)
        {
            Square captured_square = move.to().ep_square();

            remove_piece_from_both_accumulators(
                child,
                white_king_square,
                black_king_square,
                captured_square.index(),
                PieceType::PAWN,
                ~moving_color
            );
        }
        else
        {
            Piece captured_piece = board.at(move.to());

            if (captured_piece != Piece::NONE)
            {
                remove_piece_from_both_accumulators(
                    child,
                    white_king_square,
                    black_king_square,
                    to_square,
                    captured_piece.type(),
                    captured_piece.color()
                );
            }
        }

        child.valid = true;
    }

    int QuantizedHalfKPNNUE::evaluate_white_pov_from_accumulator(
        const Board& board,
        const NNUEAccumulator& accumulator
    ) const
    {
        if (!loaded_ || !accumulator.valid)
        {
            return 0;
        }

        bool white_to_move = board.sideToMove() == Color::WHITE;

        std::array<std::int16_t, CONCATENATED_L1> input;

        build_network_input_fixed(
            accumulator.white,
            accumulator.black,
            white_to_move,
            input
        );

        std::array<std::int16_t, EXPECTED_L2> hidden1;
        std::array<std::int16_t, EXPECTED_L3> hidden2;

        run_dense_layer_512_to_32_fixed(input, hidden1);
        run_dense_layer_32_to_32_fixed(hidden1, hidden2);

        std::int64_t output_scaled_twice =
            run_output_layer_fixed(hidden2);

        double denominator =
            static_cast<double>(quant_scale_)
          * static_cast<double>(quant_scale_);

        double side_to_move_score =
            static_cast<double>(output_scaled_twice) / denominator;

        double white_pov_score = 0.0;

        if (white_to_move)
        {
            white_pov_score = side_to_move_score;
        }
        else
        {
            white_pov_score = -side_to_move_score;
        }

        int centipawns =
            static_cast<int>(std::round(white_pov_score * OUTPUT_TO_CP));

        return centipawns;
    }

    int QuantizedHalfKPNNUE::evaluate_white_pov(const Board& board) const
    {
        if (!loaded_)
        {
            return 0;
        }

        NNUEAccumulator accumulator;

        refresh_accumulator(board, accumulator);

        return evaluate_white_pov_from_accumulator(board, accumulator);
    }
}