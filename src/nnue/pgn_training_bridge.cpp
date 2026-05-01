#include "../../include/pgn_training_bridge.hpp"
#include "../../include/nnue_features.hpp"
#include "../../include/evaluation.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace chessengine
{
    using namespace chess;

    constexpr int MAX_ACTIVE = 32;

    static constexpr const char* START_FEN =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    struct TrainingPosition
    {
        Board board;
        float target;
    };

    struct PgnMoveToken
    {
        std::string san;
        bool has_eval = false;
        float eval_target_white = 0.0f;
    };

    static bool is_result_token(const std::string& token)
    {
        return token == "1-0" ||
               token == "0-1" ||
               token == "1/2-1/2" ||
               token == "*";
    }

    static bool line_has_result_token(const std::string& line)
    {
        std::istringstream iss(line);
        std::string token;

        while (iss >> token)
        {
            if (is_result_token(token))
                return true;
        }

        return false;
    }

    static float result_to_white_target(const std::string& result)
    {
        if (result == "1-0")
            return 1.0f;

        if (result == "0-1")
            return -1.0f;

        return 0.0f;
    }

    static float centipawn_to_target(float centipawns)
    {
        return static_cast<float>(std::tanh(centipawns / 600.0f));
    }

    static bool parse_eval_comment(const std::string& comment, float& target_white)
    {
        std::size_t pos = comment.find("%eval");

        if (pos == std::string::npos)
            return false;

        pos += 5;

        while (pos < comment.size() &&
               std::isspace(static_cast<unsigned char>(comment[pos])))
        {
            pos++;
        }

        if (pos >= comment.size())
            return false;

        std::string value;

        while (pos < comment.size())
        {
            char c = comment[pos];

            if (std::isspace(static_cast<unsigned char>(c)) || c == ']')
                break;

            value.push_back(c);
            pos++;
        }

        if (value.empty())
            return false;

        if (value[0] == '#')
        {
            if (value.size() >= 2 && value[1] == '-')
                target_white = -1.0f;
            else
                target_white = 1.0f;

            return true;
        }

        char* end_ptr = nullptr;
        float pawns = std::strtof(value.c_str(), &end_ptr);

        if (end_ptr == value.c_str())
            return false;

        float centipawns = pawns * 100.0f;
        target_white = centipawn_to_target(centipawns);

        return true;
    }

    static std::string clean_san(std::string san)
    {
        while (!san.empty())
        {
            char c = san.back();

            if (c == '+' || c == '#' || c == '!' || c == '?')
                san.pop_back();
            else
                break;
        }

        return san;
    }

    static bool is_piece_letter(char c)
    {
        return c == 'N' || c == 'B' || c == 'R' || c == 'Q' || c == 'K';
    }

    static PieceType san_piece_type(const std::string& san)
    {
        if (san.empty())
            return PieceType::PAWN;

        if (san[0] == 'N') return PieceType::KNIGHT;
        if (san[0] == 'B') return PieceType::BISHOP;
        if (san[0] == 'R') return PieceType::ROOK;
        if (san[0] == 'Q') return PieceType::QUEEN;
        if (san[0] == 'K') return PieceType::KING;

        return PieceType::PAWN;
    }

    static PieceType promotion_piece_type(char c)
    {
        if (c == 'N') return PieceType::KNIGHT;
        if (c == 'B') return PieceType::BISHOP;
        if (c == 'R') return PieceType::ROOK;
        if (c == 'Q') return PieceType::QUEEN;

        return PieceType::NONE;
    }

    static int square_index_from_name(char file_char, char rank_char)
    {
        int file = file_char - 'a';
        int rank = rank_char - '1';

        if (file < 0 || file >= 8 || rank < 0 || rank >= 8)
            return -1;

        return rank * 8 + file;
    }

    static int move_from_square(const Move& move)
    {
        return move.from().index();
    }

    static int move_to_square(const Move& move)
    {
        return move.to().index();
    }

    static PieceType move_promotion_type(const Move& move)
    {
        return move.promotionType();
    }

    static PieceType piece_on_square_type(const Board& board, int square)
    {
        Bitboard mask = Bitboard(1) << square;

        for (Color color : {Color::WHITE, Color::BLACK})
        {
            for (PieceType piece_type : {
                PieceType::PAWN,
                PieceType::KNIGHT,
                PieceType::BISHOP,
                PieceType::ROOK,
                PieceType::QUEEN,
                PieceType::KING
            })
            {
                if (board.pieces(piece_type, color) & mask)
                    return piece_type;
            }
        }

        return PieceType::NONE;
    }

    static Color board_side_to_move(const Board& board)
    {
        return board.sideToMove();
    }

    static std::vector<Move> generate_legal_moves(const Board& board)
    {
        Movelist moves;
        movegen::legalmoves(moves, board);

        std::vector<Move> result;

        for (const Move& move : moves)
            result.push_back(move);

        return result;
    }

    static bool san_matches_move(
        const Board& board,
        const std::string& raw_san,
        const Move& move
    )
    {
        std::string san = clean_san(raw_san);

        if (san.empty())
            return false;

        if (san == "O-O" || san == "0-0")
        {
            PieceType moved_piece = piece_on_square_type(board, move_from_square(move));

            return moved_piece == PieceType::KING &&
                   move_to_square(move) - move_from_square(move) == 2;
        }

        if (san == "O-O-O" || san == "0-0-0")
        {
            PieceType moved_piece = piece_on_square_type(board, move_from_square(move));

            return moved_piece == PieceType::KING &&
                   move_from_square(move) - move_to_square(move) == 2;
        }

        PieceType wanted_piece = san_piece_type(san);

        int start_index = is_piece_letter(san[0]) ? 1 : 0;

        std::size_t promotion_pos = san.find('=');
        PieceType wanted_promotion = PieceType::NONE;

        if (promotion_pos != std::string::npos && promotion_pos + 1 < san.size())
            wanted_promotion = promotion_piece_type(san[promotion_pos + 1]);

        std::string no_promo = san.substr(0, promotion_pos);

        if (no_promo.size() < 2)
            return false;

        char dest_file = no_promo[no_promo.size() - 2];
        char dest_rank = no_promo[no_promo.size() - 1];

        int wanted_to = square_index_from_name(dest_file, dest_rank);

        if (wanted_to == -1)
            return false;

        int from = move_from_square(move);
        int to = move_to_square(move);

        if (to != wanted_to)
            return false;

        PieceType moved_piece = piece_on_square_type(board, from);

        if (moved_piece != wanted_piece)
            return false;

        if (wanted_promotion != PieceType::NONE)
        {
            if (move_promotion_type(move) != wanted_promotion)
                return false;
        }

        std::string middle = no_promo.substr(
            start_index,
            no_promo.size() - start_index - 2
        );

        middle.erase(
            std::remove(middle.begin(), middle.end(), 'x'),
            middle.end()
        );

        int from_file = from % 8;
        int from_rank = from / 8;

        for (char c : middle)
        {
            if (c >= 'a' && c <= 'h')
            {
                if (from_file != c - 'a')
                    return false;
            }
            else if (c >= '1' && c <= '8')
            {
                if (from_rank != c - '1')
                    return false;
            }
        }

        return true;
    }

    static bool parse_san_move(
        const Board& board,
        const std::string& san,
        Move& parsed_move
    )
    {
        std::vector<Move> legal_moves = generate_legal_moves(board);

        for (const Move& move : legal_moves)
        {
            if (san_matches_move(board, san, move))
            {
                parsed_move = move;
                return true;
            }
        }

        return false;
    }

    static bool is_move_number_token(const std::string& token)
    {
        if (token.empty())
            return false;

        for (char c : token)
        {
            if (!std::isdigit(static_cast<unsigned char>(c)) && c != '.')
                return false;
        }

        return true;
    }

    static std::string remove_move_number_prefix(const std::string& token)
    {
        std::size_t dot_pos = token.find_last_of('.');

        if (dot_pos == std::string::npos)
            return token;

        if (dot_pos + 1 >= token.size())
            return "";

        return token.substr(dot_pos + 1);
    }

    static void add_san_token(
        std::vector<PgnMoveToken>& tokens,
        const std::string& raw_token
    )
    {
        if (raw_token.empty())
            return;

        if (raw_token[0] == '$')
            return;

        if (is_result_token(raw_token))
            return;

        if (is_move_number_token(raw_token))
            return;

        std::string token = remove_move_number_prefix(raw_token);

        if (token.empty())
            return;

        if (is_result_token(token))
            return;

        if (is_move_number_token(token))
            return;

        PgnMoveToken move_token;
        move_token.san = token;

        tokens.push_back(move_token);
    }

    static std::vector<PgnMoveToken> tokenize_pgn_movetext_with_eval(
        const std::string& text
    )
    {
        std::vector<PgnMoveToken> tokens;
        std::string current;

        bool in_comment = false;
        int variation_depth = 0;
        std::string comment;

        auto flush_current = [&]()
        {
            if (!current.empty())
            {
                add_san_token(tokens, current);
                current.clear();
            }
        };

        for (std::size_t i = 0; i < text.size(); i++)
        {
            char c = text[i];

            if (in_comment)
            {
                if (c == '}')
                {
                    in_comment = false;

                    float eval_target_white = 0.0f;

                    if (!tokens.empty() &&
                        parse_eval_comment(comment, eval_target_white))
                    {
                        tokens.back().has_eval = true;
                        tokens.back().eval_target_white = eval_target_white;
                    }

                    comment.clear();
                    continue;
                }

                comment.push_back(c);
                continue;
            }

            if (c == '{')
            {
                flush_current();
                in_comment = true;
                comment.clear();
                continue;
            }

            if (c == ';')
            {
                flush_current();

                while (i < text.size() && text[i] != '\n')
                    i++;

                continue;
            }

            if (c == '(')
            {
                flush_current();
                variation_depth++;
                continue;
            }

            if (c == ')')
            {
                flush_current();

                if (variation_depth > 0)
                    variation_depth--;

                continue;
            }

            if (variation_depth > 0)
                continue;

            if (std::isspace(static_cast<unsigned char>(c)))
            {
                flush_current();
                continue;
            }

            current.push_back(c);
        }

        flush_current();

        return tokens;
    }

    class PgnTrainingReader
    {
    public:
        explicit PgnTrainingReader(const std::string& path)
            : input_(path)
        {
            if (!input_)
            {
                std::cout << "PGN reader could not open file: " << path << '\n';
            }
            else
            {
                std::cout << "PGN reader opened file: " << path << '\n';
            }
        }

        bool next_position(TrainingPosition& position)
        {
            while (position_buffer_.empty())
            {
                if (!load_next_game())
                    return false;
            }

            position = position_buffer_.back();
            position_buffer_.pop_back();

            return true;
        }

    private:
        std::ifstream input_;
        std::vector<TrainingPosition> position_buffer_;

        bool load_next_game()
        {
            if (!input_)
                return false;

            static int debug_game_index = 0;

            std::string line;
            std::string movetext;
            std::string result = "*";

            bool started = false;
            bool saw_movetext = false;

            while (std::getline(input_, line))
            {
                if (!started && line.empty())
                    continue;

                if (!line.empty())
                    started = true;

                bool is_tag_line = !line.empty() && line[0] == '[';

                if (is_tag_line && !saw_movetext)
                {
                    if (line.rfind("[Result ", 0) == 0)
                    {
                        std::size_t first_quote = line.find('"');
                        std::size_t second_quote = line.find('"', first_quote + 1);

                        if (first_quote != std::string::npos &&
                            second_quote != std::string::npos)
                        {
                            result = line.substr(
                                first_quote + 1,
                                second_quote - first_quote - 1
                            );
                        }
                    }

                    continue;
                }

                if (line.empty() && !saw_movetext)
                    continue;

                saw_movetext = true;

                movetext += line;
                movetext += '\n';

                if (line_has_result_token(line))
                    break;
            }

            if (!started)
                return false;

            std::vector<PgnMoveToken> tokens =
                tokenize_pgn_movetext_with_eval(movetext);

            float white_result = result_to_white_target(result);

            Board board(START_FEN);

            int ply = 0;
            int collected = 0;
            int parsed_moves = 0;
            int failed_tokens = 0;

            std::vector<TrainingPosition> game_positions;

            for (const PgnMoveToken& token : tokens)
            {
                Move move;

                if (!parse_san_move(board, token.san, move))
                {
                    failed_tokens++;
                    continue;
                }

                board.makeMove(move);
                ply++;
                parsed_moves++;

                if (ply < 8)
                    continue;

                if (collected >= 24)
                    continue;

                if (ply % 2 != 0 && ply % 3 != 0)
                    continue;

                float white_pov_target;

                if (token.has_eval)
                    white_pov_target = token.eval_target_white;
                else
                    white_pov_target = white_result;

                float side_to_move_target;

                if (board_side_to_move(board) == Color::WHITE)
                    side_to_move_target = white_pov_target;
                else
                    side_to_move_target = -white_pov_target;

                TrainingPosition pos;
                pos.board = board;
                pos.target = side_to_move_target;

                game_positions.push_back(pos);
                collected++;
            }

            if (debug_game_index < 20)
            {
                std::cout << "DEBUG PGN game " << debug_game_index
                          << " result=" << result
                          << " tokens=" << tokens.size()
                          << " parsed_moves=" << parsed_moves
                          << " failed_tokens=" << failed_tokens
                          << " positions=" << game_positions.size();

                if (!tokens.empty())
                {
                    std::cout << " first_token=" << tokens.front().san;
                }

                std::cout << '\n';
            }

            debug_game_index++;

            std::reverse(game_positions.begin(), game_positions.end());

            for (const TrainingPosition& pos : game_positions)
                position_buffer_.push_back(pos);

            return true;
        }
    };
}

extern "C"
{
    const char* pgn_bridge_version()
    {
        return "PGN_BRIDGE_DEBUG_VERSION_2026_04_30";
    }

    void* create_pgn_training_reader(const char* pgn_path)
    {
        if (pgn_path == nullptr)
            return nullptr;

        return new chessengine::PgnTrainingReader(pgn_path);
    }

    SimpleHalfKPBatch* get_next_pgn_training_batch(
        void* reader_ptr,
        int batch_size
    )
    {
        if (reader_ptr == nullptr || batch_size <= 0)
            return nullptr;

        auto* reader = static_cast<chessengine::PgnTrainingReader*>(reader_ptr);

        auto* batch = new SimpleHalfKPBatch();

        batch->size = 0;
        batch->max_active = chessengine::MAX_ACTIVE;

        batch->white_indices = new int[batch_size * chessengine::MAX_ACTIVE];
        batch->black_indices = new int[batch_size * chessengine::MAX_ACTIVE];

        batch->white_counts = new int[batch_size];
        batch->black_counts = new int[batch_size];

        batch->stm = new float[batch_size];
        batch->target = new float[batch_size];

        std::fill(
            batch->white_indices,
            batch->white_indices + batch_size * chessengine::MAX_ACTIVE,
            -1
        );

        std::fill(
            batch->black_indices,
            batch->black_indices + batch_size * chessengine::MAX_ACTIVE,
            -1
        );

        for (int i = 0; i < batch_size; i++)
        {
            chessengine::TrainingPosition position;

            if (!reader->next_position(position))
                break;

            std::vector<int> white =
                chessengine::get_active_feature_indices_halfkp(
                    position.board,
                    chess::Color::WHITE
                );

            std::vector<int> black =
                chessengine::get_active_feature_indices_halfkp(
                    position.board,
                    chess::Color::BLACK
                );

            int white_count = std::min(
                static_cast<int>(white.size()),
                chessengine::MAX_ACTIVE
            );

            int black_count = std::min(
                static_cast<int>(black.size()),
                chessengine::MAX_ACTIVE
            );

            batch->white_counts[i] = white_count;
            batch->black_counts[i] = black_count;

            for (int j = 0; j < white_count; j++)
                batch->white_indices[i * chessengine::MAX_ACTIVE + j] = white[j];

            for (int j = 0; j < black_count; j++)
                batch->black_indices[i * chessengine::MAX_ACTIVE + j] = black[j];

            batch->stm[i] =
                position.board.sideToMove() == chess::Color::WHITE ? 1.0f : 0.0f;

            batch->target[i] = position.target;

            batch->size++;
        }

        if (batch->size == 0)
        {
            destroy_simple_halfkp_batch(batch);
            return nullptr;
        }

        return batch;
    }

    void destroy_simple_halfkp_batch(SimpleHalfKPBatch* batch)
    {
        if (batch == nullptr)
            return;

        delete[] batch->white_indices;
        delete[] batch->black_indices;
        delete[] batch->white_counts;
        delete[] batch->black_counts;
        delete[] batch->stm;
        delete[] batch->target;

        delete batch;
    }

    void destroy_pgn_training_reader(void* reader_ptr)
    {
        auto* reader = static_cast<chessengine::PgnTrainingReader*>(reader_ptr);
        delete reader;
    }
}