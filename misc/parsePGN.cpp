#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#define CHESS_NO_EXCEPTIONS
#include "chess-library-master/include/chess.hpp"

using namespace chess;

class MyVisitor : public pgn::Visitor {
public:
    void startPgn() override {
        std::cout << "==============================" << std::endl;
        std::cout << "New Game Found:" << std::endl;
    }

    void header(std::string_view key, std::string_view value) override {
        std::cout << "  " << key << ": " << value << std::endl;
    }

    void startMoves() override {
        std::cout << "\nMoves:\n";
    }

    void move(std::string_view move, std::string_view comment) override {
        std::cout << move;
        if (!comment.empty()) {
            std::cout << " {" << comment << "}";
        }
        std::cout << " ";
    }

    void endPgn() override {
        std::cout << "\n==============================\n" << std::endl;
    }
};

int main(int argc, char const *argv[]) {
    const auto file = "games.pgn";
    std::ifstream file_stream(file);

    if (!file_stream.is_open()) {
        std::cerr << "Error: could not open file '" << file << "'." << std::endl;
        return 1;
    }

    auto visitor = std::make_unique<MyVisitor>();
    pgn::StreamParser parser(file_stream);

    std::cout << "Parsing PGN file: " << file << "\n\n";

    parser.readGames(*visitor);

    std::cout << "Finished reading all games." << std::endl;
    return 0;
}
