#pragma once

extern "C"
{
    struct SimpleHalfKPBatch
    {
        int size;
        int max_active;

        int* white_indices;
        int* black_indices;

        int* white_counts;
        int* black_counts;

        float* stm;
        float* target;
    };

    void* create_pgn_training_reader(const char* pgn_path);

    SimpleHalfKPBatch* get_next_pgn_training_batch(
        void* reader_ptr,
        int batch_size
    );

    void destroy_simple_halfkp_batch(SimpleHalfKPBatch* batch);

    void destroy_pgn_training_reader(void* reader_ptr);
}