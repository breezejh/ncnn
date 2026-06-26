// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_LSTM(benchmark::State& state)
{
    int input_size = state.range(0);
    int T = state.range(1);           // sequence length
    int hidden_size = state.range(2);
    int direction = state.range(3);   // 0=forward, 1=reverse, 2=bidirectional

    ncnn::Mat a = RandomMat(input_size, T);
    int num_directions = direction == 2 ? 2 : 1;

    ncnn::ParamDict pd;
    pd.set(0, hidden_size);  // num_output
    pd.set(1, hidden_size * input_size * 4 * num_directions);  // weight_data_size
    pd.set(2, direction);
    pd.set(3, hidden_size);  // hidden_size

    std::vector<ncnn::Mat> weights(3);
    weights[0] = RandomMat(hidden_size * input_size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomMat(hidden_size * hidden_size * 4 * num_directions);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("LSTM");
    if (!layer)
    {
        state.SkipWithError("Failed to create layer");
        return;
    }

    int ret = layer->load_param(pd);
    if (ret != 0)
    {
        delete layer;
        state.SkipWithError("Failed to load param");
        return;
    }

    ret = layer->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    if (ret != 0)
    {
        delete layer;
        state.SkipWithError("Failed to load model");
        return;
    }

    // Create pipeline to initialize optimized kernels
    ret = layer->create_pipeline(opt);
    if (ret != 0)
    {
        delete layer;
        state.SkipWithError("Failed to create pipeline");
        return;
    }

    ncnn::Mat b;
    for (auto _ : state)
    {
        layer->forward(a, b, opt);
    }

    // Clean up pipeline resources
    layer->destroy_pipeline(opt);
    delete layer;
}

// Benchmark with different configurations: input_size, T, hidden_size, direction
BENCHMARK(BM_LSTM)
    // ========== Speech Recognition Configurations ==========
    // Typical speech feature dimensions (MFCC, mel-spectrogram)
    ->Args({40, 100, 256, 2})         // 40 MFCC features, 100 frames, bidirectional
    ->Args({80, 200, 512, 2})         // 80 mel features, 200 frames, bidirectional
    ->Args({128, 300, 1024, 2})       // Large model for speech
    
    // ========== NLP/Text Processing ==========
    // Word embeddings
    ->Args({300, 50, 256, 2})         // Word2Vec/GloVe 300d, seq_len=50
    ->Args({512, 100, 512, 2})        // Larger embeddings, seq_len=100
    ->Args({768, 128, 768, 2})        // BERT-like embeddings
    
    // Character-level models
    ->Args({128, 200, 256, 2})        // Character embeddings
    
    // ========== Time Series Prediction ==========
    ->Args({1, 100, 64, 0})           // Univariate time series, forward only
    ->Args({10, 200, 128, 0})         // Multivariate time series
    ->Args({50, 500, 256, 0})         // Long sequence prediction
    
    // ========== Video Analysis ==========
    // Video frame features
    ->Args({2048, 30, 512, 2})        // ResNet features, 30 frames
    ->Args({1024, 60, 256, 2})        // Smaller features, 60 frames
    
    // ========== Different Directions ==========
    // Forward only (direction=0)
    ->Args({256, 100, 256, 0})        // Forward LSTM
    
    // Reverse only (direction=1)
    ->Args({256, 100, 256, 1})        // Reverse LSTM
    
    // Bidirectional (direction=2)
    ->Args({256, 100, 256, 2})        // Bidirectional LSTM
    
    // ========== Different Sequence Lengths ==========
    // Short sequences
    ->Args({128, 10, 128, 2})         // Very short sequence
    ->Args({256, 20, 256, 2})         // Short sequence
    
    // Medium sequences
    ->Args({256, 50, 256, 2})         // Medium sequence
    ->Args({256, 100, 256, 2})        // Standard sequence
    
    // Long sequences
    ->Args({256, 200, 256, 2})        // Long sequence
    ->Args({128, 500, 128, 2})        // Very long sequence
    
    // ========== Different Hidden Sizes ==========
    // Small models
    ->Args({128, 100, 64, 2})         // Small hidden
    ->Args({128, 100, 128, 2})        // Medium hidden
    
    // Large models
    ->Args({256, 100, 512, 2})        // Large hidden
    ->Args({256, 100, 1024, 2})       // Very large hidden
    
    // ========== Different Input Sizes ==========
    // Small input dimensions
    ->Args({32, 100, 128, 2})         // Small input
    ->Args({64, 100, 256, 2})         // Medium input
    
    // Large input dimensions
    ->Args({512, 100, 512, 2})        // Large input
    ->Args({1024, 50, 512, 2})        // Very large input
    
    // ========== Stacked LSTM Simulation ==========
    // First layer output becomes second layer input
    ->Args({256, 100, 256, 2})        // Layer 1
    ->Args({512, 100, 512, 2})        // Layer 2 (bidirectional doubles size)
    
    // ========== Edge Cases ==========
    ->Args({1, 1, 1, 0})              // Minimal configuration
    ->Args({2048, 10, 2048, 2});      // Very large dimensions

BENCHMARK_MAIN();

