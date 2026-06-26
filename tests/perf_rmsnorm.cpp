// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_RMSNorm(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int affine_size = state.range(3);
    int affine = state.range(4);

    ncnn::Mat a;
    if (c == 0 && h == 0)
    {
        a = RandomMat(w);  // 1D
    }
    else if (c == 0)
    {
        a = RandomMat(w, h);  // 2D
    }
    else
    {
        a = RandomMat(w, h, c);  // 3D
    }

    ncnn::ParamDict pd;
    pd.set(0, affine_size);
    pd.set(1, 1e-5f);  // eps
    pd.set(2, affine);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(affine_size);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("RMSNorm");
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

// Benchmark with different configurations: w, h, c, affine_size, affine
BENCHMARK(BM_RMSNorm)
    // ========== LLaMA/LLaMA-2 Configurations ==========
    // LLaMA-7B: hidden_dim=4096
    ->Args({4096, 2048, 1, 4096, 1})          // Sequence length 2048
    ->Args({4096, 1024, 1, 4096, 1})          // Sequence length 1024
    ->Args({4096, 512, 1, 4096, 1})           // Sequence length 512
    
    // LLaMA-13B: hidden_dim=5120
    ->Args({5120, 2048, 1, 5120, 1})          // Sequence length 2048
    ->Args({5120, 512, 1, 5120, 1})           // Sequence length 512
    
    // LLaMA-33B/65B: hidden_dim=6656/8192
    ->Args({6656, 1024, 1, 6656, 1})          // 33B model
    ->Args({8192, 512, 1, 8192, 1})           // 65B model
    
    // ========== Mistral/Mixtral Configurations ==========
    // Mistral-7B: hidden_dim=4096
    ->Args({4096, 4096, 1, 4096, 1})          // Long context (4k)
    ->Args({4096, 8192, 1, 4096, 1})          // Extended context (8k)
    
    // ========== GPT-NeoX/Pythia Style ==========
    // Various model sizes
    ->Args({2048, 2048, 1, 2048, 1})          // 1.4B model
    ->Args({2560, 2048, 1, 2560, 1})          // 2.8B model
    ->Args({3072, 1024, 1, 3072, 1})          // Larger model
    
    // ========== Smaller Models (Mobile/Edge) ==========
    ->Args({512, 512, 1, 512, 1})             // Small model
    ->Args({768, 1024, 1, 768, 1})            // Medium model
    ->Args({1024, 2048, 1, 1024, 1})          // Larger mobile model
    
    // ========== Without Affine Transform ==========
    ->Args({4096, 512, 1, 4096, 0})           // LLaMA-7B, no affine
    ->Args({2048, 1024, 1, 2048, 0})          // Medium, no affine
    
    // ========== Different Sequence Lengths ==========
    // Short sequences
    ->Args({4096, 128, 1, 4096, 1})           // Very short
    ->Args({4096, 256, 1, 4096, 1})           // Short
    
    // Long sequences
    ->Args({4096, 4096, 1, 4096, 1})          // Long (4k)
    ->Args({4096, 8192, 1, 4096, 1})          // Very long (8k)
    
    // ========== Batch Processing ==========
    // Simulating batch dimension in height
    ->Args({4096, 32, 1, 4096, 1})            // Batch=32
    ->Args({4096, 64, 1, 4096, 1})            // Batch=64
    ->Args({2048, 128, 1, 2048, 1})           // Smaller model, batch=128
    
    // ========== Edge Cases ==========
    ->Args({128, 128, 1, 128, 1})             // Very small
    ->Args({16384, 256, 1, 16384, 1});        // Very large hidden dim

BENCHMARK_MAIN();

