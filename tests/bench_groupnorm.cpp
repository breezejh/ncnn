// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_GroupNorm(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int group = state.range(3);
    int affine = state.range(4);
    float eps = 0.001f;

    // Validate parameters
    if (c % group != 0)
    {
        state.SkipWithError("Invalid group configuration: channels must be divisible by group");
        return;
    }

    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, group);
    pd.set(1, c);
    pd.set(2, eps);
    pd.set(3, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(c);
    weights[1] = RandomMat(c);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("GroupNorm");
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

// Benchmark with different configurations: w, h, c, group, affine
BENCHMARK(BM_GroupNorm)
    // ========== Common Vision Transformer Configurations ==========
    // ViT and related architectures often use GroupNorm
    ->Args({14, 14, 384, 1, 1})      // ViT-Base patch features, LayerNorm equivalent
    ->Args({14, 14, 384, 32, 1})     // ViT-Base with 32 groups
    ->Args({14, 14, 768, 1, 1})      // ViT-Large, LayerNorm equivalent
    ->Args({14, 14, 768, 32, 1})     // ViT-Large with 32 groups
    
    // ========== ResNet-style Configurations ==========
    // GroupNorm as replacement for BatchNorm
    ->Args({56, 56, 64, 32, 1})      // ResNet early layers, 32 groups
    ->Args({56, 56, 128, 32, 1})     // ResNet middle layers
    ->Args({28, 28, 256, 32, 1})     // ResNet deeper layers
    ->Args({14, 14, 512, 32, 1})     // ResNet very deep layers
    ->Args({7, 7, 2048, 32, 1})      // ResNet final layers
    
    // ========== Different Group Counts ==========
    // Single group (equivalent to LayerNorm)
    ->Args({28, 28, 128, 1, 1})      // LayerNorm equivalent
    ->Args({56, 56, 256, 1, 1})      // Large LayerNorm
    
    // Few groups
    ->Args({56, 56, 128, 2, 1})      // 2 groups
    ->Args({28, 28, 256, 4, 1})      // 4 groups
    ->Args({28, 28, 128, 8, 1})      // 8 groups
    
    // Many groups
    ->Args({28, 28, 256, 16, 1})     // 16 groups
    ->Args({14, 14, 512, 16, 1})     // 16 groups, smaller spatial
    
    // ========== With and Without Affine ==========
    ->Args({28, 28, 128, 32, 0})     // Without affine transformation
    ->Args({28, 28, 128, 32, 1})     // With affine transformation
    ->Args({56, 56, 256, 32, 0})     // Larger, without affine
    ->Args({56, 56, 256, 32, 1})     // Larger, with affine
    
    // ========== Small Feature Maps (Deep Layers) ==========
    ->Args({7, 7, 512, 32, 1})       // Small spatial size
    ->Args({7, 7, 1024, 32, 1})      // Very deep network
    ->Args({4, 4, 2048, 32, 1})      // Extremely deep
    
    // ========== Large Feature Maps (Early Layers) ==========
    ->Args({112, 112, 64, 32, 1})    // Very large spatial
    ->Args({224, 224, 32, 16, 1})    // Input-level features
    
    // ========== Different Channel Counts ==========
    ->Args({28, 28, 32, 16, 1})      // Small channels
    ->Args({28, 28, 96, 32, 1})      // Medium channels
    ->Args({14, 14, 192, 32, 1})     // Larger channels
    ->Args({14, 14, 384, 32, 1})     // Large channels
    
    // ========== 3D Feature Maps ==========
    // For video or volumetric data
    ->Args({8, 8, 256, 32, 1})       // 3D-like small spatial
    ->Args({16, 16, 128, 32, 1})     // 3D-like medium spatial
    
    // ========== Edge Cases ==========
    ->Args({56, 56, 8, 4, 1})        // Very small channels
    ->Args({7, 7, 4096, 32, 1});     // Extremely large channels

BENCHMARK_MAIN();

