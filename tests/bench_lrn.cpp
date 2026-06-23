// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_LRN(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int region_type = state.range(3);
    int local_size = state.range(4);

    // Use standard LRN parameters from AlexNet
    float alpha = 0.0001f;
    float beta = 0.75f;
    float bias = 1.0f;

    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, region_type);
    pd.set(1, local_size);
    pd.set(2, alpha);
    pd.set(3, beta);
    pd.set(4, bias);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("LRN");
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

// Benchmark with different configurations: w, h, c, region_type, local_size
// region_type: 0 = ACROSS_CHANNELS, 1 = WITHIN_CHANNEL
BENCHMARK(BM_LRN)
    // ========== AlexNet Configurations ==========
    // AlexNet uses ACROSS_CHANNELS normalization with local_size=5
    ->Args({55, 55, 96, 0, 5})        // AlexNet conv1 output
    ->Args({27, 27, 256, 0, 5})       // AlexNet conv2 output (after pooling)
    
    // ========== GoogLeNet/Inception Configurations ==========
    // GoogLeNet also uses LRN in early versions
    ->Args({56, 56, 64, 0, 5})        // GoogLeNet early layers
    ->Args({28, 28, 192, 0, 5})       // GoogLeNet after first inception
    
    // ========== Different Region Types ==========
    // ACROSS_CHANNELS (region_type=0): normalize across channels at same spatial location
    ->Args({56, 56, 128, 0, 3})       // Small local_size
    ->Args({56, 56, 128, 0, 5})       // Medium local_size (most common)
    ->Args({56, 56, 128, 0, 7})       // Large local_size
    
    // WITHIN_CHANNEL (region_type=1): normalize within same channel spatially
    ->Args({56, 56, 128, 1, 3})       // Spatial normalization, small window
    ->Args({56, 56, 128, 1, 5})       // Spatial normalization, medium window
    ->Args({56, 56, 128, 1, 7})       // Spatial normalization, large window
    
    // ========== Different Channel Counts ==========
    // Small channel counts
    ->Args({112, 112, 32, 0, 5})      // Few channels, large spatial
    ->Args({56, 56, 64, 0, 5})        // Medium channels
    
    // Large channel counts
    ->Args({28, 28, 256, 0, 5})       // Many channels
    ->Args({14, 14, 512, 0, 5})       // Very many channels
    ->Args({7, 7, 1024, 0, 5})        // Extremely many channels
    
    // ========== Different Spatial Sizes ==========
    // Large feature maps
    ->Args({224, 224, 64, 0, 5})      // Very large spatial
    ->Args({112, 112, 128, 0, 5})     // Large spatial
    
    // Small feature maps
    ->Args({14, 14, 256, 0, 5})       // Small spatial
    ->Args({7, 7, 512, 0, 5})         // Very small spatial
    
    // ========== Different Local Sizes ==========
    // Testing various local_size values with ACROSS_CHANNELS
    ->Args({28, 28, 128, 0, 1})       // Minimal local_size
    ->Args({28, 28, 128, 0, 3})       // Small local_size
    ->Args({28, 28, 128, 0, 5})       // Standard local_size
    ->Args({28, 28, 128, 0, 7})       // Large local_size
    ->Args({28, 28, 128, 0, 9})       // Very large local_size
    
    // ========== Edge Cases ==========
    ->Args({7, 7, 8, 0, 5})           // Small everything
    ->Args({224, 224, 3, 0, 3});      // RGB input normalization

BENCHMARK_MAIN();

