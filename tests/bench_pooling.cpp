// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_Pooling(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int pooling_type = state.range(3);  // 0=max, 1=avg
    int kernel = state.range(4);
    int stride = state.range(5);
    int pad = state.range(6);
    int global_pooling = state.range(7);

    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, pooling_type);
    pd.set(1, kernel);
    pd.set(2, stride);
    pd.set(3, pad);
    pd.set(4, global_pooling);
    pd.set(5, 1);  // pad_mode (REPLICATE)
    pd.set(6, 1);  // avgpool_count_include_pad
    pd.set(7, 0);  // adaptive_pooling (disabled)
    pd.set(8, 0);  // out_w (not used when adaptive_pooling=0)

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("Pooling");
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

// Benchmark with different configurations: w, h, c, pooling_type, kernel, stride, pad, global_pooling
BENCHMARK(BM_Pooling)
    // ========== Max Pooling - Common CNN Configurations ==========
    // AlexNet/VGG style: 3x3 kernel, stride=2
    // ->Args({224, 224, 64, 0, 3, 2, 1, 0})     // Early layers, large input
    // ->Args({112, 112, 128, 0, 3, 2, 1, 0})    // Middle layers
    // ->Args({56, 56, 256, 0, 3, 2, 1, 0})      // Deeper layers

    // // ResNet style: 2x2 kernel, stride=2
    // ->Args({112, 112, 64, 0, 2, 2, 0, 0})     // ResNet initial pooling
    // ->Args({56, 56, 128, 0, 2, 2, 0, 0})      // ResNet downsampling
    // ->Args({28, 28, 256, 0, 2, 2, 0, 0})      // ResNet deeper downsampling

    // // ========== Average Pooling - Common Configurations ==========
    // // Inception/GoogLeNet style
    // ->Args({28, 28, 192, 1, 3, 1, 1, 0})      // Inception module
    // ->Args({14, 14, 384, 1, 3, 1, 1, 0})      // Deeper Inception

    // // MobileNet style: stride=2 downsampling
    // ->Args({112, 112, 32, 1, 3, 2, 1, 0})     // MobileNet early
    // ->Args({56, 56, 64, 1, 3, 2, 1, 0})       // MobileNet middle
    // ->Args({28, 28, 128, 1, 3, 2, 1, 0})      // MobileNet deeper

    // // ========== Global Pooling ==========
    // // Global Average Pooling (common in modern architectures)
    // ->Args({7, 7, 512, 1, 1, 1, 0, 1})        // ResNet final GAP
    // ->Args({7, 7, 2048, 1, 1, 1, 0, 1})       // ResNet-50/101/152 final GAP
    // ->Args({14, 14, 1024, 1, 1, 1, 0, 1})     // Larger feature maps

    // // Global Max Pooling
    // ->Args({7, 7, 512, 0, 1, 1, 0, 1})        // ResNet final GMP
    // ->Args({14, 14, 256, 0, 1, 1, 0, 1})      // Medium feature maps

    // // ========== Different Kernel Sizes ==========
    // // Small kernels
    ->Args({56, 56, 128, 0, 2, 2, 0, 0})      // 2x2 max pooling
    ->Args({56, 56, 128, 1, 2, 2, 0, 0})      // 2x2 avg pooling

    // // Large kernels
    // ->Args({28, 28, 256, 0, 5, 2, 2, 0})      // 5x5 max pooling
    // ->Args({28, 28, 256, 1, 5, 2, 2, 0})      // 5x5 avg pooling

    // // ========== Different Strides ==========
    // // Stride=1 (no downsampling)
    // ->Args({56, 56, 128, 0, 3, 1, 1, 0})      // Max pool, stride=1
    // ->Args({56, 56, 128, 1, 3, 1, 1, 0})      // Avg pool, stride=1

    // // Stride=3
    // ->Args({56, 56, 128, 0, 3, 3, 1, 0})      // Max pool, stride=3

    // // ========== Different Channel Counts ==========
    // // Small channel counts
    // ->Args({112, 112, 16, 0, 2, 2, 0, 0})     // Few channels
    // ->Args({112, 112, 32, 1, 2, 2, 0, 0})     // Small channels

    // Large channel counts
    ->Args({28, 28, 512, 0, 2, 2, 0, 0})      // Many channels
    ->Args({14, 14, 1024, 1, 2, 2, 0, 0})     // Very many channels

    // ========== Edge Cases ==========
    ->Args({224, 224, 3, 0, 2, 2, 0, 0})      // RGB input
    ->Args({8, 8, 8, 0, 2, 2, 0, 0});         // Very small

BENCHMARK_MAIN();

