// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_GridSample(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int grid_w = state.range(3);
    int grid_h = state.range(4);
    int sample_type = state.range(5);      // 1=bilinear, 2=nearest, 3=bicubic
    int padding_mode = state.range(6);     // 1=zeros, 2=border, 3=reflection
    int align_corner = state.range(7);     // 0 or 1
    int permute_fusion = 0;                // Fixed to 0 for standard grid format

    ncnn::Mat a = RandomMat(w, h, c);
    ncnn::Mat grid = RandomMat(2, grid_w, grid_h);

    ncnn::ParamDict pd;
    pd.set(0, sample_type);
    pd.set(1, padding_mode);
    pd.set(2, align_corner);
    pd.set(3, permute_fusion);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("GridSample");
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

    std::vector<ncnn::Mat> a_vec(2);
    a_vec[0] = a;
    a_vec[1] = grid;

    std::vector<ncnn::Mat> b(1);
    for (auto _ : state)
    {
        layer->forward(a_vec, b, opt);
    }

    // Clean up pipeline resources
    layer->destroy_pipeline(opt);
    delete layer;
}

// Benchmark with different configurations: w, h, c, grid_w, grid_h, sample_type, padding_mode, align_corner
BENCHMARK(BM_GridSample)
    // ========== 2D GridSample - Bilinear Interpolation ==========
    // Common configurations for spatial transformer networks
    // Small feature maps
    ->Args({28, 28, 64, 28, 28, 1, 1, 0})      // MNIST-like, zeros padding
    ->Args({28, 28, 64, 28, 28, 1, 2, 0})      // Border padding
    ->Args({28, 28, 64, 28, 28, 1, 1, 1})      // Align corners
    
    // Medium feature maps (typical CNN intermediate layers)
    ->Args({56, 56, 128, 56, 56, 1, 1, 0})     // ResNet-like
    ->Args({56, 56, 128, 56, 56, 1, 2, 0})     // Border padding
    ->Args({56, 56, 128, 56, 56, 1, 3, 0})     // Reflection padding
    
    // Large feature maps
    ->Args({112, 112, 64, 112, 112, 1, 1, 0})  // High resolution
    ->Args({112, 112, 64, 112, 112, 1, 2, 1})  // Border + align corners
    
    // Downsampling scenarios (grid smaller than input)
    ->Args({112, 112, 64, 56, 56, 1, 1, 0})    // 2x downsampling
    ->Args({56, 56, 128, 28, 28, 1, 1, 0})     // 2x downsampling
    ->Args({112, 112, 64, 28, 28, 1, 2, 0})    // 4x downsampling
    
    // Upsampling scenarios (grid larger than input)
    ->Args({28, 28, 128, 56, 56, 1, 1, 0})     // 2x upsampling
    ->Args({56, 56, 64, 112, 112, 1, 1, 0})    // 2x upsampling
    ->Args({28, 28, 64, 112, 112, 1, 2, 0})    // 4x upsampling
    
    // ========== Nearest Neighbor Interpolation ==========
    // Faster but lower quality
    ->Args({56, 56, 128, 56, 56, 2, 1, 0})     // Same size
    ->Args({112, 112, 64, 56, 56, 2, 1, 0})    // Downsampling
    ->Args({28, 28, 128, 56, 56, 2, 2, 0})     // Upsampling
    
    // ========== Bicubic Interpolation ==========
    // Higher quality but slower
    ->Args({56, 56, 128, 56, 56, 3, 1, 0})     // Same size
    ->Args({112, 112, 64, 56, 56, 3, 2, 0})    // Downsampling
    ->Args({28, 28, 64, 56, 56, 3, 1, 1})      // Upsampling + align corners
    
    // ========== Different Channel Counts ==========
    // Small channels
    ->Args({56, 56, 32, 56, 56, 1, 1, 0})      // Fewer channels
    ->Args({56, 56, 16, 56, 56, 1, 2, 0})      // Very few channels
    
    // Large channels
    ->Args({28, 28, 256, 28, 28, 1, 1, 0})     // Many channels
    ->Args({28, 28, 512, 28, 28, 1, 2, 0})     // Very many channels
    
    // ========== Asymmetric Grids ==========
    // Non-square grids (common in real applications)
    ->Args({56, 56, 64, 112, 28, 1, 1, 0})     // Wide grid
    ->Args({56, 56, 64, 28, 112, 1, 1, 0})     // Tall grid
    ->Args({112, 56, 64, 56, 112, 1, 2, 0})    // Asymmetric input and grid
    
    // ========== Small Inputs (Mobile/Embedded) ==========
    ->Args({14, 14, 64, 14, 14, 1, 1, 0})      // Very small
    ->Args({7, 7, 128, 7, 7, 1, 2, 0})         // Tiny feature maps
    
    // ========== Large Inputs (High Resolution) ==========
    ->Args({224, 224, 32, 224, 224, 1, 1, 0})  // ImageNet size
    ->Args({224, 224, 64, 112, 112, 1, 2, 0})  // Large with downsampling
    
    // ========== Edge Cases ==========
    ->Args({3, 7, 1, 11, 13, 1, 1, 0})         // Minimal channels
    ->Args({8, 12, 16, 24, 16, 2, 3, 1});      // Mixed parameters

BENCHMARK_MAIN();

