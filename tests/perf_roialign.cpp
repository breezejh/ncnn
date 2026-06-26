// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer.h"
#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_ROIAlign(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int pooled_width = state.range(3);
    int pooled_height = state.range(4);
    int sampling_ratio = state.range(5);
    int aligned = state.range(6);
    int version = state.range(7);

    // Prepare input feature map
    ncnn::Mat feature_map = RandomMat(w, h, c);
    
    // Prepare ROI (x1, y1, x2, y2)
    ncnn::Mat roi(4);
    roi[0] = w * 0.2f;  // x1
    roi[1] = h * 0.2f;  // y1
    roi[2] = w * 0.8f;  // x2
    roi[3] = h * 0.8f;  // y2

    std::vector<ncnn::Mat> inputs(2);
    inputs[0] = feature_map;
    inputs[1] = roi;

    // Calculate spatial_scale based on typical detection pipeline
    float spatial_scale = 1.0f / 16.0f;  // Typical for C4 feature

    ncnn::ParamDict pd;
    pd.set(0, pooled_width);
    pd.set(1, pooled_height);
    pd.set(2, spatial_scale);
    pd.set(3, sampling_ratio);
    pd.set(4, aligned);
    pd.set(5, version);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("ROIAlign");
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

    std::vector<ncnn::Mat> outputs(1);
    for (auto _ : state)
    {
        layer->forward(inputs, outputs, opt);
    }

    // Clean up pipeline resources
    layer->destroy_pipeline(opt);
    delete layer;
}

// Benchmark with different configurations: w, h, c, pooled_w, pooled_h, sampling_ratio, aligned, version
BENCHMARK(BM_ROIAlign)
    // ========== Mask R-CNN Configurations ==========
    // ResNet-50-FPN C4 features (stride=16)
    ->Args({56, 56, 256, 14, 14, 2, 1, 1})    // Standard Mask R-CNN
    ->Args({56, 56, 256, 7, 7, 2, 1, 1})      // Box head pooling
    ->Args({28, 28, 512, 14, 14, 2, 1, 1})    // C5 features
    
    // ResNet-101-FPN
    ->Args({56, 56, 256, 14, 14, 2, 1, 1})    // Mask head
    ->Args({56, 56, 256, 7, 7, 2, 1, 1})      // Box head
    
    // ========== Faster R-CNN Configurations ==========
    ->Args({38, 50, 512, 7, 7, 2, 0, 0})      // VGG-16 backbone
    ->Args({56, 56, 1024, 7, 7, 2, 0, 0})     // ResNet-50 C4
    ->Args({28, 28, 2048, 7, 7, 2, 0, 0})     // ResNet-50 C5
    
    // ========== Different Sampling Ratios ==========
    // sampling_ratio=0 (adaptive)
    ->Args({56, 56, 256, 14, 14, 0, 1, 1})    // Adaptive sampling
    
    // sampling_ratio=1 (single sample per bin)
    ->Args({56, 56, 256, 14, 14, 1, 1, 1})    // Fast but less accurate
    
    // sampling_ratio=4 (high quality)
    ->Args({56, 56, 256, 14, 14, 4, 1, 1})    // High quality sampling
    
    // ========== Different Pooled Sizes ==========
    ->Args({56, 56, 256, 7, 7, 2, 1, 1})      // 7x7 output
    ->Args({56, 56, 256, 14, 14, 2, 1, 1})    // 14x14 output
    ->Args({56, 56, 256, 28, 28, 2, 1, 1})    // 28x28 output (large)
    
    // ========== Different Channel Counts ==========
    ->Args({56, 56, 64, 14, 14, 2, 1, 1})     // Fewer channels
    ->Args({56, 56, 128, 14, 14, 2, 1, 1})    // Medium channels
    ->Args({56, 56, 512, 14, 14, 2, 1, 1})    // More channels
    ->Args({56, 56, 1024, 7, 7, 2, 1, 1})     // Many channels
    
    // ========== Different Feature Map Sizes ==========
    ->Args({112, 112, 256, 14, 14, 2, 1, 1})  // Larger feature map
    ->Args({28, 28, 256, 14, 14, 2, 1, 1})    // Smaller feature map
    ->Args({14, 14, 512, 7, 7, 2, 1, 1})      // Small feature map
    
    // ========== Aligned vs Non-aligned ==========
    ->Args({56, 56, 256, 14, 14, 2, 0, 1})    // Non-aligned
    ->Args({56, 56, 256, 14, 14, 2, 1, 1})    // Aligned (better)
    
    // ========== Different Versions ==========
    ->Args({56, 56, 256, 14, 14, 2, 1, 0})    // Version 0
    ->Args({56, 56, 256, 14, 14, 2, 1, 1})    // Version 1 (PyTorch style)
    
    // ========== Edge Cases ==========
    ->Args({7, 7, 256, 3, 3, 2, 1, 1})        // Very small feature map
    ->Args({112, 112, 64, 28, 28, 2, 1, 1});  // Large output

BENCHMARK_MAIN();

