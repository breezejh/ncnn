// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_Bias(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int dims = state.range(3);  // 2D or 3D

    ncnn::Mat a;
    if (dims == 2)
    {
        a = RandomMat(w, h, c);
    }
    else // dims == 3
    {
        a = RandomMat(w, h, state.range(4), c);  // w, h, d, c
    }

    ncnn::ParamDict pd;
    pd.set(0, c);  // bias_data_size

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(c);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("Bias");
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

// Benchmark with different configurations: w, h, c, dims, [d]
BENCHMARK(BM_Bias)
    // ========== 2D Tensors (Image-like) ==========
    // Small feature maps
    ->Args({7, 7, 512, 2, 0})      // Small spatial, many channels
    ->Args({14, 14, 256, 2, 0})    // ResNet stage 4
    ->Args({28, 28, 128, 2, 0})    // ResNet stage 3
    ->Args({56, 56, 64, 2, 0})     // ResNet stage 2
    
    // Medium feature maps
    ->Args({112, 112, 64, 2, 0})   // Early layers
    ->Args({224, 224, 3, 2, 0})    // RGB input
    ->Args({224, 224, 64, 2, 0})   // First conv output
    
    // Large feature maps
    ->Args({512, 512, 32, 2, 0})   // High-resolution processing
    ->Args({1024, 1024, 16, 2, 0}) // Very high-resolution
    
    // Different channel counts
    ->Args({56, 56, 16, 2, 0})     // Few channels
    ->Args({56, 56, 32, 2, 0})     // MobileNet-style
    ->Args({56, 56, 128, 2, 0})    // Medium channels
    ->Args({56, 56, 256, 2, 0})    // Many channels
    ->Args({28, 28, 512, 2, 0})    // Very many channels
    ->Args({14, 14, 1024, 2, 0})   // Extremely many channels
    
    // Transformer/NLP scenarios (treating as 2D: seq_len, hidden_dim, batch)
    ->Args({512, 768, 1, 2, 0})    // BERT-Base sequence
    ->Args({1024, 768, 1, 2, 0})   // Long sequence
    ->Args({2048, 1024, 1, 2, 0})  // GPT-2 style
    ->Args({512, 4096, 1, 2, 0})   // LLaMA-7B style
    
    // Vision Transformer patches
    ->Args({197, 768, 1, 2, 0})    // ViT-Base (14x14 patches + CLS)
    ->Args({197, 1024, 1, 2, 0})   // ViT-Large
    
    // ========== 3D Tensors (Video/Volumetric) ==========
    // Small 3D volumes
    ->Args({7, 7, 7, 64, 3})       // Small 3D feature
    ->Args({14, 14, 14, 32, 3})    // Medium 3D feature
    ->Args({28, 28, 28, 16, 3})    // Larger 3D feature
    
    // Video-like (temporal dimension)
    ->Args({56, 56, 8, 64, 3})     // 8 frames, 56x56 spatial
    ->Args({112, 112, 16, 32, 3})  // 16 frames, 112x112 spatial
    ->Args({224, 224, 4, 16, 3})   // 4 frames, 224x224 spatial
    
    // Medical imaging (CT/MRI scans)
    ->Args({128, 128, 64, 32, 3})  // Typical medical scan
    ->Args({256, 256, 128, 16, 3}) // High-resolution scan
    
    // Different channel counts in 3D
    ->Args({32, 32, 32, 8, 3})     // Few channels
    ->Args({32, 32, 32, 16, 3})    // Medium channels
    ->Args({32, 32, 32, 32, 3})    // Many channels
    ->Args({16, 16, 16, 64, 3})    // Very many channels
    
    // ========== Edge Cases ==========
    // Single channel
    ->Args({224, 224, 1, 2, 0})    // Single channel image
    
    // Very small spatial
    ->Args({1, 1, 1024, 2, 0})     // Global pooling output
    ->Args({4, 4, 512, 2, 0})      // Very small feature map
    
    // Asymmetric shapes
    ->Args({128, 64, 32, 2, 0})    // Wide
    ->Args({64, 128, 32, 2, 0})    // Tall
    
    // 3D edge cases
    ->Args({1, 1, 1, 512, 3})      // 3D global pooling output
    ->Args({64, 64, 1, 32, 3});    // Single slice

BENCHMARK_MAIN();

