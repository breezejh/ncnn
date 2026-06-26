// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_LayerNorm(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int affine_size = state.range(3);
    int affine = state.range(4);
    float eps = 0.001f;

    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, affine_size);
    pd.set(1, eps);
    pd.set(2, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(affine_size);
    weights[1] = RandomMat(affine_size);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("LayerNorm");
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
BENCHMARK(BM_LayerNorm)
    // ========== Vision Transformer (ViT) Configurations ==========
    // ViT-Base: 12 layers, 768 hidden dim, 197 tokens (14x14 patches + 1 cls)
    ->Args({197, 1, 1, 768, 1})       // ViT-Base per-token normalization
    ->Args({768, 1, 1, 768, 1})       // ViT-Base full dimension
    
    // ViT-Large: 24 layers, 1024 hidden dim
    ->Args({197, 1, 1, 1024, 1})      // ViT-Large per-token
    ->Args({1024, 1, 1, 1024, 1})     // ViT-Large full dimension
    
    // ViT-Huge: 32 layers, 1280 hidden dim
    ->Args({197, 1, 1, 1280, 1})      // ViT-Huge per-token
    
    // ========== BERT/Transformer Configurations ==========
    // BERT-Base: 768 hidden dim, 512 sequence length
    ->Args({512, 1, 1, 768, 1})       // BERT-Base sequence
    ->Args({768, 1, 1, 768, 1})       // BERT-Base dimension
    
    // BERT-Large: 1024 hidden dim
    ->Args({512, 1, 1, 1024, 1})      // BERT-Large sequence
    ->Args({1024, 1, 1, 1024, 1})     // BERT-Large dimension
    
    // GPT-2: various sizes
    ->Args({1024, 1, 1, 768, 1})      // GPT-2 small
    ->Args({1024, 1, 1, 1024, 1})     // GPT-2 medium
    ->Args({1024, 1, 1, 1280, 1})     // GPT-2 large
    ->Args({1024, 1, 1, 1600, 1})     // GPT-2 XL
    
    // ========== Swin Transformer Configurations ==========
    // Swin-Tiny: 96 dim
    ->Args({49, 1, 1, 96, 1})         // 7x7 window
    ->Args({196, 1, 1, 96, 1})        // 14x14 patches
    
    // Swin-Base: 128 dim
    ->Args({49, 1, 1, 128, 1})        // 7x7 window
    ->Args({196, 1, 1, 128, 1})       // 14x14 patches
    
    // ========== Spatial LayerNorm (over H dimension) ==========
    // Common in some architectures
    ->Args({64, 14, 14, 64, 1})       // Normalize over spatial H
    ->Args({128, 7, 7, 128, 1})       // Smaller spatial
    
    // ========== Full Spatial LayerNorm (over W*H*C) ==========
    // Normalize over entire spatial dimensions
    ->Args({14, 14, 768, 14*14*768, 1})   // ViT patch features
    ->Args({7, 7, 512, 7*7*512, 1})       // Smaller feature map
    
    // ========== Different Affine Sizes ==========
    // Partial normalization (normalize over subset)
    ->Args({256, 1, 1, 128, 1})       // Normalize over half
    ->Args({512, 1, 1, 256, 1})       // Normalize over half
    ->Args({1024, 1, 1, 512, 1})      // Normalize over half
    
    // ========== With and Without Affine ==========
    ->Args({768, 1, 1, 768, 0})       // Without affine (no learnable params)
    ->Args({768, 1, 1, 768, 1})       // With affine (gamma and beta)
    ->Args({1024, 1, 1, 1024, 0})     // Larger, without affine
    ->Args({1024, 1, 1, 1024, 1})     // Larger, with affine
    
    // ========== Small Models (Mobile/Embedded) ==========
    ->Args({128, 1, 1, 128, 1})       // Small transformer
    ->Args({256, 1, 1, 256, 1})       // Medium transformer
    ->Args({384, 1, 1, 384, 1})       // MobileViT style
    
    // ========== Large Models (Server/Research) ==========
    ->Args({2048, 1, 1, 2048, 1})     // Very large dimension
    ->Args({4096, 1, 1, 4096, 1})     // Extremely large
    
    // ========== Batch Processing (using h dimension) ==========
    ->Args({768, 8, 1, 768, 1})       // Batch size 8
    ->Args({768, 16, 1, 768, 1})      // Batch size 16
    ->Args({768, 32, 1, 768, 1})      // Batch size 32
    
    // ========== Edge Cases ==========
    ->Args({1, 1, 1, 1, 1})           // Minimal configuration
    ->Args({8192, 1, 1, 8192, 1});    // Extremely large dimension

BENCHMARK_MAIN();

