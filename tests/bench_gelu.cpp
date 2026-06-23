// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_GELU(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int d = state.range(2);
    int c = state.range(3);
    int dims = state.range(4);
    int fast_gelu = state.range(5);

    ncnn::Mat a;
    if (dims == 1)
        a = RandomMat(w);
    else if (dims == 2)
        a = RandomMat(w, h, c);
    else if (dims == 3)
        a = RandomMat(w, h, c);
    else // dims == 4
        a = RandomMat(w, h, d, c);

    ncnn::ParamDict pd;
    pd.set(0, fast_gelu);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* layer = ncnn::create_layer("GELU");
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

    layer->destroy_pipeline(opt);
    delete layer;
}

// Benchmark configurations: w, h, d, c, dims, fast_gelu
BENCHMARK(BM_GELU)
    // ========== 1D (Vector) - Standard GELU ==========
    ->Args({768, 0, 0, 0, 1, 0})          // BERT hidden size
    ->Args({1024, 0, 0, 0, 1, 0})         // GPT-2 small hidden
    ->Args({2048, 0, 0, 0, 1, 0})         // GPT-2 medium hidden
    ->Args({4096, 0, 0, 0, 1, 0})         // LLaMA-7B hidden
    ->Args({8192, 0, 0, 0, 1, 0})         // Large hidden size
    
    // ========== 1D (Vector) - Fast GELU ==========
    ->Args({768, 0, 0, 0, 1, 1})          // BERT hidden size (fast)
    ->Args({1024, 0, 0, 0, 1, 1})         // GPT-2 small (fast)
    ->Args({4096, 0, 0, 0, 1, 1})         // LLaMA-7B (fast)
    
    // ========== 2D (Sequence x Hidden) - Standard GELU ==========
    // BERT-Base: hidden=768
    ->Args({768, 128, 0, 1, 2, 0})        // BERT seq=128
    ->Args({768, 512, 0, 1, 2, 0})        // BERT seq=512
    
    // GPT-2 Small: hidden=768
    ->Args({768, 1024, 0, 1, 2, 0})       // GPT-2 seq=1024
    ->Args({768, 2048, 0, 1, 2, 0})       // GPT-2 seq=2048
    
    // GPT-2 Medium: hidden=1024
    ->Args({1024, 1024, 0, 1, 2, 0})      // GPT-2 Medium
    
    // GPT-2 Large: hidden=1280
    ->Args({1280, 1024, 0, 1, 2, 0})      // GPT-2 Large
    
    // LLaMA-7B: hidden=4096
    ->Args({4096, 512, 0, 1, 2, 0})       // LLaMA seq=512
    ->Args({4096, 1024, 0, 1, 2, 0})      // LLaMA seq=1024
    ->Args({4096, 2048, 0, 1, 2, 0})      // LLaMA seq=2048
    
    // ========== 2D (Sequence x Hidden) - Fast GELU ==========
    ->Args({768, 512, 0, 1, 2, 1})        // BERT (fast)
    ->Args({1024, 1024, 0, 1, 2, 1})      // GPT-2 Medium (fast)
    ->Args({4096, 1024, 0, 1, 2, 1})      // LLaMA (fast)
    
    // ========== 2D (Batch x Hidden) - Standard GELU ==========
    // Batched inference
    ->Args({768, 8, 0, 1, 2, 0})          // Batch=8, BERT
    ->Args({768, 16, 0, 1, 2, 0})         // Batch=16, BERT
    ->Args({768, 32, 0, 1, 2, 0})         // Batch=32, BERT
    ->Args({4096, 8, 0, 1, 2, 0})         // Batch=8, LLaMA
    
    // ========== 2D (Spatial x Channels) - Standard GELU ==========
    // Vision models using GELU
    ->Args({56, 56, 0, 256, 2, 0})        // Feature map
    ->Args({28, 28, 0, 512, 2, 0})        // Deeper feature map
    ->Args({14, 14, 0, 1024, 2, 0})       // Very deep feature map
    
    // ========== 3D (Patches x Hidden x Batch) - Standard GELU ==========
    // Vision Transformer
    ->Args({768, 197, 0, 1, 2, 0})        // ViT-Base (14x14 patches + CLS)
    ->Args({1024, 197, 0, 1, 2, 0})       // ViT-Large
    ->Args({1280, 197, 0, 1, 2, 0})       // ViT-Huge
    
    // ========== 3D - Fast GELU ==========
    ->Args({768, 197, 0, 1, 2, 1})        // ViT-Base (fast)
    
    // ========== 4D (Video/Volumetric) - Standard GELU ==========
    // Video features with GELU
    ->Args({7, 7, 8, 512, 4, 0})          // Video feature
    ->Args({14, 14, 16, 256, 4, 0})       // Larger video feature
    
    // ========== MLP Intermediate Layers - Standard GELU ==========
    // BERT FFN: hidden -> 4*hidden -> hidden
    ->Args({3072, 128, 0, 1, 2, 0})       // BERT FFN intermediate (768*4)
    ->Args({3072, 512, 0, 1, 2, 0})       // BERT FFN intermediate
    
    // GPT-2 FFN
    ->Args({3072, 1024, 0, 1, 2, 0})      // GPT-2 Small FFN (768*4)
    ->Args({4096, 1024, 0, 1, 2, 0})      // GPT-2 Medium FFN (1024*4)
    
    // LLaMA FFN (uses SwiGLU but for comparison)
    ->Args({11008, 512, 0, 1, 2, 0})      // LLaMA-7B FFN intermediate
    ->Args({11008, 1024, 0, 1, 2, 0})     // LLaMA-7B FFN intermediate
    
    // ========== MLP Intermediate - Fast GELU ==========
    ->Args({3072, 512, 0, 1, 2, 1})       // BERT FFN (fast)
    ->Args({11008, 1024, 0, 1, 2, 1})     // LLaMA FFN (fast)
    
    // ========== Edge Cases ==========
    // Very small
    ->Args({64, 0, 0, 0, 1, 0})           // Small vector
    ->Args({32, 32, 0, 1, 2, 0})          // Small 2D
    
    // Very large
    ->Args({16384, 0, 0, 0, 1, 0})        // Large vector
    ->Args({8192, 2048, 0, 1, 2, 0})      // Very large 2D
    
    // Compare standard vs fast
    ->Args({4096, 1024, 0, 1, 2, 0})      // Standard
    ->Args({4096, 1024, 0, 1, 2, 1});     // Fast

BENCHMARK_MAIN();

