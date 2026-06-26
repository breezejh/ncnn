// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_MatMul(benchmark::State& state)
{
    // Parse dimensions from state ranges
    // For simplicity: a_w, a_h, b_w (where b_h = a_w for valid matmul)
    int a_w = state.range(0);
    int a_h = state.range(1);
    int b_w = state.range(2);
    int transB = state.range(3);

    ncnn::Mat a = RandomMat(a_w, a_h);
    ncnn::Mat b;
    
    if (transB)
    {
        // If transB=1, b is stored as (b_w, a_w) and will be transposed
        b = RandomMat(b_w, a_w);
    }
    else
    {
        // If transB=0, b is stored as (a_w, b_w)
        b = RandomMat(a_w, b_w);
    }

    ncnn::ParamDict pd;
    pd.set(0, transB);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("MatMul");
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

    std::vector<ncnn::Mat> inputs(2);
    inputs[0] = a;
    inputs[1] = b;
    
    std::vector<ncnn::Mat> outputs(1);
    
    for (auto _ : state)
    {
        layer->forward(inputs, outputs, opt);
    }

    // Clean up pipeline resources
    layer->destroy_pipeline(opt);
    delete layer;
}

// Benchmark with different configurations: a_w, a_h, b_w, transB
BENCHMARK(BM_MatMul)
    // ========== Transformer Attention (Q*K^T and Attn*V) ==========
    // BERT-Base: seq_len=512, hidden=768, num_heads=12, head_dim=64
    ->Args({64, 512, 512, 1})         // Q @ K^T per head
    // ->Args({512, 64, 64, 1})          // Attn @ V per head (transposed)

    // ViT-Base: seq_len=197, hidden=768, num_heads=12, head_dim=64
    ->Args({64, 197, 197, 1})         // Q @ K^T per head
    ->Args({197, 64, 64, 1})          // Attn @ V per head (transposed)

    // GPT-2: seq_len=1024, hidden=768, num_heads=12, head_dim=64
    ->Args({64, 1024, 1024, 1})       // Q @ K^T per head
    // ->Args({1024, 64, 64, 1})         // Attn @ V per head (transposed)

    // ========== Linear Layers (Fully Connected) ==========
    // Use batch dimension to avoid single element
    ->Args({512, 8, 512, 1})          // 512 -> 512 (batch=8)
    ->Args({768, 8, 3072, 1})         // BERT FFN expansion (batch=8)
    // ->Args({3072, 8, 768, 1})         // BERT FFN projection (batch=8)
    // ->Args({1024, 16, 4096, 1})       // GPT-2 FFN expansion (batch=16)
    // ->Args({4096, 16, 1024, 1})       // GPT-2 FFN projection (batch=16)

    // ========== Batch Matrix Multiplication ==========
    ->Args({256, 8, 256, 1})          // Batch=8
    ->Args({512, 16, 512, 1})         // Batch=16
    ->Args({768, 32, 768, 1})         // Batch=32

    // ========== Square Matrices ==========
    ->Args({64, 64, 64, 1})           // 64x64
    ->Args({128, 128, 128, 1})        // 128x128
    ->Args({256, 256, 256, 1})        // 256x256
    ->Args({512, 512, 512, 1})        // 512x512
    ->Args({1024, 1024, 1024, 1})     // 1024x1024
    ->Args({2048, 2048, 2048, 1})     // 2048x2048

    // ========== Rectangular Matrices ==========
    ->Args({64, 1024, 256, 1})        // Tall and thin
    ->Args({128, 2048, 512, 1})       // Dimension reduction
    ->Args({1024, 64, 256, 1})        // Short and wide
    // ->Args({2048, 128, 512, 1})       // Wide projection

    // ========== Vision Applications ==========
    ->Args({768, 196, 768, 1})        // ViT patch features
    ->Args({1024, 256, 1024, 1})      // Larger ViT

    // ========== NLP Applications ==========
    ->Args({512, 100, 512, 1})        // Sequence processing
    ->Args({768, 128, 768, 1})        // BERT-like

    // ========== Small Models (Mobile/Embedded) ==========
    ->Args({32, 32, 32, 1})           // Tiny model
    ->Args({64, 64, 128, 1})          // Small expansion
    ->Args({128, 128, 64, 1})         // Small reduction

    // ========== Edge Cases ==========
    ->Args({8, 8, 8, 1})              // Very small
    ->Args({4096, 256, 4096, 1});     // Very large projection

BENCHMARK_MAIN();

