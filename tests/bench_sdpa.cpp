// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_SDPA(benchmark::State& state)
{
    int embed_dim = state.range(0);
    int num_heads = state.range(1);
    int src_seqlen = state.range(2);
    int dst_seqlen = state.range(3);
    int attn_mask = state.range(4);

    int head_dim = embed_dim / num_heads;

    // Q: (head_dim, src_seqlen, num_heads)
    ncnn::Mat q = RandomMat(head_dim, src_seqlen, num_heads);
    // K: (head_dim, dst_seqlen, num_heads)
    ncnn::Mat k = RandomMat(head_dim, dst_seqlen, num_heads);
    // V: (head_dim, dst_seqlen, num_heads)
    ncnn::Mat v = RandomMat(head_dim, dst_seqlen, num_heads);

    float scale = 1.0f / sqrtf((float)head_dim);

    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(6, scale);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> inputs(3);
    inputs[0] = q;
    inputs[1] = k;
    inputs[2] = v;

    if (attn_mask)
    {
        inputs.push_back(RandomMat(dst_seqlen, src_seqlen));
    }

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("SDPA");
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

// Benchmark with different configurations: embed_dim, num_heads, src_seqlen, dst_seqlen, attn_mask
BENCHMARK(BM_SDPA)
    // ========== BERT Configurations ==========
    // BERT-Base: hidden=768, heads=12, head_dim=64
    ->Args({768, 12, 512, 512, 0})            // BERT-Base, seq=512
    ->Args({768, 12, 128, 128, 0})            // BERT-Base, seq=128
    ->Args({768, 12, 512, 512, 1})            // With attention mask
    
    // BERT-Large: hidden=1024, heads=16, head_dim=64
    ->Args({1024, 16, 512, 512, 0})           // BERT-Large, seq=512
    ->Args({1024, 16, 128, 128, 0})           // BERT-Large, seq=128
    
    // ========== GPT Configurations ==========
    // GPT-2 Small: hidden=768, heads=12
    ->Args({768, 12, 1024, 1024, 0})          // GPT-2, seq=1024
    // ->Args({768, 12, 2048, 2048, 0})          // GPT-2, seq=2048
    
    // GPT-2 Medium: hidden=1024, heads=16
    // ->Args({1024, 16, 1024, 1024, 0})         // GPT-2 Medium
    
    // GPT-2 Large: hidden=1280, heads=20
    // ->Args({1280, 20, 1024, 1024, 0})         // GPT-2 Large
    
    // ========== LLaMA Configurations ==========
    // LLaMA-7B: hidden=4096, heads=32, head_dim=128
    // ->Args({4096, 32, 512, 512, 0})           // LLaMA-7B, seq=512
    // ->Args({4096, 32, 1024, 1024, 0})         // LLaMA-7B, seq=1024
    // ->Args({4096, 32, 2048, 2048, 0})         // LLaMA-7B, seq=2048
    
    // LLaMA-13B: hidden=5120, heads=40, head_dim=128
    // ->Args({5120, 40, 512, 512, 0})           // LLaMA-13B, seq=512
    // ->Args({5120, 40, 2048, 2048, 0})         // LLaMA-13B, seq=2048
    
    // ========== Vision Transformer Configurations ==========
    // ViT-Base: hidden=768, heads=12, patches=196 (14x14)
    ->Args({768, 12, 197, 197, 0})            // ViT-Base (with CLS token)
    
    // ViT-Large: hidden=1024, heads=16
    ->Args({1024, 16, 197, 197, 0})           // ViT-Large
    
    // ViT-Huge: hidden=1280, heads=16
    ->Args({1280, 16, 197, 197, 0})           // ViT-Huge
    
    // ========== Cross-Attention (Encoder-Decoder) ==========
    // Different source and destination sequence lengths
    ->Args({768, 12, 128, 512, 0})            // Decoder attending to encoder
    ->Args({768, 12, 256, 1024, 0})           // Longer encoder sequence
    
    // ========== Small Models (Mobile/Edge) ==========
    ->Args({256, 4, 128, 128, 0})             // Small model
    ->Args({512, 8, 256, 256, 0})             // Medium model
    
    // ========== Long Context Models ==========
    // Extended context lengths
    // ->Args({768, 12, 4096, 4096, 0})          // Very long context
    // ->Args({1024, 16, 8192, 8192, 0})         // Extremely long context
    
    // ========== With Attention Mask ==========
    ->Args({768, 12, 512, 512, 1})            // BERT with mask
    // ->Args({4096, 32, 1024, 1024, 1})         // LLaMA with mask
    
    // ========== Edge Cases ==========
    ->Args({64, 1, 32, 32, 0})                // Very small
    ->Args({2048, 16, 128, 128, 0});          // Large embed_dim

BENCHMARK_MAIN();

