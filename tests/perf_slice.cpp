// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static std::vector<int> IntArray(int a0, int a1, int a2)
{
    std::vector<int> m(3);
    m[0] = a0;
    m[1] = a1;
    m[2] = a2;
    return m;
}

static void BM_Slice(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int d = state.range(2);
    int c = state.range(3);
    int dims = state.range(4);
    int axis = state.range(5);
    int num_slices = state.range(6);

    ncnn::Mat a;
    if (dims == 1)
        a = RandomMat(w);
    else if (dims == 2)
        a = RandomMat(w, h, c);
    else if (dims == 3)
        a = RandomMat(w, h, c);
    else // dims == 4
        a = RandomMat(w, h, d, c);

    // Create equal slices
    std::vector<int> slices_array(num_slices, -233);
    ncnn::Mat slices(slices_array.size());
    {
        int* p = slices;
        for (size_t i = 0; i < slices_array.size(); i++)
        {
            p[i] = slices_array[i];
        }
    }

    ncnn::ParamDict pd;
    pd.set(0, slices);
    pd.set(1, axis);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(1);
    a0[0] = a;

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* layer = ncnn::create_layer("Slice");
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

    std::vector<ncnn::Mat> outputs(num_slices);
    for (auto _ : state)
    {
        layer->forward(a0, outputs, opt);
    }

    layer->destroy_pipeline(opt);
    delete layer;
}

// Benchmark configurations: w, h, d, c, dims, axis, num_slices
BENCHMARK(BM_Slice)
    // ========== 1D Slicing (Vector) ==========
    ->Args({1024, 0, 0, 0, 1, 0, 2})      // Split vector in half
    ->Args({1024, 0, 0, 0, 1, 0, 4})      // Split into 4 parts
    ->Args({2048, 0, 0, 0, 1, 0, 2})      // Larger vector
    ->Args({4096, 0, 0, 0, 1, 0, 2})      // Even larger
    
    // ========== 2D Slicing along Width (axis=0) ==========
    // Image-like slicing
    ->Args({224, 224, 0, 3, 2, 0, 2})     // Split RGB image horizontally
    ->Args({224, 224, 0, 64, 2, 0, 2})    // Split feature map
    ->Args({112, 112, 0, 128, 2, 0, 2})   // Medium feature map
    ->Args({56, 56, 0, 256, 2, 0, 2})     // Deep feature map
    ->Args({56, 56, 0, 256, 2, 0, 4})     // Split into 4 parts
    
    // ========== 2D Slicing along Height (axis=1) ==========
    ->Args({224, 224, 0, 3, 2, 1, 2})     // Split RGB vertically
    ->Args({224, 224, 0, 64, 2, 1, 2})    // Split feature map
    ->Args({56, 56, 0, 256, 2, 1, 4})     // Split into 4 parts
    
    // ========== 2D Slicing along Channels (axis=2) ==========
    // Common in network architectures (e.g., Inception, ResNeXt)
    ->Args({56, 56, 0, 256, 2, 2, 2})     // Split channels in half
    ->Args({56, 56, 0, 512, 2, 2, 2})     // Larger channel count
    ->Args({28, 28, 0, 512, 2, 2, 4})     // Split into 4 groups
    ->Args({28, 28, 0, 1024, 2, 2, 2})    // Very large channels
    
    // ShuffleNet-style channel split
    ->Args({56, 56, 0, 128, 2, 2, 2})     // ShuffleNet split
    ->Args({28, 28, 0, 256, 2, 2, 2})     // ShuffleNet deeper
    
    // ========== Transformer/NLP Slicing ==========
    // Multi-head attention: split hidden dimension
    ->Args({768, 512, 0, 1, 2, 0, 12})    // BERT-Base: 768 -> 12 heads
    ->Args({1024, 1024, 0, 1, 2, 0, 16})  // GPT-2 Medium: 1024 -> 16 heads
    ->Args({4096, 1024, 0, 1, 2, 0, 32})  // LLaMA-7B: 4096 -> 32 heads
    
    // Sequence slicing
    ->Args({768, 512, 0, 1, 2, 1, 2})     // Split sequence
    ->Args({768, 1024, 0, 1, 2, 1, 4})    // Split long sequence
    
    // ========== 3D Slicing (Video/Volumetric) ==========
    // Spatial slicing
    ->Args({56, 56, 0, 256, 3, 0, 2})     // Split width
    ->Args({56, 56, 0, 256, 3, 1, 2})     // Split height
    
    // Channel slicing in 3D
    ->Args({28, 28, 0, 512, 3, 2, 2})     // Split channels
    ->Args({14, 14, 0, 1024, 3, 2, 4})    // Split into 4 groups
    
    // ========== 4D Slicing (Video with temporal) ==========
    // Temporal slicing
    ->Args({56, 56, 16, 64, 4, 2, 2})     // Split temporal dimension
    ->Args({28, 28, 32, 128, 4, 2, 4})    // Split into 4 temporal parts
    
    // Spatial slicing in 4D
    ->Args({112, 112, 8, 64, 4, 0, 2})    // Split width
    ->Args({112, 112, 8, 64, 4, 1, 2})    // Split height
    
    // Channel slicing in 4D
    ->Args({56, 56, 8, 256, 4, 3, 2})     // Split channels
    ->Args({28, 28, 16, 512, 4, 3, 4})    // Split into 4 groups
    
    // ========== Different Split Counts ==========
    ->Args({224, 224, 0, 64, 2, 2, 2})    // 2-way split
    ->Args({224, 224, 0, 64, 2, 2, 4})    // 4-way split
    ->Args({224, 224, 0, 128, 2, 2, 8})   // 8-way split
    
    // ========== Large Tensors ==========
    ->Args({512, 512, 0, 64, 2, 0, 2})    // Large spatial
    ->Args({512, 512, 0, 64, 2, 2, 2})    // Large spatial, channel split
    ->Args({1024, 1024, 0, 32, 2, 0, 2})  // Very large spatial
    
    // ========== Small Tensors ==========
    ->Args({7, 7, 0, 512, 2, 0, 2})       // Small spatial
    ->Args({7, 7, 0, 512, 2, 2, 2})       // Small spatial, channel split
    
    // ========== Edge Cases ==========
    ->Args({1024, 0, 0, 0, 1, 0, 8})      // Many splits on 1D
    ->Args({56, 56, 0, 256, 2, 2, 16});   // Many channel splits

BENCHMARK_MAIN();

