// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_Flatten(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int d = state.range(2);
    int c = state.range(3);
    int dims = state.range(4);

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

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* layer = ncnn::create_layer("Flatten");
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

// Benchmark configurations: w, h, d, c, dims
BENCHMARK(BM_Flatten)
    // ========== 1D (Already flat - identity operation) ==========
    // ->Args({1024, 0, 0, 0, 1})        // Small vector
    // ->Args({4096, 0, 0, 0, 1})        // Medium vector
    // ->Args({16384, 0, 0, 0, 1})       // Large vector
    
    // ========== 2D to 1D (Common in CNNs before FC layers) ==========
    // VGG-style: After last pooling before FC
    ->Args({7, 7, 0, 512, 2})         // VGG16/19: 7x7x512 = 25088
    ->Args({7, 7, 0, 4096, 2})        // Very deep: 7x7x4096
    
    // ResNet-style: After global pooling
    ->Args({1, 1, 0, 2048, 2})        // ResNet-50/101/152
    ->Args({1, 1, 0, 512, 2})         // ResNet-18/34
    
    // Different spatial sizes
    ->Args({14, 14, 0, 256, 2})       // 14x14x256 = 50176
    ->Args({28, 28, 0, 128, 2})       // 28x28x128 = 100352
    ->Args({56, 56, 0, 64, 2})        // 56x56x64 = 200704
    
    // MobileNet-style
    ->Args({7, 7, 0, 1024, 2})        // MobileNetV1
    ->Args({7, 7, 0, 1280, 2})        // MobileNetV2
    
    // EfficientNet-style
    ->Args({7, 7, 0, 1536, 2})        // EfficientNet-B0
    ->Args({7, 7, 0, 2048, 2})        // EfficientNet-B4
    
    // Vision Transformer patches
    ->Args({14, 14, 0, 768, 2})       // ViT-Base patches
    ->Args({16, 16, 0, 1024, 2})      // ViT-Large patches
    
    // Small feature maps (late stage)
    ->Args({4, 4, 0, 512, 2})         // 4x4x512
    ->Args({3, 3, 0, 1024, 2})        // 3x3x1024
    
    // Large feature maps (early stage)
    ->Args({112, 112, 0, 64, 2})      // 112x112x64
    ->Args({224, 224, 0, 32, 2})      // 224x224x32
    
    // Non-square feature maps
    ->Args({7, 14, 0, 512, 2})        // Asymmetric
    ->Args({14, 7, 0, 256, 2})        // Asymmetric
    
    // ========== 3D to 1D (Video/Volumetric) ==========
    // Video features
    ->Args({7, 7, 8, 512, 4})         // 7x7x8x512 (spatial + temporal)
    ->Args({14, 14, 16, 256, 4})      // 14x14x16x256
    ->Args({28, 28, 4, 128, 4})       // 28x28x4x128
    
    // 3D CNN outputs
    ->Args({4, 4, 4, 512, 4})         // Small 3D volume
    ->Args({8, 8, 8, 256, 4})         // Medium 3D volume
    ->Args({16, 16, 16, 128, 4})      // Large 3D volume
    
    // Medical imaging
    ->Args({32, 32, 32, 64, 4})       // CT/MRI volume
    ->Args({64, 64, 64, 32, 4})       // Large medical volume
    
    // Point cloud / voxel grids
    ->Args({32, 32, 32, 16, 4})       // Voxel grid
    
    // ========== Edge Cases ==========
    // Single element per channel
    ->Args({1, 1, 0, 1024, 2})        // After global pooling
    ->Args({1, 1, 0, 2048, 2})        // Large channel count
    
    // Very small
    ->Args({2, 2, 0, 64, 2})          // Tiny feature map
    
    // Very large total elements
    ->Args({224, 224, 0, 64, 2})      // 224x224x64 = 3211264
    ->Args({512, 512, 0, 16, 2})      // 512x512x16 = 4194304
    
    // 3D edge cases
    ->Args({1, 1, 1, 512, 4})         // 3D global pooling output
    ->Args({64, 64, 1, 32, 4});       // Single slice

BENCHMARK_MAIN();

