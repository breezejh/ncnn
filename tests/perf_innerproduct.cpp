// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_InnerProduct(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int outch = state.range(3);
    int bias = state.range(4);

    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, bias);
    pd.set(2, outch * w * h * c);

    // Use fixed activation for consistent benchmarking
    int activation_type = 0; // No activation
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.f;
    activation_params[1] = 0.f;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * w * h * c);
    if (bias)
        weights[1] = RandomMat(outch);

    ncnn::Option opt;
    opt.num_threads = 1;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("InnerProduct");
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

// Benchmark with different configurations: w, h, c, outch, bias
BENCHMARK(BM_InnerProduct)
    // ========== Classification Networks - Final FC Layers ==========
    // ImageNet classification (1000 classes)
    ->Args({7, 7, 512, 1000, 1})      // ResNet-18/34 final layer
    ->Args({7, 7, 2048, 1000, 1})     // ResNet-50/101/152 final layer
    ->Args({1, 1, 2048, 1000, 1})     // After global pooling
    ->Args({1, 1, 1280, 1000, 1})     // MobileNetV2 final layer
    
    // Smaller classification tasks
    ->Args({1, 1, 512, 100, 1})       // 100-class classification
    ->Args({1, 1, 256, 10, 1})        // CIFAR-10 style
    
    // ========== Multi-Layer Perceptron (MLP) ==========
    // Common MLP configurations in transformers and other architectures
    ->Args({1, 1, 768, 3072, 1})      // BERT FFN expansion (768 -> 3072)
    ->Args({1, 1, 3072, 768, 1})      // BERT FFN projection (3072 -> 768)
    ->Args({1, 1, 1024, 4096, 1})     // GPT-2 FFN expansion
    ->Args({1, 1, 4096, 1024, 1})     // GPT-2 FFN projection
    
    // ========== Feature Extraction to FC ==========
    // Typical CNN to FC transitions
    ->Args({7, 7, 512, 4096, 1})      // VGG-style FC6
    ->Args({1, 1, 512, 4096, 1})      // After pooling
    ->Args({1, 1, 4096, 4096, 1})     // VGG-style FC7
    ->Args({1, 1, 4096, 1000, 1})     // VGG-style FC8
    
    // ========== Object Detection - RoI Features ==========
    // Faster R-CNN style FC layers
    ->Args({7, 7, 256, 1024, 1})      // RoI feature to FC
    ->Args({1, 1, 1024, 1024, 1})     // Detection head FC
    ->Args({1, 1, 1024, 81, 1})       // COCO 80 classes + background
    
    // ========== Embedding Layers ==========
    // Word embeddings and similar
    ->Args({1, 1, 512, 256, 1})       // Dimension reduction
    ->Args({1, 1, 256, 512, 1})       // Dimension expansion
    ->Args({1, 1, 128, 128, 1})       // Same dimension transform
    
    // ========== Small Networks (Mobile/Embedded) ==========
    ->Args({1, 1, 128, 64, 1})        // Small FC layer
    ->Args({1, 1, 64, 32, 1})         // Very small FC layer
    ->Args({1, 1, 32, 10, 1})         // Tiny classification
    
    // ========== Large Networks (High Capacity) ==========
    ->Args({1, 1, 2048, 2048, 1})     // Large same-dimension
    ->Args({1, 1, 4096, 2048, 1})     // Large reduction
    
    // ========== With and Without Bias ==========
    ->Args({1, 1, 512, 512, 0})       // Without bias
    ->Args({1, 1, 512, 512, 1})       // With bias
    ->Args({7, 7, 256, 1024, 0})      // Spatial input, no bias
    
    // ========== Batch Processing (h > 1) ==========
    // Simulating batch dimension in h
    ->Args({512, 8, 1, 512, 1})       // Batch size 8
    ->Args({768, 16, 1, 768, 1})      // Batch size 16
    ->Args({1024, 32, 1, 1024, 1})    // Batch size 32
    
    // ========== Edge Cases ==========
    ->Args({1, 1, 1, 1, 1})           // Minimal configuration
    ->Args({1, 1, 8192, 1000, 1});    // Very large input dimension

BENCHMARK_MAIN();

