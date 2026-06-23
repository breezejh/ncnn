// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_Convolution(benchmark::State& state)
{
    int c = state.range(0);
    int outch = state.range(1);
    int w = state.range(2);
    int h = state.range(3);
    int kernel = state.range(4);
    int stride = state.range(5);
    int dilation = 1;
    int pad = kernel / 2;
    int bias = 1;

    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch * c * kernel * kernel);

    // Use fixed activation for consistent benchmarking
    int activation_type = 0; // No activation
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.f;
    activation_params[1] = 0.f;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_sgemm_convolution = false;
    opt.use_winograd_convolution = false;

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer_naive("Convolution");
    layer->load_param(pd);
    layer->load_model(ncnn::ModelBinFromMatArray(weights.data()));

    // Create pipeline to initialize optimized kernels
    layer->create_pipeline(opt);

    ncnn::Mat b;
    for (auto _ : state)
    {
        layer->forward(a, b, opt);
    }

    // Clean up pipeline resources
    layer->destroy_pipeline(opt);
    delete layer;
}

// Benchmark with different configurations: c, outch, w, h, kernel, stride
BENCHMARK(BM_Convolution)
    // ========== 1x1 Convolutions (Pointwise) ==========
    // Common in MobileNet, ResNet bottlenecks
    ->Args({64, 64, 56, 56, 1, 1})      // Small channels
    ->Args({128, 128, 28, 28, 1, 1})    // Medium channels
    ->Args({256, 256, 14, 14, 1, 1})    // Large channels
    ->Args({512, 512, 7, 7, 1, 1})      // Very large channels
    ->Args({64, 256, 56, 56, 1, 1})     // Channel expansion (ResNet)
    ->Args({256, 64, 56, 56, 1, 1})     // Channel reduction (ResNet)

    // ========== 3x3 Convolutions (Standard) ==========
    // Most common convolution type
    // VGG-style (stride=1)
    ->Args({3, 64, 224, 224, 3, 1})     // First layer (RGB input)
    ->Args({64, 64, 224, 224, 3, 1})    // VGG early layers
    ->Args({64, 128, 112, 112, 3, 1})   // VGG middle layers
    ->Args({128, 128, 112, 112, 3, 1})
    ->Args({128, 256, 56, 56, 3, 1})    // VGG deeper layers
    ->Args({256, 256, 56, 56, 3, 1})
    ->Args({256, 512, 28, 28, 3, 1})    // VGG very deep layers
    ->Args({512, 512, 28, 28, 3, 1})
    ->Args({512, 512, 14, 14, 3, 1})

    // ResNet-style (stride=1 and stride=2)
    ->Args({64, 64, 56, 56, 3, 1})      // ResNet basic block
    ->Args({64, 64, 56, 56, 3, 2})      // ResNet downsampling
    ->Args({128, 128, 28, 28, 3, 1})    // ResNet stage 2
    ->Args({128, 128, 28, 28, 3, 2})    // ResNet stage 2 downsample
    ->Args({256, 256, 14, 14, 3, 1})    // ResNet stage 3
    ->Args({256, 256, 14, 14, 3, 2})    // ResNet stage 3 downsample
    ->Args({512, 512, 7, 7, 3, 1})      // ResNet stage 4

    // ========== 5x5 Convolutions ==========
    // Less common but used in some architectures
    ->Args({32, 64, 112, 112, 5, 1})    // Early layers
    ->Args({64, 128, 56, 56, 5, 1})     // Middle layers
    ->Args({128, 256, 28, 28, 5, 2})    // With stride

    // ========== 7x7 Convolutions ==========
    // Typically used in first layer
    ->Args({3, 64, 224, 224, 7, 2})     // ResNet/VGG first layer
    ->Args({3, 64, 299, 299, 7, 2})     // Inception input size

    // ========== Small Input Sizes ==========
    // Mobile/embedded scenarios
    ->Args({32, 64, 32, 32, 3, 1})      // CIFAR-10 size
    ->Args({64, 128, 16, 16, 3, 1})     // Small feature maps
    ->Args({128, 256, 8, 8, 3, 1})      // Very small feature maps

    // ========== Large Input Sizes ==========
    // High-resolution scenarios
    ->Args({32, 64, 256, 256, 3, 1})    // Large input
    ->Args({64, 128, 128, 128, 3, 1})   // Large feature maps

    // ========== Small Channel Numbers ==========
    // Early layers or lightweight networks
    ->Args({3, 16, 224, 224, 3, 1})     // Very small output channels
    ->Args({16, 32, 112, 112, 3, 1})    // MobileNet-style
    ->Args({32, 32, 56, 56, 3, 1})      // Lightweight network

    // ========== Large Channel Numbers ==========
    // Deep layers in large networks
    ->Args({512, 1024, 14, 14, 3, 1})   // Very large channels
    ->Args({1024, 1024, 7, 7, 3, 1})    // Extremely large channels

    // ========== Asymmetric Input Sizes ==========
    // Real-world scenarios with non-square inputs
    ->Args({64, 128, 64, 32, 3, 1})     // Wide input
    ->Args({64, 128, 32, 64, 3, 1})     // Tall input
    ->Args({128, 256, 128, 64, 3, 1})   // Large asymmetric

    // ========== Edge Cases ==========
    // Extreme configurations
    ->Args({1, 1, 224, 224, 1, 1})      // Minimal channels
    ->Args({1, 64, 224, 224, 3, 1})     // Single input channel
    // ->Args({64, 1, 224, 224, 3, 1})     // Single output channel
    ->Args({2048, 2048, 7, 7, 1, 1})    // Extremely large channels

    // ========== MobileNet-specific ==========
    // Typical MobileNet configurations
    ->Args({32, 64, 112, 112, 1, 1})    // MobileNet expansion
    ->Args({64, 128, 56, 56, 1, 1})     // MobileNet projection
    ->Args({128, 256, 28, 28, 1, 1})    // MobileNet deeper layers

    // ========== Inception-style ==========
    // Multiple kernel sizes in parallel
    ->Args({192, 64, 28, 28, 1, 1})     // Inception 1x1
    ->Args({192, 128, 28, 28, 3, 1})    // Inception 3x3
    ->Args({192, 32, 28, 28, 5, 1});    // Inception 5x5

BENCHMARK_MAIN();