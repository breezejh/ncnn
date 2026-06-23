// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_ConvolutionDepthWise(benchmark::State& state)
{
    int c = state.range(0);
    int outch = state.range(1);
    int w = state.range(2);
    int h = state.range(3);
    int kernel = state.range(4);
    int stride = state.range(5);
    int group = state.range(6);
    int dilation = 1;
    int pad = kernel / 2;
    int bias = 1;

    // Validate parameters
    if (c % group != 0 || outch % group != 0)
    {
        state.SkipWithError("Invalid group configuration");
        return;
    }

    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);

    // Calculate weight size
    // For depthwise convolution (c == group && group == outch): weight_data_size = kernel * kernel * group
    // For group convolution: weight_data_size = (outch/group) * (c/group) * kernel * kernel * group
    int maxk = kernel * kernel;
    int weight_data_size;
    if (c == group && group == outch)
    {
        // Pure depthwise convolution
        weight_data_size = maxk * group;
    }
    else
    {
        // Group convolution
        weight_data_size = (outch / group) * (c / group) * maxk * group;
    }

    pd.set(6, weight_data_size);
    pd.set(7, group);

    // Use fixed activation for consistent benchmarking
    int activation_type = 0; // No activation
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.f;
    activation_params[1] = 0.f;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(weight_data_size);
    weights[1] = RandomMat(outch);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_sgemm_convolution = false;
    opt.use_winograd_convolution = false;
    opt.use_packing_layout = false;  // Disable packing to avoid potential issues

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer_naive("ConvolutionDepthWise");
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

// Benchmark with different configurations: c, outch, w, h, kernel, stride, group
BENCHMARK(BM_ConvolutionDepthWise)
    // ========== Pure Depthwise Convolution (group = channels) ==========
    // MobileNet-style 3x3 kernels - most common configurations
    ->Args({32, 32, 112, 112, 3, 1, 32})    // MobileNet early layers
    ->Args({64, 64, 56, 56, 3, 1, 64})      // MobileNet middle layers
    ->Args({128, 128, 28, 28, 3, 1, 128})   // MobileNet deeper layers
    ->Args({256, 256, 14, 14, 3, 1, 256})   // MobileNet very deep layers
    ->Args({512, 512, 7, 7, 3, 1, 512})     // MobileNet final layers

    // Stride=2 for downsampling (key configurations)
    ->Args({64, 64, 56, 56, 3, 2, 64})      // Common downsampling
    ->Args({128, 128, 28, 28, 3, 2, 128})   // Deeper downsampling

    // 5x5 kernels (EfficientNet-style)
    ->Args({40, 40, 28, 28, 5, 1, 40})      // Medium channels with 5x5
    ->Args({112, 112, 14, 14, 5, 1, 112})   // Larger channels with 5x5

    // ========== MobileNetV2/V3 Configurations ==========
    // Representative channel counts
    ->Args({24, 24, 56, 56, 3, 1, 24})      // Small channels
    ->Args({96, 96, 14, 14, 3, 1, 96})      // MobileNetV2 style
    ->Args({160, 160, 7, 7, 3, 1, 160})     // Deep layers

    // ========== Group Convolution (group < channels) ==========
    // ResNeXt-style grouped convolutions
    ->Args({64, 64, 56, 56, 3, 1, 32})      // 2 channels per group
    ->Args({128, 128, 28, 28, 3, 1, 32})    // 4 channels per group
    ->Args({256, 256, 14, 14, 3, 1, 32})    // 8 channels per group

    // ShuffleNet-style
    ->Args({240, 240, 28, 28, 3, 1, 3})     // 80 channels per group
    ->Args({480, 480, 14, 14, 3, 1, 4})     // 120 channels per group

    // ========== Different Input Sizes ==========
    // Small (mobile/embedded)
    ->Args({64, 64, 16, 16, 3, 1, 64})      // Small feature maps

    // Large (high-resolution)
    ->Args({32, 32, 224, 224, 3, 1, 32})    // Large input

    // ========== Edge Cases ==========
    ->Args({8, 8, 112, 112, 3, 1, 8})       // Very small channel count
    ->Args({1024, 1024, 7, 7, 3, 1, 1024}); // Extremely large channels

BENCHMARK_MAIN();

