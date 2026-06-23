// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_DeformableConv2D(benchmark::State& state)
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

    // Calculate output dimensions
    const int kernel_extent_w = dilation * (kernel - 1) + 1;
    const int kernel_extent_h = dilation * (kernel - 1) + 1;
    const int out_w = (w + pad + pad - kernel_extent_w) / stride + 1;
    const int out_h = (h + pad + pad - kernel_extent_h) / stride + 1;

    // DeformableConv2D requires 3 inputs:
    // 1. Input feature map
    // 2. Offset field (2 * kernel * kernel channels for x,y offsets)
    // 3. Mask field (kernel * kernel channels for modulation)
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(w, h, c);                              // Input feature map
    a[1] = RandomMat(out_w, out_h, kernel * kernel * 2);   // Offset field
    a[2] = RandomMat(out_w, out_h, kernel * kernel);       // Mask field

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
    ncnn::Layer* layer = ncnn::create_layer("DeformableConv2D");
    layer->load_param(pd);
    layer->load_model(ncnn::ModelBinFromMatArray(weights.data()));

    // Create pipeline to initialize optimized kernels
    layer->create_pipeline(opt);

    std::vector<ncnn::Mat> b(1);
    for (auto _ : state)
    {
        layer->forward(a, b, opt);
    }

    // Clean up pipeline resources
    layer->destroy_pipeline(opt);
    delete layer;
}

// Benchmark with different configurations: c, outch, w, h, kernel, stride
BENCHMARK(BM_DeformableConv2D)
    // ========== 3x3 Deformable Convolutions ==========
    // Most common configuration for deformable convolution
    // Object detection scenarios (e.g., DCN in Faster R-CNN, Mask R-CNN)
    ->Args({64, 64, 56, 56, 3, 1})      // Small feature maps
    ->Args({128, 128, 28, 28, 3, 1})    // Medium feature maps
    ->Args({256, 256, 14, 14, 3, 1})    // Large channels
    ->Args({512, 512, 7, 7, 3, 1})      // Very large channels

    // With stride=2 for downsampling
    ->Args({64, 128, 56, 56, 3, 2})     // Downsampling with channel expansion
    ->Args({128, 256, 28, 28, 3, 2})    // Deeper downsampling
    ->Args({256, 512, 14, 14, 3, 2});   // Very deep downsampling

BENCHMARK_MAIN();

