// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_DeconvolutionDepthWise(benchmark::State& state)
{
    int w = state.range(0);
    int h = state.range(1);
    int c = state.range(2);
    int outch = state.range(3);
    int kernel = state.range(4);
    int stride = state.range(5);
    int pad = state.range(6);
    int group = state.range(7);

    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, 1);  // dilation
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, 1);  // bias_term
    pd.set(6, outch / group * c / group * kernel * kernel * group);
    pd.set(7, group);
    pd.set(9, 0);  // activation_type
    pd.set(18, 0); // output_pad_right
    pd.set(19, 0); // output_pad_bottom
    pd.set(20, 0); // output_w
    pd.set(21, 0); // output_h

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outch / group * c / group * kernel * kernel * group);
    weights[1] = RandomMat(outch);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;  // Disable packing to avoid potential issues

    // Use create_layer to get optimized implementation
    ncnn::Layer* layer = ncnn::create_layer("DeconvolutionDepthWise");
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

// Benchmark with different configurations: w, h, c, outch, kernel, stride, pad, group
BENCHMARK(BM_DeconvolutionDepthWise)
    // ========== Semantic Segmentation (Upsampling) ==========
    // FCN/SegNet style: 2x upsampling
    ->Args({28, 28, 512, 512, 4, 2, 1, 512})  // Deep layer upsampling
    ->Args({56, 56, 256, 256, 4, 2, 1, 256})  // Middle layer upsampling
    ->Args({112, 112, 128, 128, 4, 2, 1, 128}) // Shallow layer upsampling
    
    // U-Net style: 2x upsampling with skip connections
    ->Args({32, 32, 512, 256, 2, 2, 0, 256})  // Decoder block 1
    ->Args({64, 64, 256, 128, 2, 2, 0, 128})  // Decoder block 2
    ->Args({128, 128, 128, 64, 2, 2, 0, 64})  // Decoder block 3
    
    // ========== Super Resolution ==========
    // 2x super resolution
    ->Args({128, 128, 64, 64, 3, 2, 1, 64})   // SR 2x
    ->Args({256, 256, 32, 32, 3, 2, 1, 32})   // SR 2x (larger)
    
    // 4x super resolution (stride=4)
    ->Args({64, 64, 64, 64, 8, 4, 2, 64})     // SR 4x
    
    // ========== MobileNet-style Depthwise Deconv ==========
    // Depthwise separable transpose convolution
    ->Args({56, 56, 32, 32, 3, 2, 1, 32})     // MobileNet decoder
    ->Args({28, 28, 64, 64, 3, 2, 1, 64})     // MobileNet decoder
    ->Args({14, 14, 128, 128, 3, 2, 1, 128})  // MobileNet decoder
    
    // ========== Different Kernel Sizes ==========
    // Small kernel (2x2)
    ->Args({56, 56, 128, 128, 2, 2, 0, 128})  // 2x2 kernel
    
    // Medium kernel (3x3)
    ->Args({56, 56, 128, 128, 3, 2, 1, 128})  // 3x3 kernel
    
    // Large kernel (4x4)
    ->Args({56, 56, 128, 128, 4, 2, 1, 128})  // 4x4 kernel
    
    // Very large kernel (5x5)
    ->Args({28, 28, 256, 256, 5, 2, 2, 256})  // 5x5 kernel
    
    // ========== Different Strides ==========
    // Stride=1 (no upsampling, just convolution)
    ->Args({56, 56, 128, 128, 3, 1, 1, 128})  // stride=1
    
    // Stride=2 (2x upsampling)
    ->Args({56, 56, 128, 128, 3, 2, 1, 128})  // stride=2
    
    // Stride=4 (4x upsampling)
    ->Args({28, 28, 256, 256, 7, 4, 3, 256})  // stride=4
    
    // ========== Different Channel Counts ==========
    // Few channels
    ->Args({112, 112, 16, 16, 3, 2, 1, 16})   // 16 channels
    ->Args({112, 112, 32, 32, 3, 2, 1, 32})   // 32 channels
    
    // Many channels
    ->Args({28, 28, 256, 256, 3, 2, 1, 256})  // 256 channels
    ->Args({14, 14, 512, 512, 3, 2, 1, 512})  // 512 channels
    
    // ========== Different Input Sizes ==========
    // Small input
    ->Args({14, 14, 256, 256, 3, 2, 1, 256})  // 14x14 input
    
    // Medium input
    ->Args({56, 56, 128, 128, 3, 2, 1, 128})  // 56x56 input
    
    // Large input
    ->Args({112, 112, 64, 64, 3, 2, 1, 64})   // 112x112 input
    
    // ========== Edge Cases ==========
    ->Args({7, 7, 512, 512, 2, 2, 0, 512})    // Very small input
    ->Args({224, 224, 16, 16, 3, 2, 1, 16});  // Large input, few channels

BENCHMARK_MAIN();

