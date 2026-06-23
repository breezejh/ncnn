// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_BatchNorm(benchmark::State& state)
{
    int channels = state.range(0);
    int w = state.range(1);
    int h = state.range(2);
    int d = state.range(3);

    ncnn::Mat a;
    if (d == 1)
        a = RandomMat(w, h, channels);
    else
        a = RandomMat(w, h, d, channels);

    ncnn::ParamDict pd;
    pd.set(0, channels); // channels
    pd.set(1, 0.001f);   // eps

    std::vector<ncnn::Mat> weights(4);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);
    weights[2] = RandomMat(channels);
    weights[3] = RandomMat(channels);

    // var must be positive
    Randomize(weights[2], 0.001f, 2.f);

    ncnn::Option opt;

    opt.num_threads = 1;
    // opt.lightmode = false;
    // opt.use_packing_layout = false;
    // opt.use_fp16_packed = false;
    // opt.use_fp16_storage = false;
    // opt.use_fp16_arithmetic = false;
    // opt.use_bf16_packed = false;
    // opt.use_bf16_storage = false;
    // opt.use_vulkan_compute = false;
    ncnn::Layer* layer = ncnn::create_layer_naive("BatchNorm");
    layer->load_param(pd);
    layer->load_model(ncnn::ModelBinFromMatArray(weights.data()));

    ncnn::Mat b;
    for (auto _ : state)
    {
        layer->forward(a, b, opt);
    }

    delete layer;
}

BENCHMARK(BM_BatchNorm)
    ->Args({32, 112, 112, 1})  // channels, w, h, d
    ->Args({64, 56, 56, 1})
    ->Args({128, 28, 28, 1})
    ->Args({256, 14, 14, 1})
    ->Args({512, 7, 7, 1})
    // Additional cases for comprehensive coverage
    ->Args({1, 112, 112, 1})   // small channels
    ->Args({3, 112, 112, 1})   // small channels, non-divisible by SIMD
    ->Args({33, 112, 112, 1})  // non-divisible by SIMD vector length (e.g., 8 for AVX)
    ->Args({1024, 7, 7, 1})    // large channels
    ->Args({32, 1, 1, 1})      // small spatial dimensions
    ->Args({32, 224, 224, 1})  // large spatial dimensions
    ->Args({32, 28, 28, 2})    // dims=4 (4D tensor)
    ->Args({64, 14, 14, 4})    // larger depth for dims=4
    ->Args({128, 7, 7, 8});    // even larger depth

    BENCHMARK_MAIN();
