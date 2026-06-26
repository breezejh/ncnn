// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include <benchmark/benchmark.h>

static void BM_Softmax(benchmark::State& state)
{
    int axis = state.range(0);
    int channels = state.range(1);
    int w = state.range(2);
    int h = state.range(3);
    int d = state.range(4);

    ncnn::Mat a;
    if (d == 1)
        a = RandomMat(w, h, channels);
    else
        a = RandomMat(w, h, d, channels);

    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    ncnn::Option opt;

    opt.num_threads = 1;

    ncnn::Layer* layer = ncnn::create_layer("Softmax");
    layer->load_param(pd);

    ncnn::Mat b;
    for (auto _ : state)
    {
        layer->forward(a, b, opt);
    }

    delete layer;
}

BENCHMARK(BM_Softmax)
    ->Args({2, 32, 112, 112, 1})  // axis, channels, w, h, d
    ->Args({2, 64, 56, 56, 1})
    ->Args({2, 128, 28, 28, 1})
    ->Args({2, 256, 14, 14, 1})
    ->Args({2, 512, 7, 7, 1})
    // Additional cases for comprehensive coverage
    ->Args({0, 32, 112, 112, 1})  // axis=0 for dims=3
    ->Args({1, 32, 112, 112, 1})  // axis=1 for dims=3
    ->Args({2, 1, 112, 112, 1})   // small channels
    ->Args({2, 33, 112, 112, 1})  // non-divisible by SIMD vector length (e.g., 8 for AVX)
    ->Args({2, 1024, 7, 7, 1})    // large channels
    ->Args({2, 32, 1, 1, 1})      // small spatial dimensions
    ->Args({2, 32, 224, 224, 1})  // large spatial dimensions
    ->Args({2, 32, 28, 28, 2})    // dims=4 (4D tensor)
    ->Args({3, 32, 14, 14, 2})    // axis=3 for dims=4
    ->Args({0, 32, 14, 14, 2})    // axis=0 for dims=4
    ->Args({1, 32, 14, 14, 2})    // axis=1 for dims=4
    ->Args({2, 64, 14, 14, 4});   // larger depth for dims=4

    BENCHMARK_MAIN();