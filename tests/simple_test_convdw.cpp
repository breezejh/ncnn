#include "testutil.h"
#include <stdio.h>

int main()
{
    int w = 112;
    int h = 112;
    int c = 32;
    int outch = 32;
    int kernel = 3;
    int stride = 1;
    int group = 32;
    int dilation = 1;
    int pad = 1;
    int bias = 1;

    printf("Creating input mat: w=%d h=%d c=%d\n", w, h, c);
    ncnn::Mat a = RandomMat(w, h, c);
    printf("Input mat created successfully\n");

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);

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

    printf("Weight data size: %d\n", weight_data_size);
    pd.set(6, weight_data_size);
    pd.set(7, group);

    int activation_type = 0;
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.f;
    activation_params[1] = 0.f;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    printf("Creating weights\n");
    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(weight_data_size);
    weights[1] = RandomMat(outch);
    printf("Weights created successfully\n");

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_sgemm_convolution = false;
    opt.use_winograd_convolution = false;
    opt.use_packing_layout = false;  // Disable packing to use simpler code path

    printf("Creating layer\n");
    ncnn::Layer* layer = ncnn::create_layer("ConvolutionDepthWise");
    if (!layer)
    {
        printf("Failed to create layer\n");
        return -1;
    }
    printf("Layer created successfully\n");

    printf("Loading param\n");
    int ret = layer->load_param(pd);
    if (ret != 0)
    {
        printf("Failed to load param: %d\n", ret);
        delete layer;
        return -1;
    }
    printf("Param loaded successfully\n");

    printf("Loading model\n");
    ret = layer->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    if (ret != 0)
    {
        printf("Failed to load model: %d\n", ret);
        delete layer;
        return -1;
    }
    printf("Model loaded successfully\n");

    printf("Creating pipeline\n");
    ret = layer->create_pipeline(opt);
    if (ret != 0)
    {
        printf("Failed to create pipeline: %d\n", ret);
        delete layer;
        return -1;
    }
    printf("Pipeline created successfully\n");

    printf("Running forward\n");
    ncnn::Mat b;
    ret = layer->forward(a, b, opt);
    if (ret != 0)
    {
        printf("Forward failed: %d\n", ret);
        layer->destroy_pipeline(opt);
        delete layer;
        return -1;
    }
    printf("Forward completed successfully\n");

    printf("Output shape: w=%d h=%d c=%d\n", b.w, b.h, b.c);

    layer->destroy_pipeline(opt);
    delete layer;

    printf("Test passed!\n");
    return 0;
}

