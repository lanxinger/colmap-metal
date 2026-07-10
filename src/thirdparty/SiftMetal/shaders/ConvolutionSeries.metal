//
//  Convolution.metal
//  SkyLight
//
//  Created by Luke Van In on 2023/01/07.
//

#include <metal_stdlib>

#include "Common.hpp"
#include "../include/ConvolutionSeries.h"

using namespace metal;


// Each thread computes CONVOLUTION_OUTPUTS_PER_THREAD consecutive outputs
// along the filter axis. Interior threads read the shared tap window once
// (n + K - 1 reads for K outputs instead of K * n); border threads fall back
// to per-output mirrored loops. Per-output accumulation order matches the
// single-output version, so results are bitwise identical.
kernel void convolutionSeriesX(
    texture2d_array<float, access::write> outputTexture [[texture(0)]],
    texture2d_array<float, access::read> inputTexture [[texture(1)]],
    constant ConvolutionParameters & parameters [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int width = parameters.width;
    const int height = parameters.height;
    const int x0 = (int)gid.x * CONVOLUTION_OUTPUTS_PER_THREAD;
    if (x0 >= width || gid.y >= uint(height)) {
        return;
    }

    const int n = (int)parameters.count;
    const int o = x0 - (n / 2);
    if (o >= 0 && o + n + 3 <= width && x0 + 4 <= width) {
        // Interior: all four outputs in bounds, no tap crosses the border.
        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int i = 0; i < n + 3; i++) {
            const float c = inputTexture.read(uint2(uint(o + i), gid.y),
                                              parameters.inputDepth).r;
            if (i < n) {
                sum0 += parameters.weights[i] * c;
            }
            if (i >= 1 && i - 1 < n) {
                sum1 += parameters.weights[i - 1] * c;
            }
            if (i >= 2 && i - 2 < n) {
                sum2 += parameters.weights[i - 2] * c;
            }
            if (i >= 3 && i - 3 < n) {
                sum3 += parameters.weights[i - 3] * c;
            }
        }
        outputTexture.write(float4(sum0, 0, 0, 1), uint2(uint(x0), gid.y),
                            parameters.outputDepth);
        outputTexture.write(float4(sum1, 0, 0, 1), uint2(uint(x0 + 1), gid.y),
                            parameters.outputDepth);
        outputTexture.write(float4(sum2, 0, 0, 1), uint2(uint(x0 + 2), gid.y),
                            parameters.outputDepth);
        outputTexture.write(float4(sum3, 0, 0, 1), uint2(uint(x0 + 3), gid.y),
                            parameters.outputDepth);
    } else {
        for (int k = 0; k < CONVOLUTION_OUTPUTS_PER_THREAD; k++) {
            const int x = x0 + k;
            if (x >= width) {
                break;
            }
            float sum = 0;
            const int ok = x - (n / 2);
            for (int i = 0; i < n; i++) {
                const int sx = symmetrizedCoordinates(ok + i, width);
                const float c = inputTexture.read(uint2(uint(sx), gid.y),
                                                  parameters.inputDepth).r;
                sum += parameters.weights[i] * c;
            }
            outputTexture.write(float4(sum, 0, 0, 1), uint2(uint(x), gid.y),
                                parameters.outputDepth);
        }
    }
}


kernel void convolutionSeriesY(
    texture2d_array<float, access::write> outputTexture [[texture(0)]],
    texture2d_array<float, access::read> inputTexture [[texture(1)]],
    constant ConvolutionParameters & parameters [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int width = parameters.width;
    const int height = parameters.height;
    const int y0 = (int)gid.y * CONVOLUTION_OUTPUTS_PER_THREAD;
    if (gid.x >= uint(width) || y0 >= height) {
        return;
    }

    const int n = (int)parameters.count;
    const int o = y0 - (n / 2);
    if (o >= 0 && o + n + 3 <= height && y0 + 4 <= height) {
        // Interior: all four outputs in bounds, no tap crosses the border.
        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int i = 0; i < n + 3; i++) {
            const float c = inputTexture.read(uint2(gid.x, uint(o + i)),
                                              parameters.inputDepth).r;
            if (i < n) {
                sum0 += parameters.weights[i] * c;
            }
            if (i >= 1 && i - 1 < n) {
                sum1 += parameters.weights[i - 1] * c;
            }
            if (i >= 2 && i - 2 < n) {
                sum2 += parameters.weights[i - 2] * c;
            }
            if (i >= 3 && i - 3 < n) {
                sum3 += parameters.weights[i - 3] * c;
            }
        }
        outputTexture.write(float4(sum0, 0, 0, 1), uint2(gid.x, uint(y0)),
                            parameters.outputDepth);
        outputTexture.write(float4(sum1, 0, 0, 1), uint2(gid.x, uint(y0 + 1)),
                            parameters.outputDepth);
        outputTexture.write(float4(sum2, 0, 0, 1), uint2(gid.x, uint(y0 + 2)),
                            parameters.outputDepth);
        outputTexture.write(float4(sum3, 0, 0, 1), uint2(gid.x, uint(y0 + 3)),
                            parameters.outputDepth);
    } else {
        for (int k = 0; k < CONVOLUTION_OUTPUTS_PER_THREAD; k++) {
            const int y = y0 + k;
            if (y >= height) {
                break;
            }
            float sum = 0;
            const int ok = y - (n / 2);
            for (int i = 0; i < n; i++) {
                const int sy = symmetrizedCoordinates(ok + i, height);
                const float c = inputTexture.read(uint2(gid.x, uint(sy)),
                                                  parameters.inputDepth).r;
                sum += parameters.weights[i] * c;
            }
            outputTexture.write(float4(sum, 0, 0, 1), uint2(gid.x, uint(y)),
                                parameters.outputDepth);
        }
    }
}
