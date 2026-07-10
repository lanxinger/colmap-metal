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


kernel void convolutionSeriesX(
    texture2d_array<float, access::write> outputTexture [[texture(0)]],
    texture2d_array<float, access::read> inputTexture [[texture(1)]],
    constant ConvolutionParameters & parameters [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int width = parameters.width;
    const int height = parameters.height;
    if (gid.x >= uint(width) || gid.y >= uint(height)) {
        return;
    }
    
    float sum = 0;
    const int n = (int)parameters.count;
    const int o = (int)gid.x - (n / 2);
    if (o >= 0 && o + n <= width) {
        // Interior: taps cannot cross the border.
        for (int i = 0; i < n; i++) {
            float c = inputTexture.read(uint2(uint(o + i), gid.y),
                                        parameters.inputDepth).r;
            sum += parameters.weights[i] * c;
        }
    } else {
        for (int i = 0; i < n; i++) {
            int x = symmetrizedCoordinates(o + i, width);
            float c =
                inputTexture.read(uint2(uint(x), gid.y), parameters.inputDepth).r;
            sum += parameters.weights[i] * c;
        }
    }
    outputTexture.write(float4(sum, 0, 0, 1), gid, parameters.outputDepth);
}


kernel void convolutionSeriesY(
    texture2d_array<float, access::write> outputTexture [[texture(0)]],
    texture2d_array<float, access::read> inputTexture [[texture(1)]],
    constant ConvolutionParameters & parameters [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int width = parameters.width;
    const int height = parameters.height;
    if (gid.x >= uint(width) || gid.y >= uint(height)) {
        return;
    }
    
    float sum = 0;
    const int n = (int)parameters.count;
    const int o = (int)gid.y - (n / 2);
    if (o >= 0 && o + n <= height) {
        // Interior: taps cannot cross the border.
        for (int i = 0; i < n; i++) {
            float c = inputTexture.read(uint2(gid.x, uint(o + i)),
                                        parameters.inputDepth).r;
            sum += parameters.weights[i] * c;
        }
    } else {
        for (int i = 0; i < n; i++) {
            int y = symmetrizedCoordinates(o + i, height);
            float c =
                inputTexture.read(uint2(gid.x, uint(y)), parameters.inputDepth).r;
            sum += parameters.weights[i] * c;
        }
    }
    outputTexture.write(float4(sum, 0, 0, 1), gid, parameters.outputDepth);
}
