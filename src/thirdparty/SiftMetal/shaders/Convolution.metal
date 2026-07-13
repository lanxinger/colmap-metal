//
//  Convolution.metal
//  SkyLight
//
//  Created by Luke Van In on 2023/01/07.
//

#include <metal_stdlib>

#include "Common.hpp"
#include "../include/SeedStage.h"

using namespace metal;


kernel void convolutionX(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    texture2d<float, access::read> inputTexture [[texture(1)]],
    constant float * weights [[buffer(0)]],
    constant SeedConvolutionParameters & parameters [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int width = parameters.width;
    const int height = parameters.height;
    if (gid.x >= uint(width) || gid.y >= uint(height)) {
        return;
    }
    
    float sum = 0;
    const int n = parameters.count;
    const int o = (int)gid.x - (n / 2);
    if (o >= 0 && o + n <= width) {
        // Interior: taps cannot cross the border.
        for (int i = 0; i < n; i++) {
            sum += weights[i] * inputTexture.read(uint2(uint(o + i), gid.y)).r;
        }
    } else {
        for (int i = 0; i < n; i++) {
            int x = symmetrizedCoordinates(o + i, width);
            sum += weights[i] * inputTexture.read(uint2(uint(x), gid.y)).r;
        }
    }
    outputTexture.write(float4(sum, 0, 0, 1), gid);
}


kernel void convolutionY(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    texture2d<float, access::read> inputTexture [[texture(1)]],
    constant float * weights [[buffer(0)]],
    constant SeedConvolutionParameters & parameters [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int width = parameters.width;
    const int height = parameters.height;
    if (gid.x >= uint(width) || gid.y >= uint(height)) {
        return;
    }
    
    float sum = 0;
    const int n = parameters.count;
    const int o = (int)gid.y - (n / 2);
    if (o >= 0 && o + n <= height) {
        // Interior: taps cannot cross the border.
        for (int i = 0; i < n; i++) {
            sum += weights[i] * inputTexture.read(uint2(gid.x, uint(o + i))).r;
        }
    } else {
        for (int i = 0; i < n; i++) {
            int y = symmetrizedCoordinates(o + i, height);
            sum += weights[i] * inputTexture.read(uint2(gid.x, uint(y))).r;
        }
    }
    outputTexture.write(float4(sum, 0, 0, 1), gid);
}
