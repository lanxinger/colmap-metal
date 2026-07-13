//
//  NearestNeighborDownScale.metal
//  SkyLight
//
//  Created by Luke Van In on 2023/01/07.
//

#include <metal_stdlib>

#import "../include/NearestNeighbor.h"

using namespace metal;


kernel void nearestNeighborDownScale(
    texture2d_array<float, access::write> outputTexture [[texture(0)]],
    texture2d_array<float, access::read> inputTexture [[texture(1)]],
    constant NearestNeighborScaleParameters & parameters [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint width = uint(parameters.outputWidth);
    const uint height = uint(parameters.outputHeight);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint inputWidth = uint(parameters.inputWidth);
    const uint inputHeight = uint(parameters.inputHeight);
    const uint2 source = min(gid * 2, uint2(inputWidth - 1, inputHeight - 1));
    outputTexture.write(inputTexture.read(source, parameters.inputSlice), gid, parameters.outputSlice);
}
