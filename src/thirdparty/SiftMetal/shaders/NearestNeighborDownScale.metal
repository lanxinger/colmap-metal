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
    device NearestNeighborScaleParameters & parameters [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint width = outputTexture.get_width();
    const uint height = outputTexture.get_height();
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint inputWidth = inputTexture.get_width();
    const uint inputHeight = inputTexture.get_height();
    const uint2 source = min(gid * 2, uint2(inputWidth - 1, inputHeight - 1));
    outputTexture.write(inputTexture.read(source, parameters.inputSlice), gid, parameters.outputSlice);
}
