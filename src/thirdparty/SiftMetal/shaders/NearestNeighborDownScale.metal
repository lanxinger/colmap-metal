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
    ushort2 gid [[thread_position_in_grid]]
) {
    const ushort width = ushort(outputTexture.get_width());
    const ushort height = ushort(outputTexture.get_height());
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const ushort inputWidth = ushort(inputTexture.get_width());
    const ushort inputHeight = ushort(inputTexture.get_height());
    const ushort2 source = min(gid * 2, ushort2(inputWidth - 1, inputHeight - 1));
    outputTexture.write(inputTexture.read(source, parameters.inputSlice), gid, parameters.outputSlice);
}
