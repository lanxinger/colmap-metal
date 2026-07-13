//
//  SIFTExtrema.metal
//  SkyLight
//
//  Created by Luke Van In on 2023/01/07.
//

#include <metal_stdlib>

#include "../include/SIFTExtrema.h"

using namespace metal;


constant int3 neighborOffsets[] = {
    int3(-1, -1, -1),
    int3( 0, -1, -1),
    int3(+1, -1, -1),
    int3(-1,  0, -1),
    int3( 0,  0, -1),
    int3(+1,  0, -1),
    int3(-1, +1, -1),
    int3( 0, +1, -1),
    int3(+1, +1, -1),

    int3(-1, -1,  0),
    int3( 0, -1,  0),
    int3(+1, -1,  0),
    int3(-1,  0,  0),

    int3(+1,  0,  0),
    int3(-1, +1,  0),
    int3( 0, +1,  0),
    int3(+1, +1,  0),

    int3(-1, -1, +1),
    int3( 0, -1, +1),
    int3(+1, -1, +1),
    int3(-1,  0, +1),
    int3( 0,  0, +1),
    int3(+1,  0, +1),
    int3(-1, +1, +1),
    int3( 0, +1, +1),
    int3(+1, +1, +1),
};


static inline float fetch(
    texture2d_array<float, access::read> texture [[texture(0)]],
    const int2 g,
    const int s,
    const int i
) {
    const int3 neighborOffset = neighborOffsets[i];
    const int2 neighborDelta = g + neighborOffset.xy;
    const int textureIndex = s + neighborOffset.z;
    const float neighborValue =
        texture.read(uint2(neighborDelta), uint(textureIndex)).r;
    return neighborValue;
}


kernel void siftExtremaList(
    device SIFTExtremaResult * output [[buffer(0)]],
    device atomic_uint * outputCount [[buffer(1)]],
    constant SIFTExtremaParameters & parameters [[buffer(2)]],
    texture2d_array<float, access::read> inputTexture [[texture(0)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread grid runs [0...width - 2][0...height - 2][0...scales - 2]
    const int2 g = (int2)gid.xy + 1;
    const int s = (int)gid.z + 1;
    const float value = inputTexture.read(uint2(g), uint(s)).r;

    // Match VLFeat and SiftGPU by rejecting low-contrast samples before the
    // 26-neighbor scan. Without this guard, low-contrast extrema can overflow
    // the bounded candidate buffer and evict valid features before subpixel
    // localization applies the final contrast threshold.
    if (abs(value) < 0.8f * parameters.peakThreshold) {
        return;
    }

    float minimum = +1000;
    float maximum = -1000;

    for (int i = 0; i < 26; i++) {
        float neighborValue = fetch(inputTexture, g, s, i);
        minimum = min(minimum, neighborValue);
        maximum = max(maximum, neighborValue);
    }

    if ((value < minimum) || (value > maximum)) {
        // Extrema are rare, so contention on the global counter is
        // negligible; this avoids staging results in threadgroup memory.
        const uint i =
            atomic_fetch_add_explicit(outputCount, 1u, memory_order_relaxed);
        if (i < parameters.capacity) {
            SIFTExtremaResult result;
            result.x = g.x;
            result.y = g.y;
            result.scale = s;
            output[i] = result;
        }
    }
}
