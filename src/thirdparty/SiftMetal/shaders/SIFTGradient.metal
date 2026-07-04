//
//  SIFTGradient.metal
//  SkyLight
//
//  Created by Luke Van In on 2023/01/07.
//

#include <metal_stdlib>

#include "Common.hpp"

using namespace metal;


kernel void siftGradient(
     texture2d_array<float, access::write> outputTexture [[texture(0)]],
     texture2d_array<float, access::read> inputTexture [[texture(1)]],
     uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outputTexture.get_width() ||
        gid.y >= outputTexture.get_height() ||
        gid.z >= outputTexture.get_array_size()) {
        return;
    }

    const int gx = (int)gid.x;
    const int gy = (int)gid.y;
    const uint gz = gid.z;
    const int dx = inputTexture.get_width();
    const int dy = inputTexture.get_height();
    const uint px = uint(symmetrizedCoordinates(gx + 1, dx));
    const uint mx = uint(symmetrizedCoordinates(gx - 1, dx));
    const uint py = uint(symmetrizedCoordinates(gy + 1, dy));
    const uint my = uint(symmetrizedCoordinates(gy - 1, dy));
    const float cpx = inputTexture.read(uint2(px, gid.y), gz).r;
    const float cmx = inputTexture.read(uint2(mx, gid.y), gz).r;
    const float cpy = inputTexture.read(uint2(gid.x, py), gz).r;
    const float cmy = inputTexture.read(uint2(gid.x, my), gz).r;
    const float tx = (cpx - cmx) * 0.5;
    const float ty = (cpy - cmy) * 0.5;
    // Orientation and descriptor kernels both consume this convention.
    float oa = atan2(tx, ty);
    float om = sqrt(tx * tx + ty * ty);
    outputTexture.write(float4(oa, om, 0, 0), gid.xy, gz);
}
