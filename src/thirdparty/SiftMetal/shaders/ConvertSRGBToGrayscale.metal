//
//  ConvertSRGBToGrayscale.metal
//  SkyLight
//
//  Created by Luke Van In on 2023/01/07.
//

#include <metal_stdlib>
using namespace metal;

kernel void convertSRGBToGrayscale(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    texture2d<float, access::read> inputTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outputTexture.get_width() ||
        gid.y >= outputTexture.get_height()) {
        return;
    }

    const float4 input = inputTexture.read(gid);
    const float i = 0 +
        (0.212639005871510 * input.r) +
        (0.715168678767756 * input.g) +
        (0.072192315360734 * input.b);
    const float4 output = float4(i, i, i, input.a);
    outputTexture.write(output, gid);
}
