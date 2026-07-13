#include <metal_stdlib>

using namespace metal;

struct WarpImageParameters {
  uint width;
  uint height;
  uint channels;
  uint pixel_count;
};

kernel void warpImageBilinear(device const uchar* source [[buffer(0)]],
                              device const float2* source_coordinates
                              [[buffer(1)]],
                              device uchar* target [[buffer(2)]],
                              constant WarpImageParameters& params
                              [[buffer(3)]],
                              uint pixel_index [[thread_position_in_grid]]) {
  if (pixel_index >= params.pixel_count) {
    return;
  }

  const uint target_offset = pixel_index * params.channels;
  const float2 source_point = source_coordinates[pixel_index];
  if (!isfinite(source_point.x) || !isfinite(source_point.y)) {
    for (uint channel = 0; channel < params.channels; ++channel) {
      target[target_offset + channel] = 0;
    }
    return;
  }

  const int x0 = int(floor(source_point.x));
  const int y0 = int(floor(source_point.y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  if (x0 < 0 || y0 < 0 || x1 >= int(params.width) || y1 >= int(params.height)) {
    for (uint channel = 0; channel < params.channels; ++channel) {
      target[target_offset + channel] = 0;
    }
    return;
  }

  const float dx = source_point.x - float(x0);
  const float dy = source_point.y - float(y0);
  const uint offset00 = (uint(y0) * params.width + uint(x0)) * params.channels;
  const uint offset01 = offset00 + params.channels;
  const uint offset10 = (uint(y1) * params.width + uint(x0)) * params.channels;
  const uint offset11 = offset10 + params.channels;

  for (uint channel = 0; channel < params.channels; ++channel) {
    const float top = mix(float(source[offset00 + channel]),
                          float(source[offset01 + channel]),
                          dx);
    const float bottom = mix(float(source[offset10 + channel]),
                             float(source[offset11 + channel]),
                             dx);
    const float value = mix(top, bottom, dy);
    target[target_offset + channel] =
        uchar(clamp(floor(value + 0.5f), 0.0f, 255.0f));
  }
}
