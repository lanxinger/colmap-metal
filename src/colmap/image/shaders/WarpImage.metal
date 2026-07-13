#include <metal_stdlib>

using namespace metal;

struct WarpImageParameters {
  uint width;
  uint height;
  uint channels;
  uint pixel_count;
};

struct SimpleRadialToPinholeParameters {
  float source_focal_length;
  float source_principal_point_x;
  float source_principal_point_y;
  float source_radial;
  float target_focal_length_x;
  float target_focal_length_y;
  float target_principal_point_x;
  float target_principal_point_y;
};

struct ResizeImageParameters {
  uint source_width;
  uint source_height;
  uint target_width;
  uint target_height;
  uint channels;
  uint pixel_count;
};

struct LanczosWeightParameters {
  uint source_length;
  uint target_length;
  uint radius;
  uint weight_count;
};

static float sinc(const float x) {
  if (abs(x) < 1e-5f) {
    return 1.0f;
  }
  const float angle = M_PI_F * x;
  return sin(angle) / angle;
}

static float lanczos3(const float x) {
  return abs(x) < 3.0f ? sinc(x) * sinc(x / 3.0f) : 0.0f;
}

static void writeWarpedPixel(device const uchar* source,
                             device uchar* target,
                             const float2 source_point,
                             constant WarpImageParameters& params,
                             const uint pixel_index) {
  const uint target_offset = pixel_index * params.channels;
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

  writeWarpedPixel(
      source, target, source_coordinates[pixel_index], params, pixel_index);
}

kernel void warpImageSimpleRadialToPinhole(
    device const uchar* source [[buffer(0)]],
    device uchar* target [[buffer(1)]],
    constant WarpImageParameters& params [[buffer(2)]],
    constant SimpleRadialToPinholeParameters& camera [[buffer(3)]],
    uint pixel_index [[thread_position_in_grid]]) {
  if (pixel_index >= params.pixel_count) {
    return;
  }

  const uint target_x = pixel_index % params.width;
  const uint target_y = pixel_index / params.width;
  const float2 normalized =
      float2((float(target_x) + 0.5f - camera.target_principal_point_x) /
                 camera.target_focal_length_x,
             (float(target_y) + 0.5f - camera.target_principal_point_y) /
                 camera.target_focal_length_y);
  const float radial =
      1.0f + camera.source_radial * dot(normalized, normalized);
  const float2 source_point =
      camera.source_focal_length * radial * normalized +
      float2(camera.source_principal_point_x, camera.source_principal_point_y) -
      0.5f;
  writeWarpedPixel(source, target, source_point, params, pixel_index);
}

kernel void buildLanczosWeights(device float* weights [[buffer(0)]],
                                constant LanczosWeightParameters& params
                                [[buffer(1)]],
                                uint weight_index [[thread_position_in_grid]]) {
  if (weight_index >= params.weight_count) {
    return;
  }

  const uint tap_count = 2 * params.radius + 1;
  const uint target_index = weight_index / tap_count;
  const int tap = int(weight_index % tap_count) - int(params.radius);
  const float scale = float(params.source_length) / float(params.target_length);
  const float source_position = (float(target_index) + 0.5f) * scale;
  const float source_fraction = source_position - floor(source_position);
  weights[weight_index] =
      lanczos3((float(tap) - (source_fraction - 0.5f)) / scale);
}

kernel void resizeLanczosHorizontal(device const uchar* source [[buffer(0)]],
                                    device float* intermediate [[buffer(1)]],
                                    device const float* weights [[buffer(2)]],
                                    constant ResizeImageParameters& params
                                    [[buffer(3)]],
                                    uint pixel_index
                                    [[thread_position_in_grid]]) {
  if (pixel_index >= params.pixel_count) {
    return;
  }

  const uint target_x = pixel_index % params.target_width;
  const uint source_y = pixel_index / params.target_width;
  const float scale = float(params.source_width) / float(params.target_width);
  const float source_position = (float(target_x) + 0.5f) * scale;
  const int center_source_x = int(floor(source_position));
  const int radius = int(ceil(3.0f * scale));
  const uint tap_count = uint(2 * radius + 1);
  const uint weight_offset = target_x * tap_count;

  const uint target_offset = pixel_index * params.channels;
  float4 weighted_sum = 0.0f;
  float weight_sum = 0.0f;
  for (int tap = -radius; tap <= radius; ++tap) {
    const float weight = weights[weight_offset + uint(tap + radius)];
    weight_sum += weight;
    const int source_x =
        clamp(center_source_x + tap, 0, int(params.source_width) - 1);
    const uint source_offset =
        (source_y * params.source_width + uint(source_x)) * params.channels;
    for (uint channel = 0; channel < params.channels; ++channel) {
      weighted_sum[channel] += weight * float(source[source_offset + channel]);
    }
  }
  for (uint channel = 0; channel < params.channels; ++channel) {
    intermediate[target_offset + channel] = weighted_sum[channel] / weight_sum;
  }
}

kernel void resizeLanczosVertical(device const float* intermediate
                                  [[buffer(0)]],
                                  device uchar* target [[buffer(1)]],
                                  device const float* weights [[buffer(2)]],
                                  constant ResizeImageParameters& params
                                  [[buffer(3)]],
                                  uint pixel_index
                                  [[thread_position_in_grid]]) {
  if (pixel_index >= params.pixel_count) {
    return;
  }

  const uint target_x = pixel_index % params.target_width;
  const uint target_y = pixel_index / params.target_width;
  const float scale = float(params.source_height) / float(params.target_height);
  const float source_position = (float(target_y) + 0.5f) * scale;
  const int center_source_y = int(floor(source_position));
  const int radius = int(ceil(3.0f * scale));
  const uint tap_count = uint(2 * radius + 1);
  const uint weight_offset = target_y * tap_count;

  const uint target_offset = pixel_index * params.channels;
  float4 weighted_sum = 0.0f;
  float weight_sum = 0.0f;
  for (int tap = -radius; tap <= radius; ++tap) {
    const float weight = weights[weight_offset + uint(tap + radius)];
    weight_sum += weight;
    const int source_y =
        clamp(center_source_y + tap, 0, int(params.source_height) - 1);
    const uint source_offset =
        (uint(source_y) * params.target_width + target_x) * params.channels;
    for (uint channel = 0; channel < params.channels; ++channel) {
      weighted_sum[channel] += weight * intermediate[source_offset + channel];
    }
  }
  for (uint channel = 0; channel < params.channels; ++channel) {
    const float value = weighted_sum[channel] / weight_sum;
    target[target_offset + channel] =
        uchar(clamp(floor(value + 0.5f), 0.0f, 255.0f));
  }
}
