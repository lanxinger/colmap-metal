//
//  SIFTDescriptor.metal
//  SkyLight
//
//  Created by Luke Van In on 2023/01/08.
//

#include <metal_stdlib>

#include "Common.hpp"
#include "../include/SIFTDescriptor.h"

using namespace metal;


static bool normalizeFeatures(
    int count,
    threadgroup float * features
) {
    float magnitude = 0;
    for (int i = 0; i < count; i++) {
        float f = features[i];
        if (!isfinite(f)) {
            return false;
        }
        magnitude += (f * f);
    }
    if (!isfinite(magnitude) || magnitude <= 0.0f) {
        return false;
    }
    const float d = 1.0 / sqrt(magnitude);
    for (int i = 0; i < count; i++) {
        features[i] *= d;
    }
    return true;
}


static void thresholdFeatures(
    int count,
    threadgroup float * features,
    float threshold
) {
    for (int i = 0; i < count; i++) {
        features[i] = min(features[i], threshold);
    }
}


// One threadgroup of d*d threads per descriptor. Thread t owns spatial
// histogram cell (t % d, t / d) and its orientation bins, stored at
// features[t * bins ... t * bins + bins), which matches the serial layout
// (y * d + x) * bins + b. Every thread walks the full sampling window (the
// per-sample texture read and setup are uniform across the threadgroup and
// served from cache); each applies its own cell's bilinear weight
// max(0, 1 - |bx - cellX|) * max(0, 1 - |by - cellY|), which reproduces the
// original scatter exactly, including the degenerate integer-coordinate
// cases. This removes the previous per-thread 128-float histogram (register
// spill) and parallelizes the scatter.
kernel void siftDescriptors(
    device SIFTDescriptorResult * results [[buffer(0)]],
    device SIFTDescriptorInput * inputs [[buffer(1)]],
    constant SIFTDescriptorParameters & parameters [[buffer(2)]],
    texture2d_array<float, access::read> gaussianTextures [[texture(0)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const int d = SIFT_DESCRIPTOR_HISTOGRAM_WIDTH; // 4
    const int bins = SIFT_DESCRIPTOR_ORIENTATION_BINS; // 8
    const int featureCount = d * d * bins; // 128
    const float tau = 2 * M_PI_F;

    threadgroup float features[SIFT_DESCRIPTOR_FEATURE_COUNT];

    const SIFTDescriptorInput input = inputs[groupId];

    const float px = float(input.absoluteX) / parameters.delta;
    const float py = float(input.absoluteY) / parameters.delta;
    const int maxScale = parameters.scalesPerOctave + 2;

    // Keep the descriptor bounds check here so descriptor sampling never
    // reads outside the gradient source texture. All inputs are uniform
    // across the threadgroup, so this branch is uniform.
    const bool valid = isfinite(input.theta) &&
                       px >= 0.0f && py >= 0.0f &&
                       px < (float)parameters.width &&
                       py < (float)parameters.height &&
                       input.scale >= 0 && input.scale <= maxScale;
    if (!valid) {
        if (tid == 0) {
            device SIFTDescriptorResult & result = results[groupId];
            result.valid = 0;
            result.keypoint = input.keypoint;
            result.theta = input.theta;
        }
        return;
    }

    const float cosT = cos(input.theta);
    const float sinT = sin(input.theta);
    const float binsPerRadian = (float)bins / tau;
    const float exponentDenominator = (float)(d * d) * 0.5;
    const float interval = (float)input.scale + input.subScale;
    const float intervals = (float)parameters.scalesPerOctave;
    const float sigma = 1.6;
    const float scale = sigma * pow(2.0, interval / intervals);
    const float histogramWidth = 3.0 * scale; // 3.0 constant from Whess (OpenSIFT)
    const int radius = histogramWidth * sqrt(2.0) * ((float)d + 1.0) * 0.5 + 0.5;

    // This thread's histogram cell and its bins.
    const int cellX = int(tid) % d;
    const int cellY = int(tid) / d;
    threadgroup float * cell = features + tid * bins;
    for (int b = 0; b < bins; b++) {
        cell[b] = 0;
    }

    for (int j = -radius; j <= +radius; j++) {
        for (int i = -radius; i <= +radius; i++) {
            const int sampleX = int(px + j);
            const int sampleY = int(py + i);
            if (sampleX < 1 || sampleY < 1 ||
                sampleX >= parameters.width - 1 ||
                sampleY >= parameters.height - 1) {
                continue;
            }

            float rx = ((float)j * cosT - (float)i * sinT) / histogramWidth;
            float ry = ((float)j * sinT + (float)i * cosT) / histogramWidth;
            float bx = rx + (float)(d / 2) - 0.5;
            float by = ry + (float)(d / 2) - 0.5;
            if (!isfinite(bx) || !isfinite(by)) {
                continue;
            }

            float2 g = siftGradientAt(
                gaussianTextures, sampleX, sampleY, uint(input.scale));
            if (!all(isfinite(g)) || g.g <= 0.0f) {
                continue;
            }
            float orientation = g.r - input.theta;
            float magnitude = g.g;
            while (orientation < 0) {
                orientation += tau;
            }
            while (orientation >= tau) {
                orientation -= tau;
            }

            // Bin
            float bin = orientation * binsPerRadian;
            if (!isfinite(bin)) {
                continue;
            }

            // Total contribution
            float exponentNumerator = rx * rx + ry * ry;
            float w = exp(-exponentNumerator / exponentDenominator);
            float value = magnitude * w;
            if (!isfinite(value) || value <= 0.0f) {
                continue;
            }

            // Gather: this cell's share of the bilinear scatter.
            const float wx = 1.0f - abs(bx - (float)cellX);
            const float wy = 1.0f - abs(by - (float)cellY);
            if (wx <= 0.0f || wy <= 0.0f) {
                continue;
            }
            const float wxy = wx * wy * value;

            int ba = int(floor(bin));
            int bb = int(ceil(bin));
            const float bMax = bin - floor(bin);
            const float bMin = 1 - bMax;
            if (ba < 0) {
                ba += bins;
            }
            if (ba >= bins) {
                ba -= bins;
            }
            if (bb < 0) {
                bb += bins;
            }
            if (bb >= bins) {
                bb -= bins;
            }
            cell[ba] += wxy * bMin;
            cell[bb] += wxy * bMax;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalization touches only 128 values; run it serially.
    if (tid != 0) {
        return;
    }
    device SIFTDescriptorResult & result = results[groupId];
    result.keypoint = input.keypoint;
    result.theta = input.theta;

    if (!normalizeFeatures(featureCount, features)) {
        result.valid = 0;
        return;
    }
    thresholdFeatures(featureCount, features, 0.2);
    if (!normalizeFeatures(featureCount, features)) {
        result.valid = 0;
        return;
    }
    for (int i = 0; i < featureCount; i++) {
        result.features[i] = features[i];
    }
    result.valid = 1;
}
