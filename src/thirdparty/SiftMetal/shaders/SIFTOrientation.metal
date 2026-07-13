//
//  SIFTOrientation.metal
//  SkyLight
//
//  Created by Luke Van In on 2023/01/07.
//

#include <metal_stdlib>

#include "Common.hpp"
#include "../include/SIFTOrientation.h"

using namespace metal;


static float orientationFromBin(float bin) {
    const int n = SIFT_ORIENTATION_HISTOGRAM_BINS;
    float t = bin / (float)n;
    float tau = 2 * M_PI_F;
    float orientation = t * tau;
    if (orientation < 0) {
        orientation += tau;
    }
    if (orientation >= tau) {
        orientation -= tau;
    }
    return orientation;
}


static float interpolatePeak(float h1, float h2, float h3) {
    return (h1 - h3) / (2 * (h1 + h3 - 2 * h2));
}


static void getPrincipalOrientations(
    threadgroup float * histogram,
    float orientationThreshold,
    thread int & orientationsCount,
    device float * orientations
) {
    const int bins = SIFT_ORIENTATION_HISTOGRAM_BINS;

    float maximum = INT_MIN;
    for (int i = 0; i < bins; i++) {
        maximum = max(maximum, histogram[i]);
    }

    const float threshold = orientationThreshold * maximum;

    orientationsCount = 0;

    for (int i = 0; i < bins; i++) {
        float hm = histogram[((i - 1) + bins) % bins];
        float h0 = histogram[i];
        float hp = histogram[(i + 1) % bins];
        if ((h0 > threshold) && (h0 > hm) && (h0 > hp)) {
            float offset = interpolatePeak(hm, h0, hp);
            float orientation = orientationFromBin((float)i + offset);
            if (isfinite(orientation) && orientationsCount < bins) {
                orientations[orientationsCount] = orientation;
                orientationsCount += 1;
            }
        }
    }
}


static void smoothHistogram(
    threadgroup float * histogram,
    int iterations
) {
    const int n = SIFT_ORIENTATION_HISTOGRAM_BINS;
    float temp[n];
    for (int j = 0; j < iterations; j++) {
        for (int i = 0; i < n; i++) {
            temp[i] = histogram[i];
        }
        for (int i = 0; i < n; i++) {
            float h0 = temp[((i - 1) + n) % n];
            float h1 = temp[i];
            float h2 = temp[(i + 1) % n];
            float v = (h0 + h1 + h2) / 3.0;
            histogram[i] = v;
        }
    }
}


// One threadgroup per keypoint. Threads accumulate disjoint subsets of the
// sampling window into private slices of threadgroup memory (no atomics),
// which are then reduced into a shared histogram. This replaces the previous
// one-thread-per-keypoint design whose per-thread histogram arrays spilled
// to scratch memory and whose sampling loop ran serially.
kernel void siftOrientation(
    device SIFTOrientationResult * results [[buffer(0)]],
    device SIFTOrientationKeypoint * keypoints [[buffer(1)]],
    constant SIFTOrientationParameters & parameters [[buffer(2)]],
    texture2d_array<float, access::read> gaussianTextures [[texture(0)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const int bins = SIFT_ORIENTATION_HISTOGRAM_BINS;
    const int numThreads = SIFT_ORIENTATION_THREADS;

    threadgroup float partials[SIFT_ORIENTATION_THREADS]
                              [SIFT_ORIENTATION_HISTOGRAM_BINS];
    threadgroup float histogram[SIFT_ORIENTATION_HISTOGRAM_BINS];

    const SIFTOrientationKeypoint keypoint = keypoints[groupId];

    for (int b = 0; b < bins; b++) {
        partials[tid][b] = 0;
    }

    const int x = round(keypoint.absoluteX / parameters.delta);
    const int y = round(keypoint.absoluteY / parameters.delta);
    const float sigma = keypoint.sigma / parameters.delta;
    const float lambda = parameters.lambda;
    const float exponentDenominator = 2.0 * lambda * lambda;

    // Window radius: match SiftGPU's SAMPLE_WF(2.0) * GAUSSIAN_WF(1.5)
    const int r = ceil(2 * lambda * sigma);
    const int side = 2 * r + 1;
    const int sampleCount = side * side;

    for (int idx = int(tid); idx < sampleCount; idx += numThreads) {
        const int i = (idx % side) - r;
        const int j = (idx / side) - r;

        // Gaussian weighting
        float u = (float)i / sigma;
        float v = (float)j / sigma;
        float r2 = u * u + v * v;
        float w = exp(-r2 / exponentDenominator);

        // Gradient orientation, computed on the fly from the
        // Gaussian scale-space image.
        float2 gradient = siftGradientAt(
            gaussianTextures, x + i, y + j, uint(keypoint.scale));
        float orientation = gradient.x;
        float magnitude = gradient.y;

        // Add to histogram
        float t = orientation / (2 * M_PI_F);
        int bin = round(t * (float)bins);
        if (bin < 0) {
            bin += bins;
        }
        if (bin >= bins) {
            bin -= bins;
        }

        partials[tid][bin] += w * magnitude;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce the per-thread partials into the shared histogram.
    for (int b = int(tid); b < bins; b += numThreads) {
        float sum = 0;
        for (int t = 0; t < numThreads; t++) {
            sum += partials[t][b];
        }
        histogram[b] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Smoothing and peak extraction touch only 36 bins; run them serially.
    if (tid != 0) {
        return;
    }
    smoothHistogram(histogram, 2);
    device SIFTOrientationResult & result = results[groupId];
    result.keypoint = keypoint.index;
    int count = 0;
    getPrincipalOrientations(
        histogram,
        parameters.orientationThreshold,
        count,
        result.orientations
    );
    result.count = count;
}
