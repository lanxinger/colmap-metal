// SIFTMatcher.metal - brute-force SIFT descriptor matching.

#include <metal_stdlib>

#include "../include/SIFTMatcher.h"

using namespace metal;

constant float kSqSiftDescriptorNorm = 512.0f * 512.0f;
constant float kInvSqSiftDescriptorNorm = 1.0f / kSqSiftDescriptorNorm;

static float3 Transform3x3(constant SIFTMatcherParameters& params,
                           float x,
                           float y) {
  return float3(params.matrix[0] * x + params.matrix[1] * y +
                    params.matrix[2],
                params.matrix[3] * x + params.matrix[4] * y +
                    params.matrix[5],
                params.matrix[6] * x + params.matrix[7] * y +
                    params.matrix[8]);
}

static bool RejectByGuidedGeometry(
    constant SIFTMatcherParameters& params,
    const device SIFTMatcherKeypoint* keypoints1,
    const device SIFTMatcherKeypoint* keypoints2,
    uint idx1,
    uint idx2) {
  if (params.guidedGeometry == SIFT_MATCHER_GUIDED_NONE) {
    return false;
  }

  const SIFTMatcherKeypoint kp1 =
      params.reverseGuided ? keypoints2[idx2] : keypoints1[idx1];
  const SIFTMatcherKeypoint kp2 =
      params.reverseGuided ? keypoints1[idx1] : keypoints2[idx2];

  if (params.guidedGeometry == SIFT_MATCHER_GUIDED_EPIPOLAR) {
    const float3 p2 = float3(kp2.x, kp2.y, 1.0f);

    const float3 line1 = Transform3x3(params, kp1.x, kp1.y);
    const float3 line2 =
        float3(params.matrix[0] * p2.x + params.matrix[3] * p2.y +
                   params.matrix[6],
               params.matrix[1] * p2.x + params.matrix[4] * p2.y +
                   params.matrix[7],
               params.matrix[2] * p2.x + params.matrix[5] * p2.y +
                   params.matrix[8]);
    const float nom = dot(p2, line1);
    const float denomSq = line1.x * line1.x + line1.y * line1.y +
                          line2.x * line2.x + line2.y * line2.y;
    return nom * nom > params.maxResidual * denomSq;
  }

  if (params.guidedGeometry == SIFT_MATCHER_GUIDED_HOMOGRAPHY) {
    const float3 projected = Transform3x3(params, kp1.x, kp1.y);
    if (abs(projected.z) <= 1e-12f) {
      return true;
    }
    const float2 reproj = projected.xy / projected.z;
    const float2 delta = reproj - float2(kp2.x, kp2.y);
    return dot(delta, delta) > params.maxResidual;
  }

  return true;
}

static bool AcceptDotProductMatch(float bestScore,
                                  float secondBestScore,
                                  constant SIFTMatcherParameters& params) {
  const float bestDist =
      acos(min(bestScore * kInvSqSiftDescriptorNorm, 1.0f));
  if (bestDist > params.maxDistance) {
    return false;
  }

  const float secondBestDist =
      acos(min(secondBestScore * kInvSqSiftDescriptorNorm, 1.0f));
  return bestDist < params.maxRatio * secondBestDist;
}

kernel void siftMatchBest(
    const device uchar* descriptors1 [[buffer(0)]],
    const device uchar* descriptors2 [[buffer(1)]],
    const device SIFTMatcherKeypoint* keypoints1 [[buffer(2)]],
    const device SIFTMatcherKeypoint* keypoints2 [[buffer(3)]],
    constant SIFTMatcherParameters& params [[buffer(4)]],
    device SIFTMatcherResult* results [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  // The host dispatches this serial kernel only for guided matching. Unguided
  // dot-product matching uses siftMatchBestDotParallel.
  if (gid >= params.numDescriptors1) {
    return;
  }

  int bestIdx = -1;
  float bestScore = FLT_MAX;
  float secondBestScore = FLT_MAX;

  // Cache this thread's descriptor once, as 4-byte vectors.
  const device uchar4* desc1 = (const device uchar4*)(
      descriptors1 + gid * SIFT_MATCHER_DESCRIPTOR_DIM);
  uchar4 d1[SIFT_MATCHER_DESCRIPTOR_DIM / 4];
  for (uint k = 0; k < SIFT_MATCHER_DESCRIPTOR_DIM / 4; ++k) {
    d1[k] = desc1[k];
  }

  for (uint idx2 = 0; idx2 < params.numDescriptors2; ++idx2) {
    if (RejectByGuidedGeometry(params, keypoints1, keypoints2, gid, idx2)) {
      continue;
    }

    const device uchar4* desc2 = (const device uchar4*)(
        descriptors2 + idx2 * SIFT_MATCHER_DESCRIPTOR_DIM);
    int4 acc = int4(0);
    for (uint k = 0; k < SIFT_MATCHER_DESCRIPTOR_DIM / 4; ++k) {
      const int4 diff = int4(d1[k]) - int4(desc2[k]);
      acc += diff * diff;
    }
    const int score = acc.x + acc.y + acc.z + acc.w;

    const float scoref = float(score);
    if (scoref < bestScore) {
      bestIdx = int(idx2);
      secondBestScore = bestScore;
      bestScore = scoref;
    } else if (scoref < secondBestScore) {
      secondBestScore = scoref;
    }
  }

  const bool accepted = bestIdx >= 0 && bestScore <= params.maxL2Distance &&
                        bestScore < params.maxRatioSquared * secondBestScore;

  results[gid].index = accepted ? bestIdx : -1;
  results[gid].bestScore = bestScore;
  results[gid].secondBestScore = secondBestScore;
}

kernel void siftMatchBestDotParallel(
    const device uchar* descriptors1 [[buffer(0)]],
    const device uchar* descriptors2 [[buffer(1)]],
    const device SIFTMatcherKeypoint* keypoints1 [[buffer(2)]],
    const device SIFTMatcherKeypoint* keypoints2 [[buffer(3)]],
    constant SIFTMatcherParameters& params [[buffer(4)]],
    device SIFTMatcherResult* results [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadsPerThreadgroup [[threads_per_threadgroup]]) {
  (void)keypoints1;
  (void)keypoints2;

  // Each threadgroup matches a block of query descriptors so every byte of
  // descriptors2 streamed from device memory is reused across the block
  // instead of being re-read once per query.
  const uint block = SIFT_MATCHER_DOT_BLOCK;
  const uint vecPerDesc = SIFT_MATCHER_DESCRIPTOR_DIM / 4;
  const uint rowBase = gid * block;
  if (rowBase >= params.numDescriptors1) {
    return;
  }
  const uint rowCount = min(block, params.numDescriptors1 - rowBase);

  threadgroup float bestScores[SIFT_MATCHER_DOT_THREADS];
  threadgroup float secondBestScores[SIFT_MATCHER_DOT_THREADS];
  threadgroup int bestIndices[SIFT_MATCHER_DOT_THREADS];
  threadgroup int secondBestIndices[SIFT_MATCHER_DOT_THREADS];
  threadgroup uchar4 descriptorBlock[SIFT_MATCHER_DOT_BLOCK]
                                    [SIFT_MATCHER_DESCRIPTOR_DIM / 4];

  // Cooperative load of the query block; rows past the end are zeroed and
  // their scores discarded below.
  for (uint k = tid; k < block * vecPerDesc; k += threadsPerThreadgroup) {
    const uint row = k / vecPerDesc;
    const uint vec = k % vecPerDesc;
    uchar4 value = uchar4(0);
    if (row < rowCount) {
      value = ((const device uchar4*)(
          descriptors1 + (rowBase + row) * SIFT_MATCHER_DESCRIPTOR_DIM))[vec];
    }
    descriptorBlock[row][vec] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float localBestScore[SIFT_MATCHER_DOT_BLOCK];
  float localSecondBestScore[SIFT_MATCHER_DOT_BLOCK];
  int localBestIdx[SIFT_MATCHER_DOT_BLOCK];
  int localSecondBestIdx[SIFT_MATCHER_DOT_BLOCK];
  for (uint b = 0; b < block; ++b) {
    localBestScore[b] = 0.0f;
    localSecondBestScore[b] = 0.0f;
    localBestIdx[b] = -1;
    localSecondBestIdx[b] = -1;
  }

  for (uint idx2 = tid; idx2 < params.numDescriptors2;
       idx2 += threadsPerThreadgroup) {
    const device uchar4* desc2 = (const device uchar4*)(
        descriptors2 + idx2 * SIFT_MATCHER_DESCRIPTOR_DIM);
    int4 acc[SIFT_MATCHER_DOT_BLOCK];
    for (uint b = 0; b < block; ++b) {
      acc[b] = int4(0);
    }
    for (uint k = 0; k < vecPerDesc; ++k) {
      const int4 c2 = int4(desc2[k]);
      for (uint b = 0; b < block; ++b) {
        acc[b] += int4(descriptorBlock[b][k]) * c2;
      }
    }
    for (uint b = 0; b < block; ++b) {
      const float scoref = float(acc[b].x + acc[b].y + acc[b].z + acc[b].w);
      if (scoref > localBestScore[b]) {
        localSecondBestIdx[b] = localBestIdx[b];
        localSecondBestScore[b] = localBestScore[b];
        localBestIdx[b] = int(idx2);
        localBestScore[b] = scoref;
      } else if (scoref > localSecondBestScore[b]) {
        localSecondBestIdx[b] = int(idx2);
        localSecondBestScore[b] = scoref;
      }
    }
  }

  // Reduce and emit one row at a time, reusing the shared scratch arrays.
  for (uint b = 0; b < rowCount; ++b) {
    bestScores[tid] = localBestScore[b];
    secondBestScores[tid] = localSecondBestScore[b];
    bestIndices[tid] = localBestIdx[b];
    secondBestIndices[tid] = localSecondBestIdx[b];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
      int bestIdx = -1;
      int secondBestIdx = -1;
      float bestScore = 0.0f;
      float secondBestScore = 0.0f;

      for (uint i = 0; i < threadsPerThreadgroup; ++i) {
        const float candidates[2] = {bestScores[i], secondBestScores[i]};
        const int candidateIndices[2] = {bestIndices[i], secondBestIndices[i]};
        for (uint candidate = 0; candidate < 2; ++candidate) {
          const float score = candidates[candidate];
          const int index = candidateIndices[candidate];
          if (index < 0) {
            continue;
          }
          if (score > bestScore ||
              (score == bestScore && (bestIdx < 0 || index < bestIdx))) {
            secondBestIdx = bestIdx;
            secondBestScore = bestScore;
            bestIdx = index;
            bestScore = score;
          } else if (index != bestIdx &&
                     (score > secondBestScore ||
                      (score == secondBestScore &&
                       (secondBestIdx < 0 || index < secondBestIdx)))) {
            secondBestIdx = index;
            secondBestScore = score;
          }
        }
      }

      bool accepted = bestIdx >= 0;
      if (accepted) {
        accepted = AcceptDotProductMatch(bestScore, secondBestScore, params);
      }

      results[rowBase + b].index = accepted ? bestIdx : -1;
      results[rowBase + b].bestScore = bestScore;
      results[rowBase + b].secondBestScore = secondBestScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
