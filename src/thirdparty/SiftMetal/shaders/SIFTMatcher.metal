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
  if (gid >= params.numDescriptors1) {
    return;
  }

  int bestIdx = -1;
  float bestScore =
      params.distanceType == SIFT_MATCHER_DISTANCE_L2 ? FLT_MAX : 0.0f;
  float secondBestScore =
      params.distanceType == SIFT_MATCHER_DISTANCE_L2 ? FLT_MAX : 0.0f;

  const uint desc1Offset = gid * SIFT_MATCHER_DESCRIPTOR_DIM;

  if (params.distanceType == SIFT_MATCHER_DISTANCE_DOT_PRODUCT &&
      params.guidedGeometry == SIFT_MATCHER_GUIDED_NONE) {
    for (uint idx2 = 0; idx2 < params.numDescriptors2; ++idx2) {
      const uint desc2Offset = idx2 * SIFT_MATCHER_DESCRIPTOR_DIM;
      int score = 0;
      for (uint k = 0; k < SIFT_MATCHER_DESCRIPTOR_DIM; ++k) {
        const int v1 = int(descriptors1[desc1Offset + k]);
        const int v2 = int(descriptors2[desc2Offset + k]);
        score += v1 * v2;
      }

      const float scoref = float(score);
      if (scoref > bestScore) {
        bestIdx = int(idx2);
        secondBestScore = bestScore;
        bestScore = scoref;
      } else if (scoref > secondBestScore) {
        secondBestScore = scoref;
      }
    }
  } else {
    for (uint idx2 = 0; idx2 < params.numDescriptors2; ++idx2) {
      if (RejectByGuidedGeometry(params, keypoints1, keypoints2, gid, idx2)) {
        continue;
      }

      int score = 0;
      const uint desc2Offset = idx2 * SIFT_MATCHER_DESCRIPTOR_DIM;
      for (uint k = 0; k < SIFT_MATCHER_DESCRIPTOR_DIM; ++k) {
        const int v1 = int(descriptors1[desc1Offset + k]);
        const int v2 = int(descriptors2[desc2Offset + k]);
        if (params.distanceType == SIFT_MATCHER_DISTANCE_L2) {
          const int diff = v1 - v2;
          score += diff * diff;
        } else {
          score += v1 * v2;
        }
      }

      const float scoref = float(score);
      if (params.distanceType == SIFT_MATCHER_DISTANCE_L2) {
        if (scoref < bestScore) {
          bestIdx = int(idx2);
          secondBestScore = bestScore;
          bestScore = scoref;
        } else if (scoref < secondBestScore) {
          secondBestScore = scoref;
        }
      } else {
        if (scoref > bestScore) {
          bestIdx = int(idx2);
          secondBestScore = bestScore;
          bestScore = scoref;
        } else if (scoref > secondBestScore) {
          secondBestScore = scoref;
        }
      }
    }
  }

  bool accepted = bestIdx >= 0;
  if (accepted && params.distanceType == SIFT_MATCHER_DISTANCE_L2) {
    accepted = bestScore <= params.maxL2Distance &&
               bestScore < params.maxRatioSquared * secondBestScore;
  } else if (accepted) {
    accepted = AcceptDotProductMatch(bestScore, secondBestScore, params);
  }

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

  if (gid >= params.numDescriptors1) {
    return;
  }

  threadgroup float bestScores[SIFT_MATCHER_DOT_THREADS];
  threadgroup float secondBestScores[SIFT_MATCHER_DOT_THREADS];
  threadgroup int bestIndices[SIFT_MATCHER_DOT_THREADS];
  threadgroup int secondBestIndices[SIFT_MATCHER_DOT_THREADS];
  threadgroup uchar descriptor1[SIFT_MATCHER_DESCRIPTOR_DIM];

  float localBestScore = 0.0f;
  float localSecondBestScore = 0.0f;
  int localBestIdx = -1;
  int localSecondBestIdx = -1;

  const uint desc1Offset = gid * SIFT_MATCHER_DESCRIPTOR_DIM;
  for (uint k = tid; k < SIFT_MATCHER_DESCRIPTOR_DIM;
       k += threadsPerThreadgroup) {
    descriptor1[k] = descriptors1[desc1Offset + k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx2 = tid; idx2 < params.numDescriptors2;
       idx2 += threadsPerThreadgroup) {
    const uint desc2Offset = idx2 * SIFT_MATCHER_DESCRIPTOR_DIM;
    int score = 0;
    for (uint k = 0; k < SIFT_MATCHER_DESCRIPTOR_DIM; ++k) {
      score += int(descriptor1[k]) * int(descriptors2[desc2Offset + k]);
    }

    const float scoref = float(score);
    if (scoref > localBestScore) {
      localSecondBestIdx = localBestIdx;
      localSecondBestScore = localBestScore;
      localBestIdx = int(idx2);
      localBestScore = scoref;
    } else if (scoref > localSecondBestScore) {
      localSecondBestIdx = int(idx2);
      localSecondBestScore = scoref;
    }
  }

  bestScores[tid] = localBestScore;
  secondBestScores[tid] = localSecondBestScore;
  bestIndices[tid] = localBestIdx;
  secondBestIndices[tid] = localSecondBestIdx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid != 0) {
    return;
  }

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

  results[gid].index = accepted ? bestIdx : -1;
  results[gid].bestScore = bestScore;
  results[gid].secondBestScore = secondBestScore;
}
