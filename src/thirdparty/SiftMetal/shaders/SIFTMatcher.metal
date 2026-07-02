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
  for (uint idx2 = 0; idx2 < params.numDescriptors2; ++idx2) {
    int score = int(kSqSiftDescriptorNorm);
    if (!RejectByGuidedGeometry(params, keypoints1, keypoints2, gid, idx2)) {
      const uint desc2Offset = idx2 * SIFT_MATCHER_DESCRIPTOR_DIM;
      score = 0;
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

  bool accepted = bestIdx >= 0;
  if (accepted && params.distanceType == SIFT_MATCHER_DISTANCE_L2) {
    accepted = bestScore <= params.maxL2Distance &&
               bestScore < params.maxRatioSquared * secondBestScore;
  } else if (accepted) {
    const float bestDist =
        acos(min(bestScore * kInvSqSiftDescriptorNorm, 1.0f));
    const float secondBestDist =
        acos(min(secondBestScore * kInvSqSiftDescriptorNorm, 1.0f));
    accepted = bestDist <= params.maxDistance &&
               bestDist < params.maxRatio * secondBestDist;
  }

  results[gid].index = accepted ? bestIdx : -1;
  results[gid].bestScore = bestScore;
  results[gid].secondBestScore = secondBestScore;
}
