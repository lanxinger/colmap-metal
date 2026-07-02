// SIFTMatcher.h - shared Metal/C++ structs for SIFT descriptor matching.

#ifndef SIFTMatcher_h
#define SIFTMatcher_h

#include <simd/simd.h>

#ifndef __METAL_VERSION__
#include <stdint.h>
#endif

#define SIFT_MATCHER_DESCRIPTOR_DIM 128
#define SIFT_MATCHER_DOT_THREADS 256

#define SIFT_MATCHER_DISTANCE_DOT_PRODUCT 0
#define SIFT_MATCHER_DISTANCE_L2 1

#define SIFT_MATCHER_GUIDED_NONE 0
#define SIFT_MATCHER_GUIDED_EPIPOLAR 1
#define SIFT_MATCHER_GUIDED_HOMOGRAPHY 2

struct SIFTMatcherKeypoint {
  float x;
  float y;
};

struct SIFTMatcherParameters {
  uint32_t numDescriptors1;
  uint32_t numDescriptors2;
  float maxRatio;
  float maxDistance;
  float maxL2Distance;
  float maxRatioSquared;
  float maxResidual;
  int32_t distanceType;
  int32_t guidedGeometry;
  int32_t reverseGuided;
  int32_t _padding;
  float matrix[9];
};

struct SIFTMatcherResult {
  int32_t index;
  float bestScore;
  float secondBestScore;
};

#endif /* SIFTMatcher_h */
