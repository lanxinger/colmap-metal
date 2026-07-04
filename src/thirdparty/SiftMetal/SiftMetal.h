// SiftMetal.h - Metal-accelerated SIFT feature extraction for macOS.
// Ported from SIFTMetal (Swift) by Luke Van In, adapted for COLMAP integration.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace sift_metal {

struct Keypoint {
  float x;           // Absolute x coordinate in original image
  float y;           // Absolute y coordinate in original image
  float sigma;       // Scale (blur level)
  float orientation; // Dominant orientation in radians
};

struct ExtractResult {
  std::vector<Keypoint> keypoints;
  // 128-dimensional descriptors, one row per keypoint.
  // Values are float (pre-normalization), suitable for COLMAP's
  // L1_ROOT or L2 normalization pipeline.
  std::vector<float> descriptors; // size = keypoints.size() * 128
};

struct Options {
  // Number of octaves. -1 = auto (based on image size).
  int num_octaves = -1;
  // Number of scales per octave.
  int scales_per_octave = 3;
  // First octave index. -1 means 2x upscaling of input.
  int first_octave = -1;
  // Peak threshold for DoG detection.
  float peak_threshold = 0.0133f;
  // Edge threshold (ratio of principal curvatures).
  float edge_threshold = 10.0f;
  // Maximum number of features to retain. Must be positive.
  int max_num_features = 8192;
  // Maximum number of orientations per keypoint.
  int max_num_orientations = 2;
  // Fix orientation to 0 for upright features.
  bool upright = false;
};

struct MatchKeypoint {
  float x;
  float y;
};

enum class MatchGuidedGeometry {
  NONE = 0,
  EPIPOLAR = 1,
  HOMOGRAPHY = 2,
};

struct MatchOptions {
  float max_ratio = 0.8f;
  float max_distance = 0.7f;
  bool cross_check = true;
};

struct MatchResult {
  uint32_t index1;
  uint32_t index2;
};

// Opaque implementation handle.
class SiftMetalExtractorImpl;
class SiftMetalMatcherImpl;

class SiftMetalExtractor {
 public:
  SiftMetalExtractor();
  ~SiftMetalExtractor();

  // Initialize the Metal pipeline. Returns false if Metal is unavailable.
  bool Init(const Options& options, int max_image_width, int max_image_height);

  // Extract SIFT features from a grayscale image.
  // data: row-major uint8 grayscale pixels
  // width, height: image dimensions
  bool Extract(const uint8_t* data, int width, int height,
               ExtractResult* result);

 private:
  std::unique_ptr<SiftMetalExtractorImpl> impl_;
};

class SiftMetalMatcher {
 public:
  SiftMetalMatcher();
  ~SiftMetalMatcher();

  // Initialize the Metal matching pipeline. Returns false if Metal is
  // unavailable.
  bool Init();

  bool Match(const uint8_t* descriptors1, int num_descriptors1,
             const uint8_t* descriptors2, int num_descriptors2,
             const MatchOptions& options, std::vector<MatchResult>* matches);

  // matrix is row-major. For EPIPOLAR, it is E/F. For HOMOGRAPHY, it maps
  // image1 points to image2 points.
  bool MatchGuided(const uint8_t* descriptors1, int num_descriptors1,
                   const MatchKeypoint* keypoints1,
                   const uint8_t* descriptors2, int num_descriptors2,
                   const MatchKeypoint* keypoints2,
                   const MatchOptions& options,
                   MatchGuidedGeometry guided_geometry,
                   const float matrix[9],
                   float max_residual,
                   std::vector<MatchResult>* matches);

 private:
  std::unique_ptr<SiftMetalMatcherImpl> impl_;
};

}  // namespace sift_metal
