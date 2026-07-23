// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <gtest/gtest.h>

#if defined(COLMAP_GUI_ENABLED)
#include <QApplication>
#else
#include "colmap/exe/gui.h"
#endif

#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/math/random.h"
#include "colmap/util/opengl_utils.h"
#if defined(COLMAP_METAL_ENABLED)
#include "thirdparty/SiftMetal/SiftMetal.h"
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <tuple>

namespace colmap {
namespace {

Bitmap CreateImageWithSquare(const int size) {
  Bitmap bitmap(size, size, false);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = size / 2 - size / 8; r < size / 2 + size / 8; ++r) {
    for (int c = size / 2 - size / 8; c < size / 2 + size / 8; ++c) {
      bitmap.SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
  return bitmap;
}

// Helper to create empty descriptors for testing.
FeatureDescriptors CreateEmptyDescriptors() {
  return FeatureDescriptors(FeatureExtractorType::SIFT,
                            FeatureDescriptorsData(0, 128));
}

// Helper to create reversed descriptors for testing matcher symmetry.
FeatureDescriptors CreateReversedDescriptors(const FeatureDescriptors& src) {
  return FeatureDescriptors(src.type, src.data.colwise().reverse());
}

void ValidateKeypoints(const FeatureKeypoints& keypoints,
                       const Bitmap& bitmap) {
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }
}

void ValidateDescriptorNorms(const FeatureDescriptors& descriptors,
                             const float max_norm_error = 1.0f) {
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512),
              max_norm_error);
  }
}

void ExpectReversedMatches(const FeatureMatches& matches) {
  EXPECT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].point2D_idx1, 0);
  EXPECT_EQ(matches[0].point2D_idx2, 1);
  EXPECT_EQ(matches[1].point2D_idx1, 1);
  EXPECT_EQ(matches[1].point2D_idx2, 0);
}

void ExpectIdentityMatches(const FeatureMatches& matches) {
  EXPECT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].point2D_idx1, 0);
  EXPECT_EQ(matches[0].point2D_idx2, 0);
  EXPECT_EQ(matches[1].point2D_idx1, 1);
  EXPECT_EQ(matches[1].point2D_idx2, 1);
}

void ExpectReversedInlierMatches(const TwoViewGeometry& two_view_geometry) {
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 2);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 0);
}

TwoViewGeometry CreatePlanarTwoViewGeometry() {
  TwoViewGeometry tvg;
  tvg.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  tvg.H = Eigen::Matrix3d::Identity();
  return tvg;
}

void RunGpuTest(const std::function<void()>& test_body) {
#if defined(COLMAP_METAL_ENABLED) || defined(COLMAP_CUDA_ENABLED)
  test_body();
#elif defined(COLMAP_GPU_ENABLED) && defined(COLMAP_GUI_ENABLED)
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   public:
    std::function<void()> body;

   private:
    void Run() {
      opengl_context_.MakeCurrent();
      body();
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  thread.body = std::move(test_body);
  RunThreadWithOpenGLContext(&thread);
#else
  GTEST_SKIP() << "No supported GPU backend is enabled for this test build.";
#endif
}

struct SiftCpuExtractionParams {
  std::string name;
  bool estimate_affine_shape;
  bool domain_size_pooling;
  bool force_covariant_extractor;
  bool upright;
  size_t expected_keypoints;
};

class SiftCpuExtractionTest
    : public ::testing::TestWithParam<SiftCpuExtractionParams> {};

TEST_P(SiftCpuExtractionTest, Nominal) {
  const auto& p = GetParam();
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = p.estimate_affine_shape;
  options.sift->domain_size_pooling = p.domain_size_pooling;
  options.sift->force_covariant_extractor = p.force_covariant_extractor;
  options.sift->upright = p.upright;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), p.expected_keypoints);
  ValidateKeypoints(keypoints, bitmap);
  EXPECT_EQ(descriptors.data.rows(), p.expected_keypoints);
  ValidateDescriptorNorms(descriptors);
}

INSTANTIATE_TEST_SUITE_P(
    SiftCpuExtraction,
    SiftCpuExtractionTest,
    ::testing::Values(
        SiftCpuExtractionParams{"Sift", false, false, false, false, 22},
        SiftCpuExtractionParams{"CovariantSift", false, false, true, false, 22},
        SiftCpuExtractionParams{
            "CovariantAffineSift", true, false, false, false, 22},
        SiftCpuExtractionParams{
            "CovariantAffineSiftUpright", true, false, false, true, 10},
        SiftCpuExtractionParams{
            "CovariantDSPSift", false, true, false, false, 22},
        SiftCpuExtractionParams{
            "CovariantAffineDSPSift", true, true, false, false, 22}),
    [](const auto& info) { return info.param.name; });

TEST(ExtractSiftFeaturesGPU, Nominal) {
  RunGpuTest([] {
    const Bitmap bitmap = CreateImageWithSquare(256);

    FeatureExtractionOptions options(FeatureExtractorType::SIFT);
    options.use_gpu = true;
    options.sift->estimate_affine_shape = false;
    options.sift->domain_size_pooling = false;
    options.sift->force_covariant_extractor = false;
    auto extractor = CreateSiftFeatureExtractor(options);

    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
    EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

    EXPECT_GE(keypoints.size(), 12);
    ValidateKeypoints(keypoints, bitmap);
    EXPECT_GE(descriptors.data.rows(), 12);
#if defined(COLMAP_METAL_ENABLED)
    ValidateDescriptorNorms(descriptors, 4.0f);
#else
    ValidateDescriptorNorms(descriptors);
#endif
  });
}

#if defined(COLMAP_METAL_ENABLED)
Bitmap CreateImageWithRectangle(const int width, const int height) {
  Bitmap bitmap(width, height, false);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = height / 4; r < 3 * height / 4; ++r) {
    for (int c = width / 4; c < 3 * width / 4; ++c) {
      bitmap.SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
  return bitmap;
}

float HashValueNoise(const int x, const int y, const uint32_t seed) {
  uint32_t value = static_cast<uint32_t>(x) * 0x9E3779B1u;
  value ^= static_cast<uint32_t>(y) * 0x85EBCA77u;
  value ^= seed;
  value ^= value >> 16;
  value *= 0x7FEB352Du;
  value ^= value >> 15;
  value *= 0x846CA68Bu;
  value ^= value >> 16;
  return static_cast<float>(value) /
             static_cast<float>(std::numeric_limits<uint32_t>::max()) * 2.0f -
         1.0f;
}

float Lerp(const float a, const float b, const float t) {
  return a + (b - a) * t;
}

Bitmap CreateMultiscaleValueNoiseImage(const int width, const int height) {
  struct NoiseLayer {
    int cell_size;
    float amplitude;
  };
  constexpr std::array<NoiseLayer, 6> kLayers = {{{2, 8.0f},
                                                  {4, 12.0f},
                                                  {8, 18.0f},
                                                  {16, 24.0f},
                                                  {32, 32.0f},
                                                  {64, 40.0f}}};

  Bitmap bitmap(width, height, false);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float pixel = 128.0f;
      for (size_t layer_idx = 0; layer_idx < kLayers.size(); ++layer_idx) {
        const auto& layer = kLayers[layer_idx];
        const int x0 = x / layer.cell_size;
        const int y0 = y / layer.cell_size;
        const float tx =
            static_cast<float>(x % layer.cell_size) / layer.cell_size;
        const float ty =
            static_cast<float>(y % layer.cell_size) / layer.cell_size;
        const uint32_t seed =
            0x1234567u + static_cast<uint32_t>(layer_idx) * 0x9E3779B9u;
        const float top = Lerp(
            HashValueNoise(x0, y0, seed), HashValueNoise(x0 + 1, y0, seed), tx);
        const float bottom = Lerp(HashValueNoise(x0, y0 + 1, seed),
                                  HashValueNoise(x0 + 1, y0 + 1, seed),
                                  tx);
        pixel += 0.6f * layer.amplitude * Lerp(top, bottom, ty);
      }
      bitmap.SetPixel(y,
                      x,
                      BitmapColor<uint8_t>(static_cast<uint8_t>(
                          std::clamp(static_cast<int>(pixel), 0, 255))));
    }
  }
  return bitmap;
}

void ExpectEquivalentMetalExtraction(
    const sift_metal::ExtractResult& actual,
    const sift_metal::ExtractResult& expected) {
  ASSERT_EQ(actual.keypoints.size(), expected.keypoints.size());
  ASSERT_EQ(actual.descriptors.size(), expected.descriptors.size());

  auto SortedIndices = [](const sift_metal::ExtractResult& result) {
    std::vector<size_t> indices(result.keypoints.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
      const auto& a = result.keypoints[lhs];
      const auto& b = result.keypoints[rhs];
      return std::tie(a.x, a.y, a.sigma, a.orientation) <
             std::tie(b.x, b.y, b.sigma, b.orientation);
    });
    return indices;
  };

  const std::vector<size_t> actual_indices = SortedIndices(actual);
  const std::vector<size_t> expected_indices = SortedIndices(expected);
  for (size_t i = 0; i < actual_indices.size(); ++i) {
    const size_t actual_idx = actual_indices[i];
    const size_t expected_idx = expected_indices[i];
    const auto& actual_keypoint = actual.keypoints[actual_idx];
    const auto& expected_keypoint = expected.keypoints[expected_idx];
    EXPECT_FLOAT_EQ(actual_keypoint.x, expected_keypoint.x);
    EXPECT_FLOAT_EQ(actual_keypoint.y, expected_keypoint.y);
    EXPECT_FLOAT_EQ(actual_keypoint.sigma, expected_keypoint.sigma);
    EXPECT_FLOAT_EQ(actual_keypoint.orientation, expected_keypoint.orientation);
    for (size_t j = 0; j < 128; ++j) {
      EXPECT_FLOAT_EQ(actual.descriptors[actual_idx * 128 + j],
                      expected.descriptors[expected_idx * 128 + j]);
    }
  }
}

TEST(ExtractSiftFeaturesMetal, RejectsNonPositiveFeatureLimit) {
  sift_metal::Options options;
  options.max_num_features = 0;

  sift_metal::SiftMetalExtractor extractor;
  EXPECT_FALSE(extractor.Init(options, 256, 256));
}

TEST(ExtractSiftFeaturesMetal, RejectsUnsupportedFirstOctave) {
  sift_metal::Options options;
  options.first_octave = 1;

  sift_metal::SiftMetalExtractor extractor;
  EXPECT_FALSE(extractor.Init(options, 256, 256));
}

TEST(ExtractSiftFeaturesMetal, ClampsExcessiveOctavesToImageSize) {
  sift_metal::Options options;
  options.num_octaves = std::numeric_limits<int>::max();

  sift_metal::SiftMetalExtractor extractor;
  EXPECT_TRUE(extractor.Init(options, 64, 64));
}

TEST(ExtractSiftFeaturesMetal, ClearsResultOnInvalidInput) {
  sift_metal::Options options;
  sift_metal::SiftMetalExtractor extractor;
  ASSERT_TRUE(extractor.Init(options, 256, 256));

  sift_metal::ExtractResult result;
  result.keypoints.push_back({1.0f, 2.0f, 3.0f, 4.0f});
  result.descriptors.resize(128, 1.0f);

  EXPECT_FALSE(extractor.Extract(nullptr, 256, 256, &result));
  EXPECT_TRUE(result.keypoints.empty());
  EXPECT_TRUE(result.descriptors.empty());
}

TEST(ExtractSiftFeaturesMetal, AllowsOrientationExpandedDescriptorLimit) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  sift_metal::Options options;
  options.max_num_features = 1;
  options.max_num_orientations = 2;

  sift_metal::SiftMetalExtractor extractor;
  ASSERT_TRUE(extractor.Init(options, bitmap.Width(), bitmap.Height()));

  sift_metal::ExtractResult result;
  ASSERT_TRUE(extractor.Extract(
      bitmap.RowMajorData().data(), bitmap.Width(), bitmap.Height(), &result));
  EXPECT_GT(result.keypoints.size(), 1);
  EXPECT_LE(result.keypoints.size(), 2);
  EXPECT_EQ(result.descriptors.size(), result.keypoints.size() * 128);
}

TEST(ExtractSiftFeaturesMetal, ReusesTexturesAcrossMixedImageSizes) {
  const std::array<Bitmap, 3> bitmaps = {
      CreateImageWithRectangle(257, 193),
      CreateImageWithRectangle(193, 257),
      CreateImageWithRectangle(321, 181),
  };
  for (const int first_octave : {-1, 0}) {
    sift_metal::Options options;
    options.first_octave = first_octave;
    sift_metal::SiftMetalExtractor reused_extractor;
    ASSERT_TRUE(reused_extractor.Init(options, 1, 1));

    for (const Bitmap& bitmap : bitmaps) {
      sift_metal::ExtractResult reused_result;
      ASSERT_TRUE(reused_extractor.Extract(bitmap.RowMajorData().data(),
                                           bitmap.Width(),
                                           bitmap.Height(),
                                           &reused_result));

      sift_metal::SiftMetalExtractor fresh_extractor;
      ASSERT_TRUE(fresh_extractor.Init(options, bitmap.Width(), bitmap.Height()));
      sift_metal::ExtractResult fresh_result;
      ASSERT_TRUE(fresh_extractor.Extract(bitmap.RowMajorData().data(),
                                          bitmap.Width(),
                                          bitmap.Height(),
                                          &fresh_result));
      ExpectEquivalentMetalExtraction(reused_result, fresh_result);
    }
  }
}

TEST(ExtractSiftFeaturesMetal, RetainsCpuComparableFeatureYield) {
  const Bitmap bitmap = CreateMultiscaleValueNoiseImage(512, 768);

  FeatureExtractionOptions cpu_options(FeatureExtractorType::SIFT);
  cpu_options.use_gpu = false;
  auto cpu_extractor = CreateSiftFeatureExtractor(cpu_options);
  FeatureKeypoints cpu_keypoints;
  FeatureDescriptors cpu_descriptors;
  ASSERT_TRUE(cpu_extractor->Extract(bitmap, &cpu_keypoints, &cpu_descriptors));

  FeatureExtractionOptions metal_options(FeatureExtractorType::SIFT);
  metal_options.use_gpu = true;
  auto metal_extractor = CreateSiftFeatureExtractor(metal_options);
  FeatureKeypoints metal_keypoints;
  FeatureDescriptors metal_descriptors;
  ASSERT_TRUE(
      metal_extractor->Extract(bitmap, &metal_keypoints, &metal_descriptors));

  ASSERT_GT(cpu_keypoints.size(), 500);
  EXPECT_GE(metal_keypoints.size() * 100, cpu_keypoints.size() * 88)
      << "Metal found " << metal_keypoints.size() << " features versus "
      << cpu_keypoints.size() << " on CPU";
}

TEST(MatchSiftFeaturesMetal, ClearsMatchesOnInvalidInput) {
  sift_metal::SiftMetalMatcher matcher;
  ASSERT_TRUE(matcher.Init());

  std::array<uint8_t, 128> descriptor = {};
  sift_metal::MatchOptions options;
  std::vector<sift_metal::MatchResult> matches = {{0, 0}};

  EXPECT_FALSE(
      matcher.Match(nullptr, 1, descriptor.data(), 1, options, &matches));
  EXPECT_TRUE(matches.empty());

  matches = {{0, 0}};
  options.max_ratio = 0.0f;
  EXPECT_FALSE(matcher.Match(
      descriptor.data(), 1, descriptor.data(), 1, options, &matches));
  EXPECT_TRUE(matches.empty());

  matches = {{0, 0}};
  options.max_ratio = 0.8f;
  const std::array<float, 9> identity = {
      1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  EXPECT_FALSE(matcher.MatchGuided(descriptor.data(),
                                   1,
                                   nullptr,
                                   descriptor.data(),
                                   1,
                                   nullptr,
                                   options,
                                   sift_metal::MatchGuidedGeometry::HOMOGRAPHY,
                                   identity.data(),
                                   1.0f,
                                   &matches));
  EXPECT_TRUE(matches.empty());
}
#endif

FeatureDescriptors CreateRandomFeatureDescriptors(const size_t num_features) {
  SetPRNGSeed(0);
  FeatureDescriptorsFloatData descriptors_float =
      FeatureDescriptorsFloatData::Zero(num_features, 128);
  std::vector<int> dims(128);
  std::iota(dims.begin(), dims.end(), 0);
  for (size_t i = 0; i < num_features; ++i) {
    std::shuffle(dims.begin(), dims.end(), *PRNG);
    for (size_t j = 0; j < 10; ++j) {
      descriptors_float(i, dims[j]) = 1.0f;
    }
  }
  L2NormalizeFeatureDescriptors(&descriptors_float);
  return FeatureDescriptors(
      FeatureExtractorType::SIFT,
      FeatureDescriptorsToUnsignedByte(descriptors_float));
}

void CheckEqualMatches(const FeatureMatches& matches1,
                       const FeatureMatches& matches2) {
  ASSERT_EQ(matches1.size(), matches2.size());
  for (size_t i = 0; i < matches1.size(); ++i) {
    EXPECT_EQ(matches1[i].point2D_idx1, matches2[i].point2D_idx1);
    EXPECT_EQ(matches1[i].point2D_idx2, matches2[i].point2D_idx2);
  }
}

TEST(CreateSiftGPUMatcherOpenGL, Nominal) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    EXPECT_NE(CreateSiftFeatureMatcher(options), nullptr);
  });
}

TEST(CreateSiftGPUMatcherCUDA, Nominal) {
#if defined(COLMAP_CUDA_ENABLED)
  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = true;
  options.gpu_index = "0";
  options.max_num_matches = 1000;
  EXPECT_NE(CreateSiftFeatureMatcher(options), nullptr);
#endif
}

struct FeatureDescriptorIndexCacheHelper {
  explicit FeatureDescriptorIndexCacheHelper(
      const std::vector<FeatureMatcher::Image>& images)
      : index_cache(100, [this](const image_t image_id) {
          auto index = FeatureDescriptorIndex::Create();
          const auto& desc = this->image_descriptors_.at(image_id);
          index->Build(desc->ToFloat());
          return index;
        }) {
    for (const auto& image : images) {
      image_descriptors_.emplace(image.image_id, image.descriptors);
    }
  }

  ThreadSafeLRUCache<image_t, FeatureDescriptorIndex> index_cache;

 private:
  std::map<image_t, std::shared_ptr<const FeatureDescriptors>>
      image_descriptors_;
};

TEST(SiftCPUFeatureMatcher, Nominal) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image0, image1, image2});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = false;
  options.sift->cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);

  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);

  matcher->Match(image0, image2, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image1, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image0, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
}

TEST(SiftCPUFeatureMatcher, TypeMismatch) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
  ASSERT_EQ(sift_desc.type, FeatureExtractorType::SIFT);

  FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
  undefined_desc.type = FeatureExtractorType::UNDEFINED;

  const FeatureMatcher::Image image_sift = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(sift_desc)};
  const FeatureMatcher::Image image_undefined = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(undefined_desc)};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image_sift, image_undefined});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  FeatureMatches matches;
  EXPECT_THROW(matcher->Match(image_sift, image_undefined, &matches),
               std::invalid_argument);
}

TEST(MatchGuidedSiftFeaturesCPU, TypeMismatch) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
  ASSERT_EQ(sift_desc.type, FeatureExtractorType::SIFT);

  FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
  undefined_desc.type = FeatureExtractorType::UNDEFINED;

  const FeatureMatcher::Image image_sift = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
      std::make_shared<FeatureDescriptors>(sift_desc)};
  const FeatureMatcher::Image image_undefined = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
      std::make_shared<FeatureDescriptors>(undefined_desc)};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image_sift, image_undefined});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

  EXPECT_THROW(matcher->MatchGuided(
                   1.0, image_sift, image_undefined, &two_view_geometry),
               std::invalid_argument);
}

TEST(MatchSiftFeaturesGPU, TypeMismatch) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

    FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
    FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
    undefined_desc.type = FeatureExtractorType::UNDEFINED;

    const FeatureMatcher::Image image_sift = {
        /*image_id=*/1,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(sift_desc)};
    const FeatureMatcher::Image image_undefined = {
        /*image_id=*/2,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(undefined_desc)};

    FeatureMatches matches;
    EXPECT_THROW(matcher->Match(image_sift, image_undefined, &matches),
                 std::invalid_argument);
  });
}

TEST(MatchGuidedSiftFeaturesGPU, TypeMismatch) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimpleRadial, 100.0, 100, 200);

    FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
    FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
    undefined_desc.type = FeatureExtractorType::UNDEFINED;

    const FeatureMatcher::Image image_sift = {
        /*image_id=*/1,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
        std::make_shared<FeatureDescriptors>(sift_desc)};
    const FeatureMatcher::Image image_undefined = {
        /*image_id=*/2,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(undefined_desc)};

    TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

    EXPECT_THROW(matcher->MatchGuided(
                     1.0, image_sift, image_undefined, &two_view_geometry),
                 std::invalid_argument);
  });
}

TEST(SiftCPUFeatureMatcherFaissVsBruteForce, Nominal) {
  FeatureMatchingOptions match_options;
  match_options.max_num_matches = 1000;

  auto TestFaissVsBruteForce = [](const FeatureMatchingOptions& options,
                                  const FeatureDescriptors& descriptors1,
                                  const FeatureDescriptors& descriptors2) {
    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
    const FeatureMatcher::Image image0 = {
        /*image_id=*/0,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(descriptors1)};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(descriptors2)};

    FeatureDescriptorIndexCacheHelper index_cache_helper(
        {image0, image1, image2});

    FeatureMatches matches_bf;
    FeatureMatches matches_faiss;

    FeatureMatchingOptions custom_options = options;
    custom_options.use_gpu = false;
    custom_options.sift->cpu_brute_force_matcher = true;
    auto bf_matcher = CreateSiftFeatureMatcher(custom_options);
    custom_options.sift->cpu_brute_force_matcher = false;
    custom_options.sift->cpu_descriptor_index_cache =
        &index_cache_helper.index_cache;
    auto faiss_matcher = CreateSiftFeatureMatcher(custom_options);

    bf_matcher->Match(image1, image2, &matches_bf);
    faiss_matcher->Match(image1, image2, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    const size_t num_matches = matches_bf.size();

    bf_matcher->Match(image0, image2, &matches_bf);
    faiss_matcher->Match(image0, image2, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    bf_matcher->Match(image1, image0, &matches_bf);
    faiss_matcher->Match(image1, image0, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    bf_matcher->Match(image0, image0, &matches_bf);
    faiss_matcher->Match(image0, image0, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    return num_matches;
  };

  {
    const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    const FeatureDescriptors descriptors2 = CreateRandomFeatureDescriptors(50);
    FeatureMatchingOptions match_options;
    TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
  }

  {
    const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2;
    descriptors2.data = descriptors1.data.colwise().reverse();
    descriptors2.type = descriptors1.type;
    FeatureMatchingOptions match_options;
    const size_t num_matches =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches, 50);
  }

  // Check the ratio test.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2 = descriptors1;

    FeatureMatchingOptions match_options;
    const size_t num_matches1 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches1, 50);

    descriptors2.data.row(49) = descriptors2.data.row(0);
    descriptors2.data(0, 0) += 50;
    descriptors2.data.row(0) = FeatureDescriptorsToUnsignedByte(
        descriptors2.data.row(0).cast<float>().normalized());
    descriptors2.data(49, 0) += 100;
    descriptors2.data.row(49) = FeatureDescriptorsToUnsignedByte(
        descriptors2.data.row(49).cast<float>().normalized());

    match_options.sift->max_ratio = 0.4;
    FeatureDescriptors descriptors1_top49;
    descriptors1_top49.data = descriptors1.data.topRows(49);
    descriptors1_top49.type = descriptors1.type;
    const size_t num_matches2 =
        TestFaissVsBruteForce(match_options, descriptors1_top49, descriptors2);
    EXPECT_EQ(num_matches2, 48);

    match_options.sift->max_ratio = 0.6;
    const size_t num_matches3 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches3, 49);
  }

  // Check the cross check.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2 = descriptors1;
    descriptors1.data.row(0) = descriptors1.data.row(1);

    FeatureMatchingOptions match_options;

    match_options.sift->cross_check = false;
    const size_t num_matches1 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches1, 50);

    match_options.sift->cross_check = true;
    const size_t num_matches2 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches2, 48);
  }
}

TEST(MatchGuidedSiftFeaturesCPU, Nominal) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(0),
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{100, 0}, {2, 0}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image0, image1, image2, image3});

  TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  constexpr double kMaxError = 1.0;

  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  matcher->MatchGuided(kMaxError, image3, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

  matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
  matcher->MatchGuided(kMaxError, image1, image0, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
  matcher->MatchGuided(kMaxError, image0, image0, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
}

void TestGuidedMatchingWithCameraDistortion(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  // Test guided matching with essential matrix using calibrated cameras.
  // This exercises the code path that uses normalized coordinates.
  // Use the OPENCV model with strong radial and tangential distortion. Its
  // params are fx, fy, cx, cy, k1, k2, p1, p2. The distortion is strong enough
  // that the pixel-coordinate fundamental matrix finds no matches, but must
  // stay invertible over the keypoints used below, which is what bounds how
  // large these coefficients can be; p2 is left at zero for that reason.
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 100.0, 100, 200);
  camera.params[4] = -0.5;  // k1
  camera.params[5] = 0.5;   // k2
  camera.params[6] = -0.5;  // p1

  // Two points on the epipolar line (v=0 in normalized coordinates).
  const Eigen::Vector2f img_point11 =
      camera.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point12 =
      camera.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point21 =
      camera.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point22 =
      camera.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();

  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(0),
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image0, image1, image2});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));
  two_view_geometry.F =
      FundamentalFromEssentialMatrix(camera.CalibrationMatrix(),
                                     two_view_geometry.E.value(),
                                     camera.CalibrationMatrix());

  constexpr double kMaxError = 1.0;

  // With uncalibrated cameras, the fundamental matrix is used with pixel
  // coordinates and no matches are expected to be found due to strong
  // distortion.
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 0);

  // With calibrated cameras, the essential matrix is used with normalized
  // coordinates and matches are expected to be found.
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
}

TEST(MatchGuidedSiftFeaturesCPU, EssentialMatrix) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingWithCameraDistortion(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

void TestGuidedMatchingSharedFocal(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  // An UNCALIBRATED pair carrying solver-estimated intrinsics (camera1/camera2)
  // is guided-matched via the essential matrix in normalized coordinates, using
  // those estimated intrinsics rather than the images' cameras, whose focal
  // length is only a placeholder. Distortion is strong enough that the
  // pixel-coordinate F path finds nothing, and the placeholder focal is wrong
  // enough that normalizing with it finds nothing either, so the test passes
  // only if the estimated camera is the one used.
  //
  // As elsewhere on the E path, E is taken to relate undistorted rays.
  constexpr double kEstimatedFocal = 100.0;
  constexpr double kPlaceholderFocal = 500.0;
  // OPENCV params are fx, fy, cx, cy, k1, k2, p1, p2. Same distortion as in
  // TestGuidedMatchingWithCameraDistortion: strong, but invertible over the
  // keypoints used below.
  Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kOpenCV, kEstimatedFocal, 100, 200);
  camera.params[4] = -0.5;  // k1
  camera.params[5] = 0.5;   // k2
  camera.params[6] = -0.5;  // p1

  // The camera as stored in the database: same model and distortion, but the
  // focal length has not been recovered yet.
  Camera placeholder_camera = camera;
  placeholder_camera.SetFocalLength(kPlaceholderFocal);

  // Two points on the epipolar line (v=0 in normalized coordinates).
  const Eigen::Vector2f img_point11 =
      camera.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point12 =
      camera.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point21 =
      camera.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point22 =
      camera.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&placeholder_camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&placeholder_camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image1, image2});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));
  two_view_geometry.camera1 = camera;
  two_view_geometry.camera2 = camera;
  // F = K^-T E K^-1, as the estimator populates it for this config.
  two_view_geometry.F =
      FundamentalFromEssentialMatrix(camera.CalibrationMatrix(),
                                     two_view_geometry.E.value(),
                                     camera.CalibrationMatrix());

  constexpr double kMaxError = 1.0;

  // Matches are found only by normalizing with the estimated intrinsics. The
  // complementary case, an UNCALIBRATED pair without them falling back to the
  // F path, is covered by TestGuidedMatchingWithCameraDistortion.
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);
}

// The normalizing camera is not a function of the image alone: a shared-focal
// pair carries a focal length estimated per pair, so the same image matched
// against different partners must be renormalized. Guards the GPU matcher's
// feature-location cache, which keys on the image id.
void TestGuidedMatchingSharedFocalPerPairFocal(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  constexpr double kFocalA = 100.0;
  constexpr double kFocalB = 200.0;
  const Camera camera_a = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, kFocalA, 100, 200);
  const Camera camera_b = Camera::CreateFromModelId(
      2, CameraModelId::kSimplePinhole, kFocalB, 100, 200);

  // image1's pixels normalize to y = +-0.1 under camera_a, and to half that,
  // y = +-0.05, under camera_b. The relative pose is a pure x-translation, so a
  // match requires the partner's normalized y to agree.
  const Eigen::Vector2f img_point11 =
      camera_a.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point12 =
      camera_a.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  // Partner for the camera_a pair.
  const Eigen::Vector2f img_point21 =
      camera_a.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point22 =
      camera_a.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();
  // Partner for the camera_b pair.
  const Eigen::Vector2f img_point31 =
      camera_b.ImgFromCam({0.3, -0.05, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point32 =
      camera_b.ImgFromCam({-0.4, 0.05, 1.0}).value().cast<float>();

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera_a,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera_a,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera_a,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point31.x(), img_point31.y()},
                                       {img_point32.x(), img_point32.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image1, image2, image3});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));

  constexpr double kMaxError = 1.0;

  two_view_geometry.camera1 = camera_a;
  two_view_geometry.camera2 = camera_a;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  // Same image1, different estimated focal: stale normalized keypoints from the
  // previous call would put image1 at y = +-0.1 instead of +-0.05, far outside
  // the ~1/f normalized threshold.
  two_view_geometry.camera1 = camera_b;
  two_view_geometry.camera2 = camera_b;
  matcher->MatchGuided(kMaxError, image1, image3, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);
}

TEST(MatchGuidedSiftFeaturesCPU, SharedFocal) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSharedFocal(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchGuidedSiftFeaturesCPU, SharedFocalPerPairFocal) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSharedFocalPerPairFocal(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchSiftFeaturesGPU, Nominal) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
    const FeatureMatcher::Image image0 = {
        /*image_id=*/0,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(
            CreateReversedDescriptors(*image1.descriptors))};

    FeatureMatches matches;

    matcher->Match(image1, image2, &matches);
    ExpectReversedMatches(matches);

    matcher->Match(image1, image2, &matches);
    ExpectReversedMatches(matches);

    matcher->Match(image0, image2, &matches);
    EXPECT_EQ(matches.size(), 0);
    matcher->Match(image1, image0, &matches);
    EXPECT_EQ(matches.size(), 0);
    matcher->Match(image0, image0, &matches);
    EXPECT_EQ(matches.size(), 0);
  });
}

TEST(MatchSiftFeaturesGPU, RefreshesInPlaceDescriptorMutation) {
  RunGpuTest([] {
    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
    auto descriptors1 =
        std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2));
    auto descriptors2 = std::make_shared<FeatureDescriptors>(
        CreateReversedDescriptors(*descriptors1));
    const FeatureMatcher::Image image1 = {/*image_id=*/1,
                                          /*camera=*/&camera,
                                          /*keypoints=*/nullptr,
                                          descriptors1};
    const FeatureMatcher::Image image2 = {/*image_id=*/2,
                                          /*camera=*/&camera,
                                          /*keypoints=*/nullptr,
                                          descriptors2};

    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    FeatureMatches matches;
    matcher->Match(image1, image2, &matches);
    ExpectReversedMatches(matches);

    const uint8_t* descriptor_data = descriptors2->data.data();
    for (Eigen::Index r = 0; r < descriptors2->data.rows(); ++r) {
      for (Eigen::Index c = 0; c < descriptors2->data.cols(); ++c) {
        descriptors2->data(r, c) = descriptors1->data(r, c);
      }
    }
    ASSERT_EQ(descriptors2->data.data(), descriptor_data);

    matcher->Match(image1, image2, &matches);
    ExpectIdentityMatches(matches);
  });
}

TEST(MatchSiftFeaturesCPUvsGPU, Nominal) {
  RunGpuTest([] {
    auto TestCPUvsGPU = [](const FeatureMatchingOptions& options,
                           const FeatureDescriptors& descriptors1,
                           const FeatureDescriptors& descriptors2) {
      const Camera camera = Camera::CreateFromModelId(
          1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
      const FeatureMatcher::Image image0 = {
          /*image_id=*/0,
          /*camera=*/&camera,
          /*keypoints=*/nullptr,
          std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
      const FeatureMatcher::Image image1 = {
          /*image_id=*/1,
          /*camera=*/&camera,
          /*keypoints=*/nullptr,
          std::make_shared<FeatureDescriptors>(descriptors1)};
      const FeatureMatcher::Image image2 = {
          /*image_id=*/2,
          /*camera=*/&camera,
          /*keypoints=*/nullptr,
          std::make_shared<FeatureDescriptors>(descriptors2)};

      FeatureDescriptorIndexCacheHelper index_cache_helper(
          {image0, image1, image2});

      FeatureMatchingOptions custom_options = options;
      custom_options.use_gpu = true;
      custom_options.max_num_matches = 1000;
      auto gpu_matcher =
          THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(custom_options));
      custom_options.use_gpu = false;
      custom_options.sift->cpu_descriptor_index_cache =
          &index_cache_helper.index_cache;
      auto cpu_matcher = CreateSiftFeatureMatcher(custom_options);

      FeatureMatches matches_cpu;
      FeatureMatches matches_gpu;

      cpu_matcher->Match(image1, image2, &matches_cpu);
      gpu_matcher->Match(image1, image2, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      const size_t num_matches = matches_cpu.size();

      cpu_matcher->Match(image0, image2, &matches_cpu);
      gpu_matcher->Match(image0, image2, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      cpu_matcher->Match(image1, image0, &matches_cpu);
      gpu_matcher->Match(image1, image0, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      cpu_matcher->Match(image0, image0, &matches_cpu);
      gpu_matcher->Match(image0, image0, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      return num_matches;
    };

    {
      const FeatureDescriptors descriptors1 =
          CreateRandomFeatureDescriptors(50);
      const FeatureDescriptors descriptors2 =
          CreateRandomFeatureDescriptors(50);
      FeatureMatchingOptions match_options;
      TestCPUvsGPU(match_options, descriptors1, descriptors2);
    }

    {
      const FeatureDescriptors descriptors1 =
          CreateRandomFeatureDescriptors(50);
      FeatureDescriptors descriptors2;
      descriptors2.data = descriptors1.data.colwise().reverse();
      descriptors2.type = descriptors1.type;
      FeatureMatchingOptions match_options;
      const size_t num_matches =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches, 50);
    }

    // Check the ratio test.
    {
      FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
      FeatureDescriptors descriptors2 = descriptors1;

      FeatureMatchingOptions match_options;
      const size_t num_matches1 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches1, 50);

      descriptors2.data.row(49) = descriptors2.data.row(0);
      descriptors2.data(0, 0) += 50;
      descriptors2.data.row(0) = FeatureDescriptorsToUnsignedByte(
          descriptors2.data.row(0).cast<float>().normalized());
      descriptors2.data(49, 0) += 100;
      descriptors2.data.row(49) = FeatureDescriptorsToUnsignedByte(
          descriptors2.data.row(49).cast<float>().normalized());

      match_options.sift->max_ratio = 0.4;
      FeatureDescriptors descriptors1_top49;
      descriptors1_top49.data = descriptors1.data.topRows(49);
      descriptors1_top49.type = descriptors1.type;
      const size_t num_matches2 =
          TestCPUvsGPU(match_options, descriptors1_top49, descriptors2);
      EXPECT_EQ(num_matches2, 48);

      match_options.sift->max_ratio = 0.6;
      const size_t num_matches3 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches3, 49);
    }

    // Check the cross check.
    {
      FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
      FeatureDescriptors descriptors2 = descriptors1;
      descriptors1.data.row(0) = descriptors1.data.row(1);

      FeatureMatchingOptions match_options;

      match_options.sift->cross_check = false;
      const size_t num_matches1 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches1, 50);

      match_options.sift->cross_check = true;
      const size_t num_matches2 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches2, 48);
    }
  });
}

TEST(MatchGuidedSiftFeaturesGPU, Nominal) {
  RunGpuTest([] {
    Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimpleRadial, 100.0, 100, 200);
    const FeatureMatcher::Image image0 = {
        /*image_id=*/0,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(0),
        std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
    auto keypoints1 = std::make_shared<FeatureKeypoints>(
        std::vector<FeatureKeypoint>{{1, 0}, {2, 0}});
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        keypoints1,
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateReversedDescriptors(*image1.descriptors))};
    const FeatureMatcher::Image image3 = {
        /*image_id=*/3,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{100, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};
    const FeatureMatcher::Image image1_updated_keypoints = {
        /*image_id=*/1,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{100, 0}, {1, 0}}),
        image1.descriptors};

    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

    constexpr double kMaxError = 4.0;

    matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
    ExpectReversedInlierMatches(two_view_geometry);

    matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
    ExpectReversedInlierMatches(two_view_geometry);

    (*keypoints1)[0].x = 100;
    matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
    ASSERT_EQ(two_view_geometry.inlier_matches.size(), 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);
    (*keypoints1)[0].x = 1;

    matcher->MatchGuided(
        kMaxError, image1_updated_keypoints, image2, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

    matcher->MatchGuided(kMaxError, image3, image2, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

    matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
    matcher->MatchGuided(kMaxError, image1, image0, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
    matcher->MatchGuided(kMaxError, image0, image0, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
  });
}

#if defined(COLMAP_METAL_ENABLED)
TEST(MatchGuidedSiftFeaturesGPU, RejectsNonFiniteGeometry) {
  RunGpuTest([] {
    Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimpleRadial, 100.0, 100, 200);
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateReversedDescriptors(*image1.descriptors))};

    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();
    (*two_view_geometry.H)(0, 0) = std::numeric_limits<double>::quiet_NaN();
    two_view_geometry.inlier_matches = {FeatureMatch{0, 1}};

    matcher->MatchGuided(4.0, image1, image2, &two_view_geometry);
    EXPECT_TRUE(two_view_geometry.inlier_matches.empty());
  });
}
#endif

TEST(MatchGuidedSiftFeaturesGPU, EssentialMatrix) {
  RunGpuTest([] {
    TestGuidedMatchingWithCameraDistortion(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, RefreshesInPlaceCameraMutation) {
  RunGpuTest([] {
    Camera camera1 = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 200, 200);
    Camera camera2 = camera1;
    camera2.camera_id = 2;
    const auto descriptors1 =
        std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2));
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera1,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{80, 100}, {120, 100}}),
        descriptors1};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera2,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{120, 100}, {80, 100}}),
        std::make_shared<FeatureDescriptors>(
            CreateReversedDescriptors(*descriptors1))};

    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    TwoViewGeometry geometry;
    geometry.config = TwoViewGeometry::CALIBRATED;
    geometry.E = EssentialMatrixFromPose(
        Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));

    matcher->MatchGuided(1.0, image1, image2, &geometry);
    ExpectReversedInlierMatches(geometry);

    camera1.SetPrincipalPointY(0.0);
    matcher->MatchGuided(1.0, image1, image2, &geometry);
    EXPECT_TRUE(geometry.inlier_matches.empty());
  });
}

TEST(MatchGuidedSiftFeaturesGPU, SharedFocal) {
  RunGpuTest([] {
    TestGuidedMatchingSharedFocal(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, SharedFocalPerPairFocal) {
  RunGpuTest([] {
    TestGuidedMatchingSharedFocalPerPairFocal(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

}  // namespace
}  // namespace colmap
