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

#include "colmap/feature/sift.h"

#include "colmap/feature/utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/math/math.h"
#include "colmap/util/cuda.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"
#include "colmap/util/string.h"

#if defined(COLMAP_GPU_ENABLED)
#if defined(COLMAP_METAL_ENABLED)
#include "thirdparty/SiftMetal/SiftMetal.h"
#else
#include "thirdparty/SiftGPU/SiftGPU.h"
#if !defined(COLMAP_GUI_ENABLED)
// GLEW symbols are already defined by Qt.
#include <GL/glew.h>
#endif  // COLMAP_GUI_ENABLED
#endif  // COLMAP_METAL_ENABLED
#endif  // COLMAP_GPU_ENABLED
#include "colmap/util/eigen_alignment.h"

#include "thirdparty/VLFeat/covdet.h"
#include "thirdparty/VLFeat/sift.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <type_traits>

#include <Eigen/Geometry>

namespace colmap {

constexpr int kSiftDescriptorDim = 128;

// SIFT descriptors are normalized to length 512 (w/ quantization errors).
constexpr int kSqSiftDescriptorNorm = 512 * 512;

bool SiftExtractionOptions::Check() const {
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GT(octave_resolution, 0);
  CHECK_OPTION_GT(peak_threshold, 0.0);
  CHECK_OPTION_GT(edge_threshold, 0.0);
  CHECK_OPTION_GT(max_num_orientations, 0);
  if (domain_size_pooling) {
    CHECK_OPTION_GT(dsp_min_scale, 0);
    CHECK_OPTION_GE(dsp_max_scale, dsp_min_scale);
    CHECK_OPTION_GT(dsp_num_scales, 0);
  }
  return true;
}

bool SiftMatchingOptions::Check() const {
  CHECK_OPTION_GT(max_ratio, 0.0);
  CHECK_OPTION_GT(max_distance, 0.0);
  if (!lightglue.Check()) return false;
  return true;
}

namespace {

#if defined(COLMAP_GPU_ENABLED)
uint64_t HashBytes(uint64_t hash, const void* data, size_t num_bytes) {
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < num_bytes; ++i) {
    hash ^= bytes[i];
    hash *= kFnvPrime;
  }
  return hash;
}

template <typename T>
uint64_t HashValue(uint64_t hash, const T& value) {
  return HashBytes(hash, &value, sizeof(value));
}

#if !defined(COLMAP_METAL_ENABLED)
uint64_t HashFeatureDescriptors(const FeatureDescriptors& descriptors) {
  uint64_t hash = 14695981039346656037ULL;
  hash = HashValue(hash, descriptors.data.rows());
  hash = HashValue(hash, descriptors.data.cols());
  return HashBytes(hash,
                   descriptors.data.data(),
                   static_cast<size_t>(descriptors.data.size()));
}
#endif  // COLMAP_METAL_ENABLED

uint64_t HashFeatureKeypoints(const FeatureKeypoints& keypoints) {
  static_assert(std::is_trivially_copyable_v<FeatureKeypoint>);
  uint64_t hash = 14695981039346656037ULL;
  hash = HashValue(hash, keypoints.size());
  return HashBytes(
      hash, keypoints.data(), keypoints.size() * sizeof(FeatureKeypoint));
}

uint64_t HashCameraForNormalization(const Camera& camera) {
  uint64_t hash = 14695981039346656037ULL;
  hash = HashValue(hash, camera.model_id);
  hash = HashValue(hash, camera.width);
  hash = HashValue(hash, camera.height);
  hash = HashValue(hash, camera.params.size());
  return HashBytes(
      hash, camera.params.data(), camera.params.size() * sizeof(double));
}
#endif  // COLMAP_GPU_ENABLED

void WarnDarknessAdaptivityNotAvailable() {
  LOG(WARNING) << "Darkness adaptivity only available for GLSL SiftGPU.";
}

void ThrowCheckFeatureTypesMatch(const FeatureMatcher::Image& image1,
                                 const FeatureMatcher::Image& image2,
                                 bool check_keypoints = false) {
  THROW_CHECK_NOTNULL(image1.descriptors);
  THROW_CHECK_NOTNULL(image2.descriptors);
  THROW_CHECK_EQ(image1.descriptors->type, FeatureExtractorType::SIFT);
  THROW_CHECK_EQ(image2.descriptors->type, FeatureExtractorType::SIFT);
  THROW_CHECK_EQ(image1.descriptors->data.cols(), kSiftDescriptorDim);
  THROW_CHECK_EQ(image2.descriptors->data.cols(), kSiftDescriptorDim);
  if (check_keypoints) {
    THROW_CHECK_NOTNULL(image1.camera);
    THROW_CHECK_NOTNULL(image2.camera);
    THROW_CHECK_NOTNULL(image1.keypoints);
    THROW_CHECK_NOTNULL(image2.keypoints);
    THROW_CHECK_EQ(image1.descriptors->data.rows(), image1.keypoints->size());
    THROW_CHECK_EQ(image2.descriptors->data.rows(), image2.keypoints->size());
  }
}

// VLFeat uses a different convention to store its descriptors. This transforms
// the VLFeat format into the original SIFT format that is also used by SiftGPU.
FeatureDescriptorsData TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptorsData& vlfeat_descriptors) {
  FeatureDescriptorsData ubc_descriptors(vlfeat_descriptors.rows(),
                                         vlfeat_descriptors.cols());
  const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (Eigen::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
              vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}

class SiftCPUFeatureExtractor : public FeatureExtractor {
 public:
  using VlSiftType = std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)>;

  explicit SiftCPUFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options), sift_(nullptr, &vl_sift_delete) {
    THROW_CHECK(options_.Check());
    THROW_CHECK(!options_.sift->estimate_affine_shape);
    THROW_CHECK(!options_.sift->domain_size_pooling);
    THROW_CHECK(!options_.sift->force_covariant_extractor);
    if (options_.sift->darkness_adaptivity) {
      WarnDarknessAdaptivityNotAvailable();
    }
  }

  static std::unique_ptr<FeatureExtractor> Create(
      const FeatureExtractionOptions& options) {
    return std::make_unique<SiftCPUFeatureExtractor>(options);
  }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) {
    THROW_CHECK(bitmap.IsGrey());
    THROW_CHECK_NOTNULL(keypoints);

    if (sift_ == nullptr || sift_->width != bitmap.Width() ||
        sift_->height != bitmap.Height()) {
      sift_ = VlSiftType(vl_sift_new(bitmap.Width(),
                                     bitmap.Height(),
                                     options_.sift->num_octaves,
                                     options_.sift->octave_resolution,
                                     options_.sift->first_octave),
                         &vl_sift_delete);
      if (!sift_) {
        return false;
      }
    }

    vl_sift_set_peak_thresh(sift_.get(), options_.sift->peak_threshold);
    vl_sift_set_edge_thresh(sift_.get(), options_.sift->edge_threshold);

    // Iterate through octaves.
    std::vector<size_t> level_num_features;
    std::vector<FeatureKeypoints> level_keypoints;
    std::vector<FeatureDescriptorsData> level_descriptors;
    bool first_octave = true;
    while (true) {
      if (first_octave) {
        const std::vector<uint8_t>& data_uint8 = bitmap.RowMajorData();
        std::vector<float> data_float(data_uint8.size());
        for (size_t i = 0; i < data_uint8.size(); ++i) {
          data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
        }
        if (vl_sift_process_first_octave(sift_.get(), data_float.data())) {
          break;
        }
        first_octave = false;
      } else {
        if (vl_sift_process_next_octave(sift_.get())) {
          break;
        }
      }

      // Detect keypoints.
      vl_sift_detect(sift_.get());

      // Extract detected keypoints.
      const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift_.get());
      const int num_keypoints = vl_sift_get_nkeypoints(sift_.get());
      if (num_keypoints == 0) {
        continue;
      }

      // Extract features with different orientations per DOG level.
      size_t level_idx = 0;
      int prev_level = -1;
      FeatureDescriptorsFloatData desc(1, kSiftDescriptorDim);
      for (int i = 0; i < num_keypoints; ++i) {
        if (vl_keypoints[i].is != prev_level) {
          if (i > 0) {
            // Resize containers of previous DOG level.
            level_keypoints.back().resize(level_idx);
            if (descriptors != nullptr) {
              level_descriptors.back().conservativeResize(level_idx,
                                                          kSiftDescriptorDim);
            }
          }

          // Add containers for new DOG level.
          level_idx = 0;
          level_num_features.push_back(0);
          level_keypoints.emplace_back(options_.sift->max_num_orientations *
                                       num_keypoints);
          if (descriptors != nullptr) {
            level_descriptors.emplace_back(
                options_.sift->max_num_orientations * num_keypoints,
                kSiftDescriptorDim);
          }
        }

        level_num_features.back() += 1;
        prev_level = vl_keypoints[i].is;

        // Extract feature orientations.
        double angles[4];
        int num_orientations;
        if (options_.sift->upright) {
          num_orientations = 1;
          angles[0] = 0.0;
        } else {
          num_orientations = vl_sift_calc_keypoint_orientations(
              sift_.get(), angles, &vl_keypoints[i]);
        }

        // Note that this is different from SiftGPU, which selects the top
        // global maxima as orientations while this selects the first two
        // local maxima. It is not clear which procedure is better.
        const int num_used_orientations =
            std::min(num_orientations, options_.sift->max_num_orientations);

        for (int o = 0; o < num_used_orientations; ++o) {
          level_keypoints.back()[level_idx] =
              FeatureKeypoint(vl_keypoints[i].x + 0.5f,
                              vl_keypoints[i].y + 0.5f,
                              vl_keypoints[i].sigma,
                              angles[o]);
          if (descriptors != nullptr) {
            vl_sift_calc_keypoint_descriptor(
                sift_.get(), desc.data(), &vl_keypoints[i], angles[o]);
            if (options_.sift->normalization ==
                SiftExtractionOptions::Normalization::L2) {
              L2NormalizeFeatureDescriptors(&desc);
            } else if (options_.sift->normalization ==
                       SiftExtractionOptions::Normalization::L1_ROOT) {
              L1RootNormalizeFeatureDescriptors(&desc);
            } else {
              LOG(FATAL_THROW) << "Normalization type not supported";
            }

            level_descriptors.back().row(level_idx) =
                FeatureDescriptorsToUnsignedByte(desc);
          }

          level_idx += 1;
        }
      }

      // Resize containers for last DOG level in octave.
      level_keypoints.back().resize(level_idx);
      if (descriptors != nullptr) {
        level_descriptors.back().conservativeResize(level_idx,
                                                    kSiftDescriptorDim);
      }
    }

    // Determine how many DOG levels to keep to satisfy max_num_features option.
    int first_level_to_keep = 0;
    int num_features = 0;
    int num_features_with_orientations = 0;
    for (int i = level_keypoints.size() - 1; i >= 0; --i) {
      num_features += level_num_features[i];
      num_features_with_orientations += level_keypoints[i].size();
      if (num_features > options_.sift->max_num_features) {
        first_level_to_keep = i;
        break;
      }
    }

    // Extract the features to be kept.
    {
      size_t k = 0;
      keypoints->resize(num_features_with_orientations);
      for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
        for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
          (*keypoints)[k] = level_keypoints[i][j];
          k += 1;
        }
      }
    }

    // Compute the descriptors for the detected keypoints.
    if (descriptors != nullptr) {
      size_t k = 0;
      descriptors->data.resize(num_features_with_orientations,
                               kSiftDescriptorDim);
      for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
        for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
          descriptors->data.row(k) = level_descriptors[i].row(j);
          k += 1;
        }
      }
      descriptors->data =
          TransformVLFeatToUBCFeatureDescriptors(descriptors->data);
      descriptors->type = FeatureExtractorType::SIFT;
    }

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  VlSiftType sift_;
};

class CovariantSiftCPUFeatureExtractor : public FeatureExtractor {
 public:
  explicit CovariantSiftCPUFeatureExtractor(
      const FeatureExtractionOptions& options)
      : options_(options) {
    THROW_CHECK(options_.Check());
    if (options_.sift->darkness_adaptivity) {
      WarnDarknessAdaptivityNotAvailable();
    }
  }

  static std::unique_ptr<FeatureExtractor> Create(
      const FeatureExtractionOptions& options) {
    return std::make_unique<CovariantSiftCPUFeatureExtractor>(options);
  }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) {
    THROW_CHECK(bitmap.IsGrey());
    THROW_CHECK_NOTNULL(keypoints);

    // Setup covariant SIFT detector.
    std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
        vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
    if (!covdet) {
      return false;
    }

    const int kMaxOctaveResolution = 1000;
    THROW_CHECK_LE(options_.sift->octave_resolution, kMaxOctaveResolution);

    vl_covdet_set_first_octave(covdet.get(), options_.sift->first_octave);
    vl_covdet_set_octave_resolution(covdet.get(),
                                    options_.sift->octave_resolution);
    vl_covdet_set_peak_threshold(covdet.get(), options_.sift->peak_threshold);
    vl_covdet_set_edge_threshold(covdet.get(), options_.sift->edge_threshold);

    {
      const std::vector<uint8_t>& data_uint8 = bitmap.RowMajorData();
      std::vector<float> data_float(data_uint8.size());
      for (size_t i = 0; i < data_uint8.size(); ++i) {
        data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
      }
      vl_covdet_put_image(
          covdet.get(), data_float.data(), bitmap.Width(), bitmap.Height());
    }

    vl_covdet_detect(covdet.get(), options_.sift->max_num_features);

    if (options_.sift->estimate_affine_shape) {
      vl_covdet_extract_affine_shape(covdet.get());
    }

    if (!options_.sift->upright) {
      vl_covdet_extract_orientations(covdet.get());
    }

    const int num_features = vl_covdet_get_num_features(covdet.get());
    VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

    // Sort features according to detected octave and scale.
    std::sort(
        features,
        features + num_features,
        [](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
          if (feature1.o == feature2.o) {
            return feature1.s > feature2.s;
          } else {
            return feature1.o > feature2.o;
          }
        });

    const size_t max_num_features =
        static_cast<size_t>(options_.sift->max_num_features);

    // Copy detected keypoints and clamp when maximum number of features
    // reached.
    int prev_octave_scale_idx = std::numeric_limits<int>::max();
    for (int i = 0; i < num_features; ++i) {
      FeatureKeypoint keypoint;
      keypoint.x = features[i].frame.x + 0.5;
      keypoint.y = features[i].frame.y + 0.5;
      keypoint.a11 = features[i].frame.a11;
      keypoint.a12 = features[i].frame.a12;
      keypoint.a21 = features[i].frame.a21;
      keypoint.a22 = features[i].frame.a22;
      keypoints->push_back(keypoint);

      const int octave_scale_idx =
          features[i].o * kMaxOctaveResolution + features[i].s;
      THROW_CHECK_LE(octave_scale_idx, prev_octave_scale_idx);

      if (octave_scale_idx != prev_octave_scale_idx &&
          keypoints->size() >= max_num_features) {
        break;
      }

      prev_octave_scale_idx = octave_scale_idx;
    }

    // Compute the descriptors for the detected keypoints.
    if (descriptors != nullptr) {
      descriptors->data.resize(keypoints->size(), kSiftDescriptorDim);

      const size_t kPatchResolution = 15;
      const size_t kPatchSide = 2 * kPatchResolution + 1;
      const double kPatchRelativeExtent = 7.5;
      const double kPatchRelativeSmoothing = 1;
      const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
      const double kSigma =
          kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

      std::vector<float> patch(kPatchSide * kPatchSide);
      std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

      float dsp_min_scale = 1;
      float dsp_scale_step = 0;
      int dsp_num_scales = 1;
      if (options_.sift->domain_size_pooling) {
        dsp_min_scale = options_.sift->dsp_min_scale;
        dsp_scale_step =
            (options_.sift->dsp_max_scale - options_.sift->dsp_min_scale) /
            options_.sift->dsp_num_scales;
        dsp_num_scales = options_.sift->dsp_num_scales;
      }

      FeatureDescriptorsFloatData descriptor(1, kSiftDescriptorDim);
      FeatureDescriptorsFloatData scaled_descriptors(dsp_num_scales,
                                                     kSiftDescriptorDim);

      std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
          vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
      if (!sift) {
        return false;
      }

      vl_sift_set_magnif(sift.get(), 3.0);

      for (size_t i = 0; i < keypoints->size(); ++i) {
        for (int s = 0; s < dsp_num_scales; ++s) {
          const double dsp_scale = dsp_min_scale + s * dsp_scale_step;

          VlFrameOrientedEllipse scaled_frame = features[i].frame;
          scaled_frame.a11 *= dsp_scale;
          scaled_frame.a12 *= dsp_scale;
          scaled_frame.a21 *= dsp_scale;
          scaled_frame.a22 *= dsp_scale;

          vl_covdet_extract_patch_for_frame(covdet.get(),
                                            patch.data(),
                                            kPatchResolution,
                                            kPatchRelativeExtent,
                                            kPatchRelativeSmoothing,
                                            scaled_frame);

          vl_imgradient_polar_f(patchXY.data(),
                                patchXY.data() + 1,
                                2,
                                2 * kPatchSide,
                                patch.data(),
                                kPatchSide,
                                kPatchSide,
                                kPatchSide);

          vl_sift_calc_raw_descriptor(sift.get(),
                                      patchXY.data(),
                                      scaled_descriptors.row(s).data(),
                                      kPatchSide,
                                      kPatchSide,
                                      kPatchResolution,
                                      kPatchResolution,
                                      kSigma,
                                      0);
        }

        if (options_.sift->domain_size_pooling) {
          descriptor = scaled_descriptors.colwise().mean();
        } else {
          descriptor = scaled_descriptors;
        }

        THROW_CHECK_EQ(descriptor.cols(), kSiftDescriptorDim);

        if (options_.sift->normalization ==
            SiftExtractionOptions::Normalization::L2) {
          L2NormalizeFeatureDescriptors(&descriptor);
        } else if (options_.sift->normalization ==
                   SiftExtractionOptions::Normalization::L1_ROOT) {
          L1RootNormalizeFeatureDescriptors(&descriptor);
        } else {
          LOG(FATAL_THROW) << "Normalization type not supported";
        }

        descriptors->data.row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
      }

      descriptors->data =
          TransformVLFeatToUBCFeatureDescriptors(descriptors->data);
      descriptors->type = FeatureExtractorType::SIFT;
    }

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
};

#if defined(COLMAP_GPU_ENABLED) && !defined(COLMAP_METAL_ENABLED)
// Mutexes that ensure that only one thread extracts/matches on the same GPU
// at the same time, since SiftGPU internally uses static variables.
static std::map<int, std::unique_ptr<std::mutex>> sift_gpu_mutexes_;

class SiftGPUFeatureExtractor : public FeatureExtractor {
 public:
  explicit SiftGPUFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options) {
    THROW_CHECK(options_.Check());
    THROW_CHECK(!options_.sift->estimate_affine_shape);
    THROW_CHECK(!options_.sift->domain_size_pooling);
    THROW_CHECK(!options_.sift->force_covariant_extractor);
  }

  static std::unique_ptr<FeatureExtractor> Create(
      const FeatureExtractionOptions& options) {
    // SiftGPU uses many global static state variables and the initialization
    // must be thread-safe in order to work correctly. This is enforced here.
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
    THROW_CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

    std::vector<std::string> sift_gpu_args;

    sift_gpu_args.push_back("./sift_gpu");

#if defined(COLMAP_CUDA_ENABLED)
    // Use CUDA version by default if darkness adaptivity is disabled.
    if (!options.sift->darkness_adaptivity && gpu_indices[0] < 0) {
      gpu_indices[0] = 0;
    }

    if (gpu_indices[0] >= 0) {
      sift_gpu_args.push_back("-cuda");
      sift_gpu_args.push_back(std::to_string(gpu_indices[0]));
    }
#endif  // COLMAP_CUDA_ENABLED

    // Darkness adaptivity (hidden feature). Significantly improves
    // distribution of features. Only available in GLSL version.
    if (options.sift->darkness_adaptivity) {
      if (gpu_indices[0] >= 0) {
        WarnDarknessAdaptivityNotAvailable();
      }
      sift_gpu_args.push_back("-da");
    }

    // No verbose logging.
    sift_gpu_args.push_back("-v");
    sift_gpu_args.push_back("0");

    // Set maximum image dimension.
    // Note the max dimension of SiftGPU is the maximum dimension of the
    // first octave in the pyramid (which is the 'first_octave').
    const int compensation_factor = 1
                                    << -std::min(0, options.sift->first_octave);
    sift_gpu_args.push_back("-maxd");
    sift_gpu_args.push_back(
        std::to_string(options.EffMaxImageSize() * compensation_factor));

    // Keep the highest level features.
    sift_gpu_args.push_back("-tc2");
    sift_gpu_args.push_back(std::to_string(options.sift->max_num_features));

    // First octave level.
    sift_gpu_args.push_back("-fo");
    sift_gpu_args.push_back(std::to_string(options.sift->first_octave));

    // Number of octave levels.
    sift_gpu_args.push_back("-d");
    sift_gpu_args.push_back(std::to_string(options.sift->octave_resolution));

    // Peak threshold.
    sift_gpu_args.push_back("-t");
    sift_gpu_args.push_back(std::to_string(options.sift->peak_threshold));

    // Edge threshold.
    sift_gpu_args.push_back("-e");
    sift_gpu_args.push_back(std::to_string(options.sift->edge_threshold));

    if (options.sift->upright) {
      // Fix the orientation to 0 for upright features.
      sift_gpu_args.push_back("-ofix");
      // Maximum number of orientations.
      sift_gpu_args.push_back("-mo");
      sift_gpu_args.push_back("1");
    } else {
      // Maximum number of orientations.
      sift_gpu_args.push_back("-mo");
      sift_gpu_args.push_back(
          std::to_string(options.sift->max_num_orientations));
    }

    std::vector<const char*> sift_gpu_args_cstr;
    sift_gpu_args_cstr.reserve(sift_gpu_args.size());
    for (const auto& arg : sift_gpu_args) {
      sift_gpu_args_cstr.push_back(arg.c_str());
    }

    auto extractor = std::make_unique<SiftGPUFeatureExtractor>(options);

    // Note that the SiftGPU object is not movable (for whatever reason).
    // If we instead create the object here and move it to the constructor, the
    // program segfaults inside SiftGPU.

    extractor->sift_gpu_.ParseParam(sift_gpu_args_cstr.size(),
                                    sift_gpu_args_cstr.data());

    extractor->sift_gpu_.gpu_index = gpu_indices[0];
    if (sift_gpu_mutexes_.count(gpu_indices[0]) == 0) {
      sift_gpu_mutexes_.emplace(gpu_indices[0], std::make_unique<std::mutex>());
    }

    if (extractor->sift_gpu_.VerifyContextGL() !=
        SiftGPU::SIFTGPU_FULL_SUPPORTED) {
      return nullptr;
    }

    return extractor;
  }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    THROW_CHECK(bitmap.IsGrey());
    THROW_CHECK_NOTNULL(keypoints);
    THROW_CHECK_NOTNULL(descriptors);

    // Note the max dimension of SiftGPU is the maximum dimension of the
    // first octave in the pyramid (which is the 'first_octave').
    const int compensation_factor =
        1 << -std::min(0, options_.sift->first_octave);
    THROW_CHECK_EQ(options_.EffMaxImageSize() * compensation_factor,
                   sift_gpu_.GetMaxDimension());

    std::lock_guard<std::mutex> lock(*sift_gpu_mutexes_[sift_gpu_.gpu_index]);

    // Note, that this produces slightly different results than using SiftGPU
    // directly for RGB->GRAY conversion, since it uses different weights.
    const int code = sift_gpu_.RunSIFT(bitmap.Pitch(),
                                       bitmap.Height(),
                                       bitmap.RowMajorData().data(),
                                       GL_LUMINANCE,
                                       GL_UNSIGNED_BYTE);

    const int kSuccessCode = 1;
    if (code != kSuccessCode) {
      return false;
    }

    const size_t num_features = static_cast<size_t>(sift_gpu_.GetFeatureNum());

    keypoints_buffer_.resize(num_features);

    FeatureDescriptorsFloatData descriptors_float(num_features,
                                                  kSiftDescriptorDim);

    // Download the extracted keypoints and descriptors.
    sift_gpu_.GetFeatureVector(keypoints_buffer_.data(),
                               descriptors_float.data());

    keypoints->resize(num_features);
    for (size_t i = 0; i < num_features; ++i) {
      (*keypoints)[i] = FeatureKeypoint(keypoints_buffer_[i].x,
                                        keypoints_buffer_[i].y,
                                        keypoints_buffer_[i].s,
                                        keypoints_buffer_[i].o);
    }

    // Save and normalize the descriptors.
    if (options_.sift->normalization ==
        SiftExtractionOptions::Normalization::L2) {
      L2NormalizeFeatureDescriptors(&descriptors_float);
    } else if (options_.sift->normalization ==
               SiftExtractionOptions::Normalization::L1_ROOT) {
      L1RootNormalizeFeatureDescriptors(&descriptors_float);
    } else {
      LOG(FATAL_THROW) << "Normalization type not supported";
    }

    descriptors->data = FeatureDescriptorsToUnsignedByte(descriptors_float);
    descriptors->type = FeatureExtractorType::SIFT;

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  SiftGPU sift_gpu_;
  std::vector<SiftKeypoint> keypoints_buffer_;
};
#endif  // COLMAP_GPU_ENABLED && !COLMAP_METAL_ENABLED

#if defined(COLMAP_METAL_ENABLED)
class SiftMetalFeatureExtractor : public FeatureExtractor {
 public:
  explicit SiftMetalFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options) {
    THROW_CHECK(options_.Check());
  }

  static std::unique_ptr<FeatureExtractor> Create(
      const FeatureExtractionOptions& options) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    auto extractor = std::make_unique<SiftMetalFeatureExtractor>(options);

    sift_metal::Options metal_opts;
    metal_opts.num_octaves = options.sift->num_octaves;
    metal_opts.scales_per_octave = options.sift->octave_resolution;
    metal_opts.first_octave = options.sift->first_octave;
    metal_opts.peak_threshold =
        static_cast<float>(options.sift->peak_threshold);
    metal_opts.edge_threshold =
        static_cast<float>(options.sift->edge_threshold);
    metal_opts.max_num_features = options.sift->max_num_features;
    metal_opts.max_num_orientations = options.sift->max_num_orientations;
    metal_opts.upright = options.sift->upright;

    // Allocate the aspect-aware pyramid lazily from the first image rather
    // than eagerly reserving a square EffMaxImageSize() pyramid.
    if (!extractor->extractor_.Init(metal_opts, 1, 1)) {
      LOG(ERROR) << "Failed to initialize Metal SIFT extractor";
      return nullptr;
    }

    return extractor;
  }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    THROW_CHECK(bitmap.IsGrey());
    THROW_CHECK_NOTNULL(keypoints);
    THROW_CHECK_NOTNULL(descriptors);

    sift_metal::ExtractResult metal_result;
    if (!extractor_.Extract(bitmap.RowMajorData().data(),
                            bitmap.Width(),
                            bitmap.Height(),
                            &metal_result)) {
      return false;
    }

    const size_t num_features = metal_result.keypoints.size();
    keypoints->resize(num_features);
    for (size_t i = 0; i < num_features; ++i) {
      const auto& kp = metal_result.keypoints[i];
      (*keypoints)[i] = FeatureKeypoint(kp.x, kp.y, kp.sigma, kp.orientation);
    }

    // Normalize and quantize descriptors. The Metal extractor returns
    // row-major [num_features x 128] floats, matching the Eigen layout.
    FeatureDescriptorsFloatData descriptors_float =
        Eigen::Map<const FeatureDescriptorsFloatData>(
            metal_result.descriptors.data(), num_features, kSiftDescriptorDim);

    if (options_.sift->normalization ==
        SiftExtractionOptions::Normalization::L2) {
      L2NormalizeFeatureDescriptors(&descriptors_float);
    } else if (options_.sift->normalization ==
               SiftExtractionOptions::Normalization::L1_ROOT) {
      L1RootNormalizeFeatureDescriptors(&descriptors_float);
    } else {
      LOG(FATAL_THROW) << "Normalization type not supported";
    }

    descriptors->data = FeatureDescriptorsToUnsignedByte(descriptors_float);
    descriptors->type = FeatureExtractorType::SIFT;

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  sift_metal::SiftMetalExtractor extractor_;
};
#endif  // COLMAP_METAL_ENABLED

}  // namespace

std::unique_ptr<FeatureExtractor> CreateSiftFeatureExtractor(
    const FeatureExtractionOptions& options) {
  if (options.sift->estimate_affine_shape ||
      options.sift->domain_size_pooling ||
      options.sift->force_covariant_extractor) {
    if (options.use_gpu) {
      LOG(INFO) << "SIFT GPU extraction requested, but affine shape "
                   "estimation, domain-size pooling, or forced covariant "
                   "extraction requires covariant CPU SIFT. Falling back to "
                   "CPU extraction.";
    }
    LOG(INFO) << "Creating Covariant SIFT CPU feature extractor";
    return CovariantSiftCPUFeatureExtractor::Create(options);
  } else if (options.use_gpu) {
#if defined(COLMAP_METAL_ENABLED)
    LOG(INFO) << "Creating SIFT Metal GPU feature extractor";
    return SiftMetalFeatureExtractor::Create(options);
#elif defined(COLMAP_GPU_ENABLED)
    LOG(INFO) << "Creating SIFT GPU feature extractor";
    return SiftGPUFeatureExtractor::Create(options);
#else
    return nullptr;
#endif
  } else {
    LOG(INFO) << "Creating SIFT CPU feature extractor";
    return SiftCPUFeatureExtractor::Create(options);
  }
}

namespace {

size_t FindBestMatchesOneWayBruteForce(
    const Eigen::RowMajorMatrixXf& dot_products,
    const float max_ratio,
    const float max_distance,
    std::vector<int>* matches) {
  constexpr float kInvSqDescriptorNorm =
      static_cast<float>(1. / kSqSiftDescriptorNorm);

  size_t num_matches = 0;
  matches->resize(dot_products.rows(), -1);

  for (Eigen::Index i1 = 0; i1 < dot_products.rows(); ++i1) {
    int best_d2_idx = -1;
    float best_dot_product = 0;
    float second_best_dot_product = 0;
    for (Eigen::Index i2 = 0; i2 < dot_products.cols(); ++i2) {
      const float dot_product = dot_products(i1, i2);
      if (dot_product > best_dot_product) {
        best_d2_idx = i2;
        second_best_dot_product = best_dot_product;
        best_dot_product = dot_product;
      } else if (dot_product > second_best_dot_product) {
        second_best_dot_product = dot_product;
      }
    }

    // Check if any match found.
    if (best_d2_idx == -1) {
      continue;
    }

    // Convert to L2 distance in which the thresholds are defined.
    const float best_dist_normed =
        std::acos(std::min(kInvSqDescriptorNorm * best_dot_product, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed = std::acos(
        std::min(kInvSqDescriptorNorm * second_best_dot_product, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    ++num_matches;
    (*matches)[i1] = best_d2_idx;
  }

  return num_matches;
}

void FindBestMatchesBruteForce(const Eigen::RowMajorMatrixXf& dot_products,
                               const float max_ratio,
                               const float max_distance,
                               const bool cross_check,
                               FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches_1to2;
  const size_t num_matches_1to2 = FindBestMatchesOneWayBruteForce(
      dot_products, max_ratio, max_distance, &matches_1to2);

  if (cross_check) {
    std::vector<int> matches_2to1;
    const size_t num_matches_2to1 = FindBestMatchesOneWayBruteForce(
        dot_products.transpose(), max_ratio, max_distance, &matches_2to1);
    matches->reserve(std::min(num_matches_1to2, num_matches_2to1));
    for (size_t i1 = 0; i1 < matches_1to2.size(); ++i1) {
      if (matches_1to2[i1] != -1 && matches_2to1[matches_1to2[i1]] != -1 &&
          matches_2to1[matches_1to2[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches_1to2[i1];
        matches->push_back(match);
      }
    }
  } else {
    matches->reserve(num_matches_1to2);
    for (size_t i1 = 0; i1 < matches_1to2.size(); ++i1) {
      if (matches_1to2[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches_1to2[i1];
        matches->push_back(match);
      }
    }
  }
}

size_t FindBestMatchesOneWayIndex(const Eigen::RowMajorMatrixXi& indices,
                                  const Eigen::RowMajorMatrixXf& l2_dists,
                                  const float max_ratio,
                                  const float max_distance,
                                  std::vector<int>* matches) {
  const float max_l2_dist = kSqSiftDescriptorNorm * max_distance * max_distance;

  size_t num_matches = 0;
  matches->resize(indices.rows(), -1);

  for (int d1_idx = 0; d1_idx < indices.rows(); ++d1_idx) {
    int best_d2_idx = -1;
    float best_l2_dist = std::numeric_limits<float>::max();
    float second_best_l2_dist = std::numeric_limits<float>::max();
    for (int n_idx = 0; n_idx < indices.cols(); ++n_idx) {
      const int d2_idx = indices(d1_idx, n_idx);
      const float l2_dist = l2_dists(d1_idx, n_idx);
      if (l2_dist < best_l2_dist) {
        best_d2_idx = d2_idx;
        second_best_l2_dist = best_l2_dist;
        best_l2_dist = l2_dist;
      } else if (l2_dist < second_best_l2_dist) {
        second_best_l2_dist = l2_dist;
      }
    }

    // Check if any match found.
    if (best_d2_idx == -1) {
      continue;
    }

    // Check if match distance passes threshold.
    if (best_l2_dist > max_l2_dist) {
      continue;
    }

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (std::sqrt(best_l2_dist) >= max_ratio * std::sqrt(second_best_l2_dist)) {
      continue;
    }

    ++num_matches;
    (*matches)[d1_idx] = best_d2_idx;
  }

  return num_matches;
}

void FindBestMatchesIndex(const Eigen::RowMajorMatrixXi& indices_1to2,
                          const Eigen::RowMajorMatrixXf& l2_dists_1to2,
                          const Eigen::RowMajorMatrixXi& indices_2to1,
                          const Eigen::RowMajorMatrixXf& l2_dists_2to1,
                          const float max_ratio,
                          const float max_distance,
                          const bool cross_check,
                          FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches_1to2;
  const size_t num_matches_1to2 = FindBestMatchesOneWayIndex(
      indices_1to2, l2_dists_1to2, max_ratio, max_distance, &matches_1to2);

  if (cross_check && indices_2to1.rows()) {
    std::vector<int> matches_2to1;
    const size_t num_matches_2to1 = FindBestMatchesOneWayIndex(
        indices_2to1, l2_dists_2to1, max_ratio, max_distance, &matches_2to1);
    matches->reserve(std::min(num_matches_1to2, num_matches_2to1));
    for (size_t i1 = 0; i1 < matches_1to2.size(); ++i1) {
      if (matches_1to2[i1] != -1 && matches_2to1[matches_1to2[i1]] != -1 &&
          matches_2to1[matches_1to2[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches_1to2[i1];
        matches->push_back(match);
      }
    }
  } else {
    matches->reserve(num_matches_1to2);
    for (size_t i1 = 0; i1 < matches_1to2.size(); ++i1) {
      if (matches_1to2[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches_1to2[i1];
        matches->push_back(match);
      }
    }
  }
}

enum class DistanceType {
  L2,
  DOT_PRODUCT,
};

// Computes the pairwise descriptor distance matrix. When `guided_filter` is
// given, it is called with the pair of descriptor indices and returns whether
// the pair should be rejected on geometric grounds; rejected pairs get the
// worst possible distance. Passing indices rather than keypoint coordinates
// lets the caller filter on whatever geometry it needs - pixels, bearings, or
// bearings plus their Jacobians - without this function knowing any of it.
Eigen::RowMajorMatrixXf ComputeSiftDistanceMatrix(
    const DistanceType distance_type,
    const FeatureDescriptorsData& descriptors1,
    const FeatureDescriptorsData& descriptors2,
    const std::function<bool(Eigen::Index, Eigen::Index)>& guided_filter) {
  const Eigen::Matrix<int, Eigen::Dynamic, kSiftDescriptorDim>
      descriptors1_int = descriptors1.cast<int>();
  const Eigen::Matrix<int, Eigen::Dynamic, kSiftDescriptorDim>
      descriptors2_int = descriptors2.cast<int>();

  Eigen::RowMajorMatrixXf distances(descriptors1.rows(), descriptors2.rows());
  for (Eigen::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
    for (Eigen::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
      if (guided_filter != nullptr && guided_filter(i1, i2)) {
        if (distance_type == DistanceType::L2) {
          distances(i1, i2) = kSqSiftDescriptorNorm;
        } else if (distance_type == DistanceType::DOT_PRODUCT) {
          distances(i1, i2) = 0;
        } else {
          LOG(FATAL_THROW) << "Distance type not supported";
        }
      } else {
        if (distance_type == DistanceType::L2) {
          distances(i1, i2) =
              (descriptors1_int.row(i1) - descriptors2_int.row(i2))
                  .squaredNorm();
        } else if (distance_type == DistanceType::DOT_PRODUCT) {
          distances(i1, i2) =
              descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
        } else {
          LOG(FATAL_THROW) << "Distance type not supported";
        }
      }
    }
  }

  return distances;
}

Eigen::Matrix3d RayHomographyForGuidedMatching(
    const Eigen::Matrix3d& stored_homography,
    const Camera& camera1,
    const Camera& camera2) {
  if (camera1.IsSpherical() || camera2.IsSpherical()) {
    return stored_homography;
  }
  return camera2.CalibrationMatrix().inverse() * stored_homography *
         camera1.CalibrationMatrix();
}

// Ray homographies are stored directly for spherical pairs and conjugated into
// a virtual pinhole image for calibrated distorted pairs. Only fresh estimator
// output carries transient provenance; database-loaded and manually constructed
// geometries default to UNKNOWN and retain the backwards-compatible pixel path.
bool UseRayHomographyForGuidedMatching(const TwoViewGeometry& geometry) {
  return geometry.H_estimation_space ==
         TwoViewGeometry::HomographyEstimationSpace::CAMERA_RAY;
}

// The CUDA/GLSL and Metal homography kernels consume raw pixel coordinates and
// cannot project an arbitrary camera ray. Keep their fast path for ordinary
// pixel homographies, but use this common CPU path for the rare ray-homography
// cases so every backend has the same correct behavior without widening either
// GPU ABI.
void MatchGuidedSiftFeaturesWithRayHomography(
    const double max_error,
    const Camera& camera1,
    const FeatureKeypoints& keypoints1,
    const FeatureDescriptorsData& descriptors1,
    const Camera& camera2,
    const FeatureKeypoints& keypoints2,
    const FeatureDescriptorsData& descriptors2,
    const Eigen::Matrix3d& ray_homography,
    const SiftMatchingOptions& options,
    const int max_num_features,
    const int max_num_matches,
    FeatureMatches* matches) {
  THROW_CHECK_NOTNULL(matches);
  matches->clear();
  THROW_CHECK_EQ(keypoints1.size(), descriptors1.rows());
  THROW_CHECK_EQ(keypoints2.size(), descriptors2.rows());
  if (max_num_features <= 0 || max_num_matches <= 0) {
    return;
  }
  const Eigen::Index num_features1 =
      std::min<Eigen::Index>(descriptors1.rows(), max_num_features);
  const Eigen::Index num_features2 =
      std::min<Eigen::Index>(descriptors2.rows(), max_num_features);
  if (num_features1 == 0 || num_features2 == 0 || !ray_homography.allFinite()) {
    return;
  }

  std::vector<std::optional<Eigen::Vector2d>> transferred_points1(
      num_features1);
  for (Eigen::Index i = 0; i < num_features1; ++i) {
    const FeatureKeypoint& keypoint = keypoints1[i];
    const std::optional<Eigen::Vector3d> ray =
        camera1.CamRayFromImg(Eigen::Vector2d(keypoint.x, keypoint.y));
    if (ray.has_value()) {
      transferred_points1[i] = camera2.ImgFromCam(ray_homography * *ray);
    }
  }

  const double max_residual = max_error * max_error;
  const bool wrap_horizontal = camera2.IsSpherical();
  const double image_width = static_cast<double>(camera2.width);

  // Scan descriptor pairs once and retain only the two best candidates in each
  // direction. Geometrically rejected pairs retain the existing maximum-
  // descriptor-distance sentinel so custom ratio/distance settings behave
  // exactly as in the dense path. Materializing the full distance, transpose,
  // and index matrices costs about 16 bytes per pair (1 GiB at 8192 x 8192),
  // while this keeps memory linear in the feature count.
  const Eigen::Matrix<int, Eigen::Dynamic, kSiftDescriptorDim>
      descriptors1_int = descriptors1.topRows(num_features1).cast<int>();
  const Eigen::Matrix<int, Eigen::Dynamic, kSiftDescriptorDim>
      descriptors2_int = descriptors2.topRows(num_features2).cast<int>();
  const float initial_distance = std::numeric_limits<float>::max();
  const float rejected_distance = static_cast<float>(kSqSiftDescriptorNorm);
  std::vector<int> best_indices_1to2(num_features1, -1);
  std::vector<float> best_dists_1to2(num_features1, initial_distance);
  std::vector<float> second_best_dists_1to2(num_features1, initial_distance);
  std::vector<int> best_indices_2to1(options.cross_check ? num_features2 : 0,
                                     -1);
  std::vector<float> best_dists_2to1(options.cross_check ? num_features2 : 0,
                                     initial_distance);
  std::vector<float> second_best_dists_2to1(
      options.cross_check ? num_features2 : 0, initial_distance);

  auto update_best = [](const int candidate_idx,
                        const float distance,
                        int* best_idx,
                        float* best_distance,
                        float* second_best_distance) {
    if (distance < *best_distance) {
      *best_idx = candidate_idx;
      *second_best_distance = *best_distance;
      *best_distance = distance;
    } else if (distance < *second_best_distance) {
      *second_best_distance = distance;
    }
  };

  for (Eigen::Index i1 = 0; i1 < num_features1; ++i1) {
    const std::optional<Eigen::Vector2d>& transferred_point =
        transferred_points1[i1];
    for (Eigen::Index i2 = 0; i2 < num_features2; ++i2) {
      bool is_geometry_valid =
          transferred_point.has_value() && transferred_point->allFinite();
      if (is_geometry_valid) {
        const FeatureKeypoint& keypoint2 = keypoints2[i2];
        Eigen::Vector2d error =
            *transferred_point - Eigen::Vector2d(keypoint2.x, keypoint2.y);
        if (wrap_horizontal) {
          error.x() -= image_width * std::round(error.x() / image_width);
        }
        is_geometry_valid = error.squaredNorm() <= max_residual;
      }
      const float distance =
          is_geometry_valid ? static_cast<float>((descriptors1_int.row(i1) -
                                                  descriptors2_int.row(i2))
                                                     .squaredNorm())
                            : rejected_distance;
      update_best(static_cast<int>(i2),
                  distance,
                  &best_indices_1to2[i1],
                  &best_dists_1to2[i1],
                  &second_best_dists_1to2[i1]);
      if (options.cross_check) {
        update_best(static_cast<int>(i1),
                    distance,
                    &best_indices_2to1[i2],
                    &best_dists_2to1[i2],
                    &second_best_dists_2to1[i2]);
      }
    }
  }

  const float max_ratio = static_cast<float>(options.max_ratio);
  const float max_distance = static_cast<float>(options.max_distance);
  const float max_l2_dist = kSqSiftDescriptorNorm * max_distance * max_distance;
  auto passes_thresholds =
      [&](const size_t idx,
          const std::vector<int>& best_indices,
          const std::vector<float>& best_distances,
          const std::vector<float>& second_best_distances) {
        return best_indices[idx] != -1 && best_distances[idx] <= max_l2_dist &&
               std::sqrt(best_distances[idx]) <
                   max_ratio * std::sqrt(second_best_distances[idx]);
      };

  matches->reserve(static_cast<size_t>(
      std::min<Eigen::Index>(num_features1, max_num_matches)));
  for (size_t i1 = 0; i1 < static_cast<size_t>(num_features1); ++i1) {
    if (!passes_thresholds(
            i1, best_indices_1to2, best_dists_1to2, second_best_dists_1to2)) {
      continue;
    }
    const int i2 = best_indices_1to2[i1];
    if (options.cross_check &&
        (!passes_thresholds(
             i2, best_indices_2to1, best_dists_2to1, second_best_dists_2to1) ||
         best_indices_2to1[i2] != static_cast<int>(i1))) {
      continue;
    }
    matches->push_back(
        FeatureMatch{static_cast<point2D_t>(i1), static_cast<point2D_t>(i2)});
    if (matches->size() == static_cast<size_t>(max_num_matches)) {
      break;
    }
  }
}

// Unit bearing vectors and their pixel Jacobians for a set of keypoints,
// together with a validity mask.
//
// Keypoints that cannot be unprojected - back-hemisphere pixels of an
// omnidirectional camera have no normalized image plane representation, and
// iterative undistortion can fail - are marked invalid and must be excluded
// from matching by the caller. Note that encoding invalidity as an extreme
// coordinate does *not* work: the Sampson error is a ratio whose numerator and
// denominator scale together, so a point pushed to infinity along a direction d
// converges to the finite distance between its partner and the epipolar line of
// d, which admits rather than rejects partners lying near that one line.
struct CamRaysWithJac {
  std::vector<Eigen::Vector3d> rays;
  std::vector<Eigen::Matrix3x2d> jacobians;
  std::vector<bool> valid;
};

CamRaysWithJac ComputeCamRaysWithJac(const Camera& camera,
                                     const FeatureKeypoints& keypoints) {
  CamRaysWithJac cam_rays;
  cam_rays.rays.resize(keypoints.size());
  cam_rays.jacobians.resize(keypoints.size());
  cam_rays.valid.resize(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const FeatureKeypoint& keypoint = keypoints[i];
    if (const auto ray_and_jac = camera.CamRayFromImgWithJac(
            Eigen::Vector2d(keypoint.x, keypoint.y))) {
      cam_rays.rays[i] = ray_and_jac->ray;
      cam_rays.jacobians[i] = ray_and_jac->jacobian;
      cam_rays.valid[i] = true;
    } else {
      cam_rays.rays[i].setZero();
      cam_rays.jacobians[i].setZero();
      cam_rays.valid[i] = false;
    }
  }
  return cam_rays;
}

// Selects the epipolar model used to guide matching. The essential matrix is
// preferred whenever the intrinsics are known, which properly handles
// non-pinhole camera models where the fundamental matrix relationship does not
// hold. The intrinsics are known either from priors (calibrated configs) or
// because the two-view solver recovered them: such pairs are labeled
// UNCALIBRATED but carry the estimated intrinsics in `camera1`/`camera2`.
// Either side alone suffices, to support solvers that estimate only one side
// against an already calibrated view.
bool UseEssentialMatrixForGuidedMatching(const TwoViewGeometry& geometry) {
  if (!geometry.E.has_value()) {
    return false;
  }
  if (geometry.config == TwoViewGeometry::CALIBRATED ||
      geometry.config == TwoViewGeometry::CALIBRATED_RIG) {
    return true;
  }
  if (geometry.config == TwoViewGeometry::UNCALIBRATED) {
    return geometry.camera1.has_value() || geometry.camera2.has_value();
  }
  return false;
}

class SiftCPUFeatureMatcher : public FeatureMatcher {
 public:
  explicit SiftCPUFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options) {
    THROW_CHECK(options_.Check());
  }

  static std::unique_ptr<FeatureMatcher> Create(
      const FeatureMatchingOptions& options) {
    return std::make_unique<SiftCPUFeatureMatcher>(options);
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    ThrowCheckFeatureTypesMatch(image1, image2);

    matches->clear();

    if (!options_.sift->cpu_brute_force_matcher &&
        (prev_image_id1_ == kInvalidImageId ||
         prev_image_id1_ != image1.image_id)) {
      index1_ = options_.sift->cpu_descriptor_index_cache->Get(image1.image_id);
      prev_image_id1_ = image1.image_id;
    }

    if (!options_.sift->cpu_brute_force_matcher &&
        (prev_image_id2_ == kInvalidImageId ||
         prev_image_id2_ != image2.image_id)) {
      index2_ = options_.sift->cpu_descriptor_index_cache->Get(image2.image_id);
      prev_image_id2_ = image2.image_id;
    }

    if (image1.descriptors->data.rows() == 0 ||
        image2.descriptors->data.rows() == 0) {
      return;
    }

    if (options_.sift->cpu_brute_force_matcher) {
      const Eigen::RowMajorMatrixXf dot_products =
          ComputeSiftDistanceMatrix(DistanceType::DOT_PRODUCT,
                                    image1.descriptors->data,
                                    image2.descriptors->data,
                                    nullptr);
      FindBestMatchesBruteForce(dot_products,
                                options_.sift->max_ratio,
                                options_.sift->max_distance,
                                options_.sift->cross_check,
                                matches);
      return;
    }

    Eigen::RowMajorMatrixXi indices_1to2;
    Eigen::RowMajorMatrixXf l2_dists_1to2;
    Eigen::RowMajorMatrixXi indices_2to1;
    Eigen::RowMajorMatrixXf l2_dists_2to1;

    THROW_CHECK_NOTNULL(index2_)->Search(
        /*num_neighbors=*/2,
        image1.descriptors->ToFloat(),
        indices_1to2,
        l2_dists_1to2);
    if (options_.sift->cross_check) {
      THROW_CHECK_NOTNULL(index1_)->Search(
          /*num_neighbors=*/2,
          image2.descriptors->ToFloat(),
          indices_2to1,
          l2_dists_2to1);
    }

    FindBestMatchesIndex(indices_1to2,
                         l2_dists_1to2,
                         indices_2to1,
                         l2_dists_2to1,
                         options_.sift->max_ratio,
                         options_.sift->max_distance,
                         options_.sift->cross_check,
                         matches);
  }

  void MatchGuided(const double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    THROW_CHECK_NOTNULL(two_view_geometry);
    ThrowCheckFeatureTypesMatch(image1, image2, /*check_keypoints=*/true);

    two_view_geometry->inlier_matches.clear();

    if (!options_.sift->cpu_brute_force_matcher &&
        (prev_image_id1_ == kInvalidImageId ||
         prev_image_id1_ != image1.image_id)) {
      index1_ = options_.sift->cpu_descriptor_index_cache->Get(image1.image_id);
      prev_image_id1_ = image1.image_id;
    }

    if (!options_.sift->cpu_brute_force_matcher &&
        (prev_image_id2_ == kInvalidImageId ||
         prev_image_id2_ != image2.image_id)) {
      index2_ = options_.sift->cpu_descriptor_index_cache->Get(image2.image_id);
      prev_image_id2_ = image2.image_id;
    }

    const bool use_essential_matrix =
        UseEssentialMatrixForGuidedMatching(*two_view_geometry);
    const bool use_fundamental_matrix =
        !use_essential_matrix &&
        two_view_geometry->config == TwoViewGeometry::UNCALIBRATED &&
        two_view_geometry->F.has_value();
    const bool use_homography =
        (two_view_geometry->config == TwoViewGeometry::PLANAR ||
         two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
         two_view_geometry->config == TwoViewGeometry::PLANAR_OR_PANORAMIC) &&
        two_view_geometry->H.has_value();
    const Camera& effective_camera1 = two_view_geometry->camera1.has_value()
                                          ? *two_view_geometry->camera1
                                          : *image1.camera;
    const Camera& effective_camera2 = two_view_geometry->camera2.has_value()
                                          ? *two_view_geometry->camera2
                                          : *image2.camera;
    if (use_homography &&
        UseRayHomographyForGuidedMatching(*two_view_geometry)) {
      MatchGuidedSiftFeaturesWithRayHomography(
          max_error,
          effective_camera1,
          *image1.keypoints,
          image1.descriptors->data,
          effective_camera2,
          *image2.keypoints,
          image2.descriptors->data,
          RayHomographyForGuidedMatching(
              *two_view_geometry->H, effective_camera1, effective_camera2),
          *options_.sift,
          std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max(),
          &two_view_geometry->inlier_matches);
      return;
    }

    // The essential matrix path scores in pixels with the tangent Sampson
    // error, matching the two-view verification that produced E. Bearings are
    // used rather than normalized image plane coordinates so that the filter is
    // defined for every central camera model, including omnidirectional ones
    // whose back hemisphere has no image plane representation at all.
    const CamRaysWithJac cam_rays1 =
        use_essential_matrix
            ? ComputeCamRaysWithJac(effective_camera1, *image1.keypoints)
            : CamRaysWithJac();
    const CamRaysWithJac cam_rays2 =
        use_essential_matrix
            ? ComputeCamRaysWithJac(effective_camera2, *image2.keypoints)
            : CamRaysWithJac();

    const Eigen::Matrix3d E =
        use_essential_matrix ? *two_view_geometry->E : Eigen::Matrix3d::Zero();
    const Eigen::Matrix3f F =
        use_fundamental_matrix
            ? Eigen::Matrix3f(two_view_geometry->F->cast<float>())
            : Eigen::Matrix3f::Zero();
    const Eigen::Matrix3f H =
        use_homography ? Eigen::Matrix3f(two_view_geometry->H->cast<float>())
                       : Eigen::Matrix3f::Zero();

    // Both thresholds must outlive the lambdas below, which capture by
    // reference and are invoked after this scope's inner blocks have exited.
    const double max_residual_double = max_error * max_error;
    const float max_residual = static_cast<float>(max_residual_double);

    std::function<bool(Eigen::Index, Eigen::Index)> guided_filter;
    if (use_essential_matrix) {
      guided_filter = [&](const Eigen::Index i1, const Eigen::Index i2) {
        if (!cam_rays1.valid[i1] || !cam_rays2.valid[i2]) {
          return true;
        }
        return ComputeSquaredTangentSampsonError(cam_rays1.rays[i1],
                                                 cam_rays1.jacobians[i1],
                                                 cam_rays2.rays[i2],
                                                 cam_rays2.jacobians[i2],
                                                 E) > max_residual_double;
      };
    } else if (use_fundamental_matrix) {
      guided_filter = [&](const Eigen::Index i1, const Eigen::Index i2) {
        const auto& keypoint1 = (*image1.keypoints)[i1];
        const auto& keypoint2 = (*image2.keypoints)[i2];
        const Eigen::Vector3f p1(keypoint1.x, keypoint1.y, 1.0f);
        const Eigen::Vector3f p2(keypoint2.x, keypoint2.y, 1.0f);
        const Eigen::Vector3f epipolar_line1 = F * p1;
        const Eigen::Vector3f epipolar_line2 = F.transpose() * p2;
        const float nom = p2.transpose() * epipolar_line1;
        const float denom_sq = epipolar_line1(0) * epipolar_line1(0) +
                               epipolar_line1(1) * epipolar_line1(1) +
                               epipolar_line2(0) * epipolar_line2(0) +
                               epipolar_line2(1) * epipolar_line2(1);
        return nom * nom > max_residual * denom_sq;
      };
    } else if (use_homography) {
      guided_filter = [&](const Eigen::Index i1, const Eigen::Index i2) {
        const auto& keypoint1 = (*image1.keypoints)[i1];
        const auto& keypoint2 = (*image2.keypoints)[i2];
        const Eigen::Vector3f p1(keypoint1.x, keypoint1.y, 1.0f);
        const Eigen::Vector2f p2(keypoint2.x, keypoint2.y);
        return ((H * p1).hnormalized() - p2).squaredNorm() > max_residual;
      };
    } else {
      return;
    }

    THROW_CHECK(guided_filter);
    // The guided filter indexes per-feature geometry (bearings with Jacobians,
    // or the keypoints themselves) by descriptor row, so the two must align.
    THROW_CHECK_EQ(image1.keypoints->size(), image1.descriptors->data.rows());
    THROW_CHECK_EQ(image2.keypoints->size(), image2.descriptors->data.rows());

    const Eigen::RowMajorMatrixXf l2_dists_1to2 =
        ComputeSiftDistanceMatrix(DistanceType::L2,
                                  image1.descriptors->data,
                                  image2.descriptors->data,
                                  guided_filter);
    const Eigen::RowMajorMatrixXf l2_dists_2to1 = l2_dists_1to2.transpose();

    Eigen::RowMajorMatrixXi indices_1to2(l2_dists_1to2.rows(),
                                         l2_dists_1to2.cols());
    for (int i = 0; i < indices_1to2.rows(); ++i) {
      indices_1to2.row(i) = Eigen::VectorXi::LinSpaced(
          indices_1to2.cols(), 0, indices_1to2.cols() - 1);
    }
    Eigen::RowMajorMatrixXi indices_2to1(l2_dists_1to2.cols(),
                                         l2_dists_1to2.rows());
    for (int i = 0; i < indices_2to1.rows(); ++i) {
      indices_2to1.row(i) = Eigen::VectorXi::LinSpaced(
          indices_2to1.cols(), 0, indices_2to1.cols() - 1);
    }

    FindBestMatchesIndex(indices_1to2,
                         l2_dists_1to2,
                         indices_2to1,
                         l2_dists_2to1,
                         options_.sift->max_ratio,
                         options_.sift->max_distance,
                         options_.sift->cross_check,
                         &two_view_geometry->inlier_matches);
  }

 private:
  const FeatureMatchingOptions options_;
  image_t prev_image_id1_ = kInvalidImageId;
  image_t prev_image_id2_ = kInvalidImageId;
  std::shared_ptr<FeatureDescriptorIndex> index1_;
  std::shared_ptr<FeatureDescriptorIndex> index2_;
};

#if defined(COLMAP_METAL_ENABLED)
void ToMetalMatchKeypoints(
    const FeatureKeypoints& keypoints,
    std::vector<sift_metal::MatchKeypoint>* metal_keypoints) {
  THROW_CHECK_NOTNULL(metal_keypoints);
  metal_keypoints->resize(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    (*metal_keypoints)[i] = {keypoints[i].x, keypoints[i].y};
  }
}

void ToMetalMatchCamRaysWithJac(
    const Camera& camera,
    const FeatureKeypoints& keypoints,
    std::vector<sift_metal::MatchCamRayWithJac>* metal_cam_rays) {
  THROW_CHECK_NOTNULL(metal_cam_rays);
  metal_cam_rays->assign(keypoints.size(), sift_metal::MatchCamRayWithJac{});
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const FeatureKeypoint& keypoint = keypoints[i];
    const auto ray_with_jac =
        camera.CamRayFromImgWithJac(Eigen::Vector2d(keypoint.x, keypoint.y));
    if (!ray_with_jac.has_value()) {
      continue;
    }
    sift_metal::MatchCamRayWithJac& output = (*metal_cam_rays)[i];
    output.x = static_cast<float>(ray_with_jac->ray.x());
    output.y = static_cast<float>(ray_with_jac->ray.y());
    output.z = static_cast<float>(ray_with_jac->ray.z());
    output.jacobian_col0_x = static_cast<float>(ray_with_jac->jacobian(0, 0));
    output.jacobian_col0_y = static_cast<float>(ray_with_jac->jacobian(1, 0));
    output.jacobian_col0_z = static_cast<float>(ray_with_jac->jacobian(2, 0));
    output.jacobian_col1_x = static_cast<float>(ray_with_jac->jacobian(0, 1));
    output.jacobian_col1_y = static_cast<float>(ray_with_jac->jacobian(1, 1));
    output.jacobian_col1_z = static_cast<float>(ray_with_jac->jacobian(2, 1));
  }
}

std::array<float, 9> ToRowMajorArray(const Eigen::Matrix3f& matrix) {
  std::array<float, 9> data;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      data[3 * r + c] = matrix(r, c);
    }
  }
  return data;
}

void ConvertMetalMatches(const std::vector<sift_metal::MatchResult>& input,
                         FeatureMatches* output) {
  output->clear();
  output->reserve(input.size());
  for (const auto& match : input) {
    output->push_back(FeatureMatch{match.index1, match.index2});
  }
}

class SiftMetalFeatureMatcher : public FeatureMatcher {
 public:
  explicit SiftMetalFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options) {
    THROW_CHECK(options_.sift->Check());
  }

  static std::unique_ptr<FeatureMatcher> Create(
      const FeatureMatchingOptions& options) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    auto matcher = std::make_unique<SiftMetalFeatureMatcher>(options);
    if (!matcher->matcher_.Init()) {
      LOG(ERROR) << "Failed to initialize Metal SIFT feature matcher";
      return nullptr;
    }
    return matcher;
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    ThrowCheckFeatureTypesMatch(image1, image2);

    matches->clear();
    if (image1.descriptors->data.rows() == 0 ||
        image2.descriptors->data.rows() == 0) {
      return;
    }

    const sift_metal::MatchOptions match_options{
        static_cast<float>(options_.sift->max_ratio),
        static_cast<float>(options_.sift->max_distance),
        options_.sift->cross_check};
    if (!matcher_.Match(image1.descriptors->data.data(),
                        image1.descriptors->data.rows(),
                        image2.descriptors->data.data(),
                        image2.descriptors->data.rows(),
                        match_options,
                        &metal_matches_)) {
      LOG(ERROR) << "Metal SIFT feature matching failed";
      return;
    }
    ConvertMetalMatches(metal_matches_, matches);
  }

  void MatchGuided(const double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    THROW_CHECK_NOTNULL(two_view_geometry);
    ThrowCheckFeatureTypesMatch(image1, image2, /*check_keypoints=*/true);

    two_view_geometry->inlier_matches.clear();
    if (image1.descriptors->data.rows() == 0 ||
        image2.descriptors->data.rows() == 0) {
      return;
    }

    const bool use_essential_matrix =
        UseEssentialMatrixForGuidedMatching(*two_view_geometry);
    const bool use_fundamental_matrix =
        !use_essential_matrix &&
        two_view_geometry->config == TwoViewGeometry::UNCALIBRATED &&
        two_view_geometry->F.has_value();
    const bool use_homography =
        (two_view_geometry->config == TwoViewGeometry::PLANAR ||
         two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
         two_view_geometry->config == TwoViewGeometry::PLANAR_OR_PANORAMIC) &&
        two_view_geometry->H.has_value();
    const Camera& effective_camera1 = two_view_geometry->camera1.has_value()
                                          ? *two_view_geometry->camera1
                                          : *image1.camera;
    const Camera& effective_camera2 = two_view_geometry->camera2.has_value()
                                          ? *two_view_geometry->camera2
                                          : *image2.camera;

    if (!use_essential_matrix && !use_fundamental_matrix && !use_homography) {
      return;
    }
    if (use_homography &&
        UseRayHomographyForGuidedMatching(*two_view_geometry)) {
      MatchGuidedSiftFeaturesWithRayHomography(
          max_error,
          effective_camera1,
          *image1.keypoints,
          image1.descriptors->data,
          effective_camera2,
          *image2.keypoints,
          image2.descriptors->data,
          RayHomographyForGuidedMatching(
              *two_view_geometry->H, effective_camera1, effective_camera2),
          *options_.sift,
          std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max(),
          &two_view_geometry->inlier_matches);
      return;
    }

    const Eigen::Matrix3f E_or_F =
        use_essential_matrix
            ? Eigen::Matrix3f(two_view_geometry->E->cast<float>())
        : use_fundamental_matrix
            ? Eigen::Matrix3f(two_view_geometry->F->cast<float>())
            : Eigen::Matrix3f::Zero();
    const Eigen::Matrix3f H =
        use_homography ? Eigen::Matrix3f(two_view_geometry->H->cast<float>())
                       : Eigen::Matrix3f::Zero();
    const Eigen::Matrix3f guided_matrix = use_homography ? H : E_or_F;

    const float max_residual = static_cast<float>(max_error * max_error);
    if (!guided_matrix.allFinite() || !std::isfinite(max_residual) ||
        max_residual < 0.0f) {
      return;
    }

    const std::array<float, 9> matrix = ToRowMajorArray(guided_matrix);

    const sift_metal::MatchOptions match_options{
        static_cast<float>(options_.sift->max_ratio),
        static_cast<float>(options_.sift->max_distance),
        options_.sift->cross_check};
    bool matched = false;
    if (use_essential_matrix) {
      const auto& cam_rays1 =
          GetTangentMetalCamRays(image1, effective_camera1, &metal_cam_rays1_);
      const auto& cam_rays2 =
          GetTangentMetalCamRays(image2, effective_camera2, &metal_cam_rays2_);
      matched = matcher_.MatchGuidedTangent(image1.descriptors->data.data(),
                                            image1.descriptors->data.rows(),
                                            cam_rays1.data(),
                                            image2.descriptors->data.data(),
                                            image2.descriptors->data.rows(),
                                            cam_rays2.data(),
                                            match_options,
                                            matrix.data(),
                                            max_residual,
                                            &metal_matches_);
    } else {
      const auto& metal_keypoints1 =
          GetRawMetalKeypoints(image1, &metal_keypoints1_);
      const auto& metal_keypoints2 =
          GetRawMetalKeypoints(image2, &metal_keypoints2_);
      const sift_metal::MatchGuidedGeometry guided_geometry =
          use_homography ? sift_metal::MatchGuidedGeometry::HOMOGRAPHY
                         : sift_metal::MatchGuidedGeometry::EPIPOLAR;
      matched = matcher_.MatchGuided(image1.descriptors->data.data(),
                                     image1.descriptors->data.rows(),
                                     metal_keypoints1.data(),
                                     image2.descriptors->data.data(),
                                     image2.descriptors->data.rows(),
                                     metal_keypoints2.data(),
                                     match_options,
                                     guided_geometry,
                                     matrix.data(),
                                     max_residual,
                                     &metal_matches_);
    }
    if (!matched) {
      LOG(ERROR) << "Metal guided SIFT feature matching failed";
      return;
    }
    ConvertMetalMatches(metal_matches_, &two_view_geometry->inlier_matches);
  }

 private:
  struct MetalKeypointCacheEntry {
    const FeatureKeypoints* source_keypoints = nullptr;
    size_t num_keypoints = 0;
    uint64_t keypoints_hash = 0;
    uint64_t camera_hash = 0;
    std::vector<sift_metal::MatchKeypoint> keypoints;
  };

  struct MetalCamRayCacheEntry {
    const FeatureKeypoints* source_keypoints = nullptr;
    size_t num_keypoints = 0;
    uint64_t keypoints_hash = 0;
    uint64_t camera_hash = 0;
    std::vector<sift_metal::MatchCamRayWithJac> cam_rays;
  };

  const std::vector<sift_metal::MatchKeypoint>& GetRawMetalKeypoints(
      const Image& image, std::vector<sift_metal::MatchKeypoint>* scratch) {
    if (image.image_id == kInvalidImageId) {
      ToMetalMatchKeypoints(*image.keypoints, scratch);
      return *scratch;
    }

    MetalKeypointCacheEntry& entry = raw_metal_keypoint_cache_[image.image_id];
    const FeatureKeypoints* keypoints = image.keypoints.get();
    const uint64_t keypoints_hash = HashFeatureKeypoints(*keypoints);
    if (entry.source_keypoints != keypoints ||
        entry.num_keypoints != keypoints->size() ||
        entry.keypoints_hash != keypoints_hash) {
      ToMetalMatchKeypoints(*keypoints, &entry.keypoints);
      entry.source_keypoints = keypoints;
      entry.num_keypoints = keypoints->size();
      entry.keypoints_hash = keypoints_hash;
      entry.camera_hash = 0;
    }
    return entry.keypoints;
  }

  const std::vector<sift_metal::MatchCamRayWithJac>& GetTangentMetalCamRays(
      const Image& image,
      const Camera& camera,
      std::vector<sift_metal::MatchCamRayWithJac>* scratch) {
    if (image.image_id == kInvalidImageId) {
      ToMetalMatchCamRaysWithJac(camera, *image.keypoints, scratch);
      return *scratch;
    }

    MetalCamRayCacheEntry& entry =
        tangent_metal_keypoint_cache_[image.image_id];
    const FeatureKeypoints* keypoints = image.keypoints.get();
    const uint64_t keypoints_hash = HashFeatureKeypoints(*keypoints);
    const uint64_t camera_hash = HashCameraForNormalization(camera);
    if (entry.source_keypoints != keypoints ||
        entry.num_keypoints != keypoints->size() ||
        entry.keypoints_hash != keypoints_hash ||
        entry.camera_hash != camera_hash) {
      ToMetalMatchCamRaysWithJac(camera, *image.keypoints, &entry.cam_rays);
      entry.source_keypoints = keypoints;
      entry.num_keypoints = keypoints->size();
      entry.keypoints_hash = keypoints_hash;
      entry.camera_hash = camera_hash;
    }
    return entry.cam_rays;
  }

  const FeatureMatchingOptions options_;
  sift_metal::SiftMetalMatcher matcher_;
  std::vector<sift_metal::MatchResult> metal_matches_;
  std::vector<sift_metal::MatchKeypoint> metal_keypoints1_;
  std::vector<sift_metal::MatchKeypoint> metal_keypoints2_;
  std::vector<sift_metal::MatchCamRayWithJac> metal_cam_rays1_;
  std::vector<sift_metal::MatchCamRayWithJac> metal_cam_rays2_;
  std::map<image_t, MetalKeypointCacheEntry> raw_metal_keypoint_cache_;
  std::map<image_t, MetalCamRayCacheEntry> tangent_metal_keypoint_cache_;
};
#endif  // COLMAP_METAL_ENABLED

#if defined(COLMAP_GPU_ENABLED) && !defined(COLMAP_METAL_ENABLED)

// Number of floats per feature in the SiftGPU bearing-plus-Jacobian layout:
// the unit bearing followed by the two columns of d(bearing) / d(pixel).
constexpr int kNumCamRayWithJacElems = 9;

// Pack bearings and unprojection Jacobians for the SiftGPU tangent Sampson
// kernel. Keypoints that cannot be unprojected are zeroed, which makes both the
// numerator and the denominator of the residual vanish; the kernel treats a
// zero denominator as "no geometric information" and rejects the pair.
std::vector<float> PackCamRaysWithJac(const Camera& camera,
                                      const FeatureKeypoints& keypoints) {
  std::vector<float> packed(keypoints.size() * kNumCamRayWithJacElems, 0.0f);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const FeatureKeypoint& keypoint = keypoints[i];
    const auto ray_and_jac =
        camera.CamRayFromImgWithJac(Eigen::Vector2d(keypoint.x, keypoint.y));
    if (!ray_and_jac) {
      continue;
    }
    float* out = packed.data() + i * kNumCamRayWithJacElems;
    for (int k = 0; k < 3; ++k) {
      out[k] = static_cast<float>(ray_and_jac->ray(k));
      out[3 + k] = static_cast<float>(ray_and_jac->jacobian(k, 0));
      out[6 + k] = static_cast<float>(ray_and_jac->jacobian(k, 1));
    }
  }
  return packed;
}

// Mutexes for OpenGL version to protect static variables in SiftGPU.
// CUDA version doesn't need this as it has its own thread safety.
static std::map<int, std::unique_ptr<std::mutex>> sift_opengl_mutexes_;

enum class SiftBackend { CUDA, GLSL };

class SiftGPUFeatureMatcher : public FeatureMatcher {
 public:
  explicit SiftGPUFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options), backend_(SiftBackend::GLSL) {
    THROW_CHECK(options_.sift->Check());
  }

  static std::unique_ptr<FeatureMatcher> Create(
      const FeatureMatchingOptions& options) {
    // SiftGPU uses many global static state variables and the initialization
    // must be thread-safe in order to work correctly. This is enforced here.
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
    THROW_CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

    SiftGPU sift_gpu;
    sift_gpu.SetVerbose(0);

    auto matcher = std::make_unique<SiftGPUFeatureMatcher>(options);

    // Note that the SiftMatchGPU object is not movable (for whatever reason).
    // If we instead create the object here and move it to the constructor, the
    // program segfaults inside SiftMatchGPU.

    matcher->sift_match_gpu_ = SiftMatchGPU(options.max_num_matches);

#if defined(COLMAP_CUDA_ENABLED)
    if (gpu_indices[0] >= 0) {
      matcher->sift_match_gpu_.SetLanguage(
          SiftMatchGPU::SIFTMATCH_CUDA_DEVICE0 + gpu_indices[0]);
    } else {
      matcher->sift_match_gpu_.SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
    }
    matcher->backend_ = SiftBackend::CUDA;
#else   // COLMAP_CUDA_ENABLED
    matcher->sift_match_gpu_.SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
    matcher->backend_ = SiftBackend::GLSL;
#endif  // COLMAP_CUDA_ENABLED

    if (matcher->sift_match_gpu_.VerifyContextGL() == 0) {
      return nullptr;
    }

    if (!matcher->sift_match_gpu_.Allocate(options.max_num_matches,
                                           options.sift->cross_check)) {
      LOG(ERROR) << StringPrintf(
          "Not enough GPU memory to match %d features. "
          "Reduce the maximum number of matches.",
          options.max_num_matches);
      return nullptr;
    }

#if !defined(COLMAP_CUDA_ENABLED)
    if (matcher->sift_match_gpu_.GetMaxSift() < options.max_num_matches) {
      LOG(WARNING) << StringPrintf(
          "OpenGL version of SiftGPU only supports a "
          "maximum of %d matches - consider changing to CUDA-based "
          "feature matching to avoid this limitation.",
          matcher->sift_match_gpu_.GetMaxSift());
    }
#endif  // COLMAP_CUDA_ENABLED

    matcher->sift_match_gpu_.gpu_index = gpu_indices[0];

    // Initialize mutex for OpenGL backend regardless of compile-time flags
    if (const auto it = sift_opengl_mutexes_.find(gpu_indices[0]);
        it == sift_opengl_mutexes_.end()) {
      sift_opengl_mutexes_.emplace_hint(
          it, gpu_indices[0], std::make_unique<std::mutex>());
    }

    return matcher;
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    ThrowCheckFeatureTypesMatch(image1, image2);

    matches->clear();

    // Protect OpenGL operations with global mutex based on runtime backend
    std::unique_lock<std::mutex> lock;
    if (backend_ == SiftBackend::GLSL) {
      lock = std::unique_lock<std::mutex>(
          *sift_opengl_mutexes_.at(sift_match_gpu_.gpu_index));
    }

    const uint64_t descriptors_hash1 =
        HashFeatureDescriptors(*image1.descriptors);
    const uint64_t descriptors_hash2 =
        HashFeatureDescriptors(*image2.descriptors);

    if (prev_image_id1_ == kInvalidImageId || prev_is_guided_ ||
        prev_image_id1_ != image1.image_id ||
        prev_descriptors1_ != image1.descriptors.get() ||
        prev_descriptors_hash1_ != descriptors_hash1) {
      WarnIfMaxNumMatchesReachedGPU(image1.descriptors->data);
      sift_match_gpu_.SetDescriptors(
          0, image1.descriptors->data.rows(), image1.descriptors->data.data());
      prev_image_id1_ = image1.image_id;
      prev_descriptors1_ = image1.descriptors.get();
      prev_descriptors_hash1_ = descriptors_hash1;
    }

    if (prev_image_id2_ == kInvalidImageId || prev_is_guided_ ||
        prev_image_id2_ != image2.image_id ||
        prev_descriptors2_ != image2.descriptors.get() ||
        prev_descriptors_hash2_ != descriptors_hash2) {
      WarnIfMaxNumMatchesReachedGPU(image2.descriptors->data);
      sift_match_gpu_.SetDescriptors(
          1, image2.descriptors->data.rows(), image2.descriptors->data.data());
      prev_image_id2_ = image2.image_id;
      prev_descriptors2_ = image2.descriptors.get();
      prev_descriptors_hash2_ = descriptors_hash2;
    }

    prev_is_guided_ = false;

    matches->resize(static_cast<size_t>(options_.max_num_matches));

    const int num_matches = sift_match_gpu_.GetSiftMatch(
        options_.max_num_matches,
        reinterpret_cast<uint32_t (*)[2]>(matches->data()),
        static_cast<float>(options_.sift->max_distance),
        static_cast<float>(options_.sift->max_ratio),
        options_.sift->cross_check);

    if (num_matches < 0) {
      LOG(ERROR) << "Feature matching failed. This is probably caused by "
                    "insufficient GPU memory. Consider reducing the maximum "
                    "number of features and/or matches.";
      matches->clear();
    } else {
      THROW_CHECK_LE(num_matches, matches->size());
      matches->resize(num_matches);
    }
  }

  void MatchGuided(const double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    static_assert(offsetof(FeatureKeypoint, x) == 0 * sizeof(float),
                  "Invalid keypoint format");
    static_assert(offsetof(FeatureKeypoint, y) == 1 * sizeof(float),
                  "Invalid keypoint format");
    static_assert(sizeof(FeatureKeypoint) == 6 * sizeof(float),
                  "Invalid keypoint format");

    THROW_CHECK_NOTNULL(two_view_geometry);
    ThrowCheckFeatureTypesMatch(image1, image2, /*check_keypoints=*/true);

    two_view_geometry->inlier_matches.clear();

    constexpr size_t kFeatureShapeNumElems = 4;

    // For calibrated cases, use the essential matrix with normalized
    // coordinates. This properly handles non-pinhole camera models (with
    // distortion) where the fundamental matrix relationship doesn't hold. The
    // essential matrix is also used for UNCALIBRATED pairs that carry
    // solver-estimated intrinsics (see UseEssentialMatrixForGuidedMatching).
    const bool use_essential_matrix =
        UseEssentialMatrixForGuidedMatching(*two_view_geometry);
    const bool use_fundamental_matrix =
        !use_essential_matrix &&
        two_view_geometry->config == TwoViewGeometry::UNCALIBRATED &&
        two_view_geometry->F.has_value();
    const bool use_homography =
        (two_view_geometry->config == TwoViewGeometry::PLANAR ||
         two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
         two_view_geometry->config == TwoViewGeometry::PLANAR_OR_PANORAMIC) &&
        two_view_geometry->H.has_value();
    // Normalize with the intrinsics the solver estimated where available (its
    // focal, not the camera's stale default), else the given cameras.
    const Camera& effective_camera1 = two_view_geometry->camera1.has_value()
                                          ? *two_view_geometry->camera1
                                          : *image1.camera;
    const Camera& effective_camera2 = two_view_geometry->camera2.has_value()
                                          ? *two_view_geometry->camera2
                                          : *image2.camera;

    if (use_homography &&
        UseRayHomographyForGuidedMatching(*two_view_geometry)) {
      MatchGuidedSiftFeaturesWithRayHomography(
          max_error,
          effective_camera1,
          *image1.keypoints,
          image1.descriptors->data,
          effective_camera2,
          *image2.keypoints,
          image2.descriptors->data,
          RayHomographyForGuidedMatching(
              *two_view_geometry->H, effective_camera1, effective_camera2),
          *options_.sift,
          sift_match_gpu_.GetMaxSift(),
          options_.max_num_matches,
          &two_view_geometry->inlier_matches);
      return;
    }

    // Protect only OpenGL operations with the global per-device mutex. The
    // ray-homography fallback above is host-only and may be comparatively
    // expensive, so it must not serialize unrelated GLSL work.
    std::unique_lock<std::mutex> lock;
    if (backend_ == SiftBackend::GLSL) {
      lock = std::unique_lock<std::mutex>(
          *sift_opengl_mutexes_.at(sift_match_gpu_.gpu_index));
    }

    const uint64_t descriptors_hash1 =
        HashFeatureDescriptors(*image1.descriptors);
    const uint64_t descriptors_hash2 =
        HashFeatureDescriptors(*image2.descriptors);
    const uint64_t keypoints_hash1 = HashFeatureKeypoints(*image1.keypoints);
    const uint64_t keypoints_hash2 = HashFeatureKeypoints(*image2.keypoints);
    const uint64_t camera_hash1 =
        use_essential_matrix ? HashCameraForNormalization(effective_camera1)
                             : 0;
    const uint64_t camera_hash2 =
        use_essential_matrix ? HashCameraForNormalization(effective_camera2)
                             : 0;

    if (prev_image_id1_ == kInvalidImageId || !prev_is_guided_ ||
        prev_image_id1_ != image1.image_id ||
        prev_descriptors1_ != image1.descriptors.get() ||
        prev_keypoints1_ != image1.keypoints.get() ||
        prev_descriptors_hash1_ != descriptors_hash1 ||
        prev_keypoints_hash1_ != keypoints_hash1 ||
        prev_camera_hash1_ != camera_hash1 ||
        use_essential_matrix != prev_use_essential_matrix_) {
      WarnIfMaxNumMatchesReachedGPU(image1.descriptors->data);
      constexpr size_t kIndex = 0;
      sift_match_gpu_.SetDescriptors(kIndex,
                                     image1.descriptors->data.rows(),
                                     image1.descriptors->data.data());
      if (use_essential_matrix) {
        const std::vector<float> cam_rays1 =
            PackCamRaysWithJac(effective_camera1, *image1.keypoints);
        sift_match_gpu_.SetFeatureLocation(kIndex,
                                           cam_rays1.data(),
                                           /*gap=*/0,
                                           kNumCamRayWithJacElems);
      } else {
        sift_match_gpu_.SetFeatureLocation(
            kIndex,
            reinterpret_cast<const float*>(image1.keypoints->data()),
            kFeatureShapeNumElems);
      }
      prev_image_id1_ = image1.image_id;
      prev_descriptors1_ = image1.descriptors.get();
      prev_keypoints1_ = image1.keypoints.get();
      prev_descriptors_hash1_ = descriptors_hash1;
      prev_keypoints_hash1_ = keypoints_hash1;
      prev_camera_hash1_ = camera_hash1;
    }

    if (prev_image_id2_ == kInvalidImageId || !prev_is_guided_ ||
        prev_image_id2_ != image2.image_id ||
        prev_descriptors2_ != image2.descriptors.get() ||
        prev_keypoints2_ != image2.keypoints.get() ||
        prev_descriptors_hash2_ != descriptors_hash2 ||
        prev_keypoints_hash2_ != keypoints_hash2 ||
        prev_camera_hash2_ != camera_hash2 ||
        use_essential_matrix != prev_use_essential_matrix_) {
      WarnIfMaxNumMatchesReachedGPU(image2.descriptors->data);
      constexpr size_t kIndex = 1;
      sift_match_gpu_.SetDescriptors(kIndex,
                                     image2.descriptors->data.rows(),
                                     image2.descriptors->data.data());
      if (use_essential_matrix) {
        const std::vector<float> cam_rays2 =
            PackCamRaysWithJac(effective_camera2, *image2.keypoints);
        sift_match_gpu_.SetFeatureLocation(kIndex,
                                           cam_rays2.data(),
                                           /*gap=*/0,
                                           kNumCamRayWithJacElems);
      } else {
        sift_match_gpu_.SetFeatureLocation(
            kIndex,
            reinterpret_cast<const float*>(image2.keypoints->data()),
            kFeatureShapeNumElems);
      }
      prev_image_id2_ = image2.image_id;
      prev_descriptors2_ = image2.descriptors.get();
      prev_keypoints2_ = image2.keypoints.get();
      prev_descriptors_hash2_ = descriptors_hash2;
      prev_keypoints_hash2_ = keypoints_hash2;
      prev_camera_hash2_ = camera_hash2;
    }

    prev_is_guided_ = true;
    prev_use_essential_matrix_ = use_essential_matrix;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> E_or_F;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H;
    float* E_or_F_ptr = nullptr;
    float* H_ptr = nullptr;
    if (use_essential_matrix) {
      // Use essential matrix with normalized coordinates.
      E_or_F = two_view_geometry->E->cast<float>();
      E_or_F_ptr = E_or_F.data();
    } else if (use_fundamental_matrix) {
      // Use fundamental matrix with pixel coordinates.
      E_or_F = two_view_geometry->F->cast<float>();
      E_or_F_ptr = E_or_F.data();
    } else if (use_homography) {
      H = two_view_geometry->H->cast<float>();
      H_ptr = H.data();
    } else {
      return;
    }

    THROW_CHECK(E_or_F_ptr != nullptr || H_ptr != nullptr);

    two_view_geometry->inlier_matches.resize(
        static_cast<size_t>(options_.max_num_matches));

    // Every config scores in pixels: the tangent Sampson error for E, the
    // pixel Sampson error for F, and the pixel transfer error for H.
    const float max_residual = static_cast<float>(max_error * max_error);

    const int num_matches = sift_match_gpu_.GetGuidedSiftMatch(
        options_.max_num_matches,
        reinterpret_cast<uint32_t (*)[2]>(
            two_view_geometry->inlier_matches.data()),
        H_ptr,
        E_or_F_ptr,
        static_cast<float>(options_.sift->max_distance),
        static_cast<float>(options_.sift->max_ratio),
        max_residual,
        max_residual,
        options_.sift->cross_check);

    if (num_matches < 0) {
      LOG(ERROR) << "Feature matching failed. This is probably caused by "
                    "insufficient GPU memory. Consider reducing the maximum "
                    "number of features.";
      two_view_geometry->inlier_matches.clear();
    } else {
      THROW_CHECK_LE(num_matches, two_view_geometry->inlier_matches.size());
      two_view_geometry->inlier_matches.resize(num_matches);
    }
  }

 private:
  void WarnIfMaxNumMatchesReachedGPU(
      const FeatureDescriptorsData& descriptors) {
    if (sift_match_gpu_.GetMaxSift() < descriptors.rows()) {
      LOG(WARNING) << StringPrintf(
          "Clamping features from %d to %d - consider "
          "increasing the maximum number of matches.",
          descriptors.rows(),
          sift_match_gpu_.GetMaxSift());
    }
  }

  const FeatureMatchingOptions options_;
  SiftMatchGPU sift_match_gpu_;
  SiftBackend backend_;
  bool prev_is_guided_ = false;
  bool prev_use_essential_matrix_ = false;
  image_t prev_image_id1_ = kInvalidImageId;
  image_t prev_image_id2_ = kInvalidImageId;
  // Data uploaded for the previous image ids. An image may be re-posted with
  // the same id but updated keypoints/descriptors, which must be re-uploaded.
  const FeatureKeypoints* prev_keypoints1_ = nullptr;
  const FeatureKeypoints* prev_keypoints2_ = nullptr;
  const FeatureDescriptors* prev_descriptors1_ = nullptr;
  const FeatureDescriptors* prev_descriptors2_ = nullptr;
  uint64_t prev_descriptors_hash1_ = 0;
  uint64_t prev_descriptors_hash2_ = 0;
  uint64_t prev_keypoints_hash1_ = 0;
  uint64_t prev_keypoints_hash2_ = 0;
  uint64_t prev_camera_hash1_ = 0;
  uint64_t prev_camera_hash2_ = 0;
};
#endif  // COLMAP_GPU_ENABLED && !COLMAP_METAL_ENABLED

}  // namespace

std::unique_ptr<FeatureMatcher> CreateSiftFeatureMatcher(
    const FeatureMatchingOptions& options) {
  THROW_CHECK_NOTNULL(options.sift);
  if (options.type == FeatureMatcherType::SIFT_LIGHTGLUE) {
    return CreateLightGlueONNXFeatureMatcher(options, options.sift->lightglue);
  } else if (options.type == FeatureMatcherType::SIFT_BRUTEFORCE) {
    if (options.use_gpu) {
#if defined(COLMAP_METAL_ENABLED)
      LOG(INFO) << "Creating SIFT Metal GPU feature matcher";
      return SiftMetalFeatureMatcher::Create(options);
#elif defined(COLMAP_GPU_ENABLED)
      LOG(INFO) << "Creating SIFT GPU feature matcher";
      return SiftGPUFeatureMatcher::Create(options);
#else
      return SiftCPUFeatureMatcher::Create(options);
#endif
    } else {
      LOG(INFO) << "Creating SIFT CPU feature matcher";
      return SiftCPUFeatureMatcher::Create(options);
    }
  } else {
    LOG(FATAL_THROW) << "Unknown SIFT feature matcher type: "
                     << FeatureMatcherTypeToString(options.type);
  }
  return nullptr;
}

void LoadSiftFeaturesFromTextFile(const std::filesystem::path& path,
                                  FeatureKeypoints* keypoints,
                                  FeatureDescriptors* descriptors) {
  THROW_CHECK_NOTNULL(keypoints);
  THROW_CHECK_NOTNULL(descriptors);

  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  file.imbue(std::locale::classic());

  std::string line;

  std::getline(file, line);
  std::istringstream header_line_stream(line);
  header_line_stream.imbue(std::locale::classic());

  point2D_t num_features;
  size_t dim;
  THROW_CHECK(header_line_stream >> num_features >> dim);

  THROW_CHECK_EQ(dim, kSiftDescriptorDim)
      << "SIFT features must have kSiftDescriptorDim dimensions";

  keypoints->resize(num_features);
  descriptors->data.resize(num_features, dim);
  descriptors->type = FeatureExtractorType::SIFT;

  for (size_t i = 0; i < num_features; ++i) {
    std::getline(file, line);
    std::istringstream feature_line_stream(line);
    feature_line_stream.imbue(std::locale::classic());

    float x, y, scale, orientation;
    THROW_CHECK(feature_line_stream >> x >> y >> scale >> orientation);

    (*keypoints)[i] = FeatureKeypoint(x, y, scale, orientation);

    // Descriptor
    for (size_t j = 0; j < dim; ++j) {
      float value;
      THROW_CHECK(feature_line_stream >> value);
      THROW_CHECK_GE(value, 0);
      THROW_CHECK_LE(value, 255);
      descriptors->data(i, j) = TruncateCast<float, uint8_t>(value);
    }
  }
}

}  //  namespace colmap
