// Copyright (c) 2026 Plinth. SPDX-License-Identifier: BSD-3-Clause
#include "ColmapSparse.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/utils.h"
#include "colmap/scene/database.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/rig.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "SiftMetal.h"
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {
struct Failure : std::runtime_error {
  cm_sparse_status status;
  Failure(cm_sparse_status value, const std::string& message)
      : std::runtime_error(message), status(value) {}
};

void Require(bool condition, cm_sparse_status status, const char* message) {
  if (!condition) throw Failure(status, message);
}

struct Input {
  fs::path path;
  std::string name;
  uint32_t calibration_id;
  colmap::Camera camera;
  colmap::image_t image_id = colmap::kInvalidImageId;
};

// One active job bounds process-wide SfM working memory, including callers that
// bypass the Swift facade. A second run fails promptly instead of queuing work.
std::mutex active_job_mutex;

colmap::Camera MakeCamera(const cm_sparse_camera& source) {
  colmap::Camera camera;
  size_t count = 0;
  switch (source.model) {
    case CM_SPARSE_PINHOLE:
      camera.model_id = colmap::CameraModelId::kPinhole;
      count = 4;
      break;
    case CM_SPARSE_SIMPLE_RADIAL:
      camera.model_id = colmap::CameraModelId::kSimpleRadial;
      count = 4;
      break;
    case CM_SPARSE_RADIAL:
      camera.model_id = colmap::CameraModelId::kRadial;
      count = 5;
      break;
    case CM_SPARSE_OPENCV:
      camera.model_id = colmap::CameraModelId::kOpenCV;
      count = 8;
      break;
    default:
      throw Failure(CM_SPARSE_INVALID_ARGUMENT, "Unsupported camera model");
  }
  camera.width = source.width;
  camera.height = source.height;
  camera.params.assign(source.params, source.params + count);
  camera.has_prior_focal_length = true;
  Require(source.width >= 32 && source.height >= 32 && source.width <= 8192 &&
              source.height <= 8192 && uint64_t(source.width) * source.height <= 16000000,
          CM_SPARSE_INVALID_ARGUMENT,
          "Encoded images must be 32..8192 pixels per axis and at most 16 MP");
  Require(camera.VerifyParams() &&
              std::all_of(camera.params.begin(),
                          camera.params.end(),
                          [](double value) { return std::isfinite(value); }) &&
              camera.FocalLengthX() > 0 && camera.FocalLengthY() > 0,
          CM_SPARSE_INVALID_ARGUMENT,
          "Invalid camera calibration");
  return camera;
}

void ValidateOptions(const cm_sparse_options& options) {
  Require(options.struct_size == sizeof(cm_sparse_options) &&
              options.abi_version == CM_SPARSE_ABI_VERSION,
          CM_SPARSE_INVALID_ARGUMENT,
          "Incompatible options ABI");
  Require(options.max_image_size >= 128 && options.max_image_size <= 1600 &&
              options.max_num_features >= 128 && options.max_num_features <= 8192 &&
              options.num_threads >= 1 && options.num_threads <= 4 &&
              options.sequential_overlap >= 1 && options.sequential_overlap <= 30 &&
              options.keyframe_stride <= 256 && options.max_images >= 2 &&
              options.max_images <= 256 && options.max_num_pairs >= 1 &&
              options.max_num_pairs <= 8192 && options.max_runtime_seconds >= 1 &&
              options.max_runtime_seconds <= 3600 &&
              (options.first_octave == -1 || options.first_octave == 0) &&
              options.matching_cache_bytes <= 64ull * 1024 * 1024 &&
              std::isfinite(options.minimum_registered_fraction) &&
              options.minimum_registered_fraction > 0 && options.minimum_registered_fraction <= 1 &&
              options.refine_intrinsics <= 1,
          CM_SPARSE_INVALID_ARGUMENT,
          "Options exceed the mobile pilot limits");
}

std::vector<std::pair<size_t, size_t>> MakePairs(size_t count, const cm_sparse_options& options) {
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = i + 1; j < std::min(count, i + options.sequential_overlap + 1); ++j)
      pairs.emplace_back(i, j);
  }
  if (options.keyframe_stride > 0) {
    std::vector<size_t> keyframes;
    for (size_t i = 0; i < count; i += options.keyframe_stride) keyframes.push_back(i);
    if (keyframes.back() != count - 1) keyframes.push_back(count - 1);
    for (size_t i = 0; i < keyframes.size(); ++i) {
      for (size_t j = i + 1; j < keyframes.size(); ++j) {
        if (keyframes[j] - keyframes[i] > options.sequential_overlap)
          pairs.emplace_back(keyframes[i], keyframes[j]);
      }
    }
  }
  Require(pairs.size() <= options.max_num_pairs,
          CM_SPARSE_INVALID_ARGUMENT,
          "Matching schedule exceeds max_num_pairs");
  return pairs;
}

struct StagingDirectory {
  fs::path path;
  explicit StagingDirectory(const fs::path& parent) {
    std::string pattern = (parent / ".colmap-sparse-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back(0);
    Require(mkdtemp(buffer.data()) != nullptr,
            CM_SPARSE_IO_ERROR,
            "Cannot create reconstruction working directory");
    path = buffer.data();
  }
  ~StagingDirectory() {
    // Even the error_code overload may allocate and throw. Cleanup must not
    // terminate the host while another exception crosses the native boundary.
    try {
      std::error_code ignored;
      fs::remove_all(path, ignored);
    } catch (...) {
    }
  }
};

// Inspect encoded dimensions before decoding. EXIF orientation is deliberately
// not applied: calibration and observations refer to the original raster.
void CheckRaster(const fs::path& path, const colmap::Camera& camera) {
  @autoreleasepool {
    NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    Require(source != nullptr, CM_SPARSE_IO_ERROR, "Cannot open encoded image");
    CFDictionaryRef properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nullptr);
    CFRelease(source);
    size_t width = 0, height = 0;
    if (properties) {
      NSDictionary* values = (__bridge NSDictionary*)properties;
      width = [values[(__bridge NSString*)kCGImagePropertyPixelWidth] unsignedLongLongValue];
      height = [values[(__bridge NSString*)kCGImagePropertyPixelHeight] unsignedLongLongValue];
      CFRelease(properties);
    }
    Require(width == camera.width && height == camera.height,
            CM_SPARSE_INVALID_ARGUMENT,
            "Encoded raster dimensions disagree with calibration");
  }
}

void ConvertMatches(const std::vector<sift_metal::MatchResult>& source,
                    colmap::FeatureMatches* output) {
  output->clear();
  output->reserve(source.size());
  for (const auto& match : source) output->push_back({match.index1, match.index2});
}

std::vector<sift_metal::MatchCamRayWithJac> MakeRays(const colmap::Camera& camera,
                                                     const colmap::FeatureKeypoints& keypoints) {
  std::vector<sift_metal::MatchCamRayWithJac> rays(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const auto ray = camera.CamRayFromImgWithJac({keypoints[i].x, keypoints[i].y});
    if (!ray) continue;
    rays[i] = {float(ray->ray.x()),
               float(ray->ray.y()),
               float(ray->ray.z()),
               float(ray->jacobian(0, 0)),
               float(ray->jacobian(1, 0)),
               float(ray->jacobian(2, 0)),
               float(ray->jacobian(0, 1)),
               float(ray->jacobian(1, 1)),
               float(ray->jacobian(2, 1))};
  }
  return rays;
}

void GuideMatches(sift_metal::SiftMetalMatcher& matcher,
                  const colmap::Camera& camera1,
                  const colmap::FeatureKeypoints& keys1,
                  const colmap::FeatureDescriptors& desc1,
                  const colmap::Camera& camera2,
                  const colmap::FeatureKeypoints& keys2,
                  const colmap::FeatureDescriptors& desc2,
                  double max_error,
                  colmap::TwoViewGeometry* geometry) {
  using G = colmap::TwoViewGeometry;
  const bool essential =
      geometry->E &&
      (geometry->config == G::CALIBRATED || geometry->config == G::CALIBRATED_RIG ||
       (geometry->config == G::UNCALIBRATED && (geometry->camera1 || geometry->camera2)));
  const bool fundamental = !essential && geometry->config == G::UNCALIBRATED && geometry->F;
  const bool homography =
      geometry->H && (geometry->config == G::PLANAR || geometry->config == G::PANORAMIC ||
                      geometry->config == G::PLANAR_OR_PANORAMIC);
  // Preserve verified inliers for ray homographies. The pixel Metal kernel
  // cannot evaluate those coordinates; the mobile pilot does not expand them.
  if ((!essential && !fundamental && !homography) ||
      (homography && geometry->H_estimation_space == G::HomographyEstimationSpace::CAMERA_RAY))
    return;
  const Eigen::Matrix3d& matrix = homography  ? *geometry->H
                                  : essential ? *geometry->E
                                              : *geometry->F;
  if (!matrix.allFinite()) return;
  std::array<float, 9> packed;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) packed[3 * r + c] = matrix(r, c);
  std::vector<sift_metal::MatchResult> matches;
  bool matched;
  if (essential) {
    const auto rays1 = MakeRays(geometry->camera1 ? *geometry->camera1 : camera1, keys1);
    const auto rays2 = MakeRays(geometry->camera2 ? *geometry->camera2 : camera2, keys2);
    matched = matcher.MatchGuidedTangent(desc1.data.data(),
                                         desc1.data.rows(),
                                         rays1.data(),
                                         desc2.data.data(),
                                         desc2.data.rows(),
                                         rays2.data(),
                                         {},
                                         packed.data(),
                                         max_error * max_error,
                                         &matches);
  } else {
    std::vector<sift_metal::MatchKeypoint> points1, points2;
    for (const auto& key : keys1) points1.push_back({key.x, key.y});
    for (const auto& key : keys2) points2.push_back({key.x, key.y});
    matched = matcher.MatchGuided(desc1.data.data(),
                                  desc1.data.rows(),
                                  points1.data(),
                                  desc2.data.data(),
                                  desc2.data.rows(),
                                  points2.data(),
                                  {},
                                  homography ? sift_metal::MatchGuidedGeometry::HOMOGRAPHY
                                             : sift_metal::MatchGuidedGeometry::EPIPOLAR,
                                  packed.data(),
                                  max_error * max_error,
                                  &matches);
  }
  Require(matched, CM_SPARSE_RESOURCE_ERROR, "Metal guided matching failed");
  ConvertMatches(matches, &geometry->inlier_matches);
}
}  // namespace

struct cm_sparse_job {
  cm_sparse_options options;
  std::vector<Input> images;
  std::vector<std::pair<size_t, size_t>> pairs;
  fs::path output_path;
  std::string metallib_path;
  std::atomic<bool> cancelled{false};
  std::atomic<bool> used{false};
  Clock::time_point started;
  std::string error;
  cm_sparse_progress_callback progress = nullptr;
  void* context = nullptr;

  bool TimedOut() const {
    return Clock::now() - started >= std::chrono::seconds(options.max_runtime_seconds);
  }
  void Check() const {
    Require(!cancelled.load(), CM_SPARSE_CANCELLED, "Reconstruction cancelled");
    Require(!TimedOut(), CM_SPARSE_RESOURCE_ERROR, "Reconstruction runtime limit reached");
  }
  void Notify(cm_sparse_stage stage, size_t completed, size_t total) const {
    if (progress) progress(context, stage, uint32_t(completed), uint32_t(total));
    Check();
  }
};

namespace {
void Prepare(cm_sparse_job& job, const fs::path& image_path, colmap::Database& database) {
  colmap::FlatHashMap<uint32_t, std::pair<colmap::Camera, colmap::rig_t>> calibrations;
  job.Notify(CM_SPARSE_PREPARING, 0, job.images.size());
  for (size_t i = 0; i < job.images.size(); ++i) {
    auto& input = job.images[i];
    const fs::path copy = image_path / input.name;
    fs::copy_file(input.path, copy);
    CheckRaster(copy, input.camera);
    auto it = calibrations.find(input.calibration_id);
    colmap::rig_t rig_id;
    if (it == calibrations.end()) {
      input.camera.camera_id = database.WriteCamera(input.camera);
      colmap::Rig rig;
      rig.AddRefSensor(input.camera.SensorId());
      rig_id = database.WriteRig(rig);
      calibrations.emplace(input.calibration_id, std::make_pair(input.camera, rig_id));
    } else {
      input.camera = it->second.first;
      rig_id = it->second.second;
    }
    colmap::Image image;
    image.SetName(input.name);
    image.SetCameraId(input.camera.camera_id);
    image.SetImageId(database.WriteImage(image));
    input.image_id = image.ImageId();
    colmap::Frame frame;
    frame.SetRigId(rig_id);
    frame.AddDataId(image.DataId());
    database.WriteFrame(frame);
    job.Notify(CM_SPARSE_PREPARING, i + 1, job.images.size());
  }
}

void Extract(cm_sparse_job& job, const fs::path& image_path, colmap::Database& database) {
  job.Notify(CM_SPARSE_EXTRACTING, 0, job.images.size());
  sift_metal::Options options;
  options.first_octave = job.options.first_octave;
  options.peak_threshold = 0.02f / 3;
  options.max_num_features = job.options.max_num_features;
  sift_metal::SiftMetalExtractor extractor;
  Require(extractor.Init(options, 1, 1, job.metallib_path),
          CM_SPARSE_RESOURCE_ERROR,
          "Cannot initialize Metal extraction from package shaders");
  for (size_t i = 0; i < job.images.size(); ++i) {
    @autoreleasepool {
      const auto& input = job.images[i];
      colmap::Bitmap bitmap;
      Require(bitmap.Read(image_path / input.name, false),
              CM_SPARSE_IO_ERROR,
              "Cannot decode input image");
      bitmap.Thumbnail(job.options.max_image_size);
      sift_metal::ExtractResult extracted;
      Require(extractor.Extract(
                  bitmap.RowMajorData().data(), bitmap.Width(), bitmap.Height(), &extracted),
              CM_SPARSE_RESOURCE_ERROR,
              "Metal feature extraction failed");
      colmap::FeatureKeypoints keypoints;
      for (const auto& key : extracted.keypoints) {
        colmap::FeatureKeypoint point(key.x, key.y, key.sigma, key.orientation);
        point.Rescale(float(input.camera.width) / bitmap.Width(),
                      float(input.camera.height) / bitmap.Height());
        keypoints.push_back(point);
      }
      colmap::FeatureDescriptors descriptors;
      descriptors.type = colmap::FeatureExtractorType::SIFT;
      colmap::FeatureDescriptorsFloatData floats =
          Eigen::Map<const colmap::FeatureDescriptorsFloatData>(
              extracted.descriptors.data(), keypoints.size(), 128);
      // Zero rows are preserved as zero descriptors instead of dividing by zero.
      for (Eigen::Index row = 0; row < floats.rows(); ++row) {
        const float norm = floats.row(row).lpNorm<1>();
        if (norm > 0 && std::isfinite(norm))
          floats.row(row) = (floats.row(row).array() / norm).sqrt();
        else
          floats.row(row).setZero();
      }
      descriptors.data = colmap::FeatureDescriptorsToUnsignedByte(floats);
      // The public budget is descriptor rows AFTER orientation expansion.
      colmap::ExtractTopScaleFeatures(&keypoints, &descriptors, job.options.max_num_features);
      database.WriteKeypoints(input.image_id, keypoints);
      database.WriteDescriptors(input.image_id, descriptors);
    }
    job.Notify(CM_SPARSE_EXTRACTING, i + 1, job.images.size());
  }
}

void Match(cm_sparse_job& job, colmap::Database& database) {
  job.Notify(CM_SPARSE_MATCHING, 0, job.pairs.size());
  sift_metal::SiftMetalMatcher matcher;
  Require(matcher.Init(job.metallib_path, job.options.matching_cache_bytes),
          CM_SPARSE_RESOURCE_ERROR,
          "Cannot initialize Metal matching from package shaders");
  colmap::TwoViewGeometryOptions geometry_options;
  geometry_options.ransac_options.random_seed = 0;
  for (size_t i = 0; i < job.pairs.size(); ++i) {
    @autoreleasepool {
      const auto& first = job.images[job.pairs[i].first];
      const auto& second = job.images[job.pairs[i].second];
      const auto keys1 = database.ReadKeypoints(first.image_id);
      const auto keys2 = database.ReadKeypoints(second.image_id);
      const auto desc1 = database.ReadDescriptors(first.image_id);
      const auto desc2 = database.ReadDescriptors(second.image_id);
      colmap::FeatureMatches matches;
      colmap::TwoViewGeometry geometry;
      if (!keys1.empty() && !keys2.empty()) {
        std::vector<sift_metal::MatchResult> metal_matches;
        Require(matcher.Match(desc1.data.data(),
                              desc1.data.rows(),
                              desc2.data.data(),
                              desc2.data.rows(),
                              {},
                              &metal_matches),
                CM_SPARSE_RESOURCE_ERROR,
                "Metal feature matching failed");
        ConvertMatches(metal_matches, &matches);
        job.Check();
        geometry = colmap::EstimateTwoViewGeometry(first.camera,
                                                   colmap::FeatureKeypointsToPointsVector(keys1),
                                                   second.camera,
                                                   colmap::FeatureKeypointsToPointsVector(keys2),
                                                   matches,
                                                   geometry_options);
        if (geometry.inlier_matches.size() >= size_t(geometry_options.min_num_inliers))
          GuideMatches(matcher,
                       first.camera,
                       keys1,
                       desc1,
                       second.camera,
                       keys2,
                       desc2,
                       geometry_options.ransac_options.max_error,
                       &geometry);
      }
      database.WriteMatches(first.image_id, second.image_id, matches);
      database.WriteTwoViewGeometry(first.image_id, second.image_id, geometry);
    }
    job.Notify(CM_SPARSE_MATCHING, i + 1, job.pairs.size());
  }
}

std::shared_ptr<colmap::Reconstruction> Map(cm_sparse_job& job,
                                            const fs::path& image_path,
                                            const std::shared_ptr<colmap::Database>& database) {
  job.Notify(CM_SPARSE_MAPPING, 0, job.images.size());
  auto options = std::make_shared<colmap::IncrementalPipelineOptions>();
  options->image_path = image_path;
  options->num_threads = job.options.num_threads;
  options->random_seed = 0;
  options->ba_refine_focal_length = job.options.refine_intrinsics != 0;
  options->ba_refine_extra_params = job.options.refine_intrinsics != 0;
  options->ba_refine_sensor_from_rig = false;
  // The mobile dependency build excludes SuiteSparse and LAPACK. Select the
  // available libraries explicitly, including when auto-selection switches
  // from dense to sparse Schur as the reconstruction grows.
  options->ba_sparse_linear_algebra_library_type = "ACCELERATE_SPARSE";
  options->ba_dense_linear_algebra_library_type = "EIGEN";
  auto manager = std::make_shared<colmap::ReconstructionManager>();
  colmap::IncrementalPipeline pipeline(options, database, manager);
  pipeline.SetCheckIfStoppedFunc([&job] { return job.cancelled.load() || job.TimedOut(); });
  const auto progress = [&] {
    size_t count = 0;
    for (size_t i = 0; i < manager->Size(); ++i)
      count = std::max(count, manager->Get(i)->NumRegImages());
    job.Notify(CM_SPARSE_MAPPING, count, job.images.size());
  };
  pipeline.AddCallback(colmap::IncrementalPipeline::INITIAL_IMAGE_PAIR_REG_CALLBACK, progress);
  pipeline.AddCallback(colmap::IncrementalPipeline::NEXT_IMAGE_REG_CALLBACK, progress);
  pipeline.Run();
  job.Check();
  std::shared_ptr<colmap::Reconstruction> best;
  for (size_t i = 0; i < manager->Size(); ++i)
    if (!best || manager->Get(i)->NumRegImages() > best->NumRegImages()) best = manager->Get(i);
  Require(best && best->NumPoints3D() > 0 &&
              double(best->NumRegImages()) / job.images.size() >=
                  job.options.minimum_registered_fraction,
          CM_SPARSE_RECONSTRUCTION_FAILED,
          "No connected sparse model meets minimum_registered_fraction");
  return best;
}

cm_sparse_result Run(cm_sparse_job& job) {
  StagingDirectory staging(job.output_path.parent_path());
  const fs::path dataset = staging.path / "dataset";
  const fs::path images = dataset / "images";
  fs::create_directories(images);
  auto database = colmap::Database::Open(staging.path / "features.db");
  Prepare(job, images, *database);
  Extract(job, images, *database);
  Match(job, *database);
  auto reconstruction = Map(job, images, database);
  database.reset();
  job.Notify(CM_SPARSE_EXPORTING, 0, 1);
  fs::create_directories(dataset / "sparse" / "0");
  reconstruction->WriteBinary(dataset / "sparse" / "0");
  const auto registered = reconstruction->RegImageIds();
  for (const auto& input : job.images) {
    if (std::find(registered.begin(), registered.end(), input.image_id) == registered.end())
      fs::remove(images / input.name);
  }
  cm_sparse_result result{};
  result.registered_images = reconstruction->NumRegImages();
  result.input_images = job.images.size();
  result.points = reconstruction->NumPoints3D();
  result.observations = reconstruction->ComputeNumObservations();
  result.mean_reprojection_error = reconstruction->ComputeMeanReprojectionError();
  result.mean_track_length = reconstruction->ComputeMeanTrackLength();
  reconstruction.reset();
  job.Check();
  // Darwin's exclusive rename provides an atomic no-overwrite commit, even if
  // another writer creates the requested output while this job is running.
  Require(renamex_np(dataset.c_str(), job.output_path.c_str(), RENAME_EXCL) == 0,
          CM_SPARSE_IO_ERROR,
          "Cannot publish dataset (output may already exist)");
  result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - job.started).count();
  // Publication is the commit point. A late cancellation cannot turn a
  // successfully published dataset into a failed result.
  if (job.progress) {
    try {
      job.progress(job.context, CM_SPARSE_FINISHED, 1, 1);
    } catch (...) {
    }
  }
  return result;
}

void CopyError(const char* message, char* buffer, size_t capacity) noexcept {
  if (buffer && capacity) std::snprintf(buffer, capacity, "%s", message);
}
}  // namespace

extern "C" {
uint32_t cm_sparse_abi_version(void) { return CM_SPARSE_ABI_VERSION; }

cm_sparse_options cm_sparse_default_options(void) {
  cm_sparse_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = CM_SPARSE_ABI_VERSION;
  options.max_image_size = 960;
  options.max_num_features = 8192;
  options.num_threads = 2;
  options.sequential_overlap = 10;
  options.keyframe_stride = 8;
  options.max_images = 256;
  options.max_num_pairs = 4096;
  options.max_runtime_seconds = 300;
  options.first_octave = -1;
  options.matching_cache_bytes = 32ull * 1024 * 1024;
  options.minimum_registered_fraction = 0.9;
  return options;
}

cm_sparse_status cm_sparse_job_create(const cm_sparse_image* images,
                                      size_t image_count,
                                      const cm_sparse_options* options,
                                      const char* output_path,
                                      const char* metallib_path,
                                      cm_sparse_job** output,
                                      char* error_buffer,
                                      size_t error_capacity) {
  if (output) *output = nullptr;
  CopyError("", error_buffer, error_capacity);
  try {
    Require(output && images && options && output_path && *output_path && metallib_path &&
                *metallib_path,
            CM_SPARSE_INVALID_ARGUMENT,
            "Missing required argument");
    ValidateOptions(*options);
    Require(image_count >= 2 && image_count <= options->max_images,
            CM_SPARSE_INVALID_ARGUMENT,
            "Image count must be 2..max_images");
    auto job = std::make_unique<cm_sparse_job>();
    job->options = *options;
    job->pairs = MakePairs(image_count, *options);
    const fs::path requested = fs::absolute(output_path).lexically_normal();
    Require(!requested.filename().empty() && fs::is_directory(requested.parent_path()),
            CM_SPARSE_INVALID_ARGUMENT,
            "Output parent directory must exist");
    job->output_path = fs::canonical(requested.parent_path()) / requested.filename();
    Require(!fs::exists(fs::symlink_status(job->output_path)),
            CM_SPARSE_INVALID_ARGUMENT,
            "Output path already exists");
    Require(
        fs::is_regular_file(metallib_path), CM_SPARSE_INVALID_ARGUMENT, "Missing package metallib");
    job->metallib_path = fs::canonical(metallib_path).string();
    colmap::FlatHashMap<uint32_t, colmap::Camera> calibrations;
    for (size_t i = 0; i < image_count; ++i) {
      Require(images[i].path && fs::is_regular_file(images[i].path),
              CM_SPARSE_INVALID_ARGUMENT,
              "Input image must be an existing regular file");
      Input input;
      input.path = fs::canonical(images[i].path);
      input.calibration_id = images[i].camera.calibration_id;
      input.camera = MakeCamera(images[i].camera);
      const auto [it, inserted] = calibrations.emplace(input.calibration_id, input.camera);
      Require(inserted || (it->second.model_id == input.camera.model_id &&
                           it->second.width == input.camera.width &&
                           it->second.height == input.camera.height &&
                           it->second.params == input.camera.params),
              CM_SPARSE_INVALID_ARGUMENT,
              "Images sharing calibration_id have different calibration");
      std::string extension = input.path.extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
      Require(extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
                  extension == ".heic",
              CM_SPARSE_INVALID_ARGUMENT,
              "Supported image extensions: jpg, jpeg, png, heic");
      char name[32];
      std::snprintf(name, sizeof(name), "%06zu", i + 1);
      input.name = name + extension;
      job->images.push_back(std::move(input));
    }
    *output = job.release();
    return CM_SPARSE_SUCCESS;
  } catch (const Failure& error) {
    CopyError(error.what(), error_buffer, error_capacity);
    return error.status;
  } catch (const std::bad_alloc&) {
    CopyError("Not enough memory", error_buffer, error_capacity);
    return CM_SPARSE_RESOURCE_ERROR;
  } catch (const fs::filesystem_error& error) {
    CopyError(error.what(), error_buffer, error_capacity);
    return CM_SPARSE_IO_ERROR;
  } catch (const std::exception& error) {
    CopyError(error.what(), error_buffer, error_capacity);
    return CM_SPARSE_INTERNAL_ERROR;
  } catch (...) {
    CopyError("Unknown native error", error_buffer, error_capacity);
    return CM_SPARSE_INTERNAL_ERROR;
  }
}

cm_sparse_status cm_sparse_job_run(cm_sparse_job* job,
                                   cm_sparse_progress_callback progress,
                                   void* context,
                                   cm_sparse_result* result) {
  if (!job || !result) return CM_SPARSE_INVALID_ARGUMENT;
  if (job->used.exchange(true)) return CM_SPARSE_INVALID_ARGUMENT;
  try {
    std::unique_lock<std::mutex> lock(active_job_mutex, std::try_to_lock);
    Require(lock.owns_lock(), CM_SPARSE_RESOURCE_ERROR, "Another sparse reconstruction is running");
    job->started = Clock::now();
    job->progress = progress;
    job->context = context;
    job->Check();
    @autoreleasepool {
      *result = Run(*job);
    }
    return CM_SPARSE_SUCCESS;
  } catch (const Failure& error) {
    try {
      job->error = error.what();
    } catch (...) {
    }
    return error.status;
  } catch (const std::bad_alloc&) {
    // Keep the diagnostic small enough for common string inline storage.
    try {
      job->error = "Out of memory";
    } catch (...) {
    }
    return CM_SPARSE_RESOURCE_ERROR;
  } catch (const fs::filesystem_error& error) {
    try {
      job->error = error.what();
    } catch (...) {
    }
    return CM_SPARSE_IO_ERROR;
  } catch (const std::exception& error) {
    try {
      job->error = error.what();
    } catch (...) {
    }
    return CM_SPARSE_INTERNAL_ERROR;
  } catch (...) {
    try {
      job->error = "Native error";
    } catch (...) {
    }
    return CM_SPARSE_INTERNAL_ERROR;
  }
}

void cm_sparse_job_cancel(cm_sparse_job* job) {
  if (job) job->cancelled.store(true);
}
const char* cm_sparse_job_error(const cm_sparse_job* job) {
  return job ? job->error.c_str() : "Invalid job";
}
void cm_sparse_job_destroy(cm_sparse_job* job) { delete job; }
}  // extern "C"
