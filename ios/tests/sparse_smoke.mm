// Copyright (c) 2026 Plinth. SPDX-License-Identifier: BSD-3-Clause
#include "ColmapSparse.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

namespace fs = std::filesystem;

static void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

static cm_sparse_camera Camera(
    uint32_t width, uint32_t height, double fx, double fy, double cx, double cy) {
  cm_sparse_camera camera{};
  camera.model = CM_SPARSE_PINHOLE;
  camera.width = width;
  camera.height = height;
  camera.params[0] = fx;
  camera.params[1] = fy;
  camera.params[2] = cx;
  camera.params[3] = cy;
  return camera;
}

static void WriteBlankPNG(const fs::path& path) {
  std::array<uint8_t, 64 * 64> pixels{};
  CGColorSpaceRef space = CGColorSpaceCreateDeviceGray();
  CGContextRef context =
      CGBitmapContextCreate(pixels.data(), 64, 64, 8, 64, space, kCGImageAlphaNone);
  CGImageRef image = CGBitmapContextCreateImage(context);
  NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
  CGImageDestinationRef destination =
      CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
  CGImageDestinationAddImage(destination, image, nullptr);
  const bool written = CGImageDestinationFinalize(destination);
  CFRelease(destination);
  CGImageRelease(image);
  CGContextRelease(context);
  CGColorSpaceRelease(space);
  Check(written, "Cannot write test fixture");
}

struct Job {
  cm_sparse_job* value = nullptr;
  ~Job() { cm_sparse_job_destroy(value); }
};

static bool NoStaging(const fs::path& parent) {
  for (const auto& file : fs::directory_iterator(parent))
    if (file.path().filename().string().find(".colmap-sparse-") == 0) return false;
  return true;
}

static cm_sparse_pose Pose(const double x) {
  cm_sparse_pose pose{};
  pose.camera_to_world[0] = std::cos(0.2);
  pose.camera_to_world[1] = std::sin(0.2);
  pose.camera_to_world[4] = -std::sin(0.2);
  pose.camera_to_world[5] = std::cos(0.2);
  pose.camera_to_world[10] = pose.camera_to_world[15] = 1;
  pose.camera_to_world[12] = x;
  pose.camera_to_world[13] = 2;
  pose.camera_to_world[14] = -3;
  return pose;
}

static void KnownPoseBoundaryTests(const std::string& image,
                                   const cm_sparse_camera& camera,
                                   const std::string& output,
                                   const fs::path& root,
                                   const char* metallib) {
  std::array<cm_sparse_image, 3> inputs{
      {{image.c_str(), camera}, {image.c_str(), camera}, {image.c_str(), camera}}};
  const std::array<cm_sparse_pose, 3> valid_poses = {Pose(1), Pose(1.1), Pose(1.2)};
  auto poses = valid_poses;
  auto options = cm_sparse_default_options();
  const auto defaults = cm_sparse_default_pose_refinement_options();
  auto refinement = defaults;
  Check(defaults.struct_size == sizeof(defaults) && defaults.abi_version == CM_SPARSE_ABI_VERSION &&
            defaults.max_num_iterations == 20 && defaults.max_translation_change == 0.15 &&
            std::abs(defaults.max_rotation_change_radians - 0.0872664626) < 1e-9,
        "Unexpected refinement option defaults");
  std::array<char, 512> error{};
  const cm_sparse_pose* pose_input = poses.data();
  size_t image_count = inputs.size();
  auto create = [&](Job& job) {
    return cm_sparse_job_create_with_poses(inputs.data(),
                                           pose_input,
                                           image_count,
                                           &options,
                                           &refinement,
                                           output.c_str(),
                                           metallib,
                                           &job.value,
                                           error.data(),
                                           error.size());
  };
  auto rejected = [&] {
    Job job;
    Check(create(job) == CM_SPARSE_INVALID_ARGUMENT && !job.value && error[0],
          "Invalid known-pose job accepted or missing diagnostic");
    Check(!fs::exists(output) && NoStaging(root), "Invalid known-pose job leaked output");
  };
  pose_input = nullptr;
  rejected();
  pose_input = poses.data();
  image_count = 2;
  rejected();
  image_count = inputs.size();
  options.refine_intrinsics = 1;
  rejected();
  options = cm_sparse_default_options();
  for (const auto& invalidate :
       std::vector<std::function<void(cm_sparse_pose_refinement_options&)>>{
           [](auto& value) { --value.struct_size; },
           [](auto& value) { ++value.abi_version; },
           [](auto& value) { value.max_num_iterations = 0; },
           [](auto& value) { value.max_num_iterations = 101; },
           [](auto& value) { value.max_translation_change = 0; },
           [](auto& value) { value.max_translation_change = 10.1; },
           [](auto& value) {
             value.max_translation_change = std::numeric_limits<double>::quiet_NaN();
           },
           [](auto& value) { value.max_rotation_change_radians = 0; },
           [](auto& value) { value.max_rotation_change_radians = 0.51; },
           [](auto& value) {
             value.max_rotation_change_radians = std::numeric_limits<double>::infinity();
           }}) {
    refinement = defaults;
    invalidate(refinement);
    rejected();
  }
  refinement = defaults;
  for (const auto& invalidate : std::vector<std::function<void(cm_sparse_pose&)>>{
           [](auto& pose) { pose.camera_to_world[12] = std::numeric_limits<double>::quiet_NaN(); },
           [](auto& pose) { pose.camera_to_world[3] = 0.1; },
           [](auto& pose) { pose.camera_to_world[15] = 0; },
           [](auto& pose) { pose.camera_to_world[0] *= 2; },
           [](auto& pose) { pose.camera_to_world[10] = -1; }}) {
    poses = valid_poses;
    invalidate(poses[1]);
    rejected();
  }
  poses = valid_poses;
  std::array<cm_sparse_pose, 3> refined{};
  Check(cm_sparse_job_copy_refined_poses(nullptr, refined.data(), refined.size()) ==
            CM_SPARSE_INVALID_ARGUMENT,
        "Null job exposed refined poses");
  {
    Job job;
    Check(create(job) == CM_SPARSE_SUCCESS, error.data());
    Check(cm_sparse_job_copy_refined_poses(job.value, refined.data(), refined.size()) ==
              CM_SPARSE_INVALID_ARGUMENT,
          "Unfinished job exposed refined poses");
    Check(cm_sparse_job_copy_refined_poses(job.value, nullptr, refined.size()) ==
              CM_SPARSE_INVALID_ARGUMENT,
          "Null pose output accepted");
    Check(cm_sparse_job_copy_refined_poses(job.value, refined.data(), refined.size() - 1) ==
              CM_SPARSE_INVALID_ARGUMENT,
          "Wrong pose output count accepted");
    cm_sparse_job_cancel(job.value);
    cm_sparse_result result{};
    Check(cm_sparse_job_run(job.value, nullptr, nullptr, &result) == CM_SPARSE_CANCELLED,
          "Known-pose pre-run cancellation failed");
    Check(cm_sparse_job_copy_refined_poses(job.value, refined.data(), refined.size()) ==
              CM_SPARSE_INVALID_ARGUMENT,
          "Cancelled job exposed refined poses");
    Check(!fs::exists(output) && NoStaging(root), "Known-pose cancellation leaked output");
  }
  {
    Job job;
    Check(create(job) == CM_SPARSE_SUCCESS, error.data());
    cm_sparse_result result{};
    Check(
        cm_sparse_job_run(job.value, nullptr, nullptr, &result) == CM_SPARSE_RECONSTRUCTION_FAILED,
        cm_sparse_job_error(job.value));
    Check(cm_sparse_job_copy_refined_poses(job.value, refined.data(), refined.size()) ==
              CM_SPARSE_INVALID_ARGUMENT,
          "Failed job exposed refined poses");
    Check(!fs::exists(output) && NoStaging(root), "Known-pose failure leaked output");
  }
}

static void BoundaryTests(const char* metallib) {
  @autoreleasepool {
    std::string pattern = (fs::temp_directory_path() / "colmap-api-tests-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back(0);
    Check(mkdtemp(buffer.data()) != nullptr, "Cannot create test directory");
    const fs::path root(buffer.data());
    struct Cleanup {
      fs::path path;
      ~Cleanup() { fs::remove_all(path); }
    } cleanup{root};
    const std::string image = (root / "blank.png").string();
    const std::string output = (root / "result").string();
    WriteBlankPNG(image);
    const cm_sparse_camera camera = Camera(64, 64, 60, 60, 32, 32);
    std::array<cm_sparse_image, 2> inputs{{{image.c_str(), camera}, {image.c_str(), camera}}};
    auto options = cm_sparse_default_options();
    std::array<char, 512> error{};
    auto create = [&](Job& job) {
      return cm_sparse_job_create(inputs.data(),
                                  inputs.size(),
                                  &options,
                                  output.c_str(),
                                  metallib,
                                  &job.value,
                                  error.data(),
                                  error.size());
    };
    Check(cm_sparse_abi_version() == 1, "Unexpected ABI");
    {
      Job job;
      options.max_num_features = 8193;
      Check(create(job) == CM_SPARSE_INVALID_ARGUMENT && !job.value && error[0],
            "Invalid options accepted");
      options = cm_sparse_default_options();
      inputs[1].camera.params[0] = 61;
      Check(create(job) == CM_SPARSE_INVALID_ARGUMENT, "Conflicting shared calibration accepted");
      inputs[1].camera = camera;
      options.max_num_pairs = 0;
      Check(create(job) == CM_SPARSE_INVALID_ARGUMENT, "Invalid pair budget accepted");
      options = cm_sparse_default_options();
      fs::create_directory(output);
      Check(create(job) == CM_SPARSE_INVALID_ARGUMENT, "Existing output accepted");
      fs::remove(output);
      fs::create_symlink(root / "missing", output);
      Check(create(job) == CM_SPARSE_INVALID_ARGUMENT, "Dangling output symlink accepted");
      fs::remove(output);
    }
    {
      Job job;
      Check(create(job) == CM_SPARSE_SUCCESS, error.data());
      std::array<cm_sparse_pose, 2> refined{};
      Check(cm_sparse_job_copy_refined_poses(job.value, refined.data(), refined.size()) ==
                CM_SPARSE_INVALID_ARGUMENT,
            "Ordinary reconstruction job exposed refined poses");
      cm_sparse_job_cancel(job.value);
      cm_sparse_result result{};
      Check(cm_sparse_job_run(job.value, nullptr, nullptr, &result) == CM_SPARSE_CANCELLED,
            "Pre-run cancellation failed");
      Check(cm_sparse_job_run(job.value, nullptr, nullptr, &result) == CM_SPARSE_INVALID_ARGUMENT,
            "Second run accepted");
      Check(!fs::exists(output) && NoStaging(root), "Pre-run cancellation leaked output");
    }
    for (cm_sparse_stage stage : {CM_SPARSE_PREPARING, CM_SPARSE_EXTRACTING, CM_SPARSE_MATCHING}) {
      Job job;
      Check(create(job) == CM_SPARSE_SUCCESS, error.data());
      struct Cancel {
        cm_sparse_job* job;
        cm_sparse_stage stage;
      } cancel{job.value, stage};
      const auto callback = [](void* context, cm_sparse_stage current, uint32_t, uint32_t) {
        auto& cancel = *static_cast<Cancel*>(context);
        if (current == cancel.stage) cm_sparse_job_cancel(cancel.job);
      };
      cm_sparse_result result{};
      Check(cm_sparse_job_run(job.value, callback, &cancel, &result) == CM_SPARSE_CANCELLED,
            cm_sparse_job_error(job.value));
      Check(!fs::exists(output) && NoStaging(root), "Cancellation leaked output");
    }
    {
      Job job;
      inputs[0].camera = inputs[1].camera = Camera(65, 64, 60, 60, 32, 32);
      Check(create(job) == CM_SPARSE_SUCCESS, error.data());
      cm_sparse_result result{};
      Check(cm_sparse_job_run(job.value, nullptr, nullptr, &result) == CM_SPARSE_INVALID_ARGUMENT,
            "Encoded dimensions were not checked before decode");
      Check(!fs::exists(output) && NoStaging(root), "Decode failure leaked output");
      inputs[0].camera = inputs[1].camera = camera;
    }
    // Two consecutive failed mapping runs exercise teardown without restarting.
    for (int i = 0; i < 2; ++i) {
      Job job;
      Check(create(job) == CM_SPARSE_SUCCESS, error.data());
      cm_sparse_result result{};
      Check(cm_sparse_job_run(job.value, nullptr, nullptr, &result) ==
                CM_SPARSE_RECONSTRUCTION_FAILED,
            cm_sparse_job_error(job.value));
      Check(!fs::exists(output) && NoStaging(root), "Mapping failure leaked output");
    }
    KnownPoseBoundaryTests(image, camera, output, root, metallib);
    std::puts("PASS: ABI validation, calibration, output protection, cancellation, cleanup, "
              "repeated runs, known-pose bounds and result access");
  }
}

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--test") {
      BoundaryTests(argv[2]);
      return 0;
    }
    if (argc != 10) {
      std::fprintf(stderr,
                   "Usage: sparse_smoke IMAGE_DIR OUTPUT_DIR METALLIB WIDTH HEIGHT FX FY CX CY\n"
                   "       sparse_smoke --test METALLIB\n");
      return 2;
    }
    std::vector<std::string> paths;
    for (const auto& file : fs::directory_iterator(argv[1])) {
      const auto extension = file.path().extension().string();
      if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
          extension == ".heic")
        paths.push_back(file.path().string());
    }
    std::sort(paths.begin(), paths.end());
    const auto camera = Camera(std::stoul(argv[4]),
                               std::stoul(argv[5]),
                               std::stod(argv[6]),
                               std::stod(argv[7]),
                               std::stod(argv[8]),
                               std::stod(argv[9]));
    std::vector<cm_sparse_image> images;
    for (const auto& path : paths) images.push_back({path.c_str(), camera});
    auto options = cm_sparse_default_options();
    std::array<char, 1024> error{};
    Job job;
    const auto status = cm_sparse_job_create(images.data(),
                                             images.size(),
                                             &options,
                                             argv[2],
                                             argv[3],
                                             &job.value,
                                             error.data(),
                                             error.size());
    Check(status == CM_SPARSE_SUCCESS, error.data());
    cm_sparse_result result{};
    const auto started = std::chrono::steady_clock::now();
    const auto progress =
        [](void* context, cm_sparse_stage stage, uint32_t completed, uint32_t total) {
          if (completed == 0 || completed == total || completed % 25 == 0) {
            const auto seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() -
                                     *static_cast<std::chrono::steady_clock::time_point*>(context))
                                     .count();
            std::fprintf(stderr,
                         "progress stage=%u completed=%u total=%u elapsed=%.3f\n",
                         stage,
                         completed,
                         total,
                         seconds);
          }
        };
    auto start_copy = started;
    const auto run_status = cm_sparse_job_run(job.value, progress, &start_copy, &result);
    Check(run_status == CM_SPARSE_SUCCESS, cm_sparse_job_error(job.value));
    std::printf(
        "{\"registered_images\":%u,\"input_images\":%u,\"points\":%llu,\"observations\":%llu,"
        "\"mean_reprojection_error\":%.9f,\"mean_track_length\":%.9f,\"elapsed_seconds\":%.3f}\n",
        result.registered_images,
        result.input_images,
        static_cast<unsigned long long>(result.points),
        static_cast<unsigned long long>(result.observations),
        result.mean_reprojection_error,
        result.mean_track_length,
        result.elapsed_seconds);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
