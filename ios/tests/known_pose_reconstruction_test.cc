// Copyright (c) 2026 Plinth. SPDX-License-Identifier: BSD-3-Clause
#include "colmap/scene/database_sqlite.h"
#include "colmap/sensor/bitmap.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "pose_refinement.h"
#include <unistd.h>

namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct ImageDirectory {
  std::filesystem::path path;

  ImageDirectory() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "colmap-known-poses-XXXXXX")
            .string();
    Require(mkdtemp(pattern.data()) != nullptr,
            "Cannot create fixture image directory");
    path = pattern;
  }

  ~ImageDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

// The database contains independently projected observations and verified
// feature correspondences, but no 3D points. This exercises the same
// triangulation/refinement entry point used after mobile feature matching.
void ReconstructPerturbedPoses(size_t supported_cameras,
                               bool append_unsupported_camera = false) {
  constexpr size_t kNumPoints = 64;
  const size_t camera_count =
      supported_cameras + size_t(append_unsupported_camera);
  auto database = colmap::Database::Open(colmap::kInMemorySqliteDatabasePath);
  auto camera = colmap::Camera::CreateFromModelId(
      1, colmap::CameraModelId::kOpenCV, 760, 640, 480);
  camera.params = {760, 750, 319, 241, 0.025, -0.008, 0.0002, -0.0001};
  camera.has_prior_focal_length = true;
  camera.camera_id = database->WriteCamera(camera);
  colmap::Rig rig;
  rig.AddRefSensor(camera.SensorId());
  const auto rig_id = database->WriteRig(rig);

  const colmap::Rigid3d world_from_scene(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(0.37, Eigen::Vector3d(0.3, 0.4, 0.5).normalized())),
      Eigen::Vector3d(2.0, -3.0, 1.5));
  std::vector<Eigen::Vector3d> points;
  for (size_t i = 0; i < kNumPoints; ++i) {
    points.push_back(world_from_scene *
                     Eigen::Vector3d(0.13 + 0.14 * (i % 8),
                                     -0.5 + 0.14 * ((i / 8) % 8),
                                     3.0 + 0.17 * (i % 5)));
  }

  ImageDirectory image_directory;
  colmap::Bitmap bitmap(640, 480, true);
  bitmap.Fill(colmap::BitmapColor<uint8_t>(40, 80, 120));
  std::vector<colmap::image_t> image_ids;
  std::vector<colmap::Rigid3d> initial_poses;
  std::vector<colmap::Rigid3d> true_poses;
  const std::array<colmap::image_t, 6> unordered_ids = {23, 5, 41, 9, 17, 3};
  for (size_t i = 0; i < camera_count; ++i) {
    const colmap::Rigid3d scene_from_cam(
        Eigen::Quaterniond(
            Eigen::AngleAxisd(0.025 * i, Eigen::Vector3d::UnitY())),
        Eigen::Vector3d(0.16 * i,
                        i == 0 || i + 1 >= supported_cameras
                            ? 0.0
                            : 0.03 * (i % 2 ? 1 : -1),
                        0.01 * i));
    const auto true_world_from_cam = world_from_scene * scene_from_cam;
    auto initial_world_from_cam = true_world_from_cam;
    if (i != 0 && i + 1 < supported_cameras) {
      initial_world_from_cam.translation() +=
          Eigen::Vector3d(i % 2 ? 0.008 : -0.008, 0.006, -0.003);
      initial_world_from_cam.rotation() =
          Eigen::Quaterniond(Eigen::AngleAxisd(
              0.00872664626,
              Eigen::Vector3d(1.0, i % 2 ? 2.0 : -2.0, 0.5).normalized())) *
          true_world_from_cam.rotation();
    }
    true_poses.push_back(colmap::Inverse(true_world_from_cam));
    initial_poses.push_back(colmap::Inverse(initial_world_from_cam));

    colmap::Image image;
    image.SetImageId(unordered_ids[i]);
    image.SetName(std::to_string(i) + ".png");
    image.SetCameraId(camera.camera_id);
    database->WriteImage(image, /*use_image_id=*/true);
    image_ids.push_back(image.ImageId());
    colmap::Frame frame;
    frame.SetRigId(rig_id);
    frame.AddDataId(image.DataId());
    database->WriteFrame(frame);
    Require(bitmap.Write(image_directory.path / image.Name()),
            "Cannot write fixture image");

    colmap::FeatureKeypoints keypoints;
    if (i < supported_cameras) {
      for (const auto& point : points) {
        const Eigen::Vector3d camera_point = true_poses.back() * point;
        const double x = camera_point.x() / camera_point.z();
        const double y = camera_point.y() / camera_point.z();
        const double radius2 = x * x + y * y;
        const double radial = 1 + 0.025 * radius2 - 0.008 * radius2 * radius2;
        const double xd =
            x * radial + 0.0004 * x * y - 0.0001 * (radius2 + 2 * x * x);
        const double yd =
            y * radial + 0.0002 * (radius2 + 2 * y * y) - 0.0002 * x * y;
        keypoints.emplace_back(760 * xd + 319, 750 * yd + 241, 1, 0);
      }
    }
    database->WriteKeypoints(image.ImageId(), keypoints);
  }

  for (size_t i = 0; i < supported_cameras; ++i) {
    for (size_t j = i + 1; j < supported_cameras; ++j) {
      colmap::TwoViewGeometry geometry;
      geometry.config = colmap::TwoViewGeometry::CALIBRATED;
      for (colmap::point2D_t point = 0; point < kNumPoints; ++point) {
        geometry.inlier_matches.emplace_back(point, point);
      }
      database->WriteTwoViewGeometry(image_ids[i], image_ids[j], geometry);
    }
  }

  const auto reconstruction = colmap_sparse::TriangulateAndRefineKnownPoses(
      database,
      image_directory.path,
      image_ids,
      initial_poses,
      cm_sparse_default_pose_refinement_options(),
      1,
      [] { return false; });
  Require(reconstruction->NumRegImages() == camera_count,
          "Refinement must retain every input image");
  Require(reconstruction->RegImageIds() == image_ids,
          "Registration must preserve capture-array order");
  Require(reconstruction->Camera(camera.camera_id).params == camera.params,
          "Refinement must retain supplied camera intrinsics");
  Require(reconstruction->NumPoints3D() == kNumPoints &&
              reconstruction->ComputeMeanReprojectionError() < 0.05,
          "Approximate poses must recover accurate multi-view geometry");
  for (size_t i = 0; i < camera_count; ++i) {
    const auto& image = reconstruction->Image(image_ids[i]);
    const auto& refined = image.CamFromWorld();
    Require((refined.TgtOriginInSrc() - true_poses[i].TgtOriginInSrc()).norm() <
                    0.001 &&
                refined.rotation().angularDistance(true_poses[i].rotation()) <
                    0.001,
            "Triangulated tracks must permit correcting the input camera pose");
    if (i == 0 || i + 1 >= supported_cameras) {
      Require((refined.ToMatrix() - initial_poses[i].ToMatrix()).norm() < 1e-12,
              "Anchor and unsupported poses must remain fixed");
    } else {
      Require(image.NumPoints3D() == kNumPoints,
              "Interior-camera observations must survive pose initialization");
    }
  }
  std::printf("PASS %zu supported cameras, %zu retained images\n",
              supported_cameras,
              camera_count);
}
}  // namespace

int main() {
  try {
    ReconstructPerturbedPoses(3);
    ReconstructPerturbedPoses(5);
    ReconstructPerturbedPoses(5, /*append_unsupported_camera=*/true);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL known-pose reconstruction: %s\n", error.what());
    return 1;
  }
}
