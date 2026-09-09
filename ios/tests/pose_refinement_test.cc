// Copyright (c) 2026 Plinth. SPDX-License-Identifier: BSD-3-Clause
#include "pose_refinement.h"

#include "colmap/scene/reconstruction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using colmap::Rigid3d;

void Require(const bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void Near(const double actual,
          const double expected,
          const double tolerance,
          const std::string& message) {
  Require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          message + ": actual=" + std::to_string(actual) +
              " expected=" + std::to_string(expected));
}

void RequireRejected(const std::function<void()>& action,
                     const std::string& message) {
  try {
    action();
  } catch (const std::runtime_error& error) {
    Require(error.what()[0] != '\0', message + ": missing diagnostic");
    return;
  }
  throw std::runtime_error(message);
}

struct Scene {
  colmap::Reconstruction reconstruction;
  std::vector<colmap::image_t> image_ids;
  std::vector<Rigid3d> initial_cam_from_world;
  std::vector<Rigid3d> true_cam_from_world;
};

// Generate observations independently of COLMAP's projection implementation.
// Nonzero distortion also verifies that all supplied intrinsics remain fixed.
Eigen::Vector2d ProjectOpenCV(const Eigen::Vector3d& point,
                              const std::vector<double>& params) {
  Require(point.z() > 0, "Synthetic point must be in front of the camera");
  const double x = point.x() / point.z();
  const double y = point.y() / point.z();
  const double radius2 = x * x + y * y;
  const double radial = 1 + params[4] * radius2 + params[5] * radius2 * radius2;
  const double xd =
      x * radial + 2 * params[6] * x * y + params[7] * (radius2 + 2 * x * x);
  const double yd =
      y * radial + params[6] * (radius2 + 2 * y * y) + 2 * params[7] * x * y;
  return {params[0] * xd + params[2], params[1] * yd + params[3]};
}

Scene MakeScene(const size_t camera_count = 5,
                const size_t point_count = 64,
                const bool disconnected = false,
                const bool two_view_tracks = false) {
  Scene scene;
  auto camera = colmap::Camera::CreateFromModelId(
      1, colmap::CameraModelId::kOpenCV, 760, 640, 480);
  camera.params = {760, 750, 319, 241, 0.025, -0.008, 0.0002, -0.0001};
  camera.has_prior_focal_length = true;
  scene.reconstruction.AddCameraWithTrivialRig(camera);

  const Rigid3d world_from_scene(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(0.37, Eigen::Vector3d(0.3, 0.4, 0.5).normalized())),
      Eigen::Vector3d(2.0, -3.0, 1.5));
  std::vector<Eigen::Vector3d> points;
  for (size_t index = 0; index < point_count; ++index) {
    points.push_back(world_from_scene *
                     Eigen::Vector3d(0.13 + 0.14 * (index % 8),
                                     -0.5 + 0.14 * ((index / 8) % 8),
                                     3.0 + 0.17 * (index % 5)));
  }

  // Deliberately unsorted IDs catch an implementation that anchors by ID
  // instead of capture-array order. The last input has the largest baseline.
  const std::array<colmap::image_t, 7> image_ids = {23, 5, 41, 9, 17, 3, 29};
  Require(camera_count <= image_ids.size(), "Synthetic camera count");
  for (size_t index = 0; index < camera_count; ++index) {
    const Rigid3d scene_from_cam(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.025 * static_cast<double>(index),
                                             Eigen::Vector3d::UnitY())),
        Eigen::Vector3d(0.16 * index,
                        index == 0 || index + 1 == camera_count
                            ? 0.0
                            : 0.03 * (index % 2 ? 1 : -1),
                        0.01 * index));
    const auto true_world_from_cam = world_from_scene * scene_from_cam;
    auto initial_world_from_cam = true_world_from_cam;
    if (index != 0 && index + 1 != camera_count) {
      initial_world_from_cam.translation() +=
          Eigen::Vector3d(index % 2 ? 0.008 : -0.008, 0.006, -0.003);
      initial_world_from_cam.rotation() =
          Eigen::Quaterniond(Eigen::AngleAxisd(
              0.00872664626,
              Eigen::Vector3d(1.0, index % 2 ? 2.0 : -2.0, 0.5).normalized())) *
          true_world_from_cam.rotation();
    }
    scene.image_ids.push_back(image_ids[index]);
    scene.true_cam_from_world.push_back(colmap::Inverse(true_world_from_cam));
    scene.initial_cam_from_world.push_back(
        colmap::Inverse(initial_world_from_cam));

    std::vector<Eigen::Vector2d> observations;
    for (const auto& point : points) {
      observations.push_back(ProjectOpenCV(
          scene.true_cam_from_world.back() * point, camera.params));
    }
    colmap::Image image;
    image.SetImageId(image_ids[index]);
    image.SetCameraId(camera.camera_id);
    image.SetName(std::to_string(index) + ".png");
    image.SetPoints2D(observations);
    scene.reconstruction.AddImageWithTrivialFrame(
        image, scene.initial_cam_from_world.back());
  }

  Require(!disconnected || camera_count >= 6,
          "Disconnected fixture needs two groups of at least three cameras");
  for (size_t index = 0; index < point_count; ++index) {
    colmap::Track track;
    for (size_t camera_index = 0; camera_index < camera_count; ++camera_index) {
      if (two_view_tracks && camera_index != index % camera_count &&
          camera_index != (index + 1) % camera_count) {
        continue;
      }
      if (disconnected &&
          ((index < point_count / 2) != (camera_index < camera_count / 2))) {
        continue;
      }
      track.AddElement(scene.image_ids[camera_index], index);
    }
    scene.reconstruction.AddPoint3D(
        points[index] + Eigen::Vector3d(0.002, -0.001, 0.003), track);
  }
  return scene;
}

colmap::TrackElement DuplicateObservation(
    Scene& scene,
    const colmap::TrackElement& original,
    const Eigen::Vector2d& offset = Eigen::Vector2d::Zero()) {
  auto& image = scene.reconstruction.Image(original.image_id);
  const auto point_id = image.Point2D(original.point2D_idx).point3D_id;
  Require(point_id != colmap::kInvalidPoint3DId,
          "Duplicate fixture must reference an existing track");
  const Eigen::Vector2d xy = image.Point2D(original.point2D_idx).xy + offset;
  const colmap::TrackElement duplicate(original.image_id, image.NumPoints2D());
  image.Points2D().push_back({xy});
  scene.reconstruction.AddObservation(point_id, duplicate);
  return duplicate;
}

std::vector<colmap::TrackElement> DuplicateAllObservations(Scene& scene) {
  std::vector<colmap::TrackElement> duplicates;
  const auto point_ids = scene.reconstruction.Point3DIds();
  for (const auto point_id : point_ids) {
    const auto elements =
        scene.reconstruction.Point3D(point_id).track.Elements();
    for (const auto& element : elements) {
      duplicates.push_back(DuplicateObservation(scene, element));
    }
  }
  return duplicates;
}

void RequireUntriangulated(const Scene& scene,
                           const std::vector<colmap::TrackElement>& elements) {
  for (const auto& element : elements) {
    Require(!scene.reconstruction.Image(element.image_id)
                 .Point2D(element.point2D_idx)
                 .HasPoint3D(),
            "Removed observation still references a triangulated point");
  }
}

double ReprojectionError(const Scene& scene) {
  double squared_error = 0;
  size_t observations = 0;
  for (const auto& [point_id, point] : scene.reconstruction.Points3D()) {
    for (const auto& element : point.track.Elements()) {
      const auto& image = scene.reconstruction.Image(element.image_id);
      const auto projected = image.ProjectPoint(point.xyz);
      Require(projected.has_value(), "Refinement put a point behind a camera");
      squared_error +=
          (*projected - image.Point2D(element.point2D_idx).xy).squaredNorm();
      ++observations;
    }
  }
  Require(observations != 0, "Reprojection fixture needs observations");
  return std::sqrt(squared_error / observations);
}

std::array<double, 2> PoseError(const Scene& scene) {
  std::array<double, 2> error{};
  for (size_t index = 1; index + 1 < scene.image_ids.size(); ++index) {
    const auto actual =
        scene.reconstruction.Image(scene.image_ids[index]).CamFromWorld();
    const auto& expected = scene.true_cam_from_world[index];
    error[0] += (actual.TgtOriginInSrc() - expected.TgtOriginInSrc()).norm();
    error[1] += actual.rotation().angularDistance(expected.rotation());
  }
  return error;
}

void Refine(
    Scene& scene,
    const cm_sparse_pose_refinement_options& options =
        cm_sparse_default_pose_refinement_options(),
    const std::function<bool()>& check_if_stopped = [] { return false; }) {
  colmap_sparse::RefineKnownPoses(scene.reconstruction,
                                  scene.image_ids,
                                  scene.initial_cam_from_world,
                                  options,
                                  1,
                                  check_if_stopped);
}

void ImprovesPosesAndPreservesWorldFrame() {
  auto scene = MakeScene();
  const auto initial_intrinsics = scene.reconstruction.Camera(1).params;
  const double initial_reprojection = ReprojectionError(scene);
  const auto initial_pose_error = PoseError(scene);
  const auto first_pose = scene.initial_cam_from_world.front();
  const auto farthest_pose = scene.initial_cam_from_world.back();
  const double initial_baseline =
      (first_pose.TgtOriginInSrc() - farthest_pose.TgtOriginInSrc()).norm();
  Refine(scene);
  const double final_reprojection = ReprojectionError(scene);
  const auto final_pose_error = PoseError(scene);
  Require(final_reprojection < 0.05 * initial_reprojection,
          "Bundle adjustment must reduce reprojection error");
  Require(final_reprojection < 0.05, "Synthetic reprojection accuracy");
  Require(final_pose_error[0] < 0.2 * initial_pose_error[0],
          "Refined camera centers must improve against ground truth");
  Require(final_pose_error[1] < 0.2 * initial_pose_error[1],
          "Refined camera rotations must improve against ground truth");
  const auto final_first =
      scene.reconstruction.Image(scene.image_ids.front()).CamFromWorld();
  const auto final_farthest =
      scene.reconstruction.Image(scene.image_ids.back()).CamFromWorld();
  Near((final_first.ToMatrix() - first_pose.ToMatrix()).norm(),
       0,
       1e-12,
       "First input pose anchors the original world frame");
  Near((final_farthest.ToMatrix() - farthest_pose.ToMatrix()).norm(),
       0,
       1e-12,
       "Farthest pose remains fixed");
  Near((final_first.TgtOriginInSrc() - final_farthest.TgtOriginInSrc()).norm(),
       initial_baseline,
       1e-12,
       "Metric anchor baseline remains fixed");
  Require(scene.reconstruction.Camera(1).params == initial_intrinsics,
          "Focal length, principal point, and distortion must remain fixed");
  Require(scene.reconstruction.NumRegImages() == scene.image_ids.size(),
          "All input cameras remain registered");
  std::printf(
      "  reprojection %.6f -> %.6f px, position error %.6f -> %.6f m, "
      "rotation error %.6f -> %.6f rad\n",
      initial_reprojection,
      final_reprojection,
      initial_pose_error[0],
      final_pose_error[0],
      initial_pose_error[1],
      final_pose_error[1]);
}

void RequirePoseUnchanged(const Scene& scene, const size_t index) {
  Near((scene.reconstruction.Image(scene.image_ids[index])
            .CamFromWorld()
            .ToMatrix() -
        scene.initial_cam_from_world[index].ToMatrix())
           .norm(),
       0,
       1e-12,
       "Excluded and anchor cameras must retain their original poses");
}

void MakeAccurateAnchor(Scene& scene, const size_t index) {
  scene.initial_cam_from_world[index] = scene.true_cam_from_world[index];
  scene.reconstruction.Image(scene.image_ids[index])
      .FramePtr()
      ->SetRigFromWorld(scene.initial_cam_from_world[index]);
}

void RefinesOnlyLargestSupportedComponent() {
  for (const bool weak_bridge : {false, true}) {
    auto scene = MakeScene(7, 64, true);
    // The second component has four cameras; its first/farthest poses anchor
    // its world frame. The global first camera belongs to the smaller group.
    MakeAccurateAnchor(scene, 3);
    if (weak_bridge) {
      const auto point_id =
          scene.reconstruction.Image(scene.image_ids[0]).Point2D(0).point3D_id;
      scene.reconstruction.AddObservation(
          point_id, colmap::TrackElement(scene.image_ids[3], 0));
    }
    Refine(scene);
    for (const size_t index :
         {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{6}}) {
      RequirePoseUnchanged(scene, index);
    }
    for (const size_t index : {size_t{4}, size_t{5}}) {
      const auto true_center =
          scene.true_cam_from_world[index].TgtOriginInSrc();
      const double before =
          (scene.initial_cam_from_world[index].TgtOriginInSrc() - true_center)
              .norm();
      const double after = (scene.reconstruction.Image(scene.image_ids[index])
                                .ProjectionCenter() -
                            true_center)
                               .norm();
      Require(after < 0.2 * before,
              "Only the largest supported component's variable cameras should "
              "improve");
    }
    Require(scene.reconstruction.NumRegImages() == scene.image_ids.size(),
            "Component refinement must retain every input camera");
  }

  auto tie = MakeScene(6, 64, true);
  MakeAccurateAnchor(tie, 2);
  Refine(tie);
  for (const size_t index :
       {size_t{0}, size_t{2}, size_t{3}, size_t{4}, size_t{5}}) {
    RequirePoseUnchanged(tie, index);
  }
  Require(
      (tie.reconstruction.Image(tie.image_ids[1]).CamFromWorld().ToMatrix() -
       tie.initial_cam_from_world[1].ToMatrix())
              .norm() > 1e-5,
      "Equal-size components must choose the earliest capture component");
}

void PreservesScaleAcrossArticulationBlocks() {
  for (const bool larger_second_block : {false, true}) {
    auto scene = MakeScene(larger_second_block ? 6 : 5, 128);
    const std::vector<size_t> first_block = larger_second_block
                                                ? std::vector<size_t>{0, 1, 2}
                                                : std::vector<size_t>{0, 2, 4};
    const std::vector<size_t> second_block =
        larger_second_block ? std::vector<size_t>{2, 3, 4, 5}
                            : std::vector<size_t>{1, 2, 3};
    // Each block has 64 three-or-more-view tracks. Only camera 2 is shared;
    // neither track set constrains the other block's metric scale. In the tie
    // case, both global anchors belong to the earliest-capture block, while
    // the smallest image ID belongs to the other block.
    for (colmap::point2D_t point_index = 0; point_index < 128; ++point_index) {
      const auto& block = point_index < 64 ? first_block : second_block;
      for (size_t index = 0; index < scene.image_ids.size(); ++index) {
        if (std::find(block.begin(), block.end(), index) == block.end()) {
          scene.reconstruction.DeleteObservation(scene.image_ids[index],
                                                 point_index);
        }
      }
    }
    MakeAccurateAnchor(scene, 2);
    Require(scene.reconstruction.IsValid(),
            "Articulation fixture associations");
    Refine(scene);

    const std::vector<size_t> fixed = larger_second_block
                                          ? std::vector<size_t>{0, 1, 2, 5}
                                          : std::vector<size_t>{0, 1, 3, 4};
    for (const auto index : fixed) RequirePoseUnchanged(scene, index);
    const auto require_baseline = [&](const size_t first, const size_t second) {
      const double before =
          (scene.initial_cam_from_world[first].TgtOriginInSrc() -
           scene.initial_cam_from_world[second].TgtOriginInSrc())
              .norm();
      const auto first_center =
          scene.reconstruction.Image(scene.image_ids[first]).ProjectionCenter();
      const auto second_center =
          scene.reconstruction.Image(scene.image_ids[second])
              .ProjectionCenter();
      const double after = (first_center - second_center).norm();
      Near(after,
           before,
           1e-12,
           "Articulation block baseline must remain fixed");
    };
    // Preserve both the selected block's anchors and a baseline between the
    // excluded cameras, whose poses must not drift along an unfixed scale.
    require_baseline(larger_second_block ? 2 : 0, larger_second_block ? 5 : 4);
    require_baseline(larger_second_block ? 0 : 1, larger_second_block ? 1 : 3);
    if (larger_second_block) {
      for (const size_t index : {size_t{3}, size_t{4}}) {
        const auto true_center =
            scene.true_cam_from_world[index].TgtOriginInSrc();
        const double before =
            (scene.initial_cam_from_world[index].TgtOriginInSrc() - true_center)
                .norm();
        const double after = (scene.reconstruction.Image(scene.image_ids[index])
                                  .ProjectionCenter() -
                              true_center)
                                 .norm();
        Require(after < 0.2 * before,
                "The larger articulation block's interior cameras should "
                "still improve");
      }
    }
    Require(scene.reconstruction.NumRegImages() == scene.image_ids.size(),
            "Articulation block selection must retain every input camera");
  }
}

void KeepsWeakCamerasFixed() {
  auto scene = MakeScene();
  for (colmap::point2D_t index = 11; index < 64; ++index) {
    scene.reconstruction.DeleteObservation(scene.image_ids[2], index);
  }
  Refine(scene);
  RequirePoseUnchanged(scene, 2);
  RequirePoseUnchanged(scene, 0);
  RequirePoseUnchanged(scene, 4);
  Require(scene.reconstruction.NumRegImages() == scene.image_ids.size(),
          "Weak cameras must remain in the output");
  Require((scene.reconstruction.Image(scene.image_ids[1])
               .CamFromWorld()
               .ToMatrix() -
           scene.initial_cam_from_world[1].ToMatrix())
                  .norm() > 1e-5,
          "Supported interior cameras should still be refined");
}

void RejectsInsufficientTracks() {
  auto scene = MakeScene(5, 11);
  RequireRejected([&] { Refine(scene); }, "Insufficient tracks accepted");
  auto two_view_tracks = MakeScene(5, 100, false, true);
  RequireRejected(
      [&] { Refine(two_view_tracks); },
      "Abundant two-view tracks were accepted as multi-view support");
}

void CollapsesCoincidentObservations() {
  auto reference = MakeScene();
  Refine(reference);
  auto scene = MakeScene();
  const auto intrinsics = scene.reconstruction.Camera(1).params;
  auto duplicates = DuplicateAllObservations(scene);
  const auto point_ids = scene.reconstruction.Point3DIds();
  for (const auto point_id : point_ids) {
    auto& track = scene.reconstruction.Point3D(point_id).track;
    const auto original = track.Element(0);
    duplicates.push_back(
        DuplicateObservation(scene, original, Eigen::Vector2d(4e-7, 0)));
    duplicates.push_back(
        DuplicateObservation(scene, original, Eigen::Vector2d(-4e-7, 0)));
    // The representative must be the lowest feature index, regardless of
    // whether the redundant orientation was discovered first in the track.
    std::reverse(track.Elements().begin(), track.Elements().end());
  }
  Require(scene.reconstruction.IsValid(), "Duplicate fixture associations");
  Refine(scene);
  Require(scene.reconstruction.NumPoints3D() == 64,
          "Coincident observations should preserve every supported point");
  Require(scene.reconstruction.ComputeNumObservations() == 64 * 5,
          "Each physical observation should contribute once");
  RequireUntriangulated(scene, duplicates);
  Require(scene.reconstruction.IsValid(), "Cleaned duplicate associations");
  Require(scene.reconstruction.Camera(1).params == intrinsics,
          "Duplicate cleanup must preserve calibration");
  for (const auto image_id : scene.image_ids) {
    Near((scene.reconstruction.Image(image_id).CamFromWorld().ToMatrix() -
          reference.reconstruction.Image(image_id).CamFromWorld().ToMatrix())
             .norm(),
         0,
         1e-9,
         "Duplicate orientations must not change the refined camera solution");
  }
  for (const auto& [point_id, point] : scene.reconstruction.Points3D()) {
    Require(point.track.Length() == scene.image_ids.size(),
            "Cleaned tracks must retain the five unique views");
    for (const auto& element : point.track.Elements()) {
      Require(element.point2D_idx < 64,
              "Cleanup must retain the original lowest feature index");
    }
  }
  Require(ReprojectionError(scene) < 0.05,
          "Coincident duplicate cleanup must preserve pose accuracy");
}

void DiscardsAmbiguousPoints() {
  // The small case has all duplicates within 1e-6 of the original, but their
  // pairwise diameter is 1.5e-6. Comparing only with the first feature is
  // unsafe.
  for (const double offset : {2.0, 7.5e-7}) {
    auto scene = MakeScene();
    const auto& image = scene.reconstruction.Image(scene.image_ids[1]);
    const auto point_id = image.Point2D(0).point3D_id;
    auto discarded = scene.reconstruction.Point3D(point_id).track.Elements();
    const colmap::TrackElement original(scene.image_ids[1], 0);
    discarded.push_back(
        DuplicateObservation(scene, original, Eigen::Vector2d(offset, 0)));
    discarded.push_back(
        DuplicateObservation(scene, original, Eigen::Vector2d(-offset, 0)));
    Refine(scene);
    Require(!scene.reconstruction.ExistsPoint3D(point_id) &&
                scene.reconstruction.NumPoints3D() == 63,
            "Discard only the point with conflicting image observations");
    RequireUntriangulated(scene, discarded);
    Require(scene.reconstruction.IsValid(),
            "Ambiguous-point cleanup associations");
    Require(ReprojectionError(scene) < 0.05,
            "Remaining supported tracks must still refine accurately");
  }
}

void RejectsInflatedDuplicateSupport() {
  auto two_view_tracks = MakeScene(5, 100, false, true);
  DuplicateAllObservations(two_view_tracks);
  RequireRejected([&] { Refine(two_view_tracks); },
                  "Duplicate features must not turn two views into three");
  Require(two_view_tracks.reconstruction.NumPoints3D() == 0,
          "Remove points with fewer than three unique views");
  Require(two_view_tracks.reconstruction.IsValid(),
          "Two-view point removal associations");

  auto insufficient = MakeScene(5, 11);
  const auto duplicates = DuplicateAllObservations(insufficient);
  RequireRejected([&] { Refine(insufficient); },
                  "Duplicate orientations must not inflate track support");
  Require(insufficient.reconstruction.ComputeNumObservations() == 11 * 5,
          "Support rejection must count unique observations");
  RequireUntriangulated(insufficient, duplicates);
  Require(insufficient.reconstruction.IsValid(),
          "Insufficient-support cleanup associations");

  auto ambiguous_support = MakeScene(5, 12);
  DuplicateObservation(ambiguous_support,
                       colmap::TrackElement(ambiguous_support.image_ids[1], 0),
                       Eigen::Vector2d(2, 0));
  RequireRejected([&] { Refine(ambiguous_support); },
                  "Removing ambiguous evidence must reapply the support guard");
  Require(ambiguous_support.reconstruction.NumPoints3D() == 11,
          "Conflicting observations must remove their complete point");
}

void RejectsExcessiveCorrections() {
  auto translation_scene = MakeScene();
  auto options = cm_sparse_default_pose_refinement_options();
  options.max_translation_change = 1e-6;
  RequireRejected([&] { Refine(translation_scene, options); },
                  "Excessive camera-center correction accepted");
  auto rotation_scene = MakeScene();
  options = cm_sparse_default_pose_refinement_options();
  options.max_rotation_change_radians = 1e-6;
  RequireRejected([&] { Refine(rotation_scene, options); },
                  "Excessive camera-rotation correction accepted");
}

void RejectsDegenerateBaseline() {
  auto scene = MakeScene();
  for (size_t index = 0; index < scene.image_ids.size(); ++index) {
    scene.initial_cam_from_world[index] = scene.initial_cam_from_world.front();
    scene.reconstruction.Image(scene.image_ids[index])
        .FramePtr()
        ->SetRigFromWorld(scene.initial_cam_from_world[index]);
  }
  RequireRejected(
      [&] { Refine(scene); },
      "Coincident camera centers accepted without a metric baseline");
}

void RejectsFullyAnchoredComponent() {
  auto scene = MakeScene();
  for (const size_t index : {size_t{0}, size_t{1}}) {
    for (colmap::point2D_t point_index = 0; point_index < 64; ++point_index) {
      scene.reconstruction.DeleteObservation(scene.image_ids[index],
                                             point_index);
    }
  }
  const auto world_from_first =
      colmap::Inverse(scene.initial_cam_from_world.front());
  const std::array<double, 3> positions = {0.6, 0.7, 0.1};
  for (size_t index = 2; index < scene.image_ids.size(); ++index) {
    const Rigid3d world_from_cam(
        world_from_first.rotation(),
        world_from_first * Eigen::Vector3d(positions[index - 2], 0, 0));
    scene.initial_cam_from_world[index] = colmap::Inverse(world_from_cam);
    scene.reconstruction.Image(scene.image_ids[index])
        .FramePtr()
        ->SetRigFromWorld(scene.initial_cam_from_world[index]);
  }
  // Global farthest is camera3, while component first/farthest are cameras2/4.
  RequireRejected(
      [&] { Refine(scene); },
      "A supported component with no variable pose must be rejected");
}

void RejectsCancellation() {
  for (const size_t cancel_after : {size_t{1}, size_t{2}}) {
    auto scene = MakeScene();
    size_t stop_checks = 0;
    RequireRejected(
        [&] {
          Refine(scene, cm_sparse_default_pose_refinement_options(), [&] {
            return ++stop_checks >= cancel_after;
          });
        },
        "Cancelled refinement returned a result");
    Require(stop_checks >= cancel_after,
            "Cancellation callback was not polled");
  }
}

}  // namespace

int main() {
  const std::array<std::pair<const char*, void (*)()>, 12> tests = {{
      {"pose accuracy, fixed calibration, world frame, and metric scale",
       ImprovesPosesAndPreservesWorldFrame},
      {"largest supported component", RefinesOnlyLargestSupportedComponent},
      {"articulation block scale", PreservesScaleAcrossArticulationBlocks},
      {"weak cameras remain fixed", KeepsWeakCamerasFixed},
      {"insufficient track support", RejectsInsufficientTracks},
      {"coincident orientation observations", CollapsesCoincidentObservations},
      {"ambiguous duplicate observations", DiscardsAmbiguousPoints},
      {"duplicate support inflation", RejectsInflatedDuplicateSupport},
      {"bounded pose corrections", RejectsExcessiveCorrections},
      {"degenerate baseline", RejectsDegenerateBaseline},
      {"fully anchored component", RejectsFullyAnchoredComponent},
      {"cancellation", RejectsCancellation},
  }};
  int failed = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::printf("PASS %s\n", name);
    } catch (const std::exception& error) {
      ++failed;
      std::fprintf(stderr, "FAIL %s: %s\n", name, error.what());
    }
  }
  std::printf("Pose refinement tests: %zu passed, %d failed\n",
              tests.size() - failed,
              failed);
  return failed == 0 ? 0 : 1;
}
