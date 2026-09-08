// Copyright (c) 2026 Plinth. SPDX-License-Identifier: BSD-3-Clause
#include "pose_refinement.h"

#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/sfm/observation_manager.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace colmap_sparse {
namespace {
void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void CleanTracks(colmap::Reconstruction& reconstruction,
                 const std::function<bool()>& check_if_stopped) {
  // COLMAP permits several feature indices from one image in a track. SIFT
  // orientation variants have identical positions; retain one residual for
  // those copies. Conflicting positions are ambiguous evidence, so discard the
  // point instead of choosing an observation using the poses being optimized.
  constexpr double kCoincidentSquaredTolerance = 1e-12;
  const auto point_ids = reconstruction.Point3DIds();
  for (const auto point_id : point_ids) {
    Check(!check_if_stopped || !check_if_stopped(),
          "Pose refinement cancelled");
    auto elements = reconstruction.Point3D(point_id).track.Elements();
    std::sort(
        elements.begin(), elements.end(), [](const auto& a, const auto& b) {
          return std::tie(a.image_id, a.point2D_idx) <
                 std::tie(b.image_id, b.point2D_idx);
        });
    std::vector<colmap::TrackElement> redundant;
    size_t unique_views = 0;
    bool ambiguous = false;
    for (size_t begin = 0; begin < elements.size();) {
      size_t end = begin + 1;
      while (end < elements.size() &&
             elements[end].image_id == elements[begin].image_id)
        ++end;
      ++unique_views;
      const auto& image = reconstruction.Image(elements[begin].image_id);
      for (size_t i = begin; i < end; ++i) {
        const auto& xy = image.Point2D(elements[i].point2D_idx).xy;
        if (!xy.allFinite()) ambiguous = true;
        for (size_t j = i + 1; j < end; ++j) {
          if (elements[i].point2D_idx == elements[j].point2D_idx ||
              (xy - image.Point2D(elements[j].point2D_idx).xy).squaredNorm() >
                  kCoincidentSquaredTolerance)
            ambiguous = true;
        }
        if (i != begin) redundant.push_back(elements[i]);
      }
      begin = end;
    }
    if (ambiguous || unique_views < 3) {
      reconstruction.DeletePoint3D(point_id);
    } else {
      for (const auto& element : redundant) {
        reconstruction.DeleteObservation(element.image_id, element.point2D_idx);
      }
    }
  }
}

std::vector<size_t> LargestSupportedBlock(
    const colmap::Reconstruction& reconstruction,
    const std::vector<colmap::image_t>& image_ids) {
  constexpr size_t kMinSharedTracks = 12;
  const size_t count = image_ids.size();
  colmap::FlatHashMap<colmap::image_t, size_t> indices;
  for (size_t i = 0; i < count; ++i) {
    Check(reconstruction.ExistsImage(image_ids[i]) &&
              reconstruction.Image(image_ids[i]).HasPose(),
          "Known-pose reconstruction lost an input image");
    Check(indices.emplace(image_ids[i], i).second, "Duplicate input image");
  }
  std::vector<size_t> shared(count * count, 0);
  for (const auto& [id, point] : reconstruction.Points3D()) {
    Check(point.xyz.allFinite(), "Nonfinite triangulated point");
    if (point.track.Length() < 3) continue;
    std::vector<size_t> visible;
    for (const auto& element : point.track.Elements()) {
      const auto it = indices.find(element.image_id);
      Check(it != indices.end(), "Track references a non-input image");
      const auto camera_point =
          reconstruction.Image(element.image_id).CamFromWorld() * point.xyz;
      Check(camera_point.allFinite() && camera_point.z() > 0,
            "Refinement produced an observation behind its camera");
      visible.push_back(it->second);
    }
    std::sort(visible.begin(), visible.end());
    Check(std::adjacent_find(visible.begin(), visible.end()) == visible.end(),
          "Track contains duplicate image observations");
    for (const auto i : visible) {
      for (const auto j : visible) ++shared[i * count + j];
    }
  }
  // A camera shared by otherwise separate groups does not constrain their
  // relative scale. Select one vertex-biconnected block and keep all cameras
  // outside it fixed, rather than freeing both sides of such a connection.
  std::vector<size_t> largest;
  std::vector<int> discovery(count, -1), low(count, -1);
  std::vector<std::pair<size_t, size_t>> edges;
  int next_discovery = 0;
  const std::function<void(size_t, size_t)> visit = [&](size_t i,
                                                        size_t parent) {
    discovery[i] = low[i] = next_discovery++;
    for (size_t j = 0; j < count; ++j) {
      if (i == j || shared[i * count + j] < kMinSharedTracks) continue;
      if (discovery[j] == -1) {
        edges.emplace_back(i, j);
        visit(j, i);
        low[i] = std::min(low[i], low[j]);
        if (low[j] >= discovery[i]) {
          std::vector<size_t> block;
          std::pair<size_t, size_t> edge;
          do {
            edge = edges.back();
            edges.pop_back();
            block.push_back(edge.first);
            block.push_back(edge.second);
          } while (edge != std::make_pair(i, j));
          std::sort(block.begin(), block.end());
          block.erase(std::unique(block.begin(), block.end()), block.end());
          // Capture order breaks ties, including blocks sharing a camera.
          if (block.size() > largest.size() ||
              (block.size() == largest.size() && block < largest)) {
            largest = std::move(block);
          }
        }
      } else if (j != parent && discovery[j] < discovery[i]) {
        edges.emplace_back(i, j);
        low[i] = std::min(low[i], discovery[j]);
      }
    }
  };
  for (size_t i = 0; i < count; ++i) {
    if (discovery[i] == -1) visit(i, count);
  }
  Check(largest.size() >= 3,
        "No group of at least three cameras has 12 shared multi-view tracks "
        "per connection without relying on a single connecting camera");
  return largest;
}

size_t FarthestPose(const std::vector<colmap::Rigid3d>& cam_from_world,
                    const std::vector<size_t>& indices) {
  const Eigen::Vector3d origin =
      colmap::Inverse(cam_from_world[indices.front()]).translation();
  size_t anchor = indices.front();
  double max_baseline = 0;
  for (const auto i : indices) {
    const double baseline =
        (colmap::Inverse(cam_from_world[i]).translation() - origin).norm();
    if (baseline > max_baseline) {
      max_baseline = baseline;
      anchor = i;
    }
  }
  Check(std::isfinite(max_baseline) && max_baseline > 1e-6,
        "Known poses have no usable metric baseline");
  return anchor;
}
}  // namespace

void RefineKnownPoses(
    colmap::Reconstruction& reconstruction,
    const std::vector<colmap::image_t>& image_ids,
    const std::vector<colmap::Rigid3d>& initial_cam_from_world,
    const cm_sparse_pose_refinement_options& options,
    int num_threads,
    const std::function<bool()>& check_if_stopped) {
  Check(image_ids.size() >= 3 &&
            image_ids.size() == initial_cam_from_world.size(),
        "Pose refinement requires at least three matching input poses");
  const auto check_stop = [&] {
    Check(!check_if_stopped || !check_if_stopped(),
          "Pose refinement cancelled");
  };
  check_stop();
  colmap::ObservationManager(reconstruction)
      .FilterObservationsWithNegativeDepth();
  CleanTracks(reconstruction, check_if_stopped);
  const auto component = LargestSupportedBlock(reconstruction, image_ids);
  std::vector<size_t> all_indices;
  for (size_t i = 0; i < image_ids.size(); ++i) all_indices.push_back(i);
  std::vector<bool> variable(image_ids.size(), false);
  for (const auto i : component) variable[i] = true;
  variable[0] = false;
  variable[FarthestPose(initial_cam_from_world, all_indices)] = false;
  variable[component.front()] = false;
  variable[FarthestPose(initial_cam_from_world, component)] = false;
  Check(std::any_of(
            variable.begin(), variable.end(), [](bool value) { return value; }),
        "Supported camera group has no variable poses after anchoring");

  colmap::BundleAdjustmentConfig config;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    config.AddImage(image_ids[i]);
    if (!variable[i]) {
      config.SetConstantRigFromWorldPose(
          reconstruction.Image(image_ids[i]).FrameId());
    }
  }
  // The selected block has its own two fixed poses; cameras in all other
  // blocks remain fixed even when they share a camera with this block.
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  colmap::BundleAdjustmentOptions ba;
  ba.refine_focal_length = false;
  ba.refine_principal_point = false;
  ba.refine_extra_params = false;
  ba.refine_sensor_from_rig = false;
  ba.refine_rig_from_world = true;
  ba.min_track_length = 3;
  ba.print_summary = false;
  ba.check_if_stopped = check_if_stopped;
  ba.ceres->loss_function_type =
      colmap::CeresBundleAdjustmentOptions::LossFunctionType::HUBER;
  ba.ceres->loss_function_scale = 1.0;
  ba.ceres->solver_options.max_num_iterations = options.max_num_iterations;
  ba.ceres->solver_options.num_threads = num_threads;
  ba.ceres->solver_options.sparse_linear_algebra_library_type =
      ceres::ACCELERATE_SPARSE;
  ba.ceres->solver_options.dense_linear_algebra_library_type = ceres::EIGEN;
  ba.ceres->solver_options.logging_type = ceres::SILENT;

  reconstruction.UpdatePoint3DErrors();
  const double before = reconstruction.ComputeMeanReprojectionError();
  Check(std::isfinite(before), "Invalid initial reprojection error");
  auto adjuster =
      colmap::CreateDefaultBundleAdjuster(ba, config, reconstruction);
  const auto summary = adjuster->Solve();
  check_stop();
  Check(summary && summary->IsSolutionUsable(),
        "Known-pose bundle adjustment failed");
  reconstruction.UpdatePoint3DErrors();
  const double after = reconstruction.ComputeMeanReprojectionError();
  Check(std::isfinite(after) && after <= before + 1e-6,
        "Known-pose refinement increased reprojection error");

  // Approximate input poses can put valid tracks outside the strict pixel
  // threshold. Apply it only after poses and points have been refined jointly,
  // then require the optimized group to retain its supporting observations.
  colmap::ObservationManager(reconstruction).FilterAllPoints3D(4.0, 1.5);
  CleanTracks(reconstruction, check_if_stopped);
  check_stop();
  reconstruction.UpdatePoint3DErrors();
  Check(LargestSupportedBlock(reconstruction, image_ids) == component,
        "Refinement changed the supported camera group");
  for (size_t i = 0; i < image_ids.size(); ++i) {
    const auto refined =
        colmap::Inverse(reconstruction.Image(image_ids[i]).CamFromWorld());
    const auto original = colmap::Inverse(initial_cam_from_world[i]);
    Check(refined.params.allFinite(), "Nonfinite refined camera pose");
    const double translation =
        (refined.translation() - original.translation()).norm();
    const double rotation =
        refined.rotation().angularDistance(original.rotation());
    if (!variable[i]) {
      Check(translation <= 1e-10 && rotation <= 1e-10,
            "Refinement changed a fixed camera pose");
    }
    Check(std::isfinite(translation) &&
              translation <= options.max_translation_change &&
              std::isfinite(rotation) &&
              rotation <= options.max_rotation_change_radians,
          "Known-pose refinement exceeded the camera correction limit");
  }
}

std::shared_ptr<colmap::Reconstruction> TriangulateAndRefineKnownPoses(
    const std::shared_ptr<colmap::Database>& database,
    const std::filesystem::path& image_path,
    const std::vector<colmap::image_t>& image_ids,
    const std::vector<colmap::Rigid3d>& initial_cam_from_world,
    const cm_sparse_pose_refinement_options& options,
    int num_threads,
    const std::function<bool()>& check_if_stopped) {
  Check(database && image_ids.size() >= 3 &&
            image_ids.size() == initial_cam_from_world.size(),
        "Pose refinement requires a database and matching input poses");
  const auto check_stop = [&] {
    Check(!check_if_stopped || !check_if_stopped(),
          "Pose refinement cancelled");
  };
  check_stop();
  auto pipeline_options =
      std::make_shared<colmap::IncrementalPipelineOptions>();
  pipeline_options->image_path = image_path;
  pipeline_options->num_threads = num_threads;
  pipeline_options->random_seed = 0;
  pipeline_options->load_all_images = true;
  pipeline_options->fix_existing_frames = true;
  pipeline_options->ba_refine_focal_length = false;
  pipeline_options->ba_refine_principal_point = false;
  pipeline_options->ba_refine_extra_params = false;
  pipeline_options->ba_refine_sensor_from_rig = false;
  // Initial poses can explain several pixels of residual error even when a
  // small camera correction recovers the true solution. Keep those tracks for
  // the joint solve; its final filtering checks the refined geometry.
  pipeline_options->ba_global_max_refinements = 0;
  pipeline_options->ba_sparse_linear_algebra_library_type = "ACCELERATE_SPARSE";
  pipeline_options->ba_dense_linear_algebra_library_type = "EIGEN";
  auto manager = std::make_shared<colmap::ReconstructionManager>();
  colmap::IncrementalPipeline pipeline(pipeline_options, database, manager);
  pipeline.SetCheckIfStoppedFunc(check_if_stopped);
  auto reconstruction = std::make_shared<colmap::Reconstruction>();
  reconstruction->Load(*pipeline.DatabaseCache());
  colmap::FlatHashSet<colmap::image_t> registered_images;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    check_stop();
    const auto image_id = image_ids[i];
    Check(registered_images.insert(image_id).second, "Duplicate input image");
    Check(reconstruction->ExistsImage(image_id),
          "Known-pose reconstruction is missing an input image");
    const auto& image = reconstruction->Image(image_id);
    reconstruction->Frame(image.FrameId())
        .SetRigFromWorld(initial_cam_from_world[i]);
    reconstruction->RegisterFrame(image.FrameId());
  }
  pipeline.TriangulateReconstruction(reconstruction);
  check_stop();
  RefineKnownPoses(*reconstruction,
                   image_ids,
                   initial_cam_from_world,
                   options,
                   num_threads,
                   check_if_stopped);
  check_stop();
  Check(reconstruction->NumRegImages() == image_ids.size(),
        "Known-pose refinement did not retain every input image");
  return reconstruction;
}
}  // namespace colmap_sparse
