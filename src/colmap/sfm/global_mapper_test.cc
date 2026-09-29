#include "colmap/sfm/global_mapper.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// TODO(jsch): Add tests for pose priors.

std::shared_ptr<DatabaseCache> CreateDatabaseCache(const Database& database) {
  DatabaseCache::Options options;
  return DatabaseCache::Create(database, options);
}

TEST(GlobalMapper, WithoutNoise) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialKnownRig) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialUnknownRig) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise

  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  // Set the rig sensors to be unknown
  for (const auto& [rig_id, rig] : reconstruction->Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction->Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithNoiseAndOutliers) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.7;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(GlobalMapper, EstablishTracksLimitsDistinctViews) {
  for (const int num_views : {100, 101}) {
    for (const bool repeat_image : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "num_views=" << num_views
                                        << ", repeat_image=" << repeat_image);
      SetPRNGSeed(1);
      auto reconstruction = std::make_shared<Reconstruction>();
      SyntheticDatasetOptions dataset_options;
      dataset_options.num_rigs = 1;
      dataset_options.num_cameras_per_rig = 1;
      dataset_options.num_frames_per_rig = num_views;
      dataset_options.num_points3D = 0;
      dataset_options.num_points2D_without_point3D = 2;
      SynthesizeDataset(dataset_options, reconstruction.get());

      auto database_cache = std::make_shared<DatabaseCache>();
      for (const auto& [camera_id, camera] : reconstruction->Cameras()) {
        database_cache->AddCamera(camera);
      }
      for (const auto& [rig_id, rig] : reconstruction->Rigs()) {
        database_cache->AddRig(rig);
      }
      for (const auto& [frame_id, frame] : reconstruction->Frames()) {
        database_cache->AddFrame(frame);
      }
      for (const auto& [image_id, image] : reconstruction->Images()) {
        // Two nearby features in one image may belong to the same track.
        reconstruction->Image(image_id).Point2D(0).xy = {100.0, 100.0};
        reconstruction->Image(image_id).Point2D(1).xy = {101.0, 100.0};
        database_cache->AddImage(image);
      }
      auto correspondence_graph = database_cache->CorrespondenceGraph();
      for (image_t image_id = 1; image_id < num_views; ++image_id) {
        TwoViewGeometry geometry;
        geometry.config = TwoViewGeometry::CALIBRATED;
        geometry.cam2_from_cam1 = Rigid3d();
        geometry.inlier_matches.emplace_back(0, 0);
        if (repeat_image && image_id == 1) {
          geometry.inlier_matches.emplace_back(1, 0);
        }
        correspondence_graph->AddTwoViewGeometry(
            image_id, image_id + 1, std::move(geometry));
      }
      correspondence_graph->Finalize();

      GlobalMapper mapper(database_cache);
      mapper.BeginReconstruction(reconstruction);
      mapper.EstablishTracks(GlobalMapperOptions());

      if (num_views == 100) {
        ASSERT_EQ(reconstruction->NumPoints3D(), 1);
        EXPECT_EQ(reconstruction->Points3D().begin()->second.track.Length(),
                  num_views + static_cast<int>(repeat_image));
      } else {
        EXPECT_EQ(reconstruction->NumPoints3D(), 0);
      }
    }
  }
}

TEST(GlobalMapper, RetriangulationRefinementKeepsCamerasFixedWithOutliers) {
  SetPRNGSeed(1);
  auto database = Database::Open(CreateTestDir() / "database.db");
  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 2;
  dataset_options.num_frames_per_rig = 7;
  dataset_options.num_points3D = 100;
  dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(dataset_options, reconstruction.get(), database.get());

  // Keep permissive triangulation correspondences that the later 4 px filter
  // rejects. Joint L2 refinement must not move cameras to fit these outliers.
  auto keypoints = database->ReadKeypoints(1);
  for (point2D_t point2D_idx = 0; point2D_idx < keypoints.size();
       point2D_idx += 3) {
    keypoints[point2D_idx].x += 8.0;
  }
  database->UpdateKeypoints(1, keypoints);

  GlobalMapper mapper(CreateDatabaseCache(*database));
  mapper.BeginReconstruction(reconstruction);
  const Reconstruction initial_reconstruction = *reconstruction;

  struct CameraChangeCallback : ceres::IterationCallback {
    CameraChangeCallback(const Reconstruction& reconstruction,
                         const Reconstruction& initial_reconstruction)
        : reconstruction(reconstruction),
          initial_reconstruction(initial_reconstruction) {}

    ceres::CallbackReturnType operator()(
        const ceres::IterationSummary& summary) override {
      ++num_callbacks;
      num_successful_steps +=
          summary.iteration > 0 && summary.step_is_successful;
      for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
        const auto& initial_camera = initial_reconstruction.Camera(camera_id);
        for (size_t i = 0; i < camera.params.size(); ++i) {
          max_camera_change =
              std::max(max_camera_change,
                       std::abs(camera.params[i] - initial_camera.params[i]));
        }
      }
      for (const auto& [frame_id, frame] : reconstruction.Frames()) {
        max_pose_change = std::max(
            max_pose_change,
            (frame.RigFromWorld().ToMatrix() -
             initial_reconstruction.Frame(frame_id).RigFromWorld().ToMatrix())
                .norm());
      }
      for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
        for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
          max_pose_change = std::max(
              max_pose_change,
              (sensor_from_rig->ToMatrix() - initial_reconstruction.Rig(rig_id)
                                                 .SensorFromRig(sensor_id)
                                                 .ToMatrix())
                  .norm());
        }
      }
      return ceres::SOLVER_CONTINUE;
    }

    const Reconstruction& reconstruction;
    const Reconstruction& initial_reconstruction;
    int num_callbacks = 0;
    int num_successful_steps = 0;
    double max_camera_change = 0.0;
    double max_pose_change = 0.0;
  } callback(*reconstruction, initial_reconstruction);

  GlobalMapperOptions options;
  auto ba_options = options.BundleAdjustment();
  ba_options.refine_principal_point = true;
  ba_options.ceres->use_gpu = false;
  ba_options.ceres->solver_options.num_threads = 1;
  // Only the inner refinement overrides this cap. Observe its live parameter
  // state before outer BA or final normalization can change camera poses.
  ba_options.ceres->solver_options.max_num_iterations = 0;
  ba_options.ceres->solver_options.update_state_every_iteration = true;
  ba_options.ceres->solver_options.callbacks.push_back(&callback);

  ASSERT_TRUE(mapper.IterativeRetriangulateAndRefine(
      options.Retriangulation(),
      ba_options,
      options.max_normalized_reproj_error,
      options.min_tri_angle_deg));
  EXPECT_GT(callback.num_callbacks, 0);
  EXPECT_GT(callback.num_successful_steps, 0);
  EXPECT_LT(callback.max_camera_change, 1e-12);
  EXPECT_LT(callback.max_pose_change, 1e-12);
  EXPECT_GT(reconstruction->NumPoints3D(), 0);
}

TEST(GlobalMapper, RetriangulationRefinesShortTracksAfterCameraAdjustment) {
  SetPRNGSeed(1);
  auto database = Database::Open(CreateTestDir() / "database.db");
  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = 7;
  dataset_options.num_points3D = 100;
  dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(dataset_options, reconstruction.get(), database.get());

  // Leave one point visible in just two views. The other long tracks constrain
  // the subsequent joint camera refinement, which excludes two-view points.
  const point3D_t short_point3D_id = reconstruction->Points3D().begin()->first;
  point2D_t short_point2D_idx = kInvalidPoint2DIdx;
  for (const auto& element :
       reconstruction->Point3D(short_point3D_id).track.Elements()) {
    if (element.image_id == 3) {
      short_point2D_idx = element.point2D_idx;
    }
  }
  ASSERT_NE(short_point2D_idx, kInvalidPoint2DIdx);
  for (auto& [pair_id, geometry] : database->ReadTwoViewGeometries()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (image_id1 == 3 && image_id2 == 4) {
      continue;
    }
    const auto& image1 = reconstruction->Image(image_id1);
    auto& matches = geometry.inlier_matches;
    matches.erase(
        std::remove_if(matches.begin(),
                       matches.end(),
                       [&](const FeatureMatch& match) {
                         return image1.Point2D(match.point2D_idx1).point3D_id ==
                                short_point3D_id;
                       }),
        matches.end());
    database->UpdateTwoViewGeometry(image_id1, image_id2, geometry);
  }

  // A small pose error keeps the initial observations within the outlier gate
  // but leaves two-view structure stale once long tracks correct this camera.
  reconstruction->Frame(reconstruction->Image(3).FrameId())
      .RigFromWorld()
      .translation()
      .x() += 0.01;

  GlobalMapper mapper(CreateDatabaseCache(*database));
  mapper.BeginReconstruction(reconstruction);
  GlobalMapperOptions options;
  auto tri_options = options.Retriangulation();
  tri_options.ignore_two_view_tracks = false;
  auto ba_options = options.BundleAdjustment();
  ba_options.refine_focal_length = false;
  ba_options.refine_extra_params = false;
  ba_options.ceres->use_gpu = false;
  ba_options.ceres->solver_options.num_threads = 1;

  ASSERT_TRUE(mapper.IterativeRetriangulateAndRefine(
      tri_options,
      ba_options,
      options.max_normalized_reproj_error,
      options.min_tri_angle_deg));
  const auto& short_observation =
      reconstruction->Image(3).Point2D(short_point2D_idx);
  ASSERT_TRUE(short_observation.HasPoint3D());
  const auto& short_point =
      reconstruction->Point3D(short_observation.point3D_id);
  ASSERT_EQ(short_point.track.Length(), 2);
  for (const auto& element : short_point.track.Elements()) {
    const auto& image = reconstruction->Image(element.image_id);
    EXPECT_LT(
        CalculateSquaredReprojectionError(image.Point2D(element.point2D_idx).xy,
                                          short_point.xyz,
                                          image.CamFromWorld(),
                                          *image.CameraPtr()),
        0.02 * 0.02);
  }
}

TEST(GlobalMapperOptions, RefineSensorFromRigPropagatesToSubOptions) {
  GlobalMapperOptions options;
  options.refine_sensor_from_rig = false;
  // Sub-options keep their own defaults (true) until accessed.
  EXPECT_TRUE(options.rotation_averaging.refine_sensor_from_rig);
  EXPECT_TRUE(options.global_positioning.refine_sensor_from_rig);
  EXPECT_TRUE(options.bundle_adjustment.refine_sensor_from_rig);
  // Accessors return resolved sub-options with the top-level flag applied.
  EXPECT_FALSE(options.RotationAveraging().refine_sensor_from_rig);
  EXPECT_FALSE(options.GlobalPositioning().refine_sensor_from_rig);
  EXPECT_FALSE(options.BundleAdjustment().refine_sensor_from_rig);
}

}  // namespace
}  // namespace colmap
