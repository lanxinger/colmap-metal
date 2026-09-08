// Copyright (c) 2026 Plinth. SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ColmapSparse.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace colmap_sparse {

// Build tracks from verified database matches, then jointly refine the supplied
// camera poses and points. Images are registered in input order and retained in
// the caller's world frame. Throws on invalid input, cancellation, or
// rejection.
std::shared_ptr<colmap::Reconstruction> TriangulateAndRefineKnownPoses(
    const std::shared_ptr<colmap::Database>& database,
    const std::filesystem::path& image_path,
    const std::vector<colmap::image_t>& image_ids,
    const std::vector<colmap::Rigid3d>& initial_cam_from_world,
    const cm_sparse_pose_refinement_options& options,
    int num_threads,
    const std::function<bool()>& check_if_stopped);

// Internal solver seam. The reconstruction must already contain triangulated
// tracks. Throws on unusable observations, cancellation, or rejected
// refinement. Callers must discard the candidate on failure.
void RefineKnownPoses(
    colmap::Reconstruction& reconstruction,
    const std::vector<colmap::image_t>& image_ids,
    const std::vector<colmap::Rigid3d>& initial_cam_from_world,
    const cm_sparse_pose_refinement_options& options,
    int num_threads,
    const std::function<bool()>& check_if_stopped);

}  // namespace colmap_sparse
