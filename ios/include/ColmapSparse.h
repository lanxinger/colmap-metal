// Copyright (c) 2026 Plinth. SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CM_SPARSE_ABI_VERSION 1

typedef struct cm_sparse_job cm_sparse_job;

typedef enum cm_sparse_status {
  CM_SPARSE_SUCCESS = 0,
  CM_SPARSE_CANCELLED = 1,
  CM_SPARSE_INVALID_ARGUMENT = 2,
  CM_SPARSE_RESOURCE_ERROR = 3,
  CM_SPARSE_RECONSTRUCTION_FAILED = 4,
  CM_SPARSE_IO_ERROR = 5,
  CM_SPARSE_INTERNAL_ERROR = 6
} cm_sparse_status;

typedef enum cm_sparse_camera_model {
  CM_SPARSE_PINHOLE = 1,        // fx, fy, cx, cy
  CM_SPARSE_SIMPLE_RADIAL = 2,  // f, cx, cy, k1
  CM_SPARSE_RADIAL = 3,         // f, cx, cy, k1, k2
  CM_SPARSE_OPENCV = 4          // fx, fy, cx, cy, k1, k2, p1, p2
} cm_sparse_camera_model;

typedef struct cm_sparse_camera {
  uint32_t calibration_id;
  cm_sparse_camera_model model;
  uint32_t width;
  uint32_t height;
  double params[8];
} cm_sparse_camera;

// Camera dimensions/intrinsics describe the encoded image raster, BEFORE EXIF
// rotation. Pixel coordinates use COLMAP's image-edge convention. Images
// sharing a calibration_id must have exactly the same calibration. Paths are
// copied.
typedef struct cm_sparse_image {
  const char* path;
  cm_sparse_camera camera;
} cm_sparse_image;

// Column-major rigid camera-to-world transform. Camera axes are x right,
// y down, z forward (COLMAP/OpenCV), and translations use the caller's world
// units. The last row must be [0, 0, 0, 1].
typedef struct cm_sparse_pose {
  double camera_to_world[16];
} cm_sparse_pose;

typedef struct cm_sparse_pose_refinement_options {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t max_num_iterations;         // 1..100, default 20 per BA solve.
  double max_translation_change;       // (0, 10] world units, default 0.15.
  double max_rotation_change_radians;  // (0, 0.5], default 5 degrees.
} cm_sparse_pose_refinement_options;

typedef struct cm_sparse_options {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t max_image_size;
  // Descriptor rows per image, capped after orientation expansion (128..8192).
  uint32_t max_num_features;
  uint32_t num_threads;
  uint32_t sequential_overlap;
  // Every Nth input plus the last image is matched to the other keyframes.
  // Zero disables these nonlocal pairs. Input array order is capture order.
  uint32_t keyframe_stride;
  uint32_t max_images;
  uint32_t max_num_pairs;
  // Cooperative wall-clock limit across preparation, extraction, matching and
  // mapping. A running GPU command/solver step can finish after this deadline.
  uint32_t max_runtime_seconds;
  int32_t first_octave;
  uint64_t matching_cache_bytes;
  double minimum_registered_fraction;
  // If zero, supplied calibration is fixed during bundle adjustment.
  uint32_t refine_intrinsics;
} cm_sparse_options;

typedef enum cm_sparse_stage {
  CM_SPARSE_PREPARING = 0,
  CM_SPARSE_EXTRACTING = 1,
  CM_SPARSE_MATCHING = 2,
  CM_SPARSE_MAPPING = 3,
  CM_SPARSE_EXPORTING = 4,
  CM_SPARSE_FINISHED = 5
} cm_sparse_stage;

// Called synchronously on the run thread. May request cancellation, but must
// not destroy/run the job, throw, or block on the thread running the
// reconstruction.
typedef void (*cm_sparse_progress_callback)(void* context,
                                            cm_sparse_stage stage,
                                            uint32_t completed,
                                            uint32_t total);

typedef struct cm_sparse_result {
  uint32_t registered_images;
  uint32_t input_images;
  uint64_t points;
  uint64_t observations;
  double mean_reprojection_error;
  double mean_track_length;
  double elapsed_seconds;
} cm_sparse_result;

uint32_t cm_sparse_abi_version(void);
cm_sparse_options cm_sparse_default_options(void);
cm_sparse_pose_refinement_options cm_sparse_default_pose_refinement_options(
    void);

// Creates a single-use job, copying inputs/options/paths synchronously. Output
// must not exist. A successful job atomically publishes images/ and sparse/0/.
// On failure/cancellation no final output is published. Existing files are
// never overwritten. metallib_path must identify the package's
// platform-specific SIFT metallib. error_buffer receives a NUL-terminated
// diagnostic if creation fails. The output's parent must already exist. Inputs
// are JPEG/PNG/HEIC, at most 16 MP, and 32..8192 pixels per axis. Published
// filenames preserve capture-array order; only the largest connected model and
// its registered images are exported.
cm_sparse_status cm_sparse_job_create(const cm_sparse_image* images,
                                      size_t image_count,
                                      const cm_sparse_options* options,
                                      const char* output_path,
                                      const char* metallib_path,
                                      cm_sparse_job** job,
                                      char* error_buffer,
                                      size_t error_capacity);

// Refines known poses through feature matching, fixed-pose triangulation, then
// bundle adjustment. At least three inputs are required; all inputs and poses
// are copied in capture order. Images must have absent/up EXIF orientation and
// refine_intrinsics must be zero. The first pose and the pose farthest from it
// are fixed anchors, preserving the input world frame and metric baseline.
// Only the largest strongly supported group can move, with its own first and
// farthest poses fixed. Groups joined through only one camera are treated
// separately. Other cameras retain their input poses. A failed solve,
// insufficient supported cameras, or excessive correction rejects the candidate.
// Successful output retains all input images and fixed camera calibrations.
cm_sparse_status cm_sparse_job_create_with_poses(
    const cm_sparse_image* images,
    const cm_sparse_pose* poses,
    size_t image_count,
    const cm_sparse_options* options,
    const cm_sparse_pose_refinement_options* refinement_options,
    const char* output_path,
    const char* metallib_path,
    cm_sparse_job** job,
    char* error_buffer,
    size_t error_capacity);

// Call after a successful known-pose run, before destroying the job. Copies
// poses in original input order. pose_count must equal the input image count.
// Returns INVALID_ARGUMENT for a failed/unfinished/ordinary job or wrong count.
cm_sparse_status cm_sparse_job_copy_refined_poses(const cm_sparse_job* job,
                                                  cm_sparse_pose* poses,
                                                  size_t pose_count);

// Synchronous; call on a dedicated background executor. Only one run may use a
// job. result is populated only on success. No C++ exception crosses this API.
// Only one job can run process-wide; a concurrent run returns RESOURCE_ERROR.
cm_sparse_status cm_sparse_job_run(cm_sparse_job* job,
                                   cm_sparse_progress_callback progress,
                                   void* context,
                                   cm_sparse_result* result);
// Thread-safe, cooperative cancellation. A submitted Metal command completes
// before cancellation is observed. Mapping also propagates stop to Ceres BA.
void cm_sparse_job_cancel(cm_sparse_job* job);
// Read after run returns; pointer remains valid until destruction.
const char* cm_sparse_job_error(const cm_sparse_job* job);
// Caller must wait for run to return before destruction. Never call in
// callback.
void cm_sparse_job_destroy(cm_sparse_job* job);

#ifdef __cplusplus
}
#endif
