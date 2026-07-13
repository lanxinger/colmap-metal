// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include "colmap/scene/camera.h"
#include "colmap/sensor/bitmap.h"

namespace colmap::internal {

// Warps an image with Metal using the same inverse camera mapping as
// WarpImageBetweenCameras. Returns false when Metal is unavailable or the GPU
// operation fails, so callers can fall back to the CPU implementation.
bool WarpImageBetweenCamerasMetal(const Camera& source_camera,
                                  const Camera& target_camera,
                                  const Bitmap& source_image,
                                  Bitmap* target_image);

}  // namespace colmap::internal
