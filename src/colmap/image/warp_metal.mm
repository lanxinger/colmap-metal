// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/image/warp_metal.h"

#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>

#ifndef IMAGE_WARP_METALLIB_PATH
#define IMAGE_WARP_METALLIB_PATH ""
#endif

namespace colmap::internal {
namespace {

constexpr size_t kMaxRemapCacheEntries = 2;
constexpr size_t kMaxRemapCacheBytes = 160 * 1024 * 1024;

bool MetalLanczosDisabled() {
  // Diagnostic escape hatch for quality comparisons and driver workarounds.
  const char* value = std::getenv("COLMAP_DISABLE_METAL_LANCZOS");
  return value != nullptr && std::string_view(value) != "0";
}

struct SourceCoordinate {
  float x;
  float y;
};

static_assert(sizeof(SourceCoordinate) == 2 * sizeof(float));

struct WarpImageParameters {
  uint32_t width;
  uint32_t height;
  uint32_t channels;
  uint32_t pixel_count;
};

struct ResizeImageParameters {
  uint32_t source_width;
  uint32_t source_height;
  uint32_t target_width;
  uint32_t target_height;
  uint32_t channels;
  uint32_t pixel_count;
};

void AddMetalLibraryCandidate(NSMutableArray<NSString*>* paths, NSString* path) {
  if (path.length > 0 && ![paths containsObject:path]) {
    [paths addObject:path];
  }
}

NSArray<NSString*>* MetalLibraryCandidatePaths() {
  NSMutableArray<NSString*>* paths = [NSMutableArray array];
  AddMetalLibraryCandidate(
      paths, [[NSBundle mainBundle] pathForResource:@"image_warp" ofType:@"metallib"]);

  Dl_info image_info;
  if (dladdr(reinterpret_cast<const void*>(&MetalLibraryCandidatePaths), &image_info) != 0 &&
      image_info.dli_fname != nullptr) {
    NSString* image_path = [NSString stringWithUTF8String:image_info.dli_fname];
    NSString* image_dir = [image_path stringByDeletingLastPathComponent];
    AddMetalLibraryCandidate(paths,
                             [image_dir stringByAppendingPathComponent:@"image_warp.metallib"]);
    AddMetalLibraryCandidate(
        paths, [image_dir stringByAppendingPathComponent:@"Resources/image_warp.metallib"]);
    AddMetalLibraryCandidate(
        paths, [image_dir stringByAppendingPathComponent:@"../Resources/image_warp.metallib"]);
    AddMetalLibraryCandidate(
        paths, [image_dir stringByAppendingPathComponent:@"../lib/image_warp.metallib"]);
  }

  AddMetalLibraryCandidate(paths, [NSString stringWithUTF8String:IMAGE_WARP_METALLIB_PATH]);
  return paths;
}

std::string RemapCacheKey(const Camera& source_camera, const Camera& scaled_target_camera) {
  std::string key;
  const auto append_bytes = [&key](const auto& value) {
    key.append(reinterpret_cast<const char*>(&value), sizeof(value));
  };
  const auto append_camera = [&key, &append_bytes](const Camera& camera) {
    append_bytes(camera.model_id);
    append_bytes(camera.width);
    append_bytes(camera.height);
    const size_t num_params = camera.params.size();
    append_bytes(num_params);
    key.append(reinterpret_cast<const char*>(camera.params.data()),
               camera.params.size() * sizeof(double));
  };
  append_camera(source_camera);
  append_camera(scaled_target_camera);
  return key;
}

struct CachedRemap {
  std::string key;
  id<MTLBuffer> buffer = nil;
  size_t num_bytes = 0;
  uint64_t last_used = 0;
};

class MetalImageWarper {
 public:
  bool Warp(const Camera& source_camera,
            const Camera& target_camera,
            const Bitmap& source_image,
            Bitmap* target_image) {
    if (!Initialize()) {
      return false;
    }

    Camera scaled_target_camera = target_camera;
    if (target_camera.width != source_camera.width ||
        target_camera.height != source_camera.height) {
      scaled_target_camera.Rescale(source_camera.width, source_camera.height);
    }

    const std::shared_ptr<CachedRemap> remap =
        GetOrCreateRemap(source_camera, scaled_target_camera);
    if (!remap) {
      return false;
    }

    const bool resize_with_metal = !MetalLanczosDisabled() &&
                                   target_camera.width <= source_camera.width &&
                                   target_camera.height <= source_camera.height &&
                                   (target_camera.width != source_camera.width ||
                                    target_camera.height != source_camera.height);
    Bitmap result(resize_with_metal ? static_cast<int>(target_camera.width)
                                    : static_cast<int>(source_camera.width),
                  resize_with_metal ? static_cast<int>(target_camera.height)
                                    : static_cast<int>(source_camera.height),
                  source_image.IsRGB());
    const size_t source_num_bytes = source_image.NumBytes();

    Timer gpu_timer;
    {
      // One persistent staging pair bounds memory use when the undistortion
      // controller calls this method concurrently from its image thread pool.
      std::lock_guard<std::mutex> lock(execution_mutex_);
      gpu_timer.Start();
      if (!EnsureStagingBuffers(source_num_bytes)) {
        return false;
      }

      if (resize_with_metal && !EnsureResizeBuffers(target_camera.width,
                                                    source_camera.height,
                                                    target_camera.height,
                                                    source_image.Channels())) {
        return false;
      }

      std::memcpy(input_buffer_.contents, source_image.RowMajorData().data(), source_num_bytes);

      id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (!command_buffer || !encoder) {
        return false;
      }

      const size_t pixel_count = source_camera.width * source_camera.height;
      if (pixel_count > std::numeric_limits<uint32_t>::max()) {
        return false;
      }
      const WarpImageParameters params = {
          static_cast<uint32_t>(source_camera.width),
          static_cast<uint32_t>(source_camera.height),
          static_cast<uint32_t>(source_image.Channels()),
          static_cast<uint32_t>(pixel_count),
      };

      [encoder setComputePipelineState:pipeline_];
      [encoder setBuffer:input_buffer_ offset:0 atIndex:0];
      [encoder setBuffer:remap->buffer offset:0 atIndex:1];
      [encoder setBuffer:output_buffer_ offset:0 atIndex:2];
      [encoder setBytes:&params length:sizeof(params) atIndex:3];

      const NSUInteger thread_width =
          std::min<NSUInteger>(pipeline_.maxTotalThreadsPerThreadgroup, 256);
      [encoder dispatchThreads:MTLSizeMake(pixel_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(thread_width, 1, 1)];
      [encoder endEncoding];

      if (resize_with_metal) {
        const size_t horizontal_pixel_count = target_camera.width * source_camera.height;
        const size_t target_pixel_count = target_camera.width * target_camera.height;
        if (horizontal_pixel_count > std::numeric_limits<uint32_t>::max() ||
            target_pixel_count > std::numeric_limits<uint32_t>::max()) {
          return false;
        }

        const ResizeImageParameters horizontal_params = {
            static_cast<uint32_t>(source_camera.width),
            static_cast<uint32_t>(source_camera.height),
            static_cast<uint32_t>(target_camera.width),
            static_cast<uint32_t>(target_camera.height),
            static_cast<uint32_t>(source_image.Channels()),
            static_cast<uint32_t>(horizontal_pixel_count),
        };
        encoder = [command_buffer computeCommandEncoder];
        if (!encoder) {
          return false;
        }
        [encoder setComputePipelineState:resize_horizontal_pipeline_];
        [encoder setBuffer:output_buffer_ offset:0 atIndex:0];
        [encoder setBuffer:resize_intermediate_buffer_ offset:0 atIndex:1];
        [encoder setBytes:&horizontal_params length:sizeof(horizontal_params) atIndex:2];
        const NSUInteger horizontal_thread_width =
            std::min<NSUInteger>(resize_horizontal_pipeline_.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(horizontal_pixel_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(horizontal_thread_width, 1, 1)];
        [encoder endEncoding];

        ResizeImageParameters vertical_params = horizontal_params;
        vertical_params.pixel_count = static_cast<uint32_t>(target_pixel_count);
        encoder = [command_buffer computeCommandEncoder];
        if (!encoder) {
          return false;
        }
        [encoder setComputePipelineState:resize_vertical_pipeline_];
        [encoder setBuffer:resize_intermediate_buffer_ offset:0 atIndex:0];
        [encoder setBuffer:resize_output_buffer_ offset:0 atIndex:1];
        [encoder setBytes:&vertical_params length:sizeof(vertical_params) atIndex:2];
        const NSUInteger vertical_thread_width =
            std::min<NSUInteger>(resize_vertical_pipeline_.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(target_pixel_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(vertical_thread_width, 1, 1)];
        [encoder endEncoding];
      }

      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        LOG(ERROR) << "Metal image warp failed: "
                   << command_buffer.error.localizedDescription.UTF8String;
        return false;
      }

      const id<MTLBuffer> result_buffer =
          resize_with_metal ? resize_output_buffer_ : output_buffer_;
      std::memcpy(result.RowMajorData().data(), result_buffer.contents, result.NumBytes());
    }

    VLOG(1) << "METAL_WARP width=" << source_camera.width << " height=" << source_camera.height
            << " channels=" << source_image.Channels() << " resize_on_gpu=" << resize_with_metal
            << " dispatch_and_copy_s=" << gpu_timer.ElapsedSeconds();

    if (!resize_with_metal && (target_camera.width != source_camera.width ||
                               target_camera.height != source_camera.height)) {
      result.Rescale(target_camera.width, target_camera.height);
    }
    *target_image = std::move(result);
    return true;
  }

 private:
  bool Initialize() {
    std::call_once(initialize_once_, [this]() {
      device_ = MTLCreateSystemDefaultDevice();
      if (!device_) {
        return;
      }
      command_queue_ = [device_ newCommandQueue];
      if (!command_queue_) {
        return;
      }

      NSError* error = nil;
      NSFileManager* file_manager = [NSFileManager defaultManager];
      for (NSString* library_path in MetalLibraryCandidatePaths()) {
        if (![file_manager fileExistsAtPath:library_path]) {
          continue;
        }
        library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:library_path] error:&error];
        if (library_) {
          break;
        }
      }
      if (!library_) {
        library_ = [device_ newDefaultLibrary];
      }
      if (!library_) {
        LOG(ERROR) << "Failed to load Metal image warp library: "
                   << error.localizedDescription.UTF8String;
        return;
      }

      id<MTLFunction> function = [library_ newFunctionWithName:@"warpImageBilinear"];
      if (!function) {
        LOG(ERROR) << "Metal image warp function is missing";
        return;
      }
      pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline_) {
        LOG(ERROR) << "Failed to create Metal image warp pipeline: "
                   << error.localizedDescription.UTF8String;
        return;
      }

      function = [library_ newFunctionWithName:@"resizeLanczosHorizontal"];
      if (!function) {
        LOG(ERROR) << "Metal horizontal Lanczos function is missing";
        return;
      }
      resize_horizontal_pipeline_ = [device_ newComputePipelineStateWithFunction:function
                                                                           error:&error];
      if (!resize_horizontal_pipeline_) {
        LOG(ERROR) << "Failed to create Metal horizontal Lanczos pipeline: "
                   << error.localizedDescription.UTF8String;
        return;
      }

      function = [library_ newFunctionWithName:@"resizeLanczosVertical"];
      if (!function) {
        LOG(ERROR) << "Metal vertical Lanczos function is missing";
        return;
      }
      resize_vertical_pipeline_ = [device_ newComputePipelineStateWithFunction:function
                                                                         error:&error];
      if (!resize_vertical_pipeline_) {
        LOG(ERROR) << "Failed to create Metal vertical Lanczos pipeline: "
                   << error.localizedDescription.UTF8String;
        return;
      }
      initialized_ = true;
    });
    return initialized_;
  }

  std::shared_ptr<CachedRemap> GetOrCreateRemap(const Camera& source_camera,
                                                const Camera& scaled_target_camera) {
    const std::string key = RemapCacheKey(source_camera, scaled_target_camera);
    std::lock_guard<std::mutex> lock(remap_cache_mutex_);
    for (const auto& entry : remap_cache_) {
      if (entry->key == key) {
        entry->last_used = ++cache_tick_;
        return entry;
      }
    }

    Timer build_timer;
    build_timer.Start();
    const size_t pixel_count = source_camera.width * source_camera.height;
    const size_t num_bytes = pixel_count * sizeof(SourceCoordinate);
    id<MTLBuffer> buffer = [device_ newBufferWithLength:num_bytes
                                                options:MTLResourceStorageModeShared];
    if (!buffer) {
      return nullptr;
    }

    auto* coordinates = static_cast<SourceCoordinate*>(buffer.contents);
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    Eigen::Vector2d image_point;
    for (size_t y = 0; y < source_camera.height; ++y) {
      image_point.y() = static_cast<double>(y) + 0.5;
      for (size_t x = 0; x < source_camera.width; ++x) {
        image_point.x() = static_cast<double>(x) + 0.5;
        SourceCoordinate& coordinate = coordinates[y * source_camera.width + x];
        const std::optional<Eigen::Vector2d> cam_point =
            scaled_target_camera.CamFromImg(image_point);
        const std::optional<Eigen::Vector2d> source_point =
            cam_point ? source_camera.ImgFromCam(cam_point->homogeneous()) : std::nullopt;
        if (source_point) {
          coordinate.x = static_cast<float>(source_point->x() - 0.5);
          coordinate.y = static_cast<float>(source_point->y() - 0.5);
        } else {
          coordinate.x = invalid;
          coordinate.y = invalid;
        }
      }
    }

    auto entry = std::make_shared<CachedRemap>();
    entry->key = key;
    entry->buffer = buffer;
    entry->num_bytes = num_bytes;
    entry->last_used = ++cache_tick_;

    while (!remap_cache_.empty() && (remap_cache_.size() >= kMaxRemapCacheEntries ||
                                     remap_cache_bytes_ + num_bytes > kMaxRemapCacheBytes)) {
      const auto oldest = std::min_element(
          remap_cache_.begin(), remap_cache_.end(), [](const auto& lhs, const auto& rhs) {
            return lhs->last_used < rhs->last_used;
          });
      remap_cache_bytes_ -= (*oldest)->num_bytes;
      remap_cache_.erase(oldest);
    }
    remap_cache_.push_back(entry);
    remap_cache_bytes_ += num_bytes;

    VLOG(1) << "METAL_WARP_REMAP width=" << source_camera.width
            << " height=" << source_camera.height << " bytes=" << num_bytes
            << " build_s=" << build_timer.ElapsedSeconds();
    return entry;
  }

  bool EnsureStagingBuffers(const size_t num_bytes) {
    if (staging_capacity_ >= num_bytes) {
      return true;
    }
    input_buffer_ = [device_ newBufferWithLength:num_bytes options:MTLResourceStorageModeShared];
    output_buffer_ = [device_ newBufferWithLength:num_bytes options:MTLResourceStorageModeShared];
    if (!input_buffer_ || !output_buffer_) {
      input_buffer_ = nil;
      output_buffer_ = nil;
      staging_capacity_ = 0;
      return false;
    }
    staging_capacity_ = num_bytes;
    return true;
  }

  bool EnsureResizeBuffers(const size_t target_width,
                           const size_t source_height,
                           const size_t target_height,
                           const size_t channels) {
    // FP16 storage caused visible error amplification after JPEG encoding.
    const size_t intermediate_num_bytes = target_width * source_height * channels * sizeof(float);
    if (resize_intermediate_capacity_ < intermediate_num_bytes) {
      resize_intermediate_buffer_ = [device_ newBufferWithLength:intermediate_num_bytes
                                                         options:MTLResourceStorageModePrivate];
      if (!resize_intermediate_buffer_) {
        resize_intermediate_capacity_ = 0;
        return false;
      }
      resize_intermediate_capacity_ = intermediate_num_bytes;
    }

    const size_t output_num_bytes = target_width * target_height * channels;
    if (resize_output_capacity_ < output_num_bytes) {
      resize_output_buffer_ = [device_ newBufferWithLength:output_num_bytes
                                                   options:MTLResourceStorageModeShared];
      if (!resize_output_buffer_) {
        resize_output_capacity_ = 0;
        return false;
      }
      resize_output_capacity_ = output_num_bytes;
    }
    return true;
  }

  std::once_flag initialize_once_;
  bool initialized_ = false;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> command_queue_ = nil;
  id<MTLLibrary> library_ = nil;
  id<MTLComputePipelineState> pipeline_ = nil;
  id<MTLComputePipelineState> resize_horizontal_pipeline_ = nil;
  id<MTLComputePipelineState> resize_vertical_pipeline_ = nil;

  std::mutex remap_cache_mutex_;
  std::vector<std::shared_ptr<CachedRemap>> remap_cache_;
  size_t remap_cache_bytes_ = 0;
  uint64_t cache_tick_ = 0;

  std::mutex execution_mutex_;
  id<MTLBuffer> input_buffer_ = nil;
  id<MTLBuffer> output_buffer_ = nil;
  size_t staging_capacity_ = 0;
  id<MTLBuffer> resize_intermediate_buffer_ = nil;
  size_t resize_intermediate_capacity_ = 0;
  id<MTLBuffer> resize_output_buffer_ = nil;
  size_t resize_output_capacity_ = 0;
};

MetalImageWarper& ImageWarper() {
  static MetalImageWarper warper;
  return warper;
}

}  // namespace

bool WarpImageBetweenCamerasMetal(const Camera& source_camera,
                                  const Camera& target_camera,
                                  const Bitmap& source_image,
                                  Bitmap* target_image) {
  if (source_camera.width != static_cast<size_t>(source_image.Width()) ||
      source_camera.height != static_cast<size_t>(source_image.Height()) ||
      target_image == nullptr) {
    return false;
  }
  return ImageWarper().Warp(source_camera, target_camera, source_image, target_image);
}

}  // namespace colmap::internal
