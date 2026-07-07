// SiftMetal.mm - Metal-accelerated SIFT feature extraction.
// Objective-C++ port of SIFTMetal Swift library by Luke Van In.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "SiftMetal.h"

// Shared C headers for Metal shader parameter structs.
#include "include/ConvolutionSeries.h"
#include "include/NearestNeighbor.h"
#include "include/SIFTDescriptor.h"
#include "include/SIFTExtrema.h"
#include "include/SIFTInterpolate.h"
#include "include/SIFTMatcher.h"
#include "include/SIFTOrientation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// Path to compiled metallib is set by CMake and embedded here.
#ifndef SIFT_METAL_METALLIB_PATH
#define SIFT_METAL_METALLIB_PATH ""
#endif

namespace sift_metal {

static constexpr int kMinExtremaCapacity = 4096;
static constexpr int kMinKeypointCapacity = 4096;
static constexpr int kMinDescriptorCapacity = 8192;

// ---------------------------------------------------------------------------
// Helper: compute 1D Gaussian kernel weights.
// ---------------------------------------------------------------------------
static std::vector<float> GaussianWeights(float sigma) {
  int radius = static_cast<int>(std::ceil(4.0f * sigma));
  int size = radius * 2 + 1;
  std::vector<float> weights(size);
  float sum = 0.0f;
  float ss = sigma * sigma;
  for (int k = -radius; k <= radius; ++k) {
    float w = std::exp(-0.5f * (float(k * k) / ss));
    weights[k + radius] = w;
    sum += w;
  }
  for (auto& w : weights) w /= sum;
  return weights;
}

// ---------------------------------------------------------------------------
// Octave: manages textures and pipelines for one octave of the pyramid.
// ---------------------------------------------------------------------------
struct Octave {
  int o = 0;                 // octave index
  float delta = 0.0f;        // sampling distance
  int width = 0, height = 0; // dimensions at this octave
  int num_scales = 0;        // scales per octave (typically 3)
  std::vector<float> sigmas;  // sigma values for each gaussian

  id<MTLTexture> gaussianTextures = nil;   // 2DArray [num_scales+3]
  id<MTLTexture> differenceTextures = nil; // 2DArray [num_scales+2]
  id<MTLBuffer> downscaleParamsBuffer = nil;

  // Buffers for extrema detection
  id<MTLBuffer> extremaOutputBuffer = nil;
  id<MTLBuffer> extremaIndexBuffer = nil;
  id<MTLBuffer> extremaParamsBuffer = nil;

  // Buffers for interpolation
  id<MTLBuffer> interpolateOutputBuffer = nil;
  id<MTLBuffer> interpolateParamsBuffer = nil;

  // Buffers for orientation
  id<MTLBuffer> orientationInputBuffer = nil;
  id<MTLBuffer> orientationOutputBuffer = nil;
  id<MTLBuffer> orientationParamsBuffer = nil;

  // Buffers for descriptors
  id<MTLBuffer> descriptorInputBuffer = nil;
  id<MTLBuffer> descriptorOutputBuffer = nil;
  id<MTLBuffer> descriptorParamsBuffer = nil;

  // Convolution kernel weights buffers for Gaussian series blur
  struct ConvPair {
    id<MTLBuffer> paramsX = nil;
    id<MTLBuffer> paramsY = nil;
  };
  std::vector<ConvPair> convPairs;
  id<MTLTexture> convWorkTexture = nil; // private storage 2DArray[1]
};

struct DetectedKeypoint {
  Keypoint keypoint;
  int scale = 0;
  float sub_scale = 0.0f;
};

static bool OctaveResourcesReady(const Octave& oct) {
  if (!oct.gaussianTextures || !oct.differenceTextures ||
      !oct.downscaleParamsBuffer || !oct.convWorkTexture ||
      !oct.extremaOutputBuffer || !oct.extremaIndexBuffer ||
      !oct.extremaParamsBuffer ||
      !oct.interpolateOutputBuffer || !oct.interpolateParamsBuffer ||
      !oct.orientationInputBuffer || !oct.orientationOutputBuffer ||
      !oct.orientationParamsBuffer || !oct.descriptorInputBuffer ||
      !oct.descriptorOutputBuffer || !oct.descriptorParamsBuffer) {
    return false;
  }
  for (const auto& pair : oct.convPairs) {
    if (!pair.paramsX || !pair.paramsY) {
      return false;
    }
  }
  return true;
}

static bool OctavesResourcesReady(const std::vector<Octave>& octaves) {
  for (const auto& oct : octaves) {
    if (!OctaveResourcesReady(oct)) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// SiftMetalExtractorImpl
// ---------------------------------------------------------------------------
class SiftMetalExtractorImpl {
 public:
  bool Init(const Options& opts, int max_w, int max_h);
  bool Extract(const uint8_t* data, int w, int h, ExtractResult* result);

 private:
  void SetupOctaves(int w, int h);
  void SetupOctave(Octave& oct, int o, float delta, int w, int h,
                   int num_scales, const std::vector<float>& sigmas);

  // Pipeline encoding helpers
  void EncodeGrayscaleUpload(id<MTLCommandBuffer> cb, int w, int h);
  bool EncodeSeedTexture(id<MTLCommandBuffer> cb);
  bool EncodeOctave(id<MTLCommandBuffer> cb, Octave& oct,
                    id<MTLTexture> inputTexture, bool inputIs2D);
  bool EncodeGaussianSeries(id<MTLCommandBuffer> cb, Octave& oct);
  bool EncodeDifferences(id<MTLCommandBuffer> cb, Octave& oct);
  bool EncodeExtrema(id<MTLCommandBuffer> cb, Octave& oct);

  // Per-octave extraction. Each stage encodes its GPU work for all octaves
  // into one shared command buffer, so Extract synchronizes with the GPU
  // once per stage instead of once per octave and stage.
  int ReadExtremaCount(Octave& oct);
  bool EncodeInterpolateKeypoints(id<MTLCommandBuffer> cb, Octave& oct,
                                  int extrema_count);
  void ReadInterpolatedKeypoints(Octave& oct, int extrema_count,
                                 std::vector<DetectedKeypoint>* keypoints);
  int PrepareOrientationInputs(Octave& oct,
                               const std::vector<DetectedKeypoint>& keypoints);
  bool EncodeOrientations(id<MTLCommandBuffer> cb, Octave& oct, int count);
  void ReadOrientations(Octave& oct, int count,
                        const std::vector<DetectedKeypoint>& keypoints,
                        std::vector<std::pair<int, float>>* oriented);
  int PrepareDescriptorInputs(
      Octave& oct, const std::vector<DetectedKeypoint>& keypoints,
      const std::vector<std::pair<int, float>>& oriented);
  bool EncodeDescriptors(id<MTLCommandBuffer> cb, Octave& oct, int count);
  void ReadDescriptors(Octave& oct, int count,
                       const std::vector<DetectedKeypoint>& keypoints,
                       const std::vector<std::pair<int, float>>& oriented,
                       ExtractResult* result);

  // Metal objects
  id<MTLDevice> device_;
  id<MTLCommandQueue> commandQueue_;
  id<MTLLibrary> library_;

  // Compute pipelines
  id<MTLComputePipelineState> bilinearUpScalePipeline_;
  id<MTLComputePipelineState> nearestNeighborDownScalePipeline_;
  id<MTLComputePipelineState> convolutionXPipeline_;
  id<MTLComputePipelineState> convolutionYPipeline_;
  id<MTLComputePipelineState> convolutionSeriesXPipeline_;
  id<MTLComputePipelineState> convolutionSeriesYPipeline_;
  id<MTLComputePipelineState> subtractPipeline_;
  id<MTLComputePipelineState> siftExtremaListPipeline_;
  id<MTLComputePipelineState> siftInterpolatePipeline_;
  id<MTLComputePipelineState> siftOrientationPipeline_;
  id<MTLComputePipelineState> siftDescriptorsPipeline_;

  // Seed textures
  id<MTLTexture> luminosityTexture_;  // R8Unorm, input size
  id<MTLTexture> scaledTexture_;      // R16Float, seed size (2x)
  id<MTLTexture> seedTexture_;        // R16Float, seed size (2x)
  id<MTLTexture> seedConvWorkTexture_; // R16Float, seed size, private

  // Seed Gaussian blur convolution buffers
  id<MTLBuffer> seedConvWeightsBuffer_;
  id<MTLBuffer> seedConvParamsBuffer_;

  // Octaves
  std::vector<Octave> octaves_;

  // Options
  Options options_;
  int extrema_capacity_ = kMinExtremaCapacity;
  int keypoint_capacity_ = kMinKeypointCapacity;
  int descriptor_capacity_ = kMinDescriptorCapacity;
  float sigma_min_ = 0.8f;
  float delta_min_ = 0.5f;
  float sigma_input_ = 0.5f;
  int input_w_ = 0, input_h_ = 0;
  int seed_w_ = 0, seed_h_ = 0;
};

// ---------------------------------------------------------------------------
// Pipeline creation helper
// ---------------------------------------------------------------------------
static id<MTLComputePipelineState> MakePipeline(id<MTLDevice> device,
                                                 id<MTLLibrary> library,
                                                 const char* name) {
  NSString* nsName = [NSString stringWithUTF8String:name];
  id<MTLFunction> func = [library newFunctionWithName:nsName];
  if (!func) {
    NSLog(@"SiftMetal: Failed to find function '%s'", name);
    return nil;
  }
  NSError* error = nil;
  id<MTLComputePipelineState> ps =
      [device newComputePipelineStateWithFunction:func error:&error];
  if (error) {
    NSLog(@"SiftMetal: Pipeline creation error for '%s': %@", name, error);
  }
  return ps;
}

static id<MTLTexture> MakeTexture2D(id<MTLDevice> device, int w, int h,
                                     MTLPixelFormat fmt,
                                     MTLStorageMode storage) {
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                        width:w
                                                       height:h
                                                    mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = storage;
  return [device newTextureWithDescriptor:desc];
}

static id<MTLTexture> MakeTexture2DArray(id<MTLDevice> device, int w, int h,
                                          int arrayLen, MTLPixelFormat fmt,
                                          MTLStorageMode storage) {
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureType2DArray;
  desc.pixelFormat = fmt;
  desc.width = w;
  desc.height = h;
  desc.arrayLength = arrayLen;
  desc.mipmapLevelCount = 1;
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = storage;
  return [device newTextureWithDescriptor:desc];
}

static bool CommitAndWait(id<MTLCommandBuffer> commandBuffer,
                          NSString* label) {
  if (!commandBuffer) {
    NSLog(@"SiftMetal: Failed to create command buffer for %@", label);
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  if (commandBuffer.error) {
    NSLog(@"SiftMetal: %@ command buffer failed: %@",
          label,
          commandBuffer.error);
    return false;
  }
  return true;
}

static uint64_t HashDescriptorBytes(const uint8_t* bytes, size_t size) {
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

static NSArray<NSString*>* MetalLibraryCandidatePaths();

// ---------------------------------------------------------------------------
// SiftMetalMatcherImpl
// ---------------------------------------------------------------------------
class SiftMetalMatcherImpl {
 public:
  bool Init();
  bool Match(const uint8_t* descriptors1, int num_descriptors1,
             const MatchKeypoint* keypoints1, const uint8_t* descriptors2,
             int num_descriptors2, const MatchKeypoint* keypoints2,
             const MatchOptions& options,
             MatchGuidedGeometry guided_geometry,
             const float matrix[9], float max_residual,
             std::vector<MatchResult>* matches);

 private:
  // Encodes one matching direction into the shared command buffer, using
  // per-direction params/results buffers so forward and cross-check reverse
  // passes can run in a single commit.
  bool EncodeOneWay(id<MTLCommandBuffer> commandBuffer,
                    id<MTLBuffer> descriptors1Buffer, int num_descriptors1,
                    id<MTLBuffer> keypoints1Buffer,
                    id<MTLBuffer> descriptors2Buffer, int num_descriptors2,
                    id<MTLBuffer> keypoints2Buffer,
                    const MatchOptions& options,
                    MatchGuidedGeometry guided_geometry,
                    const float matrix[9], float max_residual,
                    bool reverse_guided, int direction);

  id<MTLBuffer> GetDescriptorBuffer(const uint8_t* descriptors,
                                    int num_descriptors);
  id<MTLBuffer> GetKeypointBuffer(bool first_buffer,
                                  const MatchKeypoint* keypoints,
                                  int num_keypoints);
  void EvictDescriptorBuffers();

  struct DescriptorBufferCacheEntry {
    const uint8_t* descriptors = nullptr;
    int num_descriptors = 0;
    size_t byte_size = 0;
    uint64_t content_hash = 0;
    uint64_t last_used = 0;
    id<MTLBuffer> buffer = nil;
  };

  id<MTLDevice> device_;
  id<MTLCommandQueue> commandQueue_;
  id<MTLLibrary> library_;
  id<MTLComputePipelineState> siftMatchBestPipeline_;
  id<MTLComputePipelineState> siftMatchBestDotParallelPipeline_;
  id<MTLBuffer> dummyKeypointBuffer_;
  struct DirectionResources {
    id<MTLBuffer> paramsBuffer = nil;
    id<MTLBuffer> resultsBuffer = nil;
    size_t resultsCapacity = 0;
  };
  DirectionResources directions_[2];
  id<MTLBuffer> keypoints1Buffer_;
  id<MTLBuffer> keypoints2Buffer_;
  size_t keypoints1BufferCapacity_ = 0;
  size_t keypoints2BufferCapacity_ = 0;
  std::vector<SIFTMatcherResult> matches_1to2_;
  std::vector<SIFTMatcherResult> matches_2to1_;
  std::vector<DescriptorBufferCacheEntry> descriptor_buffer_cache_;
  size_t descriptor_buffer_cache_bytes_ = 0;
  uint64_t descriptor_buffer_cache_tick_ = 0;
};

bool SiftMetalMatcherImpl::Init() {
  static_assert(sizeof(MatchKeypoint) == sizeof(SIFTMatcherKeypoint),
                "Match keypoint ABI mismatch");
  static_assert(offsetof(MatchKeypoint, x) == offsetof(SIFTMatcherKeypoint, x),
                "Match keypoint x offset mismatch");
  static_assert(offsetof(MatchKeypoint, y) == offsetof(SIFTMatcherKeypoint, y),
                "Match keypoint y offset mismatch");

  device_ = MTLCreateSystemDefaultDevice();
  if (!device_) return false;

  commandQueue_ = [device_ newCommandQueue];
  if (!commandQueue_) return false;

  NSError* error = nil;
  NSFileManager* fileManager = [NSFileManager defaultManager];
  for (NSString* libPath in MetalLibraryCandidatePaths()) {
    if (![fileManager fileExistsAtPath:libPath]) {
      continue;
    }
    library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:libPath]
                                    error:&error];
    if (library_) {
      break;
    }
  }
  if (!library_) {
    library_ = [device_ newDefaultLibrary];
  }
  if (!library_) {
    NSLog(@"SiftMetal: Failed to load Metal library for matching: %@", error);
    return false;
  }

  siftMatchBestPipeline_ = MakePipeline(device_, library_, "siftMatchBest");
  siftMatchBestDotParallelPipeline_ =
      MakePipeline(device_, library_, "siftMatchBestDotParallel");
  const SIFTMatcherKeypoint dummy_keypoint = {0.0f, 0.0f};
  dummyKeypointBuffer_ =
      [device_ newBufferWithBytes:&dummy_keypoint
                           length:sizeof(dummy_keypoint)
                          options:MTLResourceStorageModeShared];
  return siftMatchBestPipeline_ != nil &&
         siftMatchBestDotParallelPipeline_ != nil &&
         dummyKeypointBuffer_ != nil;
}

id<MTLBuffer> SiftMetalMatcherImpl::GetDescriptorBuffer(
    const uint8_t* descriptors, int num_descriptors) {
  if (!descriptors || num_descriptors <= 0) {
    return nil;
  }

  const size_t descriptor_bytes =
      static_cast<size_t>(num_descriptors) * SIFT_MATCHER_DESCRIPTOR_DIM;
  uint64_t content_hash = 0;
  bool has_content_hash = false;
  const auto get_content_hash = [&]() {
    if (!has_content_hash) {
      content_hash = HashDescriptorBytes(descriptors, descriptor_bytes);
      has_content_hash = true;
    }
    return content_hash;
  };

  ++descriptor_buffer_cache_tick_;
  for (auto it = descriptor_buffer_cache_.begin();
       it != descriptor_buffer_cache_.end();
       ++it) {
    auto& entry = *it;
    if (entry.descriptors == descriptors &&
        entry.num_descriptors == num_descriptors) {
      if (entry.content_hash != get_content_hash()) {
        descriptor_buffer_cache_bytes_ -= entry.byte_size;
        descriptor_buffer_cache_.erase(it);
        break;
      }
      entry.last_used = descriptor_buffer_cache_tick_;
      return entry.buffer;
    }
  }

  id<MTLBuffer> buffer =
      [device_ newBufferWithBytes:descriptors
                           length:descriptor_bytes
                          options:MTLResourceStorageModeShared];
  if (!buffer) {
    return nil;
  }

  static constexpr size_t kMaxCachedDescriptorBytes =
      256ull * 1024ull * 1024ull;
  if (descriptor_bytes > kMaxCachedDescriptorBytes / 2) {
    return buffer;
  }

  descriptor_buffer_cache_.push_back(DescriptorBufferCacheEntry{
      descriptors,
      num_descriptors,
      descriptor_bytes,
      get_content_hash(),
      descriptor_buffer_cache_tick_,
      buffer});
  descriptor_buffer_cache_bytes_ += descriptor_bytes;
  EvictDescriptorBuffers();
  return buffer;
}

void SiftMetalMatcherImpl::EvictDescriptorBuffers() {
  static constexpr size_t kMaxCachedDescriptorBytes =
      256ull * 1024ull * 1024ull;
  while (descriptor_buffer_cache_bytes_ > kMaxCachedDescriptorBytes &&
         !descriptor_buffer_cache_.empty()) {
    auto lru = descriptor_buffer_cache_.begin();
    for (auto it = descriptor_buffer_cache_.begin();
         it != descriptor_buffer_cache_.end();
         ++it) {
      if (it->last_used < lru->last_used) {
        lru = it;
      }
    }
    descriptor_buffer_cache_bytes_ -= lru->byte_size;
    descriptor_buffer_cache_.erase(lru);
  }
}

id<MTLBuffer> SiftMetalMatcherImpl::GetKeypointBuffer(
    bool first_buffer, const MatchKeypoint* keypoints, int num_keypoints) {
  if (!keypoints || num_keypoints <= 0) {
    return nil;
  }

  id<MTLBuffer> buffer =
      first_buffer ? keypoints1Buffer_ : keypoints2Buffer_;
  size_t capacity =
      first_buffer ? keypoints1BufferCapacity_ : keypoints2BufferCapacity_;
  const size_t keypoint_bytes =
      static_cast<size_t>(num_keypoints) * sizeof(SIFTMatcherKeypoint);
  if (!buffer || capacity < keypoint_bytes) {
    buffer = [device_ newBufferWithLength:keypoint_bytes
                                  options:MTLResourceStorageModeShared];
    capacity = buffer ? keypoint_bytes : 0;
    if (first_buffer) {
      keypoints1Buffer_ = buffer;
      keypoints1BufferCapacity_ = capacity;
    } else {
      keypoints2Buffer_ = buffer;
      keypoints2BufferCapacity_ = capacity;
    }
  }
  if (!buffer) {
    return nil;
  }

  std::memcpy(buffer.contents, keypoints, keypoint_bytes);
  return buffer;
}

bool SiftMetalMatcherImpl::EncodeOneWay(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLBuffer> descriptors1Buffer, int num_descriptors1,
    id<MTLBuffer> keypoints1Buffer,
    id<MTLBuffer> descriptors2Buffer, int num_descriptors2,
    id<MTLBuffer> keypoints2Buffer,
    const MatchOptions& options, MatchGuidedGeometry guided_geometry,
    const float matrix[9], float max_residual, bool reverse_guided,
    int direction) {
  const bool use_parallel_dot_pipeline =
      guided_geometry == MatchGuidedGeometry::NONE;
  id<MTLComputePipelineState> pipeline =
      use_parallel_dot_pipeline ? siftMatchBestDotParallelPipeline_
                                : siftMatchBestPipeline_;
  if (!pipeline) {
    return false;
  }

  const bool guided = guided_geometry != MatchGuidedGeometry::NONE;

  SIFTMatcherParameters params = {};
  params.numDescriptors1 = static_cast<uint32_t>(num_descriptors1);
  params.numDescriptors2 = static_cast<uint32_t>(num_descriptors2);
  params.maxRatio = options.max_ratio;
  params.maxDistance = options.max_distance;
  params.maxL2Distance =
      512.0f * 512.0f * options.max_distance * options.max_distance;
  params.maxRatioSquared = options.max_ratio * options.max_ratio;
  params.maxResidual = max_residual;
  params.distanceType = guided ? SIFT_MATCHER_DISTANCE_L2
                               : SIFT_MATCHER_DISTANCE_DOT_PRODUCT;
  params.guidedGeometry = static_cast<int32_t>(guided_geometry);
  params.reverseGuided = reverse_guided ? 1 : 0;
  if (matrix) {
    std::memcpy(params.matrix, matrix, 9 * sizeof(float));
  }

  DirectionResources& dir = directions_[direction];
  if (!dir.paramsBuffer) {
    dir.paramsBuffer =
        [device_ newBufferWithLength:sizeof(params)
                             options:MTLResourceStorageModeShared];
  }
  const size_t result_bytes =
      static_cast<size_t>(num_descriptors1) * sizeof(SIFTMatcherResult);
  if (!dir.resultsBuffer || dir.resultsCapacity < result_bytes) {
    dir.resultsBuffer =
        [device_ newBufferWithLength:result_bytes
                             options:MTLResourceStorageModeShared];
    dir.resultsCapacity = dir.resultsBuffer ? result_bytes : 0;
  }
  if (!dir.paramsBuffer || !dir.resultsBuffer) {
    return false;
  }
  std::memcpy(dir.paramsBuffer.contents, &params, sizeof(params));

  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer computeCommandEncoder];
  if (!encoder) {
    return false;
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:descriptors1Buffer offset:0 atIndex:0];
  [encoder setBuffer:descriptors2Buffer offset:0 atIndex:1];
  [encoder setBuffer:keypoints1Buffer offset:0 atIndex:2];
  [encoder setBuffer:keypoints2Buffer offset:0 atIndex:3];
  [encoder setBuffer:dir.paramsBuffer offset:0 atIndex:4];
  [encoder setBuffer:dir.resultsBuffer offset:0 atIndex:5];
  if (use_parallel_dot_pipeline) {
    const NSUInteger threadsPerThreadgroup = std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup, SIFT_MATCHER_DOT_THREADS);
    [encoder dispatchThreadgroups:MTLSizeMake(num_descriptors1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
  } else {
    const NSUInteger threadsPerThreadgroup =
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:MTLSizeMake(num_descriptors1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
  }
  [encoder endEncoding];
  return true;
}

bool SiftMetalMatcherImpl::Match(
    const uint8_t* descriptors1, int num_descriptors1,
    const MatchKeypoint* keypoints1, const uint8_t* descriptors2,
    int num_descriptors2, const MatchKeypoint* keypoints2,
    const MatchOptions& options, MatchGuidedGeometry guided_geometry,
    const float matrix[9], float max_residual,
    std::vector<MatchResult>* matches) {
  const bool valid_guided_geometry =
      guided_geometry == MatchGuidedGeometry::NONE ||
      guided_geometry == MatchGuidedGeometry::EPIPOLAR ||
      guided_geometry == MatchGuidedGeometry::HOMOGRAPHY;
  if (!matches) {
    return false;
  }
  matches->clear();

  if (num_descriptors1 < 0 || num_descriptors2 < 0 ||
      (num_descriptors1 > 0 && !descriptors1) ||
      (num_descriptors2 > 0 && !descriptors2) ||
      !valid_guided_geometry || !std::isfinite(options.max_ratio) ||
      !std::isfinite(options.max_distance) || options.max_ratio <= 0.0f ||
      options.max_distance <= 0.0f ||
      (guided_geometry != MatchGuidedGeometry::NONE &&
       (!matrix || !std::isfinite(max_residual) || max_residual < 0.0f))) {
    return false;
  }
  if (guided_geometry != MatchGuidedGeometry::NONE) {
    if (num_descriptors1 > 0 && num_descriptors2 > 0 &&
        (!keypoints1 || !keypoints2)) {
      return false;
    }
    for (int i = 0; i < 9; ++i) {
      if (!std::isfinite(matrix[i])) {
        return false;
      }
    }
  }

  if (num_descriptors1 <= 0 || num_descriptors2 <= 0) {
    return true;
  }

  id<MTLBuffer> descriptors1Buffer =
      GetDescriptorBuffer(descriptors1, num_descriptors1);
  id<MTLBuffer> descriptors2Buffer =
      GetDescriptorBuffer(descriptors2, num_descriptors2);
  if (!descriptors1Buffer || !descriptors2Buffer) {
    return false;
  }

  // Upload each keypoint set once; the reverse pass binds the same buffers
  // swapped instead of re-uploading into shared storage while the forward
  // dispatch may still consume it.
  id<MTLBuffer> keypoints1Buffer = dummyKeypointBuffer_;
  id<MTLBuffer> keypoints2Buffer = dummyKeypointBuffer_;
  if (guided_geometry != MatchGuidedGeometry::NONE) {
    keypoints1Buffer =
        GetKeypointBuffer(/*first_buffer=*/true, keypoints1, num_descriptors1);
    keypoints2Buffer =
        GetKeypointBuffer(/*first_buffer=*/false, keypoints2, num_descriptors2);
  }
  if (!keypoints1Buffer || !keypoints2Buffer) {
    return false;
  }

  // Both directions are encoded into one command buffer with per-direction
  // params/results buffers, so cross-checked matching costs a single
  // synchronization instead of two.
  id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
  if (!commandBuffer) {
    return false;
  }
  if (!EncodeOneWay(commandBuffer,
                    descriptors1Buffer, num_descriptors1, keypoints1Buffer,
                    descriptors2Buffer, num_descriptors2, keypoints2Buffer,
                    options, guided_geometry, matrix, max_residual,
                    /*reverse_guided=*/false, /*direction=*/0)) {
    return false;
  }
  if (options.cross_check) {
    if (!EncodeOneWay(commandBuffer,
                      descriptors2Buffer, num_descriptors2, keypoints2Buffer,
                      descriptors1Buffer, num_descriptors1, keypoints1Buffer,
                      options, guided_geometry, matrix, max_residual,
                      /*reverse_guided=*/true, /*direction=*/1)) {
      return false;
    }
  }
  if (!CommitAndWait(commandBuffer, @"matching")) {
    return false;
  }

  matches_1to2_.resize(num_descriptors1);
  std::memcpy(matches_1to2_.data(), directions_[0].resultsBuffer.contents,
              static_cast<size_t>(num_descriptors1) * sizeof(SIFTMatcherResult));
  if (options.cross_check) {
    matches_2to1_.resize(num_descriptors2);
    std::memcpy(matches_2to1_.data(), directions_[1].resultsBuffer.contents,
                static_cast<size_t>(num_descriptors2) *
                    sizeof(SIFTMatcherResult));
  }

  matches->reserve(num_descriptors1);
  for (int i1 = 0; i1 < num_descriptors1; ++i1) {
    const int i2 = matches_1to2_[i1].index;
    if (i2 < 0 || i2 >= num_descriptors2) {
      continue;
    }
    if (options.cross_check &&
        (i2 >= static_cast<int>(matches_2to1_.size()) ||
         matches_2to1_[i2].index != i1)) {
      continue;
    }
    matches->push_back(
        MatchResult{static_cast<uint32_t>(i1), static_cast<uint32_t>(i2)});
  }
  return true;
}

static void AddMetalLibraryCandidate(NSMutableArray<NSString*>* paths,
                                     NSString* path) {
  if (path.length == 0) return;
  if (![paths containsObject:path]) {
    [paths addObject:path];
  }
}

static NSArray<NSString*>* MetalLibraryCandidatePaths() {
  NSMutableArray<NSString*>* paths = [NSMutableArray array];

  NSString* mainBundlePath =
      [[NSBundle mainBundle] pathForResource:@"sift" ofType:@"metallib"];
  AddMetalLibraryCandidate(paths, mainBundlePath);

  Dl_info imageInfo;
  if (dladdr(reinterpret_cast<const void*>(&MetalLibraryCandidatePaths),
             &imageInfo) != 0 &&
      imageInfo.dli_fname != nullptr) {
    NSString* imagePath = [NSString stringWithUTF8String:imageInfo.dli_fname];
    NSString* imageDir = [imagePath stringByDeletingLastPathComponent];
    AddMetalLibraryCandidate(
        paths, [imageDir stringByAppendingPathComponent:@"sift.metallib"]);
    AddMetalLibraryCandidate(
        paths, [imageDir stringByAppendingPathComponent:@"Resources/sift.metallib"]);
    AddMetalLibraryCandidate(
        paths,
        [imageDir stringByAppendingPathComponent:@"../Resources/sift.metallib"]);
    AddMetalLibraryCandidate(
        paths, [imageDir stringByAppendingPathComponent:@"../lib/sift.metallib"]);
  }

  NSString* configuredPath =
      [NSString stringWithUTF8String:SIFT_METAL_METALLIB_PATH];
  AddMetalLibraryCandidate(paths, configuredPath);

  return paths;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
bool SiftMetalExtractorImpl::Init(const Options& opts, int max_w, int max_h) {
  if (max_w <= 0 || max_h <= 0 ||
      (opts.first_octave != -1 && opts.first_octave != 0) ||
      opts.scales_per_octave <= 0 || opts.max_num_features <= 0 ||
      !std::isfinite(opts.peak_threshold) || opts.peak_threshold <= 0.0f ||
      !std::isfinite(opts.edge_threshold) || opts.edge_threshold <= 0.0f ||
      (!opts.upright && opts.max_num_orientations <= 0)) {
    return false;
  }

  options_ = opts;
  const int max_num_features = options_.max_num_features;
  const int max_num_orientations =
      options_.upright ? 1 : std::max(1, options_.max_num_orientations);
  const int64_t requested_descriptors =
      static_cast<int64_t>(max_num_features) * max_num_orientations;
  if (requested_descriptors > std::numeric_limits<int>::max()) {
    return false;
  }

  const int requested_descriptor_capacity =
      static_cast<int>(requested_descriptors);
  extrema_capacity_ =
      std::max(kMinExtremaCapacity, requested_descriptor_capacity);
  keypoint_capacity_ =
      std::max(kMinKeypointCapacity, max_num_features);
  descriptor_capacity_ =
      std::max(kMinDescriptorCapacity, requested_descriptor_capacity);

  // Get the default Metal device.
  device_ = MTLCreateSystemDefaultDevice();
  if (!device_) return false;

  commandQueue_ = [device_ newCommandQueue];
  if (!commandQueue_) return false;

  // Load the pre-compiled Metal library from the build tree or bundle.
  NSError* error = nil;
  NSFileManager* fileManager = [NSFileManager defaultManager];
  for (NSString* libPath in MetalLibraryCandidatePaths()) {
    if (![fileManager fileExistsAtPath:libPath]) {
      continue;
    }
    library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:libPath]
                                    error:&error];
    if (library_) {
      break;
    }
  }
  if (!library_) {
    // Fallback: try default library.
    library_ = [device_ newDefaultLibrary];
  }
  if (!library_) {
    NSLog(@"SiftMetal: Failed to load Metal library: %@", error);
    return false;
  }

  // Create all compute pipelines.
  bilinearUpScalePipeline_ = MakePipeline(device_, library_, "bilinearUpScale");
  nearestNeighborDownScalePipeline_ =
      MakePipeline(device_, library_, "nearestNeighborDownScale");
  convolutionXPipeline_ = MakePipeline(device_, library_, "convolutionX");
  convolutionYPipeline_ = MakePipeline(device_, library_, "convolutionY");
  convolutionSeriesXPipeline_ =
      MakePipeline(device_, library_, "convolutionSeriesX");
  convolutionSeriesYPipeline_ =
      MakePipeline(device_, library_, "convolutionSeriesY");
  subtractPipeline_ = MakePipeline(device_, library_, "subtract");
  siftExtremaListPipeline_ =
      MakePipeline(device_, library_, "siftExtremaList");
  siftInterpolatePipeline_ =
      MakePipeline(device_, library_, "siftInterpolate");
  siftOrientationPipeline_ =
      MakePipeline(device_, library_, "siftOrientation");
  siftDescriptorsPipeline_ =
      MakePipeline(device_, library_, "siftDescriptors");

  if (!bilinearUpScalePipeline_ || !nearestNeighborDownScalePipeline_ ||
      !convolutionXPipeline_ || !convolutionYPipeline_ ||
      !convolutionSeriesXPipeline_ || !convolutionSeriesYPipeline_ ||
      !subtractPipeline_ ||
      !siftExtremaListPipeline_ || !siftInterpolatePipeline_ ||
      !siftOrientationPipeline_ || !siftDescriptorsPipeline_) {
    return false;
  }

  // Determine seed size based on first_octave.
  if (opts.first_octave == -1) {
    delta_min_ = 0.5f;
  } else {
    delta_min_ = 1.0f;
  }

  input_w_ = max_w;
  input_h_ = max_h;
  seed_w_ = static_cast<int>(float(max_w) / delta_min_);
  seed_h_ = static_cast<int>(float(max_h) / delta_min_);

  // Create textures for the seed stage. The grayscale input is uploaded
  // directly as R8Unorm; texture reads return the pixel value / 255.
  luminosityTexture_ = MakeTexture2D(device_, max_w, max_h,
                                      MTLPixelFormatR8Unorm,
                                      MTLStorageModeShared);
  scaledTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                  MTLPixelFormatR16Float,
                                  MTLStorageModePrivate);
  seedTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                MTLPixelFormatR16Float,
                                MTLStorageModePrivate);
  seedConvWorkTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                        MTLPixelFormatR16Float,
                                        MTLStorageModePrivate);

  // Compute seed Gaussian blur kernel.
  float sigma_seed = std::sqrt(sigma_min_ * sigma_min_ -
                                sigma_input_ * sigma_input_) / delta_min_;
  auto seedWeights = GaussianWeights(sigma_seed);
  seedConvWeightsBuffer_ =
      [device_ newBufferWithBytes:seedWeights.data()
                           length:seedWeights.size() * sizeof(float)
                          options:MTLResourceStorageModeShared];
  uint32_t seedWeightCount = static_cast<uint32_t>(seedWeights.size());
  seedConvParamsBuffer_ =
      [device_ newBufferWithBytes:&seedWeightCount
                           length:sizeof(uint32_t)
                          options:MTLResourceStorageModeShared];

  if (!luminosityTexture_ || !scaledTexture_ || !seedTexture_ ||
      !seedConvWorkTexture_ || !seedConvWeightsBuffer_ ||
      !seedConvParamsBuffer_) {
    return false;
  }

  // Setup octaves.
  SetupOctaves(max_w, max_h);
  if (!OctavesResourcesReady(octaves_)) {
    return false;
  }

  return true;
}

// ---------------------------------------------------------------------------
// SetupOctaves
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::SetupOctaves(int w, int h) {
  int num_octaves = options_.num_octaves;
  const int seed_w = static_cast<int>(float(w) / delta_min_);
  const int seed_h = static_cast<int>(float(h) / delta_min_);
  int max_usable_octaves = 0;
  for (int ow = seed_w, oh = seed_h; ow >= 8 && oh >= 8; ow /= 2, oh /= 2) {
    ++max_usable_octaves;
  }

  if (num_octaves <= 0) {
    // Match SiftGPU's octave count: floor(log2(min(w,h))) - 3
    // But applied to the seed image dimensions (after upscaling).
    int seed_min = std::min(seed_w, seed_h);
    num_octaves = static_cast<int>(
        std::floor(std::log2(float(seed_min)))) - 3;
    num_octaves = std::max(1, num_octaves);
  } else {
    num_octaves = std::min(num_octaves, max_usable_octaves);
  }

  octaves_.clear();
  octaves_.reserve(num_octaves);
  for (int o = 0; o < num_octaves; ++o) {
    float delta = delta_min_ * std::pow(2.0f, float(o));
    int ow = static_cast<int>(float(w) / delta);
    int oh = static_cast<int>(float(h) / delta);
    if (ow < 8 || oh < 8) {
      break;
    }

    int ns = options_.scales_per_octave;
    std::vector<float> sigmas;
    for (int s = 0; s < ns + 3; ++s) {
      float ratio = delta / delta_min_;
      float scale = std::pow(2.0f, float(s) / float(ns));
      sigmas.push_back(ratio * sigma_min_ * scale);
    }

    octaves_.emplace_back();
    SetupOctave(octaves_.back(), o, delta, ow, oh, ns, sigmas);
  }
}

void SiftMetalExtractorImpl::SetupOctave(Octave& oct, int o, float delta,
                                          int w, int h, int num_scales,
                                          const std::vector<float>& sigmas) {
  oct.o = o;
  oct.delta = delta;
  oct.width = w;
  oct.height = h;
  oct.num_scales = num_scales;
  oct.sigmas = sigmas;

  int numGaussians = num_scales + 3;
  int numDifferences = num_scales + 2;

  // Gaussian images are normalized [0, 1] values; half precision halves
  // bandwidth and memory on the most heavily trafficked textures. The DoG
  // textures stay fp32: extrema detection and keypoint interpolation consume
  // small differences and second derivatives of these values.
  oct.gaussianTextures = MakeTexture2DArray(
      device_, w, h, numGaussians, MTLPixelFormatR16Float,
      MTLStorageModePrivate);
  oct.differenceTextures = MakeTexture2DArray(
      device_, w, h, numDifferences, MTLPixelFormatR32Float,
      MTLStorageModePrivate);
  NearestNeighborScaleParameters downscale_params = {};
  downscale_params.inputSlice = static_cast<int32_t>(num_scales);
  downscale_params.outputSlice = 0;
  oct.downscaleParamsBuffer =
      [device_ newBufferWithBytes:&downscale_params
                           length:sizeof(NearestNeighborScaleParameters)
                          options:MTLResourceStorageModeShared];

  // Convolution work texture (single-slice private).
  oct.convWorkTexture = MakeTexture2DArray(
      device_, w, h, 1, MTLPixelFormatR16Float, MTLStorageModePrivate);

  // Build convolution parameter buffers for Gaussian series.
  oct.convPairs.resize(numGaussians - 1);
  for (int s = 1; s < numGaussians; ++s) {
    float sa = sigmas[s - 1];
    float sb = sigmas[s];
    float rho = std::sqrt(sb * sb - sa * sa) / delta;
    auto weights = GaussianWeights(rho);
    const size_t weight_count =
        std::min(weights.size(), (size_t)CONVOLUTION_WEIGHTS_LENGTH);

    // X pass: read from slice [s-1], write to work slice [0]
    ConvolutionParameters paramsX = {};
    paramsX.inputDepth = static_cast<int32_t>(s - 1);
    paramsX.outputDepth = 0;
    paramsX.count = static_cast<int32_t>(weight_count);
    std::memcpy(paramsX.weights, weights.data(), weight_count * sizeof(float));
    oct.convPairs[s - 1].paramsX =
        [device_ newBufferWithBytes:&paramsX
                             length:sizeof(ConvolutionParameters)
                            options:MTLResourceStorageModeShared];

    // Y pass: read from work slice [0], write to slice [s]
    ConvolutionParameters paramsY = {};
    paramsY.inputDepth = 0;
    paramsY.outputDepth = static_cast<int32_t>(s);
    paramsY.count = static_cast<int32_t>(weight_count);
    std::memcpy(paramsY.weights, weights.data(), weight_count * sizeof(float));
    oct.convPairs[s - 1].paramsY =
        [device_ newBufferWithBytes:&paramsY
                             length:sizeof(ConvolutionParameters)
                            options:MTLResourceStorageModeShared];
  }

  // Extrema buffers
  oct.extremaOutputBuffer =
      [device_ newBufferWithLength:extrema_capacity_ *
                                    sizeof(SIFTExtremaResult)
                           options:MTLResourceStorageModeShared];
  oct.extremaIndexBuffer =
      [device_ newBufferWithLength:sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];
  SIFTExtremaParameters extrema_params = {};
  extrema_params.capacity = static_cast<uint32_t>(extrema_capacity_);
  extrema_params.peakThreshold = options_.peak_threshold;
  oct.extremaParamsBuffer =
      [device_ newBufferWithBytes:&extrema_params
                           length:sizeof(SIFTExtremaParameters)
                          options:MTLResourceStorageModeShared];

  // Interpolation buffers
  oct.interpolateOutputBuffer =
      [device_ newBufferWithLength:keypoint_capacity_ *
                                       sizeof(SIFTInterpolateOutputKeypoint)
                           options:MTLResourceStorageModeShared];
  oct.interpolateParamsBuffer =
      [device_ newBufferWithLength:sizeof(SIFTInterpolateParameters)
                           options:MTLResourceStorageModeShared];

  // Orientation buffers
  oct.orientationInputBuffer =
      [device_ newBufferWithLength:keypoint_capacity_ *
                                       sizeof(SIFTOrientationKeypoint)
                           options:MTLResourceStorageModeShared];
  oct.orientationOutputBuffer =
      [device_ newBufferWithLength:keypoint_capacity_ *
                                       sizeof(SIFTOrientationResult)
                           options:MTLResourceStorageModeShared];
  oct.orientationParamsBuffer =
      [device_ newBufferWithLength:sizeof(SIFTOrientationParameters)
                           options:MTLResourceStorageModeShared];

  // Descriptor buffers
  oct.descriptorInputBuffer =
      [device_ newBufferWithLength:descriptor_capacity_ *
                                    sizeof(SIFTDescriptorInput)
                           options:MTLResourceStorageModeShared];
  oct.descriptorOutputBuffer =
      [device_ newBufferWithLength:descriptor_capacity_ *
                                       sizeof(SIFTDescriptorResult)
                           options:MTLResourceStorageModeShared];
  oct.descriptorParamsBuffer =
      [device_ newBufferWithLength:sizeof(SIFTDescriptorParameters)
                           options:MTLResourceStorageModeShared];
}

// ---------------------------------------------------------------------------
// Extract
// ---------------------------------------------------------------------------
bool SiftMetalExtractorImpl::Extract(const uint8_t* data, int w, int h,
                                      ExtractResult* result) {
  if (!result) {
    return false;
  }

  result->keypoints.clear();
  result->descriptors.clear();

  if (!data || w <= 0 || h <= 0) {
    return false;
  }

  // Recreate textures if image size changed.
  if (w != input_w_ || h != input_h_) {
    input_w_ = w;
    input_h_ = h;
    seed_w_ = static_cast<int>(float(w) / delta_min_);
    seed_h_ = static_cast<int>(float(h) / delta_min_);

    luminosityTexture_ = MakeTexture2D(device_, w, h,
                                        MTLPixelFormatR8Unorm,
                                        MTLStorageModeShared);
    scaledTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                    MTLPixelFormatR16Float,
                                    MTLStorageModePrivate);
    seedTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                  MTLPixelFormatR16Float,
                                  MTLStorageModePrivate);
    seedConvWorkTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                          MTLPixelFormatR16Float,
                                          MTLStorageModePrivate);
    if (!luminosityTexture_ || !scaledTexture_ || !seedTexture_ ||
        !seedConvWorkTexture_) {
      return false;
    }

    SetupOctaves(w, h);
    if (!OctavesResourcesReady(octaves_)) {
      return false;
    }
  }

  if (octaves_.empty()) {
    return true;
  }

  // Upload uint8 grayscale directly; R8Unorm reads yield value / 255.
  MTLRegion region = MTLRegionMake2D(0, 0, w, h);
  [luminosityTexture_ replaceRegion:region
                        mipmapLevel:0
                          withBytes:data
                        bytesPerRow:w];

  // Phase 1: Build scale-space pyramid (DoG + extrema).
  {
    id<MTLCommandBuffer> cb = [commandQueue_ commandBuffer];
    if (!cb) {
      return false;
    }
    if (!EncodeSeedTexture(cb)) {
      return false;
    }

    // First octave reads from seed texture (2D).
    if (!EncodeOctave(cb, octaves_[0], seedTexture_, true)) {
      return false;
    }

    // Subsequent octaves read from previous octave's Gaussian textures.
    for (size_t i = 1; i < octaves_.size(); ++i) {
      if (!EncodeOctave(cb, octaves_[i],
                        octaves_[i - 1].gaussianTextures, false)) {
        return false;
      }
    }

    if (!CommitAndWait(cb, @"scale-space")) {
      return false;
    }
  }

  // Phase 2: interpolate, orientate, and describe keypoints. Every stage
  // encodes the dispatches for all octaves into one command buffer and waits
  // once, instead of synchronizing per octave and stage.
  const size_t num_octaves = octaves_.size();
  std::vector<int> extrema_counts(num_octaves, 0);
  std::vector<int> orientation_counts(num_octaves, 0);
  std::vector<int> descriptor_counts(num_octaves, 0);
  std::vector<std::vector<DetectedKeypoint>> oct_keypoints(num_octaves);
  // Per octave: (keypoint_index, theta) pairs.
  std::vector<std::vector<std::pair<int, float>>> oct_oriented(num_octaves);

  // Stage 1: interpolate extrema.
  {
    id<MTLCommandBuffer> cb = [commandQueue_ commandBuffer];
    if (!cb) {
      return false;
    }
    bool encoded = false;
    for (size_t i = 0; i < num_octaves; ++i) {
      const int extremaCount =
          std::min(ReadExtremaCount(octaves_[i]), keypoint_capacity_);
      extrema_counts[i] = extremaCount;
      if (extremaCount <= 0) continue;
      if (!EncodeInterpolateKeypoints(cb, octaves_[i], extremaCount)) {
        return false;
      }
      encoded = true;
    }
    if (encoded && !CommitAndWait(cb, @"keypoint interpolation")) {
      return false;
    }
  }

  // Stage 2: filter interpolated keypoints and compute orientations.
  {
    id<MTLCommandBuffer> cb = [commandQueue_ commandBuffer];
    if (!cb) {
      return false;
    }
    bool encoded = false;
    for (size_t i = 0; i < num_octaves; ++i) {
      if (extrema_counts[i] <= 0) continue;
      ReadInterpolatedKeypoints(octaves_[i], extrema_counts[i],
                                &oct_keypoints[i]);
      if (oct_keypoints[i].empty()) continue;
      const int count = PrepareOrientationInputs(octaves_[i], oct_keypoints[i]);
      orientation_counts[i] = count;
      if (count <= 0) continue;
      if (!EncodeOrientations(cb, octaves_[i], count)) {
        return false;
      }
      encoded = true;
    }
    if (encoded && !CommitAndWait(cb, @"orientation")) {
      return false;
    }
  }

  // Stage 3: expand orientations and compute descriptors.
  {
    id<MTLCommandBuffer> cb = [commandQueue_ commandBuffer];
    if (!cb) {
      return false;
    }
    bool encoded = false;
    for (size_t i = 0; i < num_octaves; ++i) {
      if (orientation_counts[i] <= 0) continue;
      ReadOrientations(octaves_[i], orientation_counts[i], oct_keypoints[i],
                       &oct_oriented[i]);
      if (oct_oriented[i].empty()) continue;
      const int count = PrepareDescriptorInputs(octaves_[i], oct_keypoints[i],
                                                oct_oriented[i]);
      descriptor_counts[i] = count;
      if (count <= 0) continue;
      if (!EncodeDescriptors(cb, octaves_[i], count)) {
        return false;
      }
      encoded = true;
    }
    if (encoded && !CommitAndWait(cb, @"descriptor")) {
      return false;
    }
  }

  // Stage 4: read back descriptors.
  for (size_t i = 0; i < num_octaves; ++i) {
    if (descriptor_counts[i] <= 0) continue;
    ReadDescriptors(octaves_[i], descriptor_counts[i], oct_keypoints[i],
                    oct_oriented[i], result);
  }

  const int max_num_orientations =
      options_.upright ? 1 : std::max(1, options_.max_num_orientations);
  const int descriptor_limit = options_.max_num_features * max_num_orientations;

  // Sort by scale (descending) and truncate after orientation expansion. This
  // mirrors COLMAP's CPU SIFT semantics, where max_num_features limits detected
  // keypoints and each keypoint may contribute multiple oriented descriptors.
  if (descriptor_limit > 0 &&
      static_cast<int>(result->keypoints.size()) > descriptor_limit) {
    // Create index array, sort by sigma descending.
    std::vector<int> indices(result->keypoints.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) {
                return result->keypoints[a].sigma >
                       result->keypoints[b].sigma;
              });
    indices.resize(descriptor_limit);
    std::sort(indices.begin(), indices.end()); // Restore order.

    std::vector<Keypoint> newKp;
    std::vector<float> newDesc;
    newKp.reserve(descriptor_limit);
    newDesc.reserve(descriptor_limit * 128);
    for (int idx : indices) {
      newKp.push_back(result->keypoints[idx]);
      newDesc.insert(newDesc.end(),
                     result->descriptors.begin() + idx * 128,
                     result->descriptors.begin() + (idx + 1) * 128);
    }
    result->keypoints = std::move(newKp);
    result->descriptors = std::move(newDesc);
  }

  return true;
}

// ---------------------------------------------------------------------------
// EncodeSeedTexture: upscale grayscale input + Gaussian blur.
// ---------------------------------------------------------------------------
bool SiftMetalExtractorImpl::EncodeSeedTexture(id<MTLCommandBuffer> cb) {
  // Bilinear upscale luminosity → scaled. Also converts R8Unorm → R16Float;
  // at equal sizes the kernel samples exact texel centers, i.e. a plain copy.
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) {
      return false;
    }
    [enc setComputePipelineState:bilinearUpScalePipeline_];
    [enc setTexture:scaledTexture_ atIndex:0];
    [enc setTexture:luminosityTexture_ atIndex:1];
    MTLSize tg = {16, 16, 1};
    MTLSize grid = {
        (NSUInteger)(seed_w_ + 15) / 16,
        (NSUInteger)(seed_h_ + 15) / 16, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
  }

  // Gaussian blur scaled → seed (separable 1D convolution).
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) {
      return false;
    }
    [enc setComputePipelineState:convolutionXPipeline_];
    [enc setTexture:seedConvWorkTexture_ atIndex:0];
    [enc setTexture:scaledTexture_ atIndex:1];
    [enc setBuffer:seedConvWeightsBuffer_ offset:0 atIndex:0];
    [enc setBuffer:seedConvParamsBuffer_ offset:0 atIndex:1];
    MTLSize tg = {16, 16, 1};
    MTLSize grid = {(NSUInteger)(seed_w_ + 15) / 16,
                    (NSUInteger)(seed_h_ + 15) / 16, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
  }
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) {
      return false;
    }
    [enc setComputePipelineState:convolutionYPipeline_];
    [enc setTexture:seedTexture_ atIndex:0];
    [enc setTexture:seedConvWorkTexture_ atIndex:1];
    [enc setBuffer:seedConvWeightsBuffer_ offset:0 atIndex:0];
    [enc setBuffer:seedConvParamsBuffer_ offset:0 atIndex:1];
    MTLSize tg = {16, 16, 1};
    MTLSize grid = {(NSUInteger)(seed_w_ + 15) / 16,
                    (NSUInteger)(seed_h_ + 15) / 16, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
  }
  return true;
}

// ---------------------------------------------------------------------------
// EncodeOctave
// ---------------------------------------------------------------------------
bool SiftMetalExtractorImpl::EncodeOctave(id<MTLCommandBuffer> cb,
                                           Octave& oct,
                                           id<MTLTexture> inputTexture,
                                           bool inputIs2D) {
  if (!inputTexture) {
    return false;
  }

  int w = oct.width;
  int h = oct.height;

  // Copy/scale input into gaussian slice 0.
  if (inputIs2D) {
    // 2D texture → first slice of 2DArray.
    if ((int)inputTexture.width != w || (int)inputTexture.height != h) {
      return false;
    }
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) {
      return false;
    }
    [blit copyFromTexture:inputTexture
              sourceSlice:0
              sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(w, h, 1)
               toTexture:oct.gaussianTextures
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
  } else {
    // Nearest-neighbor downscale from previous octave's gaussian[num_scales].
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) {
      return false;
    }
    [enc setComputePipelineState:nearestNeighborDownScalePipeline_];
    [enc setTexture:oct.gaussianTextures atIndex:0];
    [enc setTexture:inputTexture atIndex:1];
    [enc setBuffer:oct.downscaleParamsBuffer offset:0 atIndex:0];
    MTLSize tg = {16, 16, 1};
    MTLSize grid = {(NSUInteger)(w + 15) / 16,
                    (NSUInteger)(h + 15) / 16, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
  }

  // Gaussian series blur.
  if (!EncodeGaussianSeries(cb, oct)) {
    return false;
  }
  // Differences.
  if (!EncodeDifferences(cb, oct)) {
    return false;
  }
  // Extrema detection. Gradients are computed on the fly by the orientation
  // and descriptor kernels from the Gaussian textures.
  return EncodeExtrema(cb, oct);
}

bool SiftMetalExtractorImpl::EncodeGaussianSeries(id<MTLCommandBuffer> cb,
                                                    Octave& oct) {
  int w = oct.width;
  int h = oct.height;
  MTLSize tg = {16, 16, 1};
  MTLSize grid = {(NSUInteger)(w + 15) / 16,
                  (NSUInteger)(h + 15) / 16, 1};

  for (auto& pair : oct.convPairs) {
    // X pass: gaussian → work
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      if (!enc) {
        return false;
      }
      [enc setComputePipelineState:convolutionSeriesXPipeline_];
      [enc setTexture:oct.convWorkTexture atIndex:0];
      [enc setTexture:oct.gaussianTextures atIndex:1];
      [enc setBuffer:pair.paramsX offset:0 atIndex:0];
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
    }
    // Y pass: work → gaussian
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      if (!enc) {
        return false;
      }
      [enc setComputePipelineState:convolutionSeriesYPipeline_];
      [enc setTexture:oct.gaussianTextures atIndex:0];
      [enc setTexture:oct.convWorkTexture atIndex:1];
      [enc setBuffer:pair.paramsY offset:0 atIndex:0];
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
    }
  }
  return true;
}

bool SiftMetalExtractorImpl::EncodeDifferences(id<MTLCommandBuffer> cb,
                                                 Octave& oct) {
  int w = oct.width;
  int h = oct.height;
  int numDiff = oct.num_scales + 2;

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc) {
    return false;
  }
  [enc setComputePipelineState:subtractPipeline_];
  [enc setTexture:oct.differenceTextures atIndex:0];
  [enc setTexture:oct.gaussianTextures atIndex:1];
  MTLSize tg = {8, 8, 8};
  MTLSize grid = {(NSUInteger)(w + 7) / 8,
                  (NSUInteger)(h + 7) / 8,
                  (NSUInteger)(numDiff + 7) / 8};
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
  return true;
}

bool SiftMetalExtractorImpl::EncodeExtrema(id<MTLCommandBuffer> cb,
                                             Octave& oct) {
  int w = oct.width;
  int h = oct.height;
  int numDiff = oct.num_scales + 2;

  // Reset index counter.
  auto* idx = static_cast<uint32_t*>(oct.extremaIndexBuffer.contents);
  *idx = 0;

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc) {
    return false;
  }
  [enc setComputePipelineState:siftExtremaListPipeline_];
  [enc setBuffer:oct.extremaOutputBuffer offset:0 atIndex:0];
  [enc setBuffer:oct.extremaIndexBuffer offset:0 atIndex:1];
  [enc setBuffer:oct.extremaParamsBuffer offset:0 atIndex:2];
  [enc setTexture:oct.differenceTextures atIndex:0];

  // SIMD-width-aligned 2D threadgroups; the z extent is only a few scales.
  const NSUInteger maxThreads =
      siftExtremaListPipeline_.maxTotalThreadsPerThreadgroup;
  const NSUInteger tgWidth = std::min<NSUInteger>(
      siftExtremaListPipeline_.threadExecutionWidth, maxThreads);
  const NSUInteger tgHeight =
      std::max<NSUInteger>(std::min<NSUInteger>(8, maxThreads / tgWidth), 1);
  MTLSize tg = {tgWidth, tgHeight, 1};
  MTLSize gridSize = {(NSUInteger)(w - 2),
                      (NSUInteger)(h - 2),
                      (NSUInteger)(numDiff - 2)};
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tg];
  [enc endEncoding];
  return true;
}

// ---------------------------------------------------------------------------
// ReadExtremaCount
// ---------------------------------------------------------------------------
int SiftMetalExtractorImpl::ReadExtremaCount(Octave& oct) {
  auto* idx = static_cast<uint32_t*>(oct.extremaIndexBuffer.contents);
  const uint32_t count = *idx;
  *idx = 0;
  return static_cast<int>(
      std::min<uint32_t>(count, static_cast<uint32_t>(extrema_capacity_)));
}

// ---------------------------------------------------------------------------
// EncodeInterpolateKeypoints
// ---------------------------------------------------------------------------
bool SiftMetalExtractorImpl::EncodeInterpolateKeypoints(
    id<MTLCommandBuffer> cb, Octave& oct, int extremaCount) {
  // The extrema output buffer doubles as the interpolation input buffer.
  static_assert(sizeof(SIFTExtremaResult) ==
                    sizeof(SIFTInterpolateInputKeypoint),
                "Extrema/interpolate keypoint ABI mismatch");
  static_assert(offsetof(SIFTExtremaResult, x) ==
                        offsetof(SIFTInterpolateInputKeypoint, x) &&
                    offsetof(SIFTExtremaResult, y) ==
                        offsetof(SIFTInterpolateInputKeypoint, y) &&
                    offsetof(SIFTExtremaResult, scale) ==
                        offsetof(SIFTInterpolateInputKeypoint, scale),
                "Extrema/interpolate keypoint field offset mismatch");

  // Set interpolation parameters.
  auto* params = static_cast<SIFTInterpolateParameters*>(
      oct.interpolateParamsBuffer.contents);
  params->dogThreshold = options_.peak_threshold;
  params->maxIterations = 5;
  params->maxOffset = 0.6f;
  params->width = static_cast<int32_t>(oct.width);
  params->height = static_cast<int32_t>(oct.height);
  params->octaveDelta = oct.delta;
  params->edgeThreshold = options_.edge_threshold;
  params->numberOfScales = static_cast<int32_t>(oct.num_scales);

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc) {
    return false;
  }
  [enc setComputePipelineState:siftInterpolatePipeline_];
  [enc setBuffer:oct.interpolateOutputBuffer offset:0 atIndex:0];
  [enc setBuffer:oct.extremaOutputBuffer offset:0 atIndex:1];
  [enc setBuffer:oct.interpolateParamsBuffer offset:0 atIndex:2];
  [enc setTexture:oct.differenceTextures atIndex:0];

  NSUInteger maxThreads =
      siftInterpolatePipeline_.maxTotalThreadsPerThreadgroup;
  MTLSize tg = {maxThreads, 1, 1};
  MTLSize gridSize = {(NSUInteger)extremaCount, 1, 1};
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tg];
  [enc endEncoding];
  return true;
}

// ---------------------------------------------------------------------------
// ReadInterpolatedKeypoints
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::ReadInterpolatedKeypoints(
    Octave& oct, int extremaCount, std::vector<DetectedKeypoint>* keypoints) {
  keypoints->clear();
  auto* interpOut = static_cast<SIFTInterpolateOutputKeypoint*>(
      oct.interpolateOutputBuffer.contents);
  const float sigmaRatio = oct.sigmas[1] / oct.sigmas[0];

  for (int k = 0; k < extremaCount; ++k) {
    auto& p = interpOut[k];
    if (!p.converged) continue;
    if (p.scale < 0 || p.scale >= static_cast<int32_t>(oct.sigmas.size()) ||
        !std::isfinite(p.absoluteX) || !std::isfinite(p.absoluteY) ||
        !std::isfinite(p.subScale)) {
      continue;
    }

    DetectedKeypoint detected;
    detected.keypoint.x = p.absoluteX;
    detected.keypoint.y = p.absoluteY;
    detected.keypoint.sigma =
        oct.sigmas[p.scale] * std::pow(sigmaRatio, p.subScale);
    if (!std::isfinite(detected.keypoint.sigma) ||
        detected.keypoint.sigma <= 0.0f) {
      continue;
    }
    detected.keypoint.orientation = 0; // Set during orientation pass.
    detected.scale = p.scale;
    detected.sub_scale = p.subScale;
    keypoints->push_back(detected);
  }
}

// ---------------------------------------------------------------------------
// PrepareOrientationInputs
// ---------------------------------------------------------------------------
int SiftMetalExtractorImpl::PrepareOrientationInputs(
    Octave& oct, const std::vector<DetectedKeypoint>& keypoints) {
  float delta = oct.delta;
  float lambda = 1.5f;
  float orientThreshold = 0.8f;
  float minX = 1.0f, minY = 1.0f;
  float maxX = float(oct.width - 2);
  float maxY = float(oct.height - 2);

  auto* params =
      static_cast<SIFTOrientationParameters*>(oct.orientationParamsBuffer.contents);
  params->delta = delta;
  params->lambda = lambda;
  params->orientationThreshold = orientThreshold;

  auto* orientIn = static_cast<SIFTOrientationKeypoint*>(
      oct.orientationInputBuffer.contents);

  int validCount = 0;
  const int num_keypoints = static_cast<int>(keypoints.size());
  for (int k = 0; k < num_keypoints && validCount < keypoint_capacity_; ++k) {
    const auto& detected = keypoints[k];
    const auto& kp = detected.keypoint;
    float x = kp.x / delta;
    float y = kp.y / delta;
    float sigma = kp.sigma / delta;
    float r = std::ceil(3.0f * lambda * sigma);

    if (std::floor(x - r) < minX || std::ceil(x + r) > maxX ||
        std::floor(y - r) < minY || std::ceil(y + r) > maxY) {
      continue;
    }

    orientIn[validCount].index = static_cast<int32_t>(k);
    orientIn[validCount].absoluteX = kp.x;
    orientIn[validCount].absoluteY = kp.y;
    orientIn[validCount].scale = static_cast<int32_t>(detected.scale);
    orientIn[validCount].sigma = kp.sigma;
    ++validCount;
  }
  return validCount;
}

// ---------------------------------------------------------------------------
// EncodeOrientations
// ---------------------------------------------------------------------------
bool SiftMetalExtractorImpl::EncodeOrientations(id<MTLCommandBuffer> cb,
                                                Octave& oct, int count) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc) {
    return false;
  }
  [enc setComputePipelineState:siftOrientationPipeline_];
  [enc setBuffer:oct.orientationOutputBuffer offset:0 atIndex:0];
  [enc setBuffer:oct.orientationInputBuffer offset:0 atIndex:1];
  [enc setBuffer:oct.orientationParamsBuffer offset:0 atIndex:2];
  [enc setTexture:oct.gaussianTextures atIndex:0];

  NSUInteger maxThreads =
      siftOrientationPipeline_.maxTotalThreadsPerThreadgroup;
  MTLSize tg = {maxThreads, 1, 1};
  MTLSize gridSize = {(NSUInteger)count, 1, 1};
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tg];
  [enc endEncoding];
  return true;
}

// ---------------------------------------------------------------------------
// ReadOrientations
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::ReadOrientations(
    Octave& oct, int validCount,
    const std::vector<DetectedKeypoint>& keypoints,
    std::vector<std::pair<int, float>>* oriented) {
  oriented->clear();
  auto* orientOut = static_cast<SIFTOrientationResult*>(
      oct.orientationOutputBuffer.contents);
  for (int k = 0; k < validCount; ++k) {
    auto& res = orientOut[k];
    int kpIdx = static_cast<int>(res.keypoint);
    if (kpIdx < 0 || kpIdx >= static_cast<int>(keypoints.size())) {
      continue;
    }
    int count = static_cast<int>(res.count);
    int maxOrient = options_.upright ? 1 : options_.max_num_orientations;
    count = std::min({std::max(count, 0),
                      maxOrient,
                      SIFT_ORIENTATION_HISTOGRAM_BINS});
    float* oris = reinterpret_cast<float*>(&res.orientations);
    for (int i = 0; i < count; ++i) {
      float theta = options_.upright ? 0.0f : oris[i];
      if (!std::isfinite(theta)) {
        continue;
      }
      oriented->emplace_back(kpIdx, theta);
    }
  }
}

// ---------------------------------------------------------------------------
// PrepareDescriptorInputs
// ---------------------------------------------------------------------------
int SiftMetalExtractorImpl::PrepareDescriptorInputs(
    Octave& oct, const std::vector<DetectedKeypoint>& keypoints,
    const std::vector<std::pair<int, float>>& oriented) {
  const int count = std::min((int)oriented.size(), descriptor_capacity_);
  if (count == 0) return 0;

  auto* params =
      static_cast<SIFTDescriptorParameters*>(oct.descriptorParamsBuffer.contents);
  params->delta = oct.delta;
  params->scalesPerOctave = static_cast<int32_t>(oct.num_scales);
  params->width = static_cast<int32_t>(oct.width);
  params->height = static_cast<int32_t>(oct.height);

  auto* descIn =
      static_cast<SIFTDescriptorInput*>(oct.descriptorInputBuffer.contents);
  for (int i = 0; i < count; ++i) {
    int kpIdx = oriented[i].first;
    float theta = oriented[i].second;
    const auto& detected = keypoints[kpIdx];
    const auto& kp = detected.keypoint;

    descIn[i].keypoint = static_cast<int32_t>(kpIdx);
    descIn[i].absoluteX = kp.x;
    descIn[i].absoluteY = kp.y;
    descIn[i].scale = static_cast<int32_t>(detected.scale);
    descIn[i].subScale = detected.sub_scale;
    descIn[i].theta = theta;
  }
  return count;
}

// ---------------------------------------------------------------------------
// EncodeDescriptors
// ---------------------------------------------------------------------------
bool SiftMetalExtractorImpl::EncodeDescriptors(id<MTLCommandBuffer> cb,
                                               Octave& oct, int count) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc) {
    return false;
  }
  [enc setComputePipelineState:siftDescriptorsPipeline_];
  [enc setBuffer:oct.descriptorOutputBuffer offset:0 atIndex:0];
  [enc setBuffer:oct.descriptorInputBuffer offset:0 atIndex:1];
  [enc setBuffer:oct.descriptorParamsBuffer offset:0 atIndex:2];
  [enc setTexture:oct.gaussianTextures atIndex:0];

  NSUInteger maxThreads =
      siftDescriptorsPipeline_.maxTotalThreadsPerThreadgroup;
  MTLSize tg = {maxThreads, 1, 1};
  MTLSize gridSize = {(NSUInteger)count, 1, 1};
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tg];
  [enc endEncoding];
  return true;
}

// ---------------------------------------------------------------------------
// ReadDescriptors
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::ReadDescriptors(
    Octave& oct, int count, const std::vector<DetectedKeypoint>& keypoints,
    const std::vector<std::pair<int, float>>& oriented,
    ExtractResult* result) {
  auto* descOut = static_cast<SIFTDescriptorResult*>(
      oct.descriptorOutputBuffer.contents);
  for (int i = 0; i < count; ++i) {
    auto& dr = descOut[i];
    if (!dr.valid) continue;

    int kpIdx = oriented[i].first;
    if (kpIdx < 0 || kpIdx >= static_cast<int>(keypoints.size()) ||
        dr.keypoint != kpIdx || !std::isfinite(dr.theta)) {
      continue;
    }

    bool valid_descriptor = true;
    for (int j = 0; j < 128; ++j) {
      if (!std::isfinite(dr.features[j]) || dr.features[j] < 0.0f) {
        valid_descriptor = false;
        break;
      }
    }
    if (!valid_descriptor) {
      continue;
    }

    const auto& kp = keypoints[kpIdx].keypoint;

    Keypoint finalKp;
    finalKp.x = kp.x + 0.5f;
    finalKp.y = kp.y + 0.5f;
    finalKp.sigma = kp.sigma;
    finalKp.orientation = dr.theta;
    result->keypoints.push_back(finalKp);

    for (int j = 0; j < 128; ++j) {
      result->descriptors.push_back(dr.features[j]);
    }
  }
}

// ===========================================================================
// Public API
// ===========================================================================

SiftMetalExtractor::SiftMetalExtractor()
    : impl_(std::make_unique<SiftMetalExtractorImpl>()) {}

SiftMetalExtractor::~SiftMetalExtractor() = default;

bool SiftMetalExtractor::Init(const Options& options, int max_w, int max_h) {
  return impl_->Init(options, max_w, max_h);
}

bool SiftMetalExtractor::Extract(const uint8_t* data, int w, int h,
                                  ExtractResult* result) {
  return impl_->Extract(data, w, h, result);
}

SiftMetalMatcher::SiftMetalMatcher()
    : impl_(std::make_unique<SiftMetalMatcherImpl>()) {}

SiftMetalMatcher::~SiftMetalMatcher() = default;

bool SiftMetalMatcher::Init() { return impl_->Init(); }

bool SiftMetalMatcher::Match(const uint8_t* descriptors1,
                             int num_descriptors1,
                             const uint8_t* descriptors2,
                             int num_descriptors2,
                             const MatchOptions& options,
                             std::vector<MatchResult>* matches) {
  return impl_->Match(descriptors1,
                      num_descriptors1,
                      nullptr,
                      descriptors2,
                      num_descriptors2,
                      nullptr,
                      options,
                      MatchGuidedGeometry::NONE,
                      nullptr,
                      0.0f,
                      matches);
}

bool SiftMetalMatcher::MatchGuided(const uint8_t* descriptors1,
                                   int num_descriptors1,
                                   const MatchKeypoint* keypoints1,
                                   const uint8_t* descriptors2,
                                   int num_descriptors2,
                                   const MatchKeypoint* keypoints2,
                                   const MatchOptions& options,
                                   MatchGuidedGeometry guided_geometry,
                                   const float matrix[9],
                                   float max_residual,
                                   std::vector<MatchResult>* matches) {
  return impl_->Match(descriptors1,
                      num_descriptors1,
                      keypoints1,
                      descriptors2,
                      num_descriptors2,
                      keypoints2,
                      options,
                      guided_geometry,
                      matrix,
                      max_residual,
                      matches);
}

}  // namespace sift_metal
