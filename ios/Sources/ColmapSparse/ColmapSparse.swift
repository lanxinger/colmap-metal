// Copyright (c) 2026 Plinth. SPDX-License-Identifier: BSD-3-Clause
import ColmapSparseNative
import Foundation

/// Intrinsics use COLMAP's image-edge coordinates, with the first pixel center
/// at (0.5, 0.5). They must describe the encoded raster before EXIF rotation.
public enum SparseCameraModel: Sendable, Equatable {
  case pinhole(fx: Double, fy: Double, cx: Double, cy: Double)
  case simpleRadial(f: Double, cx: Double, cy: Double, k1: Double)
  case radial(f: Double, cx: Double, cy: Double, k1: Double, k2: Double)
  case openCV(
    fx: Double, fy: Double, cx: Double, cy: Double,
    k1: Double, k2: Double, p1: Double, p2: Double)

  var nativeValues: (cm_sparse_camera_model, [Double]) {
    switch self {
    case .pinhole(let fx, let fy, let cx, let cy):
      (CM_SPARSE_PINHOLE, [fx, fy, cx, cy])
    case .simpleRadial(let f, let cx, let cy, let k1):
      (CM_SPARSE_SIMPLE_RADIAL, [f, cx, cy, k1])
    case .radial(let f, let cx, let cy, let k1, let k2):
      (CM_SPARSE_RADIAL, [f, cx, cy, k1, k2])
    case .openCV(let fx, let fy, let cx, let cy, let k1, let k2, let p1, let p2):
      (CM_SPARSE_OPENCV, [fx, fy, cx, cy, k1, k2, p1, p2])
    }
  }
}

public struct SparseCamera: Sendable, Equatable {
  public let calibrationID: UInt32
  public let width: UInt32
  public let height: UInt32
  public let model: SparseCameraModel

  /// Reuse a calibration ID only for exactly equal dimensions and intrinsics.
  public init(
    calibrationID: UInt32, width: UInt32, height: UInt32,
    model: SparseCameraModel
  ) throws {
    guard (32...8192).contains(width), (32...8192).contains(height),
      UInt64(width) * UInt64(height) <= 16_000_000
    else {
      throw SparseError.invalidArgument(
        "Camera dimensions must be 32–8192 per axis and at most 16 MP.")
    }
    let (kind, values) = model.nativeValues
    let secondFocal = kind == CM_SPARSE_PINHOLE || kind == CM_SPARSE_OPENCV
    guard values.allSatisfy(\.isFinite), values[0] > 0,
      !secondFocal || values[1] > 0
    else {
      throw SparseError.invalidArgument("Intrinsics must be finite with positive focal lengths.")
    }
    self.calibrationID = calibrationID
    self.width = width
    self.height = height
    self.model = model
  }

  var native: cm_sparse_camera {
    let (kind, values) = model.nativeValues
    var result = cm_sparse_camera()
    result.calibration_id = calibrationID
    result.width = width
    result.height = height
    result.model = kind
    withUnsafeMutableBytes(of: &result.params) { bytes in
      let destination = bytes.bindMemory(to: Double.self)
      for (index, value) in values.enumerated() { destination[index] = value }
    }
    return result
  }
}

public struct SparseImage: Sendable {
  public let url: URL
  public let camera: SparseCamera

  public init(url: URL, camera: SparseCamera) throws {
    try validateFileURL(url)
    self.url = url
    self.camera = camera
  }
}

/// A rigid camera-to-world transform in the capture's existing world units.
/// The 16 values are column-major. Camera axes follow COLMAP/OpenCV: +X right,
/// +Y down, +Z forward. Convert ARKit's camera axes before constructing this pose.
public struct SparseCameraPose: Sendable, Equatable {
  public let matrixCameraToWorld: [Double]

  public init(matrixCameraToWorld matrix: [Double]) throws {
    guard matrix.count == 16, matrix.allSatisfy(\.isFinite),
      [3, 7, 11].allSatisfy({ abs(matrix[$0]) <= 1e-8 }),
      abs(matrix[15] - 1) <= 1e-8
    else {
      throw SparseError.invalidArgument("Camera pose must be a finite column-major 4×4 rigid transform.")
    }
    for column in 0..<3 {
      for otherColumn in 0..<3 {
        let dot = (0..<3).reduce(0.0) {
          $0 + matrix[column * 4 + $1] * matrix[otherColumn * 4 + $1]
        }
        guard abs(dot - (column == otherColumn ? 1 : 0)) <= 0.002 else {
          throw SparseError.invalidArgument("Camera pose rotation must be orthonormal.")
        }
      }
    }
    let determinant =
      matrix[0] * (matrix[5] * matrix[10] - matrix[9] * matrix[6])
      - matrix[4] * (matrix[1] * matrix[10] - matrix[9] * matrix[2])
      + matrix[8] * (matrix[1] * matrix[6] - matrix[5] * matrix[2])
    guard abs(determinant - 1) <= 0.002 else {
      throw SparseError.invalidArgument("Camera pose rotation must have determinant +1.")
    }
    matrixCameraToWorld = matrix
  }

  var native: cm_sparse_pose {
    var result = cm_sparse_pose()
    withUnsafeMutableBytes(of: &result.camera_to_world) { bytes in
      let destination = bytes.bindMemory(to: Double.self)
      for (index, value) in matrixCameraToWorld.enumerated() { destination[index] = value }
    }
    return result
  }

  init(native: cm_sparse_pose) throws {
    var native = native
    let matrix = withUnsafeBytes(of: &native.camera_to_world) {
      Array($0.bindMemory(to: Double.self))
    }
    try self.init(matrixCameraToWorld: matrix)
  }
}

public struct SparsePosedImage: Sendable {
  public let image: SparseImage
  public let pose: SparseCameraPose

  public init(image: SparseImage, pose: SparseCameraPose) {
    self.image = image
    self.pose = pose
  }
}

/// Bounds a fixed-intrinsics refinement initialized from all supplied poses.
public struct SparsePoseRefinementOptions: Sendable {
  /// Bundle-adjustment iteration limit, in 1...100.
  public var maxNumIterations: UInt32 = 20
  /// Maximum camera-center change in world units; meters for metric ARKit input.
  /// Must be finite, greater than zero, and at most 10.
  public var maxTranslationChange: Double = 0.15
  /// Maximum rotation change in radians, greater than zero and at most 0.5.
  public var maxRotationChangeRadians: Double = .pi / 36

  public init() {}

  var native: cm_sparse_pose_refinement_options {
    get throws {
      guard (1...100).contains(maxNumIterations),
        maxTranslationChange.isFinite, maxTranslationChange > 0, maxTranslationChange <= 10,
        maxRotationChangeRadians.isFinite, maxRotationChangeRadians > 0,
        maxRotationChangeRadians <= 0.5
      else {
        throw SparseError.invalidArgument("Pose refinement options exceed the supported bounds.")
      }
      var result = cm_sparse_default_pose_refinement_options()
      result.max_num_iterations = maxNumIterations
      result.max_translation_change = maxTranslationChange
      result.max_rotation_change_radians = maxRotationChangeRadians
      return result
    }
  }
}

/// Native validation rejects values outside the documented resource bounds.
public struct SparseOptions: Sendable {
  public var maxImageSize: UInt32 = 960
  public var maxNumFeatures: UInt32 = 8192
  public var numThreads: UInt32 = 2
  public var sequentialOverlap: UInt32 = 10
  public var keyframeStride: UInt32 = 8
  public var maxImages: UInt32 = 256
  public var maxNumPairs: UInt32 = 4096
  public var maxRuntimeSeconds: UInt32 = 300
  public var firstOctave: Int32 = -1
  public var matchingCacheBytes: UInt64 = 32 * 1024 * 1024
  public var minimumRegisteredFraction: Double = 0.9
  public var refineIntrinsics: Bool = false

  public init() {}

  var native: cm_sparse_options {
    var result = cm_sparse_default_options()
    result.max_image_size = maxImageSize
    result.max_num_features = maxNumFeatures
    result.num_threads = numThreads
    result.sequential_overlap = sequentialOverlap
    result.keyframe_stride = keyframeStride
    result.max_images = maxImages
    result.max_num_pairs = maxNumPairs
    result.max_runtime_seconds = maxRuntimeSeconds
    result.first_octave = firstOctave
    result.matching_cache_bytes = matchingCacheBytes
    result.minimum_registered_fraction = minimumRegisteredFraction
    result.refine_intrinsics = refineIntrinsics ? 1 : 0
    return result
  }
}

public struct SparseProgress: Sendable {
  public enum Stage: Sendable {
    case preparing, extracting, matching, mapping, exporting, finished
  }

  public let stage: Stage
  public let completed: UInt32
  public let total: UInt32

  init?(native: cm_sparse_stage, completed: UInt32, total: UInt32) {
    switch native {
    case CM_SPARSE_PREPARING: stage = .preparing
    case CM_SPARSE_EXTRACTING: stage = .extracting
    case CM_SPARSE_MATCHING: stage = .matching
    case CM_SPARSE_MAPPING: stage = .mapping
    case CM_SPARSE_EXPORTING: stage = .exporting
    case CM_SPARSE_FINISHED: stage = .finished
    default: return nil
    }
    self.completed = completed
    self.total = total
  }
}

public struct SparseResult: Sendable {
  public let outputDirectory: URL
  public let registeredImages: UInt32
  public let inputImages: UInt32
  public let points: UInt64
  public let observations: UInt64
  public let meanReprojectionError: Double
  public let meanTrackLength: Double
  public let elapsedSeconds: Double

  init(outputDirectory: URL, native: cm_sparse_result) {
    self.outputDirectory = outputDirectory
    registeredImages = native.registered_images
    inputImages = native.input_images
    points = native.points
    observations = native.observations
    meanReprojectionError = native.mean_reprojection_error
    meanTrackLength = native.mean_track_length
    elapsedSeconds = native.elapsed_seconds
  }
}

public struct SparsePoseRefinementResult: Sendable {
  public let reconstruction: SparseResult
  /// Refined camera-to-world transforms in exactly the supplied image order,
  /// preserving the input world coordinate system and units.
  public let cameraPoses: [SparseCameraPose]
}

public enum SparseError: Error, Sendable, LocalizedError {
  case invalidArgument(String)
  case resource(String)
  case reconstructionFailed(String)
  case io(String)
  case internalFailure(String)

  public var errorDescription: String? {
    switch self {
    case .invalidArgument(let message), .resource(let message),
      .reconstructionFailed(let message), .io(let message),
      .internalFailure(let message):
      message
    }
  }
}

/// Serializes reconstructions on a dedicated queue and releases each native job
/// before returning. Cancel the calling Swift Task to request cooperative stop.
public struct SparseReconstructor: Sendable {
  static let workerQueue = DispatchQueue(label: "com.plinth.ColmapSparse", qos: .userInitiated)

  public init() {}

  /// Images must be in capture order. The destination must not exist and its
  /// parent must exist. Progress runs synchronously on the worker: keep it
  /// brief and dispatch UI updates to MainActor.
  public func reconstruct(
    images: [SparseImage], outputDirectory: URL,
    options: SparseOptions = .init(),
    progress: (@Sendable (SparseProgress) -> Void)? = nil
  ) async throws -> SparseResult {
    let result = try await execute(
      images: images, outputDirectory: outputDirectory,
      options: options, progress: progress)
    return result.reconstruction
  }

  /// Triangulates and refines at least three images from their supplied poses.
  /// Intrinsics remain fixed: `options.refineIntrinsics` must be false. Encoded
  /// images must have no EXIF rotation. The first and farthest camera poses are
  /// fixed to retain the existing world frame and metric baseline. Only the
  /// largest strongly supported camera group can move, with its own two anchors;
  /// groups joined through only one camera are treated separately, and all
  /// remaining cameras retain their input poses. Refinement outside the
  /// configured motion bounds fails instead of publishing a result.
  /// Output-directory and progress requirements match `reconstruct`.
  public func refineKnownPoses(
    images: [SparsePosedImage], outputDirectory: URL,
    options: SparseOptions = .init(),
    refinementOptions: SparsePoseRefinementOptions = .init(),
    progress: (@Sendable (SparseProgress) -> Void)? = nil
  ) async throws -> SparsePoseRefinementResult {
    let result = try await execute(
      images: images.map(\.image), poses: images.map(\.pose),
      outputDirectory: outputDirectory, options: options,
      refinementOptions: refinementOptions, progress: progress)
    return SparsePoseRefinementResult(
      reconstruction: result.reconstruction, cameraPoses: result.cameraPoses)
  }

  private typealias RunResult = (reconstruction: SparseResult, cameraPoses: [SparseCameraPose])

  private func execute(
    images: [SparseImage], poses: [SparseCameraPose]? = nil,
    outputDirectory: URL, options: SparseOptions,
    refinementOptions: SparsePoseRefinementOptions = .init(),
    progress: (@Sendable (SparseProgress) -> Void)?
  ) async throws -> RunResult {
    try validateFileURL(outputDirectory)
    let state = JobState()
    return try await withTaskCancellationHandler {
      try Task.checkCancellation()
      return try await withCheckedThrowingContinuation { continuation in
        Self.workerQueue.async {
          do {
            let result = try Self.run(
              images: images, poses: poses, outputDirectory: outputDirectory,
              options: options, refinementOptions: refinementOptions,
              progress: progress, state: state)
            continuation.resume(returning: result)
          } catch {
            continuation.resume(throwing: error)
          }
        }
      }
    } onCancel: {
      state.cancel()
    }
  }

  static func metallibURL() throws -> URL {
    #if targetEnvironment(simulator)
      let platform = "iphonesimulator"
    #elseif os(iOS)
      let platform = "iphoneos"
    #elseif os(macOS)
      let platform = "macosx"
    #else
      #error("ColmapSparse supports iOS and macOS only")
    #endif
    guard
      let url = Bundle.module.url(
        forResource: "sift", withExtension: "metallib",
        subdirectory: "Resources/\(platform)")
    else {
      throw SparseError.resource(
        "The package's \(platform) SIFT metallib is missing. Rebuild with scripts/build-ios.sh all."
      )
    }
    return url
  }

  private static func run(
    images: [SparseImage], poses: [SparseCameraPose]?,
    outputDirectory: URL, options: SparseOptions,
    refinementOptions: SparsePoseRefinementOptions,
    progress: (@Sendable (SparseProgress) -> Void)?, state: JobState
  ) throws -> RunResult {
    try state.checkCancellation()
    guard (2...256).contains(images.count), images.count <= options.maxImages else {
      throw SparseError.invalidArgument("Image count must be 2–256 and within maxImages.")
    }
    guard cm_sparse_abi_version() == CM_SPARSE_ABI_VERSION else {
      throw SparseError.internalFailure("ColmapSparse native ABI does not match the Swift package.")
    }
    var nativeRefinementOptions = cm_sparse_pose_refinement_options()
    if let poses {
      guard poses.count == images.count, images.count >= 3, !options.refineIntrinsics else {
        throw SparseError.invalidArgument(
          "Pose refinement requires at least three posed images and fixed intrinsics.")
      }
      nativeRefinementOptions = try refinementOptions.native
    }
    let metallib = try metallibURL()
    var paths: [UnsafeMutablePointer<CChar>] = []
    defer { for path in paths { free(path) } }
    var nativeImages: [cm_sparse_image] = []
    for image in images {
      guard let path = strdup(image.url.path) else {
        throw SparseError.resource("Unable to allocate an image path.")
      }
      paths.append(path)
      nativeImages.append(cm_sparse_image(path: UnsafePointer(path), camera: image.camera.native))
    }
    var nativeOptions = options.native
    let nativePoses = poses?.map(\.native) ?? []
    var job: OpaquePointer?
    var diagnostic = [CChar](repeating: 0, count: 2048)
    let status = outputDirectory.path.withCString { outputPath in
      metallib.path.withCString { metallibPath in
        if poses != nil {
          return cm_sparse_job_create_with_poses(
            nativeImages, nativePoses, nativeImages.count,
            &nativeOptions, &nativeRefinementOptions,
            outputPath, metallibPath, &job, &diagnostic, diagnostic.count)
        } else {
          return cm_sparse_job_create(
            nativeImages, nativeImages.count, &nativeOptions,
            outputPath, metallibPath, &job, &diagnostic, diagnostic.count)
        }
      }
    }
    guard status == CM_SPARSE_SUCCESS, let job else {
      let message = String(
        decoding: diagnostic.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
      throw nativeError(status, message: message)
    }
    state.install(job)
    defer { state.destroy() }
    let relay = ProgressRelay(callback: progress)
    var result = cm_sparse_result()
    let runStatus = withExtendedLifetime(relay) {
      cm_sparse_job_run(
        job,
        { context, stage, completed, total in
          guard let context,
            let progress = SparseProgress(native: stage, completed: completed, total: total)
          else {
            return
          }
          Unmanaged<ProgressRelay>.fromOpaque(context).takeUnretainedValue().callback?(progress)
        }, Unmanaged.passUnretained(relay).toOpaque(), &result)
    }
    guard runStatus == CM_SPARSE_SUCCESS else {
      let message =
        cm_sparse_job_error(job).map { String(cString: $0) } ?? "Native reconstruction failed."
      throw nativeError(runStatus, message: message)
    }
    // Native success includes atomic publication. Late cancellation must
    // still return the completed dataset, whose existence is now committed.
    var refinedPoses: [SparseCameraPose] = []
    if poses != nil {
      var nativeRefinedPoses = [cm_sparse_pose](repeating: cm_sparse_pose(), count: images.count)
      guard cm_sparse_job_copy_refined_poses(job, &nativeRefinedPoses, nativeRefinedPoses.count)
        == CM_SPARSE_SUCCESS
      else {
        throw SparseError.internalFailure("Native refinement did not return every input camera pose.")
      }
      do {
        refinedPoses = try nativeRefinedPoses.map { try SparseCameraPose(native: $0) }
      } catch {
        throw SparseError.internalFailure("Native refinement returned a malformed camera pose.")
      }
    }
    return (SparseResult(outputDirectory: outputDirectory, native: result), refinedPoses)
  }
}

private func validateFileURL(_ url: URL) throws {
  // Foundation's decoded path can truncate at an embedded NUL, so inspect the
  // encoded path too before passing any path through the C interface.
  guard url.isFileURL, !url.path.isEmpty, !url.path.utf8.contains(0),
    !url.path(percentEncoded: true).contains("%00")
  else {
    throw SparseError.invalidArgument("A nonempty file URL without NUL characters is required.")
  }
}

private func nativeError(_ status: cm_sparse_status, message: String) -> any Error {
  switch status {
  case CM_SPARSE_CANCELLED: CancellationError()
  case CM_SPARSE_INVALID_ARGUMENT: SparseError.invalidArgument(message)
  case CM_SPARSE_RESOURCE_ERROR: SparseError.resource(message)
  case CM_SPARSE_RECONSTRUCTION_FAILED: SparseError.reconstructionFailed(message)
  case CM_SPARSE_IO_ERROR: SparseError.io(message)
  default: SparseError.internalFailure(message)
  }
}

private final class ProgressRelay {
  let callback: (@Sendable (SparseProgress) -> Void)?
  init(callback: (@Sendable (SparseProgress) -> Void)?) { self.callback = callback }
}

// The pointer is installed/destroyed only by the worker; cancellation may run
// anywhere. One lock protects pointer lifetime across cancel versus destroy.
private final class JobState: @unchecked Sendable {
  private let lock = NSLock()
  private var job: OpaquePointer?
  private var cancelled = false

  func checkCancellation() throws {
    lock.lock()
    let isCancelled = cancelled
    lock.unlock()
    if isCancelled { throw CancellationError() }
  }

  func install(_ job: OpaquePointer) {
    lock.lock()
    self.job = job
    if cancelled { cm_sparse_job_cancel(job) }
    lock.unlock()
  }

  func cancel() {
    lock.lock()
    cancelled = true
    if let job { cm_sparse_job_cancel(job) }
    lock.unlock()
  }

  func destroy() {
    lock.lock()
    if let job { cm_sparse_job_destroy(job) }
    job = nil
    lock.unlock()
  }
}
