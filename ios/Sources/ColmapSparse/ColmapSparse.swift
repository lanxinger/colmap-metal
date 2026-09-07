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
    try validateFileURL(outputDirectory)
    let state = JobState()
    return try await withTaskCancellationHandler {
      try Task.checkCancellation()
      return try await withCheckedThrowingContinuation { continuation in
        Self.workerQueue.async {
          do {
            let result = try Self.run(
              images: images, outputDirectory: outputDirectory,
              options: options, progress: progress, state: state)
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
    images: [SparseImage], outputDirectory: URL, options: SparseOptions,
    progress: (@Sendable (SparseProgress) -> Void)?, state: JobState
  ) throws -> SparseResult {
    try state.checkCancellation()
    guard (2...256).contains(images.count), images.count <= options.maxImages else {
      throw SparseError.invalidArgument("Image count must be 2–256 and within maxImages.")
    }
    guard cm_sparse_abi_version() == CM_SPARSE_ABI_VERSION else {
      throw SparseError.internalFailure("ColmapSparse native ABI does not match the Swift package.")
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
    var job: OpaquePointer?
    var diagnostic = [CChar](repeating: 0, count: 2048)
    let status = outputDirectory.path.withCString { outputPath in
      metallib.path.withCString { metallibPath in
        cm_sparse_job_create(
          nativeImages, nativeImages.count, &nativeOptions,
          outputPath, metallibPath, &job, &diagnostic, diagnostic.count)
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
    return SparseResult(outputDirectory: outputDirectory, native: result)
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
