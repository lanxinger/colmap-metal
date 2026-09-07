import ColmapSparseNative
import CoreGraphics
import Foundation
import ImageIO
import Testing

// Copyright (c) 2026 Plinth. SPDX-License-Identifier: BSD-3-Clause
@testable import ColmapSparse

@Suite(.serialized)
struct ColmapSparseTests {
  @Test func cameraModelMappingAndValidation() throws {
    let models: [(SparseCameraModel, cm_sparse_camera_model, [Double])] = [
      (.pinhole(fx: 50, fy: 51, cx: 32, cy: 33), CM_SPARSE_PINHOLE, [50, 51, 32, 33]),
      (
        .simpleRadial(f: 50, cx: 32, cy: 33, k1: 0.01), CM_SPARSE_SIMPLE_RADIAL,
        [50, 32, 33, 0.01]
      ),
      (
        .radial(f: 50, cx: 32, cy: 33, k1: 0.01, k2: -0.02), CM_SPARSE_RADIAL,
        [50, 32, 33, 0.01, -0.02]
      ),
      (
        .openCV(fx: 50, fy: 51, cx: 32, cy: 33, k1: 0.01, k2: -0.02, p1: 0.03, p2: -0.04),
        CM_SPARSE_OPENCV, [50, 51, 32, 33, 0.01, -0.02, 0.03, -0.04]
      ),
    ]
    for (model, kind, expected) in models {
      let camera = try SparseCamera(calibrationID: 17, width: 64, height: 64, model: model)
      var native = camera.native
      #expect(native.calibration_id == 17)
      #expect(native.model == kind)
      #expect(native.width == 64 && native.height == 64)
      let actual = withUnsafeBytes(of: &native.params) { Array($0.bindMemory(to: Double.self)) }
      #expect(actual == expected + Array(repeating: 0, count: 8 - expected.count))
    }
    #expect(throws: SparseError.self) {
      try SparseCamera(
        calibrationID: 0, width: 31, height: 64,
        model: .pinhole(fx: 50, fy: 50, cx: 32, cy: 32))
    }
    #expect(throws: SparseError.self) {
      try SparseCamera(
        calibrationID: 0, width: 8192, height: 8192,
        model: .pinhole(fx: 50, fy: 50, cx: 32, cy: 32))
    }
    for focal in [0, -1, Double.nan, Double.infinity] {
      #expect(throws: SparseError.self) {
        try SparseCamera(
          calibrationID: 0, width: 64, height: 64,
          model: .pinhole(fx: 50, fy: focal, cx: 32, cy: 32))
      }
    }
    let camera = try fixtureCamera()
    #expect(throws: SparseError.self) {
      try SparseImage(url: URL(string: "https://example.com/photo.jpg")!, camera: camera)
    }
    #expect(throws: SparseError.self) {
      try SparseImage(url: URL(fileURLWithPath: "/tmp/invalid\0photo.jpg"), camera: camera)
    }
  }

  @Test func packageResourcesAndNativeDefaults() throws {
    let url = try SparseReconstructor.metallibURL()
    #expect(FileManager.default.fileExists(atPath: url.path))
    #expect(try Data(contentsOf: url).count > 1024)
    let defaults = cm_sparse_default_options()
    let options = SparseOptions().native
    #expect(cm_sparse_abi_version() == CM_SPARSE_ABI_VERSION)
    #expect(options.struct_size == defaults.struct_size)
    #expect(options.abi_version == defaults.abi_version)
    #expect(options.max_image_size == defaults.max_image_size)
    #expect(options.max_num_features == defaults.max_num_features)
    #expect(options.num_threads == defaults.num_threads)
    #expect(options.sequential_overlap == defaults.sequential_overlap)
    #expect(options.keyframe_stride == defaults.keyframe_stride)
    #expect(options.max_images == defaults.max_images)
    #expect(options.max_num_pairs == defaults.max_num_pairs)
    #expect(options.max_runtime_seconds == defaults.max_runtime_seconds)
    #expect(options.first_octave == defaults.first_octave)
    #expect(options.matching_cache_bytes == defaults.matching_cache_bytes)
    #expect(options.minimum_registered_fraction == defaults.minimum_registered_fraction)
    #expect(options.refine_intrinsics == defaults.refine_intrinsics)
  }

  @Test func nativeValidationPreservesExistingOutput() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let output = fixture.directory.appendingPathComponent("existing")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    let sentinel = output.appendingPathComponent("keep.txt")
    try Data("keep this output".utf8).write(to: sentinel)
    do {
      _ = try await SparseReconstructor().reconstruct(
        images: fixture.images, outputDirectory: output)
      Issue.record("Existing output must be rejected")
    } catch SparseError.invalidArgument(let message) {
      #expect(message.contains("already exists"))
    }
    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "keep this output")

    var options = SparseOptions()
    options.maxImageSize = 0
    let invalidOutput = fixture.directory.appendingPathComponent("invalid-options")
    do {
      _ = try await SparseReconstructor().reconstruct(
        images: fixture.images, outputDirectory: invalidOutput, options: options)
      Issue.record("Out-of-range options must be rejected")
    } catch SparseError.invalidArgument {}
    #expect(!FileManager.default.fileExists(atPath: invalidOutput.path))

    let changedCamera = try SparseCamera(
      calibrationID: 1, width: 64, height: 64,
      model: .pinhole(fx: 51, fy: 50, cx: 32, cy: 32))
    let conflicting = try SparseImage(url: fixture.images[1].url, camera: changedCamera)
    do {
      _ = try await SparseReconstructor().reconstruct(
        images: [fixture.images[0], conflicting], outputDirectory: invalidOutput)
      Issue.record("Shared calibration IDs must agree")
    } catch SparseError.invalidArgument(let message) {
      #expect(message.contains("different calibration"))
    }
    #expect(!FileManager.default.fileExists(atPath: invalidOutput.path))
  }

  @Test func cancellationBeforeWorkerExecution() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let output = fixture.directory.appendingPathComponent("cancelled")
    let gate = DispatchSemaphore(value: 0)
    await withCheckedContinuation { (ready: CheckedContinuation<Void, Never>) in
      SparseReconstructor.workerQueue.async {
        ready.resume()
        gate.wait()
      }
    }
    let task = Task {
      try await SparseReconstructor().reconstruct(images: fixture.images, outputDirectory: output)
    }
    task.cancel()
    gate.signal()
    await expectCancellation(task)
    #expect(!FileManager.default.fileExists(atPath: output.path))
    #expect(try FileManager.default.contentsOfDirectory(atPath: fixture.directory.path).count == 2)
  }

  @Test func progressCancellationAndJobLifetime() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    // Repetition verifies cleanup after callback cancellation and proves that
    // cancellation releases the process-wide native run slot for reuse.
    for index in 0..<20 {
      let output = fixture.directory.appendingPathComponent("cancelled-\(index)")
      let probe = CancellationProbe()
      let task = Task {
        try await SparseReconstructor().reconstruct(
          images: fixture.images, outputDirectory: output
        ) { progress in
          if progress.stage == .preparing { probe.cancelFromProgress() }
        }
      }
      probe.install(task)
      await expectCancellation(task)
      let (called, onMainThread) = probe.snapshot()
      #expect(called)
      #expect(!onMainThread)
      #expect(!FileManager.default.fileExists(atPath: output.path))
      #expect(
        try FileManager.default.contentsOfDirectory(atPath: fixture.directory.path).count == 2)
    }
  }
}

private func fixtureCamera() throws -> SparseCamera {
  try SparseCamera(
    calibrationID: 1, width: 64, height: 64,
    model: .pinhole(fx: 50, fy: 50, cx: 32, cy: 32))
}

private struct Fixture: Sendable {
  let directory: URL
  let images: [SparseImage]

  init() throws {
    directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("ColmapSparseTests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    do {
      let context = try #require(
        CGContext(
          data: nil, width: 64, height: 64, bitsPerComponent: 8,
          bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
      context.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.6, alpha: 1))
      context.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
      let image = try #require(context.makeImage())
      let camera = try fixtureCamera()
      var inputs: [SparseImage] = []
      for index in 0..<2 {
        let url = directory.appendingPathComponent("\(index).png")
        let destination = try #require(
          CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        inputs.append(try SparseImage(url: url, camera: camera))
      }
      images = inputs
    } catch {
      try? FileManager.default.removeItem(at: directory)
      throw error
    }
  }

  func remove() { try? FileManager.default.removeItem(at: directory) }
}

private func expectCancellation(_ task: Task<SparseResult, any Error>) async {
  do {
    _ = try await task.value
    Issue.record("Cancelled operation unexpectedly succeeded")
  } catch is CancellationError {
  } catch {
    Issue.record("Expected CancellationError; got \(error)")
  }
}

private final class CancellationProbe: @unchecked Sendable {
  private let lock = NSLock()
  private var task: Task<SparseResult, any Error>?
  private var called = false
  private var onMainThread = false

  func install(_ task: Task<SparseResult, any Error>) {
    lock.lock()
    self.task = task
    if called { task.cancel() }
    lock.unlock()
  }

  func cancelFromProgress() {
    lock.lock()
    called = true
    onMainThread = Thread.isMainThread
    task?.cancel()
    lock.unlock()
  }

  func snapshot() -> (Bool, Bool) {
    lock.lock()
    defer { lock.unlock() }
    return (called, onMainThread)
  }
}
