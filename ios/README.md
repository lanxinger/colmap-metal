# ColmapSparse for Apple platforms

An in-process sparse reconstruction pilot for iOS 18+, with arm64 iPhone,
arm64 Simulator and arm64 macOS 13+ slices. It produces a standard COLMAP dataset
for msplat: original encoded images, calibrated cameras, poses, colored points
and tracks. The complete sparse pipeline has run on an iPhone 16 Pro Max;
broader device and downstream training-quality qualification remains required
before shipping.

## Build the local Swift package

From the repository root, with Xcode selected and CMake, Ninja and Boost 1.92.0
headers available:

```sh
bash scripts/build-ios.sh all
```

The script verifies downloaded Eigen, glog, Ceres and PoseLib source archives,
builds static dependencies for each SDK, compiles that SDK's Metal shaders and
creates `build-ios/ColmapSparse.xcframework`. The package preparation step copies
the result to `Artifacts/ColmapSparse.xcframework` and installs the three
platform-specific shader resources for SwiftPM. Generated binaries and resources
are not committed. Add this repository as a local Swift package after building.

Run `swift test --jobs 2` to check the Swift calibration, shader resources,
output protection and cancellation behavior on Mac. The public import is
`ColmapSparse`; the binary module is `ColmapSparseNative`.

Each XCFramework slice keeps `HeadersPath = Headers`, with its C header and
module map under `Headers/ColmapSparseNative/`. Xcode copies static-library
headers into one shared build-products include directory; placing
`module.modulemap` at that directory's root collides with other binary packages,
including msplat. The build script creates this namespaced package header layout
separately from the ordinary C install headers. Import names and native ABI are
unchanged.

Set `COLMAP_IOS_BUILD_ROOT` to keep the build outside the checkout,
`COLMAP_IOS_BUILD_JOBS` to limit compiler concurrency, and
`COLMAP_IOS_BOOST_INCLUDE` if the pinned Boost headers are elsewhere. Boost is
exposed through an isolated include directory; Homebrew libraries and unrelated
Homebrew headers are excluded. Only arm64 Simulator and Mac are packaged.

The source target is `ios/CMakeLists.txt`. It bypasses the desktop dependency
discovery and CLI. The small versioned [C API](include/ColmapSparse.h) hides
C++/Eigen/Ceres types. Keep the [dependency notices](THIRD_PARTY_NOTICES.md) and
their `Licenses/` files with any distributed package.

## Capture and reconstruction contract

```swift
import ColmapSparse

let camera = try SparseCamera(
    calibrationID: 1, width: 1200, height: 1600,
    model: .pinhole(fx: fx, fy: fy, cx: cx, cy: cy)
)
let inputs = try captureURLs.map { try SparseImage(url: $0, camera: camera) }
let reconstructionTask = Task {
    try await SparseReconstructor().reconstruct(
        images: inputs, outputDirectory: newDatasetDirectory
    ) { progress in
        Task { @MainActor in
            // Update the application's progress display.
        }
    }
}
let result = try await reconstructionTask.value
// Pass result.outputDirectory to msplat's COLMAP dataset loader.
```

The intrinsics in this example must come from the actual capture calibration;
the dimensions are illustrative. Call `reconstructionTask.cancel()` to cancel.

Provide images in capture order, with explicit camera calibration for each
encoded raster. Shared `calibrationID` values mean exactly identical camera
model, dimensions and intrinsic parameters. Different lenses, zoom levels,
resolutions or rotated rasters need separate calibration IDs.

EXIF rotation is preserved as metadata and is not applied during reconstruction.
Camera dimensions and intrinsics must describe the encoded pixels before that
rotation. Pixel coordinates use COLMAP's image-edge convention: the first pixel
center is `(0.5, 0.5)`. Do not pass intrinsics for a display-oriented `UIImage`
beside its original, differently oriented JPEG.

Supported models are PINHOLE, SIMPLE_RADIAL, RADIAL and OPENCV. JPEG, PNG and HEIC
inputs must be 32–8192 pixels per axis and at most 16 million encoded pixels.
The package copies inputs to its working directory before extracting features,
so the source files must remain readable until preparation finishes.

The output directory must not exist and its parent must already exist. On
success it contains:

```text
dataset/
  images/000001.jpg, ...
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin
  sparse/0/rigs.bin
  sparse/0/frames.bin
```

Only registered images from the largest connected model are exported, retaining
capture-order filenames. Disconnected models are never combined into one seed
cloud. A model must meet `minimumRegisteredFraction` (90% by default). The
directory is published atomically without replacing an existing destination;
failure or cancellation removes the job's working directory.

## Resource defaults

| Setting | Default | Allowed range |
| --- | --- | --- |
| Extraction long edge | 960 px | 128–1600 px |
| Descriptor rows per image | 8192, after orientation expansion | 128–8192 |
| First octave | −1 | −1 or 0 |
| CPU threads | 2 | 1–4 |
| Sequential overlap | 10 | 1–30 |
| Nonlocal keyframes | Every eighth image plus the last | Zero disables; 1–256 |
| Input images | Up to 256 | 2–256 |
| Candidate pair limit | 4096 | 1–8192 |
| Persistent GPU descriptor cache | 32 MiB | 0–64 MiB |
| Whole-job runtime limit | 300 s | 1–3600 s |
| Intrinsic refinement | Disabled | Opt in when appropriate |

The pair schedule combines the sequential window with all pairs between selected
keyframes, excluding duplicates. A schedule exceeding the pair limit is rejected
before work starts. Pose and track connectivity still depends on the capture.

Extraction and matching resources are released before mapping. One native job
can run per process; another concurrent run returns a resource error. The input,
feature, pair and cache limits bound work but are not a guarantee of a particular
peak footprint on every device. First octave 0 includes the corrected SIFT base
scale; validate its reconstruction quality before selecting it for production.

Mapping uses Ceres with Accelerate sparse algebra and Eigen dense algebra. The
package does not link SuiteSparse, desktop OpenGL, SiftGPU, FAISS, OpenImageIO,
neural inference, MVS or meshing. Camera-ray homography pairs retain their
geometrically verified inliers; this pilot does not expand those pairs with the
pixel-space guided matcher.

## Cancellation and app lifecycle

The Swift operation runs on a dedicated queue. Progress callbacks run on that
worker, so dispatch UI updates to `MainActor`. Cancel its Swift `Task` when the
user cancels or when the application's lifecycle policy requires interruption.
The package does not install signal handlers or request background execution.

Cancellation and the runtime deadline are cooperative. An in-flight GPU command,
file operation or solver step can finish before cancellation is observed. A
successful atomic publication is the commit point; cancellation arriving after
that point does not remove the completed dataset. Wait for the operation to
return before starting msplat so the native working resources have been released.

For direct C callers, keep the job alive while `cm_sparse_job_run` executes and
while another thread calls `cm_sparse_job_cancel`. Never destroy or rerun a job
inside its progress callback. The header documents the complete lifetime rules.

## msplat handoff

Pass the exported dataset directory to msplat's COLMAP loader. The images retain
their original encoded resolution; feature downscaling is reflected in rescaled
observations, and the original calibration is exported. msplat can perform its
supported camera undistortion during loading. Reconstruction units and global
orientation are arbitrary unless the application supplies a later alignment.

An alternative application adapter can expose the same cameras, points and
tracks through msplat's `DatasetDescriptor`. It must invert COLMAP's
world-to-camera transform and apply msplat's camera-axis conversion. Folder
loading already performs that conversion and is the initial integration path.

## Native validation

After `bash scripts/build-ios.sh macosx`:

```sh
build-ios/core/macosx-arm64/bitmap_apple_test
build-ios/core/macosx-arm64/sparse_smoke --test \
  build-ios/core/macosx-arm64/sift.metallib
```

A calibrated fixture can exercise the entire package path:

```sh
build-ios/core/macosx-arm64/sparse_smoke \
  IMAGE_DIRECTORY NEW_OUTPUT_DIRECTORY \
  build-ios/core/macosx-arm64/sift.metallib WIDTH HEIGHT FX FY CX CY
```

The smoke executable reports stage progress and final model statistics. Its
argument order is a shared PINHOLE calibration; the library supports per-image
calibrations through its public API. Host compilation, unit tests and loader
acceptance do not establish iPhone latency, thermal behavior, background
interruption behavior or downstream Gaussian quality.

## Measured iPhone pilot

On September 7, 2026, the default package reconstructed one connected model from
133 calibrated 1200×1600 JPEGs on an iPhone 16 Pro Max (A18 Pro, iOS 27 beta
24A5430a). The native job took 216.82 seconds and exported 70,848 colored points
with 355,384 observations and mean point reprojection error 0.6543 pixels.
Process footprint peaked at 364.5 MiB with 10 ms sampling; thermal state started
nominal and ended fair. The phone was connected to power with Low Power Mode off.

The exported images were byte-identical to the inputs. Independent validation
checked finite poses, reciprocal tracks, positive projected depth and encoded
image bounds. After similarity alignment to the Mac reconstruction, the 95th
percentile camera rotation difference was 0.0314 degrees and camera-center
difference was 0.0209% of the Mac camera-trajectory diameter. These establish
consistency on this fixture, not ground-truth accuracy or novel-view quality.

This is a single device/fixture measurement of sparse reconstruction. It does
not establish the complete capture-to-Gaussian preview budget or behavior on
the minimum supported OS, other devices, camera models or capture patterns.

A second run after about three minutes of idle time also registered 133/133
images, producing 70,863 points in 210.54 seconds with a 369.39 MiB sampled
reconstruction peak. It started at nominal thermal state but reached serious
during matching and remained serious through completion. Matching took about
136 seconds in both runs; these timings do not establish throttling.

In that same process, a new job cancelled after its first extraction returned
within 18 ms of the request and left no output or staging directory. The released
msplat 2.1.0 iPhone library (ABI 18) then loaded all 133 views, completed 133 GPU
steps at 150×200, and produced a finite, non-flat render. Training setup, steps
and rendering took 2.90 seconds with a separately sampled 250.80 MiB process
footprint peak. This short test precedes densification warmup and establishes
functional handoff.

A subsequent matched training comparison used released msplat 2.1.0 with
Plinth's 2,000-step, 1200×1600 preview settings. The same 133 JPEGs, masks and
calibration were used with the retained RealityKit sparse dataset and the first
phone-generated COLMAP dataset. For evaluation, both trained on 116 views and
held out 17; production trains all 133, and both sparse reconstructions had used
all photos.

| Measurement | RealityKit | COLMAP phone |
|---|---:|---:|
| Initial points / final Gaussians | 17,990 / 40,924 | 70,848 / 65,174 |
| Held-out foreground PSNR from pooled MSE | 21.01 dB | 21.71 dB |
| Completed 2,000-step training | 90.98 s | 175.39 s |
| Peak process footprint through training | 758.0 MiB | 825.3 MiB |
| Worst observed thermal state | Serious | Serious |

COLMAP improved foreground error in all 17 views, reducing pooled MSE by 14.86%.
Both runs completed GPU step 2,000 without recorded GPU failures and exported
finite renders and PLY data. They used the same binary in separate processes
after 60 continuous seconds at nominal thermal state; this does not prove equal
internal temperature or repeatable timing. The comparison changes camera poses
and seed geometry/count together, so it does not isolate the cause of the gain.

The separately measured reconstruction plus COLMAP training, setup and PLY export
sum to about 6 minutes 33 seconds, before remaining app preparation and display.
This is not a measured continuous preview run and already exceeds the five-minute
target. The package remains experimental; initial rendering cost, matching time,
repeated-run thermal handling and broader device/capture quality need further
qualification.

## Plinth default integration

The local Plinth `ios27` integration selects COLMAP for Gaussian preview alignment,
retains the existing 2,000-step msplat trainer and masks, and offers an explicit
RealityKit retry when calibration, input ordering, reconstruction, or supported
photo-count limits require it. Its draft Preview button uses COLMAP's 256-image
limit while retaining the existing renderer and project eligibility checks.

On September 7, 2026, the complete Plinth-Staging pipeline passed on the connected
iPhone with 133 normalized 1200×1600 JPEGs carrying 35mm-equivalent EXIF metadata.
Intrinsic refinement produced one SIMPLE_RADIAL camera, 133 registered images and
70,919 sparse points. All 2,000 training steps completed, exporting 65,732 Gaussians.
The actual RealityKit viewer was captured with `ARView.snapshot`; the model was
visible and upright. Source hashes were unchanged and app cleanup removed the
temporary run. Fifty-four focused app tests also passed.

The recorded pipeline-to-snapshot time was 478 seconds, excluding a separate
60.65-second test cooldown and including diagnostic output copying and three
seconds of renderer settling. The worst sampled thermal state was fair. The user
accepted the extra runtime in favor of the earlier measured quality gain; this
single app run is not a matched quality comparison or a general latency guarantee.

Original guided ObjectCapture HEIC metadata and lens correction still require a
fresh capture check. The integration currently uses a verified local sibling
package snapshot; a standalone Plinth clone needs that snapshot and generated
artifacts until a reviewed package revision is published.
