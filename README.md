COLMAP-Metal
============

About
-----

COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo
(MVS) pipeline with a graphical and command-line interface. It offers a wide
range of features for reconstruction of ordered and unordered image collections.
The software is licensed under the new BSD license.

This is the maintained [COLMAP-Metal fork](https://github.com/lanxinger/colmap-metal)
of [upstream COLMAP](https://github.com/colmap/colmap). It adds Metal acceleration
for Apple platforms and a local Swift package for sparse reconstruction on iPhone
and Mac. The desktop CLI, optional Qt GUI and Python bindings retain COLMAP's
reconstruction workflows and file formats.

COLMAP builds on top of existing works and when using specific algorithms within
COLMAP, please also cite the original authors, as specified in the source code,
and consider citing relevant third-party dependencies (most notably
ceres-solver, poselib, sift-gpu, vlfeat).

Metal on macOS
--------------

On Apple Silicon Macs, **Metal SIFT extraction and matching** run without CUDA
or a Qt/OpenGL context. Metal also accelerates supported image warping and is
selected automatically for eligible large-image undistortion, with a CPU
fallback.

Optional ONNX builds include ALIKED, LoMa and LightGlue. When GPU use is enabled
on macOS, supported model subgraphs can run through CoreML, with CPU fallbacks
for unsupported operations or models. ONNX brute-force matching runs on CPU.
See the [feature and matcher guide](doc/features.rst) for model options.

Dense stereo/PatchMatch and GPU bundle adjustment remain CUDA-backed; this fork
does not implement those stages in Metal. Sparse mapping and bundle adjustment
run on CPU in a Metal-only build.

### Build a sparse desktop CLI

Install and select Xcode with its Metal compiler available (`xcrun metal --version`).
With [Homebrew](https://brew.sh/), install the desktop dependencies:

```bash
brew install cmake ninja boost eigen openimageio metis glog ceres-solver \
  suitesparse glew sqlite3 libomp curl openssl@3
brew link --force libomp
```

Clone this fork, or run the CMake commands from an existing checkout:

```bash
git clone https://github.com/lanxinger/colmap-metal.git
cd colmap-metal
cmake -S . -B build-metal -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  -DMETAL_ENABLED=ON \
  -DCUDA_ENABLED=OFF \
  -DGUI_ENABLED=OFF \
  -DOPENGL_ENABLED=OFF \
  -DMVS_ENABLED=OFF \
  -DCGAL_ENABLED=OFF \
  -DONNX_ENABLED=OFF
cmake --build build-metal --parallel "$(sysctl -n hw.ncpu)"
build-metal/src/colmap/exe/colmap -h
```

CMake fetches PoseLib and FAISS by default. The configuration above builds a
headless SIFT workflow; enable `ONNX_ENABLED` for learned features, or
`GUI_ENABLED` and `OPENGL_ENABLED` with Qt installed for the desktop GUI.
See [installation details](doc/install.rst) for other platforms and dependencies.

To install the CLI, libraries and Metal shader resources into `install/`, run
`cmake --install build-metal`. The examples below use the build-tree executable.

The desktop hash-container backend defaults to `STD`.
`-DCOLMAP_HASH_MAP_BACKEND=BOOST` requires Boost 1.84 or newer. This choice is part
of the C++ ABI: COLMAP, PyCOLMAP and other libraries sharing COLMAP objects in one
process must use the same backend. See the [build notes](doc/install.rst).

### Reconstruct a sparse model

Use your own images or a [sample dataset](https://demuc.de/colmap/datasets/).
Create the workspace and run automatic reconstruction with dense MVS disabled:

```bash
mkdir -p path/to/workspace
build-metal/src/colmap/exe/colmap automatic_reconstructor \
  --workspace_path path/to/workspace \
  --image_path path/to/images \
  --dense 0 \
  --use_gpu 1
```

On Metal-only builds, `--use_gpu 1` enables Metal SIFT extraction and matching
while leaving CUDA-only optimization paths on CPU. High-quality automatic
reconstruction uses the Metal-compatible SIFT extractor instead of CPU-only
covariant SIFT modifiers. Keep guided matching enabled unless a dataset-specific
comparison justifies `--guided_matching 0`.

For individual `feature_extractor` and matcher commands, use
`--FeatureExtraction.use_gpu 1` and `--FeatureMatching.use_gpu 1`, respectively.
Explicit affine-shape estimation, domain-size pooling or forced covariant SIFT
extraction selects the CPU extractor. Logs identify the selected backends as
`Creating SIFT Metal GPU feature extractor` and
`Creating SIFT Metal GPU feature matcher`.

The workspace contains `database.db` and reconstructed models in `sparse/`.
Check registration coverage and model connectivity before using a reconstruction
for downstream Gaussian splat training; the automatic workflow can produce
multiple disconnected models.

### Benchmarks and tests

To measure automatic sparse reconstruction on your own Mac and images:

```bash
python3 scripts/benchmark_sift_metal.py path/to/images \
  --automatic \
  --workspace /tmp/colmap-metal-readme-benchmark \
  --colmap build-metal/src/colmap/exe/colmap
```

The benchmark recreates its scratch workspace unless `--keep-workspace` is set.
Without `--automatic`, it extracts features once with Metal and compares CPU and
Metal matching on the same descriptors. It does not compare CPU and Metal
extraction. The [dated Lund benchmark](benchmark/metal_sparse_lund.md) records
runtime and registration coverage for one dataset; it is not a current
performance or reconstruction-quality guarantee.

To build and run desktop C++ tests using the configuration above:

```bash
brew install googletest
cmake -S . -B build-metal -DTESTS_ENABLED=ON
cmake --build build-metal --parallel "$(sysctl -n hw.ncpu)"
ctest --test-dir build-metal --output-on-failure
```

The [Mac CI workflow](.github/workflows/build-mac.yml) exercises the Metal build
with both hash backends. Benchmark-script tests check reporting and selection,
not GPU correctness or speed.

ColmapSparse Swift package (iOS and macOS)
-----------------------------------------

The experimental `ColmapSparse` package provides in-process sparse reconstruction
for **iOS 18+ and macOS 13+**, with arm64 iPhone, arm64 Simulator and arm64 Mac
slices. Its Swift API wraps a versioned C interface.

With Xcode and its Metal compiler selected, Swift 6.0+, CMake 3.24+, Ninja and
Boost **1.92.0** headers available, build from the repository root:

```bash
bash scripts/build-ios.sh all
swift test --jobs 2
```

Add the repository as a **local Swift package after building**. The script builds
pinned native dependencies and generates `Artifacts/ColmapSparse.xcframework`
and platform-specific Metal resources; generated artifacts are not committed.
This package uses a separate sparse-only build from the desktop CLI.

Supply images in capture order with explicit camera calibration. The package
runs Metal SIFT extraction and matching followed by incremental mapping, then
exports the largest connected model under `images/` and `sparse/0/`, subject to
its registration threshold. It supports cooperative cancellation and bounded
resource settings. The exported dataset can be loaded by msplat for Gaussian
splat training.

See the [Apple package guide](ios/README.md) for API examples, calibration and
output requirements, build configuration, measured iPhone and app validation,
and remaining device, capture and thermal limitations. The
[iOS package CI workflow](.github/workflows/build-ios.yml) builds all three
slices, runs native and Swift tests on Mac, and compile-checks the iPhone and
Simulator test targets; it does not execute their tests on those destinations.

Python bindings
---------------

To use this fork from Python, build and install its desktop library first, then
build PyCOLMAP from this checkout. From the repository root, with a Python 3.10+
virtual environment activated:

```bash
cmake --install build-metal
colmap_DIR="$PWD/install/share/colmap" python -m pip install .
```

See the [PyCOLMAP guide](python/README.md) for examples and the
[incremental build helper](python/incremental_build.sh) for development builds.
The upstream PyPI wheels below are separate distributions of upstream COLMAP.

Upstream downloads
------------------

Build this checkout for the fork's Metal and Apple package changes. The following
links provide **upstream COLMAP** distributions and resources:

* [Windows binaries and releases](https://github.com/colmap/colmap/releases).
* [Linux/Unix/BSD packages](https://repology.org/metapackage/colmap/versions).
* [Docker images](https://hub.docker.com/r/colmap/colmap).
* [Conda packages](https://anaconda.org/conda-forge/colmap), installed with
  `conda install colmap`.
* [PyCOLMAP wheels](https://pypi.org/project/pycolmap/) and
  [CUDA 12 wheels](https://pypi.org/project/pycolmap-cuda12/).

Documentation and support
-------------------------

The [documentation in this checkout](doc/index.rst) describes its CLI, features,
file formats and build options. The [upstream documentation](https://colmap.github.io/)
provides general tutorials; its latest version can differ from this fork.
Instructions for building the documentation are in the
[installation guide](doc/install.rst).

For fork-specific changes, use the
[COLMAP-Metal repository](https://github.com/lanxinger/colmap-metal) and its
[pull requests](https://github.com/lanxinger/colmap-metal/pulls).
For general COLMAP questions, use
[upstream Discussions](https://github.com/colmap/colmap/discussions).
Report bugs reproducible in upstream COLMAP to its
[issue tracker](https://github.com/colmap/colmap/issues).

The Metal SIFT implementation is based on
[SIFTMetal](https://github.com/lukevanin/SIFTMetal) by Luke Van In, with the
Swift host code rewritten in Objective-C++ for CMake integration. See the
[bundled implementation notes](src/thirdparty/SiftMetal/README.md) and
[Apple package dependency notices](ios/THIRD_PARTY_NOTICES.md) for attribution.

Acknowledgments
---------------

COLMAP was originally written by [Johannes Schönberger](https://demuc.de/) with
funding provided by his PhD advisors Jan-Michael Frahm and Marc Pollefeys.
The team of core project maintainers currently includes
[Johannes Schönberger](https://github.com/ahojnnes),
[Paul-Edouard Sarlin](https://github.com/sarlinpe),
[Shaohui Liu](https://github.com/B1ueber2y), and
[Linfei Pan](https://lpanaf.github.io/).

The Python bindings in PyCOLMAP were originally added by
[Mihai Dusmanu](https://github.com/mihaidusmanu),
[Philipp Lindenberger](https://github.com/Phil26AT), and
[Paul-Edouard Sarlin](https://github.com/sarlinpe).

The project has also benefitted from countless community contributions, including
bug fixes, improvements, new features, third-party tooling, and community
support (special credits to [Torsten Sattler](https://tsattler.github.io)).

Citation
--------

If you use this project for your research, please cite:

    @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }

    @inproceedings{schoenberger2016mvs,
        author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
        title={Pixelwise View Selection for Unstructured Multi-View Stereo},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2016},
    }

If you use the global SfM pipeline (GLOMAP), please cite:

    @inproceedings{pan2024glomap,
        author={Pan, Linfei and Barath, Daniel and Pollefeys, Marc and Sch\"{o}nberger, Johannes Lutz},
        title={{Global Structure-from-Motion Revisited}},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2024},
    }

If you use the image retrieval / vocabulary tree engine, please cite:

    @inproceedings{schoenberger2016vote,
        author={Sch\"{o}nberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc},
        title={A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval},
        booktitle={Asian Conference on Computer Vision (ACCV)},
        year={2016},
    }

Contribution
------------

Contributions (bug reports, bug fixes, improvements, etc.) are very welcome and
should be submitted in the form of new issues and/or pull requests on GitHub.

License
-------

The COLMAP library is licensed under the new BSD license. Note that this text
refers only to the license for COLMAP itself, independent of its thirdparty
dependencies, which are separately licensed. Building COLMAP with these
dependencies may affect the resulting COLMAP license.

    Copyright (c), ETH Zurich and UNC Chapel Hill.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.

        * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
          its contributors may be used to endorse or promote products derived
          from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
