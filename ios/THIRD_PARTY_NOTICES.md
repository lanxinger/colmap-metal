# ColmapSparse dependency notices

Distribute this file and the accompanying `Licenses/` directory with the native
archive and Metal shader resources. These notices describe the explicit mobile
target in `ios/CMakeLists.txt`; the desktop COLMAP dependency graph is different.

| Component | Pinned source | Included notice |
| --- | --- | --- |
| COLMAP | The source revision used to build this package; recorded in generated build provenance | `Licenses/COLMAP-BSD.txt` |
| SiftMetal | Locally modified port of Luke Van In's SIFTMetal; [pinned upstream license source](https://github.com/lukevanin/SIFTMetal/blob/4abdeb8cd92b3bc3afe72643360cc45ea0c50cc1/LICENSE) | `Licenses/SiftMetal-MIT.txt` |
| Ceres Solver | [2.2.0](https://github.com/ceres-solver/ceres-solver/tree/2.2.0) | `Licenses/Ceres-LICENSE.txt`, including its bundled dependency notices |
| Eigen | [3.4.0, unmodified source archive](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz) | `Licenses/Eigen-COPYING-MPL2.txt`, `Eigen-COPYING-BSD.txt`, `Eigen-COPYING-APACHE.txt`, `Eigen-COPYING-MINPACK.txt` |
| glog | [0.7.1](https://github.com/google/glog/tree/v0.7.1) | `Licenses/glog-COPYING.txt` |
| PoseLib | [fa7280fee27f97aff31ae7f98bab7f583fac7d08](https://github.com/PoseLib/PoseLib/tree/fa7280fee27f97aff31ae7f98bab7f583fac7d08) | `Licenses/PoseLib-LICENSE.txt` |
| Boost headers | [1.92.0](https://github.com/boostorg/boost/tree/boost-1.92.0) | `Licenses/Boost-LICENSE-1.0.txt` |

Eigen source is available at the link above; the build uses the original archive
without source modifications. `EIGEN_MPL2_ONLY` rejects non-MPL-compatible Eigen
modules during compilation. Per-file copyright notices remain in that source.

SQLite and Apple's Foundation, CoreGraphics, ImageIO, Accelerate and Metal
frameworks are linked from the target operating system. They are not copied into
the XCFramework. SiftGPU, OpenGL, VLFeat, OpenImageIO, FAISS, ONNX Runtime,
SuiteSparse, Metis, CUDA, MVS and meshing implementations are excluded from this
mobile build. Ceres' empty disabled-backend translation units may retain those
names in archive member listings; they do not contain those backends.

The dependency archive URLs and SHA-256 checks are in
`scripts/build-ios-dependencies.sh`. Shader modifications preserve their original
creator headers. The upstream SiftMetal revision cited in its provenance file
pins the verified license source; the original local import did not record an
exact upstream source revision.
