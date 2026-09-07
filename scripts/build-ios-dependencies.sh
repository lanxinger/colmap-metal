#!/usr/bin/env bash
set -euo pipefail

# Build architecture-specific dependencies without using Homebrew binaries.
# Boost is header-only in the sparse target; its pinned headers are supplied
# separately through COLMAP_IOS_BOOST_INCLUDE when configuring ios/.
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
sdk="${1:-iphoneos}"
build_root="${COLMAP_IOS_BUILD_ROOT:-$repo_dir/build-ios}"
jobs="${COLMAP_IOS_BUILD_JOBS:-6}"
case "$sdk" in
  iphoneos|iphonesimulator) system_name=iOS; deployment_target="${COLMAP_IOS_DEPLOYMENT_TARGET:-18.0}" ;;
  macosx) system_name=Darwin; deployment_target="${COLMAP_MACOS_DEPLOYMENT_TARGET:-13.0}" ;;
  *) echo "Usage: $0 [iphoneos|iphonesimulator|macosx]" >&2; exit 2 ;;
esac

downloads="${COLMAP_IOS_DOWNLOAD_CACHE:-$build_root/downloads}"
sources="$build_root/sources"
prefix="$build_root/deps/$sdk-arm64"
common_prefix="$build_root/deps/common"
mkdir -p "$downloads" "$sources" "$prefix" "$common_prefix"

fetch_archive() {
  local name="$1" digest="$2" url="$3" cached="${4:-}"
  local archive="$downloads/$name"
  if [[ ! -f "$archive" ]]; then
    if [[ -n "$cached" && -f "$cached" ]]; then
      cp "$cached" "$archive"
    else
      curl --fail --location --retry 2 "$url" --output "$archive.partial"
      mv "$archive.partial" "$archive"
    fi
  fi
  local actual
  actual="$(shasum -a 256 "$archive" | awk '{print $1}')"
  if [[ "$actual" != "$digest" ]]; then
    echo "Checksum mismatch for $archive: expected $digest, got $actual" >&2
    exit 1
  fi
}

fetch_archive eigen-3.4.0.tar.gz \
  8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72 \
  https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
fetch_archive glog-0.7.1.tar.gz \
  00e4a87e87b7e7612f519a41e491f16623b12423620006f59f5688bfd8d13b08 \
  https://github.com/google/glog/archive/refs/tags/v0.7.1.tar.gz
fetch_archive ceres-2.2.0.tar.gz \
  12efacfadbfdc1bbfa203c236e96f4d3c210bed96994288b3ff0c8e7c6f350d4 \
  https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz
fetch_archive poselib-fa7280fee27f97aff31ae7f98bab7f583fac7d08.zip \
  5408d4ae8ce367cb2f076bc6c5f0f6f78abd3573d2c015304b04e46f23455f5b \
  https://github.com/PoseLib/PoseLib/archive/fa7280fee27f97aff31ae7f98bab7f583fac7d08.zip \
  "$repo_dir/build-metal-ba-benchmark/_deps/poselib-subbuild/poselib-populate-prefix/src/fa7280fee27f97aff31ae7f98bab7f583fac7d08.zip"

for dependency in eigen-3.4.0 glog-0.7.1 ceres-solver-2.2.0; do
  if [[ ! -d "$sources/$dependency" ]]; then
    archive="$dependency.tar.gz"
    [[ "$dependency" != ceres-solver-2.2.0 ]] || archive=ceres-2.2.0.tar.gz
    tar -xzf "$downloads/$archive" -C "$sources"
  fi
done
poselib_source="$sources/PoseLib-fa7280fee27f97aff31ae7f98bab7f583fac7d08"
if [[ ! -d "$poselib_source" ]]; then
  (cd "$sources" && cmake -E tar xf "$downloads/poselib-fa7280fee27f97aff31ae7f98bab7f583fac7d08.zip")
fi

if [[ ! -f "$common_prefix/share/eigen3/cmake/Eigen3Config.cmake" ]]; then
  cmake -S "$sources/eigen-3.4.0" -B "$build_root/dependency-build/eigen" -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$common_prefix" -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF
  cmake --install "$build_root/dependency-build/eigen"
fi

sdk_path="$(xcrun --sdk "$sdk" --show-sdk-path)"
platform_args=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_SYSTEM_NAME="$system_name"
  -DCMAKE_OSX_SYSROOT="$sdk_path"
  -DCMAKE_OSX_ARCHITECTURES=arm64
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target"
  -DCMAKE_INSTALL_PREFIX="$prefix"
  -DCMAKE_PREFIX_PATH="$prefix;$common_prefix"
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_CXX_FLAGS=-DEIGEN_MPL2_ONLY
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_TESTING=OFF
)

cmake -S "$sources/glog-0.7.1" -B "$build_root/dependency-build/$sdk-glog" \
  "${platform_args[@]}" -DWITH_GFLAGS=OFF -DWITH_UNWIND=OFF
cmake --build "$build_root/dependency-build/$sdk-glog" --parallel "$jobs"
cmake --install "$build_root/dependency-build/$sdk-glog"

# Ceres 2.2's IOS branch predates native CMake iOS support and forcibly enables
# miniglog. COLMAP subclasses real glog, so configure Ceres as a subproject with
# that legacy variable disabled. CMAKE_SYSTEM_NAME, SDK, architecture and minimum
# OS still describe the actual target and are never changed here.
ceres_wrapper="$build_root/dependency-build/ceres-wrapper"
mkdir -p "$ceres_wrapper"
cat > "$ceres_wrapper/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.24)
project(ColmapCeresDependency LANGUAGES C CXX)
set(IOS OFF)
add_subdirectory("$sources/ceres-solver-2.2.0" ceres)
EOF
cmake -S "$ceres_wrapper" -B "$build_root/dependency-build/$sdk-ceres" \
  "${platform_args[@]}" \
  -DEigen3_DIR="$common_prefix/share/eigen3/cmake" \
  -Dglog_DIR="$prefix/lib/cmake/glog" \
  -DMINIGLOG=OFF -DGFLAGS=OFF -DSUITESPARSE=OFF -DEIGENMETIS=OFF \
  -DEIGENSPARSE=ON -DACCELERATESPARSE=ON -DLAPACK=OFF -DUSE_CUDA=OFF \
  -DSCHUR_SPECIALIZATIONS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
  -DBUILD_DOCUMENTATION=OFF -DPROVIDE_UNINSTALL_TARGET=OFF
cmake --build "$build_root/dependency-build/$sdk-ceres" --parallel "$jobs"
cmake --install "$build_root/dependency-build/$sdk-ceres"

cmake -S "$poselib_source" -B "$build_root/dependency-build/$sdk-poselib" \
  "${platform_args[@]}" -DEigen3_DIR="$common_prefix/share/eigen3/cmake" \
  -DMARCH_NATIVE=OFF -DBUILD_TESTS=OFF -DWITH_BENCHMARK=OFF -DPYTHON_PACKAGE=OFF
cmake --build "$build_root/dependency-build/$sdk-poselib" --parallel "$jobs"
cmake --install "$build_root/dependency-build/$sdk-poselib"

cat > "$prefix/dependency-versions.txt" <<EOF
SDK=$sdk
SDK_PATH=$sdk_path
ARCH=arm64
DEPLOYMENT_TARGET=$deployment_target
Eigen=3.4.0
EigenLicenseGuard=EIGEN_MPL2_ONLY
glog=0.7.1 (static, gflags/unwind disabled)
Ceres=2.2.0 (static, EigenSparse/AccelerateSparse, no SuiteSparse/Metis/LAPACK/CUDA)
PoseLib=fa7280fee27f97aff31ae7f98bab7f583fac7d08
EOF
echo "Dependencies ready: $prefix"
