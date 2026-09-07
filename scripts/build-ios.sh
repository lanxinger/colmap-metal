#!/usr/bin/env bash
set -euo pipefail

# Build an isolated sparse COLMAP slice, or all supported Apple slices and an
# XCFramework. The published C header is the entire binary interface.
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
requested_sdk="${1:-all}"
build_root="${COLMAP_IOS_BUILD_ROOT:-$repo_dir/build-ios}"
jobs="${COLMAP_IOS_BUILD_JOBS:-6}"
case "$requested_sdk" in
  all) sdks=(macosx iphoneos iphonesimulator) ;;
  iphoneos|iphonesimulator|macosx) sdks=("$requested_sdk") ;;
  *) echo "Usage: $0 [all|iphoneos|iphonesimulator|macosx]" >&2; exit 2 ;;
esac

export COLMAP_IOS_BUILD_ROOT="$build_root"
mkdir -p "$build_root"
for sdk in "${sdks[@]}"; do
  "$repo_dir/scripts/build-ios-dependencies.sh" "$sdk"
  if [[ "$sdk" == macosx ]]; then
    system_name=Darwin
    deployment_target="${COLMAP_MACOS_DEPLOYMENT_TARGET:-13.0}"
    build_tests=ON
  else
    system_name=iOS
    deployment_target="${COLMAP_IOS_DEPLOYMENT_TARGET:-18.0}"
    build_tests=OFF
  fi
  sdk_path="$(xcrun --sdk "$sdk" --show-sdk-path)"
  dependency_prefix="$build_root/deps/$sdk-arm64"
  common_prefix="$build_root/deps/common"
  slice_build="$build_root/core/$sdk-arm64"
  slice_stage="$build_root/slices/$sdk-arm64"
  cmake -S "$repo_dir/ios" -B "$slice_build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME="$system_name" \
    -DCMAKE_OSX_SYSROOT="$sdk_path" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
    -DCMAKE_PREFIX_PATH="$dependency_prefix;$common_prefix" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -DEigen3_DIR="$common_prefix/share/eigen3/cmake" \
    -Dglog_DIR="$dependency_prefix/lib/cmake/glog" \
    -DCeres_DIR="$dependency_prefix/lib/cmake/Ceres" \
    -DPoseLib_DIR="$dependency_prefix/lib/cmake/PoseLib" \
    -DCOLMAP_IOS_BOOST_INCLUDE="${COLMAP_IOS_BOOST_INCLUDE:-/opt/homebrew/include}" \
    -DCOLMAP_IOS_BUILD_TESTS="$build_tests" \
    -DCMAKE_INSTALL_PREFIX="$slice_stage"
  cmake --build "$slice_build" --parallel "$jobs"
  cmake --install "$slice_build"
  archives=()
  while IFS= read -r archive; do
    [[ -z "$archive" ]] || archives+=("$archive")
  done < "$slice_build/static-archives.txt"
  xcrun libtool -static -o "$slice_stage/lib/libColmapSparseNative.a" "${archives[@]}"
  # Xcode copies every static XCFramework's headers into one include directory.
  # A module-specific subdirectory prevents collisions with other packages'
  # module.modulemap files while preserving the public C install layout.
  package_headers="$slice_stage/package-include/ColmapSparseNative"
  mkdir -p "$package_headers"
  cp "$slice_stage/include/ColmapSparse.h" "$package_headers/ColmapSparse.h"
  cat > "$package_headers/module.modulemap" <<'EOF'
module ColmapSparseNative {
  header "ColmapSparse.h"
  export *
  link "c++"
  link "sqlite3"
  link framework "Foundation"
  link framework "CoreGraphics"
  link framework "ImageIO"
  link framework "Accelerate"
  link framework "Metal"
}
EOF
  mkdir -p "$build_root/resources/$sdk"
  cp "$slice_build/sift.metallib" "$build_root/resources/$sdk/sift.metallib"
  xcrun lipo -info "$slice_stage/lib/libColmapSparseNative.a"
done

if [[ "$requested_sdk" == all ]]; then
  # xcodebuild requires a new output path. Preserve any previously validated
  # package until the newly built candidate has been created successfully.
  candidate_dir="$(mktemp -d "$build_root/xcframework-candidate.XXXXXX")"
  candidate="$candidate_dir/ColmapSparse.xcframework"
  xcodebuild -create-xcframework \
    -library "$build_root/slices/iphoneos-arm64/lib/libColmapSparseNative.a" \
    -headers "$build_root/slices/iphoneos-arm64/package-include" \
    -library "$build_root/slices/iphonesimulator-arm64/lib/libColmapSparseNative.a" \
    -headers "$build_root/slices/iphonesimulator-arm64/package-include" \
    -library "$build_root/slices/macosx-arm64/lib/libColmapSparseNative.a" \
    -headers "$build_root/slices/macosx-arm64/package-include" \
    -output "$candidate"
  cp "$repo_dir/ios/THIRD_PARTY_NOTICES.md" "$candidate/THIRD_PARTY_NOTICES.md"
  ditto "$repo_dir/ios/Licenses" "$candidate/Licenses"
  {
    echo "COLMAP_SOURCE_REVISION=$(git -C "$repo_dir" rev-parse HEAD)"
    echo "BUILT_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "HEADER_LAYOUT=Headers/ColmapSparseNative"
    xcodebuild -version
    for sdk in "${sdks[@]}"; do
      cat "$build_root/deps/$sdk-arm64/dependency-versions.txt"
      shasum -a 256 "$build_root/slices/$sdk-arm64/lib/libColmapSparseNative.a" \
        "$build_root/resources/$sdk/sift.metallib"
    done
    echo "Source file SHA-256 values (including uncommitted package sources):"
    (
      cd "$repo_dir"
      git ls-files -z --cached --others --exclude-standard -- ios \
        scripts/build-ios.sh scripts/build-ios-dependencies.sh \
        src/thirdparty/SiftMetal src/colmap | \
        xargs -0 shasum -a 256
    )
  } > "$candidate/BUILD_PROVENANCE.txt"
  if [[ -e "$build_root/ColmapSparse.xcframework" ]]; then
    previous="$build_root/ColmapSparse.previous.$(date -u +%Y%m%dT%H%M%SZ).xcframework"
    mv "$build_root/ColmapSparse.xcframework" "$previous"
  fi
  mv "$candidate" "$build_root/ColmapSparse.xcframework"
  rmdir "$candidate_dir"
  # Package.swift consumes generated binaries and platform resources at these
  # repository-local paths, even when build intermediates live elsewhere.
  artifacts_dir="$repo_dir/Artifacts"
  mkdir -p "$artifacts_dir"
  publish_dir="$(mktemp -d "$artifacts_dir/.colmap-sparse.XXXXXX")"
  ditto "$build_root/ColmapSparse.xcframework" "$publish_dir/ColmapSparse.xcframework"
  if [[ -e "$artifacts_dir/ColmapSparse.xcframework" ]]; then
    previous="$build_root/ColmapSparse.published.previous.$(date -u +%Y%m%dT%H%M%SZ).xcframework"
    mv "$artifacts_dir/ColmapSparse.xcframework" "$previous"
  fi
  mv "$publish_dir/ColmapSparse.xcframework" "$artifacts_dir/ColmapSparse.xcframework"
  rmdir "$publish_dir"
  for sdk in "${sdks[@]}"; do
    resource_dir="$repo_dir/ios/Sources/ColmapSparse/Resources/$sdk"
    mkdir -p "$resource_dir"
    cp "$build_root/resources/$sdk/sift.metallib" "$resource_dir/sift.metallib"
  done
  echo "XCFramework ready: $build_root/ColmapSparse.xcframework"
  echo "Swift package binary: $artifacts_dir/ColmapSparse.xcframework"
  echo "Platform Metal resources: $repo_dir/ios/Sources/ColmapSparse/Resources"
fi
