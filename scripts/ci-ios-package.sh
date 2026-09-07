#!/usr/bin/env bash
set -euo pipefail

# CI entry point: build from pinned sources, then exercise the packaged native
# library, Swift API, and iOS test-bundle links. This publishes the same ignored
# Artifacts/ and Metal resources as scripts/build-ios.sh all.
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
ci_root="${COLMAP_IOS_CI_ROOT:-${RUNNER_TEMP:-$repo_dir/build-ios}/colmap-ios-ci}"
export COLMAP_IOS_BUILD_ROOT="$ci_root/native"
export COLMAP_IOS_DOWNLOAD_CACHE="${COLMAP_IOS_DOWNLOAD_CACHE:-$ci_root/downloads}"
export COLMAP_IOS_BUILD_JOBS="${COLMAP_IOS_BUILD_JOBS:-2}"
log_dir="$ci_root/logs"
mkdir -p "$log_dir" "$COLMAP_IOS_DOWNLOAD_CACHE"
cd "$repo_dir"

run_logged() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "$log_dir/$name.log"
}

{
  git rev-parse HEAD
  uname -m
  xcodebuild -version
  swift --version
  cmake --version
  ninja --version
  for sdk in macosx iphoneos iphonesimulator; do
    xcrun --sdk "$sdk" --show-sdk-version
    xcrun --sdk "$sdk" metal --version
  done
} > "$log_dir/toolchain.txt" 2>&1
test "$(uname -m)" = arm64

# Official release metadata:
# https://archives.boost.io/release/1.92.0/source/boost_1_92_0.tar.gz.json
boost_archive="$COLMAP_IOS_DOWNLOAD_CACHE/boost_1_92_0.tar.gz"
boost_sha256=c4a3b310ddd2472416e091067166b0713be97c63f38c212c484ada022fd296ce
if [[ ! -f "$boost_archive" ]]; then
  curl --fail --location --retry 3 \
    https://archives.boost.io/release/1.92.0/source/boost_1_92_0.tar.gz \
    --output "$boost_archive.partial"
  mv "$boost_archive.partial" "$boost_archive"
fi
actual_sha256="$(shasum -a 256 "$boost_archive" | awk '{print $1}')"
if [[ "$actual_sha256" != "$boost_sha256" ]]; then
  echo "Boost 1.92.0 archive checksum mismatch: $actual_sha256" >&2
  exit 1
fi
mkdir -p "$ci_root/boost"
tar -xzf "$boost_archive" -C "$ci_root/boost" boost_1_92_0/boost
export COLMAP_IOS_BOOST_INCLUDE="$ci_root/boost/boost_1_92_0"
shasum -a 256 "$boost_archive" "$COLMAP_IOS_BOOST_INCLUDE/boost/version.hpp" \
  > "$log_dir/boost-provenance.txt"

# The three SDK builds intentionally share one build root and run sequentially.
run_logged native-build bash scripts/build-ios.sh all
native_host="$COLMAP_IOS_BUILD_ROOT/core/macosx-arm64"
run_logged native-bitmap ctest --test-dir "$native_host" \
  --output-on-failure --no-tests=error -R '^bitmap_apple_test$'
run_logged native-sparse "$native_host/sparse_smoke" --test "$native_host/sift.metallib"
run_logged swift-host swift test --package-path "$repo_dir" \
  --scratch-path "$ci_root/swift-host" --configuration release \
  --jobs "$COLMAP_IOS_BUILD_JOBS"

# Generic destinations require no simulator runtime or signing credentials.
# Building the tests forces a final link against each native slice; these are
# compile/link/resource checks, not iPhone or Simulator runtime tests.
run_logged swift-schemes xcodebuild -list -json
for sdk in iphoneos iphonesimulator; do
  destination='generic/platform=iOS'
  [[ "$sdk" != iphonesimulator ]] || destination='generic/platform=iOS Simulator'
  derived_data="$ci_root/xcode-$sdk"
  run_logged "swift-$sdk" xcodebuild build-for-testing \
    -scheme ColmapSparse -configuration Release -sdk "$sdk" \
    -destination "$destination" -derivedDataPath "$derived_data" \
    -resultBundlePath "$log_dir/swift-$sdk.xcresult" \
    -jobs "$COLMAP_IOS_BUILD_JOBS" \
    ARCHS=arm64 ONLY_ACTIVE_ARCH=YES ENABLE_TESTABILITY=YES \
    CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO
  python3 - "$repo_dir" "$derived_data/Build/Products" "$sdk" \
    > "$log_dir/resources-$sdk.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

repository, products = map(Path, sys.argv[1:3])
sdk = sys.argv[3]
source = repository / f"ios/Sources/ColmapSparse/Resources/{sdk}/sift.metallib"
expected = hashlib.sha256(source.read_bytes()).hexdigest()
resources = [
    path for path in products.rglob("sift.metallib")
    if path.parent.name == sdk
    and "ColmapSparse_ColmapSparse.bundle" in path.parts
]
if not resources:
    raise SystemExit(f"The {sdk} build did not copy ColmapSparse's Metal resource bundle")
for path in resources:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"The {sdk} bundle contains a different Metal library: {path}")
print(json.dumps({
    "sdk": sdk,
    "sift_metallib_sha256": expected,
    "copied_resources": [str(path.relative_to(products)) for path in resources],
}, indent=2))
PY
done
