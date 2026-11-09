#!/usr/bin/env bash
set -euo pipefail

hash_backend="${1:?Expected hash backend: STD or BOOST}"
mvs_enabled="${2:?Expected MVS setting: ON or OFF}"
case "$hash_backend:$mvs_enabled" in
  STD:OFF|BOOST:OFF|STD:ON) ;;
  *) echo "Unsupported CI configuration: $hash_backend MVS=$mvs_enabled" >&2; exit 1 ;;
esac

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
# Isolate generated headers, dependency builds, and installed packages so each
# configuration checks its own hash-map ABI and MVS feature gates.
build_dir="$repo_dir/build-mac/$hash_backend-mvs-$mvs_enabled"
install_prefix="$build_dir/install"

# Exercise the default in STD builds and the explicit BOOST option.
configured_backend="$hash_backend"
if [[ "$hash_backend" == STD ]]; then
  configured_backend=""
fi

cmake --version
cmake -S "$repo_dir" -B "$build_dir" \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOLMAP_HASH_MAP_BACKEND="$configured_backend" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCCACHE_ENABLED=ON \
  -DCGAL_ENABLED=OFF \
  -DCUDA_ENABLED=OFF \
  -DDOWNLOAD_ENABLED=OFF \
  -DFETCH_ONNX=OFF \
  -DGUI_ENABLED=OFF \
  -DMETAL_ENABLED=ON \
  -DMVS_ENABLED="$mvs_enabled" \
  -DONNX_ENABLED=OFF \
  -DOPENGL_ENABLED=OFF \
  -DTESTS_ENABLED=ON \
  -DWERROR_ENABLED=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED="${FETCHCONTENT_FULLY_DISCONNECTED:-OFF}"
cmake --build "$build_dir"
ctest --test-dir "$build_dir" --output-on-failure --no-tests=error

cmake --install "$build_dir" --prefix "$install_prefix"
for requested in STD BOOST; do
  consumer_dir="$build_dir/consumer-$requested"
  cmake -S "$repo_dir/cmake/tests/hash_map_backend" -B "$consumer_dir" \
    -GNinja \
    -DCMAKE_PREFIX_PATH="$install_prefix" \
    -DCOLMAP_HASH_MAP_BACKEND="$requested" \
    -DEXPECTED_HASH_MAP_BACKEND="$hash_backend"
  cmake --build "$consumer_dir"
  "$consumer_dir/hash_map_backend_test"
done
