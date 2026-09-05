#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/version.h"

#include <string>

#if defined(COLMAP_HASH_STD) == defined(COLMAP_HASH_BOOST)
#error "The installed target must select exactly one hash backend"
#endif

int main() {
  // The header, linked library, and installed package must agree on the ABI,
  // even when the consumer explicitly requested a different backend.
  if (std::string(colmap::kHashMapBackend) != EXPECTED_HASH_MAP_BACKEND ||
      colmap::GetBuildInfo().find(std::string(EXPECTED_HASH_MAP_BACKEND) +
                                  " hash maps") == std::string::npos) {
    return 1;
  }
  const colmap::Reconstruction reconstruction;
  return reconstruction.NumImages() == 0 ? 0 : 1;
}
