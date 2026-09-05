cmake_minimum_required(VERSION 3.10)
project(colmap_hash_map_backend_test LANGUAGES C CXX)

find_package(colmap REQUIRED)
if(NOT COLMAP_HASH_MAP_BACKEND STREQUAL EXPECTED_HASH_MAP_BACKEND)
    message(FATAL_ERROR "Installed backend was not preserved: ${COLMAP_HASH_MAP_BACKEND}")
endif()

add_executable(hash_map_backend_test main.cc)
target_compile_features(hash_map_backend_test PRIVATE cxx_std_17)
target_link_libraries(hash_map_backend_test PRIVATE colmap::colmap)
# Keep the selected installation ahead of other COLMAP headers on the host.
set_target_properties(hash_map_backend_test PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)
string(TOLOWER "${EXPECTED_HASH_MAP_BACKEND}" EXPECTED_HASH_MAP_BACKEND)
target_compile_definitions(hash_map_backend_test PRIVATE
    EXPECTED_HASH_MAP_BACKEND="${EXPECTED_HASH_MAP_BACKEND}")
