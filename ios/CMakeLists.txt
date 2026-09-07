cmake_minimum_required(VERSION 3.24)
project(ColmapSparse LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "ColmapSparse requires an Apple SDK")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(COLMAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(COLMAP_IOS_BOOST_INCLUDE "/opt/homebrew/include" CACHE PATH
    "Directory containing pinned Boost 1.92.0 headers; no Boost binaries are linked")
if(NOT EXISTS "${COLMAP_IOS_BOOST_INCLUDE}/boost/version.hpp")
  message(FATAL_ERROR "Set COLMAP_IOS_BOOST_INCLUDE to Boost 1.92.0 headers")
endif()
file(STRINGS "${COLMAP_IOS_BOOST_INCLUDE}/boost/version.hpp" boost_version
     REGEX "^#define BOOST_VERSION ")
if(NOT boost_version STREQUAL "#define BOOST_VERSION 109200")
  message(FATAL_ERROR "Expected pinned Boost 1.92.0 headers, found: ${boost_version}")
endif()
# Expose only Boost's header subtree. Adding a Homebrew include root could
# silently override the pinned Eigen, Ceres, or glog dependency headers.
set(boost_include "${CMAKE_CURRENT_BINARY_DIR}/boost-include")
file(MAKE_DIRECTORY "${boost_include}")
file(CREATE_LINK "${COLMAP_IOS_BOOST_INCLUDE}/boost"
     "${boost_include}/boost" SYMBOLIC)

find_package(Eigen3 3.4.0 EXACT CONFIG REQUIRED)
find_package(glog 0.7.1 EXACT CONFIG REQUIRED)
find_package(Ceres 2.2.0 EXACT CONFIG REQUIRED)
find_package(PoseLib CONFIG REQUIRED)
find_library(SQLITE3_LIBRARY sqlite3 PATHS "${CMAKE_OSX_SYSROOT}/usr/lib"
             NO_DEFAULT_PATH REQUIRED)

# Keep platform-independent COLMAP implementation files explicit. In particular,
# no CLI/factories, retrieval/FAISS, global mapper/CHOLMOD, image/OpenImageIO,
# graph partitioning/Metis, OpenGL/SiftGPU, MVS, or neural inference enters this
# target. The C bridge drives the standalone Metal frontend directly.
set(colmap_sources
  controllers/incremental_pipeline.cc
  estimators/alignment.cc
  estimators/bundle_adjustment.cc
  estimators/bundle_adjustment_ceres.cc
  estimators/generalized_pose.cc
  estimators/fundamental_matrix_degensac.cc
  estimators/pose.cc
  estimators/triangulation.cc
  estimators/two_view_geometry.cc
  estimators/solvers/absolute_pose.cc
  estimators/solvers/affine_transform.cc
  estimators/solvers/essential_matrix.cc
  estimators/solvers/fundamental_matrix.cc
  estimators/solvers/generalized_absolute_pose.cc
  estimators/solvers/generalized_relative_pose.cc
  estimators/solvers/homography_matrix.cc
  estimators/solvers/poselib_utils.cc
  estimators/solvers/relpose_one_sided_focal.cc
  estimators/solvers/relpose_shared_focal.cc
  estimators/solvers/similarity_transform.cc
  feature/types.cc
  feature/utils.cc
  geometry/bbox.cc
  geometry/essential_matrix.cc
  geometry/gps.cc
  geometry/homography_matrix.cc
  geometry/normalization.cc
  geometry/pose.cc
  geometry/pose_prior.cc
  geometry/rigid3.cc
  geometry/sim3.cc
  geometry/triangulation.cc
  math/math.cc
  math/polynomial.cc
  math/random.cc
  math/spanning_tree.cc
  optim/combination_sampler.cc
  optim/progressive_sampler.cc
  optim/random_sampler.cc
  optim/sprt.cc
  optim/support_measurement.cc
  scene/camera.cc
  scene/correspondence_graph.cc
  scene/database.cc
  scene/database_sqlite.cc
  scene/database_cache.cc
  scene/frame.cc
  scene/image.cc
  scene/point2d.cc
  scene/point3d.cc
  scene/projection.cc
  scene/reconstruction.cc
  scene/reconstruction_io.cc
  scene/reconstruction_io_binary.cc
  scene/reconstruction_io_text.cc
  scene/reconstruction_io_utils.cc
  scene/reconstruction_manager.cc
  scene/reconstruction_pruning.cc
  scene/rig.cc
  scene/track.cc
  scene/two_view_geometry.cc
  scene/visibility_pyramid.cc
  sensor/database.cc
  sensor/models.cc
  sensor/rig.cc
  sensor/specs.cc
  sfm/incremental_mapper.cc
  sfm/incremental_mapper_impl.cc
  sfm/incremental_triangulator.cc
  sfm/observation_manager.cc
  util/base_controller.cc
  util/cancellation.cc
  util/endian.cc
  util/file.cc
  util/logging.cc
  util/misc.cc
  util/ply.cc
  util/string.cc
  util/threading.cc
  util/timer.cc
)
list(TRANSFORM colmap_sources PREPEND "${COLMAP_ROOT}/src/colmap/")

file(STRINGS "${COLMAP_ROOT}/CMakeLists.txt" colmap_version_line
     REGEX "^set\\(COLMAP_VERSION ")
string(REGEX REPLACE "^set\\(COLMAP_VERSION \"([^\"]+)\"\\).*" "\\1"
       COLMAP_VERSION "${colmap_version_line}")
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${COLMAP_VERSION}")
set(COLMAP_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(COLMAP_VERSION_MINOR "${CMAKE_MATCH_2}")
set(COLMAP_VERSION_PATCH "${CMAKE_MATCH_3}")
execute_process(COMMAND git -C "${COLMAP_ROOT}" rev-parse HEAD
                OUTPUT_VARIABLE GIT_COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C "${COLMAP_ROOT}" show -s --format=%cs HEAD
                OUTPUT_VARIABLE GIT_COMMIT_DATE OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file("${COLMAP_ROOT}/src/colmap/util/version.cc.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/version.cc")

add_library(colmap_sparse STATIC
  ${colmap_sources}
  "${CMAKE_CURRENT_BINARY_DIR}/generated/version.cc"
  "${COLMAP_ROOT}/src/thirdparty/SiftMetal/SiftMetal.mm"
  src/bitmap_apple.mm
  src/colmap_sparse.mm
)
set_target_properties(colmap_sparse PROPERTIES
  OUTPUT_NAME ColmapSparse
  CXX_VISIBILITY_PRESET hidden
  OBJCXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(colmap_sparse
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${COLMAP_ROOT}/src"
          "${COLMAP_ROOT}/src/thirdparty/SiftMetal"
          "${COLMAP_ROOT}/src/thirdparty/SiftMetal/include"
)
target_include_directories(colmap_sparse SYSTEM PRIVATE "${boost_include}")
target_compile_definitions(colmap_sparse PRIVATE
  EIGEN_MPL2_ONLY
  COLMAP_HASH_STD
  COLMAP_METAL_ENABLED
  COLMAP_GPU_ENABLED
  COLMAP_IOS_SPARSE_BUILD
  GLOG_VERSION_MAJOR=0
  GLOG_VERSION_MINOR=7
)
target_compile_options(colmap_sparse PRIVATE
  "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
)
target_link_libraries(colmap_sparse PRIVATE
  Eigen3::Eigen
  glog::glog
  Ceres::ceres
  PoseLib::PoseLib
  "${SQLITE3_LIBRARY}"
  "-framework Foundation"
  "-framework CoreGraphics"
  "-framework ImageIO"
  "-framework Accelerate"
  "-framework Metal"
)

if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  if(CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
    set(metal_sdk iphonesimulator)
    set(metal_minimum "-mios-simulator-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  else()
    set(metal_sdk iphoneos)
    set(metal_minimum "-mios-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
else()
  set(metal_sdk macosx)
  set(metal_minimum "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
set(metal_source_dir "${COLMAP_ROOT}/src/thirdparty/SiftMetal")
set(metal_shaders BilinearUpScale Convolution ConvolutionSeries
    NearestNeighborDownScale SIFTDescriptor SIFTExtrema SIFTInterpolate
    SIFTMatcher SIFTOrientation Subtract)
file(GLOB metal_headers CONFIGURE_DEPENDS "${metal_source_dir}/include/*.h"
     "${metal_source_dir}/shaders/*.hpp")
set(metal_air_files)
foreach(shader IN LISTS metal_shaders)
  set(air "${CMAKE_CURRENT_BINARY_DIR}/metal/${shader}.air")
  add_custom_command(OUTPUT "${air}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/metal"
    COMMAND xcrun -sdk "${metal_sdk}" metal "${metal_minimum}"
            -c "${metal_source_dir}/shaders/${shader}.metal" -o "${air}"
            -I "${metal_source_dir}/include" -I "${metal_source_dir}/shaders"
    DEPENDS "${metal_source_dir}/shaders/${shader}.metal" ${metal_headers}
    VERBATIM)
  list(APPEND metal_air_files "${air}")
endforeach()
set(sift_metallib "${CMAKE_CURRENT_BINARY_DIR}/sift.metallib")
add_custom_command(OUTPUT "${sift_metallib}"
  COMMAND xcrun -sdk "${metal_sdk}" metallib ${metal_air_files} -o "${sift_metallib}"
  DEPENDS ${metal_air_files} VERBATIM)
add_custom_target(colmap_sparse_metallib ALL DEPENDS "${sift_metallib}")
add_dependencies(colmap_sparse colmap_sparse_metallib)

# Export the exact native dependency archive locations for libtool packaging.
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/static-archives.txt"
  CONTENT "$<TARGET_FILE:colmap_sparse>\n$<TARGET_FILE:Ceres::ceres>\n$<TARGET_FILE:glog::glog>\n$<TARGET_FILE:PoseLib::PoseLib>\n")
install(TARGETS colmap_sparse ARCHIVE DESTINATION lib)
install(FILES include/ColmapSparse.h DESTINATION include)
install(FILES "${sift_metallib}" DESTINATION resources)

option(COLMAP_IOS_BUILD_TESTS "Build native macOS validation executables" OFF)
if(COLMAP_IOS_BUILD_TESTS)
  if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    message(FATAL_ERROR "Native command-line tests require the macosx SDK")
  endif()
  enable_testing()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/bitmap_apple_test.mm")
    add_executable(bitmap_apple_test tests/bitmap_apple_test.mm)
    target_compile_options(bitmap_apple_test PRIVATE -fobjc-arc)
    target_include_directories(bitmap_apple_test PRIVATE "${COLMAP_ROOT}/src")
    target_link_libraries(bitmap_apple_test PRIVATE colmap_sparse Eigen3::Eigen
                          glog::glog Ceres::ceres)
    add_test(NAME bitmap_apple_test COMMAND bitmap_apple_test)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/sparse_smoke.mm")
    add_executable(sparse_smoke tests/sparse_smoke.mm)
    target_compile_options(sparse_smoke PRIVATE -fobjc-arc)
    target_link_libraries(sparse_smoke PRIVATE colmap_sparse)
  endif()
endif()
