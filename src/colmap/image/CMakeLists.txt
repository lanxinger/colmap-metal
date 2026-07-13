# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


set(FOLDER_NAME "image")

COLMAP_ADD_LIBRARY(
    NAME colmap_image
    SRCS
        line.h line.cc
        undistortion.h undistortion.cc
        warp.h warp.cc
        warp_metal.h
    PUBLIC_LINK_LIBS
        colmap_scene
        colmap_util
        Eigen3::Eigen
    PRIVATE_LINK_LIBS
        colmap_sensor
        colmap_vlfeat
)
if(METAL_ENABLED)
    set(IMAGE_WARP_METAL_SHADER
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/WarpImage.metal")
    set(IMAGE_WARP_METAL_AIR
        "${CMAKE_CURRENT_BINARY_DIR}/WarpImage.air")
    set(IMAGE_WARP_METALLIB
        "${CMAKE_CURRENT_BINARY_DIR}/image_warp.metallib")

    add_custom_command(
        OUTPUT ${IMAGE_WARP_METAL_AIR}
        COMMAND xcrun -sdk macosx metal
            -c ${IMAGE_WARP_METAL_SHADER}
            -o ${IMAGE_WARP_METAL_AIR}
        DEPENDS ${IMAGE_WARP_METAL_SHADER}
        COMMENT "Compiling Metal image warp shader"
    )
    add_custom_command(
        OUTPUT ${IMAGE_WARP_METALLIB}
        COMMAND xcrun -sdk macosx metallib
            ${IMAGE_WARP_METAL_AIR}
            -o ${IMAGE_WARP_METALLIB}
        DEPENDS ${IMAGE_WARP_METAL_AIR}
        COMMENT "Linking Metal image warp library"
    )
    add_custom_target(image_warp_metallib DEPENDS ${IMAGE_WARP_METALLIB})

    target_sources(colmap_image PRIVATE warp_metal.mm)
    add_dependencies(colmap_image image_warp_metallib)
    set_source_files_properties(warp_metal.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc -std=c++17"
    )
    target_compile_definitions(colmap_image PRIVATE
        IMAGE_WARP_METALLIB_PATH="${IMAGE_WARP_METALLIB}"
    )
    target_link_libraries(colmap_image PUBLIC
        "-framework Metal"
        "-framework Foundation"
    )

    install(FILES ${IMAGE_WARP_METALLIB}
            DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES ${IMAGE_WARP_METALLIB}
            DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(LSD_ENABLED)
    target_link_libraries(colmap_image PRIVATE colmap_lsd)
endif()

COLMAP_ADD_TEST(
    NAME line_test
    SRCS line_test.cc
    LINK_LIBS colmap_image
)
COLMAP_ADD_TEST(
    NAME undistortion_test
    SRCS undistortion_test.cc
    LINK_LIBS colmap_image
)
COLMAP_ADD_TEST(
    NAME warp_test
    SRCS warp_test.cc
    LINK_LIBS colmap_image
)
