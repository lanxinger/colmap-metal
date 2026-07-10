//
//  NearestNeighbor.h
//  SkyLight
//
//  Created by Luke Van In on 2023/01/08.
//

#include <simd/simd.h>

#ifndef NearestNeighbor_h
#define NearestNeighbor_h

struct NearestNeighborScaleParameters {
    int32_t inputSlice;
    int32_t outputSlice;
    // Logical image sizes; textures may be allocated larger than the image.
    int32_t inputWidth;
    int32_t inputHeight;
    int32_t outputWidth;
    int32_t outputHeight;
};

#endif /* NearestNeighbor_h */
