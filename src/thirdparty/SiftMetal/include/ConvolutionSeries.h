//
//  ConvolutionSeries.h
//  SkyLight
//
//  Created by Luke Van In on 2023/01/08.
//

#include <simd/simd.h>

#ifndef ConvolutionSeries_h
#define ConvolutionSeries_h

#define CONVOLUTION_WEIGHTS_LENGTH 32
// Consecutive outputs computed per thread along the filter axis; interior
// threads share the overlapping tap window instead of re-reading it.
#define CONVOLUTION_OUTPUTS_PER_THREAD 4


struct ConvolutionParameters {
    int32_t inputDepth;
    int32_t outputDepth;
    int32_t count;
    // Logical image size; textures may be allocated larger than the image.
    int32_t width;
    int32_t height;
    float weights[CONVOLUTION_WEIGHTS_LENGTH];
};


#endif /* ConvolutionSeries_h */
