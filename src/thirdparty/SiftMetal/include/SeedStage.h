// SeedStage.h - shared Metal/C++ structs for the seed-stage kernels.
// Logical image sizes are passed explicitly because textures may be
// allocated larger than the image they currently hold.

#include <simd/simd.h>

#ifndef SeedStage_h
#define SeedStage_h

struct BilinearUpScaleParameters {
    int32_t inputWidth;
    int32_t inputHeight;
    int32_t outputWidth;
    int32_t outputHeight;
};

struct SeedConvolutionParameters {
    int32_t count;
    int32_t width;
    int32_t height;
};

#endif /* SeedStage_h */
