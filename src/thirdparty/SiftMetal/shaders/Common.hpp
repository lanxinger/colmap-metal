//
//  Common.h
//  SkyLight
//
//  Created by Luke Van In on 2023/01/07.
//

#include <metal_stdlib>

#ifndef Common_h
#define Common_h

using namespace metal;

static inline int symmetrizedCoordinates(int i, int l) {
    int ll = 2 * l;
    i = (i + ll) % (ll);
    if (i > l - 1){
        i = ll - 1 - i;
    }
    return i;
}

// Computes the SIFT gradient (orientation, magnitude) at a pixel of a
// Gaussian scale-space slice, using the same central differences and
// atan2(tx, ty) convention the orientation and descriptor kernels consume.
// Computing this on the fly is cheaper than precomputing gradients for the
// whole pyramid: only small windows around keypoints are ever sampled.
static inline float2 siftGradientAt(
    texture2d_array<float, access::read> tex,
    int x,
    int y,
    uint slice
) {
    const int w = tex.get_width();
    const int h = tex.get_height();
    const uint px = uint(symmetrizedCoordinates(x + 1, w));
    const uint mx = uint(symmetrizedCoordinates(x - 1, w));
    const uint py = uint(symmetrizedCoordinates(y + 1, h));
    const uint my = uint(symmetrizedCoordinates(y - 1, h));
    const float cpx = tex.read(uint2(px, uint(y)), slice).r;
    const float cmx = tex.read(uint2(mx, uint(y)), slice).r;
    const float cpy = tex.read(uint2(uint(x), py), slice).r;
    const float cmy = tex.read(uint2(uint(x), my), slice).r;
    const float tx = (cpx - cmx) * 0.5;
    const float ty = (cpy - cmy) * 0.5;
    return float2(atan2(tx, ty), sqrt(tx * tx + ty * ty));
}

//constant float3x3 identity = float3x3(
//    float3(1, 0, 0),
//    float3(0, 1, 0),
//    float3(0, 0, 1)
//);

// Computes the inverse of a 3x3 matrix using the cross product and
// triple product.
// https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_×_3_matrices
// See: https://www.onlinemathstutor.org/post/3x3_inverses
static inline float3x3 invert(const float3x3 input) {
    const float3 x0 = input[0];
    const float3 x1 = input[1];
    const float3 x2 = input[2];

    const float d = determinant(input); // dot(x0, cross(x1, x2));

    const float3x3 cp = float3x3(
        cross(x1, x2),
        cross(x2, x0),
        cross(x0, x1)
    );
    return (1.0 / d) * cp;
}

// https://github.com/markkilgard/glut/blob/master/lib/gle/vvector.h
//#define SCALE_ADJOINT_3X3(a,s,m)                \
//{                                \
//   a[0][0] = (s) * (m[1][1] * m[2][2] - m[1][2] * m[2][1]);    \
//   a[1][0] = (s) * (m[1][2] * m[2][0] - m[1][0] * m[2][2]);    \
//   a[2][0] = (s) * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);    \
//                                \
//   a[0][1] = (s) * (m[0][2] * m[2][1] - m[0][1] * m[2][2]);    \
//   a[1][1] = (s) * (m[0][0] * m[2][2] - m[0][2] * m[2][0]);    \
//   a[2][1] = (s) * (m[0][1] * m[2][0] - m[0][0] * m[2][1]);    \
//                                \
//   a[0][2] = (s) * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);    \
//   a[1][2] = (s) * (m[0][2] * m[1][0] - m[0][0] * m[1][2]);    \
//   a[2][2] = (s) * (m[0][0] * m[1][1] - m[0][1] * m[1][0]);    \
//}
//float3x3 scaleAdjoint(const float3x3 m, const float s) {
//    float3x3 a;
//    a[0][0] = (s) * (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
//    a[1][0] = (s) * (m[1][2] * m[2][0] - m[1][0] * m[2][2]);
//    a[2][0] = (s) * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
//
//    a[0][1] = (s) * (m[0][2] * m[2][1] - m[0][1] * m[2][2]);
//    a[1][1] = (s) * (m[0][0] * m[2][2] - m[0][2] * m[2][0]);
//    a[2][1] = (s) * (m[0][1] * m[2][0] - m[0][0] * m[2][1]);
//
//    a[0][2] = (s) * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
//    a[1][2] = (s) * (m[0][2] * m[1][0] - m[0][0] * m[1][2]);
//    a[2][2] = (s) * (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
//
//    return a;
//}

// https://github.com/markkilgard/glut/blob/master/lib/gle/vvector.h
//#define INVERT_3X3(b,det,a)            \
//{                        \
//   double tmp;                    \
//   DETERMINANT_3X3 (det, a);            \
//   tmp = 1.0 / (det);                \
//   SCALE_ADJOINT_3X3 (b, tmp, a);        \
//}
//float3x3 invert(const float3x3 input) {
//    float d = 1.0 / determinant(input);
//    return scaleAdjoint(input, d);
//}


// https://metalbyexample.com/fundamentals-of-image-processing/
static inline float gaussian(float x, float y, float sigma) {
    float ss = sigma * sigma;
    float xx = x * x;
    float yy = y * y;
    float base = sqrt(2 * M_PI_F * ss);
    float exponent = (xx + yy) / (2 * ss);
    return (1 / base) * exp(-exponent);
}

#endif /* Common_h */
