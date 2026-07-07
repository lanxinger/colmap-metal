//
//  SIFTDescriptor.metal
//  SkyLight
//
//  Created by Luke Van In on 2023/01/08.
//

#include <metal_stdlib>

#include "../include/SIFTDescriptor.h"

using namespace metal;


bool normalizeFeatures(
    int count,
    thread float * features
) {
    float magnitude = 0;
    for (int i = 0; i < count; i++) {
        float f = features[i];
        if (!isfinite(f)) {
            return false;
        }
        magnitude += (f * f);
    }
    if (!isfinite(magnitude) || magnitude <= 0.0f) {
        return false;
    }
    const float d = 1.0 / sqrt(magnitude);
    for (int i = 0; i < count; i++) {
        features[i] *= d;
    }
    return true;
}

    
void thresholdFeatures(
    int count,
    thread float * features,
    float threshold
) {
    for (int i = 0; i < count; i++) {
        features[i] = min(features[i], threshold);
    }
}

    
void storeFeatures(
    int count,
    thread float * features,
    thread float * output
) {
    for (int i = 0; i < count; i++) {
        output[i] = features[i];
    }
}

    
int offset(int x, int y, int b) {
    const int side = 4;
    const int bins = SIFT_DESCRIPTOR_ORIENTATION_BINS;
    return (y * side * bins) + (x * bins) + b;
}


void addValue(
    thread float * patch,
    int x,
    int y,
    int b,
    float value
) {
    const int side = 4;
    const int bins = SIFT_DESCRIPTOR_ORIENTATION_BINS;
    if ((x < 0) || (x >= side) || (y < 0) || (y >= side)) {
        return;
    }
    if (b < 0) {
        b += bins;
    }
    if (b >= bins) {
        b -= bins;
    }
    patch[offset(x, y, b)] += value;
}


void addFeature(
    thread float * patch,
    float x,
    float y,
    float b,
    float value
) {
    // Integer coordinates of the four pixels surrounding the point x, y
    const int2 ca = int2(floor(x), floor(y));
    const int2 cb = int2(ceil(x), floor(y));
    const int2 cc = int2(ceil(x), ceil(y));
    const int2 cd = int2(floor(x), ceil(y));
    
    // Bins surrounding the bin at index b
    const int ba = floor(b);
    const int bb = ceil(b);
    
    const float iMax = x - floor(x);
    const float iMin = 1 - iMax;
    const float jMax = y - floor(y);
    const float jMin = 1 - jMax;
    const float bMax = b - floor(b);
    const float bMin = 1 - bMax;
    
    addValue(patch, ca.x, ca.y, ba, (iMin * jMin * bMin) * value);
    addValue(patch, ca.x, ca.y, bb, (iMin * jMin * bMax) * value);
    
    addValue(patch, cb.x, cb.y, ba, (iMax * jMin * bMin) * value);
    addValue(patch, cb.x, cb.y, bb, (iMax * jMin * bMax) * value);
    
    addValue(patch, cc.x, cc.y, ba, (iMax * jMax * bMin) * value);
    addValue(patch, cc.x, cc.y, bb, (iMax * jMax * bMax) * value);
    
    addValue(patch, cd.x, cd.y, ba, (iMin * jMax * bMin) * value);
    addValue(patch, cd.x, cd.y, bb, (iMin * jMax * bMax) * value);
}

    
kernel void siftDescriptors(
    device SIFTDescriptorResult * results [[buffer(0)]],
    device SIFTDescriptorInput * inputs [[buffer(1)]],
    device SIFTDescriptorParameters & parameters [[buffer(2)]],
    texture2d_array<float, access::read> gradientTextures [[texture(0)]],
    uint gid [[thread_position_in_grid]]
) {
   
//    let octave = dog.octaves[keypoint.octave]
    // let images = octave.gaussianImages
    // let histogramsPerAxis = configuration.descriptorHistogramsPerAxis
    const SIFTDescriptorInput input = inputs[gid];
    SIFTDescriptorResult result;
    result.valid = 0;
    result.keypoint = input.keypoint;
    result.theta = input.theta;
    
    
//    let image = octaves[keypoint.octave].gradientImages[keypoint.scale]
    
    // let delta = octave.delta
    // let lambda = configuration.lambdaDescriptor
    // let a = keypoint.absoluteCoordinate
    float px = float(input.absoluteX) / parameters.delta;
    float py = float(input.absoluteY) / parameters.delta;

    // Check that the keypoint is sufficiently far from the edge to include
    // entire area of the descriptor.
    
    // Keep the descriptor bounds check here so descriptor sampling never reads
    // outside the gradient texture.
    // let diagonal = Float(2).squareRoot() * lambda * sigma
    // let f = Float(histogramsPerAxis + 1) / Float(histogramsPerAxis)
    // let side = Int((diagonal * f).rounded())
    
    //let radius = lambda * f
    const int d = 4; // width of 2d array of histograms
    const int bins = SIFT_DESCRIPTOR_ORIENTATION_BINS;
    
    const float tau = 2 * M_PI_F;
    if (!isfinite(input.theta)) {
        results[gid] = result;
        return;
    }
    const float cosT = cos(input.theta);
    const float sinT = sin(input.theta);
    const float binsPerRadian = (float)bins / tau;
    const float exponentDenominator = (float)(d * d) * 0.5;
    const float interval = (float)input.scale + input.subScale;
    const float intervals = (float)parameters.scalesPerOctave;
    const float sigma = 1.6;
    const float scale = sigma * pow(2.0, interval / intervals); // identical to below
    // let _sigma = keypoint.sigma / octave.delta // identical to above
    const float histogramWidth = 3.0 * scale; // 3.0 constant from Whess (OpenSIFT)
    const int radius = histogramWidth * sqrt(2.0) * ((float)d + 1.0) * 0.5 + 0.5;
    
    const int maxScale = parameters.scalesPerOctave + 2;

    if (px < 0.0f || py < 0.0f ||
        px >= (float)parameters.width || py >= (float)parameters.height ||
        input.scale < 0 || input.scale > maxScale) {
        results[gid] = result;
        return;
    }

    // Create histograms
    const int featureCount = d * d * bins;
    float features[featureCount];
    
    for (int i = 0; i < featureCount; i++) {
        features[i] = 0;
    }

    for (int j = -radius; j <= +radius; j++) {
        for (int i = -radius; i <= +radius; i++) {
            const int sampleX = int(px + j);
            const int sampleY = int(py + i);
            if (sampleX < 1 || sampleY < 1 ||
                sampleX >= parameters.width - 1 ||
                sampleY >= parameters.height - 1) {
                continue;
            }

            float rx = ((float)j * cosT - (float)i * sinT) / histogramWidth;
            float ry = ((float)j * sinT + (float)i * cosT) / histogramWidth;
            float bx = rx + (float)(d / 2) - 0.5;
            float by = ry + (float)(d / 2) - 0.5;
            if (!isfinite(bx) || !isfinite(by)) {
                continue;
            }
            
            float2 g = gradientTextures.read(
                uint2(uint(sampleX), uint(sampleY)), input.scale).rg;
            if (!all(isfinite(g)) || g.g <= 0.0f) {
                continue;
            }
            float orientation = g.r - input.theta;
            float magnitude = g.g;
            while (orientation < 0) {
                orientation += tau;
            }
            while (orientation >= tau) {
                orientation -= tau;
            }

            // Bin
            float bin = orientation * binsPerRadian;
            if (!isfinite(bin)) {
                continue;
            }

            // Total contribution
            float exponentNumerator = rx * rx + ry * ry;
            float w = exp(-exponentNumerator / exponentDenominator);
            float value = magnitude * w;
            if (!isfinite(value) || value <= 0.0f) {
                continue;
            }
            
            addFeature(features, bx, by, bin, value);
        }
    }
    
    // print("feature x=\(Int(a.x)) y=\(Int(a.y)) scale=\(scale) sigma=\(_sigma) histogramWidth=\(histogramWidth) radius=\(radius)")
    
    // Serialize histograms into array
    if (!normalizeFeatures(featureCount, features)) {
        results[gid] = result;
        return;
    }
    thresholdFeatures(featureCount, features, 0.2);
    if (!normalizeFeatures(featureCount, features)) {
        results[gid] = result;
        return;
    }
    storeFeatures(featureCount, features, result.features);
    
    result.valid = 1;
    result.keypoint = input.keypoint;
    result.theta = input.theta;
    
    results[gid] = result;
}
