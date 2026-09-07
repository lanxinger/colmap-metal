// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Apple implementation of the existing Bitmap API. Compile this translation
// unit instead of sensor/bitmap.cc, never alongside it. The generic color,
// metadata-query, and rotation behavior follows that implementation.

#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/database.h"

#include <array>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Accelerate/Accelerate.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

namespace colmap {
namespace {

// These are decoder safety bounds, not an application memory budget. Read uses
// an RGBA8 decode surface plus final RGB/gray storage. The public job API must
// reject captures exceeding its tighter, device-appropriate budget first.
constexpr int kMaximumImageDimension = 32768;
constexpr size_t kMaximumImagePixels = 64 * 1024 * 1024;

size_t CheckedBytes(const int width, const int height, const int channels) {
  if (width <= 0 || height <= 0 || width > kMaximumImageDimension ||
      height > kMaximumImageDimension || channels < 1 || channels > 4) {
    throw std::invalid_argument("Unsupported bitmap dimensions or channels");
  }
  const size_t pixels = static_cast<size_t>(width) * height;
  if (pixels > kMaximumImagePixels) {
    throw std::invalid_argument("Bitmap exceeds the Apple decoder pixel limit");
  }
  return pixels * channels;
}

void CheckStorage(const Bitmap& bitmap) {
  if ((!bitmap.IsRGB() && !bitmap.IsGrey()) ||
      bitmap.NumBytes() != CheckedBytes(bitmap.Width(), bitmap.Height(), bitmap.Channels())) {
    throw std::invalid_argument("Bitmap storage does not match its dimensions");
  }
}

template <typename T>
class CFHandle {
 public:
  explicit CFHandle(T value = nullptr) : value_(value) {}
  ~CFHandle() {
    if (value_) CFRelease(value_);
  }
  CFHandle(const CFHandle&) = delete;
  CFHandle& operator=(const CFHandle&) = delete;
  T Get() const { return value_; }
  T Release() { return std::exchange(value_, nullptr); }

 private:
  T value_;
};

struct MetaValue {
  std::string type;
  std::vector<uint8_t> bytes;
  std::string string;
};

struct AppleMetaData : Bitmap::MetaData {
  // Sorted iteration makes metadata serialization deterministic.
  std::map<std::string, MetaValue> values;
  CFDictionaryRef source_properties = nullptr;

  AppleMetaData() = default;
  AppleMetaData(const AppleMetaData& other)
      : values(other.values), source_properties(other.source_properties) {
    if (source_properties) CFRetain(source_properties);
  }
  AppleMetaData& operator=(const AppleMetaData&) = delete;
  ~AppleMetaData() override {
    if (source_properties) CFRelease(source_properties);
  }
};

std::unique_ptr<Bitmap::MetaData> CopyMetadata(const std::unique_ptr<Bitmap::MetaData>& metadata) {
  if (!metadata) return nullptr;
  return std::make_unique<AppleMetaData>(*static_cast<const AppleMetaData*>(metadata.get()));
}

size_t MetadataTypeSize(const std::string_view type) {
  if (type == "int8" || type == "uint8") return 1;
  if (type == "int16" || type == "uint16") return 2;
  if (type == "int" || type == "uint" || type == "float") return 4;
  if (type == "int64" || type == "uint64" || type == "double") return 8;
  if (type == "point" || type == "vector" || type == "color") return 12;
  throw std::invalid_argument("Unsupported Apple bitmap metadata type");
}

CFTypeRef DictionaryValue(CFDictionaryRef dictionary, CFStringRef key) {
  return dictionary ? CFDictionaryGetValue(dictionary, key) : nullptr;
}

CFDictionaryRef NestedDictionary(CFDictionaryRef dictionary, CFStringRef key) {
  CFTypeRef value = DictionaryValue(dictionary, key);
  return value && CFGetTypeID(value) == CFDictionaryGetTypeID()
             ? static_cast<CFDictionaryRef>(value)
             : nullptr;
}

std::optional<double> NumberValue(CFTypeRef value) {
  if (!value || CFGetTypeID(value) != CFNumberGetTypeID()) return std::nullopt;
  double result = 0;
  if (!CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberDoubleType, &result) ||
      !std::isfinite(result)) {
    return std::nullopt;
  }
  return result;
}

std::optional<std::string> StringValue(CFTypeRef value) {
  if (!value || CFGetTypeID(value) != CFStringGetTypeID()) return std::nullopt;
  const auto string = static_cast<CFStringRef>(value);
  const CFIndex length =
      CFStringGetMaximumSizeForEncoding(CFStringGetLength(string), kCFStringEncodingUTF8);
  if (length < 0 || length > 1024 * 1024) return std::nullopt;
  std::vector<char> utf8(static_cast<size_t>(length) + 1);
  if (!CFStringGetCString(string, utf8.data(), utf8.size(), kCFStringEncodingUTF8)) {
    return std::nullopt;
  }
  return std::string(utf8.data());
}

struct MetadataMapping {
  const char* name;
  CFStringRef dictionary;
  CFStringRef key;
  const char* type;
};

const std::array<MetadataMapping, 15>& MetadataMappings() {
  static const std::array<MetadataMapping, 15> mappings = {{
      {"Orientation", nullptr, kCGImagePropertyOrientation, "int"},
      {"Make", kCGImagePropertyTIFFDictionary, kCGImagePropertyTIFFMake, "string"},
      {"Model", kCGImagePropertyTIFFDictionary, kCGImagePropertyTIFFModel, "string"},
      {"Software", kCGImagePropertyTIFFDictionary, kCGImagePropertyTIFFSoftware, "string"},
      {"Exif:FocalLength",
       kCGImagePropertyExifDictionary,
       kCGImagePropertyExifFocalLength,
       "float"},
      {"Exif:FocalLengthIn35mmFilm",
       kCGImagePropertyExifDictionary,
       kCGImagePropertyExifFocalLenIn35mmFilm,
       "float"},
      {"Exif:FocalPlaneXResolution",
       kCGImagePropertyExifDictionary,
       kCGImagePropertyExifFocalPlaneXResolution,
       "float"},
      {"Exif:FocalPlaneResolutionUnit",
       kCGImagePropertyExifDictionary,
       kCGImagePropertyExifFocalPlaneResolutionUnit,
       "int"},
      {"Exif:PixelXDimension",
       kCGImagePropertyExifDictionary,
       kCGImagePropertyExifPixelXDimension,
       "int"},
      {"Exif:PixelYDimension",
       kCGImagePropertyExifDictionary,
       kCGImagePropertyExifPixelYDimension,
       "int"},
      {"Exif:DateTimeOriginal",
       kCGImagePropertyExifDictionary,
       kCGImagePropertyExifDateTimeOriginal,
       "string"},
      {"GPS:LatitudeRef", kCGImagePropertyGPSDictionary, kCGImagePropertyGPSLatitudeRef, "string"},
      {"GPS:LongitudeRef",
       kCGImagePropertyGPSDictionary,
       kCGImagePropertyGPSLongitudeRef,
       "string"},
      {"GPS:Altitude", kCGImagePropertyGPSDictionary, kCGImagePropertyGPSAltitude, "float"},
      {"Exif:ExposureTime",
       kCGImagePropertyExifDictionary,
       kCGImagePropertyExifExposureTime,
       "float"},
  }};
  return mappings;
}

void ImportMetadata(CFDictionaryRef properties, Bitmap& bitmap) {
  for (const auto& mapping : MetadataMappings()) {
    const CFDictionaryRef dictionary =
        mapping.dictionary ? NestedDictionary(properties, mapping.dictionary) : properties;
    const CFTypeRef value = DictionaryValue(dictionary, mapping.key);
    if (std::string_view(mapping.type) == "string") {
      if (const auto string = StringValue(value)) {
        bitmap.SetMetaData(mapping.name, *string);
      }
    } else if (const auto number = NumberValue(value)) {
      if (std::string_view(mapping.type) == "float") {
        const float scalar = static_cast<float>(*number);
        if (std::isfinite(scalar)) {
          bitmap.SetMetaData(mapping.name, "float", &scalar);
        }
      } else if (*number >= std::numeric_limits<int>::min() &&
                 *number <= std::numeric_limits<int>::max()) {
        const int scalar = static_cast<int>(*number);
        bitmap.SetMetaData(mapping.name, "int", &scalar);
      }
    }
  }
  const auto gps = NestedDictionary(properties, kCGImagePropertyGPSDictionary);
  for (const auto& entry : {std::make_pair("GPS:Latitude", kCGImagePropertyGPSLatitude),
                            std::make_pair("GPS:Longitude", kCGImagePropertyGPSLongitude)}) {
    if (const auto degrees = NumberValue(DictionaryValue(gps, entry.second))) {
      const double absolute = std::abs(*degrees);
      const double whole_degrees = std::floor(absolute);
      const double minutes = (absolute - whole_degrees) * 60;
      const float components[3] = {static_cast<float>(whole_degrees),
                                   static_cast<float>(std::floor(minutes)),
                                   static_cast<float>((minutes - std::floor(minutes)) * 60)};
      bitmap.SetMetaData(entry.first, "point", components);
    }
  }
  if (const auto reference = NumberValue(DictionaryValue(gps, kCGImagePropertyGPSAltitudeRef))) {
    bitmap.SetMetaData("GPS:AltitudeRef", *reference == 1 ? "1" : "0");
  }
}

void SetNumber(CFMutableDictionaryRef dictionary, CFStringRef key, const double value) {
  CFHandle<CFNumberRef> number(CFNumberCreate(nullptr, kCFNumberDoubleType, &value));
  if (!number.Get()) throw std::bad_alloc();
  CFDictionarySetValue(dictionary, key, number.Get());
}

void SetProperty(CFMutableDictionaryRef properties,
                 CFStringRef group,
                 CFStringRef key,
                 CFTypeRef value) {
  if (!group) {
    CFDictionarySetValue(properties, key, value);
    return;
  }
  const CFDictionaryRef existing = NestedDictionary(properties, group);
  CFHandle<CFMutableDictionaryRef> dictionary(
      existing ? CFDictionaryCreateMutableCopy(nullptr, 0, existing)
               : CFDictionaryCreateMutable(
                     nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
  if (!dictionary.Get()) throw std::bad_alloc();
  CFDictionarySetValue(dictionary.Get(), key, value);
  CFDictionarySetValue(properties, group, dictionary.Get());
}

void ExportMetadata(const Bitmap& bitmap, CFMutableDictionaryRef properties) {
  for (const auto& mapping : MetadataMappings()) {
    if (std::string_view(mapping.type) == "string") {
      if (const auto value = bitmap.GetMetaData(mapping.name)) {
        CFHandle<CFStringRef> string(
            CFStringCreateWithBytes(nullptr,
                                    reinterpret_cast<const UInt8*>(value->data()),
                                    value->size(),
                                    kCFStringEncodingUTF8,
                                    false));
        if (string.Get()) {
          SetProperty(properties, mapping.dictionary, mapping.key, string.Get());
        }
      }
    } else {
      double value = 0;
      if (std::string_view(mapping.type) == "float") {
        float scalar;
        if (!bitmap.GetMetaData(mapping.name, "float", &scalar)) continue;
        value = scalar;
      } else {
        int scalar;
        if (!bitmap.GetMetaData(mapping.name, "int", &scalar)) continue;
        value = scalar;
      }
      if (!std::isfinite(value)) continue;
      // ImageIO expects integer CFNumbers for TIFF SHORT/LONG tags. In
      // particular, a floating-point 35mm focal equivalent is silently omitted
      // from JPEG EXIF, although COLMAP exposes that value as a float.
      const bool integer_tag = std::string_view(mapping.type) == "int" ||
                               mapping.key == kCGImagePropertyExifFocalLenIn35mmFilm;
      if (integer_tag &&
          (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())) {
        continue;
      }
      const int integer_value = integer_tag ? static_cast<int>(std::round(value)) : 0;
      CFHandle<CFNumberRef> number(integer_tag
                                       ? CFNumberCreate(nullptr, kCFNumberIntType, &integer_value)
                                       : CFNumberCreate(nullptr, kCFNumberDoubleType, &value));
      if (!number.Get()) throw std::bad_alloc();
      SetProperty(properties, mapping.dictionary, mapping.key, number.Get());
    }
  }
  for (const auto& entry : {std::make_pair("GPS:Latitude", kCGImagePropertyGPSLatitude),
                            std::make_pair("GPS:Longitude", kCGImagePropertyGPSLongitude)}) {
    float components[3];
    if (!bitmap.GetMetaData(entry.first, "point", components)) continue;
    const double value = std::abs(components[0] + components[1] / 60.0 + components[2] / 3600.0);
    if (!std::isfinite(value)) continue;
    CFHandle<CFNumberRef> number(CFNumberCreate(nullptr, kCFNumberDoubleType, &value));
    if (!number.Get()) throw std::bad_alloc();
    SetProperty(properties, kCGImagePropertyGPSDictionary, entry.second, number.Get());
  }
  if (const auto reference = bitmap.GetMetaData("GPS:AltitudeRef")) {
    const int value = *reference == "1" ? 1 : 0;
    CFHandle<CFNumberRef> number(CFNumberCreate(nullptr, kCFNumberIntType, &value));
    if (!number.Get()) throw std::bad_alloc();
    SetProperty(
        properties, kCGImagePropertyGPSDictionary, kCGImagePropertyGPSAltitudeRef, number.Get());
  }
  SetNumber(properties, kCGImagePropertyPixelWidth, bitmap.Width());
  SetNumber(properties, kCGImagePropertyPixelHeight, bitmap.Height());
  const int width = bitmap.Width();
  const int height = bitmap.Height();
  CFHandle<CFNumberRef> x(CFNumberCreate(nullptr, kCFNumberIntType, &width));
  CFHandle<CFNumberRef> y(CFNumberCreate(nullptr, kCFNumberIntType, &height));
  if (!x.Get() || !y.Get()) throw std::bad_alloc();
  SetProperty(
      properties, kCGImagePropertyExifDictionary, kCGImagePropertyExifPixelXDimension, x.Get());
  SetProperty(
      properties, kCGImagePropertyExifDictionary, kCGImagePropertyExifPixelYDimension, y.Get());
}

uint8_t ByteValue(const double value) {
  return static_cast<uint8_t>(std::clamp(std::round(value), 0.0, 255.0));
}

std::array<uint8_t, 256> SRGBTransferTable(const bool linearize) {
  std::array<uint8_t, 256> table;
  for (size_t index = 0; index < table.size(); ++index) {
    const double value = index / 255.0;
    const double converted =
        linearize ? (value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4))
                  : (value <= 0.0031308 ? value * 12.92 : 1.055 * std::pow(value, 1 / 2.4) - 0.055);
    table[index] = ByteValue(converted * 255);
  }
  return table;
}

void ConvertSRGB(std::vector<uint8_t>& pixels, const bool linearize) {
  const auto table = SRGBTransferTable(linearize);
  for (uint8_t& pixel : pixels) pixel = table[pixel];
}

// Pixel centers are mapped as (destination + 0.5) * scale - 0.5. Triangle
// support widens for reduction (anti-aliasing); box uses exact pixel coverage.
// The separable implementation rounds each pass to bytes and clamps edges.
// It is intentionally not promised to be bit-identical to OpenImageIO.
std::vector<uint8_t> RescaleAxis(const std::vector<uint8_t>& source,
                                 const int width,
                                 const int height,
                                 const int channels,
                                 const int target_size,
                                 const bool horizontal,
                                 const Bitmap::RescaleFilter filter) {
  const int source_size = horizontal ? width : height;
  const int output_width = horizontal ? target_size : width;
  const int output_height = horizontal ? height : target_size;
  std::vector<uint8_t> output(CheckedBytes(output_width, output_height, channels));
  const double scale = static_cast<double>(source_size) / target_size;
  const int line_count = horizontal ? height : width;
  for (int target = 0; target < target_size; ++target) {
    const double center = (target + 0.5) * scale - 0.5;
    const double radius = std::max(1.0, scale);
    const bool box = filter == Bitmap::RescaleFilter::kBox;
    const double left = target * scale;
    const double right = (target + 1) * scale;
    const int begin =
        box ? static_cast<int>(std::floor(left)) : static_cast<int>(std::floor(center - radius));
    const int end =
        box ? static_cast<int>(std::ceil(right)) : static_cast<int>(std::ceil(center + radius)) + 1;
    std::vector<std::pair<int, double>> weights;
    double total_weight = 0;
    for (int index = begin; index < end; ++index) {
      const double weight =
          box ? std::max(0.0, std::min(right, index + 1.0) - std::max(left, double(index)))
              : std::max(0.0, 1.0 - std::abs(index - center) / radius);
      if (weight <= 0) continue;
      weights.emplace_back(std::clamp(index, 0, source_size - 1), weight);
      total_weight += weight;
    }
    for (int line = 0; line < line_count; ++line) {
      for (int channel = 0; channel < channels; ++channel) {
        double value = 0;
        for (const auto& [index, weight] : weights) {
          const size_t offset =
              horizontal ? (static_cast<size_t>(line) * width + index) * channels + channel
                         : (static_cast<size_t>(index) * width + line) * channels + channel;
          value += source[offset] * weight;
        }
        const size_t offset =
            horizontal ? (static_cast<size_t>(line) * target_size + target) * channels + channel
                       : (static_cast<size_t>(target) * width + line) * channels + channel;
        output[offset] = ByteValue(value / total_weight);
      }
    }
  }
  return output;
}

std::vector<uint8_t> RescaleHighQuality(const Bitmap& source, const int width, const int height) {
  std::vector<uint8_t> output(CheckedBytes(width, height, source.Channels()));
  std::vector<uint8_t> source_plane;
  std::vector<uint8_t> output_plane;
  if (source.IsRGB()) {
    source_plane.resize(CheckedBytes(source.Width(), source.Height(), 1));
    output_plane.resize(CheckedBytes(width, height, 1));
  }
  for (int channel = 0; channel < source.Channels(); ++channel) {
    if (source.IsRGB()) {
      for (size_t pixel = 0; pixel < source_plane.size(); ++pixel) {
        source_plane[pixel] = source.RowMajorData()[pixel * 3 + channel];
      }
    }
    vImage_Buffer input = {
        source.IsGrey() ? const_cast<uint8_t*>(source.RowMajorData().data()) : source_plane.data(),
        static_cast<vImagePixelCount>(source.Height()),
        static_cast<vImagePixelCount>(source.Width()),
        static_cast<size_t>(source.Width())};
    vImage_Buffer destination = {source.IsGrey() ? output.data() : output_plane.data(),
                                 static_cast<vImagePixelCount>(height),
                                 static_cast<vImagePixelCount>(width),
                                 static_cast<size_t>(width)};
    // Apple's high-quality Lanczos resampler differs from OIIO's
    // direction-dependent default. Both preserve the same pixel-center map.
    if (vImageScale_Planar8(&input, &destination, nullptr, kvImageHighQualityResampling) !=
        kvImageNoError) {
      throw std::runtime_error("Apple high-quality bitmap resize failed");
    }
    if (source.IsRGB()) {
      for (size_t pixel = 0; pixel < output_plane.size(); ++pixel) {
        output[pixel * 3 + channel] = output_plane[pixel];
      }
    }
  }
  return output;
}

}  // namespace

Bitmap::Bitmap() : width_(0), height_(0), channels_(0), linear_colorspace_(false) {}

Bitmap::Bitmap(const int width, const int height, const bool as_rgb, const bool linear_colorspace)
    : width_(width),
      height_(height),
      channels_(as_rgb ? 3 : 1),
      linear_colorspace_(linear_colorspace),
      data_(CheckedBytes(width, height, channels_)),
      meta_data_(std::make_unique<AppleMetaData>()) {
  SetMetaData("oiio:ColorSpace", linear_colorspace ? "linear" : "sRGB");
}

Bitmap::Bitmap(const Bitmap& other)
    : width_(other.width_),
      height_(other.height_),
      channels_(other.channels_),
      linear_colorspace_(other.linear_colorspace_),
      data_(other.data_),
      meta_data_(CopyMetadata(other.meta_data_)) {}

Bitmap::Bitmap(Bitmap&& other) noexcept : Bitmap() { *this = std::move(other); }

Bitmap& Bitmap::operator=(const Bitmap& other) {
  if (this != &other) {
    Bitmap copied(other);
    *this = std::move(copied);
  }
  return *this;
}

Bitmap& Bitmap::operator=(Bitmap&& other) noexcept {
  if (this != &other) {
    width_ = std::exchange(other.width_, 0);
    height_ = std::exchange(other.height_, 0);
    channels_ = std::exchange(other.channels_, 0);
    linear_colorspace_ = std::exchange(other.linear_colorspace_, false);
    data_ = std::move(other.data_);
    meta_data_ = std::move(other.meta_data_);
    other.data_.clear();
  }
  return *this;
}

void Bitmap::Fill(const BitmapColor<uint8_t>& color) {
  if (IsEmpty()) return;
  CheckStorage(*this);
  if (IsGrey()) {
    std::fill(data_.begin(), data_.end(), color.r);
  } else {
    for (size_t index = 0; index < data_.size(); index += 3) {
      data_[index] = color.r;
      data_[index + 1] = color.g;
      data_[index + 2] = color.b;
    }
  }
}

std::optional<int> Bitmap::ExifOrientation() const {
  int orientation;
  if (GetMetaData("Orientation", "int", &orientation)) return orientation;
  return std::nullopt;
}

std::optional<std::string> Bitmap::ExifCameraModel() const {
  const auto make = GetMetaData("Make");
  const auto model = GetMetaData("Model");
  float focal_length;
  if (!make || !model ||
      (!GetMetaData("Exif:FocalLengthIn35mmFilm", "float", &focal_length) &&
       !GetMetaData("Exif:FocalLength", "float", &focal_length))) {
    return std::nullopt;
  }
  std::ostringstream result;
  result.imbue(std::locale::classic());
  result << *make << '-' << *model << '-' << std::fixed << std::setprecision(6) << focal_length
         << '-' << width_ << 'x' << height_;
  return result.str();
}

std::optional<double> Bitmap::ExifFocalLength() const {
  float focal_35mm;
  if (GetMetaData("Exif:FocalLengthIn35mmFilm", "float", &focal_35mm) &&
      std::isfinite(focal_35mm) && focal_35mm > 0) {
    return focal_35mm / 43.27 * std::hypot(double(width_), double(height_));
  }
  float focal_mm;
  if (!GetMetaData("Exif:FocalLength", "float", &focal_mm) || !std::isfinite(focal_mm) ||
      focal_mm <= 0) {
    return std::nullopt;
  }
  float resolution;
  int unit;
  if (GetMetaData("Exif:FocalPlaneXResolution", "float", &resolution) &&
      GetMetaData("Exif:FocalPlaneResolutionUnit", "int", &unit) && std::isfinite(resolution) &&
      resolution > 0 && unit >= 2 && unit <= 5) {
    constexpr double units_per_mm[] = {0, 0, 1 / 25.4, 1 / 10.0, 1, 1000};
    return focal_mm * resolution * units_per_mm[unit];
  }
  const auto make = GetMetaData("Make");
  const auto model = GetMetaData("Model");
  if (make && model) {
    CameraDatabase database;
    double sensor_width;
    if (database.QuerySensorWidth(*make, *model, &sensor_width) && std::isfinite(sensor_width) &&
        sensor_width > 0) {
      return focal_mm / sensor_width * std::max(width_, height_);
    }
  }
  return std::nullopt;
}

std::optional<double> Bitmap::ExifLatitude() const {
  float dms[3];
  if (!GetMetaData("GPS:Latitude", "point", dms)) return std::nullopt;
  double value = dms[0] + dms[1] / 60.0 + dms[2] / 3600.0;
  if (!std::isfinite(value)) return std::nullopt;
  const auto reference = GetMetaData("GPS:LatitudeRef");
  if (reference && (*reference == "S" || *reference == "s") && value > 0) {
    value = -value;
  }
  return value;
}

std::optional<double> Bitmap::ExifLongitude() const {
  float dms[3];
  if (!GetMetaData("GPS:Longitude", "point", dms)) return std::nullopt;
  double value = dms[0] + dms[1] / 60.0 + dms[2] / 3600.0;
  if (!std::isfinite(value)) return std::nullopt;
  const auto reference = GetMetaData("GPS:LongitudeRef");
  if (reference && (*reference == "W" || *reference == "w") && value > 0) {
    value = -value;
  }
  return value;
}

std::optional<double> Bitmap::ExifAltitude() const {
  float altitude;
  if (!GetMetaData("GPS:Altitude", "float", &altitude) || !std::isfinite(altitude))
    return std::nullopt;
  const auto reference = GetMetaData("GPS:AltitudeRef");
  return reference && *reference == "1" ? -std::abs(double(altitude)) : altitude;
}

bool Bitmap::Read(const std::filesystem::path& path,
                  const bool as_rgb,
                  const bool linearize_colorspace) {
  @autoreleasepool {
    try {
      const std::string native_path = path.string();
      CFHandle<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(
          nullptr, reinterpret_cast<const UInt8*>(native_path.data()), native_path.size(), false));
      if (!url.Get()) return false;
      const void* keys[] = {kCGImageSourceShouldCache};
      const void* values[] = {kCFBooleanFalse};
      CFHandle<CFDictionaryRef> options(CFDictionaryCreate(nullptr,
                                                           keys,
                                                           values,
                                                           1,
                                                           &kCFTypeDictionaryKeyCallBacks,
                                                           &kCFTypeDictionaryValueCallBacks));
      CFHandle<CGImageSourceRef> source(CGImageSourceCreateWithURL(url.Get(), options.Get()));
      if (!source.Get() || CGImageSourceGetCount(source.Get()) == 0) return false;
      CFHandle<CFDictionaryRef> properties(
          CGImageSourceCopyPropertiesAtIndex(source.Get(), 0, options.Get()));
      const auto width = NumberValue(DictionaryValue(properties.Get(), kCGImagePropertyPixelWidth));
      const auto height =
          NumberValue(DictionaryValue(properties.Get(), kCGImagePropertyPixelHeight));
      if (!width || !height || *width < 1 || *height < 1 || *width > kMaximumImageDimension ||
          *height > kMaximumImageDimension || std::floor(*width) != *width ||
          std::floor(*height) != *height) {
        return false;
      }
      const int w = static_cast<int>(*width);
      const int h = static_cast<int>(*height);
      const size_t rgba_bytes = CheckedBytes(w, h, 4);
      // CreateImageAtIndex returns the encoded raster. Do not use a transformed
      // thumbnail or apply EXIF orientation: calibration addresses these pixels.
      CFHandle<CGImageRef> image(CGImageSourceCreateImageAtIndex(source.Get(), 0, options.Get()));
      if (!image.Get() || CGImageGetWidth(image.Get()) != size_t(w) ||
          CGImageGetHeight(image.Get()) != size_t(h) ||
          CGImageSourceGetStatusAtIndex(source.Get(), 0) != kCGImageStatusComplete) {
        return false;
      }
      std::vector<uint8_t> rgba(rgba_bytes);
      CFHandle<CGColorSpaceRef> colorspace(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
      CFHandle<CGContextRef> context(
          CGBitmapContextCreate(rgba.data(),
                                w,
                                h,
                                8,
                                static_cast<size_t>(w) * 4,
                                colorspace.Get(),
                                kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast));
      if (!context.Get()) return false;
      CGContextSetBlendMode(context.Get(), kCGBlendModeCopy);
      CGContextSetInterpolationQuality(context.Get(), kCGInterpolationNone);
      CGContextDrawImage(context.Get(), CGRectMake(0, 0, w, h), image.Get());
      Bitmap result(w, h, as_rgb);
      const auto linear_table = SRGBTransferTable(true);
      for (size_t pixel = 0; pixel < static_cast<size_t>(w) * h; ++pixel) {
        const int alpha = rgba[pixel * 4 + 3];
        uint8_t color[3];
        for (int channel = 0; channel < 3; ++channel) {
          // Drop alpha after recovering straight RGB. A fully transparent
          // source pixel has no recoverable color after color-managed drawing.
          color[channel] =
              alpha == 0 ? 0
                         : ByteValue(static_cast<double>(rgba[pixel * 4 + channel]) * 255 / alpha);
          if (linearize_colorspace) color[channel] = linear_table[color[channel]];
        }
        if (as_rgb) {
          std::copy_n(color, 3, result.data_.begin() + pixel * 3);
        } else {
          result.data_[pixel] =
              static_cast<uint8_t>(.2126f * color[0] + .7152f * color[1] + .0722f * color[2] + .5f);
        }
      }
      static_cast<AppleMetaData*>(result.meta_data_.get())->source_properties =
          properties.Release();
      ImportMetadata(static_cast<AppleMetaData*>(result.meta_data_.get())->source_properties,
                     result);
      result.linear_colorspace_ = linearize_colorspace;
      result.SetMetaData("oiio:ColorSpace", linearize_colorspace ? "linear" : "sRGB");
      *this = std::move(result);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
}

bool Bitmap::Write(const std::filesystem::path& path, const bool delinearize_colorspace) const {
  @autoreleasepool {
    try {
      CheckStorage(*this);
      std::string extension = path.extension().string();
      std::transform(
          extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
            return std::tolower(value);
          });
      const bool jpeg = extension == ".jpg" || extension == ".jpeg";
      if (!jpeg && extension != ".png") return false;
      const std::string native_path = path.string();
      CFHandle<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(
          nullptr, reinterpret_cast<const UInt8*>(native_path.data()), native_path.size(), false));
      if (!url.Get()) return false;
      CFHandle<CGImageDestinationRef> destination(CGImageDestinationCreateWithURL(
          url.Get(), jpeg ? CFSTR("public.jpeg") : CFSTR("public.png"), 1, nullptr));
      if (!destination.Get()) return false;
      Bitmap output = CloneAsRGB();
      const bool write_linear = linear_colorspace_ && !delinearize_colorspace;
      if (linear_colorspace_ && delinearize_colorspace) {
        ConvertSRGB(output.data_, false);
      }
      CFHandle<CGColorSpaceRef> colorspace(
          CGColorSpaceCreateWithName(write_linear ? kCGColorSpaceLinearSRGB : kCGColorSpaceSRGB));
      CFHandle<CGDataProviderRef> provider(
          CGDataProviderCreateWithData(nullptr, output.data_.data(), output.data_.size(), nullptr));
      CFHandle<CGImageRef> image(CGImageCreate(width_,
                                               height_,
                                               8,
                                               24,
                                               static_cast<size_t>(width_) * 3,
                                               colorspace.Get(),
                                               kCGImageAlphaNone,
                                               provider.Get(),
                                               nullptr,
                                               false,
                                               kCGRenderingIntentDefault));
      if (!image.Get()) return false;
      const auto* metadata = static_cast<const AppleMetaData*>(meta_data_.get());
      CFHandle<CFMutableDictionaryRef> properties(
          metadata && metadata->source_properties
              ? CFDictionaryCreateMutableCopy(nullptr, 0, metadata->source_properties)
              : CFDictionaryCreateMutable(
                    nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
      if (!properties.Get()) return false;
      // The CGImage owns the output color profile; never label converted sRGB
      // bytes with the source image's Display P3 or grayscale profile name.
      CFDictionaryRemoveValue(properties.Get(), kCGImagePropertyProfileName);
      CFDictionaryRemoveValue(properties.Get(), kCGImagePropertyColorModel);
      ExportMetadata(*this, properties.Get());
      int quality = 100;
      if (const auto compression = GetMetaData("Compression");
          compression && compression->rfind("jpeg:", 0) == 0) {
        const std::string suffix = compression->substr(5);
        size_t consumed;
        quality = std::stoi(suffix, &consumed);
        if (consumed != suffix.size() || quality < 1 || quality > 100) return false;
      }
      if (jpeg) {
        SetNumber(properties.Get(), kCGImageDestinationLossyCompressionQuality, quality / 100.0);
      }
      CGImageDestinationAddImage(destination.Get(), image.Get(), properties.Get());
      return CGImageDestinationFinalize(destination.Get());
    } catch (const std::exception&) {
      return false;
    }
  }
}

void Bitmap::Rescale(const int new_width, const int new_height, const RescaleFilter filter) {
  CheckStorage(*this);
  CheckedBytes(new_width, new_height, channels_);
  if (filter != RescaleFilter::kBilinear && filter != RescaleFilter::kBox &&
      filter != RescaleFilter::kHighQuality) {
    throw std::invalid_argument("Unknown bitmap resize filter");
  }
  if (new_width == width_ && new_height == height_) return;
  std::vector<uint8_t> output;
  if (filter == RescaleFilter::kHighQuality) {
    output = RescaleHighQuality(*this, new_width, new_height);
  } else if (static_cast<size_t>(new_width) * height_ <= static_cast<size_t>(width_) * new_height) {
    auto intermediate = RescaleAxis(data_, width_, height_, channels_, new_width, true, filter);
    output = RescaleAxis(intermediate, new_width, height_, channels_, new_height, false, filter);
  } else {
    auto intermediate = RescaleAxis(data_, width_, height_, channels_, new_height, false, filter);
    output = RescaleAxis(intermediate, width_, new_height, channels_, new_width, true, filter);
  }
  width_ = new_width;
  height_ = new_height;
  data_ = std::move(output);
}

double Bitmap::Thumbnail(const int max_image_size, const RescaleFilter filter) {
  if (max_image_size <= 0) {
    throw std::invalid_argument("Thumbnail size must be positive");
  }
  if (width_ <= max_image_size && height_ <= max_image_size) return 1;
  const double scale = static_cast<double>(max_image_size) / std::max(width_, height_);
  Rescale(std::max(1, static_cast<int>(std::round(width_ * scale))),
          std::max(1, static_cast<int>(std::round(height_ * scale))),
          filter);
  return scale;
}

void Bitmap::Rot90(int k) {
  if (IsEmpty()) return;
  CheckStorage(*this);
  k = (k % 4 + 4) % 4;
  if (k == 0) return;
  const int width = k % 2 ? height_ : width_;
  const int height = k % 2 ? width_ : height_;
  std::vector<uint8_t> output(data_.size());
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const int target_x = k == 1 ? y : k == 2 ? width_ - 1 - x : height_ - 1 - y;
      const int target_y = k == 1 ? width_ - 1 - x : k == 2 ? height_ - 1 - y : x;
      std::memcpy(&output[(static_cast<size_t>(target_y) * width + target_x) * channels_],
                  &data_[(static_cast<size_t>(y) * width_ + x) * channels_],
                  channels_);
    }
  }
  width_ = width;
  height_ = height;
  data_ = std::move(output);
}

Bitmap Bitmap::Clone() const { return *this; }

Bitmap Bitmap::CloneAsGrey() const {
  if (IsEmpty() || IsGrey()) return Clone();
  CheckStorage(*this);
  Bitmap result(width_, height_, false, linear_colorspace_);
  for (size_t pixel = 0; pixel < result.data_.size(); ++pixel) {
    result.data_[pixel] =
        static_cast<uint8_t>(.2126f * data_[pixel * 3] + .7152f * data_[pixel * 3 + 1] +
                             .0722f * data_[pixel * 3 + 2] + .5f);
  }
  result.meta_data_ = CopyMetadata(meta_data_);
  return result;
}

Bitmap Bitmap::CloneAsRGB() const {
  if (IsEmpty() || IsRGB()) return Clone();
  CheckStorage(*this);
  Bitmap result(width_, height_, true, linear_colorspace_);
  for (size_t pixel = 0; pixel < data_.size(); ++pixel) {
    std::fill_n(result.data_.begin() + pixel * 3, 3, data_[pixel]);
  }
  result.meta_data_ = CopyMetadata(meta_data_);
  return result;
}

void Bitmap::SetJpegQuality(const int quality) {
  if (quality < 1 || quality > 100) {
    throw std::invalid_argument("JPEG quality must be in 1...100");
  }
  SetMetaData("Compression", "jpeg:" + std::to_string(quality));
}

void Bitmap::SetMetaData(const std::string_view& name,
                         const std::string_view& type,
                         const void* value) {
  if (!value) throw std::invalid_argument("Metadata value must not be null");
  const size_t bytes = MetadataTypeSize(type);
  if (!meta_data_) meta_data_ = std::make_unique<AppleMetaData>();
  MetaValue attribute;
  attribute.type = type;
  attribute.bytes.resize(bytes);
  std::memcpy(attribute.bytes.data(), value, bytes);
  static_cast<AppleMetaData*>(meta_data_.get())->values[std::string(name)] = std::move(attribute);
}

void Bitmap::SetMetaData(const std::string_view& name, const std::string_view& value) {
  if (!meta_data_) meta_data_ = std::make_unique<AppleMetaData>();
  MetaValue attribute;
  attribute.type = "string";
  attribute.string = value;
  static_cast<AppleMetaData*>(meta_data_.get())->values[std::string(name)] = std::move(attribute);
}

bool Bitmap::GetMetaData(const std::string_view& name,
                         const std::string_view& type,
                         void* value) const {
  const size_t bytes = MetadataTypeSize(type);
  if (!meta_data_ || !value) return false;
  const auto& attributes = static_cast<const AppleMetaData*>(meta_data_.get())->values;
  const auto iterator = attributes.find(std::string(name));
  if (iterator == attributes.end() || iterator->second.type != type) return false;
  std::memcpy(value, iterator->second.bytes.data(), bytes);
  return true;
}

std::optional<std::string> Bitmap::GetMetaData(const std::string_view& name) const {
  if (!meta_data_) return std::nullopt;
  const auto& attributes = static_cast<const AppleMetaData*>(meta_data_.get())->values;
  const auto iterator = attributes.find(std::string(name));
  if (iterator == attributes.end() || iterator->second.type != "string") {
    return std::nullopt;
  }
  return iterator->second.string;
}

void Bitmap::CloneMetadata(Bitmap* target) const {
  if (!target) throw std::invalid_argument("Metadata target must not be null");
  target->meta_data_ = CopyMetadata(meta_data_);
}

std::ostream& operator<<(std::ostream& stream, const Bitmap& bitmap) {
  return stream << "Bitmap(width=" << bitmap.Width() << ", height=" << bitmap.Height()
                << ", channels=" << bitmap.Channels() << ")";
}

float JetColormap::Red(const float gray) { return Base(gray - 0.25f); }
float JetColormap::Green(const float gray) { return Base(gray); }
float JetColormap::Blue(const float gray) { return Base(gray + 0.25f); }

float JetColormap::Base(const float value) {
  if (value <= 0.125f) return 0;
  if (value <= 0.375f) return Interpolate(2 * value - 1, 0, -0.75f, 1, -0.25f);
  if (value <= 0.625f) return 1;
  if (value <= 0.87f) return Interpolate(2 * value - 1, 1, 0.25f, 0, 0.75f);
  return 0;
}

float JetColormap::Interpolate(
    const float value, const float y0, const float x0, const float y1, const float x1) {
  return (value - x0) * (y1 - y0) / (x1 - x0) + y0;
}

}  // namespace colmap
