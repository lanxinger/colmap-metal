#include "colmap/sensor/bitmap.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

namespace {

using colmap::Bitmap;
using colmap::BitmapColor;
using Color = BitmapColor<uint8_t>;

void Require(const bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void Near(const double actual,
          const double expected,
          const double tolerance,
          const std::string& message) {
  Require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          message + ": actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}

template <typename T>
class CFHandle {
 public:
  explicit CFHandle(T value) : value_(value) { Require(value, "Create CF object"); }
  ~CFHandle() { CFRelease(value_); }
  T Get() const { return value_; }
  CFHandle(const CFHandle&) = delete;
  CFHandle& operator=(const CFHandle&) = delete;

 private:
  T value_;
};

struct TestDirectory {
  std::filesystem::path path;
  TestDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "colmap-apple-bitmap-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    Require(mkdtemp(buffer.data()), "Create test directory");
    path = buffer.data();
  }
  ~TestDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

// Independent ImageIO fixture writer. It never calls Bitmap::Write, so a
// matching orientation or channel-order bug in Read/Write cannot pass this test.
void WriteFixture(const std::filesystem::path& path,
                  const int width,
                  const int height,
                  const std::vector<uint8_t>& pixels,
                  const int orientation = 1,
                  const bool display_p3 = false) {
  Require(pixels.size() == static_cast<size_t>(width) * height * 3, "Fixture pixel count");
  CFHandle<CGColorSpaceRef> color(
      CGColorSpaceCreateWithName(display_p3 ? kCGColorSpaceDisplayP3 : kCGColorSpaceSRGB));
  CFHandle<CGDataProviderRef> provider(
      CGDataProviderCreateWithData(nullptr, pixels.data(), pixels.size(), nullptr));
  CFHandle<CGImageRef> image(CGImageCreate(width,
                                           height,
                                           8,
                                           24,
                                           width * 3,
                                           color.Get(),
                                           kCGImageAlphaNone,
                                           provider.Get(),
                                           nullptr,
                                           false,
                                           kCGRenderingIntentDefault));
  const std::string filename = path.string();
  CFHandle<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(
      nullptr, reinterpret_cast<const UInt8*>(filename.data()), filename.size(), false));
  const bool jpeg = path.extension() == ".jpg";
  CFHandle<CGImageDestinationRef> destination(CGImageDestinationCreateWithURL(
      url.Get(), jpeg ? CFSTR("public.jpeg") : CFSTR("public.png"), 1, nullptr));
  CFHandle<CFNumberRef> orientation_number(CFNumberCreate(nullptr, kCFNumberIntType, &orientation));
  const double quality = 1;
  CFHandle<CFNumberRef> quality_number(CFNumberCreate(nullptr, kCFNumberDoubleType, &quality));
  const void* keys[] = {kCGImagePropertyOrientation, kCGImageDestinationLossyCompressionQuality};
  const void* values[] = {orientation_number.Get(), quality_number.Get()};
  CFHandle<CFDictionaryRef> properties(CFDictionaryCreate(nullptr,
                                                          keys,
                                                          values,
                                                          jpeg ? 2 : 1,
                                                          &kCFTypeDictionaryKeyCallBacks,
                                                          &kCFTypeDictionaryValueCallBacks));
  CGImageDestinationAddImage(destination.Get(), image.Get(), properties.Get());
  Require(CGImageDestinationFinalize(destination.Get()), "Write ImageIO fixture");
}

std::vector<uint8_t> Quadrants(const int width, const int height) {
  const Color colors[] = {
      Color(241, 16, 27), Color(28, 211, 34), Color(21, 33, 230), Color(213, 190, 50)};
  std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const Color color = colors[(y >= height / 2) * 2 + (x >= width / 2)];
      const size_t index = (static_cast<size_t>(y) * width + x) * 3;
      pixels[index] = color.r;
      pixels[index + 1] = color.g;
      pixels[index + 2] = color.b;
    }
  }
  return pixels;
}

void CheckQuadrants(const Bitmap& bitmap, const int tolerance) {
  const Color colors[] = {
      Color(241, 16, 27), Color(28, 211, 34), Color(21, 33, 230), Color(213, 190, 50)};
  for (int index = 0; index < 4; ++index) {
    const int x = index % 2 ? bitmap.Width() - 6 : 5;
    const int y = index / 2 ? bitmap.Height() - 6 : 5;
    const Color color = bitmap.GetPixel(x, y).value();
    Near(color.r, colors[index].r, tolerance, "Quadrant red");
    Near(color.g, colors[index].g, tolerance, "Quadrant green");
    Near(color.b, colors[index].b, tolerance, "Quadrant blue");
  }
}

void EncodedExifPortrait() {
  TestDirectory directory;
  for (const auto& [width, height] : {std::make_pair(48, 80), std::make_pair(80, 48)}) {
    const auto path = directory.path / (std::to_string(width) + ".jpg");
    WriteFixture(path, width, height, Quadrants(width, height), 6);
    Bitmap bitmap;
    Require(bitmap.Read(path), "Read EXIF orientation 6 JPEG");
    Require(bitmap.Width() == width && bitmap.Height() == height,
            "EXIF orientation must not change encoded raster dimensions");
    Require(bitmap.ExifOrientation() == 6, "Retain orientation separately");
    CheckQuadrants(bitmap, 4);
    const auto round_trip = directory.path / (std::to_string(width) + "-copy.png");
    Require(bitmap.Write(round_trip), "Write decoded JPEG as PNG");
    Bitmap reread;
    Require(reread.Read(round_trip), "Read round-trip PNG");
    Require(reread.Width() == width && reread.Height() == height, "Round-trip encoded dimensions");
    Require(reread.RowMajorData() == bitmap.RowMajorData(), "Lossless PNG pixels");
    Require(reread.ExifOrientation() == 6, "Retain EXIF through PNG export");
  }
}

void ColorsAndLinearization() {
  TestDirectory directory;
  const auto png = directory.path / "colors.png";
  const std::vector<uint8_t> pixels = {
      255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 128, 128, 128, 10, 20, 30};
  WriteFixture(png, 6, 1, pixels);
  Bitmap rgb;
  Require(rgb.Read(png), "Read primary-color PNG");
  Require(rgb.RowMajorData() == pixels, "sRGB primary colors and channel order");
  Require(rgb.GetMetaData("oiio:ColorSpace") == "sRGB", "Explicit sRGB metadata");
  Bitmap grey;
  Require(grey.Read(png, false), "Read as grayscale");
  Require(grey.RowMajorData() == std::vector<uint8_t>({54, 182, 18, 255, 128, 19}),
          "COLMAP Rec.709 grayscale weights and rounding");
  Require(grey.RowMajorData() == rgb.CloneAsGrey().RowMajorData(),
          "Read/clone grayscale agreement");
  const auto grey_path = directory.path / "grey.png";
  Require(grey.Write(grey_path), "Write gray PNG");
  Bitmap reread;
  Require(reread.Read(grey_path, false), "Read gray PNG again");
  Require(reread.RowMajorData() == grey.RowMajorData(), "Gray PNG round trip");

  Bitmap linear;
  Require(linear.Read(png, true, true), "Read into linear sRGB");
  Near(linear.GetPixel(4, 0)->r, 55, 0, "sRGB 128 linearizes to 55/255");
  const auto linear_path = directory.path / "linear.png";
  Require(linear.Write(linear_path), "Delinearize on write");
  Require(reread.Read(linear_path), "Read delinearized image");
  Near(reread.GetPixel(4, 0)->r, 128, 1, "Linear/sRGB round trip");

  const auto p3_path = directory.path / "display-p3.png";
  WriteFixture(p3_path, 1, 1, {180, 100, 80}, 1, true);
  Require(reread.Read(p3_path), "Read Display P3 image");
  const Color converted = reread.GetPixel(0, 0).value();
  // The P3 fixture is intentionally inside the sRGB gamut but its encoded
  // values differ. Dropping its profile instead of converting cannot pass.
  Require(converted.r > 185 && converted.g < 99 && converted.b < 78,
          "Convert Display P3 channels into sRGB");
}

void MetadataAndCopies() {
  Bitmap bitmap(100, 80, true);
  bitmap.Fill(Color(10, 20, 30));
  const int orientation = 6;
  const float focal = 24;
  const float latitude[] = {52, 22, 12};
  const float longitude[] = {4, 54, 0};
  const float altitude = 12.5;
  bitmap.SetMetaData("Orientation", "int", &orientation);
  bitmap.SetMetaData("Make", "Apple");
  bitmap.SetMetaData("Model", "Camera");
  bitmap.SetMetaData("Exif:FocalLengthIn35mmFilm", "float", &focal);
  bitmap.SetMetaData("GPS:Latitude", "point", latitude);
  bitmap.SetMetaData("GPS:LatitudeRef", "N");
  bitmap.SetMetaData("GPS:Longitude", "point", longitude);
  bitmap.SetMetaData("GPS:LongitudeRef", "E");
  bitmap.SetMetaData("GPS:Altitude", "float", &altitude);
  bitmap.SetMetaData("GPS:AltitudeRef", "1");
  bitmap.SetMetaData("custom", "float", &focal);
  Require(bitmap.ExifCameraModel() == "Apple-Camera-24.000000-100x80",
          "Camera identity from metadata");
  Near(bitmap.ExifFocalLength().value(),
       24 / 43.27 * std::hypot(100, 80),
       1e-5,
       "35mm focal equivalent in pixels");
  Near(bitmap.ExifLatitude().value(), 52.37, 1e-5, "Latitude");
  Near(bitmap.ExifLongitude().value(), 4.9, 1e-5, "Longitude");
  Near(bitmap.ExifAltitude().value(), -12.5, 1e-5, "Signed altitude");
  float value = 0;
  Require(bitmap.GetMetaData("custom", "float", &value) && value == focal, "Typed metadata value");
  Require(!bitmap.GetMetaData("custom", "int8", &value), "Metadata type mismatch");
  Bitmap copied = bitmap.Clone();
  copied.SetPixel(0, 0, Color(255));
  copied.SetMetaData("Model", "Different");
  Require(bitmap.GetPixel(0, 0).value() == Color(10, 20, 30), "Deep pixel copy");
  Require(bitmap.GetMetaData("Model") == "Camera", "Deep metadata copy");
  Bitmap target(2, 1, false);
  bitmap.CloneMetadata(&target);
  Require(target.Width() == 2 && target.Height() == 1 && target.IsGrey(),
          "CloneMetadata preserves target geometry");
  Require(target.ExifOrientation() == 6, "CloneMetadata copies EXIF");
  Bitmap moved(std::move(copied));
  Require(copied.IsEmpty() && copied.Width() == 0 && moved.Width() == 100,
          "Move leaves empty source");
  copied = Bitmap();
  Require(!copied.ExifOrientation(), "Empty metadata query");
  copied = bitmap;
  copied = copied;
  Require(copied.RowMajorData() == bitmap.RowMajorData(), "Copy assignment and self-copy");

  TestDirectory directory;
  bitmap.SetJpegQuality(93);
  const auto path = directory.path / "metadata.jpg";
  Require(bitmap.Write(path), "Write JPEG metadata");
  Bitmap read;
  Require(read.Read(path), "Read JPEG metadata");
  Require(read.Width() == 100 && read.Height() == 80, "JPEG dimensions");
  Require(read.ExifOrientation() == orientation, "JPEG integer orientation tag");
  Require(read.ExifCameraModel() == bitmap.ExifCameraModel(),
          "Camera metadata round trip: actual=" + read.ExifCameraModel().value_or("missing") +
              " expected=" + bitmap.ExifCameraModel().value_or("missing") +
              " make=" + read.GetMetaData("Make").value_or("missing") +
              " model=" + read.GetMetaData("Model").value_or("missing"));
  Near(read.ExifLatitude().value(), 52.37, 1e-4, "JPEG GPS latitude");
  Near(read.ExifAltitude().value(), -12.5, 1e-5, "JPEG GPS altitude");

  Bitmap database(100, 80, false);
  const float focal_mm = 72;
  database.SetMetaData("Exif:FocalLength", "float", &focal_mm);
  database.SetMetaData("Make", "canon");
  database.SetMetaData("Model", "eos1dsmarkiii");
  Near(database.ExifFocalLength().value(), 200, 1e-5, "Camera sensor database fallback");
}

void RotationAndResizing() {
  Bitmap bitmap(3, 2, false);
  bitmap.RowMajorData() = {1, 2, 3, 4, 5, 6};
  Bitmap rotated = bitmap.Clone();
  rotated.Rot90(1);
  Require(rotated.Width() == 2 && rotated.Height() == 3, "CCW rotation dimensions");
  Require(rotated.RowMajorData() == std::vector<uint8_t>({3, 6, 2, 5, 1, 4}),
          "CCW rotation pixel coordinates");
  rotated.Rot90(-1);
  Require(rotated.RowMajorData() == bitmap.RowMajorData(), "Inverse rotation");
  rotated.Rot90(2);
  Require(rotated.RowMajorData() == std::vector<uint8_t>({6, 5, 4, 3, 2, 1}),
          "Half-turn pixel coordinates");

  Bitmap ramp(2, 1, false);
  ramp.RowMajorData() = {0, 100};
  ramp.Rescale(4, 1, Bitmap::RescaleFilter::kBilinear);
  Require(ramp.RowMajorData() == std::vector<uint8_t>({0, 25, 75, 100}),
          "Pixel-center map for bilinear enlargement");
  Bitmap stripe(4, 1, false);
  stripe.RowMajorData() = {0, 255, 0, 255};
  Bitmap triangle = stripe.Clone();
  triangle.Rescale(2, 1, Bitmap::RescaleFilter::kBilinear);
  Require(triangle.RowMajorData() == std::vector<uint8_t>({96, 159}),
          "Triangle reduction anti-aliasing and clamped boundaries");
  stripe.Rescale(2, 1, Bitmap::RescaleFilter::kBox);
  Require(stripe.RowMajorData() == std::vector<uint8_t>({128, 128}),
          "Box reduction integrates pixel areas");
  for (const auto filter : {Bitmap::RescaleFilter::kBilinear,
                            Bitmap::RescaleFilter::kBox,
                            Bitmap::RescaleFilter::kHighQuality}) {
    Bitmap constant(13, 9, true);
    constant.Fill(Color(30, 90, 170));
    constant.Rescale(4, 6, filter);
    for (int y = 0; y < constant.Height(); ++y) {
      for (int x = 0; x < constant.Width(); ++x) {
        Require(constant.GetPixel(x, y).value() == Color(30, 90, 170),
                "Resize preserves constant RGB values");
      }
    }
  }
  Bitmap high_quality(7, 1, false);
  high_quality.RowMajorData() = {0, 20, 40, 60, 80, 100, 120};
  high_quality.Rescale(14, 1, Bitmap::RescaleFilter::kHighQuality);
  Near(high_quality.GetPixel(6, 0)->r, 55, 2, "Lanczos center-coordinate map");
  Bitmap thumbnail(16, 8, false);
  Near(thumbnail.Thumbnail(5), 5 / 16.0, 1e-12, "Thumbnail scale");
  Require(thumbnail.Width() == 5 && thumbnail.Height() == 3,
          "Thumbnail rounds dimensions independently");
  Near(thumbnail.Thumbnail(8), 1, 0, "Thumbnail does not enlarge");
  Bitmap narrow(100, 1, false);
  narrow.Thumbnail(1);
  Require(narrow.Width() == 1 && narrow.Height() == 1, "Thumbnail retains one-pixel axis");
}

uint32_t PngCRC(const uint8_t* bytes, const size_t count) {
  uint32_t crc = 0xffffffff;
  for (size_t index = 0; index < count; ++index) {
    crc ^= bytes[index];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
  }
  return crc ^ 0xffffffff;
}

void FailurePaths() {
  TestDirectory directory;
  Bitmap bitmap(2, 1, true);
  bitmap.Fill(Color(7, 8, 9));
  const auto original = bitmap.RowMajorData();
  Require(!bitmap.Read(directory.path / "missing.png"), "Missing image returns false");
  Require(bitmap.RowMajorData() == original && bitmap.Width() == 2,
          "Failed read preserves original image");
  const auto corrupt = directory.path / "corrupt.jpg";
  std::ofstream(corrupt) << "not an image";
  Require(!bitmap.Read(corrupt), "Corrupt image returns false");
  Require(!bitmap.Write(directory.path / "unsupported.bmp"), "Unsupported writer returns false");
  Require(!Bitmap().Write(directory.path / "empty.png"), "Empty writer returns false");
  int exceptions = 0;
  for (const auto& action :
       std::vector<std::function<void()>>{[] { Bitmap bad(32769, 1, false); },
                                          [] { Bitmap bad(16384, 16384, true); },
                                          [&] { bitmap.Rescale(0, 1); },
                                          [&] { bitmap.Thumbnail(0); },
                                          [&] { bitmap.SetJpegQuality(0); },
                                          [&] { bitmap.CloneMetadata(nullptr); }}) {
    try {
      action();
    } catch (const std::invalid_argument&) {
      ++exceptions;
    }
  }
  Require(exceptions == 6, "Invalid operations throw instead of terminating");
  Require(bitmap.RowMajorData() == original, "Invalid mutations preserve data");

  const auto oversized = directory.path / "oversized.png";
  WriteFixture(oversized, 1, 1, {0, 0, 0});
  std::ifstream source(oversized, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(source)), {});
  Require(bytes.size() > 33, "PNG IHDR fixture");
  // Change the encoded width to 32769 and repair the IHDR checksum. ImageIO
  // can inspect this header; the decoder bound must reject before pixel work.
  bytes[16] = 0;
  bytes[17] = 0;
  bytes[18] = 0x80;
  bytes[19] = 1;
  const uint32_t crc = PngCRC(bytes.data() + 12, 17);
  for (int index = 0; index < 4; ++index) bytes[29 + index] = crc >> (24 - index * 8);
  std::ofstream output(oversized, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  output.close();
  Require(!bitmap.Read(oversized), "Oversized encoded dimensions rejected");
  Require(bitmap.RowMajorData() == original, "Oversized read is transactional");
}

}  // namespace

int main() {
  @autoreleasepool {
    const std::array<std::pair<const char*, void (*)()>, 5> tests = {{
        {"encoded EXIF portrait", EncodedExifPortrait},
        {"color space and gray conversion", ColorsAndLinearization},
        {"metadata, clone, JPEG round trip", MetadataAndCopies},
        {"rotation and resize coordinates", RotationAndResizing},
        {"recoverable failure paths", FailurePaths},
    }};
    int failed = 0;
    for (const auto& [name, test] : tests) {
      try {
        test();
        std::printf("PASS %s\n", name);
      } catch (const std::exception& error) {
        ++failed;
        std::fprintf(stderr, "FAIL %s: %s\n", name, error.what());
      }
    }
    std::printf("Apple bitmap tests: %zu passed, %d failed\n", tests.size() - failed, failed);
    return failed == 0 ? 0 : 1;
  }
}
