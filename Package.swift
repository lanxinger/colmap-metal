// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "ColmapSparse",
  platforms: [.iOS(.v18), .macOS(.v13)],
  products: [.library(name: "ColmapSparse", targets: ["ColmapSparse"])],
  targets: [
    .binaryTarget(name: "ColmapSparseNative", path: "Artifacts/ColmapSparse.xcframework"),
    .target(
      name: "ColmapSparse",
      dependencies: ["ColmapSparseNative"],
      path: "ios",
      exclude: ["CMakeLists.txt", "README.md", "include", "src", "tests"],
      sources: ["Sources/ColmapSparse/ColmapSparse.swift"],
      resources: [
        // Copy each platform directory separately. A top-level Resources
        // folder makes codesign reject the iOS resource bundle.
        .copy("Sources/ColmapSparse/Resources/iphoneos"),
        .copy("Sources/ColmapSparse/Resources/iphonesimulator"),
        .copy("Sources/ColmapSparse/Resources/macosx"),
        .copy("Licenses"),
        .copy("THIRD_PARTY_NOTICES.md"),
      ],
      linkerSettings: [
        .linkedLibrary("c++"),
        .linkedLibrary("sqlite3"),
        .linkedFramework("Foundation"),
        .linkedFramework("CoreGraphics"),
        .linkedFramework("ImageIO"),
        .linkedFramework("Accelerate"),
        .linkedFramework("Metal"),
      ]
    ),
    .testTarget(
      name: "ColmapSparseTests",
      dependencies: ["ColmapSparse"],
      path: "ios/tests/ColmapSparseTests"
    ),
  ]
)
