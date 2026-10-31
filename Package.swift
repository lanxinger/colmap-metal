// swift-tools-version: 6.0
import PackageDescription
import Foundation

// Native development and CI can opt in to the locally rebuilt XCFramework.
let nativeTarget: Target = ProcessInfo.processInfo.environment["COLMAP_SPARSE_LOCAL_BUILD"] == "1"
  ? .binaryTarget(name: "ColmapSparseNative", path: "Artifacts/ColmapSparse.xcframework")
  : .binaryTarget(
      name: "ColmapSparseNative",
      url: "https://github.com/lanxinger/colmap-metal/releases/download/0.1.0/ColmapSparseNative-0.1.0.xcframework.zip",
      checksum: "f92d6bba61269759da67f9f897803ab061905d74a104cd98fc575d719014ba9b"
    )

let package = Package(
  name: "ColmapSparse",
  platforms: [.iOS(.v18), .macOS(.v13)],
  products: [.library(name: "ColmapSparse", targets: ["ColmapSparse"])],
  targets: [
    nativeTarget,
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
