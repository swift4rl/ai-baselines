// swift-tools-version:5.3
import PackageDescription

let package = Package(
    name: "ai-baselines",
    platforms: [
        .macOS(.v10_15),
    ],
    products: [
        .library(name: "environments", targets: ["environments"]),
        .executable(name: "Run", targets: ["Run"])
    ],
    dependencies: [
        .package(name: "grpc-swift", url: "https://github.com/grpc/grpc-swift.git",  .exact("1.0.0-alpha.18")),
        .package(name: "Version", url: "https://github.com/mrackwitz/Version.git", .exact("0.8.0")),
        .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master")),
        .package(name: "swift-models", url: "https://github.com/swift4rl/swift-models.git", .branch("master"))
    ],
    targets: [
        .target(
            name: "environments",
            dependencies: [
                .product(name: "GRPC", package: "grpc-swift"),
                .product(name: "Version", package: "Version"),
            ]
        ),
        .target(
            name: "Run",
            dependencies: [
                "environments",
                .product(name: "Gym", package: "swift-models"),
            ]
        ),        
        .testTarget(
            name: "TestEnvironments",
            dependencies: ["environments"]
        )
    ]
)
