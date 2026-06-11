use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell Cargo to rerun if Swift files change
    println!("cargo:rerun-if-changed=swift/");
    println!("cargo:rerun-if-changed=Package.swift");
    println!("cargo:rerun-if-env-changed=TARGET");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let cargo_target = std::env::var("TARGET").unwrap_or_default();

    // Build the Swift package first to get FluidAudio dependency
    println!(
        "cargo:warning=Building Swift package for {}...",
        cargo_target
    );

    let swift_build_dir = out_dir.join("swift-build");
    std::fs::create_dir_all(&swift_build_dir).expect("Failed to create swift-build directory");

    // SwiftPM defaults to host triple when not given --triple, so iOS
    // cross-builds silently produce a macOS bridge lib that won't link into
    // the iOS binary. Map cargo TARGET → SwiftPM `--triple` explicitly. For
    // iOS targets we additionally pass `--sdk <iphoneos|iphonesimulator path>`
    // so swiftc can find the Swift standard library for that platform.
    //
    // We do NOT wrap with `xcrun -sdk <name> swift build`: that exports
    // SDKROOT into the manifest-compile stage too, which still runs at the
    // host triple (arm64-apple-macosx*) and would then receive an iOS SDK,
    // breaking the manifest with a sysroot/target mismatch.
    let swift_triple = cargo_to_swift_triple(&cargo_target);
    let sdk_path = cargo_to_xcrun_sdk(&cargo_target).and_then(|sdk_name| {
        let out = Command::new("xcrun")
            .args(["--sdk", sdk_name, "--show-sdk-path"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let path = String::from_utf8(out.stdout).ok()?.trim().to_owned();
        if path.is_empty() {
            None
        } else {
            Some(path)
        }
    });

    let mut swift_args: Vec<String> = vec![
        "build".into(),
        "-c".into(),
        "release".into(),
        "--build-path".into(),
        swift_build_dir.to_str().unwrap().into(),
    ];
    if let Some(triple) = swift_triple.as_deref() {
        swift_args.push("--triple".into());
        swift_args.push(triple.into());
    }
    if let Some(sdk) = sdk_path.as_deref() {
        swift_args.push("--sdk".into());
        swift_args.push(sdk.into());
    }

    let status = Command::new("swift")
        .args(&swift_args)
        .current_dir(&manifest_dir)
        .status()
        .expect("Failed to run swift build");

    if !status.success() {
        panic!("Swift package build failed");
    }

    // SwiftPM places artefacts under <build-path>/<triple>/release/ — for
    // both host and cross builds. The host case lands in e.g.
    // `arm64-apple-macosx/release/`, the iOS case in `arm64-apple-ios/release/`
    // (the deployment-version suffix is stripped). Walk swift_build_dir to
    // find libFluidAudioBridge.a wherever SwiftPM put it, rather than guessing.
    let lib_path = find_bridge_lib_dir(&swift_build_dir).unwrap_or_else(|| {
        panic!(
            "libFluidAudioBridge.a not found under {}; swift build may have failed silently",
            swift_build_dir.display()
        )
    });

    // Link the Swift library
    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=static=FluidAudioBridge");

    // Link Apple frameworks (available on both macOS and iOS)
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // Link Swift runtime
    println!("cargo:rustc-link-lib=dylib=swiftCore");

    // Link C++ standard library (needed for FastClusterWrapper.cpp in FluidAudio)
    println!("cargo:rustc-link-lib=c++");
}

/// Map a cargo `TARGET` triple to the SwiftPM `--triple` value SwiftPM
/// expects. Returning `None` falls back to SwiftPM's host triple, which is the
/// right behaviour for native macOS builds.
///
/// Known mappings (deployment targets match Xcode 15 defaults):
/// - `aarch64-apple-darwin` → host (None)
/// - `x86_64-apple-darwin` → host (None)
/// - `aarch64-apple-ios` → `arm64-apple-ios17.0`
/// - `aarch64-apple-ios-sim` → `arm64-apple-ios17.0-simulator`
/// - `x86_64-apple-ios` → `x86_64-apple-ios17.0-simulator` (Xcode 15+ only ships sim x86_64)
fn cargo_to_swift_triple(cargo_target: &str) -> Option<String> {
    if cargo_target.is_empty() || cargo_target.contains("darwin") {
        return None;
    }
    if !cargo_target.contains("apple-ios") {
        // Non-Apple targets are out of scope for this crate, but be explicit:
        return None;
    }

    let arch = if cargo_target.starts_with("aarch64") {
        "arm64"
    } else if cargo_target.starts_with("x86_64") {
        "x86_64"
    } else {
        return None;
    };

    let suffix = if cargo_target.ends_with("-sim") || cargo_target.starts_with("x86_64-apple-ios") {
        "-simulator"
    } else {
        ""
    };

    Some(format!("{arch}-apple-ios17.0{suffix}"))
}

/// Walk `swift_build_dir` looking for `libFluidAudioBridge.a` and return its
/// parent directory. SwiftPM places it under a triple-scoped subdir whose
/// exact name depends on the SDK/toolchain version — searching is more
/// robust than hard-coding the layout.
fn find_bridge_lib_dir(swift_build_dir: &std::path::Path) -> Option<PathBuf> {
    fn walk(dir: &std::path::Path) -> Option<PathBuf> {
        let entries = std::fs::read_dir(dir).ok()?;
        let mut subdirs = vec![];
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path.file_name().and_then(|n| n.to_str()) == Some("libFluidAudioBridge.a")
            {
                return path.parent().map(|p| p.to_path_buf());
            }
            if path.is_dir() {
                subdirs.push(path);
            }
        }
        for sub in subdirs {
            if let Some(hit) = walk(&sub) {
                return Some(hit);
            }
        }
        None
    }
    walk(swift_build_dir)
}

/// Map a cargo `TARGET` triple to the `xcrun -sdk <name>` value, so we can
/// invoke `swift build` against the right Apple SDK. Returns `None` for the
/// macOS host build (where the default `xcrun swift` already works).
fn cargo_to_xcrun_sdk(cargo_target: &str) -> Option<&'static str> {
    if !cargo_target.contains("apple-ios") {
        return None;
    }
    if cargo_target.ends_with("-sim") || cargo_target.starts_with("x86_64-apple-ios") {
        Some("iphonesimulator")
    } else {
        Some("iphoneos")
    }
}
