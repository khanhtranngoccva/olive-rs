#![allow(missing_docs)]

fn main() {
    let crate_name = env!("CARGO_PKG_NAME");

    // Re-run if the bootstrap env var changes.
    println!("cargo:rerun-if-env-changed=RUSTC_BOOTSTRAP");

    // Emit cfg(unstable_features) when nightly or bootstrap is active.
    if olive_build::can_use_unstable_features(crate_name) {
        println!("cargo:rustc-cfg=unstable_features");
    }

    // Mark that this is a genuine cargo build (as opposed to rust-analyzer's
    // analysis, which skips build scripts). The Miri guard in lib.rs uses this
    // to avoid firing inside the IDE, where `cfg(miri)` is enabled by default
    // but no build script has run.
    println!("cargo:rustc-cfg=olive_real_build");

    // Declare our custom cfgs so the compiler doesn't warn about them.
    println!("cargo:rustc-check-cfg=cfg(unstable_features)");
    println!("cargo:rustc-check-cfg=cfg(olive_real_build)");
}
