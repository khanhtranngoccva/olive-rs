#![allow(missing_docs)]

fn main() {
    let crate_name = env!("CARGO_PKG_NAME");

    // Re-run if the bootstrap env var or Miri environment changes.
    println!("cargo:rerun-if-env-changed=RUSTC_BOOTSTRAP");
    println!("cargo:rerun-if-env-changed=MIRI_SYSROOT");

    let unstable = olive_build::can_use_unstable_features(crate_name);

    // Emit cfg(unstable_features) when nightly or bootstrap is active.
    if unstable {
        println!("cargo:rustc-cfg=unstable_features");
    }

    // Declare our custom cfg so the compiler doesn't warn about it.
    println!("cargo:rustc-check-cfg=cfg(unstable_features)");
}
