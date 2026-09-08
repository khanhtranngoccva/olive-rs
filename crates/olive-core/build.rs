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

    // Miri guard: Running this crate under a *genuine* Miri invocation requires
    // `unstable_features` to enable use of strict Miri implementations.
    // We detect a real Miri run here (via MIRI_SYSROOT / the miri driver) rather
    // than relying on cfg(miri), because rust-analyzer sets cfg(miri) by default
    // during analysis while leaving unstable_features unset — a source-level
    // compile_error! would therefore fire spuriously inside the IDE.
    if olive_build::is_miri() && !unstable {
        eprintln!(
            "error: {crate_name} cannot be compiled under Miri without unstable features enabled. \
             Re-run with nightly, or set RUSTC_BOOTSTRAP={crate_name}."
        );
        std::process::exit(1);
    }
}
