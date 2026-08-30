//! Build-time environment detection and other utilities for the Olive workspace.
//!
//! This crate is consumed exclusively as a `build-dependency`. It provides
//! granular predicates so each consuming crate's build script can decide
//! which cfg flags to emit.
//!
//! # Detection items
//!
//! ## Unstable features
//! 
//! Two independent signals indicate that unstable features (`#![feature(...)]`)
//! are permitted:
//!
//! - **Nightly** — the compiler's version string contains `"nightly"` or
//!   `"-dev"`. This is authoritative; no environment variable is involved.
//! - **Bootstrap** — the `RUSTC_BOOTSTRAP` environment variable is set and
//!   applies to the current crate. This allows a stable or beta compiler to
//!   accept `#![feature(...)]` gates (used by the rustc bootstrap process and
//!   by projects like Firefox/Chromium that need cutting-edge features on
//!   stable).
//!
//! A crate may use unstable features when *either* signal is present. The
//! combined result maps to `cfg(unstable_features)`, emitted by the
//! consuming build script.
//!
//! # Rationales
//!
//! ## Running Miri with full provenance
//!
//! There are certain parts where Miri cannot successfully run without unstable features:
//! - Cloning unsized heap items from references require copying pointer metadata. Normally, 
//!   we can simply use a pointer hack, but it causes undefined behavior when on Miri 
//!   (although it is sound otherwise).
//!
//! ```ignore
//! #[cfg(all(miri, not(unstable_features)))]
//! compile_error!("Miri requires nightly or bootstrap.");
//! ```

/// Returns `true` if the current compiler is a nightly build.
///
/// Detection method: runs `rustc --version` and checks whether the output
/// contains the substring `"nightly"` or `"-dev"`. This is the same heuristic
/// used by `autocfg` and most build scripts in the ecosystem.
pub fn is_nightly() -> bool {
    let Ok(output) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let version = String::from_utf8_lossy(&output.stdout);
    version.contains("nightly") || version.contains("-dev")
}

/// Returns `true` if the `RUSTC_BOOTSTRAP` environment variable permits
/// unstable features for the current crate.
///
/// `RUSTC_BOOTSTRAP` can take three forms:
/// - `1` (or any non-numeric value): applies to all crates.
/// - A crate name: applies only to that specific crate.
/// - `-1`: explicitly disables bootstrap (acts as stable even on nightly).
///
/// The `crate_name` parameter should be the name of the crate whose build
/// script is calling this function (available via `CARGO_PKG_NAME`).
pub fn is_bootstrap(crate_name: &str) -> bool {
    match std::env::var("RUSTC_BOOTSTRAP") {
        Err(_) => false,
        Ok(val) => match val.as_str() {
            "" => false,
            "-1" => false,
            _ => val == "1" || val == crate_name,
        },
    }
}

/// Returns `true` if unstable features (`#![feature(...)]`) are permitted
/// under the current build configuration.
///
/// This is `true` when either [`is_nightly`] or [`is_bootstrap`] holds.
pub fn can_use_unstable_features(crate_name: &str) -> bool {
    is_nightly() || is_bootstrap(crate_name)
}
