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
//! Consuming crates therefore refuse to build under a *genuine* Miri invocation
//! unless `unstable_features` is active. This check lives in each crate's
//! build script (using [`is_miri`] + [`can_use_unstable_features`]) rather than
//! as a `compile_error!` in source, because rust-analyzer enables `cfg(miri)`
//! by default during analysis while leaving `unstable_features` unset — a
//! source-level guard would fire spuriously inside the IDE.

/// Returns `true` if the current compiler is a nightly build.
///
/// Detection method: runs `rustc --version` and checks whether the output
/// contains the substring `"nightly"` or `"-dev"`. This is the same heuristic
/// used by most build scripts in the ecosystem.
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
/// Normally this is `true` when either [`is_nightly`] or [`is_bootstrap`]
/// holds. However, `RUSTC_BOOTSTRAP=-1` is a hard override: it tells rustc to
/// reject `#![feature]` *even on nightly* (producing E0554), so it must
/// disable unstable features regardless of channel. That check takes priority
/// over both signals.
pub fn can_use_unstable_features(crate_name: &str) -> bool {
    // A `-1` bootstrap value forces stable semantics, overriding nightly.
    if std::env::var("RUSTC_BOOTSTRAP").as_deref() == Ok("-1") {
        return false;
    }
    is_nightly() || is_bootstrap(crate_name)
}

/// Returns `true` if this build script is running under a genuine
/// `cargo miri` invocation (as opposed to rust-analyzer's analysis, which
/// fakes `cfg(miri)` but never invokes the Miri driver).
///
/// Detection method: a real `cargo miri` run sets the `MIRI_SYSROOT`
/// environment variable in the build-script environment (pointing at the
/// Miri-built sysroot). rust-analyzer does not set this, so its presence is a
/// reliable discriminator. As a secondary signal we also check whether the
/// `RUSTC` binary path ends in `/miri`, which is the case when cargo-miri
/// swaps the compiler driver.
///
/// # Rationale
///
/// Some code paths are provenance-unsound under Miri's strict model unless compiled with
/// `unstable_features`. A `compile_error!` placed in crate source would fire
/// spuriously inside rust-analyzer (which enables `cfg(miri)` by default while
/// leaving `unstable_features` unset). By performing the check here — where we
/// can tell a real Miri build from IDE analysis — we fail only when it truly
/// matters.
pub fn is_miri() -> bool {
    std::env::var_os("MIRI_SYSROOT").is_some()
        || std::env::var("RUSTC")
            .map(|rustc| {
                rustc.ends_with("/miri")
                    || rustc.ends_with("\\miri.exe")
                    || rustc.ends_with("miri.exe")
            })
            .unwrap_or(false)
}
