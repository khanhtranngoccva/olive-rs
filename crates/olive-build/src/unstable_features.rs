//! Detection of whether unstable features (`#![feature(...)]`) are
//! permitted, and the cfg emission built on that verdict.

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

/// The environment variable whose value every bootstrap check depends on.
pub const BOOTSTRAP_ENV: &str = "RUSTC_BOOTSTRAP";

/// Returns `true` if the `RUSTC_BOOTSTRAP` environment variable permits
/// unstable features for the current crate.
///
/// `RUSTC_BOOTSTRAP` can take four forms:
/// - `1`: applies to all crates.
/// - A comma-separated list of crate names: applies only to the crates named
///   in the list.
/// - `-1`: explicitly disables bootstrap (acts as stable even on nightly).
///
/// The `crate_name` parameter should be the name of the crate whose build
/// script is calling this function (available via `CARGO_PKG_NAME`).
///
/// This function emits
/// `cargo:rerun-if-env-changed=RUSTC_BOOTSTRAP` so a cached build-script
/// result cannot go stale when the variable changes. cargo treats repeated
/// identical directives as a single trigger, so the duplicate emission from
/// [`can_use`] is harmless.
pub fn is_bootstrap(crate_name: &str) -> bool {
    cargo_build::rerun_if_env_changed(BOOTSTRAP_ENV);
    match std::env::var(BOOTSTRAP_ENV) {
        Err(_) => false,
        Ok(val) => match val.as_str() {
            "" | "-1" => false,
            "1" => true,
            // rustc accepts either a single crate name or a comma-separated
            // list of crate names; the current crate must be named among them.
            names => names.split(',').any(|name| name == crate_name),
        },
    }
}

/// Returns `true` if unstable features (`#![feature(...)]`) are permitted
/// under the current build configuration.
///
/// Normally this is `true` when either [`is_nightly`] or [`is_bootstrap`]
/// holds. However, `RUSTC_BOOTSTRAP=-1` is a hard override: it tells rustc to
/// reject `#![feature]` *even on nightly*, so it must disable unstable features
/// regardless of channel. That check takes priority over both signals.
///
/// This function emits
/// `cargo:rerun-if-env-changed=RUSTC_BOOTSTRAP`, including on the `-1`
/// early-return path.
pub fn can_use(crate_name: &str) -> bool {
    cargo_build::rerun_if_env_changed(BOOTSTRAP_ENV);
    // A `-1` bootstrap value forces stable semantics, overriding nightly.
    if std::env::var(BOOTSTRAP_ENV).as_deref() == Ok("-1") {
        return false;
    }
    is_nightly() || is_bootstrap(crate_name)
}

/// The custom cfg this module emits, shared by [`emit`] so the name lives
/// in exactly one place.
pub const CFG_NAME: &str = "unstable_features";

/// Registers the `unstable_features` cfg with `cargo:rustc-check-cfg` and,
/// when `can` holds, enables it via a bare `cargo:rustc-cfg` flag. Always
/// call this (even on the stable path): registration is what keeps the
/// absent flag from producing an unexpected-cfg warning.
///
/// ```ignore
/// let unstable = olive_build::unstable_features::can_use(crate_name);
/// olive_build::unstable_features::emit(unstable);
/// ```
pub fn emit(can: bool) {
    cargo_build::rustc_check_cfgs(CFG_NAME);
    if can {
        cargo_build::rustc_cfg(CFG_NAME);
    }
}
