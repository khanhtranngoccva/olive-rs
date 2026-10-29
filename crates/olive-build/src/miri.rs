//! Detection of a genuine `cargo miri` invocation, as distinct from
//! rust-analyzer's analysis-time faking of `cfg(miri)`.

use std::{ffi::OsStr, path::Path};

/// Returns `true` if this build script is running under a genuine
/// `cargo miri` invocation (as opposed to rust-analyzer's analysis, which
/// fakes `cfg(miri)` but never invokes the Miri driver).
///
/// Detection method: a real `cargo miri` run sets the `MIRI_SYSROOT`
/// environment variable in the build-script environment (pointing at the
/// Miri-built sysroot). rust-analyzer does not set this, so its presence is a
/// reliable discriminator.
///
/// As a secondary signal we also check whether the `RUSTC` binary path ends
/// in `/miri`, which is the case when cargo-miri swaps the compiler driver.
///
/// # Rationale
///
/// Some code paths are provenance-unsound under Miri's strict model unless
/// compiled with `unstable_features`.
///
/// A `compile_error!` placed in crate source
/// would fire spuriously inside rust-analyzer (which enables `cfg(miri)`
/// by default while leaving `unstable_features` unset). By performing the check
/// here, we can tell a real Miri build from IDE analysis.
///
/// This function emits `cargo:rerun-if-env-changed` for `MIRI_SYSROOT` and
/// `RUSTC`, the two environment reads its detection relies on, so a cached
/// verdict cannot go stale when either changes. cargo treats repeated
/// identical directives as a single trigger.
pub fn is_miri() -> bool {
    cargo_build::rerun_if_env_changed("MIRI_SYSROOT");
    cargo_build::rerun_if_env_changed("RUSTC");
    std::env::var_os("MIRI_SYSROOT").is_some()
        || std::env::var("RUSTC")
            .map(|rustc| {
                let path = Path::new(&rustc);
                let name = path.file_name();
                name == Some(OsStr::new("miri")) || name == Some(OsStr::new("miri.exe"))
            })
            .unwrap_or(false)
}

/// The error returned by [`ensure_miri_on_unstable`] when a *genuine* Miri
/// invocation runs without the unstable features the crate requires.
pub struct MiriGuardError {
    crate_name: String,
}

impl std::fmt::Display for MiriGuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} cannot be compiled under Miri without unstable features enabled. \
             Re-run with nightly, or set RUSTC_BOOTSTRAP={}.",
            self.crate_name, self.crate_name
        )
    }
}

impl std::fmt::Debug for MiriGuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for MiriGuardError {}

/// Runs the Miri guard: returns [`MiriGuardError`] when this build runs
/// under a *genuine* Miri invocation while the unstable
/// features Miri needs are unavailable.
pub fn ensure_miri_on_unstable(crate_name: &str, unstable: bool) -> Result<(), MiriGuardError> {
    if is_miri() && !unstable {
        return Err(MiriGuardError {
            crate_name: crate_name.to_string(),
        });
    }
    Ok(())
}
