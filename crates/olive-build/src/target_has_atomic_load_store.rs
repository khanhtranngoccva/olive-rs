//! Detection and emission of polyfill cfg for the precise `target_has_atomic_load_store`
//! cfg family (unstable `cfg_target_has_atomic`, tracking issue 94039) and
//! its build-script forwarding helpers.

/// The values the precise `target_has_atomic_load_store` cfg family can
/// take, in canonical order, shared by [`probe`] and the emission helpers.
///
/// [`None`] is the valueless form `cfg(target_has_atomic_load_store)`,
/// emitted by the compiler when the target supports atomic load/store at
/// *any* width; [`Some`] carries one concrete width (`"8"`, `"16"`,
/// `"32"`, `"64"`, `"ptr"`).
pub const TARGET_HAS_ATOMIC_LOAD_STORE_WIDTHS: [Option<&str>; 6] = [
    None,
    Some("8"),
    Some("16"),
    Some("32"),
    Some("64"),
    Some("ptr"),
];

/// The custom cfg this module emits, shared by [`emit`] and
/// [`emit_check_cfg`].
pub const CFG_NAME: &str = "unstable_target_has_atomic_load_store";

/// The cfg name as rustc itself reports it in `--print cfg` output..
const RUSTC_CFG_NAME: &str = "target_has_atomic_load_store";

/// Returns the widths for which the target supports lock-free atomic load
/// and store operations, as confirmed by the precise
/// `cfg(target_has_atomic_load_store = ...)` probe.
///
/// The returned set is drawn from [`TARGET_HAS_ATOMIC_LOAD_STORE_WIDTHS`],
/// in that order: [`None`] for the valueless form
/// `cfg(target_has_atomic_load_store)`, and [`Some`] for each confirmed
/// width (`"8"`, `"16"`, `"32"`, `"64"`, `"ptr"`). Unrecognized values the
/// compiler might report are filtered out.
///
/// This probe is in place to support targets that do not fully support
/// `target_has_atomic_load_store`.
///
/// The probe emits `cargo:rerun-if-env-changed` for `RUSTC` and `TARGET`,
/// the two reads it depends on.
pub fn probe() -> Vec<Option<&'static str>> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let target = std::env::var("TARGET").unwrap_or_default();
    cargo_build::rerun_if_env_changed("RUSTC");
    cargo_build::rerun_if_env_changed("TARGET");
    let Ok(output) = std::process::Command::new(rustc)
        .args(["--print", "cfg", "--target", &target])
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    let widths: Vec<Option<&'static str>> = TARGET_HAS_ATOMIC_LOAD_STORE_WIDTHS
        .iter()
        .filter(|width| {
            let needle = match width {
                // The valueless form is a cfg in its own right, distinct
                // from every valued form (rustc keeps (key, value) pairs).
                None => RUSTC_CFG_NAME.to_string(),
                // FIXME: this is bad serialization
                Some(width) => format!("{RUSTC_CFG_NAME}=\"{width}\""),
            };
            stdout.lines().any(|line| line.trim() == needle)
        })
        .copied()
        .collect();
    widths
}

/// Emits the `cargo:rustc-cfg` flags forwarding the widths confirmed by
/// [`probe`] to the crate being built, as valued cfgs mirroring the
/// unstable ones. The valueless form ([`None`]) is emitted as a bare flag.
pub fn emit(widths: &[Option<&str>]) {
    for width in widths {
        match width {
            None => cargo_build::rustc_cfg(CFG_NAME),
            Some(width) => cargo_build::rustc_cfg((CFG_NAME, *width)),
        }
    }
}

/// Emits the `cargo:rustc-check-cfg` registration for the
/// `unstable_target_has_atomic_load_store` cfg.
pub fn emit_check_cfg() {
    let values = TARGET_HAS_ATOMIC_LOAD_STORE_WIDTHS
        .iter()
        .filter_map(|width| width.as_deref())
        .collect::<Vec<_>>();
    cargo_build::rustc_check_cfg(CFG_NAME, &values);
}
