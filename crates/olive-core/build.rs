#![allow(missing_docs)]

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let crate_name = env!("CARGO_PKG_NAME");

    // The checks below each emit their own `cargo:rerun-if-env-changed`
    // triggers for the environment variables they read.

    let unstable = olive_build::unstable_features::can_use(crate_name);
    olive_build::unstable_features::emit(unstable);

    // Registered for `core::io::error::os_functions`: builds that pass
    // `--cfg no_io_statics` select the fallback (non-atomic) vtable plumbing.
    cargo_build::rustc_check_cfgs("no_io_statics");

    // Registered unstable_target_has_atomic_load_store for io error vtable.
    let widths = olive_build::target_has_atomic_load_store::probe();
    olive_build::target_has_atomic_load_store::emit(&widths);
    olive_build::target_has_atomic_load_store::emit_check_cfg();

    // Miri guard: Running this crate under a *genuine* Miri invocation requires
    // `unstable_features` to enable use of strict Miri implementations.
    olive_build::miri::ensure_miri_on_unstable(crate_name, unstable)?;

    Ok(())
}
