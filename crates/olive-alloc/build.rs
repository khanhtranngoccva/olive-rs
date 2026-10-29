#![allow(missing_docs)]

fn main() {
    let crate_name = env!("CARGO_PKG_NAME");

    // Checks emit their own reload-if-env-changed.

    let unstable = olive_build::unstable_features::can_use(crate_name);
    olive_build::unstable_features::emit(unstable);
}
