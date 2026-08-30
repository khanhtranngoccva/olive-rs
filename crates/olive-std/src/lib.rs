//! # olive-std
//!
//! A fully-fallible re-port of Rust's `std`-only surface: hash collections, FFI
//! types, sync primitives, panic helpers (`catch_unwind`, `resume_unwind`), and many more. 
//! This crate depends on both [`olive_core`] and [`olive_alloc`].
//!
//! (Placeholder — the std-only port is layered on top of the alloc port.)

#![deny(clippy::arithmetic_side_effects)]

pub use olive_alloc;
pub use olive_core;

#[cfg(test)]
mod tests {
    // These prove the shape of olive_core's public surface for downstream code.

    #[test]
    fn panic_macro_in_scope_for_downstream() {
        // `panic!` is in scope through the language prelude; downstream code
        // doesn't need to route it through olive_core at all.
        let result = std::panic::catch_unwind(|| {
            panic!("intentional test panic");
        });
        assert!(result.is_err());
    }

    #[test]
    fn olive_core_glob_carries_core_modules_through() {
        // The `pub use core::*` glob re-exports core's modules, so deep items are
        // reachable under the olive_core:: path without any Olive-side plumbing.
        let x: olive_core::num::NonZeroU32 = olive_core::num::NonZeroU32::new(5).unwrap();
        assert_eq!(x.get(), 5);

        // The `panic` module itself rides along too — no local module needed.
        let loc = olive_core::panic::Location::caller();
        assert!(loc.line() > 0);
        let _safe: &dyn olive_core::panic::UnwindSafe = &(1u8);
    }
}
