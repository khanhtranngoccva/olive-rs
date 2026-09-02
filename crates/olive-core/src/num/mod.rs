//! Foundational-trait implementations for the [`core::num`] numeric types,
//! and re-export for the [`core::num`] module.
//!
//! This module also re-exports numeric types and other items from the original
//! [`core`] crate, allowing these items to be accessible from the Olive namespace.
pub use core::num::*;

// These modules only contain trait implementations, so they can be private.
mod error;
mod float_parse;
mod nonzero;
mod saturating;
mod wrapping;

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    /// Deep-path smoke test: every `core::num` item must still resolve through
    /// this shadowing module via the glob re-export, proving the Olive partition
    /// did not hide any of the original surface.
    #[test]
    fn num_module_exposes_core_surface() {
        let nz = NonZeroU32::new(5).unwrap();
        assert_eq!(nz.get(), 5);
        let w = Wrapping(7i32);
        assert_eq!(w.0, 7);
        let s = Saturating(10u8);
        assert_eq!(s.0, 10);
        let _: Option<IntErrorKind> = None;
        let _pf: Option<ParseFloatError> = None;
        let _pi: Option<ParseIntError> = None;
        let _tf: Option<TryFromIntError> = None;
    }
}
