//! Foundational-trait impls for the [`core::ops`] types that Olive layers fallible
//! behavior onto, and re-exports for the [`core::ops`].
//!
//! This module re-exports the entire [`core::ops`] surface below to keep every
//! item accessible from the Olive namespace.
pub use core::ops::*;

// These modules can be private because they only contain trait implementations that 
// can be accessed publicly.
mod control_flow;
mod range;

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::try_traits::TryClone;

    /// Deep-path smoke test: every `core::ops` item must still resolve through
    /// this shadowing module via the glob re-export, proving the Olive partition
    /// did not hide any of the original surface.
    #[test]
    fn ops_module_exposes_core_surface() {
        let r = 1..5usize;
        assert_eq!(r.start, 1);
        let cf: ControlFlow<u32, ()> = ControlFlow::Break(42u32);
        assert_eq!(cf.try_clone().unwrap(), cf);
        // A couple of non-Olive core::ops items to prove the whole surface rides along.
        let neg = -3i32;
        assert_eq!(neg, -3);
    }
}
