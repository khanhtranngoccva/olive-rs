//! Foundational-trait impls for marker types, such as [`core::marker::PhantomData`].

// Re-export the whole `core::marker` surface so this shadowing module stays a
// drop-in superset: deep paths like `olive_core::marker::PhantomPinned` resolve
// exactly as they do under `core::marker`. The glob also brings `PhantomData`
// into scope for the impls below, so no separate private import is needed (a
// private import here would shadow the public re-export and make the type
// unreachable to downstream crates).
pub use core::marker::*;

use crate::try_traits::try_clone::{TryClone, TryCloneError};

/// Cloning a [`PhantomData`] is a bit-for-bit copy of a zero-sized marker — no
/// allocation, no failure. The `T: TryClone` bound mirrors std's `Clone` bound
/// on `PhantomData<T>` (which requires `T: Clone`) so the impl composes with the
/// rest of the fallible ecosystem even though the value itself is empty.
impl<T: TryClone> TryClone for PhantomData<T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn phantom_data_try_clone_is_infallible() {
        let p: PhantomData<u32> = PhantomData;
        let cloned = p.try_clone().unwrap();
        // Both are ZST markers; identity is preserved by construction.
        let _ = cloned;
    }
}
