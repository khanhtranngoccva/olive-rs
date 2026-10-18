//! Foundational-trait impls for marker types, such as [`core::marker::PhantomData`].

// Re-export the whole `core::marker` surface so this shadowing module stays a
// drop-in superset.
pub use core::marker::*;

use crate::try_traits::try_clone::{TryClone, TryCloneError};
use crate::try_traits::try_default::{TryDefault, TryDefaultError};

impl<T> TryClone for PhantomData<T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

impl<T> TryDefault for PhantomData<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(PhantomData)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn phantom_data_try_default_is_infallible() {
        struct NoDefault;
        let _: PhantomData<NoDefault> = TryDefault::try_default().unwrap();
    }

    #[test]
    fn phantom_data_try_clone_is_infallible() {
        struct NoClone;
        let p: PhantomData<NoClone> = PhantomData;
        let cloned = p.try_clone().unwrap();
        // Both are ZST markers; identity is preserved by construction.
        let _ = cloned;
    }
}
