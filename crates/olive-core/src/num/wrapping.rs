//! Foundational-trait impls for the wrapping numeric newtype ([`core::num::Wrapping`]).
use core::num::Wrapping;

use crate::try_traits::try_clone::{TryClone, TryCloneError};

/// Cloning a [`Wrapping`] clones the wrapped value via [`TryClone`].
impl<T: TryClone> TryClone for Wrapping<T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Wrapping(self.0.try_clone()?))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn wrapping_try_clone_preserves_value() {
        let w = Wrapping(42i32);
        assert_eq!(w.try_clone().unwrap(), w);

        let u = Wrapping(9u64);
        assert_eq!(u.try_clone().unwrap(), u);
    }
}
