//! Foundational-trait impls for the saturating number wrapper ([`core::num::Saturating`]).
use core::num::Saturating;

use crate::try_traits::try_clone::{TryClone, TryCloneError};

/// Cloning a [`Saturating`] clones the wrapped value via [`TryClone`].
impl<T: TryClone> TryClone for Saturating<T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Saturating(self.0.try_clone()?))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn saturating_try_clone_preserves_value() {
        let s = Saturating(7i32);
        assert_eq!(s.try_clone().unwrap(), s);

        let u = Saturating(12u64);
        assert_eq!(u.try_clone().unwrap(), u);
    }
}
