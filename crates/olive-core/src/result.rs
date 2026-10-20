//! Re-exports of [`core::result`] types and foundational-trait impls
//! for these types.
//!
//! This includes the notable type [`Result`] and the iterator types
//! that accompany it.
//!
//! # Constraints
//!
//! Due to API constraints, it is impossible to soundly implement
//! [`TryClone`] for certain iterator types like [`IntoIter`], at least
//! without hacky solutions that assume layout and may silently break.

// Re-export the whole `core::result` surface so this shadowing module stays a
// drop-in superset.
pub use core::result::*;

use crate::try_traits::try_clone::{TryClone, TryCloneError};

impl<T: TryClone, E: TryClone> TryClone for Result<T, E> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            Ok(v) => Ok(Ok(v.try_clone()?)),
            Err(e) => Ok(Err(e.try_clone()?)),
        }
    }
}

impl<T> TryClone for Iter<'_, T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Copies the `Option<&'a T>` payload; a shared reference is `Copy`,
        // so this can neither allocate nor panic.
        Ok(Clone::clone(self))
    }
}

// No TryClone is possible for `IntoIter` without significant sorcery that
// assumes implementation details, and we consider the `IntoIter` a less common
// use case.
// An old algorithm attempted to use ManuallyDrop to extract out the item.

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::testing::{CloneBudget, MaybeUnclonable};

    #[test]
    fn try_clone_result() {
        let r: Result<i32, bool> = Ok(9);
        assert_eq!(r.try_clone().unwrap(), Ok(9));
    }

    #[test]
    fn try_clone_result_failure_paths() {
        let ok: Result<CloneBudget, u8> = Ok(CloneBudget::new(0));
        assert!(ok.try_clone().is_err());

        let err: Result<u8, CloneBudget> = Err(CloneBudget::new(0));
        assert!(err.try_clone().is_err());
    }

    #[test]
    fn iter_try_clone_succeeds_even_when_the_pointee_is_unclonable() {
        // The payload is a shared reference, so cloning the iterator copies
        // the reference and never invokes the pointee's `TryClone` (which
        // would fail for `MaybeUnclonable`).
        let holder: Result<MaybeUnclonable<u32>, TryCloneError> =
            Ok(MaybeUnclonable::unclonable(1));
        let mut iter = holder.iter();
        let mut cloned = iter
            .try_clone()
            .expect("cloning a reference payload is infallible");

        // Both iterators drain the same item, independently.
        assert_eq!(iter.next().map(|m| m.inner), Some(1));
        assert_eq!(cloned.next().map(|m| m.inner), Some(1));
        assert_eq!(iter.next(), None);
        assert_eq!(cloned.next(), None);
    }

    #[test]
    fn iter_try_clone_of_an_err_iter_stays_empty() {
        let holder: Result<u32, TryCloneError> = Err(TryCloneError::Other("unused"));
        let mut iter = holder.iter();
        let mut cloned = iter.try_clone().unwrap();
        assert_eq!(cloned.next(), None);
        assert_eq!(iter.next(), None);
    }
}
