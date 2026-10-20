//! Re-exports of [`core::option`] types and foundational-trait impls
//! for these types.
//!
//! This includes the notable type [`Option`] and the iterator types
//! that accompany it.
//! 
//! # Constraints
//! 
//! Due to API constraints, it is impossible to soundly implement 
//! [`TryClone`] for certain iterator types like [`IntoIter`], at least 
//! without hacky solutions that assume layout and may silently break.

// Re-export the whole `core::option` surface so this shadowing module stays a
// drop-in superset.
pub use core::option::*;

use crate::try_traits::{TryClone, TryCloneError};
use crate::try_traits::{TryDefault, TryDefaultError};

impl<T: TryClone> TryClone for Option<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            Some(v) => Ok(Some(v.try_clone()?)),
            None => Ok(None),
        }
    }
}

impl<T> TryDefault for Option<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(None)
    }
}

impl<A> TryClone for Iter<'_, A> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
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
    fn try_clone_option() {
        let o: Option<i32> = Some(7);
        assert_eq!(o.try_clone().unwrap(), Some(7));
    }

    #[test]
    fn try_clone_optionfailure_paths() {
        let none: Option<CloneBudget> = None;
        assert!(none.try_clone().is_ok());

        let some: Option<CloneBudget> = Some(CloneBudget::new(0));
        assert!(some.try_clone().is_err());
    }

    #[test]
    fn iter_try_clone_succeeds_even_when_the_pointee_is_unclonable() {
        // The payload is a shared reference, so cloning the iterator copies
        // the reference and never invokes the pointee's `TryClone` (which
        // would fail for `MaybeUnclonable`).
        let holder: Option<MaybeUnclonable<u32>> = Some(MaybeUnclonable::unclonable(1));
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
    fn iter_try_clone_of_an_empty_iter_stays_empty() {
        let holder: Option<u32> = None;
        let mut iter = holder.iter();
        let mut cloned = iter.try_clone().unwrap();
        assert_eq!(cloned.next(), None);
        assert_eq!(iter.next(), None);
    }
}
