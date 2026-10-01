//! Construction methods for [`BTreeSet`].

use olive_core::{try_traits::TryDefault, try_traits::TryDefaultError};

use super::super::map::BTreeMap;
use super::BTreeSet;
use crate::alloc::{Allocator, AllocatorTryDefault, Global};

impl<T: Ord, A: Allocator> BTreeSet<T, A> {
    /// Attempts to create an empty `BTreeSet` with the given allocator.
    pub fn new_in(alloc: A) -> Self {
        Self {
            map: BTreeMap::new_in(alloc),
        }
    }
}

impl<T: Ord> BTreeSet<T, Global> {
    /// Creates an empty `BTreeSet` backed by the global allocator.
    #[inline]
    pub fn new() -> Self {
        Self::new_in(Global)
    }
}

impl<T: Ord> Default for BTreeSet<T, Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord, A: AllocatorTryDefault> TryDefault for BTreeSet<T, A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        let alloc = A::try_default()?;
        Ok(BTreeSet::new_in(alloc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_constructs_a_set() {
        let _set: BTreeSet<i32> = BTreeSet::new();
    }

    #[test]
    fn new_in_constructs_a_set() {
        let _set: BTreeSet<i32> = BTreeSet::new_in(Global);
    }

    #[test]
    fn new_and_new_in_both_construct() {
        let _a: BTreeSet<u32> = BTreeSet::new();
        let _b: BTreeSet<u32> = BTreeSet::new_in(Global);
    }

    #[test]
    fn new_set_drop_is_clean() {
        // Dropping a freshly constructed set must not leak or panic.
        let set: BTreeSet<i32> = BTreeSet::new();
        drop(set);
    }

    #[test]
    fn new_in_set_drop_is_clean() {
        let set: BTreeSet<i32> = BTreeSet::new_in(Global);
        drop(set);
    }

    // --- Default ---------------------------------------------------------------

    #[test]
    fn default_constructs_a_set() {
        let _set: BTreeSet<i32> = BTreeSet::default();
    }

    // --- TryDefault ------------------------------------------------------------

    #[test]
    fn try_default_constructs_a_set() {
        let _set: BTreeSet<i32, Global> = TryDefault::try_default().expect("default ok");
    }
}
