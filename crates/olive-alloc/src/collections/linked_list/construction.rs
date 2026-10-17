//! Construction operations for [`LinkedList`].

use core::marker::PhantomData;

use olive_core::alloc::AllocatorTryDefault;
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

use crate::alloc::{Allocator, Global};

use super::LinkedList;

impl<T> LinkedList<T, Global> {
    /// Constructs a new, empty `LinkedList<T>`.
    ///
    /// The list will not allocate until an element is pushed onto it.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self::new_in(Global)
    }
}

impl<T, A: Allocator> LinkedList<T, A> {
    /// Constructs a new, empty `LinkedList<T, A>` with the given allocator.
    ///
    /// The list will not allocate until an element is pushed onto it. No
    /// allocation happens here, so this cannot fail.
    #[must_use]
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            head: None,
            tail: None,
            len: 0,
            alloc,
            marker: PhantomData,
        }
    }
}

impl<T> Default for LinkedList<T, Global> {
    /// Equivalent to [`LinkedList::new`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, A: AllocatorTryDefault> TryDefault for LinkedList<T, A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        let alloc = A::try_default()?;
        Ok(Self::new_in(alloc))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use olive_core::try_traits::try_default::TryDefault;
    use olive_test_helpers::{FailAlloc, FailDefaultAlloc};

    use super::*;
    use crate::alloc::AllocError;

    #[test]
    fn new_is_empty_and_uses_global_allocator() {
        let l: LinkedList<i32, Global> = LinkedList::new();
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn new_in_uses_the_given_allocator() {
        let mut l: LinkedList<i32, FailAlloc> = LinkedList::new_in(FailAlloc);
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
        // Nothing has been allocated yet, so construction itself cannot have
        // failed; the first push must go through the given allocator and
        // FailAlloc makes that observable.
        assert!(matches!(l.try_push_front(1), Err(AllocError)));
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn default_matches_new() {
        let d: LinkedList<i32> = Default::default();
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn try_default_matches_new() {
        let t: LinkedList<i32, Global> = TryDefault::try_default().expect("infallible for Global");
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn linked_list_try_default_allocator_failure_propagates() {
        let e = <LinkedList<u32, FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }
}
