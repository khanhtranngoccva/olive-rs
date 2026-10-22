use super::Rc;
use super::uninit::UniqueRcUninit;
use crate::alloc::{Allocator, Global};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError, TryCloneToUninit};

// ---------------------------------------------------------------------------
// Unsized construction via TryCloneToUninit
// ---------------------------------------------------------------------------

#[allow(private_bounds)]
impl<T: ?Sized + TryCloneToUninit> Rc<T, Global> {
    /// Clones a `&T` into a freshly allocated `Rc<T, Global>` for potentially
    /// unsized `T`.
    ///
    /// This is the fallible analogue of std's `Rc::from(&slice[..])`,
    /// `Rc::from("literal")`, etc. It works for any type implementing
    /// [`TryCloneToUninit`]: in practice that means sized types, `str`, and
    /// slices `[T]`. Trait-object payloads (`dyn Trait`) are not yet supported.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails, an element clone
    /// fails, or the payload layout is absurd (see
    /// [`try_clone_from_ref_in`](Self::try_clone_from_ref_in)).
    #[inline]
    pub fn try_clone_from_ref(src: &T) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, Global)
    }
}

#[allow(private_bounds)]
impl<T: ?Sized + TryCloneToUninit, A: Allocator> Rc<T, A> {
    /// Clones a `&T` into a freshly allocated `Rc<T, A>` for potentially
    /// unsized `T`, using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails, any element clone
    /// fails, or the payload layout is absurd (e.g. a slice whose length would
    /// overflow when combined with the header). The last case is reported as
    /// [`TryCloneError::Other`], not an OOM-style error, because it can never
    /// succeed on retry.
    #[inline]
    pub fn try_clone_from_ref_in(src: &T, alloc: A) -> Result<Self, TryCloneError> {
        let uninit = UniqueRcUninit::try_new_for_value(src, alloc)?;
        // SAFETY: `uninit.data_ptr()` points to the uninitialized payload slot
        // within a live allocation of the correct size and alignment.
        unsafe { <T as TryCloneToUninit>::try_clone_to_uninit(src, uninit.data_ptr()) }?;
        // SAFETY: the value is just cloned into the uninit item.
        Ok(unsafe { uninit.into_rc() })
    }
}

// Convenience wrappers for common unsized types.
impl<T: TryClone, A: Allocator> Rc<[T], A> {
    /// Creates a new `Rc<[T]>` by cloning the slice of T items.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_slice_in(src: &[T], alloc: A) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, alloc)
    }
}

impl<T: TryClone> Rc<[T], Global> {
    /// Convenience wrapper around [`Self::try_from_slice_in`]
    /// using the global allocator.
    #[inline]
    pub fn try_from_slice(src: &[T]) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref(src)
    }
}

impl<A: Allocator> Rc<str, A> {
    /// Allocates a new `Rc<str>` by copying the UTF-8 bytes from `src` into
    /// fresh heap memory. Fallible analogue of std's `Rc::from("literal")`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_str_in(src: &str, alloc: A) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, alloc)
    }
}

impl Rc<str, Global> {
    /// Convenience wrapper around [`Self::try_from_str_in`] using
    /// the global allocator.
    #[inline]
    pub fn try_from_str(src: &str) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref(src)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_test_helpers::{CountingAllocator, FailAlloc, MaybeUnclonable};
    use std::string::String;

    #[test]
    fn unsized_byte_slice_construction_and_deref() {
        let arr = [10u8, 20, 30, 40];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        assert_eq!(&*rc, [10, 20, 30, 40]);
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);
    }

    #[test]
    fn unsized_byte_slice_rc_clone_shares_allocation() {
        let arr = [1u8, 2, 3];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
        assert_eq!(&*rc2, [1, 2, 3]);
    }

    #[test]
    fn unsized_byte_slice_weak_lifecycle() {
        let arr = [65u8, 66, 67];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn unsized_str_construction_and_deref() {
        let s = String::from("hello world");
        let rc: Rc<str> = Rc::try_from_str(s.as_str()).unwrap();
        assert_eq!(&*rc, "hello world");
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn unsized_str_rc_clone_shares_allocation() {
        let rc: Rc<str> = Rc::try_from_str("first").unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(&*rc2, "first");
        // Like the byte-slice case below, `try_clone` bumps the strong count
        // instead of copying the string into a fresh allocation.
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
    }

    #[test]
    fn unsized_str_weak_lifecycle() {
        let rc: Rc<str> = Rc::try_from_str("lifecycle").unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        assert_eq!(&*weak.try_upgrade().unwrap().unwrap(), "lifecycle");
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn try_from_slice_with_fail_alloc_errors() {
        let arr = [1u32, 2, 3];
        let res: Result<Rc<[u32], FailAlloc>, TryCloneError> =
            Rc::try_from_slice_in(&arr[..], FailAlloc);
        assert!(matches!(res, Err(TryCloneError::Alloc(_))));
    }

    #[test]
    fn try_from_str_with_fail_alloc_errors() {
        let res: Result<Rc<str, FailAlloc>, TryCloneError> =
            Rc::try_from_str_in("hello", FailAlloc);
        assert!(matches!(res, Err(TryCloneError::Alloc(_))));
    }

    #[test]
    fn try_clone_from_ref_sized_success() {
        let rc: Rc<u32> = Rc::try_clone_from_ref(&41).unwrap();
        assert_eq!(*rc, 41);
        assert_eq!(Rc::strong_count(&rc), 1);
        // Two constructions from equal sources are distinct allocations.
        let rc2: Rc<u32> = Rc::try_clone_from_ref(&41).unwrap();
        assert!(!Rc::ptr_eq(&rc, &rc2));
    }

    #[test]
    fn try_clone_from_ref_str_success() {
        let rc: Rc<str> = Rc::try_clone_from_ref("borrowed source").unwrap();
        assert_eq!(&*rc, "borrowed source");
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn try_clone_from_ref_in_sized_success_with_counting_allocator() {
        let alloc = CountingAllocator::new();
        let rc: Rc<u32, &CountingAllocator> = Rc::try_clone_from_ref_in(&7, &alloc).unwrap();
        assert_eq!(*rc, 7);
        assert_eq!(alloc.allocations(), 1);
        drop(rc);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_clone_from_ref_in_str_success_with_counting_allocator() {
        let alloc = CountingAllocator::new();
        let rc: Rc<str, &CountingAllocator> =
            Rc::try_clone_from_ref_in("heap-owning", &alloc).unwrap();
        assert_eq!(&*rc, "heap-owning");
        assert_eq!(alloc.allocations(), 1);
        drop(rc);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_clone_from_ref_in_sized_with_fail_alloc_errors() {
        let res: Result<Rc<i32, FailAlloc>, TryCloneError> =
            Rc::try_clone_from_ref_in(&5, FailAlloc);
        assert!(matches!(res, Err(TryCloneError::Alloc(_))));
    }

    #[test]
    fn try_clone_from_ref_in_slice_element_failure_frees_the_block() {
        // The first element clones fine and the second is armed to fail: the
        // partially initialized payload must be rolled back and the block
        // deallocated, with the element error surfaced as-is.
        let alloc = CountingAllocator::new();
        let items = [
            MaybeUnclonable::clonable(1u32),
            MaybeUnclonable::unclonable(2u32),
        ];
        let res: Result<Rc<[MaybeUnclonable<u32>], &CountingAllocator>, TryCloneError> =
            Rc::try_clone_from_ref_in(&items[..], &alloc);
        assert!(matches!(res, Err(TryCloneError::Other(_))));
        assert_eq!(alloc.allocations(), 1);
        assert_eq!(alloc.deallocations(), 1);
    }
}
