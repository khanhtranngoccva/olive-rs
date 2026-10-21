use core::{
    alloc::Layout,
    mem::MaybeUninit,
    pin::Pin,
    ptr::{self, NonNull},
};

use crate::alloc::Global;

use super::Box;
use olive_core::{
    alloc::{AllocError, Allocator, AllocatorTryDefault, StaticAllocator},
    try_traits::{TryDefault, TryDefaultError},
};

// ---------------------------------------------------------------------------
// Global construction block (sized T)
// ---------------------------------------------------------------------------

impl<T> Box<T, Global> {
    /// Allocates memory on the global heap and places `x` into it.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new(x: T) -> Result<Self, AllocError> {
        Self::try_new_give_back(x).map_err(|(_given_back, err)| err)
    }

    /// Like [`try_new`](Self::try_new), but on allocation failure returns the
    /// original `x` back to the caller alongside the error, so the value is
    /// not dropped silently.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new_give_back(x: T) -> Result<Self, (T, AllocError)> {
        // Compiler optimization: do not monomorphize generic
        match Self::try_new_uninit() {
            Ok(b) => Ok(b.write(x)),
            Err(e) => Err((x, e)),
        }
    }

    /// Constructs a `Box<MaybeUninit<T>>` filled with zeros.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new_zeroed() -> Result<Box<MaybeUninit<T>>, AllocError> {
        // ZST optimization: no actual allocation needed.
        if size_of::<T>() == 0 {
            return Ok(Box {
                inner: NonNull::dangling(),
                alloc: Global,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = Global.allocate_zeroed(layout)?;

        Ok(Box {
            inner: ptr.cast(),
            alloc: Global,
        })
    }

    /// Constructs a `Box<MaybeUninit<T>>` containing uninitialized memory.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new_uninit() -> Result<Box<MaybeUninit<T>>, AllocError> {
        // ZST optimization: no actual allocation needed.
        if size_of::<T>() == 0 {
            return Ok(Box {
                inner: NonNull::dangling(),
                alloc: Global,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = Global.allocate(layout)?;

        Ok(Box {
            inner: ptr.cast(),
            alloc: Global,
        })
    }

    /// Allocates a new `Box<T>` containing `x` and pins it in place, returning
    /// a `Pin<Box<T>>`.
    ///
    /// If `T` does not implement `Unpin`, then `*boxed` will be pinned in memory
    /// and unable to be moved.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let p: core::pin::Pin<Box<i32>> = Box::try_pin(42).unwrap();
    /// assert_eq!(*p, 42);
    /// ```
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_pin(x: T) -> Result<Pin<Self>, AllocError> {
        Self::try_new(x).map(Box::into_pin)
    }

    /// Like [`try_pin`](Self::try_pin), but on allocation failure returns the
    /// unallocated `x` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_pin_give_back(x: T) -> Result<Pin<Self>, (T, AllocError)> {
        Self::try_new_give_back(x).map(Box::into_pin)
    }
}

// ---------------------------------------------------------------------------
// Generic construction block (sized T)
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Box<T, A> {
    /// Like [`try_new`](Self::try_new), but parameterized over the choice of allocator for the
    /// returned `Box`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_in(x: T, alloc: A) -> Result<Self, AllocError> {
        let b = Self::try_new_uninit_in(alloc)?;
        Ok(b.write(x))
    }

    /// Like [`try_new_give_back`](Self::try_new_give_back), but parameterized over the choice of
    /// allocator for the returned `Box`. On allocation failure returns the
    /// unallocated `x` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_give_back_in(x: T, alloc: A) -> Result<Self, (T, AllocError)> {
        match Self::try_new_uninit_in(alloc) {
            Ok(b) => Ok(b.write(x)),
            Err(e) => Err((x, e)),
        }
    }

    /// Like [`try_new_zeroed`](Self::try_new_zeroed), but parameterized over the choice of allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed_in(alloc: A) -> Result<Box<MaybeUninit<T>, A>, AllocError> {
        // ZST optimization: no actual allocation needed.
        if size_of::<T>() == 0 {
            return Ok(Box {
                inner: NonNull::dangling(),
                alloc,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = alloc.allocate_zeroed(layout)?;

        Ok(Box {
            inner: ptr.cast(),
            alloc,
        })
    }

    /// Like [`try_new_uninit`](Self::try_new_uninit), but parameterized over the choice of allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_uninit_in(alloc: A) -> Result<Box<MaybeUninit<T>, A>, AllocError> {
        if size_of::<T>() == 0 {
            return Ok(Box {
                inner: NonNull::dangling(),
                alloc,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = alloc.allocate(layout)?;

        Ok(Box {
            inner: ptr.cast(),
            alloc,
        })
    }

    /// Like [`try_pin`](Self::try_pin), but parametrized over the choice of allocator.
    ///
    /// Allocates a new `Box<T>` containing `x` and pins it in place,
    /// returning a `Pin<Box<T>>`.
    ///
    /// If `T` does not implement `Unpin`, then `*boxed` will be pinned in
    /// memory and unable to be moved.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    /// use olive_alloc::alloc::Global;
    ///
    /// let p: core::pin::Pin<Box<i32, Global>> = Box::try_pin_in(42, Global).unwrap();
    /// assert_eq!(*p, 42);
    /// ```
    #[inline]
    pub fn try_pin_in(x: T, alloc: A) -> Result<Pin<Box<T, A>>, AllocError>
    where
        A: StaticAllocator,
    {
        Self::try_new_in(x, alloc).map(Box::into_pin)
    }

    /// Like [`try_pin_give_back`](Self::try_pin_give_back), but parameterized over the choice of
    /// allocator for the returned `Box`. On allocation failure returns the
    /// unallocated `x` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_pin_give_back_in(x: T, alloc: A) -> Result<Pin<Box<T, A>>, (T, AllocError)>
    where
        A: StaticAllocator,
    {
        Self::try_new_give_back_in(x, alloc).map(Box::into_pin)
    }
}

impl<T: ?Sized, A: Allocator> Box<T, A> {
    /// Converts a `Box<T>` into a `Pin<Box<T>>`.
    ///
    /// This is also available via `From<Box<T>> for Pin<Box<T>>`. If the T is `!Unpin`,
    /// then the inner content will be pinned in memory and cannot be moved.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let b = Box::try_new(7).unwrap();
    /// let p = Box::into_pin(b);
    /// assert_eq!(*p, 7);
    /// ```
    #[inline]
    pub fn into_pin(b: Self) -> Pin<Self>
    where
        A: StaticAllocator,
    {
        // SAFETY: It's not possible to move or replace the insides of a `Pin<Box<T>>`
        // when `T: !Unpin`, and `A: StaticAllocator` guarantees the backing memory
        // stays valid, so it is safe to pin regardless of whether `T: Unpin`.
        unsafe { Pin::new_unchecked(b) }
    }
}

// ---------------------------------------------------------------------------
// Default / TryDefault
// ---------------------------------------------------------------------------

#[inline]
fn dangling_zero_len_slice<T>() -> NonNull<[T]> {
    // SAFETY: `NonNull::<T>::dangling()` is non-null (the alignment of any
    // type is at least 1), so the raw slice pointer built from it is
    // non-null.
    unsafe {
        NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(
            NonNull::<T>::dangling().as_ptr(),
            0,
        ))
    }
}

impl<T> Default for Box<[T]> {
    #[inline]
    fn default() -> Self {
        Self {
            inner: dangling_zero_len_slice::<T>(),
            alloc: Global,
        }
    }
}

impl<T, A: AllocatorTryDefault> TryDefault for Box<[T], A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self {
            inner: dangling_zero_len_slice::<T>(),
            alloc: A::try_default()?,
        })
    }
}

impl Default for Box<str> {
    #[inline]
    fn default() -> Self {
        // SAFETY: the dangling `[u8]` pointer is non-null, and `str`'s
        // metadata is the byte length, matching `[u8]`'s element count
        // (0 here), so the cast yields a valid non-null `*mut str`.
        let inner =
            unsafe { NonNull::new_unchecked(dangling_zero_len_slice::<u8>().as_ptr() as *mut str) };
        Self {
            inner,
            alloc: Global,
        }
    }
}

impl<A: AllocatorTryDefault> TryDefault for Box<str, A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        // SAFETY: as in `Default`, a valid non-null dangling `*mut str`.
        let inner =
            unsafe { NonNull::new_unchecked(dangling_zero_len_slice::<u8>().as_ptr() as *mut str) };
        Ok(Self {
            inner,
            alloc: A::try_default()?,
        })
    }
}

impl<T: TryDefault, A: AllocatorTryDefault> TryDefault for Box<T, A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        let uninit = Box::try_new_uninit_in(A::try_default()?).map_err(TryDefaultError::Alloc)?;
        let value = T::try_default()?;
        Ok(uninit.write(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use olive_test_helpers::{FailAlloc, FailDefaultAlloc, NoDefault};

    #[test]
    fn box_try_new_and_deref() {
        let b = Box::try_new(42i32).unwrap();
        assert_eq!(*b, 42);
    }

    #[test]
    fn box_try_new_ok() {
        let b = Box::try_new(99u64).unwrap();
        assert_eq!(*b, 99);
    }

    #[test]
    fn box_zst() {
        struct Zst;
        let b = Box::try_new(Zst).unwrap();
        // Should not allocate; should work fine.
        let _ = &b;
    }

    #[test]
    fn box_try_new_give_back_success() {
        let b = Box::try_new_give_back(42u32).unwrap();
        assert_eq!(*b, 42);
    }

    #[test]
    fn box_try_new_uninit() {
        let b: Box<MaybeUninit<u64>> = Box::<u64>::try_new_uninit().unwrap();
        let b = b.write(5);
        assert_eq!(*b, 5);
    }

    #[test]
    fn box_try_new_zeroed() {
        let b = Box::<u64>::try_new_zeroed().unwrap();
        // SAFETY: `try_new_zeroed` produces a zero-filled allocation, and
        // zero is a valid `u64`.
        let b = unsafe { b.assume_init() };
        assert_eq!(*b, 0);
    }

    #[test]
    fn box_try_pin_global() {
        let p: Pin<Box<u8>> = Box::try_pin(31u8).unwrap();
        assert_eq!(*p, 31);
    }

    #[test]
    fn box_try_pin_give_back_success() {
        let p: Pin<Box<u32>> = Box::try_pin_give_back(7u32).unwrap();
        assert_eq!(*p, 7);
    }

    #[test]
    fn box_try_new_in() {
        // Verify that Box works with an explicit `Global` allocator; the
        // stored allocator must be the same unit struct we passed in.
        let b = Box::try_new_in(55u16, Global).unwrap();
        assert_eq!(*b, 55);
        assert!(matches!(Box::allocator(&b), Global));
    }

    #[test]
    fn box_try_new_in_oom() {
        let e = Box::<u32, FailAlloc>::try_new_in(42, FailAlloc).unwrap_err();
        assert!(matches!(e, AllocError));
    }

    #[test]
    fn box_try_new_in_zst_under_fail_alloc() {
        // The ZST path skips the allocation entirely, so construction must
        // succeed even though every non-ZST allocation of `FailAlloc` fails.
        struct Zst;
        let b = Box::<Zst, FailAlloc>::try_new_in(Zst, FailAlloc).unwrap();
        let _ = &b;
    }

    #[test]
    fn box_try_new_give_back_in_success() {
        let b = Box::<u32, Global>::try_new_give_back_in(42u32, Global).unwrap();
        assert_eq!(*b, 42);
    }

    #[test]
    fn box_try_new_give_back_in_oom_returns_value() {
        // On OOM the value must come back inside the error pair instead of
        // being dropped silently.
        let (given_back, e) =
            Box::<u32, FailAlloc>::try_new_give_back_in(42, FailAlloc).unwrap_err();
        assert_eq!(given_back, 42);
        assert!(matches!(e, AllocError));
    }

    #[test]
    fn box_try_new_zeroed_in() {
        let b = Box::<u64, Global>::try_new_zeroed_in(Global).unwrap();
        // SAFETY: zero-initialized integers are valid.
        let b = unsafe { b.assume_init() };
        assert_eq!(*b, 0);
    }

    #[test]
    fn box_try_new_zeroed_in_oom() {
        let e = Box::<u64, FailAlloc>::try_new_zeroed_in(FailAlloc).unwrap_err();
        assert!(matches!(e, AllocError));
    }

    #[test]
    fn box_try_new_zeroed_in_zst_under_fail_alloc() {
        // As above: the zeroed ZST path never touches the allocator.
        struct Zst;
        let b = Box::<Zst, FailAlloc>::try_new_zeroed_in(FailAlloc).unwrap();
        let _ = &b;
    }

    #[test]
    fn box_try_new_uninit_in() {
        let b = Box::<u64, Global>::try_new_uninit_in(Global).unwrap();
        let b = b.write(120);
        assert_eq!(*b, 120);
    }

    #[test]
    fn box_try_new_uninit_in_oom() {
        let e = Box::<u64, FailAlloc>::try_new_uninit_in(FailAlloc).unwrap_err();
        assert!(matches!(e, AllocError));
    }

    #[test]
    fn box_try_pin() {
        let p: Pin<Box<u8, Global>> = Box::<u8, Global>::try_pin_in(31u8, Global).unwrap();
        assert_eq!(*p, 31);
    }

    #[test]
    fn box_try_pin_in_oom() {
        let e = Box::<u32, FailAlloc>::try_pin_in(42, FailAlloc).unwrap_err();
        assert!(matches!(e, AllocError));
    }

    #[test]
    fn box_try_pin_give_back_in_success() {
        let p: Pin<Box<u32, Global>> =
            Box::<u32, Global>::try_pin_give_back_in(7u32, Global).unwrap();
        assert_eq!(*p, 7);
    }

    #[test]
    fn box_try_pin_give_back_in_oom_returns_value() {
        let (given_back, e) =
            Box::<u32, FailAlloc>::try_pin_give_back_in(42, FailAlloc).unwrap_err();
        assert_eq!(given_back, 42);
        assert!(matches!(e, AllocError));
    }

    // Default / TryDefault.

    #[test]
    fn box_default_empty_slice() {
        let b = Box::<[u8]>::default();
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn box_default_empty_str() {
        let b = Box::<str>::default();
        assert_eq!(&*b, "");
    }

    #[test]
    fn box_try_default_matches_default() {
        assert_eq!(Box::<[u8]>::try_default().unwrap().len(), 0);
        assert_eq!(&*Box::<str>::try_default().unwrap(), "");
    }

    #[test]
    fn box_try_default_sized_uses_allocator() {
        let b = Box::<u32>::try_default().unwrap();
        assert_eq!(*b, 0);
    }

    #[test]
    fn box_try_default_linked_allocator() {
        // Dangling-pointer construction allocates nothing, so the failing
        // allocation path is never exercised; the default value is built
        // entirely from the `Global` singleton the allocator delegates to.
        let b = Box::<[u8], FailAlloc>::try_default().unwrap();
        assert_eq!(b.len(), 0);
        let s = Box::<str, FailAlloc>::try_default().unwrap();
        assert_eq!(&*s, "");
    }

    #[test]
    fn box_try_default_allocator_failure_propagates() {
        let e = Box::<[u8], FailDefaultAlloc>::try_default().unwrap_err();
        assert!(matches!(e, TryDefaultError::Other(_)));
        let e = Box::<str, FailDefaultAlloc>::try_default().unwrap_err();
        assert!(matches!(e, TryDefaultError::Other(_)));
    }

    #[test]
    fn box_try_default_sized_allocator_default_failure_propagates() {
        let e = Box::<u32, FailDefaultAlloc>::try_default().unwrap_err();
        assert!(matches!(e, TryDefaultError::Other(_)));
    }

    #[test]
    fn box_try_default_sized_value_failure_propagates() {
        // The `T::try_default()` branch: the value's fallible default fails
        // after the (successful) allocation, and the error propagates.
        let e = Box::<NoDefault<u32>, Global>::try_default().unwrap_err();
        assert!(matches!(e, TryDefaultError::Other(_)));
    }

    #[test]
    fn box_try_default_sized_alloc_failure_propagates() {
        // The allocation branch: `A::try_default()` succeeds (`FailAlloc`
        // defaults fine) but backing the value fails, surfacing as
        // `TryDefaultError::Alloc`.
        let e = Box::<u32, FailAlloc>::try_default().unwrap_err();
        assert!(matches!(e, TryDefaultError::Alloc(_)));
    }
}
