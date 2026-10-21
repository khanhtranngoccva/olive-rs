use core::{
    alloc::Layout,
    mem::{self, MaybeUninit},
    ops::DerefMut,
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
            Ok(mut b) => {
                b.deref_mut().write(x);
                Ok(unsafe { mem::transmute::<Box<MaybeUninit<T>>, Self>(b) })
            }
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
