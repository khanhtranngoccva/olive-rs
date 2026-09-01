//! Conversions impls for [`Box`].

use core::any::Any;
use core::error::Error;
use core::mem;
use core::pin::Pin;
use core::ptr::{self, NonNull};

use super::Box;
use crate::alloc::{Allocator, Global, StaticAllocator};

// ---------------------------------------------------------------------------
// From / Into conversions
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> From<Box<T, A>> for Pin<Box<T, A>>
where
    A: StaticAllocator,
{
    /// Converts a `Box<T>` into a `Pin<Box<T>>`. If `T` does not implement [`Unpin`], then
    /// `*boxed` will be pinned in memory and unable to be moved.
    ///
    /// This conversion does not allocate on the heap and happens in place.
    ///
    /// This is also available via [`Box::into_pin`].
    fn from(boxed: Box<T, A>) -> Self {
        Box::into_pin(boxed)
    }
}

impl<A: Allocator> From<Box<str, A>> for Box<[u8], A> {
    /// Converts a `Box<str>` into a `Box<[u8]>`.
    ///
    /// This conversion does not allocate on the heap and happens in place.
    fn from(s: Box<str, A>) -> Self {
        unsafe {
            // Extract fields directly to avoid the `Sized` requirement.
            let inner = ptr::read(&s.inner);
            let alloc = ptr::read(&s.alloc);
            mem::forget(s);

            // SAFETY: a `str` is just a sequence of `u8`s with the same layout.
            // Reinterpret the fat pointer (address + length metadata) as
            // `NonNull<[u8]>`, which has identical representation.
            let nn_bytes: NonNull<[u8]> = mem::transmute(inner);
            Box::from_raw_in(nn_bytes.as_ptr(), alloc)
        }
    }
}

// ---------------------------------------------------------------------------
// TryFrom: Box<[T]> → Box<[T; N]>
// ---------------------------------------------------------------------------

/// Casts a boxed slice to a boxed array.
///
/// # Safety
///
/// `boxed_slice.len()` must be exactly `N`.
unsafe fn boxed_slice_as_array_unchecked<T, A: Allocator, const N: usize>(
    boxed_slice: Box<[T], A>,
) -> Box<[T; N], A> {
    debug_assert_eq!(boxed_slice.len(), N);

    unsafe {
        // Extract fields directly to avoid the `Sized` requirement.
        let inner = ptr::read(&boxed_slice.inner);
        let alloc = ptr::read(&boxed_slice.alloc);
        mem::forget(boxed_slice);

        // SAFETY: Pointer and allocator came from an existing box,
        // and our safety condition requires that the length is exactly `N`.
        // A slice of length N has the same layout as an array [T; N]; drop
        // the (redundant) length metadata by re-pointing at `[T; N]`.
        let arr_ptr: *mut [T; N] = inner.as_ptr() as *mut u8 as *mut [T; N];
        Box::from_raw_in(arr_ptr, alloc)
    }
}

impl<T, const N: usize> TryFrom<Box<[T], Global>> for Box<[T; N], Global> {
    type Error = Box<[T], Global>;

    /// Attempts to convert a `Box<[T]>` into a `Box<[T; N]>`.
    ///
    /// The conversion occurs in-place and does not require a new memory
    /// allocation.
    ///
    /// # Errors
    ///
    /// Returns the old `Box<[T]>` in the `Err` variant if
    /// `boxed_slice.len()` does not equal `N`.
    fn try_from(boxed_slice: Box<[T], Global>) -> Result<Self, Self::Error> {
        if boxed_slice.len() == N {
            Ok(unsafe { boxed_slice_as_array_unchecked(boxed_slice) })
        } else {
            Err(boxed_slice)
        }
    }
}

// ---------------------------------------------------------------------------
// Any downcasting
// ---------------------------------------------------------------------------

impl<A: Allocator> Box<dyn Any, A> {
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Any>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                // Extract the base address and allocator.
                let (base_addr, alloc) = Box::into_raw_with_allocator(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                // Casting the thin base address to `*mut T` reinterprets the
                // same address bits as a pointer to `T`, which is sound
                // because the allocation contains a valid `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Any + Send, A> {
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Any>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                // Extract the base address and allocator.
                let (base_addr, alloc) = Box::into_raw_with_allocator(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Any + Send + Sync, A> {
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Any>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                // Extract the base address and allocator.
                let (base_addr, alloc) = Box::into_raw_with_allocator(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Error, A> {
    /// Attempts to downcast the box to a concrete error type.
    pub fn downcast<T: Error + 'static>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                // Extract the base address and allocator.
                let (base_addr, alloc) = Box::into_raw_with_allocator(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Error + Send, A> {
    /// Attempts to downcast the box to a concrete error type.
    pub fn downcast<T: Error + 'static>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                // Extract the base address and allocator.
                let (base_addr, alloc) = Box::into_raw_with_allocator(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Error + Send + Sync, A> {
    /// Attempts to downcast the box to a concrete error type.
    pub fn downcast<T: Error + 'static>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                // Extract the base address and allocator.
                let (base_addr, alloc) = Box::into_raw_with_allocator(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

// ---------------------------------------------------------------------------
// Iterator forwarding for Box<I, A> where I: Iterator + ?Sized
// ---------------------------------------------------------------------------

impl<I: Iterator + ?Sized, A: Allocator> Iterator for Box<I, A> {
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (**self).next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        (**self).nth(n)
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        #[inline]
        fn some<T>(_: Option<T>, x: T) -> Option<T> {
            Some(x)
        }

        self.fold(None, some)
    }
}

impl<I: DoubleEndedIterator + ?Sized, A: Allocator> DoubleEndedIterator for Box<I, A> {
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> {
        (**self).next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        (**self).nth_back(n)
    }
}

impl<I: core::iter::FusedIterator + ?Sized, A: Allocator> core::iter::FusedIterator for Box<I, A> {}

impl<I: ExactSizeIterator + ?Sized, A: Allocator> ExactSizeIterator for Box<I, A> {}
