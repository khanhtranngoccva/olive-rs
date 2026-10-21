//! Conversions impls for [`Box`].

use core::alloc::Layout;
use core::any::Any;
use core::error::Error;
use core::pin::Pin;
use core::ptr::{self};

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
    /// This conversion does not allocate on the heap.
    fn from(s: Box<str, A>) -> Self {
        unsafe {
            let len = s.len();
            let (ptr, alloc) = Box::into_non_null_with_allocator(s);
            let reinterpreted = ptr::slice_from_raw_parts_mut(ptr.as_ptr().cast::<u8>(), len);
            Box::from_raw_in(reinterpreted, alloc)
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
        let (ptr, alloc) = Box::into_non_null_with_allocator(boxed_slice);
        let arr_ptr = ptr.cast::<[T; N]>();
        Box::from_non_null_in(arr_ptr, alloc)
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

impl<T, A: Allocator> Box<T, A> {
    /// Returns the value held in the box, deallocating the box's memory
    /// with its allocator.
    ///
    /// The value is moved out and is not dropped here; its ownership
    /// passes to the caller.
    #[inline]
    pub fn into_inner(b: Self) -> T {
        let (ptr, alloc) = Box::into_non_null_with_allocator(b);
        // SAFETY: `ptr` points to the fully initialized `T` the box owned.
        let value = unsafe { ptr.read() };
        if size_of::<T>() != 0 {
            // SAFETY: `ptr` denotes the live block previously allocated
            // via `alloc` with `Layout::new::<T>()`, and is not used
            // again afterwards.
            unsafe { alloc.deallocate(ptr.cast(), Layout::new::<T>()) };
        }
        value
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

#[cfg(test)]
mod tests {
    #![allow(clippy::vec_init_then_push)]

    extern crate std;
    use super::*;
    use crate::alloc::CountingAllocator;
    use core::fmt;

    #[test]
    fn test_box_str_to_bytes() {
        let bs: Box<str> = Box::try_clone_from_ref("xyz").unwrap();
        let bytes: Box<[u8]> = Box::from(bs);
        assert_eq!(&bytes[..], b"xyz");
    }

    #[test]
    fn test_box_slice_try_from() {
        let bs: Box<[i32]> = Box::try_from_slice(&[1, 2, 3]).unwrap();
        let arr: Box<[i32; 3]> = Box::try_from(bs).unwrap();
        assert_eq!(arr[0], 1);
        assert_eq!(arr[2], 3);

        let bs: Box<[i32]> = Box::try_from_slice(&[1, 2]).unwrap();
        let result: Result<Box<[i32; 3]>, Box<[i32]>> = Box::try_from(bs);
        assert!(result.is_err());
    }

    #[test]
    fn test_box_into_inner() {
        // `into_inner` must move the value out and deallocate the block
        // through the box's own allocator.
        let alloc = CountingAllocator::new();
        let b: Box<u32, &CountingAllocator> = Box::try_new_in(42u32, &alloc).unwrap();
        let value = Box::into_inner(b);
        assert_eq!(value, 42);
        assert_eq!(alloc.allocations(), 1, "one allocation");
        assert_eq!(
            alloc.deallocations(),
            1,
            "into_inner must deallocate the block"
        );
    }

    #[test]
    fn test_box_any_downcast() {
        // Build a `Box<dyn Any>` by boxing the concrete value and unsize-coercing
        // through a raw fat pointer (our `Box::try_new` is sized-only).
        let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
        let raw: *mut i32 = Box::into_raw(boxed_i32);
        let fat: *mut dyn Any = raw as *mut dyn Any;
        let b: Box<dyn Any, Global> = unsafe { Box::from_raw_in(fat, Global) };
        let recovered = b.downcast::<i32>().unwrap();
        assert_eq!(*recovered, 42);

        let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
        let raw: *mut i32 = Box::into_raw(boxed_i32);
        let fat: *mut dyn Any = raw as *mut dyn Any;
        let b: Box<dyn Any, Global> = unsafe { Box::from_raw_in(fat, Global) };
        assert!(b.downcast::<f64>().is_err());
    }

    #[test]
    fn test_box_any_send_downcast() {
        // Same exercise as `test_box_any_downcast`, but through the
        // `dyn Any + Send` impl.
        let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
        let raw: *mut i32 = Box::into_raw(boxed_i32);
        let fat: *mut (dyn Any + Send) = raw as *mut (dyn Any + Send);
        let b: Box<dyn Any + Send, Global> = unsafe { Box::from_raw_in(fat, Global) };
        let recovered = b.downcast::<i32>().unwrap();
        assert_eq!(*recovered, 42);

        let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
        let raw: *mut i32 = Box::into_raw(boxed_i32);
        let fat: *mut (dyn Any + Send) = raw as *mut (dyn Any + Send);
        let b: Box<dyn Any + Send, Global> = unsafe { Box::from_raw_in(fat, Global) };
        assert!(b.downcast::<f64>().is_err());
    }

    #[test]
    fn test_box_any_send_sync_downcast() {
        // Same exercise through the `dyn Any + Send + Sync` impl.
        let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
        let raw: *mut i32 = Box::into_raw(boxed_i32);
        let fat: *mut (dyn Any + Send + Sync) = raw as *mut (dyn Any + Send + Sync);
        let b: Box<dyn Any + Send + Sync, Global> = unsafe { Box::from_raw_in(fat, Global) };
        let recovered = b.downcast::<i32>().unwrap();
        assert_eq!(*recovered, 42);

        let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
        let raw: *mut i32 = Box::into_raw(boxed_i32);
        let fat: *mut (dyn Any + Send + Sync) = raw as *mut (dyn Any + Send + Sync);
        let b: Box<dyn Any + Send + Sync, Global> = unsafe { Box::from_raw_in(fat, Global) };
        assert!(b.downcast::<f64>().is_err());
    }

    #[derive(Debug)]
    struct MyErr(u32);
    impl fmt::Display for MyErr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "my error {}", self.0)
        }
    }
    impl Error for MyErr {}

    #[test]
    fn test_box_error_downcast() {
        // Box the concrete error and unsize-coerce through a raw fat pointer.
        let boxed: Box<MyErr, Global> = Box::try_new(MyErr(0)).unwrap();
        let raw: *mut MyErr = Box::into_raw(boxed);
        let fat: *mut dyn Error = raw as *mut dyn Error;
        let e: Box<dyn Error, Global> = unsafe { Box::from_raw_in(fat, Global) };
        let recovered = e.downcast::<MyErr>().unwrap();
        let _ = recovered; // just verify it works
    }

    #[test]
    fn test_box_error_send_downcast() {
        // Box the concrete error and unsize-coerce through a raw fat pointer.
        let boxed: Box<MyErr, Global> = Box::try_new(MyErr(7)).unwrap();
        let raw: *mut MyErr = Box::into_raw(boxed);
        let fat: *mut (dyn Error + Send) = raw as *mut (dyn Error + Send);
        let e: Box<dyn Error + Send, Global> = unsafe { Box::from_raw_in(fat, Global) };
        let recovered = e.downcast::<MyErr>().unwrap();
        assert_eq!(recovered.0, 7);
    }

    #[test]
    fn test_box_error_send_sync_downcast() {
        // Box the concrete error and unsize-coerce through a raw fat pointer.
        let boxed: Box<MyErr, Global> = Box::try_new(MyErr(9)).unwrap();
        let raw: *mut MyErr = Box::into_raw(boxed);
        let fat: *mut (dyn Error + Send + Sync) = raw as *mut (dyn Error + Send + Sync);
        let e: Box<dyn Error + Send + Sync, Global> = unsafe { Box::from_raw_in(fat, Global) };
        let recovered = e.downcast::<MyErr>().unwrap();
        assert_eq!(recovered.0, 9);
    }
}
