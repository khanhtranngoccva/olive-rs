//! Implementations of [`String`] APIs that query string information.

use core::{mem, ptr};

use olive_core::alloc::Allocator;

use super::String;
use crate::{alloc::Global, vec::Vec};

// FIXME: into_raw_parts and into_raw_parts_with_allocator is missing

impl String {
    /// Reconstructs a `String` from the components of its underlying buffer:
    /// a raw pointer allocated with [`Global`], a length, and a capacity.
    ///
    /// # Safety
    ///
    /// See [`Vec::from_raw_parts`](crate::vec::Vec::from_raw_parts) for the
    /// full list of preconditions. In addition, the first `length` bytes must
    /// be valid UTF-8.
    pub unsafe fn from_raw_parts(ptr: *mut u8, length: usize, capacity: usize) -> Self {
        // SAFETY: preconditions passed to the caller.
        let buf = unsafe { Vec::<u8, Global>::from_raw_parts(ptr, length, capacity) };
        debug_assert!(core::str::from_utf8(&buf).is_ok());
        Self { buf }
    }
}

impl<A: Allocator> String<A> {
    /// Same as [`from_raw_parts`](Self::from_raw_parts), but takes a generic
    /// allocator.
    ///
    /// # Safety
    ///
    /// See [`Vec::from_raw_parts_in`](crate::vec::Vec::from_raw_parts_in) for
    /// the full list of preconditions. In addition, the first `length` bytes
    /// must be valid UTF-8. This condition is unchecked.
    pub unsafe fn from_raw_parts_in(
        ptr: *mut u8,
        length: usize,
        capacity: usize,
        alloc: A,
    ) -> Self {
        // SAFETY: preconditions passed to the caller.
        let buf = unsafe { Vec::<u8, A>::from_raw_parts_in(ptr, length, capacity, alloc) };
        debug_assert!(core::str::from_utf8(&buf).is_ok());
        Self { buf }
    }

    /// Leaks the string and its allocator instance, returning a string reference
    /// over the allocator's lifetime.
    ///
    /// The allocation is intentionally leaked; the caller owns the memory forever.
    /// The underlying bytes remain valid UTF-8.
    ///
    /// This method does not try to shrink the underlying allocation.
    pub fn leak<'a>(self) -> &'a mut str
    where
        // Unlike the original String implementation, this implementation allows arbitrary
        // allocators to deal with complex allocation scenarios. The string must *not* move
        // out of allocator scope.
        A: 'a,
    {
        let (ptr, len, _cap, alloc) = unsafe { self.buf.into_raw_parts_with_alloc() };
        // Must leak the allocator to prevent the backing storage from being lost.
        mem::forget(alloc);
        // SAFETY: `ptr` points to `len` bytes of valid UTF-8 allocated by
        // `alloc`; reinterpreting the fat pointer as `str` preserves the
        // invariant and keeps the same allocator for eventual deallocation.
        unsafe {
            let str_ptr: *mut str = ptr::slice_from_raw_parts_mut(ptr, len) as *mut str;
            str_ptr.as_mut().expect("pointer is non-nul l")
        }
    }
}
