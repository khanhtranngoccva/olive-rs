//! APIs for [`Box`] that deals with initializing uninitialized containers
//! or asserting them as such.

use super::Box;
use core::{mem::MaybeUninit, ops::DerefMut};
use olive_core::alloc::Allocator;

// ---------------------------------------------------------------------------
// Uninit initialization helpers
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Box<MaybeUninit<T>, A> {
    /// Writes `val` into the boxed slot and returns the initialized `Box<T, A>`.
    #[inline]
    pub fn write(mut self, val: T) -> Box<T, A> {
        self.deref_mut().write(val);
        unsafe { self.assume_init() }
    }

    /// Asserts that the contained value has been initialized and returns an
    /// owning `Box<T, A>`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the slot has been fully initialized.
    #[inline]
    pub unsafe fn assume_init(self) -> Box<T, A> {
        // SAFETY: `Box<T>` and `Box<MaybeUninit<T>>` have the same layout.
        unsafe { olive_core::mem::transmute_unchecked::<Self, Box<T, A>>(self) }
    }
}

impl<T, A: Allocator> Box<[MaybeUninit<T>], A> {
    /// Asserts that all elements of the boxed slice have been initialized and
    /// returns an owning `Box<[T], A>`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that every element in the slice has been fully
    /// initialized.
    #[inline]
    pub unsafe fn assume_init(self) -> Box<[T], A> {
        // SAFETY: `[MaybeUninit<T>]` and `[T]` are layout-compatible; only the
        // semantic initialization state differs.
        unsafe { olive_core::mem::transmute_unchecked::<Self, Box<[T], A>>(self) }
    }
}
