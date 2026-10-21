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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_uninit_write_and_assume_init() {
        // write(): initialize via the consuming helper.
        let b: Box<MaybeUninit<u64>> = Box::try_new_uninit().unwrap();
        let init = b.write(99u64);
        assert_eq!(*init, 99);

        // assume_init(): manually fill then reinterpret.
        let mut u: Box<MaybeUninit<u64>> = Box::try_new_uninit().unwrap();
        // invoke write() on the uninit object.
        (*u).write(1234u64);
        let init = unsafe { u.assume_init() };
        assert_eq!(*init, 1234);
    }

    #[test]
    fn test_box_new_uninit_slice_write_and_read() {
        // Allocate uninitialized, fill via `write`, then reinterpret as `[T]`.
        let mut bs: Box<[MaybeUninit<u64>]> = Box::try_new_uninit_slice(4).unwrap();
        assert_eq!(bs.len(), 4);
        for (slot, val) in bs.iter_mut().zip([10u64, 20, 30, 40]) {
            slot.write(val);
        }
        let init: Box<[u64]> = unsafe { bs.assume_init() };
        assert_eq!(*init, [10, 20, 30, 40]);
    }

    #[test]
    fn test_box_new_zeroed_slice_is_zeroed() {
        // Zero-filled buffer reinterpreted straight to `[u8]` must read all zeros.
        let zs: Box<[MaybeUninit<u8>]> = Box::try_new_zeroed_slice(16).unwrap();
        assert_eq!(zs.len(), 16);
        let init: Box<[u8]> = unsafe { zs.assume_init() };
        assert!(init.iter().all(|&b| b == 0));
    }
}
