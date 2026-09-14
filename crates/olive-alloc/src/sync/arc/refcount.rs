//! Direct, fallible manipulation of an [`Arc`](super::Arc)'s strong reference
//! count through a raw pointer — without constructing an intermediate
//! [`Arc`] handle.

use core::mem::ManuallyDrop;
use core::sync::atomic::Ordering;

use crate::alloc::{Allocator, Global};

use super::Arc;
use super::conversion::TryArcError;
use super::pointers::{checked_increment, is_last_strong};

// ---------------------------------------------------------------------------
// Allocator-generic forms (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Increments the strong count of the allocation backing `ptr` without
    /// constructing an [`Arc`] wrapper.
    ///
    /// Parametrized over the allocator to match [`Self::try_decrement_strong_count_in`].
    ///
    /// # Errors
    ///
    /// Returns [`TryArcError::OutOfBounds`] if the strong counter has reached
    /// [`MAX_REFCOUNT`](super::pointers::MAX_REFCOUNT) and cannot be incremented
    /// further.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Arc<T>::into_raw`] or
    ///   [`Arc<T>::into_raw_with_allocator`] and must satisfy the layout required
    ///   by [`Arc<T>::from_raw_in`].
    /// - `ptr` must point to a block allocated by `alloc`.
    /// - The `Arc` must be valid — the strong count must not be 0.
    #[inline]
    pub unsafe fn try_increment_strong_count_in(
        ptr: *const T,
        alloc: &A,
    ) -> Result<(), TryArcError> {
        // NOTE: taking `alloc` by reference avoids paying for an allocator clone
        // that this operation does not need and reduces the caller need to clone
        // the allocator.
        // The reconstituted handle is wrapped in `ManuallyDrop` to prevent an
        // unintentional refcount decrement.
        // The allocator reference also helps avoid allocator leaks.
        // SAFETY: caller guarantees `ptr` is a live `Arc` allocation backed by
        // `alloc`.
        let me = unsafe { ManuallyDrop::new(Arc::from_raw_in(ptr, alloc)) };
        let inner = me.inner();

        // A plain increment publishes nothing and consumes nothing, so relaxed
        // semantics suffice — there is no data dependency to order.
        // `fetch_update` retries on contention and rejects at MAX_REFCOUNT.
        match inner
            .strong
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, checked_increment)
        {
            Ok(_) => Ok(()),
            Err(_) => Err(TryArcError::OutOfBounds),
        }
    }

    /// Decrements the strong count of the allocation backing `ptr`. When the
    /// count reaches zero the value is dropped and the block is freed.
    ///
    /// Parameterized over the allocator so the correct deallocator is used when
    /// the last strong reference goes away.
    ///
    /// # Errors
    ///
    /// Returns [`TryArcError::OutOfBounds`] if the strong counter is already
    /// zero (unbalanced decrement).
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Arc<T>::into_raw`] or
    ///   [`Arc<T>::into_raw_with_allocator`] and must satisfy the layout required
    ///   by [`Arc<T>::from_raw_in`].
    /// - `ptr` must point to a block allocated by `alloc`.
    /// - The `Arc` must be valid — the strong count must not be 0.
    /// - Decrementing must not leave other owned `Arc`s without a unit to
    ///   decrement.
    /// - This method can be used to free the [`Arc`] and its backing storage.
    #[inline]
    pub unsafe fn try_decrement_strong_count_in(
        ptr: *const T,
        alloc: &A,
    ) -> Result<(), TryArcError> {
        // NOTE: taking `alloc` by reference avoids paying for an allocator clone
        // that this operation does not need and reduces the caller need to clone
        // the allocator.
        // The reconstituted handle is wrapped in `ManuallyDrop` to prevent an
        // unintentional refcount decrement.
        // The allocator reference also helps avoid allocator leaks.
        // SAFETY: caller guarantees `ptr` is a live `Arc` allocation backed by
        // `alloc`.
        let mut me = unsafe { ManuallyDrop::new(Arc::from_raw_in(ptr, alloc)) };
        let inner = me.inner();

        // `fetch_update` atomically rejects a zero counter (no transient
        // underflow to usize::MAX) and retries on contention. The success
        // ordering is Release, mirroring `Drop for Arc`: it orders our prior
        // use of the payload before the eventual destruction, pairing with the
        // Acquire fence below.
        match inner
            .strong
            .fetch_update(Ordering::Release, Ordering::Relaxed, |cur| {
                #[allow(clippy::arithmetic_side_effects, reason = "asserted cur > 0")]
                {
                    cur.checked_sub(1)
                }
            }) {
            Ok(prev) => {
                #[allow(clippy::arithmetic_side_effects, reason = "asserted prev > 0")]
                if is_last_strong(prev - 1) {
                    // We were the last strong reference. An Acquire fence
                    // completes the release-acquire pair, ordering all prior
                    // writes to the payload (by any thread that held a strong
                    // ref) before the value destruction.
                    core::sync::atomic::fence(Ordering::Acquire);
                    // We are simulating an Arc drop here.
                    unsafe { Arc::drop_slow(&mut me) };
                }
                Ok(())
            }
            Err(_) => Err(TryArcError::OutOfBounds),
        }
    }
}

// ---------------------------------------------------------------------------
// Global-allocator forms (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized> Arc<T, Global> {
    /// Increments the strong count of the allocation backing `ptr` without
    /// constructing an [`Arc`] wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`TryArcError::OutOfBounds`] if the strong counter has reached
    /// [`MAX_REFCOUNT`](super::pointers::MAX_REFCOUNT) and cannot be incremented
    /// further without hitting the reserved `usize::MAX` sentinel.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Arc<T>::into_raw`] or
    ///   [`Arc<T>::into_raw_with_allocator`] and must satisfy the layout required
    ///   by [`Arc<T>::from_raw`].
    /// - `ptr` must point to a block allocated by the global allocator.
    /// - The `Arc` must be valid — the strong count must not be 0.
    /// - Decrementing must not leave other owned `Arc`s without a unit to
    ///   decrement.
    #[inline]
    pub unsafe fn try_increment_strong_count(ptr: *const T) -> Result<(), TryArcError> {
        // SAFETY: caller guarantees `ptr` is a live `Arc` allocation.
        unsafe { Self::try_increment_strong_count_in(ptr, &Global) }
    }

    /// Decrements the strong count of the allocation backing `ptr`. When the
    /// count reaches zero the value is dropped and the block is freed.
    ///
    /// # Errors
    ///
    /// Returns [`TryArcError::OutOfBounds`] if the strong counter is already
    /// zero (unbalanced decrement).
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Arc<T>::into_raw`] or
    ///   [`Arc<T>::into_raw_with_allocator`] and must satisfy the layout required
    ///   by [`Arc<T>::from_raw`].
    /// - `ptr` must point to a block allocated by the global allocator.
    /// - The `Arc` must be valid — the strong count must not be 0.
    /// - This method can be called to release the `Arc` and backing storage,
    ///   similar to calling [`Arc<T>::from_raw`] and dropping the value.
    #[inline]
    pub unsafe fn try_decrement_strong_count(ptr: *const T) -> Result<(), TryArcError> {
        // SAFETY: caller guarantees `ptr` is a live `Arc` allocation backed by
        // the global allocator.
        unsafe { Self::try_decrement_strong_count_in(ptr, &Global) }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::test_helpers::LocalCountingAlloc;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc as StdArc;

    /// A payload that records how many times it was dropped, so a test can prove
    /// the final strong release actually destroyed the value exactly once.
    /// Single-threaded local fixture, hence `Rc` (no cross-thread sharing).
    struct DropRecorder(Rc<Cell<usize>>);
    impl DropRecorder {
        fn new(counter: Rc<Cell<usize>>) -> Self {
            Self(counter)
        }
    }
    impl Drop for DropRecorder {
        fn drop(&mut self) {
            // `Cell` allows interior mutation through a shared reference.
            self.0.set(self.0.get() + 1);
        }
    }

    // --- Happy paths ---------------------------------------------------------

    #[test]
    fn increment_bumps_strong_count() {
        use core::sync::atomic::Ordering as Ord;
        let arc = Arc::try_new(7i32).unwrap();
        let raw = Arc::into_raw(arc);

        // Read the initial strong count directly from the allocation header.
        let inner = unsafe { super::super::pointers::data_get_ptr::<i32>(raw) };
        assert_eq!(unsafe { (*inner).strong.load(Ord::Relaxed) }, 1);

        // Bump the count: the raw pointer now represents two owners.
        unsafe { Arc::<i32>::try_increment_strong_count(raw) }.unwrap();
        // Confirm the count went up (read directly to avoid creating a handle).
        assert_eq!(unsafe { (*inner).strong.load(Ord::Relaxed) }, 2);

        // Bring it back down through the public API: release both references.
        unsafe { Arc::<i32>::try_decrement_strong_count(raw) }.unwrap();
        unsafe { Arc::<i32>::try_decrement_strong_count(raw) }.unwrap();
        // If we reach here without a heap-corruption abort or leak, the teardown was clean.
    }

    #[test]
    fn decrement_to_zero_drops_payload_and_frees_block() {
        // Use a recording payload so we can prove the final release actually ran
        // its destructor exactly once (and therefore freed the block).
        let drops = Rc::new(Cell::new(0));
        let arc = Arc::try_new(DropRecorder::new(drops.clone())).unwrap();
        let raw = Arc::into_raw(arc);

        // Release the single strong reference directly through the raw pointer.
        unsafe { Arc::<DropRecorder>::try_decrement_strong_count(raw) }.unwrap();
        // The payload was destroyed exactly once; the block is freed as part of
        // the same teardown (verified implicitly — no double-free or leak).
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn generic_forms_match_global_behaviour() {
        let drops = Rc::new(Cell::new(0));
        let alloc_counter = StdArc::new(crate::test_helpers::DropCounter::new());
        let alloc = LocalCountingAlloc::new(alloc_counter);
        let arc = Arc::try_new_in(DropRecorder::new(drops.clone()), alloc).unwrap();

        let (raw, alloc_out) = Arc::into_raw_with_allocator(arc);
        // `try_*_in` borrows the allocator handle, so it stays usable across
        // calls; only `from_raw_in` consumes an owned copy.
        unsafe { Arc::<DropRecorder, _>::try_increment_strong_count_in(raw, &alloc_out) }.unwrap();
        // Borrow the payload through a temporary that clones the handle.
        let probe_alloc = alloc_out.clone();
        let probe = unsafe { Arc::from_raw_in(raw, probe_alloc) };
        assert_eq!(Arc::strong_count(&probe), 2);
        drop(probe);

        unsafe { Arc::<DropRecorder, _>::try_decrement_strong_count_in(raw, &alloc_out) }.unwrap();
        // Final release destroyed the payload exactly once.
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn error_variant_displays() {
        let oob = TryArcError::OutOfBounds;
        let mut s = std::string::String::new();
        std::fmt::Write::write_fmt(&mut s, format_args!("{oob}")).unwrap();
        assert_eq!(s, "reference count out of bounds");
        assert!(core::error::Error::source(&oob).is_none());
    }
}
