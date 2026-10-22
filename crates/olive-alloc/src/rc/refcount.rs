use core::fmt::{self, Debug, Display, Formatter};
use core::marker::PhantomData;
use core::mem::ManuallyDrop;

use super::helpers;
use super::{Rc, Weak};
use crate::alloc::{AllocError, Allocator, Global};
use olive_core::alloc::AllocatorTryClone;
use olive_core::try_traits::{TryClone, TryCloneError};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by fallible reference-counting operations on [`Rc`] and
/// [`Weak`].
///
/// Every operation that mutates a refcount or clones an allocator handle can
/// fail with one of these variants. The caller decides whether to propagate
/// the error, panic via `.expect(...)` (not recommended unless the call is
/// not supposed to fail), or recover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryRcError {
    /// A counter increment would exceed `usize::MAX` or a decrement would
    /// underflow below zero.
    ///
    /// This can arise from a logic error (unbalanced inc/dec), adversarial
    /// misuse of the raw pointer APIs, or — through the safe API alone — from
    /// [`core::mem::forget`]ing enough `Rc`s that their skipped `Drop` leaves
    /// the strong count stranded near `usize::MAX`; any further increment then
    /// overflows. Such leaks are rare in practice but do make this variant
    /// reachable without undefined behavior.
    OutOfBounds,
    /// Cloning the allocator handle failed.
    CloneAlloc(TryCloneError),
}

impl From<TryCloneError> for TryRcError {
    #[inline]
    fn from(e: TryCloneError) -> Self {
        Self::CloneAlloc(e)
    }
}

impl Display for TryRcError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds => write!(f, "reference count out of bounds"),
            Self::CloneAlloc(e) => write!(f, "allocator clone failed: {e}"),
        }
    }
}

impl From<AllocError> for TryRcError {
    #[inline]
    fn from(e: AllocError) -> Self {
        Self::CloneAlloc(TryCloneError::Alloc(e))
    }
}

impl core::error::Error for TryRcError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::OutOfBounds => None,
            Self::CloneAlloc(e) => Some(e),
        }
    }
}

/// Error returned by fallible reference-count operations that only mutate a
/// counter.
///
/// A counter increment would exceed `usize::MAX` or a decrement would underflow
/// below zero. This can arise from a logic error (unbalanced inc/dec),
/// adversarial misuse of the raw pointer APIs, or — through the safe API alone —
/// from [`core::mem::forget`]ing enough `Rc`s that their skipped `Drop` strands
/// the strong count near `usize::MAX`, so a further increment overflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryRcOutOfBoundsError;

impl Display for TryRcOutOfBoundsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("reference count out of bounds")
    }
}

impl core::error::Error for TryRcOutOfBoundsError {}

impl From<TryRcOutOfBoundsError> for TryRcError {
    #[inline]
    fn from(_: TryRcOutOfBoundsError) -> Self {
        Self::OutOfBounds
    }
}

// ---------------------------------------------------------------------------
// Rc TryClone
// ---------------------------------------------------------------------------

// The `TryClone` impl requires `A: AllocatorTryClone` (not merely
// `Allocator + Clone`) so that the cloned allocator handle is guaranteed to be
// equivalent to the original — a prerequisite for the refcount-bump clone to
// remain sound. Both the allocator clone and the counter bump are fallible;
// either failure surfaces as [`TryCloneError`] here.
impl<T: ?Sized, A: AllocatorTryClone> TryClone for Rc<T, A> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Bump the strong count and return a new handle sharing the same
        // allocation. The allocator is cloned via `AllocatorTryClone`.
        let alloc = A::try_clone(&self.alloc)?;
        let inner = Self::inner(self);
        inner
            .inc_strong()
            .map_err(|_| TryCloneError::Other("strong count out of bounds"))?;
        Ok(Rc {
            ptr: self.ptr,
            alloc,
            _marker: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Unsafe Rc global block
// ---------------------------------------------------------------------------

impl<T: ?Sized> Rc<T, Global> {
    /// Increments the strong count of the allocation backing `ptr` without
    /// constructing an `Rc` wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the strong count would overflow
    /// `usize`.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Rc<T>::into_raw`] or
    ///   [`Rc<T>::into_raw_with_allocator`] and must satisfy the layout required by [`Rc<T>::from_raw`].
    /// - `ptr` must point to a block allocated by the global allocator.
    /// - The `Rc` must be valid - the strong count must not be 0.
    #[inline]
    pub unsafe fn try_increment_strong_count(ptr: *const T) -> Result<(), TryRcOutOfBoundsError> {
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation.
        unsafe { Self::try_increment_strong_count_in(ptr, &Global) }
    }

    /// Decrements the strong count of the allocation backing `ptr`. When the
    /// count reaches zero the value is dropped and the block is freed.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the strong count is already zero
    /// (unbalanced decrement).
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Rc<T>::into_raw`] or
    ///   [`Rc<T>::into_raw_with_allocator`] and must satisfy the layout required by [`Rc<T>::from_raw`].
    /// - `ptr` must point to a block allocated by the global allocator.
    /// - The `Rc` must be valid - the strong count must not be 0.
    /// - This method can be called to release the Rc and backing storage, similar to
    ///   calling [`Rc<T>::from_raw`] and dropping the value.
    #[inline]
    pub unsafe fn try_decrement_strong_count(ptr: *const T) -> Result<(), TryRcOutOfBoundsError> {
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation backed by
        // the global allocator.
        unsafe { Self::try_decrement_strong_count_in(ptr, &Global) }
    }
}

// ---------------------------------------------------------------------------
// Unsafe Rc generic block
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Same as [`try_increment_strong_count`](Self::try_increment_strong_count),
    /// but parameterized over the allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the strong count would overflow
    /// `usize`.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Rc<T>::into_raw`] or
    ///   [`Rc<T>::into_raw_with_allocator`] and must satisfy the layout required by
    ///   [`Rc<T>::from_raw_in`].
    /// - `ptr` must point to a block allocated by the `alloc`.
    /// - The `Rc` must be valid - the strong count must not be 0.
    #[inline]
    pub unsafe fn try_increment_strong_count_in(
        ptr: *const T,
        alloc: &A,
    ) -> Result<(), TryRcOutOfBoundsError> {
        // NOTE: taking `alloc` by reference avoids paying for an allocator clone
        // that this operation does not need and reduces the caller need to clone
        // the allocator.
        // The reconstituted handle is wrapped in `ManuallyDrop` to prevent an
        // unintentional refcount decrement.
        // The allocator reference also helps avoid allocator leaks.
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation backed by
        // `alloc`. Wrapping in `ManuallyDrop` prevents a panic during unwinding
        // from decrementing the refcount.
        let me = unsafe { ManuallyDrop::new(Rc::from_raw_in(ptr, &alloc)) };
        let inner = Rc::inner(&me);
        if inner.strong() == 0 {
            return Err(TryRcOutOfBoundsError);
        }
        inner.inc_strong()?;
        Ok(())
    }

    /// Same as [`try_decrement_strong_count`](Self::try_decrement_strong_count),
    /// but parameterized over the allocator so the correct deallocator is used
    /// when the last strong reference goes away.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the strong count is already zero
    /// (unbalanced decrement).
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Rc<T>::into_raw`] or
    ///   [`Rc<T>::into_raw_with_allocator`] and must satisfy the layout required by
    ///   [`Rc<T>::from_raw_in`].
    /// - `ptr` must point to a block allocated by the `alloc`.
    /// - The `Rc` must be valid - the strong count must not be 0.
    /// - This method can be used to free the [`Rc`] and its backing storage.
    #[inline]
    pub unsafe fn try_decrement_strong_count_in(
        ptr: *const T,
        alloc: &A,
    ) -> Result<(), TryRcOutOfBoundsError> {
        // NOTE: taking `alloc` by reference avoids paying for an allocator clone
        // that this operation does not need and reduces the caller need to clone
        // the allocator.
        // The reconstituted handle is wrapped in `ManuallyDrop` to prevent an
        // unintentional refcount decrement.
        // The allocator reference also helps avoid allocator leaks.
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation backed by
        // `alloc`. Wrapping in `ManuallyDrop` prevents a panic during unwinding
        // from arbitrarily lowering the refcount.
        let mut me = unsafe { ManuallyDrop::new(Rc::from_raw_in(ptr, &alloc)) };
        let inner = Rc::inner(&me);
        inner.dec_strong()?;
        if helpers::is_last_strong(inner.strong()) {
            // SAFETY: we are the last strong reference.
            unsafe { Rc::drop_slow(&mut me) };
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Conversion to Weak (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: AllocatorTryClone> Rc<T, A> {
    /// Borrows an [`Rc`] as a [`Weak`] pointer.
    ///
    /// This does not increment the strong count, so the resulting `Weak` will
    /// not prevent the value from being dropped once all strong references are
    /// gone.
    ///
    /// The operation is fallible because it must clone the allocator handle
    /// (via [`AllocatorTryClone`]) to attach to the new `Weak`, and counter
    /// mutation can report out-of-bounds conditions. Cloning an allocator
    /// handle is a first-class fallible operation in this framework.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if the weak count would overflow
    /// `usize`, or [`TryRcError::CloneAlloc`] if cloning the allocator handle
    /// fails.
    #[inline]
    pub fn try_downgrade(this: &Self) -> Result<Weak<T, A>, TryRcError> {
        let alloc = A::try_clone(&this.alloc)?;
        let inner = Self::inner(this);
        inner.inc_weak().map_err(TryRcError::from)?;
        Ok(Weak {
            ptr: this.ptr,
            alloc,
            _marker: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Weak block
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: AllocatorTryClone> Weak<T, A> {
    /// Attempts to upgrade the `Weak` reference to an [`Rc`].
    ///
    /// Returns `Ok(None)` if there are no strong (`Rc`) references left, in
    /// which case the value has been dropped.
    ///
    /// The operation is fallible because it must clone the allocator handle
    /// (via [`AllocatorTryClone`]) to attach to the new `Rc`, and counter
    /// mutation can report out-of-bounds conditions. Cloning an allocator
    /// handle is a first-class fallible operation in this framework.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if a counter would overflow
    /// `usize`, or [`TryRcError::CloneAlloc`] if cloning the allocator handle
    /// fails.
    #[inline]
    pub fn try_upgrade(&self) -> Result<Option<Rc<T, A>>, TryRcError> {
        // A dangling weak (from `Weak::new`) never referred to an allocation,
        // so it can never be upgraded. Checking this first also avoids ever
        // dereferencing the sentinel address below.
        let inner = match self.inner() {
            Some(i) => i,
            None => return Ok(None),
        };
        if inner.strong() == 0 {
            return Ok(None);
        }

        let alloc = A::try_clone(&self.alloc)?;
        // Restore the strong count. The allocation is guaranteed to stay alive
        // because this `Weak` itself pins it. We only bump strong; the weak
        // count already accounts for this handle (it was incremented when the
        // `Weak` was created via `try_downgrade`).
        inner.inc_strong().map_err(TryRcError::from)?;
        Ok(Some(Rc {
            ptr: self.ptr,
            alloc,
            _marker: PhantomData,
        }))
    }
}

// ---------------------------------------------------------------------------
// Weak TryClone
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: AllocatorTryClone> TryClone for Weak<T, A> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let alloc = A::try_clone(&self.alloc)?;
        if helpers::is_dangling_weak(self.ptr.as_ptr()) {
            return Ok(Weak {
                ptr: self.ptr,
                alloc,
                _marker: PhantomData,
            });
        }
        unsafe {
            let inner = self.ptr.as_ptr();
            (*inner)
                .inc_weak()
                .map_err(|_| TryCloneError::Other("weak count out of bounds"))?;
        }
        Ok(Weak {
            ptr: self.ptr,
            alloc,
            _marker: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::CountingAllocator;
    use olive_test_helpers::{Ledger, TrackedItem};
    use std::string::String;
    use std::sync::Arc;

    #[test]
    fn rc_try_clone_shares_allocation() {
        let rc = Rc::try_new(String::from("hello")).unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
        assert_eq!(&*rc2, "hello");
    }

    #[test]
    fn weak_does_not_keep_alive() {
        let rc = Rc::try_new(1).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_try_clone() {
        let rc = Rc::try_new(3).unwrap();
        let w1 = Rc::try_downgrade(&rc).unwrap();
        let w2 = w1.try_clone().unwrap();
        assert_eq!(Rc::weak_count(&rc), 2);
        assert!(w1.ptr_eq(&w2));
        assert_eq!(Rc::as_ptr(&rc), w1.as_ptr());
        drop(rc);
        assert!(w1.try_upgrade().unwrap().is_none());
        assert!(w2.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_upgrade_restores_strong() {
        let rc = Rc::try_new(7).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let upgraded = weak.try_upgrade().unwrap().unwrap();
        assert_eq!(Rc::strong_count(&upgraded), 2);
        assert_eq!(*upgraded, 7);
        drop(upgraded);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    /// Regression: cloning a dangling `Weak` (from `Weak::new`) must not
    /// dereference the sentinel address. The clone must behave as another
    /// dangling weak — it upgrades to `None` and compares pointer-equal —
    /// and dropping both must remain a no-op.

    #[test]
    fn weak_new_try_clone_stays_dangling() {
        let w: Weak<i32> = Weak::new();
        let c = w.try_clone().unwrap();
        assert!(c.try_upgrade().unwrap().is_none());
        assert!(w.ptr_eq(&c));
    }

    #[test]
    fn try_increment_strong_count_roundtrip() {
        let rc = Rc::try_new(9).unwrap();
        let raw = Rc::into_raw(rc);
        // SAFETY: `raw` was produced by `into_raw` and strong == 1.
        unsafe { Rc::try_increment_strong_count(raw) }.unwrap();
        let rc2 = unsafe { Rc::from_raw(raw) };
        assert_eq!(Rc::strong_count(&rc2), 2);
        drop(rc2);
        // The remaining strong reference is consumed manually: the payload is
        // dropped and the block freed inside the decrement itself.
        // SAFETY: `raw` still refers to the live allocation, strong == 1.
        unsafe { Rc::try_decrement_strong_count(raw) }.unwrap();
    }

    #[test]
    fn try_decrement_strong_count_in_drops_payload_and_frees_block() {
        let ledger = Arc::new(Ledger::new());
        let alloc = CountingAllocator::new();
        let item = TrackedItem::construct(&ledger, 4u32);
        let id = item.id();
        let rc: Rc<TrackedItem<u32>, &CountingAllocator> = Rc::try_new_in(item, &alloc).unwrap();
        let (raw, rc_alloc) = Rc::into_raw_with_allocator(rc);
        assert_eq!(ledger.live_ids().len(), 1);
        // SAFETY: `raw` was produced by `into_raw_with_allocator`; this is the
        // sole strong reference, so the payload is dropped and the block freed.
        unsafe { Rc::try_decrement_strong_count_in(raw, rc_alloc) }.unwrap();
        assert!(ledger.live_ids().is_empty());
        assert_eq!(ledger.drop_count(id), 1);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_strong_count_ops_error_at_zero_strong() {
        let rc = Rc::try_new(1).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        drop(rc);
        let raw = weak.as_ptr();
        // SAFETY: `raw` comes from `into_raw`; the `weak` keeps the block
        // alive while the payload is dropped.
        unsafe { Rc::try_decrement_strong_count(raw) }.unwrap();
        // The allocation now has strong == 0: both raw operations must report
        // the imbalance instead of wrapping the counter.
        // SAFETY: `raw` still refers to the live (weak-pinned) allocation.
        assert!(matches!(
            unsafe { Rc::try_increment_strong_count(raw) },
            Err(TryRcOutOfBoundsError)
        ));
        // SAFETY: `raw` still refers to the live (weak-pinned) allocation.
        assert!(matches!(
            unsafe { Rc::try_decrement_strong_count(raw) },
            Err(TryRcOutOfBoundsError)
        ));
        drop(weak);
    }
}
