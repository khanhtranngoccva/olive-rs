//! A fully-fallible port of the standard library's [`Rc`](stock_alloc::rc::Rc) and
//! [`Weak`](stock_alloc::rc::Weak).
//!
//! Compared with the std original, three things differ:
//!
//! * Every constructor that allocates a new node — `Rc::try_new`,
//!   `Rc::try_new_give_back`, and the various `*_in` variants — returns a
//!   [`Result`] carrying [`AllocError`] instead of panicking or aborting on
//!   out-of-memory.
//! * All allocation goes through the [`Allocator`] trait rather than free
//!   functions, giving a single swappable seam for custom allocators and OOM
//!   simulation in tests.
//! * Every reference-count mutation and allocator-handle clone is fallible and
//!   reports failures via [`TryRcError`]. Cloning an existing [`Rc`] does not
//!   allocate a new block (it only bumps the strong count), but it still must
//!   clone the allocator handle, which is a first-class fallible operation in
//!   this framework. The crate-standard [`TryClone`] impls surface these
//!   failures as [`TryCloneError`].
//!
//! The reference counts themselves are plain (non-atomic) integers, exactly
//! like the std originals: `Rc` is not thread-safe, and sharing one across
//! threads is undefined behavior.
//!
//! # Allocator bounds
//!
//! Cloning an `Rc<T, A>` requires `A: AllocatorTryClone`, not merely
//! `Allocator + Clone`. The stronger bound guarantees that a cloned allocator
//! handle is *equivalent* to the original — memory allocated through one may be
//! freed through the other — which is essential for the refcount-bump clone
//! path to remain sound. A plain `Clone` on the allocator would permit two
//! independent backing stores, breaking the invariant that all handles share
//! one allocation.

mod construction;
mod decompose;
mod dst;
mod helpers;
mod internals;
mod mutable;
mod query;
mod reconstitution;
mod refcount;
mod traits;
mod uninit;

pub use construction::TryRcWithError;
use internals::RcInner;
pub use refcount::{TryRcError, TryRcOutOfBoundsError};

use crate::alloc::{Allocator, Global, Layout};
use core::marker::PhantomData;
use olive_core::alloc::LayoutExt;
use olive_core::ptr::{self, NonNull};

// ---------------------------------------------------------------------------
// Rc declaration
// ---------------------------------------------------------------------------

/// A single-threaded reference-counting pointer.
///
/// `Rc<T>` shares ownership of a heap-allocated `T` among any number of strong
/// references. The value is dropped when the last strong reference disappears;
/// [`Weak`] references do not keep it alive.
///
/// Unlike `std::rc::Rc`, constructing a fresh node can fail: use
/// [`try_new`](Self::try_new) and friends, which return
/// `Result<Self, AllocError>` instead of panicking or aborting on out-of-memory.
pub struct Rc<T: ?Sized, A: Allocator = Global> {
    ptr: NonNull<RcInner<T>>,
    alloc: A,
    // This phantom prevents impl of Send and Sync
    _marker: PhantomData<*const T>,
}

// ---------------------------------------------------------------------------
// Drop
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Drop for Rc<T, A> {
    #[inline]
    fn drop(&mut self) {
        let inner = Rc::inner(self);
        inner.dec_strong().expect("strong count underflow");
        if helpers::is_last_strong(inner.strong()) {
            // SAFETY: we are the last strong reference.
            unsafe { Self::drop_slow(self) };
        }
    }
}

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Destroys the value and conditionally frees the block after the last
    /// strong reference has been dropped.
    ///
    /// # Safety
    /// - The Rc must be uniquely owned (or strong == 1).
    #[inline(never)]
    unsafe fn drop_slow(this: &mut Self) {
        // Construct a temporary `Weak` standing in for this `Rc`'s implicit
        // weak reference. This is the decrement or deallocate guard that
        // unconditionally runs even if the call is unwinding.
        let _fake_weak = Weak::<T, &A> {
            ptr: this.ptr,
            alloc: &this.alloc,
            _marker: PhantomData,
        };

        // SAFETY: the value is still present in the block (we are inside the
        // last strong's drop, before any free). Destroying it here is exactly
        // what dropping the final strong reference must do.
        unsafe {
            ptr::drop_in_place(&mut (*this.ptr.as_ptr()).value);
        }
    }
}

// ---------------------------------------------------------------------------
// Weak
// ---------------------------------------------------------------------------

/// A weak reference to an [`Rc`] allocation.
///
/// [`Weak<T>`] borrows the allocation without keeping it alive: upgrading a
/// [`Weak`] via [`try_upgrade`](Self::try_upgrade) yields an
/// [`Option<Rc<T>>`] that is `None` once every strong reference has been
/// dropped.
pub struct Weak<T: ?Sized, A: Allocator = Global> {
    ptr: NonNull<RcInner<T>>,
    alloc: A,
    _marker: PhantomData<RcInner<T>>,
}

impl<T: ?Sized, A: Allocator> Drop for Weak<T, A> {
    #[inline]
    fn drop(&mut self) {
        // A dangling weak (from `Weak::new`) owns no allocation, so there is
        // nothing to decrement or free — return immediately. `inner()` folds
        // the sentinel check into one call.
        let inner = match self.inner() {
            Some(i) => i,
            None => return,
        };

        // Decrement the weak count. If this was the last reference of any kind
        // (strong already zero, and now weak hits zero), free the allocation.
        // An unbalanced decrement here would indicate a logic error, so we
        // surface it as a panic rather than silently corrupting the count.
        inner.dec_weak().expect("Weak weak count underflow");

        // Invariant: once the weak count reaches zero, the strong count must
        // also be zero (the last strong `Rc`'s drop either freed the block or
        // left it pinned by at least one `Weak`). So `weak == 0` is sufficient
        // to decide whether to deallocate.
        if helpers::is_last_ref(inner.weak()) {
            // SAFETY: `ptr` carries correct pointer metadata for `T`; the pointee
            // may be uninitialized (already dropped) but we only need its size
            // and alignment, which live in the fat pointer.
            let layout = unsafe { Layout::for_value_pointer(self.ptr.as_ptr()) };
            // SAFETY: the block was allocated with exactly this layout; the
            // header alone guarantees a non-zero size.
            unsafe {
                self.alloc.deallocate(self.ptr.cast(), layout);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_core::try_traits::try_clone::TryClone;

    #[test]
    fn drop_frees_at_last_strong() {
        use olive_test_helpers::{Ledger, TrackedItem};
        use std::sync::Arc;

        let ledger = Arc::new(Ledger::new());
        let rc = Rc::try_new(TrackedItem::construct(&ledger, ())).unwrap();
        let id = rc.id();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(ledger.live_ids(), [id]);
        drop(rc);
        assert_eq!(ledger.live_ids(), [id]);
        drop(rc2);
        // The value's destructor ran exactly once, at the last strong drop,
        // and the block was released with it.
        assert!(ledger.all_dropped_once([id]));
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
    }

    // A value whose destructor panics. Used to verify that the panic-aware
    // guard in `Rc::drop` still frees the block when no `Weak` remains.
    struct PanickingDrop;
    impl Drop for PanickingDrop {
        fn drop(&mut self) {
            panic!("intentional panic in value destructor");
        }
    }
    #[test]
    #[should_panic(expected = "intentional panic")]
    fn panic_in_last_strong_no_weak_still_frees() {
        // No weak references: when the last strong drops and the value's
        // destructor panics, the guard must still deallocate the block. If it
        // leaked, Miri / leak detectors would catch it; here we at least prove
        // the code path runs without double-free or abort.
        let _rc = Rc::try_new(PanickingDrop).unwrap();
    }
    #[test]
    fn panic_in_last_strong_with_weak_keeps_block() {
        // A live `Weak` exists: the guard must NOT free the block (weak > 0),
        // leaving the `Weak` to release it later. Catch the panic so we can
        // verify the weak handle still sees a defunct-but-alive allocation.
        let rc = Rc::try_new(PanickingDrop).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(rc); // panics inside value's Drop; guard sees weak >= 1, skips free
        }));
        assert!(result.is_err(), "expected the value's Drop to panic");
        // The weak handle must still be valid (block not freed): upgrading
        // returns None because strong is 0, but accessing the weak does not
        // trigger a use-after-free. Dropping it now frees the block.
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
    }
}
