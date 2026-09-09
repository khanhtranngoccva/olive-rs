//! Query methods for [`Arc`](super::Arc).
//!
//! All query functions here are **associated functions** taking `&Self` rather
//! than inherent methods on the receiver. This keeps autoref-based deref
//! coercion methods from being shadowed.

use core::ptr;

use super::{Arc, ArcInner};
use crate::alloc::Allocator;

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Gets a shared raw pointer to the underlying data.
    ///
    /// The pointer has write provenance from the live allocation, so it can be
    /// used in splitting and reconstitution operations.
    // TODO: test as_ptr provenance behavior by testing the splitting and
    // reconstitution API
    #[must_use]
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr: *mut ArcInner<T> = this.ptr.as_ptr();
        // SAFETY: an `Arc` always owns a live allocation whose header is valid
        // while `this` exists; projecting the `value` field yields the payload
        // slot within that block.
        // The pointer carries write provenance to be usable for splitting into
        // raw parts.
        unsafe { &raw mut (*ptr).value }
    }

    /// Gets a shared reference to the allocator backing this `Arc`.
    #[must_use]
    #[inline]
    pub const fn allocator(this: &Self) -> &A {
        &this.alloc
    }

    /// Determines if two `Arc` pointers point to the same allocation.
    ///
    /// This ignores the metadata comparison, matching std's `Arc::ptr_eq`.
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::addr_eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Returns the number of strong [`Arc`] pointers to this allocation.
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        Self::inner(this).strong()
    }

    /// Returns the number of weak (`Weak`) pointers to this allocation,
    /// excluding the implicit weak reference held by all strong pointers.
    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        Self::inner(this).weak().saturating_sub(1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::Global;

    /// Reads the payload behind an `Arc` without relying on `Deref` (which has
    /// not landed yet). Only valid while the payload is initialized.
    fn peek<T, A: Allocator>(arc: &Arc<T, A>) -> &T {
        // SAFETY: the Arc owns a live allocation whose payload is initialized.
        #[allow(
            clippy::needless_borrow,
            reason = "Miri does not allow implicit autoref"
        )]
        unsafe {
            &(&*arc.ptr.as_ptr()).value
        }
    }

    #[test]
    fn as_ptr_points_at_payload() {
        let arc = Arc::try_new(42u32).unwrap();
        let p: *const u32 = Arc::as_ptr(&arc);
        assert_eq!(unsafe { *p }, 42);
        // The pointer addresses the payload slot, not the counter header:
        // writing through it (sole owner) must be observable via the handle.
        unsafe { *(p as *mut u32) = 7 };
        assert_eq!(peek(&arc), &7);
    }

    #[test]
    fn allocator_returns_backing_handle() {
        let drops = std::rc::Rc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::LocalCountingAlloc::new(drops.clone());
        let arc = Arc::try_new_in(1i32, alloc).unwrap();
        // The returned reference must alias the exact handle stored inside the
        // Arc: reading a field through it observes the same state as the
        // original (here: the shared drop counter is still alive).
        let got: &crate::test_helpers::LocalCountingAlloc = Arc::allocator(&arc);
        let _ = std::format!("{got:?}");
        assert_eq!(drops.get(), 0);
        drop(arc);
        // Dropping the Arc destroys its embedded allocator handle exactly once.
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn allocator_default_is_global() {
        let arc = Arc::try_new(1i32).unwrap();
        let _: &Global = Arc::allocator(&arc);
    }

    #[test]
    fn ptr_eq_same_allocation_true_distinct_false() {
        // TODO: `clone` has not landed yet, so build a second handle on the same
        // block by duplicating its fields WITHOUT bumping the counters. Both
        // handles must then be suppressed from dropping (`ManuallyDrop`) and
        // torn down manually exactly once, keeping the strong count balanced.
        use core::mem::ManuallyDrop;
        let a = ManuallyDrop::new(Arc::try_new(1u8).unwrap());
        let b = ManuallyDrop::new(Arc {
            ptr: a.ptr,
            alloc: Global,
            _marker: core::marker::PhantomData,
        });
        assert!(Arc::ptr_eq(&a, &b));
        let c = Arc::try_new(2u8).unwrap();
        assert!(!Arc::ptr_eq(&a, &c));
        // Tear down the shared block exactly once through one handle: read the
        // Arc out of its `ManuallyDrop` slot (bitwise copy; no refcount effect)
        // and drop that owned value.
        use core::ops::Deref;
        unsafe { drop(ptr::read(a.deref())) };
    }

    #[test]
    fn strong_and_weak_counts_track_ownership() {
        let arc = Arc::try_new(1i32).unwrap();
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
        // Internal invariant: the raw weak cell includes the implicit weak ref.
        assert_eq!(arc.inner().weak(), 1);
    }
}
