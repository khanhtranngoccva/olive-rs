//! Query methods for [`Arc`](super::Arc).
//!
//! All query functions here are **associated functions** taking `&Self` rather
//! than inherent methods on the receiver. This keeps autoref-based deref
//! coercion methods from being shadowed.

use core::ptr;

use super::pointers::is_dangling_weak;
use super::{Arc, ArcInner, Weak};
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

    /// Gets an approximation of the number of strong ([`Arc`]) pointers to this
    /// allocation.
    ///
    /// # Accuracy
    ///
    /// Due to implementation details, the returned value can be off by 1 in
    /// either direction when other threads are manipulating any [`Arc`]s or
    /// [`Weak`]s pointing to the same allocation.
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        Self::inner(this).strong()
    }

    /// Gets an approximation of the number of weak ([`Weak`]) pointers to this
    /// allocation, excluding the implicit weak reference held by all strong
    /// pointers.
    ///
    /// # Accuracy
    ///
    /// Due to implementation details, the returned value can be off by 1 in
    /// either direction when other threads are manipulating any [`Arc`]s or
    /// [`Weak`]s pointing to the same allocation.
    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        Self::inner(this).weak().saturating_sub(1)
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Gets a shared raw pointer to the underlying data.
    ///
    /// The pointer may be dangling, or may be uninitialized if strong references have
    /// all vanished. In either case, it must not be dereferenced.
    ///
    /// A weak reference that never referred to an allocation (from
    /// [`Weak::new`](super::Weak::new)) yields the deliberately misaligned
    /// dangling sentinel address, which can never collide with a real payload
    /// address.
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        let ptr = self.ptr.as_ptr();

        if is_dangling_weak(ptr) {
            // If the pointer is dangling, we return the sentinel directly. This
            // cannot be a valid payload address, as the payload is at least as
            // aligned as ArcInner (usize).
            ptr as *const T
        } else {
            // SAFETY: if `is_dangling_weak` returns false, then the pointer is
            // dereferenceable. The payload may have been dropped at this point,
            // and we have to maintain provenance, so use raw pointer
            // manipulation.
            unsafe { &raw mut (*ptr).value }
        }
    }

    /// Gets a shared reference to the allocator backing this `Weak`.
    #[must_use]
    #[inline]
    pub const fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Determines if two `Weak` pointers point to the same allocation.
    ///
    /// This method does not deal with fat pointer metadata.
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Gets an approximation of the number of strong ([`Arc`]) pointers pointing
    /// to this allocation.
    ///
    /// If `self` was created using [`Weak::new`](super::Weak::new), this will
    /// return 0.
    ///
    /// # Accuracy
    ///
    /// Due to implementation details, the returned value can be off by 1 in
    /// either direction when other threads are manipulating any [`Arc`]s or
    /// [`Weak`]s pointing to the same allocation.
    #[inline]
    pub fn strong_count(&self) -> usize {
        // A dangling weak owns no allocation, so there are no strong pointers.
        self.inner().map_or(0, |inner| inner.strong())
    }

    /// Gets an approximation of the number of [`Weak`] pointers pointing to this
    /// allocation, excluding the implicit weak reference held by all strong
    /// pointers.
    ///
    /// If `self` was created using [`Weak::new`](super::Weak::new), or if there
    /// are no remaining strong pointers, this will return 0.
    ///
    /// # Accuracy
    ///
    /// Due to implementation details, the returned value can be off by 1 in
    /// either direction when other threads are manipulating any `Arc`s or
    /// `Weak`s pointing to the same allocation.
    #[inline]
    pub fn weak_count(&self) -> usize {
        let Some(inner) = self.inner() else {
            // Dangling weak: no allocation, hence no weak pointers.
            return 0;
        };
        // With no strong pointers left the payload has been dropped and only the
        // implicit weak ref (if any) remains; report 0 to match std's contract.
        if inner.strong() == 0 {
            0
        } else {
            inner.weak().saturating_sub(1)
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

    // --- Weak query methods --------------------------------------------------
    //
    // TODO(downgrade): the live-allocation cases below need a way to obtain a
    // non-dangling `Weak` from an `Arc`, which requires `Arc::downgrade` (or
    // `Weak::try_downgrade`) to land first. Until then only the dangling paths
    // are exercisable without fabricating handles and corrupting the refcount.
    // Replace these TODO stubs with real tests once downgrade is available.

    #[test]
    fn weak_as_ptr_dangling_returns_sentinel() {
        // A weak that never referred to an allocation yields the misaligned
        // sentinel address, which can never be a real payload address.
        let w: Weak<u32, Global> = Weak::new();
        let p: *const u32 = w.as_ptr();
        assert_ne!(p.addr(), 0);
        // It must differ from any live payload address (same element type so the
        // comparison is well-typed).
        let arc = Arc::try_new(42u32).unwrap();
        assert_ne!(w.as_ptr(), Arc::as_ptr(&arc));
    }

    // TODO(downgrade): test that a live weak's `as_ptr` projects to the same
    // payload slot as the corresponding `Arc::as_ptr` and reads the value.

    #[test]
    fn weak_allocator_returns_backing_handle() {
        // The returned reference aliases the exact handle stored inside the Weak.
        let drops = std::rc::Rc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::LocalCountingAlloc::new(drops.clone());
        let w: Weak<u64, _> = Weak::new_in(alloc);
        let got: &crate::test_helpers::LocalCountingAlloc = w.allocator();
        let _ = std::format!("{got:?}");
        assert_eq!(drops.get(), 0);
        drop(w);
        // Dropping the Weak destroys its embedded allocator handle exactly once.
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn weak_allocator_default_is_global() {
        let w: Weak<i32, Global> = Weak::new();
        let _: &Global = w.allocator();
    }

    #[test]
    fn weak_ptr_eq_dangling_equal_to_each_other_and_not_to_live() {
        // Two dangling weaks share the same sentinel address, so they compare
        // equal; neither compares equal to a live allocation's weak-equivalent
        // (here compared against a live `Arc`'s payload-derived identity via a
        // second dangling vs. live distinction).
        let a: Weak<u8, Global> = Weak::new();
        let b: Weak<u8, Global> = Weak::new();
        assert!(a.ptr_eq(&b));
        // A dangling weak's address is the sentinel; a live block's inner
        // pointer is a real aligned address, so they cannot be equal.
        let arc = Arc::try_new(1u8).unwrap();
        let live_inner_addr = arc.ptr.as_ptr().addr();
        let dang_addr = a.ptr.as_ptr().addr();
        assert_ne!(live_inner_addr, dang_addr);
    }

    // TODO(downgrade): test that two weaks over the same allocation compare
    // equal via `ptr_eq`, while weaks over distinct allocations do not.

    #[test]
    fn weak_strong_count_dangling_is_zero() {
        // A weak from `Weak::new` owns no allocation, so there are no strong
        // pointers to count.
        let w: Weak<i32, Global> = Weak::new();
        assert_eq!(w.strong_count(), 0);
    }

    #[test]
    fn weak_weak_count_dangling_is_zero() {
        // Likewise a dangling weak has no weak pointers to report.
        let w: Weak<i32, Global> = Weak::new();
        assert_eq!(w.weak_count(), 0);
    }

    // TODO(downgrade): with a live weak (obtained via `Arc::downgrade`) assert
    // `strong_count()` equals the Arc's strong count and `weak_count()` equals
    // the Arc's weak count; after all strongs drop, both must read 0.
}
