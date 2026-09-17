//! Query methods for [`Arc`](super::Arc).
//!
//! All query functions here are **associated functions** taking `&Self` rather
//! than inherent methods on the receiver. This keeps autoref-based deref
//! coercion methods from being shadowed.

use core::ptr;
use core::sync::atomic::Ordering;

use super::pointers::is_dangling_weak;
use super::{Arc, ArcInner, Weak};
use crate::alloc::Allocator;

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Gets a shared raw pointer to the underlying data.
    ///
    /// The pointer has write provenance from the live allocation, so it can be
    /// used in splitting and reconstitution operations (see
    /// [`into_raw`](super::Arc::into_raw) / [`from_raw`](super::Arc::from_raw)
    /// and so on).
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
        this.inner().strong.load(Ordering::Relaxed)
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
        let cnt = this.inner().weak.load(Ordering::Relaxed);
        // If the weak count is currently locked, the value of the
        // count was 0 just before taking the lock.
        if cnt == usize::MAX {
            0
        } else {
            cnt.saturating_sub(1)
        }
    }

    /// Returns `true` if there are no other [`Arc`] or [`Weak`] pointers to this
    /// allocation.
    #[inline]
    pub fn is_unique(this: &Self) -> bool {
        // Lock the weak pointer count if we appear to be the sole weak pointer
        // holder.
        //
        // The acquire label here ensures a happens-before relationship with any
        // writes to `strong` (in particular in `Weak::try_upgrade`) prior to decrements
        // of the `weak` count (via `Weak::drop`, which uses release). If the upgraded
        // weak ref was never dropped, the CAS here will fail so we do not care to synchronize.
        if this
            .inner()
            .weak
            .compare_exchange(1, usize::MAX, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            // This needs to be an `Acquire` to synchronize with the decrement of the `strong`
            // counter in `drop` -- the only access that happens when any but the last reference
            // is being dropped.
            let unique = this.inner().strong.load(Ordering::Acquire) == 1;

            // The release write here synchronizes with a read in `try_downgrade`,
            // effectively preventing the above read of `strong` from happening
            // after the write.
            this.inner().weak.store(1, Ordering::Release); // release the lock
            unique
        } else {
            false
        }
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
        self.inner()
            .map_or(0, |inner| inner.strong.load(Ordering::Relaxed))
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
        let weak = inner.weak.load(Ordering::Acquire);
        let strong = inner.strong.load(Ordering::Relaxed);
        if strong == 0 {
            0
        } else {
            // Since we observed that there was at least one strong pointer
            // after reading the weak count, we know that the implicit weak
            // reference (present whenever any strong references are alive)
            // was still around when we observed the weak count, and can
            // therefore safely subtract it.
            weak.saturating_sub(1)
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

    #[test]
    fn as_ptr_points_at_payload() {
        let arc = Arc::try_new(42u32).unwrap();
        let p: *const u32 = Arc::as_ptr(&arc);
        assert_eq!(unsafe { *p }, 42);
        // The pointer addresses the payload slot, not the counter header:
        // writing through it (sole owner) must be observable via the handle.
        unsafe { *(p as *mut u32) = 7 };
        assert_eq!(*arc, 7);
    }

    #[test]
    fn allocator_returns_backing_handle() {
        let drops = std::sync::Arc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::DropCountingAlloc::new(drops.clone());
        let arc = Arc::try_new_in(1i32, alloc).unwrap();
        // The returned reference must alias the exact handle stored inside the
        // Arc: reading a field through it observes the same state as the
        // original (here: the shared drop counter is still alive).
        let got: &crate::test_helpers::DropCountingAlloc = Arc::allocator(&arc);
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
        assert_eq!(arc.inner().weak.load(Ordering::Relaxed), 1);
    }

    // --- Weak query methods --------------------------------------------------
    //
    // Live-allocation cases use [`Arc::try_downgrade`] to obtain a non-dangling
    // `Weak`; dangling cases use `Weak::new`.

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

    #[test]
    fn weak_as_ptr_live_projects_to_same_payload_slot() {
        // A live weak (obtained by downgrading an Arc) projects `as_ptr` to the
        // exact same payload slot as the originating Arc, and reading through it
        // yields the current value.
        let arc = Arc::try_new(7i32).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        assert_eq!(weak.as_ptr(), Arc::as_ptr(&arc));
        // The payload is still initialized while a strong reference is alive.
        assert_eq!(unsafe { &*weak.as_ptr() }, &7);
    }

    #[test]
    fn weak_allocator_returns_backing_handle() {
        // The returned reference aliases the exact handle stored inside the Weak.
        let drops = std::sync::Arc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::DropCountingAlloc::new(drops.clone());
        let w: Weak<u64, _> = Weak::new_in(alloc);
        let got: &crate::test_helpers::DropCountingAlloc = w.allocator();
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

    #[test]
    fn weak_ptr_eq_live_same_allocation_true_distinct_false() {
        // Two weaks derived from the same Arc share an inner pointer, so they
        // compare equal; a weak from a different allocation does not.
        let arc = Arc::try_new(1u8).unwrap();
        let other = Arc::try_new(2u8).unwrap();
        let w1 = Arc::try_downgrade(&arc).unwrap();
        let w2 = Arc::try_downgrade(&arc).unwrap();
        let wo = Arc::try_downgrade(&other).unwrap();
        assert!(w1.ptr_eq(&w2));
        assert!(!w1.ptr_eq(&wo));
    }

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

    #[test]
    fn weak_counts_live_match_arc_and_zero_after_drop() {
        // A live weak reports the same strong/weak counts as its Arc (both
        // excluding the implicit weak ref). When strong count is 0, the
        // weak count also collapses to 0.
        let arc = Arc::try_new(1i32).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        assert_eq!(weak.strong_count(), Arc::strong_count(&arc));
        assert_eq!(weak.weak_count(), Arc::weak_count(&arc));
        assert_eq!(weak.strong_count(), 1);
        assert_eq!(weak.weak_count(), 1);
        drop(arc);
        assert_eq!(weak.strong_count(), 0);
        assert_eq!(weak.weak_count(), 0);
    }

    /// Differential parity test against `std::sync::{Arc, Weak}`: the count
    /// methods must agree with std in every phase of the lifecycle, including
    /// after the last strong reference has vanished. This is the regression
    /// net for the "implicit weak subtraction" logic in `Weak::weak_count`.
    #[test]
    fn weak_counts_parity_with_std_across_lifecycle() {
        use std::sync as stock;

        // Phase 1: fresh node, no weaks.
        let a = Arc::try_new(1i32).unwrap();
        let sa = stock::Arc::new(1i32);
        assert_eq!(Arc::strong_count(&a), stock::Arc::strong_count(&sa));
        assert_eq!(Arc::weak_count(&a), stock::Arc::weak_count(&sa));

        // Phase 2: two explicit weaks alive alongside the strong.
        let w1 = Arc::try_downgrade(&a).unwrap();
        let w2 = Arc::try_downgrade(&a).unwrap();
        let sw1 = stock::Arc::downgrade(&sa);
        let sw2 = stock::Arc::downgrade(&sa);
        for (ours, theirs) in [(&w1, &sw1), (&w2, &sw2)] {
            assert_eq!(ours.strong_count(), theirs.strong_count());
            assert_eq!(ours.weak_count(), theirs.weak_count());
        }
        assert_eq!(Arc::strong_count(&a), stock::Arc::strong_count(&sa));
        assert_eq!(Arc::weak_count(&a), stock::Arc::weak_count(&sa));

        // Phase 3: all strongs gone, only weaks remain. std collapses
        // `Weak::weak_count` to 0 once the last strong reference is dropped,
        // even though explicit weaks still pin the allocation; ours must match.
        drop(sa);
        drop(a);
        for (ours, theirs) in [(&w1, &sw1), (&w2, &sw2)] {
            assert_eq!(ours.strong_count(), theirs.strong_count());
            assert_eq!(ours.weak_count(), theirs.weak_count());
        }

        // Phase 4: one weak drops, the other survives.
        drop(sw1);
        drop(w1);
        assert_eq!(w2.strong_count(), sw2.strong_count());
        assert_eq!(w2.weak_count(), sw2.weak_count());

        // Phase 5: everything is gone.
        drop(sw2);
        drop(w2);
    }
}
