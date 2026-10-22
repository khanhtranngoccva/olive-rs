use super::helpers;
use super::internals::{RcInner, WeakInner};
use super::{Rc, Weak};
use olive_core::{alloc::Allocator, ptr};

// ---------------------------------------------------------------------------
// Rc query block (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Returns a shared reference to the allocation's internal [`RcInner`]
    /// header (strong/weak counters and payload).
    #[inline]
    pub(super) fn inner(this: &Self) -> &RcInner<T> {
        // SAFETY: an `Rc` always owns a live allocation whose header is valid
        // for reads while `self` exists.
        unsafe { &*this.ptr.as_ptr() }
    }

    /// Returns a **mutable** reference to the allocation's internal
    /// [`RcInner`] header.
    ///
    /// Prefer this over [`Self::inner`] when the caller intends to mutate the
    /// payload through the returned reference.
    ///
    /// # Safety
    ///
    /// - The pointer's strong count must be 1.
    #[inline]
    pub(super) unsafe fn inner_mut(this: &mut Self) -> &mut RcInner<T> {
        // SAFETY: an `Rc` always owns a live allocation whose header is valid
        // for reads and writes while `this` exists; the `&mut self` receiver
        // guarantees exclusive access for the call's duration.
        unsafe { &mut *this.ptr.as_ptr() }
    }

    /// Gets a shared raw pointer to the underlying data.
    ///
    /// The returned pointer carries write provenance derived from the live
    /// allocation, so it can be used in splitting and reconstitution
    /// operations (e.g. [`Self::into_raw_with_allocator`] / [`Self::into_raw`] /
    /// [`Self::from_raw_in`] / [`Self::from_raw`]).
    #[must_use]
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr: *mut RcInner<T> = this.ptr.as_ptr();
        unsafe { &raw mut (*ptr).value }
    }

    /// Gets a shared reference to the allocator backing this `Rc`.
    #[must_use]
    #[inline]
    pub const fn allocator(this: &Self) -> &A {
        &this.alloc
    }

    /// Determines if two `Rc` pointers point to the same allocation.
    ///
    /// This ignores the metadata comparison.
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::addr_eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Returns the number of strong [`Rc`] pointers to this allocation.
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        Self::inner(this).strong()
    }

    /// Returns the number of weak (`Weak`) pointers to this allocation,
    /// excluding the implicit weak reference held by each strong pointer.
    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        Self::inner(this).weak().saturating_sub(1)
    }

    /// Returns `true` if there are no other `Rc` or [`Weak`] pointers to this
    /// allocation.
    #[inline]
    pub fn is_unique(this: &Self) -> bool {
        Rc::weak_count(this) == 0 && Rc::strong_count(this) == 1
    }
}

// ---------------------------------------------------------------------------
// Weak query block (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Returns a [`WeakInner`] handle to the allocation's reference-count
    /// cells, or `None` if this handle is dangling (constructed via
    /// [`Weak::new`]).
    ///
    /// We are careful to *not* create a reference covering the "data" field, as
    /// the field may have been dropped in-place (e.g., when the last `Rc` was
    /// dropped while this `Weak` still pins the allocation). Only the two
    /// counter cells are referenced, which remain valid for the lifetime of the
    /// allocation.
    #[inline]
    pub(super) fn inner(&self) -> Option<WeakInner<'_>> {
        if helpers::is_dangling_weak(self.ptr.as_ptr()) {
            None
        } else {
            // SAFETY: a non-dangling `Weak` always owns a live allocation whose
            // counter cells are valid for reads/writes while `self` exists. The
            // payload (`value`) may have been dropped, but we never form a
            // reference to it here.
            Some(unsafe {
                let ptr = self.ptr.as_ptr();
                WeakInner {
                    strong: &(*ptr).strong,
                    weak: &(*ptr).weak,
                }
            })
        }
    }

    /// Gets a shared raw pointer to the underlying `T`.
    ///
    /// The pointer may be dangling, or may be uninitialized if strong references
    /// have all vanished. In either case, it must not be dereferenced.
    ///
    /// A weak that never referred to an allocation (from [`Weak::new`]) yields
    /// the deliberately misaligned dangling sentinel address, which can never
    /// collide with a real payload address.
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        let ptr = self.ptr.as_ptr();

        if helpers::is_dangling_weak(ptr) {
            // If the pointer is dangling, we return the sentinel directly. This cannot be
            // a valid payload address, as the payload is at least as aligned as RcInner (usize).
            ptr as *const T
        } else {
            // SAFETY: if is_dangling returns false, then the pointer is dereferenceable.
            // The payload may be dropped at this point, and we have to maintain provenance,
            // so use raw pointer manipulation.
            unsafe { &raw mut (*ptr).value }
        }
    }

    /// Gets a shared reference to the allocator backing this `Weak`.
    #[must_use]
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Determines if two `Weak` pointers point to the same allocation.
    ///
    /// This method does not deal with fat pointer metadata.
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_core::try_traits::try_clone::TryClone;
    use olive_test_helpers::CountingAllocator;

    #[test]
    fn unsized_as_ptr() {
        let arr = [10u8, 20];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let p: *const [u8] = Rc::as_ptr(&rc);
        let slice: &[u8] = unsafe { &*p };
        assert_eq!(slice.len(), 2);
        assert_eq!(slice, [10, 20]);
    }

    #[test]
    fn sized_as_ptr() {
        let rc = Rc::try_new(41u32).unwrap();
        let p: *const u32 = Rc::as_ptr(&rc);
        // SAFETY: `rc` holds the sole strong reference, so the payload is
        // live and initialized.
        assert_eq!(unsafe { *p }, 41);
    }

    #[test]
    fn rc_as_ptr_write_provenance_manual_write() {
        let rc = Rc::try_new(42u64).unwrap();
        let p: *const u64 = Rc::as_ptr(&rc);
        // Cast away constness — only sound because we hold the sole strong
        // reference (strong == 1, weak == 0) and the pointer's provenance
        // covers the payload slot.
        let mut_p = p as *mut u64;
        unsafe {
            ptr::write(mut_p, 7);
        }
        assert_eq!(*rc, 7);
        // Overwrite again via the same pointer to confirm repeated use is fine.
        unsafe {
            ptr::write_volatile(mut_p, 9);
        }
        assert_eq!(*rc, 9);
    }

    /// Same guarantee for unsized payloads: the fat pointer returned by
    /// `Rc::as_ptr` must be writable through its data word while metadata
    /// (the slice length) is preserved.
    #[test]
    fn rc_as_ptr_write_provenance_unsized_slice() {
        let arr = [1u8, 2, 3];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let p: *const [u8] = Rc::as_ptr(&rc);
        let mut_p = p as *mut [u8];
        unsafe {
            (*mut_p)[1] = 20;
        }
        assert_eq!(&*rc, [1, 20, 3]);
    }

    #[test]
    fn is_unique_true_only_when_no_other_refs() {
        let rc = Rc::try_new(1).unwrap();
        assert!(Rc::is_unique(&rc));

        let weak = Rc::try_downgrade(&rc).unwrap();
        // A live Weak disqualifies uniqueness even though strong == 1.
        assert!(!Rc::is_unique(&rc));
        drop(weak);
        assert!(Rc::is_unique(&rc));

        let rc2 = rc.try_clone().unwrap();
        assert!(!Rc::is_unique(&rc));
        drop(rc2);
        assert!(Rc::is_unique(&rc));
    }

    #[test]
    fn weak_as_ptr_dangling_returns_sentinel() {
        let w: Weak<i32> = Weak::new();
        let p = w.as_ptr();
        assert_eq!(p.addr(), usize::MAX);
        // The sentinel is misaligned relative to any non-ZST payload, so it
        // can never alias a real allocation.
        assert_ne!(p.addr() % align_of::<i32>(), 0);
    }

    #[test]
    fn weak_as_ptr_after_last_strong_is_not_sentinel() {
        let rc = Rc::try_new(1i32).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        drop(rc);
        let p = weak.as_ptr();
        // This weak did refer to a real allocation, so its data word is the
        // genuine (now dead) payload address, not the sentinel.
        assert_ne!(p.addr(), usize::MAX);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn rc_ptr_eq_compares_addresses_not_values() {
        let a = Rc::try_new(5).unwrap();
        let b = Rc::try_new(5).unwrap();
        // Equal payloads, distinct allocations.
        assert!(!Rc::ptr_eq(&a, &b));
        let shared = a.try_clone().unwrap();
        assert!(Rc::ptr_eq(&a, &shared));
    }

    #[test]
    fn rc_allocator_returns_backing_allocator() {
        let alloc = CountingAllocator::new();
        let rc: Rc<u32, &CountingAllocator> = Rc::try_new_in(1, &alloc).unwrap();
        // The handle handed back is the very allocator the Rc was built with.
        assert!(std::ptr::eq(*Rc::allocator(&rc), &alloc));
        assert_eq!(Rc::allocator(&rc).allocations(), 1);
        drop(rc);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn weak_ptr_eq_and_allocator() {
        let alloc = CountingAllocator::new();
        let rc: Rc<u32, &CountingAllocator> = Rc::try_new_in(3, &alloc).unwrap();
        let w1 = Rc::try_downgrade(&rc).unwrap();
        let w2 = Rc::try_downgrade(&rc).unwrap();
        let cloned = w1.try_clone().unwrap();
        assert!(w1.ptr_eq(&w2));
        assert!(w1.ptr_eq(&cloned));
        assert!(std::ptr::eq(*Weak::allocator(&w1), &alloc));
        // A dangling weak shares its sentinel only with other dangling weaks.
        let dangling: Weak<u32, &CountingAllocator> = Weak::new_in(&alloc);
        assert!(!w1.ptr_eq(&dangling));
        let dangling2 = dangling.try_clone().unwrap();
        assert!(dangling.ptr_eq(&dangling2));
    }
}
