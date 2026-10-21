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
