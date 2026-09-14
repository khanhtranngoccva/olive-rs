//! Raw-pointer splitting and reconstitution for [`Arc`](super::Arc).
//!
//! These methods let an `Arc` (or [`Weak`]) be split into a raw pointer — plus
//! its allocator handle, for the generic forms — and later reconstituted from
//! that pointer. The pair is the escape hatch for storing an `Arc` in foreign
//! memory: the caller takes over ownership of the reference count carried by
//! the raw pointer and must eventually reconstitute it or manually deallocate
//! it.
//!
//! Like std's infallible counterparts, none of these operations can fail:
//! reconstitution merely moves the caller-supplied allocator handle into the
//! new pointer without cloning it, so no fallibility creeps in here.

use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ptr;

use crate::alloc::{Allocator, Global};

use super::pointers::{data_get_ptr, is_dangling_weak};
use super::{Arc, ArcInner, Weak};
use olive_core::ptr::NonNull;

// ---------------------------------------------------------------------------
// Arc: global-allocator forms
// ---------------------------------------------------------------------------

impl<T: ?Sized> Arc<T, Global> {
    /// Constructs a new `Arc<T>` from a raw pointer previously produced by
    /// [`Arc::into_raw`].
    ///
    /// # Safety
    ///
    /// * Creating an `Arc<T>` from a pointer other than one returned from
    ///   [`Arc::<T>::into_raw`](Arc::into_raw) or
    ///   [`Arc::into_raw_with_allocator`](Arc::into_raw_with_allocator) is
    ///   undefined behavior.
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Arc<U>` was constructed
    ///   through `Arc<T>` and then converted to `Arc<U>` through an [unsized
    ///   coercion](https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions).
    /// * Note that if `U` or `U`'s data pointer is not `T` but has the same size
    ///   and alignment, this is basically like transmuting references of
    ///   different types. See [`core::mem::transmute`] for more information on
    ///   what restrictions apply in this case.
    /// * The raw pointer must point to a block of memory allocated by the global
    ///   allocator.
    /// * The user of [`Arc::from_raw`] has to make sure a specific value of `T`
    ///   is only dropped once.
    #[inline]
    pub unsafe fn from_raw(p: *const T) -> Self {
        // SAFETY: caller guarantees `p` derives from `into_raw`; converting it
        // back restores the exact reference counts the original had.
        unsafe {
            let inner = data_get_ptr(p);
            Arc {
                ptr: NonNull::new_unchecked(inner as *mut ArcInner<T>),
                alloc: Global,
                _marker: PhantomData,
            }
        }
    }

    /// Converts an `Arc<T>` allocated using the global allocator into a raw
    /// pointer.
    ///
    /// The caller takes ownership of the reference count carried by the pointer
    /// and must eventually reconstruct an `Arc` from it (via
    /// [`from_raw`](Self::from_raw)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw(arc: Self) -> *const T {
        let me = ManuallyDrop::new(arc);
        let _alloc = unsafe { ptr::read(&me.alloc) };
        Arc::as_ptr(&me)
    }
}

// ---------------------------------------------------------------------------
// Arc: allocator-generic forms
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Like [`from_raw`](Self::from_raw), but parameterized over the choice of
    /// allocator.
    ///
    /// # Safety
    ///
    /// * Creating an `Arc<T, A>` from a pointer other than one returned from
    ///   [`Arc::<T, A>::into_raw`](Arc::into_raw) or
    ///   [`Arc::into_raw_with_allocator`](Arc::into_raw_with_allocator) is
    ///   undefined behavior.
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Arc<U, A>` was constructed
    ///   through `Arc<T, A>` and then converted to `Arc<U, A>` through an
    ///   [unsized coercion](https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions).
    /// * Note that if `U` or `U`'s data pointer is not `T` but has the same size
    ///   and alignment, this is basically like transmuting references of
    ///   different types. See [`core::mem::transmute`] for more information on
    ///   what restrictions apply in this case.
    /// * The raw pointer must point to a block of memory allocated by `alloc`.
    /// * The user of [`Arc::from_raw_in`] has to make sure a specific value of
    ///   `T` is only dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory
    /// unsafety, even if the returned [`Arc<T, A>`] is never accessed.
    #[inline]
    pub unsafe fn from_raw_in(p: *const T, alloc: A) -> Self {
        // SAFETY: caller guarantees validity.
        unsafe {
            let inner = data_get_ptr(p);
            Arc {
                ptr: NonNull::new_unchecked(inner as *mut ArcInner<T>),
                alloc,
                _marker: PhantomData,
            }
        }
    }

    /// Converts an [`Arc<T, A>`] into a raw pointer, retaining its allocator.
    ///
    /// The caller takes ownership of both the allocation and the allocator and
    /// must eventually reconstruct an `Arc` from them (via
    /// [`from_raw_in`](Self::from_raw_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw_with_allocator(arc: Self) -> (*const T, A) {
        let me = ManuallyDrop::new(arc);
        // SAFETY: `arc`'s fields are fully initialized; reading them out bit by
        // bit (rather than moving the whole `Arc`) avoids running its `Drop`,
        // which would decrement the counters we are handing off.
        let ptr = Arc::as_ptr(&me);
        let alloc = unsafe { ptr::read(&me.alloc) };
        (ptr, alloc)
    }
}

// ---------------------------------------------------------------------------
// Weak: global-allocator forms
// ---------------------------------------------------------------------------

impl<T: ?Sized> Weak<T, Global> {
    /// Constructs a new `Weak<T>` from a raw pointer previously produced by
    /// [`Weak::into_raw`].
    ///
    /// # Safety
    ///
    /// The pointer must have originated from the [`Self::into_raw`] and
    /// must still own its potential weak reference, and must point to a block of memory
    /// allocated by global allocator.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`Self::into_raw`].
    ///
    /// This function is unsafe because improper use may lead to memory
    /// unsafety, even if the returned [`Weak<T>`] is never accessed.
    #[inline]
    pub unsafe fn from_raw(p: *const T) -> Self {
        // SAFETY: caller guarantees `p` derives from `into_raw`; converting it
        // back restores the exact reference counts the original had.
        unsafe {
            // A dangling weak's raw pointer is the sentinel itself (see
            // `Weak::as_ptr`), not a projected payload address.
            let inner = if is_dangling_weak(p as *const ArcInner<T>) {
                p as *mut ArcInner<T>
            } else {
                data_get_ptr(p) as *mut ArcInner<T>
            };
            Weak {
                ptr: NonNull::new_unchecked(inner),
                alloc: Global,
                _marker: PhantomData,
            }
        }
    }

    /// Converts a `Weak<T>` allocated using the global allocator into a raw
    /// pointer.
    ///
    /// The caller takes ownership of the weak reference count carried by the
    /// pointer and must eventually reconstruct a `Weak` from it (via
    /// [`from_raw`](Self::from_raw)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw(weak: Self) -> *const T {
        let me = ManuallyDrop::new(weak);
        let _alloc = unsafe { ptr::read(&me.alloc) };
        Weak::as_ptr(&me)
    }
}

// ---------------------------------------------------------------------------
// Weak: allocator-generic forms
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Like [`from_raw`](Self::from_raw), but parameterized over the choice of
    /// allocator.
    ///
    /// # Safety
    ///
    /// The pointer must have originated from the [`Self::into_raw_with_allocator`] and
    /// must still own its potential weak reference, and must point to a block of memory
    /// allocated by `alloc`.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`Self::into_raw_with_allocator`].
    ///
    /// This function is unsafe because improper use may lead to memory
    /// unsafety, even if the returned [`Weak<T, A>`] is never accessed.
    #[inline]
    pub unsafe fn from_raw_in(p: *const T, alloc: A) -> Self {
        // SAFETY: caller guarantees validity.
        unsafe {
            // Mirror `from_raw`: a dangling weak's sentinel is not a payload
            // address, so it must be cast directly instead of offset-adjusted.
            let inner = if is_dangling_weak(p as *const ArcInner<T>) {
                p as *mut ArcInner<T>
            } else {
                data_get_ptr(p) as *mut ArcInner<T>
            };
            Weak {
                ptr: NonNull::new_unchecked(inner),
                alloc,
                _marker: PhantomData,
            }
        }
    }

    /// Converts a [`Weak<T, A>`] into a raw pointer and returns it with its allocator.
    ///
    /// The caller takes ownership of both the weak reference and the allocator
    /// and must eventually reconstruct a `Weak` from them (via
    /// [`from_raw_in`](Self::from_raw_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw_with_allocator(weak: Self) -> (*const T, A) {
        let me = ManuallyDrop::new(weak);
        // SAFETY: `weak`'s fields are fully initialized; reading them out bit by
        // bit avoids running its `Drop`, which would decrement the weak counter
        // we are handing off.
        let ptr = Weak::as_ptr(&me);
        let alloc = unsafe { ptr::read(&me.alloc) };
        (ptr, alloc)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::test_helpers::DropCounter;
    use olive_core::try_traits::TryClone;
    use std::sync::Arc as StdArc;

    // --- Sized round-trips ---------------------------------------------------

    #[test]
    fn arc_into_raw_from_raw_roundtrip() {
        let arc = Arc::try_new(99u32).unwrap();
        
        let raw = Arc::into_raw(arc);
        assert_eq!(unsafe { *raw }, 99);

        let arc = unsafe { Arc::from_raw(raw) };
        assert_eq!(*arc, 99);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn arc_into_raw_preserves_multiple_strong_refs() {
        // Splitting an Arc hands off its strong reference unchanged: the total
        // count stays put while ownership moves between handle and raw pointer.
        let arc = Arc::try_new(7i64).unwrap();
        let cloned = arc.try_clone().unwrap();
        assert_eq!(Arc::strong_count(&arc), 2);

        let raw = Arc::into_raw(cloned);
        // The raw pointer now carries one of the two references.
        assert_eq!(Arc::strong_count(&arc), 2);

        let restored = unsafe { Arc::from_raw(raw) };
        assert_eq!(Arc::strong_count(&restored), 2);
        assert!(Arc::ptr_eq(&arc, &restored));

        drop(restored);
        assert_eq!(Arc::strong_count(&arc), 1);
        drop(arc);
        // Last owner gone: allocation freed, no leak.
    }

    #[test]
    fn arc_into_raw_with_allocator_roundtrip() {
        let drops = StdArc::new(DropCounter::new());
        let alloc = crate::test_helpers::LocalCountingAlloc::new(drops.clone());
        let arc = Arc::try_new_in(5i32, alloc).unwrap();

        let (raw, alloc_out) = Arc::into_raw_with_allocator(arc);
        assert_eq!(unsafe { *raw }, 5);

        let arc = unsafe { Arc::from_raw_in(raw, alloc_out) };
        assert_eq!(*arc, 5);
        assert_eq!(Arc::strong_count(&arc), 1);

        drop(arc);
        // Dropping the reconstructed Arc destroys its embedded allocator handle
        // exactly once.
        assert_eq!(drops.get(), 1);
    }

    // --- Unsized round-trips ---------------------------------------------------

    #[test]
    fn arc_unsized_slice_into_raw_roundtrip() {
        let arr = [7u8, 8, 9];
        let arc: Arc<[u8]> = Arc::try_from_slice(&arr[..]).unwrap();

        let raw = Arc::into_raw(arc);
        let slice: &[u8] = unsafe { &*raw };
        assert_eq!(slice.len(), 3);
        assert_eq!(slice, [7, 8, 9]);

        let arc = unsafe { Arc::from_raw(raw) };
        assert_eq!(&*arc, [7, 8, 9]);
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    #[test]
    fn arc_unsized_str_into_raw_roundtrip() {
        let arc: Arc<str> = Arc::try_from_str("hello world").unwrap();

        let raw = Arc::into_raw(arc);
        let text: &str = unsafe { &*raw };
        assert_eq!(text, "hello world");

        let arc = unsafe { Arc::from_raw(raw) };
        assert_eq!(&*arc, "hello world");
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    // --- Weak round-trips ------------------------------------------------------

    #[test]
    fn weak_into_raw_from_raw_roundtrip() {
        let arc = Arc::try_new(11i32).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();

        let raw = Weak::into_raw(weak);
        // The payload is still readable while a strong reference lives.
        assert_eq!(unsafe { *raw }, 11);

        let weak = unsafe { Weak::from_raw(raw) };
        assert_eq!(weak.weak_count(), 1);
        assert_eq!(weak.strong_count(), 1);

        drop(arc);
        // With no strong refs left the payload is gone; upgrade reports None.
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_into_raw_keeps_allocation_alive_after_last_arc() {
        // A weak handed off as a raw pointer pins the allocation: dropping all
        // Arcs leaves the block alive until the reconstituted Weak is dropped.
        let arc = Arc::try_new(1u8).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();

        let raw = Weak::into_raw(weak);
        drop(arc);

        // Strong count is now zero, but the block must still exist (held by the
        // outstanding weak reference). Reading the counters through the weak is
        // valid; reading the payload is not.
        let weak = unsafe { Weak::from_raw(raw) };
        assert_eq!(weak.strong_count(), 0);
        assert_eq!(weak.weak_count(), 0);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_dangling_into_raw_roundtrip() {
        // A dangling weak carries no allocation; its raw pointer is the
        // misaligned sentinel, and reconstituting it yields another dangling
        // weak that behaves identically.
        let w: Weak<u32, Global> = Weak::new();

        let raw = Weak::into_raw(w);
        assert_ne!(raw.addr(), 0);

        let w = unsafe { Weak::from_raw(raw) };
        assert!(w.try_upgrade().unwrap().is_none());
        assert_eq!(w.strong_count(), 0);
        assert_eq!(w.weak_count(), 0);
    }

    #[test]
    fn weak_into_raw_with_allocator_roundtrip() {
        let drops = StdArc::new(DropCounter::new());
        let alloc = crate::test_helpers::LocalCountingAlloc::new(drops.clone());

        let arc = Arc::try_new_in(2i32, alloc).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();

        let (raw, alloc_out) = Weak::into_raw_with_allocator(weak);
        let weak = unsafe { Weak::from_raw_in(raw, alloc_out) };
        assert_eq!(weak.weak_count(), 1);
        drop(arc);
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
        // Three allocator handles are destroyed along the way: the one in `arc`
        // (dropped with it), the ephemeral one cloned by `try_upgrade` above
        // (dropped at end of statement), and the one handed back and moved into
        // the reconstituted `Weak` (dropped here).
        assert_eq!(drops.get(), 3);
    }
}
