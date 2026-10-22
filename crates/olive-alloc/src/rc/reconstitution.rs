use core::marker::PhantomData;
use core::mem::ManuallyDrop;

use super::helpers;
use super::internals::RcInner;
use super::{Rc, Weak};
use crate::alloc::{Allocator, Global};
use olive_core::ptr::{self, NonNull};

// ---------------------------------------------------------------------------
// Rc global reconstitution block
// ---------------------------------------------------------------------------

impl<T: ?Sized> Rc<T, Global> {
    /// Constructs a new `Rc<T>` from a raw pointer previously produced by
    /// [`Rc::into_raw`].
    ///
    /// # Safety
    ///
    /// * Creating a `Rc<T>` from a pointer other than one returned from
    ///   [`Rc::<T>::into_raw`](Rc::into_raw) or [`Rc::into_raw_with_allocator`](Rc::into_raw_with_allocator)
    ///   is undefined behavior.
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Rc<U>` was constructed
    ///   through `Rc<T>` and then converted to `Rc<U>` through an [unsized
    ///   coercion].
    /// * Note that if `U` or `U`'s data pointer is not `T` but has the same size
    ///   and alignment, this is basically like transmuting references of
    ///   different types. See [`mem::transmute`](core::mem::transmute) for more information
    ///   on what restrictions apply in this case.
    /// * The raw pointer must point to a block of memory allocated by the global allocator.
    /// * The user of [`Rc::from_raw`] has to make sure a specific value of `T` is only
    ///   dropped once.
    #[inline]
    pub unsafe fn from_raw(p: *const T) -> Self {
        // SAFETY: caller guarantees `p` derives from `into_raw`; converting it
        // back restores the exact reference counts the original had.
        unsafe {
            let inner = helpers::data_get_ptr(p);
            Rc {
                ptr: NonNull::new_unchecked(inner as *mut RcInner<T>),
                alloc: Global,
                _marker: PhantomData,
            }
        }
    }

    /// Converts an `Rc<T>` allocated using the global allocator into a raw pointer.
    ///
    /// The caller takes ownership of the reference count carried by the pointer
    /// and must eventually reconstruct an `Rc` from it (via
    /// [`from_raw`](Self::from_raw)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw(rc: Self) -> *const T {
        let me = ManuallyDrop::new(rc);
        let _alloc = unsafe { ptr::read(&me.alloc) };
        Rc::as_ptr(&me)
    }
}

// ---------------------------------------------------------------------------
// Rc generic reconstitution block (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Like [`from_raw`](Self::from_raw), but parameterized over the choice of
    /// allocator.
    ///
    /// # Safety
    ///
    /// * Creating a [`Rc<T, A>`] from a pointer other than one returned from
    ///   [`Rc<U, A>::into_raw`](Rc::into_raw) or
    ///   [`Rc<U, A>::into_raw_with_allocator`](Rc::into_raw_with_allocator)
    ///   is undefined behavior.
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Rc<U, A>` was constructed
    ///   through `Rc<T, A>` and then converted to `Rc<U, A>` through an [unsized
    ///   coercion].
    /// * Note that if `U` or `U`'s data pointer is not `T` but has the same size
    ///   and alignment, this is basically like transmuting references of
    ///   different types. See [`mem::transmute`](core::mem::transmute) for
    ///   more information on what restrictions apply in this case.
    /// * The raw pointer must point to a block of memory allocated by `alloc`
    /// * The user of `from_raw` has to make sure a specific value of `T` is only
    ///   dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned [`Rc<T, A>`] is never accessed.
    #[inline]
    pub unsafe fn from_raw_in(p: *const T, alloc: A) -> Self {
        // SAFETY: caller guarantees validity.
        unsafe {
            let inner = helpers::data_get_ptr(p);
            Rc {
                ptr: NonNull::new_unchecked(inner as *mut RcInner<T>),
                alloc,
                _marker: PhantomData,
            }
        }
    }

    /// Converts an [`Rc<T, A>`] into a raw pointer, retaining its allocator.
    ///
    /// The caller takes ownership of both the allocation and the allocator and
    /// must eventually reconstruct an `Rc` from them (via
    /// [`from_raw_in`](Self::from_raw_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw_with_allocator(rc: Self) -> (*const T, A) {
        let me = ManuallyDrop::new(rc);
        // SAFETY: `rc`'s inner field is fully initialized
        let ptr = Rc::as_ptr(&me);
        let alloc = unsafe { ptr::read(&me.alloc) };
        (ptr, alloc)
    }
}

// ---------------------------------------------------------------------------
// Weak global reconstitution block
// ---------------------------------------------------------------------------

impl<T: ?Sized> Weak<T, Global> {
    /// Creates a new `Weak` pointer from a raw pointer previously produced by
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
        // SAFETY: caller guarantees `p` derives from `Weak::into_raw`.
        unsafe {
            // A dangling weak's raw pointer is the sentinel itself (see
            // `Weak::as_ptr`), not a projected payload address.
            let inner = if helpers::is_dangling_weak(p as *const RcInner<T>) {
                p as *mut RcInner<T>
            } else {
                helpers::data_get_ptr(p) as *mut RcInner<T>
            };
            Weak {
                ptr: NonNull::new_unchecked(inner),
                alloc: Global,
                _marker: PhantomData,
            }
        }
    }

    /// Converts a `Weak<T>` into a raw pointer.
    ///
    /// The caller takes ownership of the weak reference carried by the pointer
    /// and must eventually reconstruct a `Weak` from it (via
    /// [`from_raw`](Self::from_raw)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw(self) -> *const T {
        let me = ManuallyDrop::new(self);
        let _alloc = unsafe { ptr::read(&me.alloc) };
        // SAFETY: the allocation and reference counts are valid
        Weak::as_ptr(&me)
    }
}

// ---------------------------------------------------------------------------
// Weak generic reconstitution block
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
            let inner = if helpers::is_dangling_weak(p as *const RcInner<T>) {
                p as *mut RcInner<T>
            } else {
                helpers::data_get_ptr(p) as *mut RcInner<T>
            };
            Weak {
                ptr: NonNull::new_unchecked(inner),
                alloc,
                _marker: PhantomData,
            }
        }
    }

    /// Converts a `Weak<T, A>` into a raw pointer, returning it along with
    /// its allocator.
    ///
    /// The caller takes ownership of both the allocation and the allocator and
    /// must eventually reconstruct a `Weak` from them (via
    /// [`from_raw_in`](Self::from_raw_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw_with_allocator(self) -> (*const T, A) {
        let me = ManuallyDrop::new(self);
        // SAFETY: the allocation and reference counts are valid
        let ptr = Weak::as_ptr(&me);
        let alloc = unsafe { ptr::read(&me.alloc) };
        (ptr, alloc)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_test_helpers::CountingAllocator;

    #[test]
    fn into_raw_roundtrip() {
        let rc = Rc::try_new(99).unwrap();
        let raw = Rc::into_raw(rc);
        assert_eq!(unsafe { *raw }, 99);
        let rc = unsafe { Rc::from_raw(raw) };
        assert_eq!(*rc, 99);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn unsized_into_raw_roundtrip_slice() {
        let arr = [7u8, 8, 9];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let raw = Rc::into_raw(rc);
        let slice: &[u8] = unsafe { &*raw };
        assert_eq!(slice.len(), 3);
        assert_eq!(slice, [7, 8, 9]);
        let rc = unsafe { Rc::from_raw(raw) };
        assert_eq!(&*rc, [7, 8, 9]);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn rc_into_raw_with_allocator_roundtrip_sized() {
        let alloc = CountingAllocator::new();
        let rc: Rc<u32, &CountingAllocator> = Rc::try_new_in(64, &alloc).unwrap();
        let (raw, ret_alloc) = Rc::into_raw_with_allocator(rc);
        assert_eq!(unsafe { *raw }, 64);
        assert_eq!(ret_alloc.allocations(), 1);
        let rc = unsafe { Rc::from_raw_in(raw, ret_alloc) };
        assert_eq!(*rc, 64);
        assert_eq!(Rc::strong_count(&rc), 1);
        drop(rc);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn rc_into_raw_with_allocator_roundtrip_unsized_slice() {
        let alloc = CountingAllocator::new();
        let arr = [1u8, 2, 3];
        let rc: Rc<[u8], &CountingAllocator> = Rc::try_from_slice_in(&arr[..], &alloc).unwrap();
        let (raw, ret_alloc) = Rc::into_raw_with_allocator(rc);
        let slice: &[u8] = unsafe { &*raw };
        assert_eq!(slice, [1u8, 2, 3]);
        let rc = unsafe { Rc::from_raw_in(raw, ret_alloc) };
        assert_eq!(&*rc, [1u8, 2, 3]);
        assert_eq!(Rc::strong_count(&rc), 1);
        drop(rc);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn weak_non_dangling_into_raw_roundtrip_sized() {
        let rc = Rc::try_new(11).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let raw = Weak::into_raw(weak);
        assert_eq!(unsafe { *raw }, 11);
        let weak = unsafe { Weak::from_raw(raw) };
        // The raw pointer carried the weak reference over: the count stays 1.
        assert_eq!(Rc::weak_count(&rc), 1);
        assert_eq!(*weak.try_upgrade().unwrap().unwrap(), 11);
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_non_dangling_into_raw_roundtrip_unsized_str() {
        let rc: Rc<str> = Rc::try_from_str("carry me").unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let raw = Weak::into_raw(weak);
        assert_eq!(unsafe { &*raw }, "carry me");
        let weak = unsafe { Weak::from_raw(raw) };
        assert_eq!(&*weak.try_upgrade().unwrap().unwrap(), "carry me");
        drop(rc);
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
        // `as_ptr` on a dangling weak returns the sentinel itself — exercising
        // the same branch that `from_raw` relies on to detect the sentinel.
        assert_eq!(w.as_ptr().addr(), raw.addr());
        assert!(w.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_dangling_unsized_into_raw_roundtrip() {
        let w = Weak::<str>::new();
        let raw = Weak::into_raw(w);
        let w = unsafe { Weak::from_raw(raw) };
        assert_eq!(w.as_ptr().addr(), raw.addr());
        assert!(w.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_into_raw_with_allocator_roundtrip_sized() {
        let alloc = CountingAllocator::new();
        let rc: Rc<u32, &CountingAllocator> = Rc::try_new_in(5, &alloc).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let (raw, ret_alloc) = weak.into_raw_with_allocator();
        assert_eq!(unsafe { *raw }, 5);
        let weak = unsafe { Weak::from_raw_in(raw, ret_alloc) };
        assert_eq!(Rc::weak_count(&rc), 1);
        assert_eq!(*weak.try_upgrade().unwrap().unwrap(), 5);
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn weak_into_raw_with_allocator_roundtrip_unsized_str() {
        let alloc = CountingAllocator::new();
        let rc: Rc<str, &CountingAllocator> = Rc::try_from_str_in("still here", &alloc).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let (raw, ret_alloc) = weak.into_raw_with_allocator();
        assert_eq!(unsafe { &*raw }, "still here");
        let weak = unsafe { Weak::from_raw_in(raw, ret_alloc) };
        assert_eq!(&*weak.try_upgrade().unwrap().unwrap(), "still here");
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn weak_dangling_into_raw_with_allocator_roundtrip() {
        let alloc = CountingAllocator::new();
        let weak: Weak<u32, &CountingAllocator> = Weak::new_in(&alloc);
        let (raw, ret_alloc) = weak.into_raw_with_allocator();
        let weak = unsafe { Weak::from_raw_in(raw, ret_alloc) };
        // The sentinel survives the round trip, and no allocator traffic
        // happens for a weak that never pointed at anything.
        assert_eq!(weak.as_ptr().addr(), raw.addr());
        assert!(weak.try_upgrade().unwrap().is_none());
        assert_eq!(alloc.allocations(), 0);
        assert_eq!(alloc.deallocations(), 0);
    }

    #[test]
    fn weak_dangling_into_raw_with_allocator_unsized_roundtrip() {
        let alloc = CountingAllocator::new();
        let weak: Weak<[u32], &CountingAllocator> = Weak::new_in(&alloc);
        let (raw, ret_alloc) = weak.into_raw_with_allocator();
        let weak = unsafe { Weak::from_raw_in(raw, ret_alloc) };
        // The sentinel survives the round trip, and no allocator traffic
        // happens for a weak that never pointed at anything.
        assert_eq!(weak.as_ptr().addr(), raw.addr());
        assert!(weak.try_upgrade().unwrap().is_none());
        assert_eq!(alloc.allocations(), 0);
        assert_eq!(alloc.deallocations(), 0);
    }
}
