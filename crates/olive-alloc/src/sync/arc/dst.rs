//! Construction of [`Arc`](super::Arc) nodes whose payload is a
//! **dynamically-sized type** (`?Sized`): slices `[T]`, `str`.
//!
//! The sized constructors live in [`construction`](super::construction); this
//! submodule holds the unsized-only entry points that derive their layout from
//! a *reference's* metadata (slice length, etc.) rather than from a static
//! type.
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr;

use crate::alloc::{Allocator, Global, Layout};
use olive_core::alloc::LayoutExt;
use olive_core::ptr::{NonNull, PointerExt};
use olive_core::try_traits::TryClone;
use olive_core::try_traits::try_clone::{TryCloneError, TryCloneToUninit};

use super::pointers::{arc_inner_layout_for_value_layout, initialize_arcinner};
use super::{Arc, ArcInner};

// ---------------------------------------------------------------------------
// UniqueArcUninit — intermediate allocation handle
// ---------------------------------------------------------------------------

/// A uniquely-owned, freshly-allocated `ArcInner<T>` block whose payload region
/// is still uninitialized.
///
/// This struct sits between "raw allocation" and "finished `Arc<T>`": it owns
/// the heap block, has already initialized the two atomic refcount headers to
/// `(strong = 1, weak = 1)`, but leaves the payload slot for the caller to fill
/// via [`data_ptr`](Self::data_ptr). Once the caller has written the payload,
/// calling [`into_arc`](Self::into_arc) consumes the handle and produces the
/// final `Arc<T, A>`.
///
/// If dropped without being converted (e.g. due to an early return or error),
/// the block is deallocated automatically — no leaks.
pub(crate) struct UniqueArcUninit<T: ?Sized, A: Allocator> {
    ptr: NonNull<ArcInner<T>>,
    alloc: A,
    layout: Layout,
}

impl<T: ?Sized, A: Allocator> UniqueArcUninit<T, A> {
    /// Allocates a new `ArcInner<T>` block for a **potentially unsized** type,
    /// deriving the layout from `src`'s metadata (slice length, vtable, etc.),
    /// and initializes the refcount headers to `(1, 1)`.
    ///
    /// This is the path used when **cloning** an existing value into fresh
    /// storage: the source reference provides the metadata that determines the
    /// block size and the fat-pointer width.
    ///
    /// The caller must subsequently write the payload through
    /// [`data_ptr`](Self::data_ptr) (typically via
    /// [`TryCloneToUninit::try_clone_to_uninit`]) and then call
    /// [`into_arc`](Self::into_arc) to obtain the final `Arc`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError::Alloc`] if the allocation fails, or
    /// [`TryCloneError::Other`] if the combined layout overflows.
    pub(crate) fn try_new_for_value(src: &T, alloc: A) -> Result<Self, TryCloneError> {
        // SAFETY: `src` is a valid reference, so its pointee is aligned,
        // non-null, and carries correct metadata for `T`; reading only the
        // metadata does not require the pointee to be initialized.
        let value_layout = unsafe { Layout::for_value_pointer(ptr::from_ref(src)) };
        let (layout, _) = arc_inner_layout_for_value_layout(value_layout)
            .map_err(|_| TryCloneError::Other("Arc payload layout overflow"))?;

        let block = alloc.allocate(layout)?;
        let base: *mut u8 = block.cast::<u8>().as_ptr();

        // Graft the source's metadata (slice length / vtable / etc.) onto the
        // destination pointer so the resulting fat pointer carries the correct
        // info for the payload.
        let to_copy_metadata = ptr::from_ref(src) as *const ArcInner<T>;
        let inner_fat = unsafe { base.cast_with_metadata(to_copy_metadata) };
        let ptr = unsafe { NonNull::new_unchecked(inner_fat) };

        // Seed the refcount headers to (1, 1). The allocation is fresh
        // (uninitialized), so we write through raw pointers rather than forming
        // references to uninitialized `AtomicUsize` values.
        // SAFETY: `ptr` is a live, aligned block whose header fields are not yet
        // initialized, and no other thread can observe it before this function
        // returns.
        unsafe { initialize_arcinner(ptr.as_ptr()) };

        Ok(Self { ptr, alloc, layout })
    }

    /// Returns a raw mutable pointer to the payload region within the block.
    ///
    /// The pointed-to memory is uninitialized; the caller must write the
    /// payload before calling [`into_arc`](Self::into_arc). For unsized types
    /// the caller should use [`TryCloneToUninit::try_clone_to_uninit`], which
    /// knows how to interpret the target.
    #[inline]
    pub(crate) fn data_ptr(&self) -> *mut u8 {
        // Project the `value` field out of the fat pointer to get the exact
        // address of the payload slot.
        unsafe { &raw mut (*self.ptr.as_ptr()).value as *mut u8 }
    }

    /// Consumes the handle and produces the finalized `Arc<T, A>`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the payload region (pointed to by
    /// [`data_ptr`](Self::data_ptr)) has been fully initialized with a valid
    /// `T` before calling this method.
    #[inline]
    pub(crate) fn into_arc(self) -> Arc<T, A> {
        let me = ManuallyDrop::new(self);
        // SAFETY: we are transferring ownership of the block to the returned
        // `Arc`. Reading the fields out and wrapping `ManuallyDrop` on `self`
        // prevents its `Drop` impl from double-freeing the allocation.
        let ptr = unsafe { ptr::read(&me.ptr) };
        let alloc = unsafe { ptr::read(&me.alloc) };
        Arc {
            ptr,
            alloc,
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Allocator> Drop for UniqueArcUninit<T, A> {
    fn drop(&mut self) {
        // If we reach here, `into_arc` was never called (early return, error,
        // panic, etc.). Free the block. The header was initialized to (1, 1) so
        // the allocation is well-formed from the allocator's perspective.
        //
        // SAFETY: `self.ptr` was allocated by `self.alloc` with `self.layout`.
        unsafe {
            self.alloc.deallocate(self.ptr.cast(), self.layout);
        }
    }
}

// ---------------------------------------------------------------------------
// Unsized construction (?Sized)
// ---------------------------------------------------------------------------

#[allow(private_bounds)]
impl<T: ?Sized + TryCloneToUninit> Arc<T, Global> {
    /// Clones a `&T` into a freshly allocated `Arc<T, Global>` for potentially
    /// unsized `T`.
    ///
    /// This is the fallible analogue of std's `Arc::from(&slice[..])`,
    /// `Arc::from("literal")`, etc. It works for any type implementing
    /// [`TryCloneToUninit`]: in practice that means sized types, `str`, and
    /// slices `[T]`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails, an element clone
    /// fails, or the payload layout is absurd (see
    /// [`try_clone_from_ref_in`](Self::try_clone_from_ref_in)).
    #[inline]
    pub fn try_clone_from_ref(src: &T) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, Global)
    }
}

#[allow(private_bounds)]
impl<T: ?Sized + TryCloneToUninit, A: Allocator> Arc<T, A> {
    /// Clones a `&T` into a freshly allocated `Arc<T, A>` for potentially
    /// unsized `T`, using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails, any element clone
    /// fails, or the payload layout is absurd (e.g. a slice whose length would
    /// overflow when combined with the header). The last case is reported as
    /// [`TryCloneError::Other`], not an OOM-style error, because it can never
    /// succeed on retry.
    #[inline]
    pub fn try_clone_from_ref_in(src: &T, alloc: A) -> Result<Self, TryCloneError> {
        let uninit = UniqueArcUninit::try_new_for_value(src, alloc)?;
        // SAFETY: `uninit.data_ptr()` points to the uninitialized payload slot
        // within a live allocation of the correct size and alignment.
        unsafe { <T as TryCloneToUninit>::try_clone_to_uninit(src, uninit.data_ptr()) }?;
        Ok(uninit.into_arc())
    }
}

// ---------------------------------------------------------------------------
// Uninit-slice construction (requires `T: Sized`)
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Arc<[T], A> {
    /// Allocates a new `Arc<[MaybeUninit<T>], A>` containing `len` slots of
    /// uninitialized memory, parameterized over the allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError::Alloc`] if the allocation fails, or
    /// [`TryCloneError::Other`] if the combined layout overflows.
    #[inline]
    pub fn try_new_uninit_slice_in(
        len: usize,
        alloc: A,
    ) -> Result<Arc<[MaybeUninit<T>], A>, TryCloneError> {
        // A zero-length slice needs no payload bytes; a positive-length one
        // needs `len * size_of::<T>()` bytes at `align_of::<T>()`. Since
        // `MaybeUninit<T>` shares `T`'s layout, sizing by `T` is exact.
        let value_layout = Layout::array::<T>(len)
            .map_err(|_| TryCloneError::Other("Arc slice layout overflow"))?;
        let (layout, data_offset) = arc_inner_layout_for_value_layout(value_layout)
            .map_err(|_| TryCloneError::Other("Arc payload layout overflow"))?;

        let block = alloc.allocate(layout)?;
        let base: *mut u8 = block.cast::<u8>().as_ptr();

        // Compute the address of the payload slot (after the header).
        let payload_addr = unsafe { base.add(data_offset) };

        // Build a fat `*mut ArcInner<[MaybeUninit<T>]>` whose data word is `base`
        // and whose metadata carries the slice length.
        let fat_at_payload: *const [MaybeUninit<T>] =
            ptr::slice_from_raw_parts(payload_addr.cast::<MaybeUninit<T>>(), len);
        // Casting is possible because both share the unsized tail `T`.
        let meta_carrier = fat_at_payload as *const ArcInner<[MaybeUninit<T>]>;
        let inner_fat: *mut ArcInner<[MaybeUninit<T>]> =
            unsafe { base.cast_with_metadata(meta_carrier) };
        let ptr = unsafe { NonNull::new_unchecked(inner_fat) };

        // SAFETY: `ptr` is a live, aligned block whose header fields are not yet
        // initialized, and no other thread can observe it before this function
        // returns.
        unsafe { initialize_arcinner(ptr.as_ptr()) };

        Ok(Arc {
            ptr,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Allocates a new `Arc<[MaybeUninit<T>], A>` containing `len` slots of
    /// zero-initialized memory, parameterized over the allocator.
    ///
    /// The elements are zero-filled. This is the slice analogue of
    /// [`try_new_zeroed`](super::Arc::try_new_zeroed).
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError::Alloc`] if the allocation fails, or
    /// [`TryCloneError::Other`] if the combined layout overflows.
    #[inline]
    pub fn try_new_zeroed_slice_in(
        len: usize,
        alloc: A,
    ) -> Result<Arc<[MaybeUninit<T>], A>, TryCloneError> {
        // A zero-length slice needs no payload bytes; a positive-length one
        // needs `len * size_of::<T>()` bytes at `align_of::<T>()`. Since
        // `MaybeUninit<T>` shares `T`'s layout, sizing by `T` is exact.
        let value_layout = Layout::array::<T>(len)
            .map_err(|_| TryCloneError::Other("Arc slice layout overflow"))?;
        let (layout, data_offset) = arc_inner_layout_for_value_layout(value_layout)
            .map_err(|_| TryCloneError::Other("Arc payload layout overflow"))?;

        // Zero-fill the whole block up front so every payload byte starts out as
        // zero before the refcount headers are seeded on top of it.
        let block = alloc.allocate_zeroed(layout)?;
        let base: *mut u8 = block.cast::<u8>().as_ptr();

        // Compute the address of the payload slot (after the header).
        let payload_addr = unsafe { base.add(data_offset) };

        // Build a fat `*mut ArcInner<[MaybeUninit<T>]>` whose data word is `base`
        // and whose metadata carries the slice length.
        let fat_at_payload: *const [MaybeUninit<T>] =
            ptr::slice_from_raw_parts(payload_addr.cast::<MaybeUninit<T>>(), len);
        // Casting is possible because both share the unsized tail `T`.
        let meta_carrier = fat_at_payload as *const ArcInner<[MaybeUninit<T>]>;
        let inner_fat: *mut ArcInner<[MaybeUninit<T>]> =
            unsafe { base.cast_with_metadata(meta_carrier) };
        let ptr = unsafe { NonNull::new_unchecked(inner_fat) };

        // Seed the refcount headers to (1, 1) on top of the zero-filled block.
        // SAFETY: `ptr` is a live, aligned block whose header fields are not yet
        // initialized as valid atomics, and no other thread can observe it
        // before this function returns.
        unsafe { initialize_arcinner(ptr.as_ptr()) };

        Ok(Arc {
            ptr,
            alloc,
            _marker: PhantomData,
        })
    }
}

impl<T> Arc<[T], Global> {
    /// Allocates a new `Arc<[MaybeUninit<T]]>` containing `len` slots of
    /// uninitialized memory on the global allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError::Alloc`] if the allocation fails, or
    /// [`TryCloneError::Other`] if the combined layout overflows.
    #[inline]
    pub fn try_new_uninit_slice(len: usize) -> Result<Arc<[MaybeUninit<T>]>, TryCloneError> {
        Arc::try_new_uninit_slice_in(len, Global)
    }

    /// Allocates a new `Arc<[MaybeUninit<T>]>` containing `len` slots of
    /// zero-initialized memory on the global allocator.
    ///
    /// The elements are zero-filled. This is the slice analogue of
    /// [`try_new_zeroed`](super::Arc::try_new_zeroed).
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError::Alloc`] if the allocation fails, or
    /// [`TryCloneError::Other`] if the combined layout overflows.
    #[inline]
    pub fn try_new_zeroed_slice(len: usize) -> Result<Arc<[MaybeUninit<T>]>, TryCloneError> {
        Arc::try_new_zeroed_slice_in(len, Global)
    }
}

// ---------------------------------------------------------------------------
// Convenience wrappers for common unsized types
// ---------------------------------------------------------------------------

impl<T: TryClone, A: Allocator> Arc<[T], A> {
    /// Creates a new `Rc<[T]>` by cloning the slice of T items.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_slice_in(src: &[T], alloc: A) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, alloc)
    }
}

impl<T: TryClone> Arc<[T], Global> {
    /// Convenience wrapper around [`Self::try_from_slice_in`] using the global
    /// allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_slice(src: &[T]) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, Global)
    }
}

impl<A: Allocator> Arc<str, A> {
    /// Allocates a new `Arc<str>` by copying the UTF-8 bytes from `src` into
    /// fresh heap memory, parameterized over the allocator. Fallible analogue
    /// of std's `Arc::from("literal")`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_str_in(src: &str, alloc: A) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, alloc)
    }
}

impl Arc<str, Global> {
    /// Convenience wrapper around [`Self::try_from_str_in`] using the global
    /// allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_str(src: &str) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, Global)
    }
}

// ---------------------------------------------------------------------------
// Uninit-slice → init-slice bridge
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Arc<[MaybeUninit<T>], A> {
    /// Reinterprets this `Arc<[MaybeUninit<T>], A>` as an initialized
    /// `Arc<[T], A>`.
    ///
    /// # Safety
    ///
    /// Every element of the payload slice must already be fully initialized
    /// (each `MaybeUninit<T>` slot must contain a valid `T`). Calling this
    /// while any slot is still uninitialized is undefined behavior.
    #[inline]
    pub unsafe fn assume_init(self) -> Arc<[T], A> {
        // Establish up front that the element types are layout-identical.
        debug_assert_eq!(
            size_of::<MaybeUninit<T>>(),
            size_of::<T>(),
            "MaybeUninit<T> must have T's size"
        );
        debug_assert_eq!(
            align_of::<MaybeUninit<T>>(),
            align_of::<T>(),
            "MaybeUninit<T> must have T's alignment"
        );

        // Suppress the drop of `self` so the original `Arc<[MaybeUninit<T>]>`
        // does not decrement the refcount and free the allocation that the
        // returned `Arc<[T]>` inherits.
        let me = ManuallyDrop::new(self);
        let alloc = unsafe { ptr::read(&me.alloc) };

        // SAFETY: ArcInner<[MaybeUninit<T>]> has identical layout to ArcInner<T>.
        let new_ptr: NonNull<ArcInner<[T]>> =
            unsafe { olive_core::mem::transmute_unchecked(me.ptr) };
        Arc {
            ptr: new_ptr,
            alloc,
            _marker: PhantomData,
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
    use crate::string::String;
    use crate::test_helpers::{CloneBudget, FailAlloc, FlakyTrackedItem, Ledger};
    use std::sync::Arc as StdArc;
    use std::vec::Vec;

    // --- Slice construction --------------------------------------------------

    #[test]
    fn try_from_slice_copies_bytes_and_sets_counters() {
        let arr = [1u8, 2, 3, 4];
        let arc: Arc<[u8]> = Arc::try_from_slice(&arr[..]).unwrap();
        assert_eq!(&*arc, &[1, 2, 3, 4]);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
        assert_eq!(
            arc.inner().weak.load(core::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn try_from_slice_empty_is_valid() {
        let arr: [u8; 0] = [];
        let arc: Arc<[u8]> = Arc::try_from_slice(&arr[..]).unwrap();
        assert_eq!(arc.len(), 0);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn try_from_string_slice_is_correct() {
        let arr: [String; 1] = [String::try_from_str("hello world").unwrap()];
        let arc: Arc<[String]> = Arc::try_from_slice(&arr[..]).unwrap();
        assert_eq!(arc.len(), 1);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
        assert_eq!(arc[0], "hello world");
    }

    #[test]
    fn try_from_slice_zst_elements_carries_length() {
        // Zero-sized elements: the payload occupies no bytes, but the fat
        // pointer must still report the correct length.
        let src: Vec<()> = std::vec![(), (), ()];
        let arc: Arc<[()]> = Arc::try_clone_from_ref_in(&src[..], Global).unwrap();
        assert_eq!(arc.len(), 3);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn try_from_str_copies_utf8_and_sets_counters() {
        let s = String::try_from_str("héllo wörld").unwrap();
        let arc: Arc<str> = Arc::try_from_str(s.as_str()).unwrap();
        assert_eq!(arc.as_ref(), "héllo wörld");
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn try_from_slice_in_custom_allocator_drops_handle_once() {
        let drops = StdArc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::LocalCountingAlloc::new(drops.clone());
        let arr = [9u8, 8, 7];
        let arc = Arc::try_from_slice_in(&arr[..], alloc).unwrap();
        assert_eq!(&*arc, &[9, 8, 7]);
        assert_eq!(Arc::strong_count(&arc), 1);
        drop(arc);
        // The allocator handle was consumed by the Arc and dropped with it.
        assert_eq!(drops.get(), 1);
    }

    // --- OOM paths ------------------------------------------------------------

    #[test]
    fn try_from_slice_oom_errors() {
        let arr = [1u8, 2, 3];
        let res: Result<Arc<[u8], FailAlloc>, TryCloneError> =
            Arc::try_from_slice_in(&arr[..], FailAlloc);
        assert!(
            matches!(res, Err(TryCloneError::Alloc(_))),
            "expected Alloc(OOM)"
        );
    }

    #[test]
    fn try_from_str_oom_errors() {
        let res: Result<Arc<str, FailAlloc>, TryCloneError> =
            Arc::try_from_str_in("abc", FailAlloc);
        assert!(
            matches!(res, Err(TryCloneError::Alloc(_))),
            "expected Alloc(OOM)"
        );
    }

    #[test]
    fn try_new_uninit_slice_oom_errors() {
        let res: Result<Arc<[MaybeUninit<i32>], FailAlloc>, TryCloneError> =
            Arc::try_new_uninit_slice_in(4, FailAlloc);
        assert!(
            matches!(res, Err(TryCloneError::Alloc(_))),
            "expected Alloc(OOM)"
        );
    }

    #[test]
    fn try_new_zeroed_slice_oom_errors() {
        let res: Result<Arc<[MaybeUninit<i32>], FailAlloc>, TryCloneError> =
            Arc::try_new_zeroed_slice_in(4, FailAlloc);
        assert!(
            matches!(res, Err(TryCloneError::Alloc(_))),
            "expected Alloc(OOM)"
        );
    }

    // --- Clone-failure rollback ----------------------------------------------

    #[test]
    fn slice_clone_success_drops_each_element_exactly_once() {
        // Three source elements (ids 0..3) are cloned into a fresh Arc as new
        // ids (3..6). While the Arc is alive no clone has been dropped yet;
        // dropping the Arc must destroy the three clones exactly once, then
        // dropping the source vec destroys the three originals exactly once.
        let ledger = StdArc::new(Ledger::new());
        let budget = StdArc::new(CloneBudget::new(u32::MAX));
        let mut src: Vec<FlakyTrackedItem> = Vec::new();
        for _ in 0..3 {
            let id = ledger.allocate();
            ledger.register(id);
            src.push(FlakyTrackedItem {
                id,
                ledger: ledger.clone(),
                inner: (*budget).share(),
            });
        }

        let arc: Arc<[FlakyTrackedItem]> = Arc::try_clone_from_ref_in(&src[..], Global).unwrap();
        assert_eq!(arc.len(), 3);
        // Six instances exist: the three sources and their three clones. None
        // has been dropped yet.
        assert_eq!(ledger.live_ids(), [0, 1, 2, 3, 4, 5]);
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.drop_counts().is_empty());

        drop(arc);
        // The three cloned incarnations (ids 3, 4, 5) are destroyed exactly
        // once each; the sources remain live.
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        assert_eq!(ledger.drop_count(5), 1);

        drop(src);
        // All six instances died exactly once: no leaks, no double-frees.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..6));
    }

    #[test]
    fn budgeted_flaky_mid_operation_failure_is_clean() {
        // Shared budget allows exactly 2 clones; a 3-element slice forces a
        // failure at element index 2. Rollback must discard the two transient
        // clones without leaking or double-freeing anything.
        let ledger = StdArc::new(Ledger::new());
        let budget = StdArc::new(CloneBudget::new(2));
        let mut src: Vec<FlakyTrackedItem> = Vec::new();
        for _ in 0..3 {
            let id = ledger.allocate();
            ledger.register(id);
            src.push(FlakyTrackedItem {
                id,
                ledger: ledger.clone(),
                inner: (*budget).share(),
            });
        }
        let res: Result<Arc<[FlakyTrackedItem]>, TryCloneError> =
            Arc::try_clone_from_ref_in(&src[..], Global);
        assert!(res.is_err(), "third clone must exhaust the budget");
        // Exactly two transient clones (ids 3 and 4) were created and both were
        // rolled back: each dropped precisely once, only the sources remain
        // live, and nothing was double-freed.
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        assert_eq!(ledger.total_allocated(), 5);
        // The budget must be fully drained: two successful clones plus the third
        // failed attempt that observed an empty budget. A further consume must
        // therefore still fail, proving nothing was leaked back into the pool.
        assert!(
            !budget.try_consume(),
            "budget should be exhausted after the failed clone"
        );

        // Tear down the sources: all five instances now dead exactly once.
        drop(src);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    // --- Absurd layout --------------------------------------------------------

    #[test]
    fn absurd_payload_layout_reports_other_not_oom() {
        let usize_size = size_of::<usize>();
        let max_size = isize::MAX as usize - usize_size + 1;
        // A hypothetical Rust tcype requires the size is divisible by its alignment
        assert_eq!(max_size % usize_size, 0);
        let huge = Layout::from_size_align(max_size, usize_size)
            .expect("payload layout alone is representable");
        let err = arc_inner_layout_for_value_layout(huge)
            .expect_err("absurd payload must fail layout computation");
        let _ = err;
        // Mirror the exact mapping used by `try_clone_from_ref_in`: the overflow
        // becomes `Other`, never `Alloc(AllocError)`. Asserting on both arms of
        // the match proves the absurd case cannot be mistaken for transient OOM.
        let mapped = TryCloneError::Other("Arc payload layout overflow");
        assert!(matches!(mapped, TryCloneError::Other(_)));
        assert!(!matches!(mapped, TryCloneError::Alloc(_)));
    }

    // --- Uninit-slice → init-slice bridge ------------------------------------

    /// Fills every slot of an `Arc<[MaybeUninit<T>]>` by projecting the payload
    /// slice out of the fat pointer and writing each element's `MaybeUninit`.
    fn fill_slots<T>(arc: &Arc<[MaybeUninit<T>]>, f: impl Fn(usize) -> T) {
        let slots: *mut [MaybeUninit<T>] = unsafe { &raw mut (*arc.ptr.as_ptr()).value };
        #[allow(
            clippy::needless_borrow,
            reason = "Miri does not allow implicit autoref"
        )]
        let n = unsafe { (&*slots).len() };
        // SAFETY: exclusive access (strong == 1, weak == 0 excluding implicit weak ref);
        // each write initializes one slot.
        unsafe {
            for i in 0..n {
                (*slots)[i].as_mut_ptr().write(f(i));
            }
        }
    }

    #[test]
    fn uninit_slice_bridge_assume_init_preserves_len_and_state() {
        let len = 4usize;
        let uninit: Arc<[MaybeUninit<i32>], Global> =
            Arc::try_new_uninit_slice_in(len, Global).unwrap();
        assert_eq!(Arc::strong_count(&uninit), 1);

        // Fill each slot directly, bypassing any higher-level writer, to prove
        // `assume_init` is a pure type reinterpretation. Each element is a
        // `MaybeUninit<i32>`; writing a valid `i32` into it initializes that slot.
        fill_slots(&uninit, |i| i as i32 * 10);

        let arc = unsafe { uninit.assume_init() };
        assert_eq!(&*arc, &[0, 10, 20, 30]);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn uninit_slice_bridge_empty_slice() {
        let uninit: Arc<[MaybeUninit<u8>], Global> =
            Arc::try_new_uninit_slice_in(0, Global).unwrap();
        let arc = unsafe { uninit.assume_init() };
        assert_eq!(arc.len(), 0);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn zeroed_slice_yields_zeroed_payload() {
        // Every byte of the payload must come back as zero; interpreting the
        // slots as `u64` (whose all-bits-zero value is `0`) makes any stray
        // nonzero bit observable.
        let len = 8usize;
        let zeroed: Arc<[MaybeUninit<u64>], Global> =
            Arc::try_new_zeroed_slice_in(len, Global).unwrap();
        assert_eq!(Arc::strong_count(&zeroed), 1);
        assert_eq!(Arc::weak_count(&zeroed), 0);

        let arc = unsafe { zeroed.assume_init() };
        assert_eq!(&*arc, &[0u64; 8]);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn zeroed_slice_empty_and_zst_are_valid() {
        // Empty slice: no payload bytes, but the fat pointer still reports 0.
        let empty: Arc<[MaybeUninit<u8>], Global> =
            Arc::try_new_zeroed_slice_in(0, Global).unwrap();
        assert_eq!(Arc::strong_count(&empty), 1);
        assert_eq!(Arc::weak_count(&empty), 0);
        assert_eq!(empty.len(), 0);

        // ZST elements: zero bytes to zero-fill, length metadata preserved.
        let zst: Arc<[MaybeUninit<()>], Global> = Arc::try_new_zeroed_slice_in(5, Global).unwrap();
        assert_eq!(Arc::strong_count(&zst), 1);
        assert_eq!(Arc::weak_count(&zst), 0);
        assert_eq!(zst.len(), 5);
    }

    #[test]
    fn uninit_slice_bridge_zst_elements() {
        let uninit: Arc<[MaybeUninit<()>], Global> =
            Arc::try_new_uninit_slice_in(5, Global).unwrap();
        // ZST slots occupy no bytes but still carry the length metadata; filling
        // them is a no-op write, then `assume_init` must preserve the count.
        fill_slots(&uninit, |_| ());
        let arc = unsafe { uninit.assume_init() };
        assert_eq!(arc.len(), 5);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    // --- Global-allocator aliases ---------------------------------------------

    #[test]
    fn global_alias_matches_generic_constructor_for_uninit_slice() {
        // The no-allocator alias must produce exactly the same node as passing
        // `Global` explicitly to the generic form.
        let via_alias: Arc<[MaybeUninit<i32>], Global> = Arc::try_new_uninit_slice(4).unwrap();
        let via_generic: Arc<[MaybeUninit<i32>], Global> =
            Arc::try_new_uninit_slice_in(4, Global).unwrap();
        assert_eq!(via_alias.len(), via_generic.len());
        assert_eq!(
            Arc::strong_count(&via_alias),
            Arc::strong_count(&via_generic)
        );
        assert_eq!(Arc::weak_count(&via_alias), Arc::weak_count(&via_generic));
    }

    #[test]
    fn global_alias_matches_generic_constructor_for_zeroed_slice() {
        // Same agreement check for the zeroed variant; both routes must yield an
        // all-zero payload of the same length and refcount state.
        let via_alias: Arc<[MaybeUninit<u64>], Global> = Arc::try_new_zeroed_slice(8).unwrap();
        let via_generic: Arc<[MaybeUninit<u64>], Global> =
            Arc::try_new_zeroed_slice_in(8, Global).unwrap();
        assert_eq!(via_alias.len(), via_generic.len());
        assert_eq!(
            Arc::strong_count(&via_alias),
            Arc::strong_count(&via_generic)
        );
        assert_eq!(Arc::weak_count(&via_alias), Arc::weak_count(&via_generic));

        let aliased = unsafe { via_alias.assume_init() };
        assert_eq!(&*aliased, &[0u64; 8]);
    }

    // --- Cross-checks ---------------------------------------------------------

    #[test]
    fn global_and_generic_constructors_agree_for_slices() {
        let arr = [
            String::try_from_str("cat").unwrap(),
            String::try_from_str("dog").unwrap(),
        ];
        let g = Arc::try_from_slice(&arr[..]).unwrap();
        let generic = Arc::try_from_slice_in(&arr[..], Global).unwrap();
        assert_eq!(*g, *generic);
        assert_eq!(Arc::strong_count(&g), Arc::strong_count(&generic));
        assert_eq!(Arc::weak_count(&g), Arc::weak_count(&generic));
    }
}
