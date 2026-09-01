//! A low-level, fallible port of the standard library's `RawVec`.
//!
//! This is the memory-management substrate that contiguous heap-backed collections in
//! this crate builds on top of: it owns a single allocation described by a
//! [`Layout`], tracks its capacity, and grows/shrinks it through the
//! [`Allocator`] trait (defaulting to [`Global`]).
//!
//! Compared with the std original, three things differ:
//!
//! * Every operation that could fail — reserving, growing, shrinking, or
//!   allocating an initial buffer — returns a [`Result`] carrying
//!   [`TryReserveError`] instead of panicking or aborting.
//! * All allocation goes through the [`Allocator`] trait rather than free
//!   functions, giving a single swappable seam for custom allocators and OOM
//!   simulation in tests.
//! * Compiler-internal attributes (`rustc_layout_scalar_valid_range_end`) and
//!   unstable helpers (`SizedTypeProperties`, `Layout::repeat`) are replaced
//!   with stable, equivalent code.

use core::cmp;
use core::hint;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::mem::MaybeUninit;
use core::mem::{align_of, size_of, transmute};
use core::ptr::{self, NonNull};

use crate::alloc::{Allocator, Global, Layout};
use crate::boxed::Box;
use olive_core::alloc::LayoutExt;
use olive_core::alloc_errors::{TryReserveError, TryReserveErrorKind};

// Convenience alias mirroring std's `use TryReserveErrorKind::*;`.
use TryReserveErrorKind::CapacityOverflow;

/// Whether a fresh allocation should be left uninitialized or zero-filled.
enum AllocInit {
    /// The contents of the new memory are uninitialized.
    Uninitialized,
    /// The new memory is guaranteed to be zeroed.
    Zeroed,
}

/// Wraps the stored capacity so that we never store a meaningless huge value for
/// ZSTs. For a ZST the effective capacity is reported as `usize::MAX`, but "0" is stored.
#[repr(transparent)]
struct Cap(usize);

impl Cap {
    const ZERO: Cap = Cap(0);

    /// # Safety
    ///
    /// `cap` must be `<= isize::MAX`. Only valid for non-ZST element types; use
    /// [`Self::new_zst_aware`] at sites where the element type is known.
    const unsafe fn new_unchecked(cap: usize) -> Self {
        debug_assert!(cap <= isize::MAX as usize);
        // SAFETY: caller guarantees `cap <= isize::MAX`.
        Cap(cap)
    }

    /// Stores `cap`, or [`Cap::ZERO`] when `T` is a zero-sized type.
    ///
    /// A ZST's effective capacity is reported as `usize::MAX` by
    /// [`RawVec::capacity`], but no real allocation backs it — storing that
    /// sentinel would trip the size-limit invariant in [`Self::new_unchecked`]
    /// (and is meaningless). Collapsing it to `Cap::ZERO` keeps the stored
    /// value honest while leaving the reported capacity untouched, since
    /// `capacity()` special-cases ZSTs.
    ///
    /// # Safety
    ///
    /// When `T` is not a ZST, `cap` must be `<= isize::MAX`.
    const unsafe fn new_zst_aware<T>(cap: usize) -> Self {
        if core::mem::size_of::<T>() == 0 {
            Cap::ZERO
        } else {
            // SAFETY: non-ZST branch defers to the same invariant.
            unsafe { Self::new_unchecked(cap) }
        }
    }
}

/// A low-level utility for more ergonomically allocating, reallocating, and
/// deallocating a buffer of memory on the heap without having to worry about all
/// the corner cases involved. This type is excellent for building your own data
/// structures like `Vec` and `VecDeque`. In particular:
///
/// * Produces a well-aligned dangling `NonNull` on zero-sized types.
/// * Produces a well-aligned dangling `NonNull` on zero-length allocations.
/// * Avoids freeing a dangling `NonNull`.
/// * Catches all overflows in capacity computations (promotes them to
///   `CapacityOverflow` errors).
/// * Guards against 32-bit systems allocating more than `isize::MAX` bytes.
/// * Guards against overflowing your length.
/// * Returns a [`TryReserveError`] for failed allocations instead of aborting.
/// * Contains a `NonNull` (`Unique` is unusable).
/// * Uses the excess returned from the allocator to use the largest available
///   capacity.
///
/// This type does not in any way inspect the memory that it manages. When
/// dropped it *will* free its memory, but it *won't* try to drop its contents.
/// It is up to the user of `RawVec` to handle the actual things *stored* inside
/// of a `RawVec`.
///
/// Note that the excess of a zero-sized type is always infinite, so
/// `capacity()` always returns `usize::MAX`. This means that you need to be
/// careful when round-tripping this type with a `Box<[T]>`, since `capacity()`
/// won't yield the length.
#[allow(missing_debug_implementations)]
pub(crate) struct RawVec<T, A: Allocator = Global> {
    inner: RawVecInner<A>,
    _marker: PhantomData<T>,
}

/// Like a `RawVec`, but only generic over the allocator, not the type.
///
/// As such, all the methods need the layout passed-in as a parameter.
///
/// Having this separation reduces the amount of code we need to monomorphize,
/// as most operations don't need the actual type, just its layout.
#[allow(missing_debug_implementations)]
struct RawVecInner<A: Allocator = Global> {
    // The base address of the allocation. We store a thin `NonNull<u8>` (rather
    // than std's unstable `Unique`) because the length is always derivable from
    // `cap` and the element layout at the point of use.
    ptr: NonNull<u8>,
    /// Never used for ZSTs; it's `capacity()`'s responsibility to return
    /// `usize::MAX` in that case.
    ///
    /// # Safety
    ///
    /// `cap` must be in the `0..=isize::MAX` range.
    cap: Cap,
    alloc: A,
}

impl<T> RawVec<T, Global> {
    /// Creates the biggest possible `RawVec` (on the system heap) without
    /// allocating. If `T` has positive size, then this makes a `RawVec` with
    /// capacity `0`. If `T` is zero-sized, then it makes a `RawVec` with
    /// capacity `usize::MAX`. Useful for implementing delayed allocation.
    #[must_use]
    pub const fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a `RawVec` (on the system heap) with exactly the capacity and
    /// alignment requirements for a `[T; capacity]`. This is equivalent to
    /// calling `RawVec::new` when `capacity` is `0` or `T` is zero-sized. Note
    /// that if `T` is zero-sized this means you will *not* get a `RawVec` with
    /// the requested capacity.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// allocation fails.
    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        let inner = RawVecInner::try_with_capacity_in(capacity, Global, elem_layout::<T>())?;
        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Like [`Self::try_with_capacity`], but guarantees the buffer is zeroed.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// allocation fails.
    #[inline]
    pub fn try_with_capacity_zeroed(capacity: usize) -> Result<Self, TryReserveError> {
        let inner = RawVecInner::try_with_capacity_zeroed_in(capacity, Global, elem_layout::<T>())?;
        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }
}

// Tiny Vecs are dumb. Skip to:
// - 8 if the element size is 1, because any heap allocator is likely
//   to round up a request of less than 8 bytes to at least 8 bytes.
// - 4 if elements are moderate-sized (<= 1 KiB).
// - 1 otherwise, to avoid wasting too much space for very short Vecs.
const fn min_non_zero_cap(size: usize) -> usize {
    if size == 1 {
        8
    } else if size <= 1024 {
        4
    } else {
        1
    }
}

/// The layout of a single element of type `T`.
const fn elem_layout<T>() -> Layout {
    Layout::new::<T>()
}

impl<T, A: Allocator> RawVec<T, A> {
    /// Minimum non-zero capacity for this element size, matching std's growth
    /// heuristic.
    #[allow(unused)]
    pub(crate) const MIN_NON_ZERO_CAP: usize = min_non_zero_cap(size_of::<T>());

    /// Like [`Self::new`], but parameterized over the choice of allocator for the
    /// returned `RawVec`.
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            inner: unsafe { RawVecInner::new_in(alloc, align_of::<T>()) },
            _marker: PhantomData,
        }
    }

    /// Like [`Self::try_with_capacity`], but parameterized over the choice of allocator for
    /// the returned `RawVec`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// allocation fails.
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, TryReserveError> {
        let inner = RawVecInner::try_with_capacity_in(capacity, alloc, elem_layout::<T>())?;
        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Like [`Self::try_with_capacity_zeroed`], but parameterized over the choice of
    /// allocator for the returned `RawVec`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// allocation fails.
    #[allow(unused)]
    #[inline]
    pub fn try_with_capacity_zeroed_in(capacity: usize, alloc: A) -> Result<Self, TryReserveError> {
        let inner = RawVecInner::try_with_capacity_zeroed_in(capacity, alloc, elem_layout::<T>())?;
        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Converts the entire buffer into `Box<[MaybeUninit<T>]>` with the specified `len`.
    ///
    /// Note that this will correctly reconstitute any `cap` changes
    /// that may have been performed. (See description of type for details.)
    ///
    /// # Safety
    ///
    /// * `len` must be greater than or equal to the most recently requested capacity, and
    /// * `len` must be less than or equal to `self.capacity()`.
    ///
    /// Note, that the requested capacity and `self.capacity()` could differ, as
    /// an allocator could overallocate and return a greater memory block than requested.
    #[inline]
    pub unsafe fn into_box(self, len: usize) -> Box<[MaybeUninit<T>], A> {
        // Sanity-check one half of the safety requirement (we cannot check the other half).
        debug_assert!(
            len <= self.capacity(),
            "`len` must be smaller than or equal to `self.capacity()`"
        );

        let me = ManuallyDrop::new(self);
        unsafe {
            let raw: *mut [MaybeUninit<T>] = ptr::slice_from_raw_parts_mut(me.ptr().cast(), len);
            // Move the allocator over.
            Box::from_raw_in(raw, ptr::read(&me.inner.alloc))
        }
    }

    /// Reconstitutes a `RawVec` from a pointer, capacity, and
    /// allocator.
    ///
    /// # Safety
    ///
    /// The `ptr` must be non-null, allocated (via the given allocator `alloc`) if T is non zero-sized
    /// or dangling if T is zero-sized, must be well aligned for T, and with the `capacity` that is 
    /// between the previous requested capacity and the actual capacity (including both ends).
    /// The `capacity` cannot exceed `isize::MAX` for sized types. For ZSTs `capacity` is ignored.
    /// If the `ptr` and `capacity` come from a `RawVec` created via `alloc`, then this is guaranteed.
    #[inline]
    pub const unsafe fn from_raw_parts_in(ptr: *mut T, capacity: usize, alloc: A) -> Self {
        // SAFETY: Precondition passed to the caller.
        unsafe {
            let ptr = ptr.cast();
            let capacity = Cap::new_zst_aware::<T>(capacity);
            Self {
                inner: RawVecInner::from_raw_parts_in(ptr, capacity, alloc),
                _marker: PhantomData,
            }
        }
    }

    /// A convenience method for hoisting the non-null precondition out of
    /// [`RawVec::from_raw_parts_in`].
    ///
    /// # Safety
    ///
    /// See [`RawVec::from_raw_parts_in`].
    #[inline]
    pub const unsafe fn from_nonnull_in(ptr: NonNull<T>, capacity: usize, alloc: A) -> Self {
        // SAFETY: Precondition passed to the caller.
        unsafe {
            let ptr = ptr.cast();
            let capacity = Cap::new_zst_aware::<T>(capacity);
            Self {
                inner: RawVecInner::from_nonnull_in(ptr, capacity, alloc),
                _marker: PhantomData,
            }
        }
    }

    /// Gets a raw pointer to the start of the allocation. Note that this is a
    /// well-aligned [`NonNull::dangling`] if `capacity == 0` or `T` is zero-sized.
    /// In the former case, you must be careful.
    #[inline]
    pub const fn ptr(&self) -> *mut T {
        self.inner.ptr()
    }

    #[inline]
    pub const fn non_null(&self) -> NonNull<T> {
        self.inner.non_null()
    }

    /// Gets the capacity of the allocation.
    ///
    /// This will always be `usize::MAX` if `T` is zero-sized.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.inner.capacity(size_of::<T>())
    }

    /// Returns a shared reference to the allocator backing this `RawVec`.
    #[inline]
    pub const fn allocator(&self) -> &A {
        self.inner.allocator()
    }

    /// Ensures that the buffer contains at least enough space to hold `len +
    /// additional` elements. If it doesn't already have enough capacity, will
    /// reallocate enough space plus comfortable slack space to get amortized
    /// *O*(1) behavior. Will limit this behavior if it would needlessly cause
    /// itself to fail.
    ///
    /// If `len` exceeds `self.capacity()`, this may fail to actually allocate
    /// the requested space. This is not really unsafe, but the unsafe code
    /// *you* write that relies on the behavior of this function may break.
    ///
    /// This is ideal for implementing a bulk-push operation like `extend`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the new capacity overflows or the
    /// allocation fails.
    pub fn try_reserve(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.inner.try_reserve(len, additional, elem_layout::<T>()) }
    }

    /// Ensures that the buffer contains at least enough space to hold `len +
    /// additional` elements. If it doesn't already, will reallocate the minimum
    /// possible amount of memory necessary. Generally this will be exactly the
    /// amount of memory necessary, but in principle the allocator is free to
    /// give back more than we asked for.
    ///
    /// If `len` exceeds `self.capacity()`, this may fail to actually allocate
    /// the requested space. This is not really unsafe, but the unsafe code
    /// *you* write that relies on the behavior of this function may break.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the new capacity overflows or the
    /// allocation fails.
    pub fn try_reserve_exact(
        &mut self,
        len: usize,
        additional: usize,
    ) -> Result<(), TryReserveError> {
        unsafe {
            self.inner
                .try_reserve_exact(len, additional, elem_layout::<T>())
        }
    }

    /// A specialized version of `self.reserve(len, 1)` which requires the
    /// caller to ensure `len == self.capacity()`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the new capacity overflows or the
    /// allocation fails.
    #[inline(never)]
    pub unsafe fn try_grow_one(&mut self) -> Result<(), TryReserveError> {
        unsafe { self.inner.try_grow_one(elem_layout::<T>()) }
    }

    /// Shrinks the buffer down to the specified capacity. If the given amount
    /// is 0, actually completely deallocates.
    ///
    /// # Panics
    ///
    /// Panics if the given amount is *larger* than the current capacity.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the allocation fails.
    #[track_caller]
    pub fn try_shrink_to_fit(&mut self, cap: usize) -> Result<(), TryReserveError> {
        self.inner.shrink(cap, elem_layout::<T>())
    }
}

impl<T, A: Allocator> Drop for RawVec<T, A> {
    /// Frees the memory owned by the `RawVec` *without* trying to drop its
    /// contents.
    fn drop(&mut self) {
        // SAFETY: We are in a Drop impl, self.inner will not be used again.
        unsafe { self.inner.deallocate(elem_layout::<T>()) }
    }
}

impl<A: Allocator> RawVecInner<A> {
    /// # Safety
    ///
    /// `align` is a power of two.
    #[inline]
    const unsafe fn new_in(alloc: A, align: usize) -> Self {
        debug_assert!(align.is_power_of_two());
        let ptr = unsafe { transmute::<usize, NonNull<u8>>(align) };
        // `cap: 0` means "unallocated". zero-sized types are ignored.
        Self {
            ptr,
            cap: Cap::ZERO,
            alloc,
        }
    }

    #[inline]
    fn try_with_capacity_in(
        capacity: usize,
        alloc: A,
        elem_layout: Layout,
    ) -> Result<Self, TryReserveError> {
        let this = Self::try_allocate_in(capacity, AllocInit::Uninitialized, alloc, elem_layout)?;
        unsafe {
            // Make it more obvious that a subsequent Vec::reserve(capacity) will not allocate.
            hint::assert_unchecked(!this.needs_to_grow(0, capacity, elem_layout));
        };
        Ok(this)
    }

    fn try_allocate_in(
        capacity: usize,
        init: AllocInit,
        alloc: A,
        elem_layout: Layout,
    ) -> Result<Self, TryReserveError> {
        // We avoid `unwrap_or_else` here because it bloats the amount of
        // LLVM IR generated.
        let layout = match layout_array(capacity, elem_layout) {
            Ok(layout) => layout,
            Err(_) => return Err(CapacityOverflow.into()),
        };

        // Don't allocate here because `Drop` will not deallocate when `capacity` is 0.
        if layout.size() == 0 {
            return Ok(unsafe { Self::new_in(alloc, elem_layout.align()) });
        }

        let result = match init {
            AllocInit::Uninitialized => alloc.allocate(layout),
            AllocInit::Zeroed => alloc.allocate_zeroed(layout),
        };
        let ptr = match result {
            Ok(ptr) => base_ptr(ptr),
            Err(_) => return Err(TryReserveError::new_alloc(layout)),
        };

        // The allocator returns a slice pointer whose length matches the size
        // requested; we store only the thin base address. This path is only
        // reachable for non-ZST `T` (the ZST case returned early above), so the
        // real capacity is stored.
        Ok(Self {
            ptr,
            cap: unsafe { Cap::new_unchecked(capacity) },
            alloc,
        })
    }

    /// # Safety
    /// - Caller must assert len == capacity
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    #[inline]
    unsafe fn try_grow_one(&mut self, elem_layout: Layout) -> Result<(), TryReserveError> {
        // SAFETY: Precondition passed to caller. 
        // additionally `self.cap.0` is meaningless if elem_layout encodes ZST
        unsafe { self.grow_amortized(self.cap.0, 1, elem_layout) }
    }

    #[inline]
    fn try_with_capacity_zeroed_in(
        capacity: usize,
        alloc: A,
        elem_layout: Layout,
    ) -> Result<Self, TryReserveError> {
        Self::try_allocate_in(capacity, AllocInit::Zeroed, alloc, elem_layout)
    }

    #[inline]
    const unsafe fn from_raw_parts_in(ptr: *mut u8, cap: Cap, alloc: A) -> Self {
        Self {
            // SAFETY: caller guarantees `ptr` is a valid, aligned, non-null
            // allocation (or the dangling pointer for empty/ZST buffers).
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            cap,
            alloc,
        }
    }

    #[inline]
    const unsafe fn from_nonnull_in(ptr: NonNull<u8>, cap: Cap, alloc: A) -> Self {
        Self { ptr, cap, alloc }
    }

    #[inline]
    const fn ptr<T>(&self) -> *mut T {
        self.non_null::<T>().as_ptr()
    }

    #[inline]
    const fn non_null<T>(&self) -> NonNull<T> {
        self.ptr.cast()
    }

    #[inline]
    const fn capacity(&self, elem_size: usize) -> usize {
        if elem_size == 0 {
            usize::MAX
        } else {
            self.cap.0
        }
    }

    #[inline]
    const fn allocator(&self) -> &A {
        &self.alloc
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    #[inline]
    const unsafe fn current_memory(&self, elem_layout: Layout) -> Option<(NonNull<u8>, Layout)> {
        if elem_layout.size() == 0 || self.cap.0 == 0 {
            None
        } else {
            // We could use Layout::array here which ensures the absence of isize and
            // usize overflows and could hypothetically handle differences between stride
            // and size, but this memory has already been allocated so we know it can't
            // overflow and currently Rust does not support such types. So we can do
            // better by skipping some checks and avoid an unwrap.
            unsafe {
                let alloc_size = elem_layout.size().unchecked_mul(self.cap.0);
                let layout = Layout::from_size_align_unchecked(alloc_size, elem_layout.align());
                Some((self.ptr, layout))
            }
        }
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    unsafe fn try_reserve(
        &mut self,
        len: usize,
        additional: usize,
        elem_layout: Layout,
    ) -> Result<(), TryReserveError> {
        if self.needs_to_grow(len, additional, elem_layout) {
            // SAFETY: Precondition passed to caller
            unsafe { self.grow_amortized(len, additional, elem_layout)? };
        }
        unsafe {
            // Inform the optimizer that the reservation has succeeded or wasn't needed
            hint::assert_unchecked(!self.needs_to_grow(len, additional, elem_layout));
        }
        Ok(())
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    unsafe fn try_reserve_exact(
        &mut self,
        len: usize,
        additional: usize,
        elem_layout: Layout,
    ) -> Result<(), TryReserveError> {
        if self.needs_to_grow(len, additional, elem_layout) {
            // SAFETY: Precondition passed to caller
            unsafe { self.grow_exact(len, additional, elem_layout)? };
        }
        unsafe {
            // Inform the optimizer that the reservation has succeeded or wasn't needed
            hint::assert_unchecked(!self.needs_to_grow(len, additional, elem_layout));
        }
        Ok(())
    }

    #[inline]
    fn needs_to_grow(&self, len: usize, additional: usize, elem_layout: Layout) -> bool {
        // Correct even if `len > capacity` (degenerate/probe inputs): compute
        // the free space as a saturating difference so we never silently wrap
        // into a false "no growth needed". When `len <= capacity` this is
        // identical to the classic `capacity - len` formulation.
        let cap = self.capacity(elem_layout.size());
        let free = cap.saturating_sub(len);
        additional > free
    }

    #[inline]
    unsafe fn set_ptr_and_cap(&mut self, ptr: NonNull<[u8]>, cap: usize) {
        // Allocators currently return a `NonNull<[u8]>` whose length matches
        // the size requested. If that ever changes, the capacity here should
        // change to `ptr.len() / size_of::<T>()`.
        self.ptr = ptr.cast();
        // Reaching here implies a non-ZST reallocation, so the real capacity is stored.
        self.cap = unsafe { Cap::new_unchecked(cap) };
    }

    fn shrink(&mut self, cap: usize, elem_layout: Layout) -> Result<(), TryReserveError> {
        assert!(
            cap <= self.capacity(elem_layout.size()),
            "Tried to shrink to a larger capacity"
        );
        // SAFETY: Just checked this isn't trying to grow.
        unsafe { self.shrink_unchecked(cap, elem_layout) }
    }

    /// `shrink`, but without the capacity check.
    ///
    /// This is split out so that `shrink` can inline the check, since it
    /// optimizes out in things like `shrink_to_fit`, without needing to also
    /// inline all this code, as doing that ends up failing the
    /// `vec-shrink-panic` codegen test when `shrink_to_fit` ends up being too
    /// big for LLVM to be willing to inline.
    ///
    /// # Safety
    ///
    /// `cap <= self.capacity()`
    unsafe fn shrink_unchecked(
        &mut self,
        cap: usize,
        elem_layout: Layout,
    ) -> Result<(), TryReserveError> {
        let (ptr, layout) = if let Some(mem) = unsafe { self.current_memory(elem_layout) } {
            mem
        } else {
            return Ok(());
        };

        // If shrinking to 0, deallocate the buffer. We don't reach this point
        // for the ZST case since current_memory() will have returned None.
        if cap == 0 {
            unsafe { self.alloc.deallocate(ptr, layout) };
            self.ptr = layout.dangling_pointer();
            self.cap = Cap::ZERO;
        } else {
            let new_ptr = unsafe {
                // Layout cannot overflow here because it would have
                // overflowed earlier when capacity was larger.
                let new_size = elem_layout.size().unchecked_mul(cap);
                let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
                self.alloc
                    .shrink(ptr, layout, new_layout)
                    .map_err(|_| TryReserveError::new_alloc(new_layout))?
            };
            // SAFETY: if the allocation is valid, then the capacity is too.
            unsafe {
                self.set_ptr_and_cap(new_ptr, cap);
            }
        }
        Ok(())
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    /// - The sum of `len` and `additional` must be greater than the current capacity
    unsafe fn grow_amortized(
        &mut self,
        len: usize,
        additional: usize,
        elem_layout: Layout,
    ) -> Result<(), TryReserveError> {
        // This is ensured by the calling contexts.
        debug_assert!(additional > 0);

        if elem_layout.size() == 0 {
            // Since we return a capacity of `usize::MAX` when `elem_size` is
            // 0, getting to here necessarily means the `RawVec` is overfull.
            return Err(CapacityOverflow.into());
        }

        // Nothing we can really do about these checks, sadly.
        let required_cap = len.checked_add(additional).ok_or(CapacityOverflow)?;

        // This guarantees exponential growth. The doubling cannot overflow
        // because `cap <= isize::MAX` and the type of `cap` is `usize`.
        let cap = cmp::max(self.cap.0.saturating_mul(2), required_cap);
        let cap = cmp::max(min_non_zero_cap(elem_layout.size()), cap);

        // SAFETY:
        // - cap >= len + additional
        // - other preconditions passed to caller
        let ptr = unsafe { self.finish_grow(cap, elem_layout) }?;

        // SAFETY: finish_grow would have resulted in a capacity overflow if we
        // tried to allocate more than `isize::MAX` items.
        unsafe { self.set_ptr_and_cap(ptr, cap) };
        Ok(())
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    /// - The sum of `len` and `additional` must be greater than the current capacity
    unsafe fn grow_exact(
        &mut self,
        len: usize,
        additional: usize,
        elem_layout: Layout,
    ) -> Result<(), TryReserveError> {
        if elem_layout.size() == 0 {
            // Since we return a capacity of `usize::MAX` when the type size is
            // 0, getting to here necessarily means the `RawVec` is overfull.
            return Err(CapacityOverflow.into());
        }

        let cap = len.checked_add(additional).ok_or(CapacityOverflow)?;

        // SAFETY:
        // - cap >= len + additional
        // - other preconditions passed to caller
        let ptr = unsafe { self.finish_grow(cap, elem_layout) }?;
        // SAFETY: finish_grow would have resulted in a capacity overflow if we
        // tried to allocate more than `isize::MAX` items.

        unsafe {
            self.set_ptr_and_cap(ptr, cap);
        }
        Ok(())
    }

    /// # Safety
    ///
    /// This function deallocates the owned allocation, but does not update
    /// `ptr` or `cap` to prevent double-free or use-after-free. Essentially, do
    /// not do anything with the caller after this function returns. Ideally this
    /// function would take `self` by move, but it cannot because it exists to
    /// be called from a `Drop` impl.
    unsafe fn deallocate(&mut self, elem_layout: Layout) {
        if let Some((ptr, layout)) = unsafe { self.current_memory(elem_layout) } {
            unsafe {
                self.alloc.deallocate(ptr, layout);
            }
        }
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    /// - `cap` must be greater than the current capacity
    // not marked inline(never) since we want optimizers to be able to observe the specifics of this
    // function, see tests/codegen-llvm/vec-reserve-extend.rs.
    #[cold]
    unsafe fn finish_grow(
        &self,
        cap: usize,
        elem_layout: Layout,
    ) -> Result<NonNull<[u8]>, TryReserveError> {
        let new_layout = layout_array(cap, elem_layout)?;

        let memory = if let Some((ptr, old_layout)) = unsafe { self.current_memory(elem_layout) } {
            debug_assert_eq!(old_layout.align(), new_layout.align());
            unsafe {
                // The allocator checks for alignment equality
                hint::assert_unchecked(old_layout.align() == new_layout.align());
                self.alloc
                    .grow(ptr, old_layout, new_layout)
                    .map_err(|_| TryReserveError::new_alloc(new_layout))
            }
        } else {
            // Normalize the slice pointer returned by `allocate` down to its
            // thin base address.
            self.alloc
                .allocate(new_layout)
                .map_err(|_| TryReserveError::new_alloc(new_layout))
        }?;

        Ok(memory)
    }
}

/// Extracts the base `NonNull<u8>` from a fat `NonNull<[u8]>`.
///
/// `NonNull::<[T]>::as_non_null_ptr` is not yet stable, so we recover the base
/// pointer through a raw-pointer cast. Mirrors the private helper in
/// `olive_core::allocator`.
const fn base_ptr(slice: NonNull<[u8]>) -> NonNull<u8> {
    slice.cast()
}

/// Obtains the packed array layout for `cap` number of elements. This also caps the capacity at isize::MAX.
#[inline]
fn layout_array(cap: usize, elem_layout: Layout) -> Result<Layout, TryReserveError> {
    if let Some(size) = elem_layout.size().checked_mul(cap) {
        // The safe constructor is called here to enforce the isize size limit.
        Layout::from_size_align(size, elem_layout.align())
            .map_err(|_| TryReserveError::new_capacity_overflow())
    } else {
        Err(TryReserveError::new_capacity_overflow())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::AllocError;
    use std::format;

    #[test]
    fn new_is_empty() {
        let v: RawVec<i32> = RawVec::new();
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn zst_capacity_is_max() {
        let v: RawVec<()> = RawVec::new();
        assert_eq!(v.capacity(), usize::MAX);
    }

    #[test]
    fn try_with_capacity_allocates() {
        let v = RawVec::<i32>::try_with_capacity(10).expect("alloc ok");
        assert!(v.capacity() >= 10);
        // Write and read back through the raw pointer.
        unsafe {
            for i in 0..10 {
                v.ptr().add(i).write(i as i32);
            }
            for i in 0..10 {
                assert_eq!(v.ptr().add(i).read(), i as i32);
            }
        }
    }

    #[test]
    fn try_with_capacity_zeroed_is_zeroed() {
        let v = RawVec::<u8>::try_with_capacity_zeroed(64).expect("zeroed alloc ok");
        unsafe {
            for i in 0..64 {
                assert_eq!(v.ptr().add(i).read(), 0);
            }
        }
    }

    #[test]
    fn reserve_grows_amortized() {
        let mut v = RawVec::<i32>::new();
        v.try_reserve(0, 5).expect("reserve ok");
        assert!(v.capacity() >= 5);
        // Pushing past capacity forces another growth.
        v.try_reserve(v.capacity(), 1).expect("second reserve ok");
        assert!(v.capacity() > 5);
    }

    #[test]
    fn reserve_exact_grows_minimally() {
        let mut v = RawVec::<i32>::new();
        v.try_reserve_exact(0, 7).expect("exact reserve ok");
        assert!(v.capacity() >= 7);
    }

    #[test]
    fn shrink_releases_capacity() {
        let mut v = RawVec::<i32>::try_with_capacity(100).expect("alloc ok");
        let before = v.capacity();
        assert!(before >= 100);
        v.try_shrink_to_fit(4).expect("shrink ok");
        assert!(v.capacity() <= before);
        assert!(v.capacity() >= 4);
    }

    #[test]
    fn shrink_to_zero_deallocates() {
        let mut v = RawVec::<i32>::try_with_capacity(32).expect("alloc ok");
        v.try_shrink_to_fit(0).expect("shrink to 0 ok");
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn capacity_overflow_detected() {
        let mut v = RawVec::<i32>::new();
        // Request absurdly large capacity: len + additional overflows usize.
        let err = v
            .try_reserve(usize::MAX, usize::MAX)
            .expect_err("should overflow");
        assert!(err.is_capacity_overflow());
    }

    #[test]
    fn oom_returns_alloc_error_kind() {
        // Use a custom allocator that always fails to prove the error kind is
        // AllocError (not CapacityOverflow).
        #[derive(Default)]
        struct FailAlloc;
        unsafe impl Allocator for FailAlloc {
            fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
                Err(AllocError)
            }
            unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
        }

        let err = match RawVec::<i32, FailAlloc>::try_with_capacity_in(8, FailAlloc) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        let msg = format!("{err}");
        assert!(msg.contains("allocation"));
    }

    #[test]
    fn from_raw_parts_roundtrip() {
        let v = RawVec::<i32>::try_with_capacity(4).expect("alloc ok");
        unsafe {
            for i in 0..4 {
                v.ptr().add(i).write((i * 10) as i32);
            }
        }
        let ptr = v.ptr();
        let cap = v.capacity();
        // Prevent the original from deallocating on drop so we can reuse the
        // allocation via `from_raw_parts_in`.
        std::mem::forget(v);
        // SAFETY: `ptr` was allocated by the global allocator with `cap` elements.
        let v2 = unsafe { RawVec::<i32>::from_raw_parts_in(ptr, cap, Global) };
        assert_eq!(v2.capacity(), cap);
        unsafe {
            for i in 0..4 {
                assert_eq!(v2.ptr().add(i).read(), (i * 10) as i32);
            }
        }
    }

    #[test]
    fn min_non_zero_cap_heuristic() {
        assert_eq!(min_non_zero_cap(1), 8);
        assert_eq!(min_non_zero_cap(64), 4);
        assert_eq!(min_non_zero_cap(4096), 1);
    }
}
