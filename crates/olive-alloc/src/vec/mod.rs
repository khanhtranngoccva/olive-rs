//! A fully-fallible port of the standard library's `Vec`.
//!
//! Compared with the std original, three things differ:
//!
//! * Every operation that can grow the buffer — `push`, `insert`, `extend`,
//!   `resize`, capacity reservations, and every allocating constructor — returns
//!   a [`Result`] carrying an error instead of panicking on out-of-memory. The
//!   primary error is [`TryReserveError`]; operations that also clone elements
//!   additionally surface [`TryCloneError`].
//! * All allocation goes through the [`Allocator`] trait rather than free
//!   functions, giving a single swappable seam for custom allocators and OOM
//!   simulation in tests.
//! * Compiler-internal attributes (`#[lang = "exchange_vector_like"]`) and
//!   unstable helpers are omitted; this is a pure type with no compiler magic.
//!
//! Element cloning uses the fallible [`TryClone`] trait throughout, so a `Vec<T>`
//! can hold values whose own construction can fail (nested collections, boxes,
//! …) without ever aborting.
//!
//! # Arithmetic safety
//!
//! All index and length arithmetic in this module operates on values bounded by
//! `self.len` and `self.capacity()`, both of which are `usize` fields maintained
//! under strict invariants (`len <= capacity`). Overflow would require
//! `capacity > usize::MAX`, which is impossible because the allocator rejects
//! such layouts. We therefore suppress `clippy::arithmetic_side_effects` at the
//! module level rather than scattering 30+ individual allows or converting
//! every `+= 1` into a checked operation that would panic anyway.

#![allow(clippy::arithmetic_side_effects)]

use core::cmp;
use core::fmt;
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::ptr;
use core::slice;

use crate::alloc::{Allocator, Global};
use crate::boxed::Box;
use crate::raw_vec::RawVec;
use olive_core::alloc_errors::TryReserveError;
use olive_core::recovery::{ResumableSource, Resume};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
use olive_core::try_traits::try_from_iterator::TryFromIterator;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned by fallible vector operations that may both reserve capacity
/// and clone elements.
///
/// Covers `try_from_elem`, `try_from_slice`, `try_resize`, and
/// `try_extend_from_slice_with_rollback` — any operation whose failure modes are
/// limited to a capacity reservation ([`TryReserveError`]) or an element clone
/// failure ([`TryCloneError`]).
#[derive(Clone, PartialEq, Eq)]
pub enum TryVecWithCloneError {
    /// A capacity reservation on the vector failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
}

impl fmt::Debug for TryVecWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .debug_tuple("TryVecWithCloneError::Reserve")
                .field(e)
                .finish(),
            Self::Clone(e) => f
                .debug_tuple("TryVecWithCloneError::Clone")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryVecWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "vector operation failed: {e}"),
            Self::Clone(e) => write!(f, "vector operation failed: {e}"),
        }
    }
}

impl core::error::Error for TryVecWithCloneError {}

impl From<TryReserveError> for TryVecWithCloneError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryVecWithCloneError {
    #[inline]
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

/// Error returned by fallible vector insert operations.
///
/// Used by [`Vec::try_insert`] and [`Vec::try_insert_give_back`]: the operation
/// can fail either because growing the buffer failed or because the index was
/// out of bounds. In the give-back variant the value travels alongside this
/// error as a tuple: `Result<(), (T, TryVecInsertError)>`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TryVecInsertError {
    /// A capacity reservation failed.
    Reserve(TryReserveError),
    /// The provided index exceeded the vector's length.
    OutOfBounds,
}

impl fmt::Debug for TryVecInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .debug_tuple("TryVecInsertError::Reserve")
                .field(e)
                .finish(),
            Self::OutOfBounds => f.debug_tuple("TryVecInsertError::OutOfBounds").finish(),
        }
    }
}

impl fmt::Display for TryVecInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "vector insert failed: {e}"),
            Self::OutOfBounds => write!(f, "vector insert failed: index out of bounds"),
        }
    }
}

impl core::error::Error for TryVecInsertError {}

/// Error returned by [`Vec::try_swap_remove`] and [`Vec::try_remove`] when the
/// provided index is out of bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryVecRemoveError {
    /// The index that was provided.
    pub index: usize,
    /// The vector's length at the time of the call.
    pub len: usize,
}

impl fmt::Display for TryVecRemoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "index ({}) is out of bounds len ({})",
            self.index, self.len
        )
    }
}

impl core::error::Error for TryVecRemoveError {}

/// Error returned by [`Vec::try_push_within_capacity`] when the buffer has no
/// remaining capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryPushWithinCapacityError {
    /// The current length (equal to capacity).
    pub len: usize,
}

impl fmt::Display for TryPushWithinCapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no spare capacity: vec is full at len {}", self.len)
    }
}

impl core::error::Error for TryPushWithinCapacityError {}

/// Error returned by [`Vec::try_from_fn`] when constructing a vector from a
/// fallible closure.
///
/// The closure may fail with any error type `E`, and the capacity reservation
/// itself may also fail independently.
#[derive(Clone, PartialEq, Eq)]
pub enum TryVecWithClosureError<E> {
    /// A capacity reservation on the vector failed (overflow or OOM).
    Reserve(TryReserveError),
    /// The closure returned an error.
    Closure(E),
}

impl<E: fmt::Debug> fmt::Debug for TryVecWithClosureError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .debug_tuple("TryVecWithClosureError::Reserve")
                .field(e)
                .finish(),
            Self::Closure(e) => f
                .debug_tuple("TryVecWithClosureError::Closure")
                .field(e)
                .finish(),
        }
    }
}

impl<E: fmt::Display> fmt::Display for TryVecWithClosureError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "vector construction failed: {e}"),
            Self::Closure(e) => write!(f, "vector construction failed: {e}"),
        }
    }
}

impl<E: core::error::Error> core::error::Error for TryVecWithClosureError<E> {}

impl<E> From<TryReserveError> for TryVecWithClosureError<E> {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

/// Error returned by [`Vec::try_into_array`] when converting a vector into a
/// boxed array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryVecIntoArrayError {
    /// The vector's length did not match the target array size.
    LengthMismatch {
        /// The expected (target array) length.
        expected: usize,
        /// The actual vector length at the time of the call.
        actual: usize,
    },
    /// The shrink-to-fit reallocation needed to eliminate excess capacity
    /// failed.
    Shrink(TryReserveError),
}

impl fmt::Display for TryVecIntoArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "cannot convert Vec into [T; {expected}]: vector has {actual} elements"
            ),
            Self::Shrink(e) => write!(f, "shrink-to-fit failed during into_array: {e}"),
        }
    }
}

impl core::error::Error for TryVecIntoArrayError {}

impl From<TryReserveError> for TryVecIntoArrayError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Shrink(err)
    }
}

// ---------------------------------------------------------------------------
// Vec
// ---------------------------------------------------------------------------

/// A growable list of memory, backed by a [`RawVec`] buffer.
///
/// This is the fully-fallible analogue of `std::vec::Vec`: every operation that
/// could allocate or fail returns a [`Result`] instead of panicking. It is generic over
/// the [`Allocator`] used for all heap traffic (defaulting to [`Global`]).
pub struct Vec<T, A: Allocator = Global> {
    raw: RawVec<T, A>,
    len: usize,
}

// SAFETY: `Vec` never hands out references that outlive the buffer, and moving
// a `Vec` moves its whole allocation. Sound iff `T` itself is `Send`/`Sync`.
unsafe impl<T: Send, A: Allocator + Send> Send for Vec<T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for Vec<T, A> {}

// ---------------------------------------------------------------------------
// Constructors and reconstitution — global allocator
// ---------------------------------------------------------------------------

impl<T> Vec<T, Global> {
    /// Constructs a new, empty `Vec<T>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            raw: RawVec::new(),
            len: 0,
        }
    }

    /// Constructs a new, empty `Vec<T>` with at least the specified capacity.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the vector will not allocate.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// initial allocation fails.
    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity_in(capacity, Global)
    }

    /// Creates a `Vec<T>` containing `value` cloned `count` times.
    ///
    /// Equivalent to `vec![value; count]` but fully fallible.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    pub fn try_from_elem(value: &T, count: usize) -> Result<Self, TryVecWithCloneError>
    where
        T: TryClone,
    {
        Self::try_from_elem_in(value, count, Global)
    }

    /// Creates a `Vec<T>` by calling `f` `n` times, collecting the results.
    ///
    /// The closure may return an error at any point; construction stops
    /// immediately and the partially-built buffer is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithClosureError<E>`] if either the capacity
    /// reservation fails or the closure returns `Err(e)`.
    pub fn try_from_fn<E, F>(n: usize, mut f: F) -> Result<Self, TryVecWithClosureError<E>>
    where
        F: FnMut() -> Result<T, E>,
    {
        let mut vec = Self::new();
        if n > 0 {
            vec.raw
                .try_reserve_exact(0, n)
                .map_err(TryVecWithClosureError::Reserve)?;
        }
        for _ in 0..n {
            match f() {
                Ok(item) => {
                    // Capacity was reserved above, so this cannot fail.
                    unsafe {
                        vec.raw.ptr().add(vec.len).write(item);
                    }
                    vec.len += 1;
                }
                Err(e) => return Err(TryVecWithClosureError::Closure(e)),
            }
        }
        Ok(vec)
    }

    /// Creates a `Vec<T>` directly from a pointer, a length, and a capacity.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// - If T is not a zero-sized type and the capacity is nonzero, ptr must have been allocated using the global allocator.
    ///   If T is a zero-sized type or the capacity is zero, ptr need only be non-null and aligned.
    /// - T needs to have the same alignment as what ptr was allocated with, if the pointer is required to be allocated.
    ///   (T having a less strict alignment is not sufficient, the alignment really needs to be equal to satisfy the dealloc requirement that memory must be allocated and deallocated with the same layout.)
    /// - The size of T times the `capacity` (i.e. the allocated size in bytes), if nonzero, needs to be the same size as the pointer was allocated with.
    ///   (Because similar to alignment, dealloc must be called with the same layout size.)
    /// - `length` needs to be less than or equal to `capacity`.
    /// - The first `length` values must be properly initialized values of type T.
    /// - `capacity` needs to be the capacity that the pointer was allocated with, if the pointer is required to be allocated.
    /// - The allocated size in bytes must be no larger than `isize::MAX`. See the safety documentation of [`pointer::offset`](core::pointer::offset).
    // FIXME: pointer::offset is unlinked
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Self {
        debug_assert!(
            length <= capacity,
            "Vec::from_raw_parts requires that length <= capacity"
        );
        unsafe { Self::from_raw_parts_in(ptr, length, capacity, Global) }
    }

    /// Creates a `Vec<T>` from a non-null pointer, length, and capacity.
    ///
    /// # Safety
    ///
    /// Same requirements as [`Self::from_raw_parts`], except `ptr` is already
    /// known to be non-null.
    #[inline]
    pub unsafe fn from_parts(ptr: ptr::NonNull<T>, length: usize, capacity: usize) -> Self {
        unsafe { Self::from_parts_in(ptr, length, capacity, Global) }
    }

    /// Decomposes a `Vec<T>` into its constituent parts: a raw pointer, a
    /// length, and a capacity.
    ///
    /// After calling this function, the caller is responsible for the memory
    /// previously managed by the `Vec`. Most often, one does this by converting
    /// the raw pointer, length, and capacity back into a `Vec` with the
    /// [`from_raw_parts`] function.
    ///
    /// The returned pointer is non-null if and only if the capacity is non-zero.
    ///
    /// [`from_raw_parts`]: Self::from_raw_parts
    #[must_use = "losing the pointer will leak memory"]
    pub fn into_raw_parts(self) -> (*mut T, usize, usize) {
        let this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller, who
        // takes over ownership of the allocation.
        (this.raw.ptr(), this.len, this.raw.capacity())
    }

    /// Decomposes a `Vec<T>` into its constituent parts: a non-null pointer, a
    /// length, and a capacity.
    ///
    /// This is the [`NonNull`](core::ptr::NonNull) counterpart of [`Self::into_raw_parts`].
    ///
    /// After calling this function, the caller is responsible for the memory
    /// previously managed by the `Vec`. Most often, one does this by converting
    /// the raw pointer, length, and capacity back into a `Vec` with the
    /// [`from_parts`] function.
    ///
    /// [`from_parts`]: Self::from_parts
    #[must_use = "losing the pointer will leak memory"]
    pub fn into_parts(self) -> (ptr::NonNull<T>, usize, usize) {
        let this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller, who
        // takes over ownership of the allocation.
        (this.raw.non_null(), this.len, this.raw.capacity())
    }
}

// ---------------------------------------------------------------------------
// Insert & push methods — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Appends an element to the back of the vector.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails.
    pub fn try_push(&mut self, value: T) -> Result<(), TryReserveError> {
        self.try_push_mut_give_back(value)
            .map(|_| ())
            .map_err(|(_returned, e)| e)
    }

    /// Like [`Self::try_push`], but on failure returns the unappended `value`
    /// back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryReserveError)` if growing the buffer fails.
    pub fn try_push_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)> {
        self.try_push_mut_give_back(value).map(|_| ())
    }

    /// Appends an element to the back of the vector and returns a mutable
    /// reference to it.
    ///
    /// This is convenient when the element needs further initialization after
    /// insertion (e.g., setting fields on a struct).
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails.
    pub fn try_push_mut(&mut self, value: T) -> Result<&mut T, TryReserveError> {
        self.try_push_mut_give_back(value)
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_push_mut`], but on failure returns the unappended
    /// `value` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryReserveError)` if growing the buffer fails.
    pub fn try_push_mut_give_back(&mut self, value: T) -> Result<&mut T, (T, TryReserveError)> {
        if self.len == self.capacity() {
            // SAFETY: `len == capacity` holds here.
            let r = unsafe { self.raw.try_grow_one() };
            if let Err(e) = r {
                return Err((value, e));
            }
        }
        let ptr = self.as_mut_ptr();
        // SAFETY: `self.len < capacity` (we grew if needed), so this is in-bounds.
        let dest = unsafe { ptr.add(self.len) };
        // SAFETY: we write to `dest`, which is initialized and within bounds.
        Ok(unsafe {
            dest.write(value);
            self.len += 1;
            &mut *dest
        })
    }

    /// Inserts an element at position `index`, shifting later elements to the
    /// right.
    ///
    /// # Errors
    ///
    /// * [`TryVecInsertError::OutOfBounds`] — `index > len`.
    /// * [`TryVecInsertError::Reserve`] — growing the buffer failed.
    pub fn try_insert(&mut self, index: usize, value: T) -> Result<(), TryVecInsertError> {
        self.try_insert_mut_give_back(index, value)
            .map(|_| ())
            .map_err(|(_returned, e)| e)
    }

    /// Like [`Self::try_insert`], but returns the value back on failure.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryVecInsertError)` on failure.
    pub fn try_insert_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<(), (T, TryVecInsertError)> {
        self.try_insert_mut_give_back(index, value).map(|_| ())
    }

    /// Inserts an element at position `index` and returns a mutable reference
    /// to it.
    ///
    /// # Errors
    ///
    /// * [`TryVecInsertError::OutOfBounds`] — `index > len`.
    /// * [`TryVecInsertError::Reserve`] — growing the buffer failed.
    pub fn try_insert_mut(&mut self, index: usize, value: T) -> Result<&mut T, TryVecInsertError> {
        self.try_insert_mut_give_back(index, value)
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_insert_mut`], but returns the value back on failure.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryVecInsertError)` on failure.
    pub fn try_insert_mut_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<&mut T, (T, TryVecInsertError)> {
        if index > self.len {
            return Err((value, TryVecInsertError::OutOfBounds));
        }
        if index == self.len {
            return self
                .try_push_mut_give_back(value)
                .map_err(|(v, e)| (v, TryVecInsertError::Reserve(e)));
        }
        if self.len == self.capacity() {
            // SAFETY: `len == capacity` holds here.
            let r = unsafe { self.raw.try_grow_one() };
            if let Err(e) = r {
                return Err((value, TryVecInsertError::Reserve(e)));
            }
        }
        let ptr = self.as_mut_ptr();
        // SAFETY: `index < self.len <= capacity`, so both pointers are in-bounds.
        let dest = unsafe { ptr.add(index) };
        let shifted = unsafe { dest.add(1) };
        // Shift the tail `[index..len]` one slot to the right (overlap allowed).
        // `copy` handles overlapping regions by copying in the correct order
        // for the direction of movement.
        // The tail's location is also "dest" so we use that variable directly to save space.
        unsafe {
            ptr::copy(dest, shifted, self.len - index);
        }
        // SAFETY: we write to `dest`, which is initialized and within bounds.
        Ok(unsafe {
            dest.write(value);
            self.len += 1;
            &mut *dest
        })
    }
}

// ---------------------------------------------------------------------------
// Constructors, reconstitution & allocation management — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Creates an empty `Vec` using the given allocator.
    ///
    /// No allocation happens here, so this cannot fail.
    #[must_use]
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            raw: RawVec::new_in(alloc),
            len: 0,
        }
    }

    /// Creates an empty `Vec` with room to hold exactly `capacity` elements.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// initial allocation fails.
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, TryReserveError> {
        let raw = RawVec::try_with_capacity_in(capacity, alloc)?;
        Ok(Self { raw, len: 0 })
    }

    /// Creates a `Vec<T>` directly from a pointer, a length, a capacity, and an
    /// allocator.
    ///
    /// # Safety
    ///
    /// - If T is not a zero-sized type and the capacity is nonzero, ptr must have been allocated using the `alloc`.
    ///   If T is a zero-sized type or the capacity is zero, ptr need only be non-null and aligned.
    /// - T needs to have the same alignment as what ptr was allocated with, if the pointer is required to be allocated.
    ///   (T having a less strict alignment is not sufficient, the alignment really needs to be equal to satisfy the dealloc
    ///   requirement that memory must be allocated and deallocated with the same layout.)
    /// - The size of T times the `capacity` (i.e. the allocated size in bytes), if nonzero, needs to be the same size as
    ///   the pointer was allocated with. (Because similar to alignment, dealloc must be called with the same layout size.)
    /// - `length` needs to be less than or equal to `capacity`.
    /// - The first `length` values must be properly initialized values of type T.
    /// - `capacity` needs to be the capacity that the pointer was allocated with, if the pointer is required to be allocated.
    /// - The allocated size in bytes must be no larger than `isize::MAX`. See the safety documentation of [`pointer::offset`](core::pointer::offset).
    // FIXME: pointer::offset is unlinked
    #[inline]
    pub unsafe fn from_raw_parts_in(ptr: *mut T, length: usize, capacity: usize, alloc: A) -> Self {
        debug_assert!(
            length <= capacity,
            "Vec::from_raw_parts_in requires that length <= capacity"
        );
        // SAFETY: preconditions passed to the caller.
        let raw = unsafe { RawVec::from_raw_parts_in(ptr, capacity, alloc) };
        Self { raw, len: length }
    }

    /// Creates a `Vec<T>` from a non-null pointer, length, capacity, and
    /// allocator.
    ///
    /// # Safety
    ///
    /// See [`Vec::from_raw_parts_in`] for the full list of invariants.
    #[inline]
    pub unsafe fn from_parts_in(
        ptr: ptr::NonNull<T>,
        length: usize,
        capacity: usize,
        alloc: A,
    ) -> Self {
        debug_assert!(
            length <= capacity,
            "Vec::from_parts_in requires that length <= capacity"
        );
        // SAFETY: preconditions passed to the caller.
        let raw = unsafe { RawVec::from_nonnull_in(ptr, capacity, alloc) };
        Self { raw, len: length }
    }

    /// Decomposes a `Vec<T, A>` into its constituent parts: a raw pointer, a
    /// length, a capacity, and the allocator.
    ///
    /// # Safety
    ///
    /// After calling this function, the caller assumes responsibility for the
    /// allocation: it must eventually deallocate `ptr` with `alloc`.
    #[must_use = "losing the pointer will leak memory"]
    pub unsafe fn into_raw_parts_with_alloc(self) -> (*mut T, usize, usize, A) {
        let this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller.
        unsafe {
            (
                this.raw.ptr(),
                this.len,
                this.capacity(),
                ptr::read(this.raw.allocator()),
            )
        }
    }

    /// Decomposes a `Vec<T, A>` into a non-null pointer, length, capacity, and
    /// allocator.
    ///
    /// # Safety
    ///
    /// See [`Self::into_raw_parts_with_alloc`].
    #[must_use = "losing the pointer will leak memory"]
    pub unsafe fn into_parts_with_alloc(self) -> (ptr::NonNull<T>, usize, usize, A) {
        let this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller.
        unsafe {
            (
                this.raw.non_null(),
                this.len,
                this.capacity(),
                ptr::read(this.allocator()),
            )
        }
    }

    /// Ensures the vector has room for at least `additional` more elements.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the reservation fails.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.raw.try_reserve(self.len, additional)
    }

    /// Ensures the vector has room for exactly `len + additional` elements.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the reservation fails.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.raw.try_reserve_exact(self.len, additional)
    }

    /// Shrinks the capacity down to `min_capacity`, keeping at least `len`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the reallocation fails.
    /// The vector is still safe to use regardless whether the operation failed or succeeded.
    pub fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError> {
        let target = cmp::max(self.len, min_capacity);
        if self.capacity() > target {
            self.raw.try_shrink_to_fit(target)
        } else {
            Ok(())
        }
    }

    /// Shrinks the capacity to fit the current length.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the reallocation fails.
    /// The vector is still safe to use regardless whether the operation failed or succeeded.
    pub fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError> {
        self.try_shrink_to(self.len)
    }
}

// ---------------------------------------------------------------------------
// Conversion methods — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Converts this vector into a `Box<[T]>` with exactly `len()` elements and
    /// no excess capacity.
    ///
    /// No elements are cloned: when there is spare capacity the buffer is
    /// shrunk in place, then handed straight to the box. For an empty vector
    /// this returns an empty boxed slice without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the shrink reallocation fails. Unlike the
    /// give-back variant, the vector is consumed either way.
    pub fn try_into_boxed_slice(self) -> Result<Box<[T], A>, TryReserveError> {
        // Delegate to the give-back variant; on failure we discard the vector
        // (it is consumed either way in this signature).
        self.try_into_boxed_slice_give_back()
            .map_err(|(_returned, e)| e)
    }

    /// Like [`Self::try_into_boxed_slice`], but returns the vector back on
    /// failure so no data is lost.
    ///
    /// # Errors
    ///
    /// Returns `(Vec<T, A>, TryReserveError)` if the shrink fails.
    pub fn try_into_boxed_slice_give_back(
        mut self,
    ) -> Result<Box<[T], A>, (Self, TryReserveError)> {
        if let Err(e) = self.try_shrink_to_fit() {
            return Err((self, e));
        }
        // Prevent the outer `Drop` from running while we dismantle the fields.
        let this = ManuallyDrop::new(self);
        // SAFETY: after shrinking, `raw` holds exactly `self.len` initialized
        // elements; `into_box` wraps them without dropping, and `assume_init`
        // reinterprets the buffer as `[T]`. Mirrors std's `into_boxed_slice`.
        unsafe {
            let buf = ptr::read(&this.raw);
            let len = this.len;
            Ok(buf.into_box(len).assume_init())
        }
    }

    /// Converts this vector into a `Box<[T; N]>`, consuming the vector.
    ///
    /// The vector must contain exactly `N` elements; otherwise
    /// [`TryVecIntoArrayError::LengthMismatch`] is returned. If the vector has
    /// excess capacity beyond `N`, a shrink-to-fit reallocation is attempted
    /// first; failure there yields [`TryVecIntoArrayError::Shrink`].
    ///
    /// # Errors
    ///
    /// * [`TryVecIntoArrayError::LengthMismatch`] — `self.len() != N`.
    /// * [`TryVecIntoArrayError::Shrink`] — the internal shrink failed (OOM).
    pub fn try_into_array<const N: usize>(self) -> Result<Box<[T; N], A>, TryVecIntoArrayError> {
        // Delegate to the give-back variant; on failure we discard the vector
        // (it is consumed either way in this signature).
        self.try_into_array_give_back::<N>()
            .map_err(|(_returned, e)| e)
    }

    /// Like [`Self::try_into_array`], but returns the vector back on failure
    /// so no data is lost.
    ///
    /// # Errors
    ///
    /// Returns `(Vec<T, A>, TryVecIntoArrayError)` when either the length
    /// mismatch or the shrink fails.
    pub fn try_into_array_give_back<const N: usize>(
        mut self,
    ) -> Result<Box<[T; N], A>, (Self, TryVecIntoArrayError)> {
        if self.len != N {
            let actual = self.len;
            return Err((
                self,
                TryVecIntoArrayError::LengthMismatch {
                    expected: N,
                    actual,
                },
            ));
        }
        if let Err(e) = self.try_shrink_to_fit() {
            return Err((self, TryVecIntoArrayError::Shrink(e)));
        }
        // Prevent the outer `Drop` from running while we dismantle the fields.
        let this = ManuallyDrop::new(self);
        // SAFETY: identical to `try_into_array` — the buffer holds exactly `N`
        // initialized elements allocated by `A`; casting to `*mut [T; N]` is
        // layout-compatible.
        unsafe {
            let raw_ptr = this.raw.ptr();
            let alloc = ptr::read(this.raw.allocator());
            let array_ptr = raw_ptr.cast::<[T; N]>();
            Ok(Box::from_raw_in(array_ptr, alloc))
        }
    }
}

// ---------------------------------------------------------------------------
// Query methods — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Returns a reference to the allocator backing this vector.
    #[inline]
    pub const fn allocator(&self) -> &A {
        self.raw.allocator()
    }

    /// The maximum number of elements the vector can hold without reallocating.
    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.raw.capacity()
    }

    /// Number of elements in the vector, also referred to as its 'length'.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the vector's buffer.
    ///
    /// The caller must not use this pointer after the vector is mutated or
    /// dropped.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.raw.ptr()
    }

    /// Returns a mutable raw pointer to the vector's buffer.
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.raw.ptr()
    }

    /// Views the vector as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `self.len <= self.capacity()` is an invariant, and the first
        // `self.len` slots are always initialized.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Views the vector as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: see `as_slice`.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }
}

// ---------------------------------------------------------------------------
// Mutation methods — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Removes all elements from the vector, dropping them in place.
    ///
    /// Removal never allocates, so this cannot fail.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Removes the last element from the vector and returns it, or `None` if
    /// the vector is empty.
    ///
    /// Removal never allocates, so this cannot fail.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            // SAFETY: we extract and move the element at self.len == new_len == old_len - 1
            Some(unsafe { ptr::read(self.as_mut_ptr().add(self.len)) })
        }
    }

    /// Shortens the vector, keeping only the first `new_len` elements and
    /// dropping the rest. If `new_len` is greater than or equal to the current
    /// length, nothing happens.
    ///
    /// Truncation never allocates, so this cannot fail.
    pub fn truncate(&mut self, new_len: usize) {
        // Truncating to a length >= current length is a no-op (matches std).
        if new_len >= self.len {
            return;
        }
        // SAFETY:
        // * The slice we hand to `drop_in_place` covers `[new_len..self.len)`,
        //   which is within the initialized region, so it is valid.
        // * We shrink `self.len` *before* calling `drop_in_place`, so if any
        //   destructor in the dropped tail panics, the length already excludes
        //   those elements and unwinding cannot double-free them. (A second
        //   panic during unwind aborts, per Rust's rules.)
        unsafe {
            let remaining = self.len - new_len;
            let tail = slice::from_raw_parts_mut(self.as_mut_ptr().add(new_len), remaining);
            self.len = new_len;
            ptr::drop_in_place(tail);
        }
    }

    /// Sets the length of the vector, possibly uninitialized.
    ///
    /// This is an unsafe operation: the caller must guarantee that `new_len`
    /// does not exceed `capacity()` and that all elements in `[0..new_len)`
    /// are valid (initialized) values of type `T`.
    ///
    /// # Safety
    ///
    /// * `new_len <= capacity()`
    /// * All slots in `[0..new_len)` must contain valid `T` values.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!(
            new_len <= self.capacity(),
            "Vec::set_len requires that new_len <= capacity()"
        );
        self.len = new_len;
    }

    /// Removes an element from the vector and returns it, replacing it with
    /// the last element. Does not preserve ordering.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecRemoveError`] if `index >= len`.
    pub fn try_swap_remove(&mut self, index: usize) -> Result<T, TryVecRemoveError> {
        let len = self.len;
        if index >= len {
            return Err(TryVecRemoveError { index, len });
        }
        self.len -= 1;
        // SAFETY: bounds checked above.
        Ok(unsafe {
            let value = ptr::read(self.as_mut_ptr().add(index));
            if index != len - 1 {
                ptr::copy_nonoverlapping(
                    self.as_mut_ptr().add(len - 1),
                    self.as_mut_ptr().add(index),
                    1,
                );
            }
            value
        })
    }

    /// Removes and returns the element at position `index`, shifting later
    /// elements to the left.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecRemoveError`] if `index >= len`.
    pub fn try_remove(&mut self, index: usize) -> Result<T, TryVecRemoveError> {
        let len = self.len;
        if index >= len {
            return Err(TryVecRemoveError { index, len });
        }
        self.len -= 1;
        let ptr = self.as_mut_ptr();
        // SAFETY: bounds checked above.
        let removed = unsafe { ptr::read(ptr.add(index)) };
        // SAFETY: We are removing elements from `index + 1..len` (`len` noninclusive).
        unsafe {
            ptr::copy(ptr.add(index + 1), ptr.add(index), len - index - 1);
        }
        Ok(removed)
    }

    /// Retains only the elements selected by `predicate`, dropping the rest in
    /// place.
    ///
    /// The predicate is called once per element; it receives a mutable
    /// reference so it can inspect (but not replace) each element. This runs
    /// in a single pass with a bulk backshift of survivors, so it is linear in
    /// the number of elements rather than quadratic in deletions.
    ///
    /// Retention never grows the buffer, so this cannot fail.
    pub fn retain<F>(&mut self, predicate: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        self.retain_mut(predicate);
    }

    /// Retains only the elements selected by `predicate`, using a two-pass
    /// strategy that avoids repeated shifting.
    ///
    /// The predicate receives a mutable reference to each element. Elements
    /// for which it returns `false` are dropped in place.
    // TODO: need to validate from here
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        let original_len = self.len();

        if original_len == 0 {
            // Empty case: explicit return allows better optimization, vs letting compiler infer it
            return;
        }

        // Vec: [Kept, Kept, Hole, Hole, Hole, Hole, Unchecked, Unchecked]
        //      |            ^- write                ^- read             |
        //      |<-              original_len                          ->|
        // Kept: Elements which predicate returns true on.
        // Hole: Moved or dropped element slot.
        // Unchecked: Unchecked valid elements.
        //
        // This drop guard will be invoked when predicate or `drop` of element panicked.
        // It shifts unchecked elements to cover holes and `set_len` to the correct length.
        // In cases when predicate and `drop` never panick, it will be optimized out.
        struct PanicGuard<'a, T, A: Allocator> {
            v: &'a mut Vec<T, A>,
            read: usize,
            write: usize,
            original_len: usize,
        }

        impl<T, A: Allocator> Drop for PanicGuard<'_, T, A> {
            #[cold]
            fn drop(&mut self) {
                let remaining = self.original_len - self.read;
                // SAFETY: Trailing unchecked items must be valid since we never touch them.
                unsafe {
                    ptr::copy(
                        self.v.as_ptr().add(self.read),
                        self.v.as_mut_ptr().add(self.write),
                        remaining,
                    );
                }
                // SAFETY: After filling holes, all items are in contiguous memory.
                unsafe {
                    self.v.set_len(self.write + remaining);
                }
            }
        }

        let mut read = 0;
        loop {
            // SAFETY: read < original_len
            let cur = unsafe { self.get_unchecked_mut(read) };
            if !f(cur) {
                break;
            }
            read += 1;
            if read == original_len {
                // All elements are kept, return early.
                return;
            }
        }

        // Critical section starts here and at least one element is going to be removed.
        // Advance `g.read` early to avoid double drop if `drop_in_place` panicked.
        let mut g = PanicGuard {
            v: self,
            read: read + 1,
            write: read,
            original_len,
        };
        // SAFETY: previous `read` is always less than original_len.
        unsafe { ptr::drop_in_place(&mut *g.v.as_mut_ptr().add(read)) };

        while g.read < g.original_len {
            // SAFETY: `read` is always less than original_len.
            let cur = unsafe { &mut *g.v.as_mut_ptr().add(g.read) };
            if !f(cur) {
                // Advance `read` early to avoid double drop if `drop_in_place` panicked.
                g.read += 1;
                // SAFETY: We never touch this element again after dropped.
                unsafe { ptr::drop_in_place(cur) };
            } else {
                // SAFETY: `read` > `write`, so the slots don't overlap.
                // We use copy for move, and never touch the source element again.
                unsafe {
                    let hole = g.v.as_mut_ptr().add(g.write);
                    ptr::copy_nonoverlapping(cur, hole, 1);
                }
                g.write += 1;
                g.read += 1;
            }
        }

        // We are leaving the critical section and no panic happened,
        // Commit the length change and forget the guard.
        // SAFETY: `write` is always less than or equal to original_len.
        unsafe { g.v.set_len(g.write) };
        olive_core::mem::forget(g);
    }

    /// Removes all but the first of consecutive elements that resolve to the
    /// same key.
    ///
    /// If the vector is sorted, this removes all duplicates.
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    /// Removes all but the first of consecutive elements satisfying a given
    /// equality relation.
    ///
    /// The `same_bucket` function is passed references to two elements; if
    /// `same_bucket(a, b)` returns `true`, `a` is removed. Note the arguments
    /// are in reverse order relative to their position in the vector.
    // TODO: need to validate this function
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        let len = self.len;
        if len <= 1 {
            return;
        }

        // Drop guard: if `same_bucket` panics mid-way, some slots have already
        // been dropped (leaving holes) while `self.len` still reflects the
        // original length. Without restoring contiguity, the Vec's `Drop`
        // would either double-free the dropped slots or leak the tail. The
        // guard backfills any remaining gap and corrects the length on unwind.
        struct DedupGuard<'a, T, A: Allocator> {
            v: &'a mut Vec<T, A>,
            write: usize,
            read: usize,
            original_len: usize,
            active: bool,
        }

        impl<T, A: Allocator> Drop for DedupGuard<'_, T, A> {
            fn drop(&mut self) {
                if !self.active || self.read == self.original_len {
                    return;
                }
                // Backfill `[write..read)` from `[read..original_len)` to close
                // the hole left by the duplicates we already dropped.
                unsafe {
                    ptr::copy(
                        self.v.as_ptr().add(self.read),
                        self.v.as_mut_ptr().add(self.write),
                        self.original_len - self.read,
                    );
                }
                // SAFETY: after backfilling, items `[0..write)` are contiguous
                // and valid; the rest are uninitialized.
                unsafe {
                    self.v.set_len(self.write);
                }
            }
        }

        let start = self.as_mut_ptr();
        // Find the first duplicate.
        let mut first_duplicate_idx: usize = 1;
        while first_duplicate_idx != len {
            let found = unsafe {
                // SAFETY: first_duplicate_idx is in [1, len).
                let prev = start.add(first_duplicate_idx - 1);
                let current = start.add(first_duplicate_idx);
                same_bucket(&mut *current, &mut *prev)
            };
            if found {
                break;
            }
            first_duplicate_idx += 1;
        }
        if first_duplicate_idx == len {
            return;
        }

        // Gap-filling loop.
        let mut read = first_duplicate_idx + 1;
        let mut write = first_duplicate_idx;

        // Activate the guard before mutating anything.
        let mut guard = DedupGuard {
            v: self,
            write,
            read,
            original_len: len,
            active: true,
        };

        // SAFETY: first_duplicate_idx < len, so this slot is initialized.
        unsafe { ptr::drop_in_place(start.add(first_duplicate_idx)) };

        while read < len {
            // SAFETY: read < len, write >= 1, so all derived pointers are valid.
            let (read_ptr, prev_ptr) = unsafe { (start.add(read), start.add(write - 1)) };
            let found = unsafe { same_bucket(&mut *read_ptr, &mut *prev_ptr) };
            if found {
                read += 1;
                guard.read = read;
                // SAFETY: read_ptr points to an initialized element.
                unsafe { ptr::drop_in_place(read_ptr) };
            } else {
                // SAFETY: write < read, so write_ptr is a valid distinct slot.
                let write_ptr = unsafe { start.add(write) };
                // SAFETY: read_ptr != write_ptr (guaranteed by initial gap).
                unsafe { ptr::copy_nonoverlapping(read_ptr, write_ptr, 1) };
                write += 1;
                read += 1;
                guard.write = write;
                guard.read = read;
            }
        }
        // Loop completed without panic; deactivate the guard and finalize via
        // the guard's own reference (we cannot borrow `self` again while the
        // guard is alive).
        guard.active = false;
        // SAFETY: all items before `write` are now valid and contiguous.
        unsafe { guard.v.set_len(write) };
    }

    /// Pushes an element onto the end of the vector without attempting to grow
    /// the buffer. Succeeds only if there is already spare capacity.
    ///
    /// # Errors
    ///
    /// Returns [`TryPushWithinCapacityError`] if `len == capacity`.
    // FIXME: need give_back variant. Replace inline pushing with unsafe force_push()
    pub fn try_push_within_capacity(&mut self, value: T) -> Result<(), TryPushWithinCapacityError> {
        // Prevent a degenerate scenario where `length` is exceeding `capacity`
        if self.len >= self.capacity() {
            return Err(TryPushWithinCapacityError { len: self.len });
        }
        // SAFETY: capacity confirmed, slot is allocated.
        unsafe {
            self.raw.ptr().add(self.len).write(value);
        }
        self.len += 1;
        Ok(())
    }

    /// Pops the last element off the vector if it satisfies the predicate,
    /// returning `None` otherwise.
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        // SAFETY: len > 0, so the last slot is initialized.
        let last = unsafe { &mut *self.as_mut_ptr().add(self.len - 1) };
        if predicate(last) { self.pop() } else { None }
    }

    /// Extends the vector with each element of `other`, cloned via
    /// [`TryClone`].
    ///
    /// If a clone fails mid-way, the vector is truncated back to its length at
    /// the start of the call so no partially-appended elements remain.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    // FIXME: need a panic-aware rollback guard, replace inline pushing with unsafe force_push()
    pub fn try_extend_from_slice_with_rollback(
        &mut self,
        other: &[T],
    ) -> Result<(), TryVecWithCloneError>
    where
        T: TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        self.raw
            .try_reserve(self.len, other.len())
            .map_err(TryVecWithCloneError::Reserve)?;
        let len_before = self.len;
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    unsafe {
                        self.raw.ptr().add(self.len).write(cloned);
                    }
                    self.len += 1;
                }
                Err(e) => {
                    self.truncate(len_before);
                    return Err(TryVecWithCloneError::Clone(e));
                }
            }
        }
        Ok(())
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// Elements are moved, not cloned. On success `other` is drained; on failure
    /// `other` is left untouched.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving space for `other`'s elements
    /// fails.
    // FIXME: Replace inner try_reserve() with public try_reserve
    pub fn try_append(&mut self, other: &mut Self) -> Result<(), TryReserveError> {
        let extra = other.len;
        if extra == 0 {
            return Ok(());
        }
        self.raw.try_reserve(self.len, extra)?;
        let src = other.as_mut_ptr();
        let dst = unsafe { self.as_mut_ptr().add(self.len) };
        // SAFETY: `self` and `other` are distinct vectors; their buffers never overlap.
        unsafe {
            ptr::copy_nonoverlapping(src, dst, extra);
        }
        self.len += extra;
        other.len = 0;
        Ok(())
    }

    /// Resizes the vector so its length becomes `new_len`.
    ///
    /// If `new_len` is greater than the current length, the vector is extended
    /// by cloning `value` via [`TryClone`]. If smaller, it is truncated.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    // FIXME: need panic-aware rollback behavior. replace inline pushing with unsafe force_push()
    pub fn try_resize(&mut self, value: &T, new_len: usize) -> Result<(), TryVecWithCloneError>
    where
        T: TryClone,
    {
        let current = self.len;
        if new_len <= current {
            self.truncate(new_len);
            return Ok(());
        }
        let extra = new_len - current;
        self.raw
            .try_reserve(self.len, extra)
            .map_err(TryVecWithCloneError::Reserve)?;
        for _ in 0..extra {
            match value.try_clone() {
                Ok(cloned) => {
                    // Capacity was reserved above, so this cannot fail.
                    unsafe {
                        self.raw.ptr().add(self.len).write(cloned);
                    }
                    self.len += 1;
                }
                Err(e) => {
                    self.truncate(current);
                    return Err(TryVecWithCloneError::Clone(e));
                }
            }
        }
        Ok(())
    }

    /// Resizes the vector so its length becomes `new_len`, producing new
    /// elements with the closure `f`.
    ///
    /// The closure is invoked only after capacity is secured.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving space fails.
    // FIXME: make the callback fallible and return something like `TryResizeWithError`, and add panic-aware rollback as well
    // also replace inline pushing with unsafe force_push(). Replace inner try_reserve() with public try_reserve
    pub fn try_resize_with<F>(&mut self, new_len: usize, mut f: F) -> Result<(), TryReserveError>
    where
        F: FnMut() -> T,
    {
        let current = self.len;
        if new_len <= current {
            self.truncate(new_len);
            return Ok(());
        }
        let extra = new_len - current;
        self.raw.try_reserve(self.len, extra)?;
        for _ in 0..extra {
            // Capacity was reserved above, so this cannot fail.
            unsafe {
                self.raw.ptr().add(self.len).write(f());
            }
            self.len += 1;
        }
        Ok(())
    }

    /// Swaps two elements in the vector by their indices.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    // TODO: replace with try_swap that *doesn't* panic. Replace with try_swap.
    #[track_caller]
    pub fn swap(&mut self, a: usize, b: usize) {
        let sl = self.as_mut_slice();
        sl.swap(a, b);
    }

    /// Reverses the order of the elements in place.
    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse();
    }

    /// Sorts the vector's elements in place using the default ordering.
    ///
    /// # Requires
    ///
    /// `T: Ord`.
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.as_mut_slice().sort();
    }
}

// ---------------------------------------------------------------------------
// Fallible constructors — explicit allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Creates a `Vec<T>` containing `value` cloned `count` times.
    ///
    /// Equivalent to `vec![value; count]` but fully fallible.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    pub fn try_from_elem_in(value: &T, count: usize, alloc: A) -> Result<Self, TryVecWithCloneError>
    where
        T: TryClone,
    {
        let mut vec = Self::new_in(alloc);
        if count > 0 {
            vec.raw
                .try_reserve(0, count)
                .map_err(TryVecWithCloneError::Reserve)?;
        }
        for _ in 0..count {
            match value.try_clone() {
                Ok(cloned) => {
                    // Capacity was reserved above, so this cannot fail.
                    unsafe {
                        vec.raw.ptr().add(vec.len).write(cloned);
                    }
                    vec.len += 1;
                }
                Err(e) => return Err(TryVecWithCloneError::Clone(e)),
            }
        }
        Ok(vec)
    }

    /// Like [`Self::try_from_elem_in`], but takes ownership of `value` and
    /// returns it on failure so the caller is not left empty-handed.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryVecWithCloneError)` on failure.
    pub fn try_from_elem_give_back_in(
        value: T,
        count: usize,
        alloc: A,
    ) -> Result<Self, (T, TryVecWithCloneError)>
    where
        T: TryClone,
    {
        // Wrap in `ManuallyDrop` so that on failure we can hand the original
        // value back without dropping it twice (once here, once by the caller).
        let mut value = ManuallyDrop::new(value);
        match Self::try_from_elem_in(&value, count, alloc) {
            Ok(v) => Ok(v),
            Err(e) => {
                // SAFETY: `value` was only ever lent by reference; it is still
                // fully initialized and owned by us. Taking it out transfers
                // ownership to the error tuple.
                let recovered = unsafe { ManuallyDrop::take(&mut value) };
                Err((recovered, e))
            }
        }
    }

    /// Creates a `Vec<T>` from a slice by cloning each element via
    /// [`TryClone`].
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    pub fn try_from_slice_in(slice: &[T], alloc: A) -> Result<Self, TryVecWithCloneError>
    where
        T: TryClone,
    {
        let mut vec = Self::new_in(alloc);
        if !slice.is_empty() {
            vec.raw
                .try_reserve(0, slice.len())
                .map_err(TryVecWithCloneError::Reserve)?;
        }
        for item in slice {
            match item.try_clone() {
                Ok(cloned) => {
                    // Capacity was reserved above, so this cannot fail.
                    unsafe {
                        vec.raw.ptr().add(vec.len).write(cloned);
                    }
                    vec.len += 1;
                }
                Err(e) => return Err(TryVecWithCloneError::Clone(e)),
            }
        }
        Ok(vec)
    }

    /// Fallibly collects an iterator into a `Vec<T>`, using the size hint to
    /// pre-allocate when possible.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if a reservation fails.
    // FIXME: move implementation to TryFromIterator.
    pub fn try_collect_in<I: IntoIterator<Item = T>>(
        iter: I,
        alloc: A,
    ) -> Result<Self, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut vec = Self::new_in(alloc);
        if capacity > 0 {
            vec.raw.try_reserve(0, capacity)?;
        }
        for item in iter {
            // The iterator may yield more elements than its hint promised.
            if vec.len == vec.capacity() {
                // SAFETY: `len == capacity` holds here.
                unsafe { vec.raw.try_grow_one()? };
            }
            unsafe {
                vec.raw.ptr().add(vec.len).write(item);
            }
            vec.len += 1;
        }
        Ok(vec)
    }
}

// Additional convenience constructors for the default (`Global`) allocator.
impl<T> Vec<T, Global> {
    /// Like [`Self::try_from_elem`], but takes ownership of `value` and
    /// returns it on failure.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryVecWithCloneError)` on failure.
    pub fn try_from_elem_give_back(
        value: T,
        count: usize,
    ) -> Result<Self, (T, TryVecWithCloneError)>
    where
        T: TryClone,
    {
        Self::try_from_elem_give_back_in(value, count, Global)
    }

    /// Creates a `Vec<T>` from a slice by cloning each element via
    /// [`TryClone`].
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    pub fn try_from_slice(slice: &[T]) -> Result<Self, TryVecWithCloneError>
    where
        T: TryClone,
    {
        Self::try_from_slice_in(slice, Global)
    }

    /// Fallibly collects an iterator into a `Vec<T>`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if a reservation fails.
    // FIXME: std does not have it, may retire
    pub fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, TryReserveError> {
        Self::try_collect_in(iter, Global)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<T> Default for Vec<T, Global> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, A: Allocator> Deref for Vec<T, A> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> DerefMut for Vec<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, A: Allocator> Index<usize> for Vec<T, A> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        // Bounds-checked by the slice machinery.
        &self.as_slice()[index]
    }
}

impl<T, A: Allocator> IndexMut<usize> for Vec<T, A> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.as_mut_slice()[index]
    }
}

impl<T, A: Allocator> Index<core::ops::RangeFull> for Vec<T, A> {
    type Output = [T];

    #[inline]
    fn index(&self, _: core::ops::RangeFull) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> IndexMut<core::ops::RangeFull> for Vec<T, A> {
    #[inline]
    fn index_mut(&mut self, _: core::ops::RangeFull) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, A: Allocator> fmt::Debug for Vec<T, A>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, A: Allocator> Drop for Vec<T, A> {
    fn drop(&mut self) {
        // SAFETY: the first `self.len` slots are initialized; dropping them in
        // place runs each element's destructor exactly once. The buffer itself
        // is freed by `RawVec`'s `Drop` immediately afterward 
        // (even when drop_in_place panics).
        unsafe {
            let slice = ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len);
            ptr::drop_in_place(slice);
        }
    }
}

impl<T, A: Allocator> TryExtend<T> for Vec<T, A> {
    type Error = TryReserveError;

    // FIXME: need size_hint and call reserve with public try_reserve method.
    // force_push() within size_hint range after reserving capacity,
    // try_push for the rest after size_hint exhausted but there is still an item in the iterator
    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = T>,
    {
        let (head, mut inner) = source.safe_into_iter();

        // Peek ahead: if there is at least one element, reserve up-front.
        let first = head.or_else(|| inner.next());
        let Some(first) = first else {
            return Ok(());
        };

        if let Err(e) = self.raw.try_reserve(self.len, 1) {
            return Err((Resume::new(first, inner), e));
        }

        // Push the first element (capacity guaranteed).
        unsafe {
            self.raw.ptr().add(self.len).write(first);
        }
        self.len += 1;

        // Push the remainder, growing as needed.
        while let Some(next) = inner.next() {
            if self.len == self.capacity() {
                // SAFETY: `len == capacity` holds here.
                if let Err(e) = unsafe { self.raw.try_grow_one() } {
                    return Err((Resume::new(next, inner), e));
                }
            }
            unsafe {
                self.raw.ptr().add(self.len).write(next);
            }
            self.len += 1;
        }
        Ok(())
    }
}

impl<'s, T, A: Allocator> TryExtendFromSlice<'s, T> for Vec<T, A>
where
    T: TryClone,
{
    type Error = TryCloneError;

    // FIXME: replace with force_push(), use the Vec::try_reserve public method
    fn try_extend_from_slice(&mut self, other: &'s [T]) -> Result<(), (&'s [T], Self::Error)> {
        if other.is_empty() {
            return Ok(());
        }
        self.raw
            .try_reserve(self.len, other.len())
            .map_err(|e| (other, TryCloneError::Reserve(e)))?;
        let mut i = 0usize;
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    unsafe {
                        self.raw.ptr().add(self.len).write(cloned);
                    }
                    self.len += 1;
                    i += 1;
                }
                Err(e) => return Err((&other[i..], e)),
            }
        }
        Ok(())
    }
}

impl<T: TryClone, A: Allocator + Clone> TryClone for Vec<T, A> {
    // FIXME: replace try_push with force_push for performance, use the public Vec::try_reserve.
    // Allocator needs try_clone per convention as well
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = Self::new_in(self.raw.allocator().clone());
        if !self.is_empty() {
            out.raw
                .try_reserve(0, self.len)
                .map_err(TryCloneError::Reserve)?;
        }
        for elem in self.iter() {
            match elem.try_clone() {
                Ok(cloned) => out.try_push(cloned).map_err(TryCloneError::Reserve)?,
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }
}

// FIXME: move try_collect() here, generalize this block to support any allocator 
impl<T> TryFromIterator<T> for Vec<T, Global> {
    type Error = TryReserveError;

    fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Self::Error> {
        Self::try_collect(iter)
    }
}

mod into_iter;
pub use into_iter::IntoIter;

impl<T, A: Allocator> IntoIterator for Vec<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Take ownership of the raw parts so the iterator owns the allocation
        // directly rather than nesting a whole `Vec`. The `ManuallyDrop` inside
        // `into_parts_with_alloc` prevents the original `Vec` from freeing the
        // buffer; the iterator takes over responsibility for it.
        // SAFETY: `self` is a well-formed `Vec`; its pointer, length, capacity,
        // and allocator satisfy all the invariants `new_from_parts` requires.
        let (ptr, len, cap, alloc) = unsafe { self.into_parts_with_alloc() };
        // SAFETY: as above — the parts come straight off a valid `Vec`.
        unsafe { IntoIter::new_from_parts(ptr, len, cap, alloc) }
    }
}

// Borrowed iterators delegate to the slice methods. 
// FIXME: need to use IntoIterator block for reference and mutable reference
impl<T, A: Allocator> Vec<T, A> {
    /// Returns an iterator over the vector's elements.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns an iterator over mutable references to the vector's elements.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}

#[cfg(test)]
mod tests;
