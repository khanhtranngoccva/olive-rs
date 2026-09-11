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
//! …) without ever panicking or aborting.

use core::borrow::Borrow;
use core::borrow::BorrowMut;
// This module performs a great deal of index arithmetic on `len`/`capacity`, so
// `clippy::arithmetic_side_effects` (denied crate-wide on non-test builds) is
// suppressed *per site* rather than at the module level. Each allow carries a
// `reason = "asserted …"` documenting the invariant that makes the operation
// overflow-free; a blanket `#![allow]` would hide real bugs, so new arithmetic
// must be annotated individually to compile.
use core::cmp;
use core::fmt;
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::ptr::{self, NonNull};
use core::slice;

use crate::alloc::{Allocator, Global};
use crate::boxed::Box;
use crate::raw_vec::RawVec;
use olive_core::alloc::AllocatorTryClone;
use olive_core::alloc_errors::TryReserveError;
use olive_core::recovery::{ResumableSource, Resume};
use olive_core::slice::{TrySliceRangeError, try_range};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};
use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
use olive_core::try_traits::try_from_iterator::TryFromIterator;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned by fallible vector operations that may both reserve capacity
/// and clone elements.
///
/// Covers `try_from_elem`, `try_from_slice`, `try_resize`,
/// `try_extend_from_slice_with_rollback`, and so on — any operation whose failure
/// modes are limited to a capacity reservation ([`TryReserveError`]) or an element
/// clone failure ([`TryCloneError`]).
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
            "index {} is out of bounds, vector length is {} (index must be smaller than length)",
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
        write!(f, "no spare capacity: vec is full at length {}", self.len)
    }
}

impl core::error::Error for TryPushWithinCapacityError {}

/// Error returned by [`Vec::try_swap`] when an index is out of bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrySwapError {
    /// The offending index (whichever was checked first).
    pub index: usize,
    /// The vector's length at the time of the call.
    pub len: usize,
}

impl fmt::Display for TrySwapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "index {} is out of bounds, vector length is {} (indices must be smaller than length)",
            self.index, self.len
        )
    }
}

impl core::error::Error for TrySwapError {}

/// Error returned by [`Vec::try_split_off`] when the split index is out of
/// bounds or the allocation for the right-hand half fails.
#[derive(Clone, PartialEq, Eq)]
pub enum TryVecSplitOffError {
    /// The requested split offset exceeded the vector's length.
    OutOfBounds {
        /// The requested split offset.
        index: usize,
        /// The vector's current length at the time of the call.
        len: usize,
    },
    /// A capacity reservation for the new vector failed (overflow or OOM).
    Reserve(TryReserveError),
}

impl fmt::Debug for TryVecSplitOffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { index, len } => f
                .debug_struct("TryVecSplitOffError::OutOfBounds")
                .field("index", index)
                .field("len", len)
                .finish(),
            Self::Reserve(e) => f
                .debug_tuple("TryVecSplitOffError::Reserve")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryVecSplitOffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { index, len } => write!(
                f,
                "cannot split off at index {index}: out of bounds (vector length is {len})"
            ),
            Self::Reserve(e) => write!(f, "split_off allocation failed: {e}"),
        }
    }
}

impl core::error::Error for TryVecSplitOffError {}

/// Error returned by [`Vec::try_extend_from_within`] when the range is invalid,
/// resolving overflows, or the operation fails during reservation/cloning.
#[derive(Clone, PartialEq, Eq)]
pub enum TryVecExtendFromWithinError {
    /// The resolved range exceeded the vector's length or was reversed.
    InvalidRange {
        /// The computed start of the range.
        start: usize,
        /// The computed end of the range (non-inclusive).
        end: usize,
        /// The vector's current length at the time of the call.
        len: usize,
    },
    /// Resolving the range resulted in an integer overflow.
    RangeOverflow,
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
    /// Cloning an element into the extended region failed.
    Clone(TryCloneError),
}

impl fmt::Debug for TryVecExtendFromWithinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { start, end, len } => f
                .debug_struct("TryVecExtendFromWithinError::InvalidRange")
                .field("start", start)
                .field("end", end)
                .field("len", len)
                .finish(),
            Self::RangeOverflow => f
                .debug_tuple("TryVecExtendFromWithinError::RangeOverflow")
                .finish(),
            Self::Reserve(e) => f
                .debug_tuple("TryVecExtendFromWithinError::Reserve")
                .field(e)
                .finish(),
            Self::Clone(e) => f
                .debug_tuple("TryVecExtendFromWithinError::Clone")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryVecExtendFromWithinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { start, end, len } => write!(
                f,
                "invalid range {start}..{end} for extend_from_within (vector length is {len})"
            ),
            Self::RangeOverflow => write!(f, "range bounds overflowed"),
            Self::Reserve(e) => write!(f, "extend_from_within allocation failed: {e}"),
            Self::Clone(e) => write!(f, "element clone failed during extend_from_within: {e}"),
        }
    }
}

impl core::error::Error for TryVecExtendFromWithinError {}

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

/// A growable contiguous list of items backed by the heap.
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
            vec.try_reserve_exact(n)
                .map_err(TryVecWithClosureError::Reserve)?;
        }
        for _ in 0..n {
            match f() {
                Ok(item) => {
                    // SAFETY: Capacity was reserved above, so this cannot fail.
                    unsafe { vec.force_push(item) };
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
    /// - The allocated size in bytes must be no larger than `isize::MAX`. See the safety documentation of `ptr.offset()`.
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
    pub unsafe fn from_parts(ptr: NonNull<T>, length: usize, capacity: usize) -> Self {
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
        // Mirrors std's `Vec::into_raw_parts`: wrap in `ManuallyDrop` so that a
        // panic during unwinding cannot cause the implicit drop at scope end to
        // free the buffer after we have already handed ownership to the caller.
        let mut this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller, who
        // takes over ownership of the allocation.
        // We still want to extract the allocator for consistency even if it is a ZST.
        let _alloc = unsafe { ptr::read(&this.allocator()) };
        (this.as_mut_ptr(), this.len(), this.capacity())
    }

    /// Decomposes a `Vec<T>` into its constituent parts: a non-null pointer, a
    /// length, and a capacity.
    ///
    /// This is the [`NonNull`] counterpart of [`Self::into_raw_parts`].
    ///
    /// After calling this function, the caller is responsible for the memory
    /// previously managed by the `Vec`. Most often, one does this by converting
    /// the raw pointer, length, and capacity back into a `Vec` with the
    /// [`from_parts`] function.
    ///
    /// [`from_parts`]: Self::from_parts
    #[must_use = "losing the pointer will leak memory"]
    pub fn into_parts(self) -> (NonNull<T>, usize, usize) {
        // Mirrors std's `Vec::into_parts`: wrap in `ManuallyDrop` so that a
        // panic during unwinding cannot cause the implicit drop at scope end to
        // free the buffer after we have already handed ownership to the caller.
        let this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller, who
        // takes over ownership of the allocation.
        // We still want to extract the allocator for consistency even if it is a ZST.
        let _alloc = unsafe { ptr::read(&this.allocator()) };
        (this.raw.non_null(), this.len(), this.capacity())
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
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.len < self.capacity, self.capacity <= usize::MAX"
            )]
            {
                self.len += 1;
            }
            &mut *dest
        })
    }

    /// Appends `value` to the back of the vector without checking or growing
    /// capacity. This is the fast path used internally by bulk operations that
    /// have already secured capacity up front via [`Self::try_reserve`].
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `self.len < self.capacity()`, i.e. there
    /// is at least one spare slot in the buffer. Writing past the end of the
    /// allocation is undefined behavior.
    #[inline]
    pub(crate) unsafe fn force_push(&mut self, value: T) {
        debug_assert!(
            self.len < self.capacity(),
            "force_push requires spare capacity"
        );
        // SAFETY: caller guarantees a spare slot, so this is in-bounds.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted self.len < self.capacity, self.capacity <= usize::MAX"
        )]
        unsafe {
            self.raw.ptr().add(self.len).write(value);
            self.len += 1;
        }
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
        // SAFETY: `index < self.len < capacity`, so both pointers are in-bounds.
        let dest = unsafe { ptr.add(index) };
        let shifted = unsafe { dest.add(1) };
        // Shift the tail `[index..len]` one slot to the right (overlap allowed).
        // `copy` handles overlapping regions by copying in the correct order
        // for the direction of movement.
        // The tail's location is also "dest" so we use that variable directly to save space.
        #[allow(clippy::arithmetic_side_effects, reason = "asserted index < self.len")]
        unsafe {
            ptr::copy(dest, shifted, self.len - index);
        }
        // SAFETY: we write to `dest`, which is initialized and within bounds.
        Ok(unsafe {
            dest.write(value);
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.len < self.capacity, self.capacity <= usize::MAX"
            )]
            {
                self.len += 1;
            }
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
    /// - The allocated size in bytes must be no larger than `isize::MAX`. See the safety documentation of `ptr.offset()`.
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
    pub unsafe fn from_parts_in(ptr: NonNull<T>, length: usize, capacity: usize, alloc: A) -> Self {
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
        let mut this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller.
        unsafe {
            (
                this.as_mut_ptr(),
                this.len(),
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
    pub unsafe fn into_parts_with_alloc(self) -> (NonNull<T>, usize, usize, A) {
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

    /// Ensures the vector has room for at least `total` elements *in total*
    /// (an absolute target, not an increment).
    ///
    /// `total < len` results in a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the new capacity overflows or the
    /// allocation fails.
    pub fn try_reserve_total(&mut self, total: usize) -> Result<(), TryReserveError> {
        let additional = total.saturating_sub(self.len);
        self.raw.try_reserve(self.len, additional)
    }

    /// Shrinks the capacity down to `min_capacity`, keeping at least `len`.
    /// Note that the resulting capacity may still be more than `min_capacity`.
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

    /// Shrinks the capacity to attempt to fit the current length.
    /// Note that the actual capacity may still be more than the current length.
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
    /// potentially no excess capacity.
    ///
    /// No elements are cloned: when there is spare capacity the buffer is
    /// shrunk in place, then handed straight to the box. For an empty vector
    /// this returns an empty boxed slice without allocating.
    ///
    /// # Implementation notes
    ///
    /// In case there is more hidden capacity than the slice describes due to
    /// allocator quirks, it is still possible to deallocate or grow down the line
    /// because the allocator specification allows specifying any current layout
    /// that is anywhere between the size of the expected layout and the actual
    /// layout given (including both ends).
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
        // SAFETY: the buffer holds exactly `len` initialized elements.
        unsafe {
            let (buf, len, _cap, alloc) = self.into_raw_parts_with_alloc();
            let slice = ptr::slice_from_raw_parts_mut(buf, len);
            Ok(Box::from_raw_in(slice, alloc))
        }
    }

    /// Converts this vector into a `Box<[T; N]>`, consuming the vector.
    ///
    /// The vector must contain exactly `N` elements; otherwise
    /// [`TryVecIntoArrayError::LengthMismatch`] is returned. If the vector has
    /// excess capacity beyond `N`, a shrink-to-fit reallocation is attempted
    /// first; failure there yields [`TryVecIntoArrayError::Shrink`].
    ///
    /// # Implementation notes
    ///
    /// In case there is more hidden capacity than the array describes due to
    /// allocator quirks, it is still possible to deallocate or grow down the line
    /// because the allocator specification allows specifying any current layout
    /// that is anywhere between the size of the expected layout and the actual
    /// layout given (including both ends).
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
        // SAFETY: the buffer holds exactly `N` initialized elements allocated by `A`;
        // casting to `*mut [T; N]` is layout-compatible. Allocator is moved out.
        unsafe {
            let (raw_ptr, _len, _cap, alloc) = self.into_raw_parts_with_alloc();
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

/// Panic-aware rollback guard for fallible mutation loops.
///
/// If the guarded section unwinds (e.g. a clone or closure panics), `Drop`
/// truncates the vector back to `original_len`, ensuring any partially-written
/// tail is dropped exactly once.
///
/// Stores a raw `*mut Vec` to avoid conflicting with the `&mut self` borrows
/// used by the guarded loop body. The pointer is derived directly from the
/// enclosing `&mut self` (via `&raw mut *self`), so it carries mutable provenance
/// — no `*const → *mut` reborrow round-trip, which keeps Miri's Stacked Borrows
/// model happy.
struct RollbackGuard<T, A: Allocator>(*mut Vec<T, A>, usize);

impl<T, A: Allocator> Drop for RollbackGuard<T, A> {
    fn drop(&mut self) {
        // SAFETY: the pointer was obtained from a valid `&mut Vec` at guard
        // construction time; the Vec is alive for the entire scope (the guard
        // is a local that drops before the function returns).
        let vec_ref = unsafe { &mut *self.0 };
        vec_ref.truncate(self.1);
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
            #[allow(clippy::arithmetic_side_effects, reason = "asserted self.len > 0")]
            {
                self.len -= 1;
            }
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
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted new_len < self.len"
            )]
            let remaining = { self.len - new_len };
            let tail = slice::from_raw_parts_mut(self.as_mut_ptr().add(new_len), remaining);
            self.len = new_len;
            ptr::drop_in_place(tail);
        }
    }

    /// Removes a range of elements from the vector and returns them as an
    /// iterator, shifting later elements to the left to fill the gap.
    ///
    /// This is the fallible-port analogue of `std`'s `Vec::drain`. The only
    /// failure mode is validating the requested range: iteration itself never
    /// allocates (it merely destroys and shifts elements), so once the drainer
    /// is constructed it cannot fail.
    ///
    /// # Errors
    ///
    /// Returns [`TrySliceRangeError`] if the resolved range is out of bounds or
    /// reversed.
    ///
    /// # Leaking
    ///
    /// If [`mem::forget`](core::mem::forget) is called, all elements from the
    /// start of `range` to the end of the vector, except the extracted elements,
    /// are leaked.
    pub fn try_drain<R: core::ops::RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<Drain<'_, T, A>, TrySliceRangeError> {
        let len = self.len();
        let r = try_range(range, ..len)?;
        let start = r.start;
        let end = r.end;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start <= end (guaranteed by try_range)"
        )]
        let count = end - start;
        // Cap the vector's logical length down to `start` *before* handing out
        // the drainer. Once `next()` has yielded an element via `ptr::read`,
        // that slot is uninitialized; if the vec still counted it in its length,
        // a later drop of the vec (e.g. after the drainer is forgotten) would
        // drop uninitialized memory. Capping excludes the whole drained range
        // from the vec's live contents up front, so the vec can never touch the
        // hole. This mirrors std's internal bookkeeping.
        //
        // SAFETY: `start <= len <= capacity()`, and every slot in `[0..start)`
        // is an initialized value, so capping to `start` preserves the invariant
        // that all live slots are valid.
        unsafe {
            self.set_len(start);
        }
        // SAFETY: `count == end - start` is the full resolved range length and
        // `start + count == end <= len`; the first `len` slots are initialized.
        // The vector already had its length capped
        let drain = unsafe { Drain::new_from_parts(start, count, len, &raw mut *self) };
        Ok(drain)
    }

    /// Splits the collection into two at index `at`, keeping the portion before
    /// `at` in `self` and returning the portion from `at` onward as a new `Vec`.
    ///
    /// This is the fallible-port analogue of std's `Vec::split_off`. The only
    /// failure modes are an out-of-bounds index or a failed allocation for the
    /// right-hand half.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecSplitOffError::OutOfBounds`] if `at > len()`, or
    /// [`TryVecSplitOffError::Reserve`] if allocating the new vector fails.
    pub fn try_split_off(&mut self, at: usize) -> Result<Vec<T, A>, TryVecSplitOffError>
    where
        A: Clone,
    {
        let len = self.len;
        if at > len {
            return Err(TryVecSplitOffError::OutOfBounds { index: at, len });
        }
        let alloc = self.raw.allocator().clone();
        // Fast path: splitting at the end produces an empty tail.
        if at == len {
            return Ok(Self::new_in(alloc));
        }
        #[allow(clippy::arithmetic_side_effects, reason = "asserted at <= len")]
        let tail_len = len - at;
        let mut out = Vec::<T, A>::try_with_capacity_in(tail_len, alloc)
            .map_err(TryVecSplitOffError::Reserve)?;
        // Mirror std: set both lengths first, then bitwise-copy the tail.
        // SAFETY: `at < len <= capacity`, so reducing self's len is valid.
        // `out` was just allocated with capacity >= `tail_len` and is empty,
        // so writing `tail_len` elements into it and setting its length is
        // within bounds. The source range `[at..len)` is initialized.
        // Additionally, at < len, so no OOB or pointer overflow.
        unsafe {
            self.set_len(at);
            out.set_len(tail_len);
            ptr::copy_nonoverlapping(self.as_ptr().add(at), out.as_mut_ptr(), tail_len);
        }
        Ok(out)
    }

    /// Extends the vector with a number of copied elements from within itself.
    ///
    /// This is the fallible-port analogue of std's `Vec::extend_from_within`.
    /// The range `indices` selects elements to clone and append to the end of
    /// the vector.
    ///
    /// If a clone fails mid-way, the vector is truncated back to its length at
    /// the start of the call so no partially-appended elements remain.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecExtendFromWithinError`] if the range is invalid,
    /// resolution overflows, capacity reservation fails, or an element clone
    /// fails.
    pub fn try_extend_from_within<R: core::ops::RangeBounds<usize>>(
        &mut self,
        indices: R,
    ) -> Result<(), TryVecExtendFromWithinError>
    where
        T: TryClone,
    {
        let len = self.len;
        let r = try_range(indices, ..len).map_err(|e| match e {
            TrySliceRangeError::StartOverflow | TrySliceRangeError::EndOverflow => {
                TryVecExtendFromWithinError::RangeOverflow
            }
            TrySliceRangeError::StartExceedsEnd { start, end, len } => {
                TryVecExtendFromWithinError::InvalidRange { start, end, len }
            }
            TrySliceRangeError::EndExceedsBound { start, end, len } => {
                TryVecExtendFromWithinError::InvalidRange { start, end, len }
            }
        })?;
        let start = r.start;
        let end = r.end;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start <= end (guaranteed by try_range)"
        )]
        let count = end - start;
        if count == 0 {
            return Ok(());
        }
        // Reserve up front so the vector never allocates during the hot path push.
        self.try_reserve(count)
            .map_err(TryVecExtendFromWithinError::Reserve)?;
        // If any clone or push fails mid-loop (or the body panics), the guard
        // truncates back to `len`, dropping every appended element exactly once.
        let guard = RollbackGuard(&raw mut *self, len);
        for i in 0..count {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "i < count, start+i < end <= len"
            )]
            let src_val = &self[start + i];
            let cloned = src_val
                .try_clone()
                .map_err(TryVecExtendFromWithinError::Clone)?;
            // SAFETY: we reserved enough space.
            unsafe { self.force_push(cloned) };
        }
        core::mem::forget(guard);
        Ok(())
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
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "if len == 0, then index >= len, which causes early return above"
        )]
        {
            self.len -= 1;
        }
        let last_index = self.len;
        // SAFETY: bounds checked above.
        Ok(unsafe {
            let value = ptr::read(self.as_mut_ptr().add(index));
            if index != last_index {
                ptr::copy_nonoverlapping(
                    self.as_mut_ptr().add(last_index),
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
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "if len == 0, then index >= len, which causes early return above"
        )]
        {
            self.len -= 1;
        }
        let ptr = self.as_mut_ptr();
        // SAFETY: bounds checked above.
        let ptr_at_index = unsafe { ptr.add(index) };
        let removed = unsafe { ptr::read(ptr_at_index) };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted index < len => index + 1 <= len"
        )]
        let index_plus_one = index + 1;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted index + 1 <= len, by corollary len - (index + 1) >= 0"
        )]
        let remaining = len - index_plus_one;
        // SAFETY: We are removing elements from `index + 1..len` (`len` noninclusive).
        // if `remaining == 0`, the source pointer is out of bounds and may overflow to 0 in special cases.
        if remaining > 0 {
            unsafe {
                ptr::copy(ptr.add(index_plus_one), ptr_at_index, remaining);
            }
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
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "asserted self.read <= self.original_len"
                )]
                let remaining = self.original_len - self.read;
                // If `remaining == 0`, then `self.read == self.original_len` and
                // `as_ptr().add(self.read)` would address one past the buffer's
                // end (and could even overflow when `original_len == capacity`).
                // A zero-length copy is a no-op anyway, so skip it entirely.
                if remaining > 0 {
                    // SAFETY: Trailing unchecked items must be valid since we
                    // never touch them, and `read < original_len <= capacity`.
                    unsafe {
                        ptr::copy(
                            self.v.as_ptr().add(self.read),
                            self.v.as_mut_ptr().add(self.write),
                            remaining,
                        );
                    }
                }
                // SAFETY: After filling holes, all items are in contiguous memory.
                unsafe {
                    #[allow(
                        clippy::arithmetic_side_effects,
                        reason = "asserted write + remaining < read + remaining <= original_len"
                    )]
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
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted read <= original_len"
            )]
            {
                read += 1;
            }
            if read == original_len {
                // All elements are kept, return early.
                return;
            }
        }

        // Critical section starts here and at least one element is going to be removed.
        // Advance `g.read` early to avoid double drop (i.e. shifting a hole that has attempted drop in)
        // if `drop_in_place` panicked.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted read < original_len => read + 1 == original_len"
        )]
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
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "asserted g.read < original_len"
                )]
                {
                    g.read += 1;
                }
                // SAFETY: We never touch this element again after dropped.
                unsafe { ptr::drop_in_place(cur) };
            } else {
                // SAFETY: `read` > `write`, so the slots don't overlap.
                // We use copy for move, and never touch the source element again.
                unsafe {
                    let hole = g.v.as_mut_ptr().add(g.write);
                    ptr::copy_nonoverlapping(cur, hole, 1);
                }
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "asserted g.write < g.read < original_len"
                )]
                {
                    g.write += 1;
                }
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "asserted g.read < original_len"
                )]
                {
                    g.read += 1;
                }
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
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        let len = self.len();
        if len <= 1 {
            return;
        }

        // Check if we ever want to remove anything.
        // This allows to use copy_non_overlapping in next cycle.
        // And avoids any memory writes if we don't need to remove anything.
        let mut first_duplicate_idx: usize = 1;
        let start = self.as_mut_ptr();
        while first_duplicate_idx != len {
            let found_duplicate = unsafe {
                // SAFETY: first_duplicate always in range [1..len)
                // Note that we start iteration from 1 so we never overflow.
                let prev = start.add(first_duplicate_idx.wrapping_sub(1));
                let current = start.add(first_duplicate_idx);
                // We explicitly say in docs that references are reversed.
                same_bucket(&mut *current, &mut *prev)
            };
            if found_duplicate {
                break;
            }
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted in loop that first_duplicate_idx != len and 
                first_duplicate_idx starts with 0 => first_duplicate_idx < len"
            )]
            {
                first_duplicate_idx += 1;
            }
        }
        // Don't need to remove anything.
        // We cannot get bigger than len.
        if first_duplicate_idx == len {
            return;
        }

        /* INVARIANT: vec.len() >= read > write > write-1 >= 0 */
        struct FillGapOnDrop<'a, T, A: Allocator> {
            /* Offset of the element we want to check if it is duplicate */
            read: usize,

            /* Offset of the place where we want to place the non-duplicate
             * when we find it. */
            write: usize,

            /* The Vec that would need correction if `same_bucket` panicked */
            vec: &'a mut Vec<T, A>,
        }

        impl<T, A: Allocator> Drop for FillGapOnDrop<'_, T, A> {
            fn drop(&mut self) {
                /* This code gets executed when `same_bucket` panics */

                /* SAFETY: invariant guarantees that `read - write`
                 * and `len - read` never overflow and that the copy is always
                 * in-bounds. */
                unsafe {
                    let ptr = self.vec.as_mut_ptr();
                    let len = self.vec.len();

                    /* How many items were left when `same_bucket` panicked.
                     * Basically vec[read..].len() */
                    let items_left = len.wrapping_sub(self.read);

                    if items_left > 0 {
                        /* Pointer to first item in vec[write..write+items_left] slice */
                        let dropped_ptr = ptr.add(self.write);
                        /* Pointer to first item in vec[read..] slice */
                        let valid_ptr = ptr.add(self.read);

                        /* Copy `vec[read..]` to `vec[write..write+items_left]`.
                         * The slices can overlap, so `copy_nonoverlapping`
                         * cannot be used. Skipping the copy when `items_left
                         * == 0` avoids forming a one-past-the-end (and
                         * possibly overflowing) source pointer. */
                        ptr::copy(valid_ptr, dropped_ptr, items_left);
                    }

                    /* How many items have been already dropped
                     * Basically vec[read..write].len() */
                    // asserted self.write < self.read
                    let dropped = self.read.wrapping_sub(self.write);

                    #[allow(
                        clippy::arithmetic_side_effects,
                        reason = "asserted dropped <= self.read <= len"
                    )]
                    self.vec.set_len(len - dropped);
                }
            }
        }

        /* Drop items while going through Vec, it should be more efficient than
         * doing slice partition_dedup + truncate */

        // Construct gap first and then drop item to avoid memory corruption if `T::drop` panics.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted first_duplicate_idx < len"
        )]
        let mut gap = FillGapOnDrop {
            read: first_duplicate_idx + 1,
            write: first_duplicate_idx,
            vec: self,
        };
        unsafe {
            // SAFETY: we checked that first_duplicate_idx in bounds before.
            // If drop panics, `gap` would remove this item without drop.
            ptr::drop_in_place(start.add(first_duplicate_idx));
        }

        /* SAFETY: Because of the invariant, read_ptr, prev_ptr and write_ptr
         * are always in-bounds and read_ptr never aliases prev_ptr */
        unsafe {
            while gap.read < len {
                // SAFETY: `gap.read < len <= capacity`, so both offsets address
                // initialized slots within the buffer. When `gap.read` is the
                // final index (`len - 1`), the pointers still point at valid,
                // allocated memory — no one-past-the-end or overflowing offset
                // is ever formed here.
                let read_ptr = start.add(gap.read);
                let prev_ptr = start.add(gap.write.wrapping_sub(1));

                // We explicitly say in docs that references are reversed.
                let found_duplicate = same_bucket(&mut *read_ptr, &mut *prev_ptr);
                if found_duplicate {
                    // Increase `gap.read` now since the drop may panic.
                    #[allow(clippy::arithmetic_side_effects, reason = "asserted gap.read < len")]
                    {
                        gap.read += 1;
                    }
                    /* We have found duplicate, drop it in-place */
                    ptr::drop_in_place(read_ptr);
                } else {
                    let write_ptr = start.add(gap.write);

                    /* read_ptr cannot be equal to write_ptr because at this point
                     * we guaranteed to skip at least one element (before loop starts).
                     */
                    ptr::copy_nonoverlapping(read_ptr, write_ptr, 1);

                    /* We have filled that place, so go further */
                    #[allow(
                        clippy::arithmetic_side_effects,
                        reason = "asserted gap.write < gap.read < len"
                    )]
                    {
                        gap.write += 1;
                    }
                    #[allow(clippy::arithmetic_side_effects, reason = "asserted gap.read < len")]
                    {
                        gap.read += 1;
                    }
                }
            }

            /* Technically we could let `gap` clean up with its Drop, but
             * when `same_bucket` is guaranteed to not panic, this bloats a little
             * the codegen, so we just do it manually */
            gap.vec.set_len(gap.write);
            olive_core::mem::forget(gap);
        }
    }

    /// Pushes an element onto the end of the vector without attempting to grow
    /// the buffer. Succeeds only if there is already spare capacity.
    ///
    /// # Errors
    ///
    /// Returns [`TryPushWithinCapacityError`] if `len == capacity`.
    pub fn try_push_within_capacity(&mut self, value: T) -> Result<(), TryPushWithinCapacityError> {
        // Prevent a degenerate scenario where `length` is exceeding `capacity`
        if self.len >= self.capacity() {
            return Err(TryPushWithinCapacityError { len: self.len });
        }
        // SAFETY: spare capacity was just confirmed above.
        unsafe { self.force_push(value) };
        Ok(())
    }

    /// Pops the last element off the vector if it satisfies the predicate,
    /// returning `None` otherwise.
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        // SAFETY: len > 0, so the last slot is initialized.
        #[allow(clippy::arithmetic_side_effects, reason = "asserted self.len > 0")]
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
        self.try_reserve(other.len())
            .map_err(TryVecWithCloneError::Reserve)?;
        let len_before = self.len;
        let guard = RollbackGuard(&raw mut *self, len_before);
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    // SAFETY: capacity was reserved above for all of `other`.
                    unsafe { self.force_push(cloned) };
                }
                Err(e) => {
                    return Err(TryVecWithCloneError::Clone(e));
                }
            }
        }
        core::mem::forget(guard);
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
    pub fn try_append(&mut self, other: &mut Self) -> Result<(), TryReserveError> {
        let extra = other.len;
        if extra == 0 {
            return Ok(());
        }
        self.try_reserve(extra)?;
        let src = other.as_mut_ptr();
        let dst = unsafe { self.as_mut_ptr().add(self.len) };
        // SAFETY: `self` and `other` are distinct vectors; their buffers never overlap.
        unsafe {
            ptr::copy_nonoverlapping(src, dst, extra);
        }
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted len + extra <= capacity (reserved above)"
        )]
        {
            self.len += extra;
        }
        other.len = 0;
        Ok(())
    }

    /// Resizes the vector so its length becomes `new_len`.
    ///
    /// Parameter order matches [`Self::try_resize_with`] and std's `resize`:
    /// the target length comes first, then the fill value.
    ///
    /// If `new_len` is greater than the current length, the vector is extended
    /// by cloning `value` via [`TryClone`]. If smaller, it is truncated.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    pub fn try_resize(&mut self, new_len: usize, value: &T) -> Result<(), TryVecWithCloneError>
    where
        T: TryClone,
    {
        let current = self.len;
        if new_len <= current {
            self.truncate(new_len);
            return Ok(());
        }
        #[allow(clippy::arithmetic_side_effects, reason = "asserted new_len > current")]
        let extra = new_len - current;
        self.try_reserve(extra)
            .map_err(TryVecWithCloneError::Reserve)?;
        let guard = RollbackGuard(&raw mut *self, current);
        for _ in 0..extra {
            match value.try_clone() {
                Ok(cloned) => {
                    // SAFETY: capacity was reserved above for all `extra`.
                    unsafe { self.force_push(cloned) };
                }
                Err(e) => {
                    return Err(TryVecWithCloneError::Clone(e));
                }
            }
        }
        core::mem::forget(guard);
        Ok(())
    }

    /// Resizes the vector so its length becomes `new_len`, producing new
    /// elements with the fallible closure `f`.
    ///
    /// The closure is invoked only after capacity is secured. If it returns an
    /// error, the vector is truncated back to its original length so no
    /// partially-produced elements remain.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithClosureError<E>`] if either the capacity reservation
    /// fails or the closure returns `Err(e)`.
    pub fn try_resize_with<E, F>(
        &mut self,
        new_len: usize,
        mut f: F,
    ) -> Result<(), TryVecWithClosureError<E>>
    where
        F: FnMut() -> Result<T, E>,
    {
        let current = self.len;
        if new_len <= current {
            self.truncate(new_len);
            return Ok(());
        }
        #[allow(clippy::arithmetic_side_effects, reason = "asserted new_len > current")]
        let extra = { new_len - current };
        self.try_reserve(extra)
            .map_err(TryVecWithClosureError::Reserve)?;
        let guard = RollbackGuard(&raw mut *self, current);
        for _ in 0..extra {
            match f() {
                Ok(item) => {
                    // SAFETY: capacity was reserved above for all `extra`.
                    unsafe { self.force_push(item) };
                }
                Err(e) => {
                    // Guard drops here and truncates back to `current`.
                    return Err(TryVecWithClosureError::Closure(e));
                }
            }
        }
        // Success: defuse the guard so it doesn't truncate the new elements.
        core::mem::forget(guard);
        Ok(())
    }

    /// Swaps two elements in the vector by their indices without panicking.
    ///
    /// The caller owns the outcome: an out-of-bounds index is reported through
    /// the returned [`Result`] rather than unwinding.
    ///
    /// # Errors
    ///
    /// Returns [`TrySwapError`] if either index is out of bounds.
    pub fn try_swap(&mut self, a: usize, b: usize) -> Result<(), TrySwapError> {
        let len = self.len;
        if a >= len || b >= len {
            return Err(TrySwapError {
                index: if a >= len { a } else { b },
                len,
            });
        }
        // SAFETY: both indices were bounds-checked above.
        unsafe {
            let pa = self.as_mut_ptr().add(a);
            let pb = self.as_mut_ptr().add(b);
            ptr::swap(pa, pb);
        }
        Ok(())
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
        let mut vec =
            Self::try_with_capacity_in(count, alloc).map_err(TryVecWithCloneError::Reserve)?;

        for _ in 0..count {
            match value.try_clone() {
                Ok(cloned) => {
                    // SAFETY: Capacity was reserved above, so this cannot fail.
                    unsafe { vec.force_push(cloned) };
                }
                Err(e) => return Err(TryVecWithCloneError::Clone(e)),
            }
        }
        Ok(vec)
    }

    /// Creates a [`Vec<T>`] from a slice by cloning each element via
    /// [`TryClone`].
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    pub fn try_from_slice_in(slice: &[T], alloc: A) -> Result<Self, TryVecWithCloneError>
    where
        T: TryClone,
    {
        let mut vec = Self::try_with_capacity_in(slice.len(), alloc)
            .map_err(TryVecWithCloneError::Reserve)?;
        for item in slice {
            match item.try_clone() {
                Ok(cloned) => {
                    // SAFETY: Capacity was reserved above, so this cannot fail.
                    unsafe { vec.force_push(cloned) };
                }
                Err(e) => return Err(TryVecWithCloneError::Clone(e)),
            }
        }
        Ok(vec)
    }

    /// Fallibly collects an iterator into a [`Vec<T>`] on the given allocator,
    /// using the size hint to pre-allocate when possible.
    ///
    /// This is the allocator-aware backend for [`TryFromIterator`]. It reserves
    /// up front from the hint's upper bound and grows as needed if the iterator
    /// yields more elements than advertised.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if a reservation fails.
    pub fn try_from_iter_in<I: IntoIterator<Item = T>>(
        iter: I,
        alloc: A,
    ) -> Result<Self, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut vec = Self::try_with_capacity_in(capacity, alloc)?;
        for item in iter {
            // The iterator may yield more elements than its hint promised.
            if vec.len == vec.capacity() {
                // SAFETY: `len == capacity` holds here.
                unsafe { vec.raw.try_grow_one()? };
            }
            // SAFETY: we just confirmed there is a spare slot.
            unsafe { vec.force_push(item) };
        }
        Ok(vec)
    }
}

/// Fallible construction of a `Vec<T>` on the default [`Global`] allocator from
/// a borrowed slice, cloning each element via [`TryClone`].
impl<T: TryClone> TryFrom<&[T]> for Vec<T, Global> {
    type Error = TryVecWithCloneError;

    fn try_from(slice: &[T]) -> Result<Self, Self::Error> {
        Self::try_from_slice_in(slice, Global)
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

impl<T, A: Allocator> Borrow<[T]> for Vec<T, A> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> BorrowMut<[T]> for Vec<T, A> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
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

    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = T>,
    {
        let (head, mut inner, hint) = source.decompose_with_size_hint();
        // Ignore over-reserve.
        let _ = self.try_reserve_total(hint.estimated_total());
        // Push the head first.
        if let Some(head) = head {
            if let Err((head, err)) = self.try_push_give_back(head) {
                return Err((Resume::new(head, inner), err));
            }
        }

        // Push the remainder. While we have spare capacity this is a cheap
        // force_push; once capacity is exhausted (under-hinted or OOM'd) grow
        // one slot at a time, stranding the current element on failure.
        while let Some(next) = inner.next() {
            if self.len == self.capacity() {
                // SAFETY: `len == capacity` holds here.
                if let Err(e) = unsafe { self.raw.try_grow_one() } {
                    return Err((Resume::new(next, inner), e));
                }
            }
            // SAFETY: a spare slot was just confirmed.
            unsafe { self.force_push(next) };
        }
        Ok(())
    }
}

impl<'s, T, A: Allocator> TryExtendFromSlice<'s, T> for Vec<T, A>
where
    T: TryClone,
{
    type Error = TryCloneError;

    fn try_extend_from_slice(&mut self, other: &'s [T]) -> Result<(), (&'s [T], Self::Error)> {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(|e| (other, TryCloneError::Reserve(e)))?;
        let mut i = 0usize;
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    // SAFETY: capacity was reserved above for all of `other`.
                    unsafe { self.force_push(cloned) };
                    #[allow(clippy::arithmetic_side_effects, reason = "i <= other.len()")]
                    {
                        i += 1;
                    }
                }
                Err(e) => return Err((&other[i..], e)),
            }
        }
        Ok(())
    }
}

impl<T: TryClone, A: AllocatorTryClone> TryClone for Vec<T, A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = Self::new_in(self.raw.allocator().try_clone()?);
        if !self.is_empty() {
            out.try_reserve(self.len).map_err(TryCloneError::Reserve)?;
        }
        for elem in self.iter() {
            match elem.try_clone() {
                // SAFETY: capacity was reserved above for every element.
                Ok(cloned) => unsafe { out.force_push(cloned) },
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }
}

// An empty vector never allocates, so its default construction is infallible.
// The default allocator is `Global`, matching std's `Vec<T>` (which defaults
// to the global allocator).
impl<T> TryDefault for Vec<T, Global> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Vec::new())
    }
}

// Collects into the default (`Global`) allocator. For a custom allocator use
// [`Vec::try_from_iter_in`].
impl<T> TryFromIterator<T> for Vec<T, Global> {
    type Error = TryReserveError;

    fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Self::Error> {
        Self::try_from_iter_in(iter, Global)
    }
}

mod drain;
mod into_iter;
pub use drain::Drain;
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

// Borrowed iteration delegates to the slice methods, mirroring `std`'s
// `IntoIterator` impls for `&Vec` and `&mut Vec`.
impl<'a, T, A: Allocator> IntoIterator for &'a Vec<T, A> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a mut Vec<T, A> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

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
