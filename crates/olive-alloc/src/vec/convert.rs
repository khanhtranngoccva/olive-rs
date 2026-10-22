//! Conversion trait implementations for [`Vec`](super::Vec).
use core::convert::TryFrom;
use core::mem::ManuallyDrop;
use core::{fmt, ptr};
use olive_core::try_traits::try_clone::TryClone;
use olive_core::{alloc::Allocator, alloc_errors::TryReserveError};

use crate::alloc::Global;
use crate::borrow::Cow;
use crate::boxed::Box;

use super::{TryVecWithCloneError, Vec};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

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
// Conversion into types methods — generic allocator
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
// Conversion from types methods — generic allocator
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

    /// Fallible conversion of an array into a vector on the given allocator,
    /// moving each element without cloning. The buffer is allocated with
    /// exactly `N` slots of capacity.
    ///
    /// This is the allocator-aware backend for [`TryFrom<[T; N]>`](core::convert::TryFrom).
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving the initial buffer fails.
    pub fn try_from_array_in<const N: usize>(
        array: [T; N],
        alloc: A,
    ) -> Result<Self, TryReserveError> {
        if N == 0 {
            return Ok(Self::new_in(alloc));
        }
        if size_of::<T>() == 0 {
            // ZST elements occupy no memory and need no buffer; just record
            // the length (capacity reports as `usize::MAX`, matching std).
            // SAFETY: a zero-length logical slice of ZSTs has no pointer
            // validity requirements beyond what `new_in` guarantees.
            let mut v = Self::new_in(alloc);
            unsafe { v.set_len(N) };
            return Ok(v);
        }
        // Allocate directly on the target allocator and move each element in.
        let mut dest = Self::try_with_capacity_in(N, alloc)?;
        // SAFETY: we wrap the array in `ManuallyDrop`, so its drop glue never
        // runs; every element below is moved out exactly once by the copy.
        let mut src = ManuallyDrop::new(array);
        let src_ptr = src.as_mut_ptr();
        // SAFETY: both pointers are valid for `N` reads/writes, do not overlap
        // (stack vs heap), and `T` is properly aligned on both sides. The
        // destination has exactly `N` reserved slots.
        unsafe { ptr::copy_nonoverlapping(src_ptr, dest.as_mut_ptr(), N) };
        // SAFETY: all `N` slots were just initialized by the copy above.
        unsafe { dest.set_len(N) };
        Ok(dest)
    }
}

// ---------------------------------------------------------------------------
// TryFrom: arrays (Global)
// ---------------------------------------------------------------------------

impl<T, const N: usize> TryFrom<[T; N]> for Vec<T, Global> {
    type Error = TryReserveError;

    /// Fallibly converts an array into a vector on the [`Global`] allocator,
    /// moving each element without cloning. The buffer is allocated with exactly
    /// `N` slots of capacity.
    fn try_from(array: [T; N]) -> Result<Self, Self::Error> {
        Vec::try_from_array_in(array, Global)
    }
}

// ---------------------------------------------------------------------------
// From: boxed slices
// ---------------------------------------------------------------------------

impl<T, A: Allocator> From<Box<[T], A>> for Vec<T, A> {
    /// Converts a boxed slice into a vector on the same allocator, taking
    /// ownership of the allocation without copying or reallocating.
    ///
    /// A `Box<[T]>` records only its logical length, so the reconstructed
    /// vector reports that same value as its capacity. This is always a legal
    /// lower bound for deallocation purposes (the block was allocated for at
    /// least `len` elements), matching how std's analogous conversion treats
    /// a boxed slice's capacity.
    fn from(boxed: Box<[T], A>) -> Self {
        let len = boxed.len();
        // Decompose the box into its raw parts and rebuild the vector around
        // the same buffer.
        // SAFETY: `into_raw_with_allocator` consumes the box; the pointer was
        // allocated by `alloc` with at least `len` initialized elements, which
        // satisfies every precondition of `from_raw_parts_in`.
        unsafe {
            let (slice_ptr, alloc) = Box::into_raw_with_allocator(boxed);
            let elem_ptr = slice_ptr.cast::<T>();
            Vec::from_raw_parts_in(elem_ptr, len, len, alloc)
        }
    }
}

// ---------------------------------------------------------------------------
// TryFrom: Cow (Global)
// ---------------------------------------------------------------------------

impl<'b, T> TryFrom<Cow<'b, [T]>> for Vec<T, Global>
where
    T: TryClone,
{
    type Error = TryVecWithCloneError;

    /// Consumes a clone-on-write borrow of a slice, returning the owned
    /// contents as a vector on the [`Global`] allocator.
    ///
    /// The `Owned` arm is a zero-cost pass-through: the inner `Vec<T, Global>`
    /// is returned as-is without cloning or reallocating. The `Borrowed` arm
    /// clones each element via [`TryClone`] into a freshly allocated buffer.
    fn try_from(cow: Cow<'b, [T]>) -> Result<Self, Self::Error> {
        match cow {
            Cow::Owned(inner) => Ok(inner),
            Cow::Borrowed(slice) => Self::try_from_slice_in(slice, Global),
        }
    }
}

// ---------------------------------------------------------------------------
// TryFrom: borrowed slice, Global allocator
// ---------------------------------------------------------------------------

/// Fallible construction of a [`Vec<T>`] on the [`Global`] allocator from a
/// borrowed slice, cloning each element via [`TryClone`].
impl<T: TryClone> TryFrom<&[T]> for Vec<T, Global> {
    type Error = TryVecWithCloneError;

    fn try_from(slice: &[T]) -> Result<Self, Self::Error> {
        Self::try_from_slice_in(slice, Global)
    }
}
