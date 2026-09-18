//! Conversion trait implementations for [`Vec`](super::Vec).
use core::convert::TryFrom;

use olive_core::alloc::Allocator;
use olive_core::try_traits::try_clone::TryClone;
use olive_core::try_traits::try_from_iterator::TryFromIterator;

use crate::alloc::Global;
use crate::borrow::Cow;
use crate::boxed::Box;

use super::{TryReserveError, TryVecWithCloneError, Vec};

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
// TryFromIterator: any iterator, Global allocator
// ---------------------------------------------------------------------------

impl<T> TryFromIterator<T> for Vec<T, Global> {
    type Error = TryReserveError;

    fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Self::Error> {
        Self::try_from_iter_in(iter, Global)
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
