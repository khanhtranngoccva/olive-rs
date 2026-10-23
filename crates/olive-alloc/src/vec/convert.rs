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
            let (buf, len, _cap, alloc) = self.into_raw_parts_with_allocator();
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
            let (raw_ptr, _len, _cap, alloc) = self.into_raw_parts_with_allocator();
            let array_ptr = raw_ptr.cast::<[T; N]>();
            Ok(Box::from_raw_in(array_ptr, alloc))
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion from types methods — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
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

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::borrow::Cow;
    use olive_core::TryClone;
    use olive_core::try_traits::try_clone::TryCloneError;
    use olive_test_helpers::{
        BudgetedAlloc, CloneBudget, FailAlloc, FlakyClone, FlakyCloneTrackedItem, Ledger,
    };
    use std::format;
    use std::sync::Arc;

    #[test]
    fn try_into_array_error_display() {
        let e = TryVecIntoArrayError::LengthMismatch {
            expected: 4,
            actual: 2,
        };
        let msg = format!("{e}");
        assert!(msg.contains("[T; 4]"));
        assert!(msg.contains("2"));

        let e2 = TryVecIntoArrayError::Shrink(TryReserveError::new_capacity_overflow());
        let msg2 = format!("{e2}");
        assert!(msg2.contains("shrink"));
    }

    #[test]
    fn try_into_boxed_slice_preserves_contents() {
        let v = {
            let mut v: Vec<i32> = Vec::new();
            for i in 0..5 {
                v.try_push(i).unwrap();
            }
            v
        };
        let boxed = v.try_into_boxed_slice().expect("box ok");
        assert_eq!(boxed.as_ref(), [0, 1, 2, 3, 4]);
    }

    #[test]
    fn try_into_boxed_slice_empty() {
        let v: Vec<i32> = Vec::new();
        let boxed = v.try_into_boxed_slice().expect("box ok");
        assert!(boxed.is_empty());
    }

    #[test]
    fn try_into_boxed_slice_give_back_success() {
        let v = {
            let mut v: Vec<i32> = Vec::new();
            for i in 0..5 {
                v.try_push(i).unwrap();
            }
            v
        };
        let boxed = v.try_into_boxed_slice_give_back().expect("box ok");
        assert_eq!(&boxed[..], &[0, 1, 2, 3, 4][..]);
    }

    #[test]
    fn try_into_boxed_slice_give_back_on_oom() {
        let v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        // Empty vec: shrink-to-fit on an empty buffer shouldn't need to allocate,
        // but FailAlloc makes any realloc fail. Either way, no data is lost.
        match v.try_into_boxed_slice_give_back() {
            Ok(boxed) => assert!(boxed.is_empty()),
            Err((vec_back, e)) => {
                assert!(e.is_alloc());
                assert!(vec_back.is_empty());
            }
        }
    }

    #[test]
    fn try_into_array_exact_length() {
        let mut v = Vec::new();
        for i in 0..4u8 {
            v.try_push(i).unwrap();
        }
        let arr: Box<[u8; 4]> = v.try_into_array().expect("should succeed");
        assert_eq!(arr.as_ref(), &[0, 1, 2, 3]);
    }

    #[test]
    fn try_into_array_wrong_length_too_few() {
        let mut v = Vec::new();
        v.try_push(1u8).unwrap();
        v.try_push(2).unwrap();
        let err = v.try_into_array::<4>().unwrap_err();
        match err {
            TryVecIntoArrayError::LengthMismatch { expected, actual } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 2);
            }
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn try_into_array_wrong_length_too_many() {
        let mut v = Vec::new();
        for i in 0..6u8 {
            v.try_push(i).unwrap();
        }
        let err = v.try_into_array::<3>().unwrap_err();
        match err {
            TryVecIntoArrayError::LengthMismatch { expected, actual } => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 6);
            }
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn try_into_array_zero_size() {
        let v: Vec<u32> = Vec::new();
        let arr: Box<[u32; 0]> = v.try_into_array().expect("empty array ok");
        assert_eq!(arr.len(), 0);
    }

    #[test]
    fn try_into_array_with_excess_capacity() {
        // Push enough elements to trigger growth, then pop down to exactly N.
        let mut v = Vec::try_with_capacity(10).unwrap();
        for i in 0..3u8 {
            v.try_push(i).unwrap();
        }
        assert!(v.capacity() > 3);
        let arr: Box<[u8; 3]> = v.try_into_array().expect("shrink + convert");
        assert_eq!(arr.as_ref(), &[0, 1, 2]);
    }

    #[test]
    fn try_into_array_give_back_length_mismatch() {
        let mut v = Vec::new();
        v.try_push(99u8).unwrap();
        let (vec_back, err) = v.try_into_array_give_back::<5>().unwrap_err();
        assert_eq!(vec_back.len(), 1);
        assert_eq!(vec_back[0], 99);
        match err {
            TryVecIntoArrayError::LengthMismatch { expected, actual } => {
                assert_eq!(expected, 5);
                assert_eq!(actual, 1);
            }
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn try_into_array_give_back_success() {
        let mut v = Vec::new();
        for i in ['a', 'b', 'c'] {
            v.try_push(i).unwrap();
        }
        let arr: Box<[char; 3]> = v.try_into_array_give_back().expect("ok");
        assert_eq!(arr.as_ref(), &['a', 'b', 'c']);
    }

    #[test]
    fn try_into_array_give_back_on_oom() {
        let alloc = BudgetedAlloc::new(1);
        let mut v: Vec<u8, BudgetedAlloc> =
            Vec::try_with_capacity_in(10, alloc.clone()).expect("alloc ok");
        for i in 0..3u8 {
            v.try_push(i).unwrap();
        }
        assert!(v.capacity() > 3);
        // With the budget drained, the shrink-to-fit reallocation must fail
        // and hand the vector back intact.
        alloc.drain();
        let (vec_back, e) = v.try_into_array_give_back::<3>().unwrap_err();
        match e {
            TryVecIntoArrayError::Shrink(r) => assert!(r.is_alloc()),
            other => panic!("expected Shrink, got {other:?}"),
        }
        assert_eq!(vec_back.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn from_slice_in_fail_alloc_reports_reserve() {
        let e = Vec::try_from_slice_in(&[1, 2], FailAlloc).expect_err("should fail");
        assert!(matches!(e, TryVecWithCloneError::Reserve(_)));
    }

    #[test]
    fn try_from_slice_in_reports_clone_failure() {
        let ledger = Arc::new(Ledger::new());
        // No tickets: the very first element clone must fail, and nothing
        // transient may be created.
        let budget = CloneBudget::new(0);
        let src = [FlakyCloneTrackedItem::construct(&ledger, budget.share())];
        let e = Vec::try_from_slice_in(&src, Global).expect_err("clone should fail");
        assert!(matches!(e, TryVecWithCloneError::Clone(_)));
        assert_eq!(ledger.live_ids(), [0]);
        assert!(ledger.double_dropped().is_empty());
        drop(src);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn try_from_array_moves_elements_without_cloning() {
        use core::cell::Cell;
        struct Counting(u32, Cell<u32>);
        impl TryClone for Counting {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                self.1.set(self.1.get() + 1);
                Ok(Self(self.0, Cell::new(self.1.get())))
            }
        }
        let clones = Cell::new(0);
        let arr = [Counting(1, clones.clone()), Counting(2, clones.clone())];
        let v: Vec<Counting, Global> = Vec::try_from(arr).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].0, 1);
        assert_eq!(v[1].0, 2);
        // Elements were moved, not cloned.
        assert_eq!(clones.get(), 0);
    }

    #[test]
    fn try_from_empty_array_is_empty() {
        let v: Vec<i32, Global> = Vec::try_from([]).unwrap();
        assert!(v.is_empty());
        // An empty vector of a non-ZST never allocates, so it reports no usable
        // slots (`usize::MAX` is reserved for zero-sized elements).
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn try_from_zst_array_no_alloc() {
        #[derive(TryClone, PartialEq, Copy, Clone)]
        struct Zst;
        let v: Vec<Zst, Global> = Vec::try_from([Zst; 5]).unwrap();
        assert_eq!(v.len(), 5);
        // ZST elements occupy no memory and never require a real allocation, so
        // the reported capacity is `usize::MAX` (matching std's convention).
        assert_eq!(v.capacity(), usize::MAX);
    }

    #[test]
    fn try_from_array_in_reports_oom() {
        let e = Vec::<i32, FailAlloc>::try_from_array_in([1, 2, 3], FailAlloc)
            .expect_err("should fail");
        assert!(e.is_alloc());
    }

    #[test]
    fn from_boxed_slice_preserves_contents_and_capacity() {
        // Build a vector with exactly `len` of capacity (no growth slack), so the
        // boxed slice carries a tight allocation and the round-trip is exact.
        let mut src = Vec::new();
        src.try_reserve_exact(3).unwrap();
        for x in [10, 20, 30] {
            src.try_push(x).unwrap();
        }
        let boxed: Box<[i32]> = src.try_into_boxed_slice().unwrap();
        let out: Vec<i32> = Vec::from(boxed);
        assert_eq!(out.as_slice(), &[10, 20, 30]);
        // The buffer was taken over untouched, so the reported capacity matches
        // the original allocation exactly.
        assert_eq!(out.capacity(), 3);
    }

    #[test]
    fn from_boxed_empty_slice() {
        let src: Vec<u8> = Vec::new();
        let boxed: Box<[u8]> = src.try_into_boxed_slice().unwrap();
        let v: Vec<u8> = Vec::from(boxed);
        assert!(v.is_empty());
        // The empty box carried no real allocation, so the reconstructed vector
        // of a non-ZST reports zero usable slots.
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn try_from_cow_borrowed_clones_elements() {
        let src: &[i32] = &[11, 22];
        let cow = Cow::Borrowed(src);
        let v: Vec<i32> = Vec::try_from(cow).unwrap();
        assert_eq!(v.as_slice(), &[11, 22]);
    }

    #[test]
    fn try_from_cow_borrowed_fails_when_element_clone_fails() {
        // The Borrowed arm delegates to `try_from_slice_in`, which clones each
        // element via `TryClone`. A `FlakyClone` with threshold 0 fails its first
        // clone attempt, so the very first element cannot be copied into the new
        // vector and the conversion surfaces a `Clone` error.
        let items = [FlakyClone::new(0), FlakyClone::new(0)];
        let cow = Cow::Borrowed(items.as_slice());
        match Vec::try_from(cow).expect_err("first element clone should fail") {
            TryVecWithCloneError::Clone(TryCloneError::Other(_)) => {}
            other => panic!("expected a clone failure, got {other:?}"),
        }
    }

    #[test]
    fn try_from_cow_owned_passes_through() {
        let mut owned: Vec<i32> = Vec::new();
        for x in [7, 8, 9] {
            owned.try_push(x).unwrap();
        }
        // Give the source vec surplus capacity so we can verify it's preserved.
        owned.try_reserve(5).unwrap();
        let src_cap = owned.capacity();
        assert!(src_cap > 3);

        let cow = Cow::Owned(owned);
        let v: Vec<i32> = Vec::try_from(cow).unwrap();
        assert_eq!(v.as_slice(), &[7, 8, 9]);
        // The Owned arm transfers the buffer wholesale — capacity is preserved.
        assert_eq!(v.capacity(), src_cap);
    }

    #[test]
    fn try_from_cow_borrowed_fails_when_reservation_fails() {
        // We cannot test the global allocator, so we mimic the code path that is used.
        let src: &[i32] = &[1, 2, 3];
        let res: Result<Vec<i32, FailAlloc>, _> = Vec::try_from_slice_in(src, FailAlloc);
        match res.expect_err("reservation should fail") {
            TryVecWithCloneError::Reserve(r) => assert!(r.is_alloc()),
            other => panic!("expected a reserve failure, got {other:?}"),
        }
    }

    #[test]
    fn try_from_borrowed_slice_clones_elements() {
        let src: &[i32] = &[10, 20, 30];
        let v: Vec<i32, Global> = Vec::try_from(src).unwrap();
        assert_eq!(v.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn try_from_borrowed_slice_fails_when_reservation_fails() {
        // The `Global`-scoped trait impl can't target a failing allocator, so drive
        // the same code path through the `_in` seam with one that does.
        let src: &[i32] = &[1, 2, 3];
        let res: Result<Vec<i32, FailAlloc>, _> = Vec::try_from_slice_in(src, FailAlloc);
        match res.expect_err("reservation should fail") {
            TryVecWithCloneError::Reserve(r) => {
                assert!(r.is_alloc());
            }
            other => panic!("expected a reserve failure, got {other:?}"),
        }
    }

    #[test]
    fn try_from_borrowed_slice_fails_when_element_clone_fails() {
        // `FlakyClone` with threshold 0 fails its first clone attempt, so the very
        // first element in the slice cannot be copied into the new vector.
        let items = [FlakyClone::new(0), FlakyClone::new(0)];
        let res: Result<Vec<FlakyClone, Global>, _> = Vec::try_from(&items[..]);
        match res.expect_err("first element clone should fail") {
            TryVecWithCloneError::Clone(TryCloneError::Other(_)) => {}
            other => panic!("expected a clone failure, got {other:?}"),
        }
    }
}
