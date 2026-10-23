use super::RollbackGuard;
use super::Vec;
use super::{TryVecWithCloneError, TryVecWithClosureError};
use core::cmp;
use core::fmt;
use core::ptr;
use olive_core::alloc::AllocatorTryClone;
use olive_core::slice;
use olive_core::slice::TrySliceRangeError;
use olive_core::try_traits::TryClone;
use olive_core::try_traits::TryCloneError;
use olive_core::{alloc::Allocator, alloc_errors::TryReserveError};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

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
    /// Cloning the allocator failed.
    CloneAlloc(TryCloneError),
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
            Self::CloneAlloc(e) => f
                .debug_tuple("TryVecSplitOffError::CloneAlloc")
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
            Self::CloneAlloc(e) => write!(f, "split_off allocator cloning failed: {e}"),
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

// ---------------------------------------------------------------------------
// Allocating insert & push methods — generic allocator
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
// Allocation management — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
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

    /// Ensures the vector has room for at least `additional` more elements,
    /// preferring amortized growth but falling back to an exact-sized
    /// allocation when the amortized reserve is refused.
    ///
    /// This first attempts [`Self::try_reserve`], which grows by doubling so
    /// that future appends are cheap. If that allocation fails — typically
    /// because a tight allocator cannot satisfy the larger amortized size — it
    /// retries with [`Self::try_reserve_exact`], which asks for only the
    /// minimum memory required (`len + additional`). The exact request is
    /// always no larger than the amortized one, so it can succeed where the
    /// amortized attempt failed.
    ///
    /// Use this for bulk operations (e.g. extending from a slice of known
    /// length) where reserving up front is desirable but a refusal of the
    /// over-provisioned amortized size should not be fatal.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if both the amortized and the exact
    /// reservations fail.
    pub fn try_reserve_adaptive(&mut self, additional: usize) -> Result<(), TryReserveError> {
        match self.try_reserve(additional) {
            Ok(()) => Ok(()),
            Err(_) => self.try_reserve_exact(additional),
        }
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
// Splitting
// ---------------------------------------------------------------------------

impl<T, A: AllocatorTryClone> Vec<T, A> {
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
    // FIXME: splitting off should deal with allocator cloning
    pub fn try_split_off(&mut self, at: usize) -> Result<Vec<T, A>, TryVecSplitOffError> {
        let len = self.len;
        if at > len {
            return Err(TryVecSplitOffError::OutOfBounds { index: at, len });
        }
        let alloc = self
            .raw
            .allocator()
            .try_clone()
            .map_err(TryVecSplitOffError::CloneAlloc)?;
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
}

// ---------------------------------------------------------------------------
// Additional allocating methods
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
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
        let r = slice::try_range(indices, ..len).map_err(|e| match e {
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
        // Reserve up front so the vector never allocates during the hot path
        // push. Adaptive, so a refused over-provisioned amortized size falls
        // back to an exact-sized allocation instead of failing.
        self.try_reserve_adaptive(count)
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
        self.try_reserve_adaptive(extra)?;
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
        self.try_reserve_adaptive(extra)
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
        self.try_reserve_adaptive(extra)
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
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use olive_core::try_traits::try_clone::TryCloneError;
    use olive_core::try_traits::try_extend::TryExtend;
    use olive_test_helpers::{
        BudgetedAlloc, CloneBudget, FailAlloc, FlakyCloneTrackedItem, Ledger,
    };
    use std::sync::Arc;

    #[test]
    fn try_push_pop_roundtrip() {
        let mut v = Vec::new();
        v.try_push(1).expect("push ok");
        v.try_push(2).expect("push ok");
        assert_eq!(v.as_slice(), &[1, 2]);
        assert_eq!(v.pop(), Some(2));
        assert_eq!(v.pop(), Some(1));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn try_push_grows_amortized() {
        let mut v = Vec::new();
        for i in 0..100 {
            v.try_push(i).expect("push ok");
        }
        assert_eq!(v.len(), 100);
        assert!(v.capacity() >= 100);
        assert_eq!(v.as_slice()[99], 99);
    }

    #[test]
    fn try_push_fails_on_oom() {
        let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        // First push needs an allocation, which always fails.
        let err = v.try_push(42).expect_err("should fail");
        assert!(err.is_alloc());
        assert!(v.is_empty());
    }

    #[test]
    fn try_push_give_back_returns_value_on_oom() {
        let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        // First push needs an allocation, which always fails.
        let (returned, err) = v.try_push_give_back(42).expect_err("should fail");
        assert_eq!(returned, 42);
        assert!(err.is_alloc());
        assert!(v.is_empty());
    }

    #[test]
    fn try_push_mut_returns_reference_to_last() {
        let mut v = Vec::new();
        let r = v.try_push_mut(42i32).unwrap();
        *r += 8;
        assert_eq!(v.as_slice(), &[50]);
        let r2 = v.try_push_mut(100i32).unwrap();
        *r2 *= 2;
        assert_eq!(v.as_slice(), &[50, 200]);
    }

    #[test]
    fn try_push_mut_fails_on_oom() {
        let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        // First push needs an allocation, which always fails.
        let err = v.try_push_mut(42).expect_err("should fail");
        assert!(err.is_alloc());
        assert!(v.is_empty());
    }

    #[test]
    fn try_push_mut_give_back_returns_reference_on_success() {
        let mut v = Vec::new();
        let r = v.try_push_mut_give_back(10i32).unwrap();
        *r += 5;
        assert_eq!(v.as_slice(), &[15]);
    }

    #[test]
    fn try_push_mut_give_back_returns_value_on_oom() {
        let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        let (returned, err) = v.try_push_mut_give_back(99).expect_err("should fail");
        assert_eq!(returned, 99);
        assert!(err.is_alloc());
        assert!(v.is_empty());
    }

    #[test]
    fn try_insert_at_front_middle_end() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(3).unwrap();
        v.try_insert(0, 0).unwrap();
        v.try_insert(2, 2).unwrap();
        v.try_insert(4, 4).unwrap();
        assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn try_insert_out_of_bounds() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        let e = v.try_insert(5, 99).expect_err("oob");
        assert!(matches!(e, TryVecInsertError::OutOfBounds));
        assert_eq!(v.as_slice(), &[1]);
    }

    #[test]
    fn try_insert_fails_on_oom() {
        let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        // Inserting into an empty vec requires growth -> OOM.
        let err = v.try_insert(0, 7).expect_err("oom");
        assert!(matches!(err, TryVecInsertError::Reserve(_)));
        assert!(v.is_empty());
    }

    #[test]
    fn try_insert_give_back_inserts_on_success() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(3).unwrap();
        v.try_insert_give_back(1, 2).expect("insert ok");
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn try_insert_give_back_out_of_bounds() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        let (val, e) = v.try_insert_give_back(5, 99).expect_err("oob");
        assert_eq!(val, 99);
        assert!(matches!(e, TryVecInsertError::OutOfBounds));
        assert_eq!(v.as_slice(), &[1]);
    }

    #[test]
    fn try_insert_give_back_oom_returns_value() {
        let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        // Inserting at index 0 of an empty vec requires growth -> OOM.
        let (val, e) = v.try_insert_give_back(0, 7).expect_err("oom");
        assert_eq!(val, 7);
        assert!(matches!(e, TryVecInsertError::Reserve(_)));
        assert!(v.is_empty());
    }

    #[test]
    fn try_insert_mut_returns_reference() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(3).unwrap();
        {
            let slot: &mut i32 = v.try_insert_mut(1, 99).unwrap();
            *slot = 2;
        }
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn try_insert_mut_out_of_bounds() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        let e = v.try_insert_mut(5, 99).expect_err("oob");
        assert!(matches!(e, TryVecInsertError::OutOfBounds));
        assert_eq!(v.as_slice(), &[1]);
    }

    #[test]
    fn try_insert_mut_fails_on_oom() {
        let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        // Inserting into an empty vec requires growth -> OOM.
        let err = v.try_insert_mut(0, 7).expect_err("oom");
        assert!(matches!(err, TryVecInsertError::Reserve(_)));
        assert!(v.is_empty());
    }

    #[test]
    fn try_insert_mut_give_back_inserts_on_success() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(3).unwrap();
        {
            let slot: &mut i32 = v.try_insert_mut_give_back(1, 99).unwrap();
            *slot = 2;
        }
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn try_insert_mut_give_back_out_of_bounds() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        let (val, e) = v.try_insert_mut_give_back(5, 42).expect_err("oob");
        assert_eq!(val, 42);
        assert!(matches!(e, TryVecInsertError::OutOfBounds));
        assert_eq!(v.as_slice(), &[1]);
    }

    #[test]
    fn try_insert_mut_give_back_on_oom() {
        let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        // Inserting at index 0 of an empty vec requires growth -> OOM.
        let (val, err) = v.try_insert_mut_give_back(0, 42).unwrap_err();
        assert_eq!(val, 42);
        match err {
            TryVecInsertError::Reserve(_) => {}
            other => panic!("expected Reserve, got {other:?}"),
        }
    }

    #[test]
    fn resize_grows_by_cloning() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_resize(4, &9).expect("resize ok");
        assert_eq!(v.as_slice(), &[1, 9, 9, 9]);
    }

    #[test]
    fn resize_shrinks() {
        let mut v = Vec::new();
        for i in 0..5 {
            v.try_push(i).unwrap();
        }
        let before = v.capacity();
        v.try_resize(2, &0).expect("shrink ok");
        assert_eq!(v.as_slice(), &[0, 1]);
        // Shrinking only truncates; the allocation must keep its capacity.
        assert_eq!(v.capacity(), before);
    }

    // FIXME: missing resize OOM

    #[test]
    fn resize_rollbacks_partial_on_clone_failure() {
        let ledger = Arc::new(Ledger::new());
        let budget = CloneBudget::new(2);

        // Seed one live payload (id 0).
        let mut v: Vec<FlakyCloneTrackedItem> = Vec::new();
        v.try_push(FlakyCloneTrackedItem::construct(&ledger, budget.share()))
            .unwrap();

        let src = v.as_slice()[0].try_clone().expect("seed clone ok");

        let e = v.try_resize(4, &src).expect_err("clone should fail");
        assert!(matches!(e, TryVecWithCloneError::Clone(_)));
        assert_eq!(v.len(), 1);

        // The lone transient clone (id 2) was destroyed during rollback. The only
        // ids still alive are 0 (seed, in v) and 1 (src, a local) — both expected.
        // No double-free, and exactly the transient clone has been dropped so far.
        assert_eq!(ledger.live_ids(), [0, 1]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(
            ledger.drop_count(2),
            1,
            "the one appended clone must be dropped once"
        );
        assert_eq!(ledger.total_allocated(), 3);

        // Tear down: `src` (id 1) and the vec's seed (id 0) drop → all three gone.
        drop(src);
        drop(v);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..3));
    }

    #[test]
    fn resize_with_calls_closure() {
        let mut v = Vec::new();
        let mut counter = 0u32;
        v.try_resize_with(3, || -> Result<i32, ()> {
            counter += 1;
            Ok(counter as i32)
        })
        .expect("resize_with ok");
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    // FIXME: missing resize_with OOM

    #[test]
    fn resize_with_closure_error_rolls_back() {
        let ledger = Arc::new(Ledger::new());
        let budget = CloneBudget::new(u32::MAX);

        // Seed two live payloads (ids 0, 1).
        let mut v: Vec<FlakyCloneTrackedItem> = Vec::new();
        for i in 0..2u32 {
            v.try_push(FlakyCloneTrackedItem::with_id(&ledger, i, budget.share()))
                .unwrap();
        }

        // The closure appends two tracked payloads (ids 2, 3) and then errors.
        // The rollback must truncate back to two elements, dropping exactly
        // the two transient appends once.
        let mut calls = 0u32;
        let result: Result<(), TryVecWithClosureError<()>> = v.try_resize_with(6, || {
            calls += 1;
            if calls <= 2 {
                Ok(FlakyCloneTrackedItem::with_id(
                    &ledger,
                    calls + 1,
                    budget.share(),
                ))
            } else {
                Err(())
            }
        });
        assert!(matches!(result, Err(TryVecWithClosureError::Closure(()))));
        assert_eq!(v.len(), 2);

        // Ids 0 and 1 stay live in the vector; ids 2 and 3 were destroyed by
        // the rollback — no leaks, no double-frees.
        assert_eq!(ledger.live_ids(), [0, 1]);
        assert_eq!(ledger.drop_count(2), 1);
        assert_eq!(ledger.drop_count(3), 1);
        assert!(ledger.double_dropped().is_empty());

        drop(v);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..4));
    }

    #[test]
    fn append_moves_all_elements() {
        let mut a = Vec::new();
        a.try_push(1).unwrap();
        let mut b = Vec::new();
        b.try_push(2).unwrap();
        b.try_push(3).unwrap();
        a.try_append(&mut b).expect("append ok");
        assert_eq!(a.as_slice(), &[1, 2, 3]);
        assert!(b.is_empty());
    }

    // FIXME: missing append OOM

    #[test]
    fn shrink_to_fit_reduces_capacity() {
        let mut v = Vec::<i32>::try_with_capacity(100).expect("alloc ok");
        for i in 0..4 {
            v.try_push(i).unwrap();
        }
        let before = v.capacity();
        v.try_shrink_to_fit().expect("shrink ok");
        assert!(v.capacity() < before);
        assert!(v.capacity() >= 4);
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn shrink_to_fit_failure_keeps_capacity() {
        let alloc = BudgetedAlloc::new(1);
        let mut v: Vec<i32, BudgetedAlloc> =
            Vec::try_with_capacity_in(16, alloc.clone()).expect("alloc ok");
        for i in 0..4 {
            v.try_push(i).unwrap();
        }
        let before = v.capacity();
        alloc.drain();
        // The shrink reallocates into a drained budget, so it must fail and
        // leave the vector's capacity and contents untouched.
        v.try_shrink_to_fit().expect_err("shrink should fail");
        assert_eq!(v.capacity(), before);
        assert_eq!(v.len(), 4);
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn shrink_to_reduces_capacity() {
        let mut v = Vec::<i32>::try_with_capacity(16).expect("alloc ok");
        for i in 0..4 {
            v.try_push(i).unwrap();
        }
        v.try_shrink_to(8).expect("shrink ok");
        assert!(v.capacity() < 16);
        assert!(v.capacity() >= 4);
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn shrink_to_below_len_clamps_to_len() {
        let mut v = Vec::<i32>::try_with_capacity(16).expect("alloc ok");
        for i in 0..4 {
            v.try_push(i).unwrap();
        }
        // A target below len clamps up to len: this still reallocates.
        v.try_shrink_to(0).expect("shrink ok");
        assert_eq!(v.len(), 4);
        assert!(v.capacity() < 16);
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn shrink_to_failure_keeps_capacity() {
        let alloc = BudgetedAlloc::new(1);
        let mut v: Vec<i32, BudgetedAlloc> =
            Vec::try_with_capacity_in(16, alloc.clone()).expect("alloc ok");
        for i in 0..4 {
            v.try_push(i).unwrap();
        }
        let before = v.capacity();
        alloc.drain();
        v.try_shrink_to(8).expect_err("shrink should fail");
        assert_eq!(v.capacity(), before);
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn capacity_overflow_detected() {
        let mut v: Vec<i32> = Vec::new();
        let err = v.try_reserve(usize::MAX).expect_err("should overflow");
        assert!(err.is_capacity_overflow());
    }

    #[test]
    fn try_reserve_adaptive_falls_back_to_exact_when_amortized_refused() {
        use olive_test_helpers::allocators::ByteCapAlloc;
        let mut v: Vec<i32, ByteCapAlloc> =
            Vec::try_with_capacity_in(15, ByteCapAlloc::new(64)).expect("should create vector");
        v.try_extend(0..15).expect("capacity is sufficient");
        v.try_reserve_adaptive(1)
            .expect("exact reserve should succeed");
        assert!(v.capacity() >= 16);
    }

    #[test]
    fn try_reserve_adaptive_succeeds_via_amortized_path_when_affordable() {
        // With a generous cap the first (amortized) attempt succeeds outright, so no
        // fallback is needed and the result still holds the requested room.
        let mut v: Vec<i32> = Vec::new();
        v.try_reserve_adaptive(16).expect("affordable reserve");
        assert!(v.capacity() >= 16);
    }

    #[test]
    fn split_off_middle() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let right = v.try_split_off(2).unwrap();
        assert_eq!(v.as_slice(), &[0, 1]);
        assert_eq!(right.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn split_off_at_zero() {
        let mut v = Vec::new();
        for i in 0..3i32 {
            v.try_push(i).unwrap();
        }
        let right = v.try_split_off(0).unwrap();
        assert!(v.is_empty());
        assert_eq!(right.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn split_off_at_len_returns_empty() {
        let mut v = Vec::new();
        for i in 0..3i32 {
            v.try_push(i).unwrap();
        }
        let right = v.try_split_off(3).unwrap();
        assert_eq!(v.as_slice(), &[0, 1, 2]);
        assert!(right.is_empty());
    }

    #[test]
    fn split_off_on_empty_vec() {
        let mut v: Vec<i32> = Vec::new();
        let right = v.try_split_off(0).unwrap();
        assert!(v.is_empty());
        assert!(right.is_empty());
    }

    #[test]
    fn split_off_out_of_bounds_errors() {
        let mut v = Vec::new();
        for i in 0..3i32 {
            v.try_push(i).unwrap();
        }
        let err = v.try_split_off(4).unwrap_err();
        assert!(matches!(
            err,
            TryVecSplitOffError::OutOfBounds { index: 4, len: 3 }
        ));
        // vec unchanged on error
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn split_off_moves_all_elements_exactly_once() {
        let ledger = Arc::new(Ledger::new());
        let budget = CloneBudget::new(u32::MAX);

        let mut v: Vec<FlakyCloneTrackedItem> = Vec::new();
        for i in 0..6u32 {
            v.try_push(FlakyCloneTrackedItem::with_id(&ledger, i, budget.share()))
                .unwrap();
        }
        let right = v.try_split_off(3).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(right.len(), 3);
        // The split only moved items; nothing was dropped. All six ids still live.
        assert!(ledger.drop_counts().is_empty());
        assert_eq!(ledger.live_ids(), [0, 1, 2, 3, 4, 5]);
        drop(v); // drops ids 0, 1, 2
        assert_eq!(ledger.live_ids(), [3, 4, 5]);
        assert!(ledger.double_dropped().is_empty());
        drop(right); // drops ids 3, 4, 5
        // Every element dropped exactly once; nothing leaked.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..6));
    }

    #[test]
    fn split_off_zst() {
        #[derive(Debug, PartialEq)]
        struct Unit;
        let mut v: Vec<Unit, FailAlloc> = Vec::new_in(FailAlloc);
        for _ in 0..3 {
            v.try_push(Unit).unwrap();
        }
        let right = v.try_split_off(1).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(right.len(), 2);
    }

    #[test]
    fn split_off_oom_errors_and_leaves_vec_intact() {
        let alloc = BudgetedAlloc::new(1);
        let mut v: Vec<i32, BudgetedAlloc> =
            Vec::try_with_capacity_in(4, alloc.clone()).expect("alloc ok");
        for i in 0..3 {
            v.try_push(i).unwrap();
        }
        let before = v.capacity();
        alloc.drain();
        // Cloning the allocator succeeds, but the allocation for the right
        // half fails, so the split must report the error and leave `v` alone.
        let err = v.try_split_off(2).expect_err("oom");
        assert!(matches!(err, TryVecSplitOffError::Reserve(_)));
        assert_eq!(v.len(), 3);
        assert_eq!(v.capacity(), before);
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn extend_from_within_basic() {
        let mut v = Vec::new();
        for i in [1, 2, 3, 4, 5] {
            v.try_push(i).unwrap();
        }
        v.try_extend_from_within(1..3).unwrap();
        // Appended clones of elements at indices 1 and 2 (values 2, 3)
        assert_eq!(v.as_slice(), &[1, 2, 3, 4, 5, 2, 3]);
    }

    #[test]
    fn extend_from_within_entire_vec() {
        let mut v = Vec::new();
        for i in [10, 20, 30] {
            v.try_push(i).unwrap();
        }
        v.try_extend_from_within(..).unwrap();
        assert_eq!(v.as_slice(), &[10, 20, 30, 10, 20, 30]);
    }

    #[test]
    fn extend_from_within_empty_range_is_noop() {
        let mut v = Vec::new();
        for i in [1, 2, 3] {
            v.try_push(i).unwrap();
        }
        v.try_extend_from_within(1..1).unwrap();
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn extend_from_within_out_of_bounds_errors() {
        let mut v = Vec::new();
        for i in [1, 2, 3] {
            v.try_push(i).unwrap();
        }
        let err = v.try_extend_from_within(1..5).unwrap_err();
        assert!(matches!(
            err,
            TryVecExtendFromWithinError::InvalidRange { .. }
        ));
        // vec unchanged on error
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn extend_from_within_reversed_range_errors() {
        let mut v = Vec::new();
        for i in [1, 2, 3] {
            v.try_push(i).unwrap();
        }
        // Reversed range must be rejected (lint suppressed: this is the point of the test).
        #[allow(
            clippy::reversed_empty_ranges,
            reason = "intentionally testing a reversed range"
        )]
        let err = v.try_extend_from_within(2..1).unwrap_err();
        assert!(matches!(
            err,
            TryVecExtendFromWithinError::InvalidRange { .. }
        ));
    }

    #[test]
    fn extend_from_within_with_open_ranges() {
        let mut v = Vec::new();
        for i in [0, 1, 2, 3, 4] {
            v.try_push(i).unwrap();
        }
        // `..2` means 0..2 → appends [0, 1]
        v.try_extend_from_within(..2).unwrap();
        assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4, 0, 1]);
        // Fresh vec for second assertion to avoid confusion with grown length.
        let mut w = Vec::new();
        for i in [0, 1, 2, 3, 4] {
            w.try_push(i).unwrap();
        }
        // `3..` means 3..5 → appends [3, 4]
        w.try_extend_from_within(3..).unwrap();
        assert_eq!(w.as_slice(), &[0, 1, 2, 3, 4, 3, 4]);
    }

    #[test]
    fn extend_from_within_grows_capacity() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push(99).unwrap();
        // Extend with itself: [99] → [99, 99]
        v.try_extend_from_within(..).unwrap();
        assert_eq!(v.as_slice(), &[99, 99]);
        // Again: [99, 99] → [99, 99, 99, 99]
        v.try_extend_from_within(..).unwrap();
        assert_eq!(v.as_slice(), &[99, 99, 99, 99]);
    }

    #[test]
    fn extend_from_within_oom_errors_and_leaves_vec_intact() {
        let alloc = BudgetedAlloc::new(1);
        let mut v: Vec<i32, BudgetedAlloc> =
            Vec::try_with_capacity_in(4, alloc.clone()).expect("alloc ok");
        for i in 0..3 {
            v.try_push(i).unwrap();
        }
        let before = v.capacity();
        alloc.drain();
        // Both the amortized and the exact reserve fall into the drained
        // budget, so the extend must fail and leave the vector untouched.
        let err = v.try_extend_from_within(0..3).expect_err("oom");
        assert!(matches!(err, TryVecExtendFromWithinError::Reserve(_)));
        assert_eq!(v.len(), 3);
        assert_eq!(v.capacity(), before);
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    /// Regression test: if `try_clone` panics mid-loop (after some elements have
    /// already been appended), the rollback guard must truncate the vector back to
    /// its original length so every appended element is dropped exactly once when
    /// the Vec unwinds — no leaks, no double-frees. Before the guard fix, a panic
    /// after a partial append would leak the already-cloned tail.
    ///
    /// Uses the shared [`Ledger`] helper to track individual payload ids, enabling
    /// precise verification of both leaks and double-frees.
    #[test]
    fn extend_from_within_panic_is_safe() {
        struct PanickingItem {
            pub id: u32,
            pub ledger: Arc<Ledger>,
            pub budget: CloneBudget,
        }

        impl Drop for PanickingItem {
            fn drop(&mut self) {
                self.ledger.unregister(self.id);
            }
        }

        impl TryClone for PanickingItem {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                if !self.budget.try_consume() {
                    panic!("clone exhausted")
                }
                let id = self.ledger.allocate();
                self.ledger.register(id);
                Ok(PanickingItem {
                    id,
                    ledger: self.ledger.clone(),
                    budget: self.budget.share(),
                })
            }
        }

        let ledger = Arc::new(Ledger::new());
        let budget = CloneBudget::new(2);

        let mut v: Vec<PanickingItem> = Vec::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            || {
                // Push 4 initial items (ids 0..=3).
                for _ in 0..4 {
                    let id = ledger.allocate();
                    ledger.register(id);
                    v.try_push(PanickingItem {
                        id,
                        ledger: ledger.clone(),
                        budget: budget.share(),
                    })
                    .unwrap();
                }
                // Range 0..4: clones source elements 0–3. Clone calls produce ids
                // 4, 5, 6, …. Id 4 succeeds (append #1), id 5 succeeds (append
                // #2), id 6 triggers the panic. Two appends are outstanding when
                // we unwind through the rollback guard.
                let _ = v.try_extend_from_within(0..4);
            }
        }));

        assert_eq!(v.len(), 4);
        assert!(result.is_err(), "expected try_clone to panic");
        drop(v);

        // After the catch and drop, all Payloads constructed inside the
        // closure have been destroyed.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        // Exactly 6 ids were ever allocated (4 originals + 2 successful clones).
        assert_eq!(ledger.total_allocated(), 6);
        assert!(ledger.all_dropped_once(0..6));
    }
}
