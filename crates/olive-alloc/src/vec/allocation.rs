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
