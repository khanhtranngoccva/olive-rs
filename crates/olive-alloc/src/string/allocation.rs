//! Module that pertains to any modifying functions of [`String`] that allocate.

use core::{fmt, ptr};

use crate::alloc::Global;

use super::String;
use olive_core::slice::{TrySliceRangeError, try_range};
use olive_core::try_traits::{TryClone, TryCloneError};
use olive_core::{alloc::Allocator, alloc_errors::TryReserveError};

// ──────────────────────────────────────────────────────────────────────────────
// Error types
// ──────────────────────────────────────────────────────────────────────────────

/// Error returned by [`String::try_insert_str`].
///
/// Insertion can fail either because the requested position is not a valid
/// character-boundary offset (including being past the end of the string), or
/// because growing the buffer failed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TryStringInsertError {
    /// The index was greater than the string's length or landed in the middle
    /// of a multi-byte character. Carries the offending index and the string's
    /// current length for diagnostic purposes.
    NotCharBoundary {
        /// The byte index that was attempted.
        index: usize,
        /// The string's current length at the time of the call.
        len: usize,
    },
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
}

impl fmt::Debug for TryStringInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCharBoundary { index, len } => f
                .debug_struct("TryStringInsertError::NotCharBoundary")
                .field("index", index)
                .field("len", len)
                .finish(),
            Self::Reserve(e) => f
                .debug_tuple("TryStringInsertError::Reserve")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryStringInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCharBoundary { index, len } => {
                write!(
                    f,
                    "insertion index {index} is not on a character boundary (string length {len})"
                )
            }
            Self::Reserve(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for TryStringInsertError {}

impl From<TryReserveError> for TryStringInsertError {
    #[inline]
    fn from(e: TryReserveError) -> Self {
        Self::Reserve(e)
    }
}

impl TryClone for TryStringInsertError {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

/// Error returned by [`String::try_extend_from_within`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TryStringExtendFromWithinError {
    /// The resolved range exceeded the string's length, invalid,
    /// or otherwise not on the character boundary of the string.
    InvalidRange {
        /// The computed start of the range.
        start: usize,
        /// The computed end of the range (noninclusive).
        end: usize,
        /// The string's current length.
        len: usize,
    },
    /// Resolving the range results in an overflow.
    RangeOverflow,
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
}

impl fmt::Debug for TryStringExtendFromWithinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { start, end, len } => f
                .debug_struct("TryStringExtendFromWithinError::InvalidRange")
                .field("start", start)
                .field("end", end)
                .field("len", len)
                .finish(),
            Self::Reserve(e) => f
                .debug_tuple("TryStringExtendFromWithinError::Reserve")
                .field(e)
                .finish(),
            Self::RangeOverflow => f
                .debug_tuple("TryStringExtendFromWithinError::RangeOverflow")
                .finish(),
        }
    }
}

impl fmt::Display for TryStringExtendFromWithinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { start, end, len } => write!(
                f,
                "range [{start}, {end}) is out of bounds, not on char boundaries, or otherwise invalid for string of length {len}"
            ),
            Self::Reserve(e) => write!(f, "{e}"),
            Self::RangeOverflow => write!(f, "arithmetic overflow while resolving range bounds"),
        }
    }
}

impl core::error::Error for TryStringExtendFromWithinError {}

impl From<TryReserveError> for TryStringExtendFromWithinError {
    #[inline]
    fn from(e: TryReserveError) -> Self {
        Self::Reserve(e)
    }
}

impl TryClone for TryStringExtendFromWithinError {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

/// Error returned by [`String::try_replace_range`].
///
/// Replacement can fail because the range is invalid / misaligned, or because
/// growing the buffer to fit the replacement failed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TryStringReplaceRangeError {
    /// The resolved range endpoints were out of order, out of bounds, or not
    /// on char boundaries. Carries the offending `(start, end)` pair.
    InvalidRange {
        /// Start of the invalid range.
        start: usize,
        /// End of the invalid range.
        end: usize,
        /// The string's current length.
        len: usize,
    },
    /// An arithmetic overflow occurred while resolving an excluded/unbounded
    /// range edge (e.g. `Excluded(usize::MAX)`).
    RangeOverflow,
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
}

impl fmt::Debug for TryStringReplaceRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { start, end, len } => f
                .debug_struct("TryStringReplaceRangeError::InvalidRange")
                .field("start", start)
                .field("end", end)
                .field("len", len)
                .finish(),
            Self::RangeOverflow => f
                .debug_tuple("TryStringReplaceRangeError::Overflow")
                .finish(),
            Self::Reserve(e) => f
                .debug_tuple("TryStringReplaceRangeError::Reserve")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryStringReplaceRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { start, end, len } => {
                write!(
                    f,
                    "replacement range [{start}, {end}) is invalid or not aligned to char boundaries (string length is {len})"
                )
            }
            Self::RangeOverflow => write!(f, "arithmetic overflow while resolving range bounds"),
            Self::Reserve(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for TryStringReplaceRangeError {}

impl From<TryReserveError> for TryStringReplaceRangeError {
    #[inline]
    fn from(e: TryReserveError) -> Self {
        Self::Reserve(e)
    }
}

impl TryClone for TryStringReplaceRangeError {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

/// Error returned by [`String::try_split_off`].
///
/// Splitting can fail because the split point is misaligned/out of range, or
/// because allocating the right-hand half failed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TryStringSplitOffError {
    /// The split offset was out of bounds or not on a char boundary.
    NotCharBoundary {
        /// The requested split offset.
        index: usize,
        /// The string's current length at the time of the call.
        len: usize,
    },
    /// A capacity reservation for the new string failed (overflow or OOM).
    Reserve(TryReserveError),
}

impl fmt::Debug for TryStringSplitOffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCharBoundary { index, len } => f
                .debug_struct("TryStringSplitOffError::NotCharBoundary")
                .field("index", index)
                .field("len", len)
                .finish(),
            Self::Reserve(e) => f
                .debug_tuple("TryStringSplitOffError::Reserve")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryStringSplitOffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCharBoundary { index, len } => write!(
                f,
                "cannot split off at byte {index}: not on a character boundary (string length is {len})"
            ),
            Self::Reserve(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for TryStringSplitOffError {}

impl TryClone for TryStringSplitOffError {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Reserve and shrink block
// ──────────────────────────────────────────────────────────────────────────────

impl<A: Allocator> String<A> {
    /// Reserves additional capacity so the string can hold at least
    /// `additional` more bytes beyond its current length.
    ///
    /// This is the primary fallible entry point: it surfaces a
    /// [`TryReserveError`] directly rather than wrapping it.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.buf.try_reserve(additional)
    }

    /// Reserves the minimum capacity for exactly `additional` more bytes.
    ///
    /// Unlike [`try_reserve`](Self::try_reserve), this does not adhere to any
    /// exponential growth strategy and may interfere with it, typically
    /// resulting in extra allocations. However, this prevents overshooting
    /// which may fail.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.buf.try_reserve_exact(additional)
    }

    /// Ensures the string has room for at least `additional` more bytes,
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
    /// Use this for bulk operations (e.g. appending a known-length fragment)
    /// where reserving up front is desirable but a refusal of the
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

    /// Ensures the string has room for at least `total` bytes *in total*
    /// (an absolute target, not an increment).
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the new capacity overflows or the
    /// allocation fails.
    pub fn try_reserve_total(&mut self, total: usize) -> Result<(), TryReserveError> {
        self.buf.try_reserve_total(total)
    }

    /// Shrinks the string's capacity down to fit its length.
    ///
    /// Note that the actual capacity may still be more than the current length.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the reallocation fails.
    /// The string is still safe to use regardless whether the operation failed or succeeded.
    pub fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError> {
        self.buf.try_shrink_to_fit()
    }

    /// Shrinks the string's capacity down to be at least `target`. The actual content of
    /// the string is unaffected. If `target` is smaller than current length, the
    /// operation does nothing.
    ///
    /// Note that the actual capacity may still be more than `target` after shrinking.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the reallocation fails.
    /// The string is still safe to use regardless whether the operation failed or succeeded.
    pub fn try_shrink_to(&mut self, target: usize) -> Result<(), TryReserveError> {
        self.buf.try_shrink_to(target)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Push block
// ──────────────────────────────────────────────────────────────────────────────

impl<A: Allocator> String<A> {
    /// Appends a string slice to the string, growing the buffer if needed.
    ///
    /// The input is taken as a concrete `&str`, which guarantees valid UTF-8 by
    /// construction — no fallible conversion step is involved. For pushing a
    /// single character, use [`try_push`](Self::try_push).
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails.
    pub fn try_push_str(&mut self, s: &str) -> Result<(), TryReserveError> {
        if s.is_empty() {
            return Ok(());
        }
        self.push_str_inner(s)
    }

    /// Appends a single character to the string, growing the buffer if needed.
    ///
    /// This is the `char` counterpart of [`try_push_str`](Self::try_push_str):
    /// encoding one scalar into its UTF-8 bytes can never fail, so this only
    /// ever errors on a failed growth.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails.
    pub fn try_push(&mut self, c: char) -> Result<(), TryReserveError> {
        let mut buf = [0u8; 4];
        let encoded = c.encode_utf8(&mut buf);
        self.push_str_inner(encoded)
    }

    /// Shared append path for a validated UTF-8 fragment.
    pub(super) fn push_str_inner(&mut self, s: &str) -> Result<(), TryReserveError> {
        // Sanity guard to ensure there is something to push =>
        // len < capacity and prevent dst from overflowing
        if s.is_empty() {
            return Ok(());
        }
        self.try_reserve_adaptive(s.len())?;
        // SAFETY: the reservation above guarantees `s.len()` spare bytes, and
        // `s` is valid UTF-8, so appending it preserves the invariant.
        unsafe {
            // asserted len < capacity, cannot overflow.
            ptr::copy_nonoverlapping(s.as_ptr(), self.buf.as_mut_ptr().add(self.len()), s.len());
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted total <= capacity after successful reserve"
            )]
            {
                self.buf.set_len(self.len() + s.len());
            }
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Insert block
// ──────────────────────────────────────────────────────────────────────────────

impl<A: Allocator> String<A> {
    /// Inserts a char at the byte index `index`, shifting subsequent
    /// bytes rightward.
    ///
    /// Both failure modes are reported through the returned error rather than
    /// unwinding: an index outside the string or landing mid-character is
    /// [`TryStringInsertError::NotCharBoundary`], and a failed growth is
    /// [`TryStringInsertError::Reserve`].
    ///
    /// # Errors
    ///
    /// Returns [`TryStringInsertError`] if the index is invalid or growing the
    /// buffer fails.
    pub fn try_insert(&mut self, index: usize, c: char) -> Result<(), TryStringInsertError> {
        let mut buf = [0u8; 4];
        let encoded = c.encode_utf8(&mut buf);
        self.try_insert_str(index, encoded)
    }

    /// Inserts a string slice at the byte index `index`, shifting subsequent
    /// bytes rightward.
    ///
    /// Both failure modes are reported through the returned error rather than
    /// unwinding: an index outside the string or landing mid-character is
    /// [`TryStringInsertError::NotCharBoundary`], and a failed growth is
    /// [`TryStringInsertError::Reserve`].
    ///
    /// # Errors
    ///
    /// Returns [`TryStringInsertError`] if the index is invalid or growing the
    /// buffer fails.
    pub fn try_insert_str(&mut self, index: usize, s: &str) -> Result<(), TryStringInsertError> {
        let len = self.len();
        if index > len || !self.is_char_boundary(index) {
            return Err(TryStringInsertError::NotCharBoundary { index, len });
        }

        // Sanity guard to ensure len < capacity and prevent dst from overflowing
        if s.is_empty() {
            return Ok(());
        }

        let self_len = self.len();
        let s_len = s.len();

        self.try_reserve_adaptive(s_len)
            .map_err(TryStringInsertError::Reserve)?;

        // SAFETY: after `reserve`, `[0..self_len + s_len)` is valid writable
        // memory. We shift the tail right by `s_len`, then copy `s` into the
        // vacated slot.
        // SAFETY: `index <= self_len` was checked above, so `self_len - index`
        // cannot underflow. After `reserve`, the destination range is valid.
        unsafe {
            let base = self.buf.as_mut_ptr();
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "index <= self_len < capactity checked above"
            )]
            {
                ptr::copy(base.add(index), base.add(index + s_len), self_len - index);
            }
            ptr::copy_nonoverlapping(s.as_ptr(), base.add(index), s_len);
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted total <= capacity after successful reserve"
            )]
            {
                self.buf.set_len(self_len + s_len);
            }
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Allocating manipulation block
// ──────────────────────────────────────────────────────────────────────────────

impl<A: Allocator> String<A> {
    /// Splits the string into two at byte index `at`, keeping the portion
    /// before `at` in `self` and returning the portion from `at` onward as a
    /// new `String` allocated through the global allocator.
    ///
    /// This is the fallible-port analogue of std's `String::split_off`.
    ///
    /// # Errors
    ///
    /// Returns [`TryStringSplitOffError::NotCharBoundary`] if `at` is out of
    /// bounds or not on a character boundary, or
    /// [`TryStringSplitOffError::Reserve`] if the allocation for the
    /// right-hand half fails.
    pub fn try_split_off(&mut self, at: usize) -> Result<String, TryStringSplitOffError> {
        self.try_split_off_in(at, Global)
    }

    /// Splits the string into two at byte index `at`, keeping the portion
    /// before `at` in `self` and returning the portion from `at` onward as a
    /// new `String` allocated through `alloc`.
    ///
    /// # Errors
    ///
    /// Returns [`TryStringSplitOffError::NotCharBoundary`] if `at` is out of
    /// bounds or not on a character boundary, or
    /// [`TryStringSplitOffError::Reserve`] if the allocation for the
    /// right-hand half fails.
    pub fn try_split_off_in<A2: Allocator>(
        &mut self,
        at: usize,
        alloc: A2,
    ) -> Result<String<A2>, TryStringSplitOffError> {
        let len = self.len();
        if at > len || !self.is_char_boundary(at) {
            return Err(TryStringSplitOffError::NotCharBoundary { index: at, len });
        }
        let tail_bytes = self[at..].as_bytes();
        // Fast path: splitting at the end produces an empty tail — no copy needed.
        if tail_bytes.is_empty() {
            return Ok(String::<A2>::new_in(alloc));
        }
        let mut out = String::<A2>::try_with_capacity_in(tail_bytes.len(), alloc)
            .map_err(TryStringSplitOffError::Reserve)?;
        let dst = out.buf.as_mut_ptr();
        // SAFETY: `out` was created with capacity >= `tail_bytes.len()` and is
        // empty; copying that many bytes is in-bounds and `set_len` to the same
        // count is valid.
        unsafe {
            ptr::copy_nonoverlapping(tail_bytes.as_ptr(), dst, tail_bytes.len());
            out.buf.set_len(tail_bytes.len());
        }
        self.buf.truncate(at);
        Ok(out)
    }

    /// Extends this string with a subslice of itself identified by `indices`.
    ///
    /// # Errors
    ///
    /// Returns [`TryStringExtendFromWithinError::InvalidRange`] if the resolved
    /// range exceeds the string's length, or
    /// [`TryStringExtendFromWithinError::Reserve`] if reserving room for the
    /// extra bytes fails.
    /// Returns [`TryStringExtendFromWithinError::NotCharBoundary`] if either
    /// endpoint falls inside a multi-byte `char`.
    pub fn try_extend_from_within<R: core::ops::RangeBounds<usize>>(
        &mut self,
        indices: R,
    ) -> Result<(), TryStringExtendFromWithinError> {
        let len = self.len();
        // Resolve the bounds into a concrete range, surfacing overflow and
        // ordering violations as distinct error variants.
        let r = try_range(indices, ..len).map_err(|e| match e {
            TrySliceRangeError::StartOverflow | TrySliceRangeError::EndOverflow => {
                TryStringExtendFromWithinError::RangeOverflow
            }
            TrySliceRangeError::StartExceedsEnd { start, end, len } => {
                TryStringExtendFromWithinError::InvalidRange { start, end, len }
            }
            TrySliceRangeError::EndExceedsBound { start, end, len } => {
                TryStringExtendFromWithinError::InvalidRange { start, end, len }
            }
        })?;
        let start = r.start;
        let end = r.end;
        if !self.is_char_boundary(start) {
            return Err(TryStringExtendFromWithinError::InvalidRange { start, end, len });
        }
        if !self.is_char_boundary(end) {
            return Err(TryStringExtendFromWithinError::InvalidRange { start, end, len });
        }
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start <= end (guaranteed by try_range)"
        )]
        let count = end - start;
        if count == 0 {
            return Ok(());
        }
        // Reserve first so that any reallocation happens before we take the
        // source pointer. After reserving, the data is intact at the same
        // logical offsets, and the destination (at offset `len`) does not
        // overlap the source (which ends at most at `len`).
        self.try_reserve_adaptive(count)
            .map_err(TryStringExtendFromWithinError::Reserve)?;
        let base = self.buf.as_mut_ptr();
        // SAFETY: `base + start` and `base + len` are within the allocation;
        // the source `[start..end)` and destination `[len..len+count)` do not
        // overlap because `end <= len`. Writing `count` bytes at offset `len`
        // is in-bounds (just reserved), and `set_len` to `len + count` matches.
        unsafe {
            ptr::copy_nonoverlapping(base.add(start), base.add(len), count);
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "reserve succeeded so no overflow"
            )]
            {
                self.buf.set_len(len + count);
            }
        }
        Ok(())
    }

    /// Replaces the substring in `range` with `replacement`, growing the buffer
    /// as needed.
    ///
    /// This is the fallible-port analogue of std's `String::replace_range`.
    /// The range accepts any `RangeBounds<usize>` (e.g. `0..3`, `..2`, `1..`,
    /// `..`). Both resolved endpoints must be on character boundaries and in
    /// order.
    ///
    /// # Errors
    ///
    /// Returns [`TryStringReplaceRangeError::InvalidRange`] if the resolved
    /// range is out of bounds, not on char boundaries, or reversed,
    /// [`TryStringReplaceRangeError::Overflow`] if an unbounded edge overflows
    /// when converted, or [`TryStringReplaceRangeError::Reserve`] if growth
    /// fails.
    pub fn try_replace_range<R: core::ops::RangeBounds<usize>>(
        &mut self,
        range: R,
        replacement: &str,
    ) -> Result<(), TryStringReplaceRangeError> {
        let len = self.len();
        // Resolve the bounds into a concrete range, surfacing overflow and
        // ordering violations as distinct error variants.
        let r = try_range(range, ..len).map_err(|e| match e {
            TrySliceRangeError::StartOverflow | TrySliceRangeError::EndOverflow => {
                TryStringReplaceRangeError::RangeOverflow
            }
            TrySliceRangeError::StartExceedsEnd { start, end, .. } => {
                TryStringReplaceRangeError::InvalidRange { start, end, len }
            }
            TrySliceRangeError::EndExceedsBound { start, end, .. } => {
                TryStringReplaceRangeError::InvalidRange { start, end, len }
            }
        })?;
        let start = r.start;
        let end = r.end;
        // The range is in-bounds and ordered; the only remaining validation is
        // that both endpoints sit on character boundaries.
        if !self.is_char_boundary(start) || !self.is_char_boundary(end) {
            return Err(TryStringReplaceRangeError::InvalidRange { start, end, len });
        }
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start <= end (checked above)"
        )]
        let removed = end - start;
        let added = replacement.len();
        let delta = added.saturating_sub(removed);
        if delta > 0 {
            self.try_reserve_adaptive(delta)?;
        }
        let ptr = self.buf.as_mut_ptr();
        // Move the trailing segment `[end..len]` to make room, accounting for
        // the net length change. If shrinking, it moves left; if growing, right.
        let tail_start = end;
        #[allow(clippy::arithmetic_side_effects, reason = "end <= len (checked above)")]
        let tail_len = len - end;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "reserve succeeded, asserted start + added <= capacity, in bounds"
        )]
        let dest = start + added;
        if tail_len > 0 && dest != tail_start {
            // SAFETY: overlapping shift within the (possibly grown) buffer;
            // `ptr::copy` handles both left and right shifts.
            unsafe {
                ptr::copy(ptr.add(tail_start), ptr.add(dest), tail_len);
            }
        }
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "removed <= len and reserve succeeded"
        )]
        let new_len = len - removed + added;
        // SAFETY: the tail was shifted to make room and `delta` bytes were
        // reserved above when growing; writing `added` bytes at `start` is
        // in-bounds, and the final length equals the validated `new_len`.
        unsafe {
            if added > 0 {
                ptr::copy_nonoverlapping(replacement.as_ptr(), ptr.add(start), added);
            }
            self.buf.set_len(new_len);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::ops::Deref;

    use super::*;
    use crate::alloc::CountingAllocator;
    use olive_test_helpers::{BudgetedAlloc, FailAlloc};

    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    // ── try_reserve / try_shrink_to_fit ───────────────────────────────────────

    #[test]
    fn reserve_and_shrink() {
        let mut s = mk("abc");
        s.try_reserve(64).unwrap();
        assert!(s.capacity() >= 67); // len 3 + 64 additional
        s.try_shrink_to_fit().unwrap();
        assert!(s.capacity() < 64);
        assert_eq!(s.deref(), "abc");
    }

    // FIXME: shrink_to should be tested, with tests that makes input
    // smaller than length and larger than current cap (no-op)

    #[test]
    fn reserve_exact_and_total_and_adaptive() {
        let mut s = String::new();

        // `try_reserve_exact` reserves at least the requested additional bytes.
        s.try_reserve_exact(10).unwrap();
        assert!(s.capacity() >= 10);

        // `try_reserve_adaptive` grows amortized: with capacity 10 and length
        // 0, it reserves `max(cap * 2, len + 32)` = 32, not `before + 32`.
        s.try_reserve_adaptive(32).unwrap();
        assert!(s.capacity() >= 32);

        // `try_reserve_total` reserves at least the given total capacity.
        s.try_reserve_total(64).unwrap();
        assert!(s.capacity() >= 64);

        assert!(s.is_empty());
    }

    #[test]
    fn oom_on_reserve_returns_error() {
        let mut s = mk("tiny");
        // Deterministic failure: no allocator can satisfy a `usize::MAX` request.
        let err = s.try_reserve(usize::MAX).unwrap_err();
        assert!(err.is_capacity_overflow());
        // The string must be untouched after a failed reservation.
        assert_eq!(s.deref(), "tiny");
    }

    #[test]
    fn budgeted_reserve_fails_when_budget_drained() {
        // One successful allocation builds the string; the drained budget then
        // deterministically refuses the growth.
        let alloc = BudgetedAlloc::new(1);
        let mut s = String::try_with_capacity_in(8, alloc.try_clone().unwrap()).unwrap();
        s.try_push_str("abc").unwrap();

        alloc.drain();
        let err = s.try_reserve(64).unwrap_err();
        assert!(err.is_alloc());

        assert_eq!(alloc.remaining(), 0);
        assert_eq!(s.deref(), "abc");
        assert_eq!(s.capacity(), 8);
    }

    #[test]
    fn budgeted_reserve_exact_fails_when_budget_drained() {
        // One successful allocation builds the string; the drained budget then
        // deterministically refuses the growth.
        let alloc = BudgetedAlloc::new(1);
        let mut s = String::try_with_capacity_in(8, alloc.try_clone().unwrap()).unwrap();
        s.try_push_str("abc").unwrap();

        alloc.drain();
        let err = s.try_reserve_exact(64).unwrap_err();
        assert!(err.is_alloc());

        assert_eq!(alloc.remaining(), 0);
        assert_eq!(s.deref(), "abc");
        assert_eq!(s.capacity(), 8);
    }

    #[test]
    fn budgeted_reserve_adaptive_fails_then_succeeds_after_refund() {
        let alloc = BudgetedAlloc::new(1);
        let mut s = String::try_with_capacity_in(8, alloc.try_clone().unwrap()).unwrap();
        s.try_push_str("abc").unwrap();

        // With no budget left, both the amortized and the exact fallback need
        // an allocation, so the adaptive reservation fails.
        alloc.drain();
        let err = s.try_reserve_adaptive(64).unwrap_err();
        assert!(err.is_alloc());
        assert_eq!(s.deref(), "abc");
        assert_eq!(s.capacity(), 8);

        // Restoring a single unit lets the amortized attempt spend it and
        // succeed.
        alloc.set_budget(1);
        s.try_reserve_adaptive(64).unwrap();
        assert!(s.capacity() >= 67); // len 3 + 64 additional
        assert_eq!(alloc.remaining(), 0);
    }

    // FIXME: use a ByteCapAlloc to force an exact resize for adaptive reserve

    #[test]
    fn budgeted_shrink_to_fit_fails_and_preserves_string() {
        // Shrinking from 64 down to len 3 reallocates, so a drained budget
        // makes it fail.
        let alloc = BudgetedAlloc::new(1);
        let mut s = String::try_with_capacity_in(64, alloc.try_clone().unwrap()).unwrap();
        s.try_push_str("abc").unwrap();

        alloc.drain();
        let err = s.try_shrink_to_fit().unwrap_err();
        assert!(err.is_alloc());
        // The string is still safe to use after a failed shrink: content and
        // capacity are untouched.
        assert_eq!(s.deref(), "abc");
        assert_eq!(s.capacity(), 64);

        // Refunding budget lets the same shrink succeed.
        alloc.set_budget(1);
        s.try_shrink_to_fit().unwrap();
        assert!(s.capacity() < 64);
        assert_eq!(s.deref(), "abc");
    }

    #[test]
    fn budgeted_shrink_to_fails_and_preserves_string() {
        let alloc = BudgetedAlloc::new(1);
        let mut s = String::try_with_capacity_in(64, alloc.try_clone().unwrap()).unwrap();
        s.try_push_str("abcdef").unwrap();

        alloc.drain();
        let err = s.try_shrink_to(8).unwrap_err();
        assert!(err.is_alloc());
        assert_eq!(s.deref(), "abcdef");
        assert_eq!(s.capacity(), 64);

        // Refunding budget lets the same shrink succeed.
        alloc.set_budget(1);
        s.try_shrink_to(10).unwrap();
        assert!(s.capacity() >= 10);
        assert!(s.capacity() < 64);
        assert_eq!(s.deref(), "abc");
    }

    // ── try_push / try_push_str ───────────────────────────────────────────────

    #[test]
    fn push_char_grows() {
        let mut s = String::new();
        s.try_push('a').unwrap();
        s.try_push('é').unwrap();
        s.try_push('🦊').unwrap();
        assert_eq!(s.deref(), "aé🦊");
    }

    #[test]
    fn push_str_appends() {
        let mut s = mk("foo");
        s.try_push_str("-bar").unwrap();
        assert_eq!(s.deref(), "foo-bar");
    }

    #[test]
    fn push_char_reports_reservation_failure() {
        let mut s: String<FailAlloc> = String::new_in(FailAlloc);
        let err = s.try_push('a').unwrap_err();
        assert!(err.is_alloc());
        assert!(s.is_empty());
    }

    #[test]
    fn push_reports_reservation_failure() {
        let mut s: String<FailAlloc> = String::new_in(FailAlloc);
        let err = s.try_push_str("data").unwrap_err();
        assert!(err.is_alloc());
        assert!(s.is_empty());
    }

    // ── try_insert / try_insert_str ───────────────────────────────────────────

    #[test]
    fn insert_char_middle() {
        let mut s = mk("helo");
        s.try_insert(2, 'l').unwrap();
        assert_eq!(s.deref(), "hello");
    }

    #[test]
    fn insert_str_middle() {
        let mut s = mk("hello world");
        s.try_insert_str(5, ", brave").unwrap();
        assert_eq!(s.deref(), "hello, brave world");
    }

    #[test]
    fn insert_at_end_and_start() {
        let mut s = mk("bc");
        s.try_insert_str(2, "d").unwrap();
        assert_eq!(s.deref(), "bcd");
        s.try_insert(0, 'a').unwrap();
        assert_eq!(s.deref(), "abcd");
    }

    #[test]
    fn insert_rejects_out_of_bounds() {
        let mut s = mk("ab");
        let err = s.try_insert_str(3, "x").unwrap_err();
        assert!(matches!(
            err,
            TryStringInsertError::NotCharBoundary { index: 3, len: 2 }
        ));
        assert_eq!(s.deref(), "ab");
        let err = s.try_insert(3, 'x').unwrap_err();
        assert!(matches!(
            err,
            TryStringInsertError::NotCharBoundary { index: 3, len: 2 }
        ));
        assert_eq!(s.deref(), "ab");
    }

    #[test]
    fn insert_rejects_mid_character() {
        let mut s = mk("é🦊");
        // é occupies bytes [0, 2); byte index 1 is inside it.
        let err = s.try_insert_str(1, "x").unwrap_err();
        assert!(matches!(
            err,
            TryStringInsertError::NotCharBoundary { index: 1, len: 6 }
        ));
        assert_eq!(s.deref(), "é🦊");
        let err = s.try_insert_str(1, "x").unwrap_err();
        assert!(matches!(
            err,
            TryStringInsertError::NotCharBoundary { index: 1, len: 6 }
        ));
        assert_eq!(s.deref(), "é🦊");
    }

    // FIXME: Reserve (OOM) variant not tested for try_insert and try_insert_str

    // ── try_split_off ─────────────────────────────────────────────────────────

    #[test]
    fn try_split_off_middle() {
        let mut s = mk("abcdef");
        let right = s.try_split_off(3).unwrap();
        assert_eq!(s.deref(), "abc");
        assert_eq!(right.deref(), "def");
    }

    #[test]
    fn try_split_off_at_end() {
        let mut s = mk("abcdef");
        let right = s.try_split_off(6).unwrap();
        assert_eq!(s.deref(), "abcdef");
        assert_eq!(right.deref(), "");
    }

    #[test]
    fn try_split_off_at_start() {
        let mut s = mk("abcdef");
        let right = s.try_split_off(0).unwrap();
        assert_eq!(s.deref(), "");
        assert_eq!(right.deref(), "abcdef");
    }

    #[test]
    fn try_split_off_mid_char_fails() {
        let mut s = mk("é🦊");
        // é occupies bytes [0, 2); byte index 1 is inside it.
        let err = s.try_split_off(1).unwrap_err();
        assert!(matches!(
            err,
            TryStringSplitOffError::NotCharBoundary { index: 1, len: 6 }
        ));
        assert_eq!(s.deref(), "é🦊");
    }

    #[test]
    fn try_split_off_past_end_fails() {
        let mut s = mk("abc");
        let err = s.try_split_off(4).unwrap_err();
        assert!(matches!(
            err,
            TryStringSplitOffError::NotCharBoundary { index: 4, len: 3 }
        ));
        assert_eq!(s.deref(), "abc");
    }

    #[test]
    fn try_split_off_in_uses_custom_allocator() {
        let alloc = CountingAllocator::new();
        let mut s: String<&CountingAllocator> = String::try_from_str_in("abcdef", &alloc).unwrap();
        let right = s.try_split_off_in(3, &alloc).unwrap();
        assert_eq!(s.deref(), "abc");
        assert_eq!(right.deref(), "def");
    }

    // FIXME: try_split_off_in should have Reserve/OOM tested

    // ── try_extend_from_within ────────────────────────────────────────────────

    #[test]
    fn extend_from_within_basic() {
        let mut s = mk("hello");
        s.try_extend_from_within(0..5).unwrap();
        assert_eq!(s.deref(), "hellohello");
    }

    #[test]
    fn extend_from_within_full() {
        let mut s = mk("abc");
        s.try_extend_from_within(0..3).unwrap();
        assert_eq!(s.deref(), "abcabc");
    }

    #[test]
    fn extend_from_within_empty_range() {
        let mut s = mk("abc");
        s.try_extend_from_within(1..1).unwrap();
        assert_eq!(s.deref(), "abc");
    }

    #[test]
    fn extend_from_within_rejects_non_char_boundary_start() {
        let mut s = mk("é🦊");
        let err = s.try_extend_from_within(1..=4).unwrap_err();
        assert!(matches!(
            err,
            TryStringExtendFromWithinError::InvalidRange {
                start: 1,
                end: 5,
                len: 6
            }
        ));
        assert_eq!(s.deref(), "é🦊");
    }

    #[test]
    fn extend_from_within_rejects_non_char_boundary_end() {
        let mut s = mk("é🦊");
        let err = s.try_extend_from_within(0..3).unwrap_err();
        assert!(matches!(
            err,
            TryStringExtendFromWithinError::InvalidRange {
                start: 0,
                end: 3,
                len: 6
            }
        ));
        assert_eq!(s.deref(), "é🦊");
    }

    // FIXME: add OOM tests

    // ── try_replace_range ─────────────────────────────────────────────────────

    #[test]
    fn replace_range_shrink() {
        let mut s = mk("hello world");
        s.try_replace_range(5..11, "").unwrap();
        assert_eq!(s.deref(), "hello");
    }

    #[test]
    fn replace_range_grow() {
        let mut s = mk("hello world");
        s.try_replace_range(5..11, ", olive!").unwrap();
        assert_eq!(s.deref(), "hello, olive!");
    }

    #[test]
    fn replace_range_same_len() {
        let mut s = mk("hello");
        s.try_replace_range(0..5, "world").unwrap();
        assert_eq!(s.deref(), "world");
    }

    #[test]
    fn replace_range_multibyte() {
        let mut s = mk("é🦊x");
        s.try_replace_range(0..6, "a").unwrap();
        assert_eq!(s.deref(), "ax");
    }

    #[test]
    fn replace_range_invalid_mid_char() {
        let mut s = mk("é🦊");
        let err = s.try_replace_range(1..4, "x").unwrap_err();
        assert!(matches!(
            err,
            TryStringReplaceRangeError::InvalidRange {
                start: 1,
                end: 4,
                len: 6
            }
        ));
        assert_eq!(s.deref(), "é🦊");
    }

    #[test]
    fn replace_range_out_of_order() {
        let mut s = mk("hello");
        let err = s
            .try_replace_range(core::ops::Range { start: 3, end: 1 }, "x")
            .unwrap_err();
        assert!(matches!(
            err,
            TryStringReplaceRangeError::InvalidRange {
                start: 3,
                end: 1,
                len: 5
            }
        ));
        assert_eq!(s.deref(), "hello");
    }

    #[test]
    fn replace_range_end_oob_reports_start() {
        let mut s = mk("hello");
        let err = s.try_replace_range(4..=8, "x").unwrap_err();
        assert!(matches!(
            err,
            TryStringReplaceRangeError::InvalidRange {
                start: 4,
                end: 9,
                len: 5
            }
        ));
        assert_eq!(s.deref(), "hello");
    }

    // FIXME: add OOM tests for growth case
}
