//! Low-level mutation primitives for [`VecDeque`].
//!
//! These operate *within* existing capacity: they never allocate, and each one
//! reports a full buffer as an error instead of growing. The fallible public
//! push methods in `allocation.rs` build on top of them, securing space first
//! when needed.

use super::VecDeque;
use super::wrapped_index::WrappedIndex;
use olive_core::alloc::Allocator;
use olive_core::mem::size_of;
use olive_core::ptr;

/// Error returned by the within-capacity push primitives when the buffer is
/// exactly full (`len == capacity`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryPushWithinCapacityError {
    /// The current length (equal to capacity).
    pub len: usize,
}

impl core::fmt::Display for TryPushWithinCapacityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "no spare capacity: deque is full at length {}", self.len)
    }
}

impl core::error::Error for TryPushWithinCapacityError {}

// ---------------------------------------------------------------------------
// Within-capacity pushes
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Appends an element to the back of the deque without attempting to grow
    /// the buffer. Succeeds only if there is already spare capacity.
    ///
    /// # Errors
    ///
    /// Returns [`TryPushWithinCapacityError`] if `len == capacity`.
    pub fn try_push_back_within_capacity(
        &mut self,
        value: T,
    ) -> Result<(), TryPushWithinCapacityError> {
        let cap = self.capacity();
        if self.len >= cap {
            return Err(TryPushWithinCapacityError { len: self.len });
        }
        // SAFETY: `len < capacity`, so the slot computed below is in-bounds.
        unsafe { self.push_back_within_cap(value) };
        Ok(())
    }

    /// Prepends an element to the front of the deque without attempting to
    /// grow the buffer. Succeeds only if there is already spare capacity.
    ///
    /// # Errors
    ///
    /// Returns [`TryPushWithinCapacityError`] if `len == capacity`.
    pub fn try_push_front_within_capacity(
        &mut self,
        value: T,
    ) -> Result<(), TryPushWithinCapacityError> {
        let cap = self.capacity();
        if self.len >= cap {
            return Err(TryPushWithinCapacityError { len: self.len });
        }
        // SAFETY: `len < capacity`, so retreating `head` stays in-bounds.
        unsafe { self.push_front_within_cap(value) };
        Ok(())
    }

    /// Writes `value` into the next back slot and advances `len`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `self.len < self.capacity()`: the destination
    /// slot is then provably inside the allocation. For zero-sized types the
    /// slot is always the dangling base pointer, so no index arithmetic is
    /// performed at all.
    #[inline]
    unsafe fn push_back_within_cap(&mut self, value: T) {
        debug_assert!(
            self.len < self.capacity(),
            "push_back_within_cap requires spare capacity"
        );
        if size_of::<T>() == 0 {
            // ZST: every "slot" is the same dangling address; nothing to move.
            unsafe { self.buf.ptr().cast::<T>().write(value) };
        } else {
            // SAFETY: `len < capacity`, so this wraps correctly.
            let idx = unsafe { self.wrap_add(self.head, self.len) };
            // SAFETY: `idx < capacity`, so the slot is writable.
            unsafe { self.buf.ptr().add(idx.as_index()).write(value) };
        }
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted self.len < self.capacity <= usize::MAX"
        )]
        {
            self.len += 1;
        }
    }

    /// Retreats `head` by one, writes `value` there, and advances `len`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `self.len < self.capacity()`: retreating
    /// `head` by one from its current position then stays within `0..capacity`.
    /// For zero-sized types `head` never moves and the write goes to the
    /// dangling base pointer.
    #[inline]
    unsafe fn push_front_within_cap(&mut self, value: T) {
        debug_assert!(
            self.len < self.capacity(),
            "push_front_within_cap requires spare capacity"
        );
        if size_of::<T>() == 0 {
            // ZST: head is pinned at 0; just bump the length.
            unsafe { self.buf.ptr().cast::<T>().write(value) };
        } else {
            // SAFETY: `len < capacity` guarantees there is a free slot before
            // `head`; `wrap_sub` with subtrahend 1 keeps the result in-bounds.
            self.head = unsafe { self.wrap_sub(self.head, 1) };
            // SAFETY: `head < capacity`, so the slot holds writable memory.
            unsafe { self.buf.ptr().add(self.head.as_index()).write(value) };
        }
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted self.len < self.capacity <= usize::MAX"
        )]
        {
            self.len += 1;
        }
    }

    /// Copies a contiguous block of memory `len` long from `src` to `dst`.
    ///
    /// # Safety
    /// - Both `src + len` and `dst + len` must not exceed `self.capacity()`.
    /// - The source and destination ranges must not overlap (use `copy` for
    ///   overlapping regions).
    #[inline]
    #[allow(clippy::arithmetic_side_effects, reason = "debug assertions only")]
    unsafe fn copy_nonoverlapping(&mut self, src: WrappedIndex, dst: WrappedIndex, len: usize) {
        debug_assert!(
            dst.as_index() + len <= self.capacity(),
            "cno dst={} src={} len={} cap={}",
            dst.as_index(),
            src.as_index(),
            len,
            self.capacity()
        );
        debug_assert!(
            src.as_index() + len <= self.capacity(),
            "cno dst={} src={} len={} cap={}",
            dst.as_index(),
            src.as_index(),
            len,
            self.capacity()
        );
        // SAFETY: upheld by caller.
        unsafe {
            ptr::copy_nonoverlapping(
                self.buf.ptr().add(src.as_index()),
                self.buf.ptr().add(dst.as_index()),
                len,
            );
        }
    }

    /// Copies a contiguous block of memory `len` long from `src` to `dst`,
    /// handling overlapping regions correctly (like `memcpy`).
    ///
    /// # Safety
    /// - Both `src + len` and `dst + len` must not exceed `self.capacity()`.
    #[inline]
    #[allow(clippy::arithmetic_side_effects, reason = "debug assertions only")]
    unsafe fn copy(&mut self, src: WrappedIndex, dst: WrappedIndex, len: usize) {
        debug_assert!(
            dst.as_index() + len <= self.capacity(),
            "cpy dst={} src={} len={} cap={}",
            dst.as_index(),
            src.as_index(),
            len,
            self.capacity()
        );
        debug_assert!(
            src.as_index() + len <= self.capacity(),
            "cpy dst={} src={} len={} cap={}",
            dst.as_index(),
            src.as_index(),
            len,
            self.capacity()
        );
        // SAFETY: upheld by caller.
        unsafe {
            ptr::copy(
                self.buf.ptr().add(src.as_index()),
                self.buf.ptr().add(dst.as_index()),
                len,
            );
        }
    }

    /// Relocates elements after a capacity increase caused by `realloc`.
    ///
    /// After `RawVec` grows via `realloc`, the first `old_capacity` bytes of
    /// the new buffer hold the old contents verbatim. If the deque's logical
    /// elements were contiguous starting at `head` (i.e. they did not wrap
    /// around the end of the old buffer), nothing needs to move. Otherwise we
    /// must shift either the tail or the head segment so that all `len`
    /// elements form one contiguous run in the new, larger buffer.
    ///
    /// For zero-sized types the buffer holds no physical slots, so this is a
    /// no-op regardless of capacity changes.
    ///
    /// # Safety
    /// - `old_capacity` must be the capacity *before* the growth.
    /// - `self.len <= old_capacity` (the invariant held before growth).
    #[inline]
    pub(super) unsafe fn handle_capacity_increase(&mut self, old_capacity: usize) {
        // ZSTs never occupy physical memory; there is nothing to relocate.
        // This also guards against arithmetic overflow when `capacity()` is
        // `usize::MAX` (as reported for ZST deques).
        if size_of::<T>() == 0 {
            return;
        }

        let new_capacity = self.capacity();
        debug_assert!(new_capacity >= old_capacity);

        // Move the shortest contiguous section of the ring buffer
        //
        // H := head
        // L := last element (`self.to_physical_idx(self.len - 1)`)
        //
        //    H             L
        //   [o o o o o o o o ]
        //    H             L
        // A [o o o o o o o o . . . . . . . . ]
        //        L H
        //   [o o o o o o o o ]
        //          H             L
        // B [. . . o o o o o o o o . . . . . ]
        //              L H
        //   [o o o o o o o o ]
        //              L                 H
        // C [o o o o o o . . . . . . . . o o ]

        // Case A: elements are already contiguous from `head` without wrapping.
        // `head + len <= old_capacity` means the last element sits at or before
        // the end of the old region, which is preserved byte-for-byte by realloc.
        //
        // Overflow safety: `head < old_capacity` and `len <= old_capacity`, so
        // `head + len < 2 * old_capacity`. For non-ZST types the allocator
        // guarantees `size_of::<T>() * capacity <= isize::MAX`, hence
        // `capacity <= isize::MAX` and therefore
        // `2 * old_capacity <= 2 * isize::MAX == usize::MAX - 1`.
        // The addition cannot overflow.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "head + len < 2 * old_capacity <= 2 * isize::MAX == usize::MAX - 1"
        )]
        if self.head.as_index() + self.len <= old_capacity {
            // Nop — everything is already in place.
            return;
        }

        // `head < old_capacity` (invariant) thus `head_len >= 1`.
        #[allow(clippy::arithmetic_side_effects, reason = "head < old_capacity")]
        let head_len = old_capacity - self.head.as_index();
        // asserted len > old_cap - head
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "head_len = old_cap - head < len (case A returned otherwise)"
        )]
        let tail_len = self.len - head_len;

        // `new_capacity - old_capacity`: both are valid capacities for a
        // non-ZST type, each ≤ isize::MAX, and `new ≥ old` (growth only),
        // so the subtraction cannot underflow.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "new_capacity >= old_capacity (monotonic growth)"
        )]
        let grown_by = new_capacity - old_capacity;

        // If the head is shorter than the tail, moving the tail instead of the head
        // will take up more runtime.
        if head_len > tail_len && grown_by >= tail_len {
            // Case B: the tail is shorter than the head and there is enough
            // spare room right after the old region to absorb it. Copy the tail
            // forward. This also makes the memory contiguous.
            //
            // SAFETY: source `[0, tail_len)` and destination
            // `[old_capacity, old_capacity + tail_len)` do not overlap because
            // `tail_len <= grown_by` and `old_capacity >= tail_len` (since
            // `head_len > tail_len` and `head_len + tail_len = len <= old_capacity`).
            unsafe {
                self.copy_nonoverlapping(
                    WrappedIndex::zero(),
                    WrappedIndex::from_arbitrary_number(old_capacity),
                    tail_len,
                );
            }
        } else {
            // Case C: move the head segment to the back of the new buffer so
            // that the whole deque becomes contiguous.
            //
            // `new_capacity - head_len`: since `head_len <= old_capacity <
            // new_capacity`, the subtraction cannot underflow.
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "head_len <= old_capacity <= new_capacity"
            )]
            let new_head = WrappedIndex::from_arbitrary_number(new_capacity - head_len);
            // Note: head_len >= 1, new_head == new_capacity - head_len < new_capacity
            // SAFETY: head < old_capacity <= new_capacity, new_head < new_capacity
            unsafe {
                self.copy(self.head, new_head, head_len);
            }
            self.head = new_head;
        }
        // The second condition asserts head == 0 and head >= cap, therefore cap == 0.
        debug_assert!(self.head.as_index() < self.capacity() || self.head.is_zero());
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use crate::collections::vec_deque::VecDeque;

    /// Concatenate the two halves of a deque into a single owned array-backed
    /// slice for comparison. Returns `None` if the combined length exceeds
    /// `N`.
    fn collect_into_array<const N: usize>(dq: &VecDeque<i32>) -> Option<[i32; N]> {
        let (a, b) = dq.as_slices();
        let mut out: [i32; N] = [0; N];
        if a.len() + b.len() > N {
            return None;
        }
        out[..a.len()].copy_from_slice(a);
        out[a.len()..a.len() + b.len()].copy_from_slice(b);
        Some(out)
    }

    #[test]
    fn push_back_within_capacity_fills_slots_in_order() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(3), Ok(()));
        assert_eq!(dq.len(), 3);
        let (a, b) = dq.as_slices();
        assert_eq!(a, &[1, 2, 3]);
        assert!(b.is_empty());
    }

    #[test]
    fn push_front_within_capacity_prepends() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(3), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(2), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(1), Ok(()));
        assert_eq!(dq.len(), 3);
        // With head wrapping, the deque may split across two physical slices;
        // verify logical order via get().
        assert_eq!(dq.get(0), Some(&1));
        assert_eq!(dq.get(1), Some(&2));
        assert_eq!(dq.get(2), Some(&3));
    }

    #[test]
    fn interleaved_front_and_back_preserves_logical_order() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        // push_back(4), push_back(5) -> [4, 5]
        // push_front(2), push_front(3) -> [3, 2, 4, 5] (each front-push prepends)
        // push_back(6) -> [3, 2, 4, 5, 6]
        // push_front(1) -> [1, 3, 2, 4, 5, 6]
        for v in [4, 5] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        for v in [2, 3] {
            assert_eq!(dq.try_push_front_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_back_within_capacity(6), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(1), Ok(()));
        assert_eq!(dq.len(), 6);
        assert_eq!(collect_into_array::<6>(&dq), Some([1, 3, 2, 4, 5, 6]));
    }

    #[test]
    fn full_buffer_reports_full_error() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        let err = match dq.try_push_back_within_capacity(3) {
            Err(e) => e,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(err.len, 2);
        // Same from the front.
        let err = match dq.try_push_front_within_capacity(0) {
            Err(e) => e,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(err.len, 2);
        // The rejected values were not inserted and the deque is unchanged.
        assert_eq!(dq.len(), 2);
        let (a, b) = dq.as_slices();
        assert_eq!(a, &[1, 2]);
        assert!(b.is_empty());
    }

    #[test]
    fn zero_capacity_rejects_immediately() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        let err = match dq.try_push_back_within_capacity(1) {
            Err(e) => e,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(err.len, 0);
        assert!(dq.is_empty());
    }

    #[test]
    fn zst_pushes_succeed_without_real_slots() {
        let mut dq: VecDeque<()> = VecDeque::new();
        // ZST capacity is usize::MAX, so within-capacity pushes always fit.
        assert_eq!(dq.try_push_back_within_capacity(()), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(()), Ok(()));
        assert_eq!(dq.len(), 2);
        let (a, b) = dq.as_slices();
        assert_eq!(a.len() + b.len(), 2);
    }

    #[test]
    fn zst_full_state_is_unreachable_within_capacity() {
        // Since ZST capacity is usize::MAX, `len >= capacity` can never hold
        // for realistic lengths; pushing a few more must keep succeeding.
        let mut dq: VecDeque<()> = VecDeque::new();
        for _ in 0..1024 {
            assert_eq!(dq.try_push_back_within_capacity(()), Ok(()));
        }
        assert_eq!(dq.len(), 1024);
    }
}
