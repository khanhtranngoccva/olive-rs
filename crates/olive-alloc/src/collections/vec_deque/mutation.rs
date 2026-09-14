//! Low-level mutation primitives for [`VecDeque`].
//!
//! These operate *within* existing capacity: they never allocate, and each one
//! reports a full buffer as an error instead of growing. The fallible public
//! push methods in `allocation.rs` build on top of them, securing space first
//! when needed.

use core::cmp;

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

/// Error returned by the within-capacity insert primitives.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TryInsertWithinCapacityError {
    /// The buffer is full (`len == capacity`); no room to shift.
    Full {
        /// The current length (equal to capacity).
        len: usize,
    },
    /// The provided index exceeded the deque's length.
    OutOfBounds,
}

impl core::fmt::Debug for TryInsertWithinCapacityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Full { len } => f
                .debug_struct("TryInsertWithinCapacityError::Full")
                .field("len", len)
                .finish(),
            Self::OutOfBounds => f
                .debug_tuple("TryInsertWithinCapacityError::OutOfBounds")
                .finish(),
        }
    }
}

impl core::fmt::Display for TryInsertWithinCapacityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Full { len } => {
                write!(f, "no spare capacity: deque is full at length {len}")
            }
            Self::OutOfBounds => write!(f, "insert index out of bounds"),
        }
    }
}

impl core::error::Error for TryInsertWithinCapacityError {}

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

    // FIXME: mut and give_back variants

    // -----------------------------------------------------------------------
    // Within-capacity inserts
    // -----------------------------------------------------------------------

    /// Inserts an element at position `index` without attempting to grow the
    /// buffer. Succeeds only if there is already spare capacity and the index
    /// is in bounds.
    ///
    /// On failure the deque is left unchanged and `value` is dropped. Use
    /// [`Self::try_insert_within_capacity_give_back`] to recover the value.
    ///
    /// # Errors
    ///
    /// * [`TryInsertWithinCapacityError::OutOfBounds`] — `index > len`.
    /// * [`TryInsertWithinCapacityError::Full`] — `len == capacity`.
    pub fn try_insert_within_capacity(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<(), TryInsertWithinCapacityError> {
        self.try_insert_mut_within_capacity_give_back(index, value)
            .map(|_| ())
            .map_err(|(_returned, e)| e)
    }

    /// Like [`Self::try_insert_within_capacity`], but returns the value back
    /// on failure.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryInsertWithinCapacityError)` on failure.
    pub fn try_insert_within_capacity_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<(), (T, TryInsertWithinCapacityError)> {
        self.try_insert_mut_within_capacity_give_back(index, value)
            .map(|_| ())
    }

    /// Inserts an element at position `index` and returns a mutable reference
    /// to it, without attempting to grow the buffer.
    ///
    /// # Errors
    ///
    /// * [`TryInsertWithinCapacityError::OutOfBounds`] — `index > len`.
    /// * [`TryInsertWithinCapacityError::Full`] — `len == capacity`.
    pub fn try_insert_mut_within_capacity(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<&mut T, TryInsertWithinCapacityError> {
        self.try_insert_mut_within_capacity_give_back(index, value)
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_insert_mut_within_capacity`], but returns the value
    /// back on failure.
    ///
    /// # Errors
    ///
    /// Returns `(&mut T, (T, TryInsertWithinCapacityError))` on failure.
    pub fn try_insert_mut_within_capacity_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<&mut T, (T, TryInsertWithinCapacityError)> {
        if index > self.len {
            return Err((value, TryInsertWithinCapacityError::OutOfBounds));
        }
        if self.len >= self.capacity() {
            return Err((value, TryInsertWithinCapacityError::Full { len: self.len }));
        }
        // SAFETY: both preconditions upheld above.
        let ptr = unsafe { self.insert_within_cap(index, value) };
        // SAFETY: `ptr` points to the freshly-written, in-bounds slot.
        Ok(unsafe { &mut *ptr })
    }

    /// Inserts `value` at logical position `index` and returns a raw pointer
    /// to the newly-inserted element. Assumes the buffer already has spare
    /// capacity and the index is in bounds.
    ///
    /// This is the canonical unchecked insert primitive shared by both the
    /// within-capacity safe wrappers and the fallible (may-grow) variants in
    /// `allocation.rs`.
    ///
    /// Uses the same strategy as std: shift the shorter side (tail forward or
    /// head backward) via `wrap_copy`, then write the new value into the
    /// vacated slot.
    ///
    /// # Safety
    ///
    /// - `index <= self.len`
    /// - `self.len < self.capacity()`
    ///
    /// Both conditions guarantee that all physical slots touched during the
    /// shift and the final write are in-bounds.
    #[inline]
    pub(super) unsafe fn insert_within_cap(&mut self, index: usize, value: T) -> *mut T {
        debug_assert!(index <= self.len);
        debug_assert!(self.len < self.capacity());
        if index == self.len {
            // Fast path: equivalent to push_back.
            return unsafe { self.push_back_within_cap(value) };
        }
        if size_of::<T>() == 0 {
            // ZST: no physical slots to shift; just bump length and write to
            // the dangling base pointer (which is aligned and sufficient for
            // a zero-sized write).
            let dest = self.buf.ptr().cast::<T>();
            unsafe { dest.write(value) };
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.len < self.capacity <= usize::MAX"
            )]
            {
                self.len += 1;
            }
            return dest;
        }
        // Choose the cheaper direction: shift the tail forward (k elements)
        // or shift the head backward (index elements).
        // SAFETY: `index < self.len` (the equal case returned above), so this
        // cannot underflow.
        #[allow(clippy::arithmetic_side_effects, reason = "asserted index < self.len")]
        let k = self.len - index;
        // Tail is shifted if tail < head
        // Head is shifted if head <= tail or tail >= head.
        if k < index {
            // Shift tail `[index..len]` forward by one.
            // SAFETY: `index < len` (the equal case returned above), so
            // `to_wrapped_index(index + 1)` is valid. The overlap constraint
            // holds because we're shifting by exactly one slot with spare
            // capacity available.
            unsafe {
                self.wrap_copy(
                    self.to_wrapped_index(index),
                    // SAFETY: `index < len <= capacity`, so `index + 1` is in-bounds.
                    #[allow(
                        clippy::arithmetic_side_effects,
                        reason = "asserted index < len <= cap"
                    )]
                    self.to_wrapped_index(index + 1),
                    k,
                );
            }
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.len < self.capacity <= usize::MAX"
            )]
            {
                self.len += 1;
            }
            // SAFETY: slot at `index` was vacated by the shift; writing here
            // initializes a previously-uninitialized in-bounds slot.
            let ptr = unsafe { self.buf.ptr().add(self.to_wrapped_index(index).as_index()) };
            unsafe { ptr.write(value) };
            ptr
        } else {
            // Shift head `[0..index]` backward by one: retreat `head`.
            let old_head = self.head;
            self.head = unsafe { self.wrap_sub(self.head, 1) };
            // SAFETY: the overlap constraint holds — we're shifting `index`
            // elements from `old_head` to `head` (one slot earlier), and
            // `len < capacity` guarantees the destination is in-bounds.
            unsafe {
                self.wrap_copy(old_head, self.head, index);
            }
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.len < self.capacity <= usize::MAX"
            )]
            {
                self.len += 1;
            }
            // SAFETY: slot at logical `index` is now vacant (elements before
            // it were shifted backward, freeing it).
            let ptr = unsafe { self.buf.ptr().add(self.to_wrapped_index(index).as_index()) };
            unsafe { ptr.write(value) };
            ptr
        }
    }

    /// Writes `value` into the next back slot, advances `len`, and returns a
    /// raw pointer to the newly-inserted element.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `self.len < self.capacity()`: the destination
    /// slot is then provably inside the allocation. For zero-sized types the
    /// slot is always the dangling base pointer, so no index arithmetic is
    /// performed at all.
    #[inline]
    pub(super) unsafe fn push_back_within_cap(&mut self, value: T) -> *mut T {
        debug_assert!(
            self.len < self.capacity(),
            "push_back_within_cap requires spare capacity"
        );
        let dest = if size_of::<T>() == 0 {
            // ZST: every "slot" is the same dangling address; nothing to move.
            self.buf.ptr().cast::<T>()
        } else {
            // SAFETY: `len < capacity`, so this wraps correctly.
            let idx = unsafe { self.wrap_add(self.head, self.len) };
            // SAFETY: `idx < capacity`, so the slot is writable.
            unsafe { self.buf.ptr().add(idx.as_index()) }
        };
        unsafe { dest.write(value) };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted self.len < self.capacity <= usize::MAX"
        )]
        {
            self.len += 1;
        }
        dest
    }

    /// Retreats `head` by one, writes `value` there, advances `len`, and
    /// returns a raw pointer to the newly-inserted element.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `self.len < self.capacity()`: retreating
    /// `head` by one from its current position then stays within `0..capacity`.
    /// For zero-sized types `head` never moves and the write goes to the
    /// dangling base pointer.
    #[inline]
    pub(super) unsafe fn push_front_within_cap(&mut self, value: T) -> *mut T {
        debug_assert!(
            self.len < self.capacity(),
            "push_front_within_cap requires spare capacity"
        );
        let dest = if size_of::<T>() == 0 {
            // ZST: head is pinned at 0; just use the dangling base pointer.
            self.buf.ptr().cast::<T>()
        } else {
            // SAFETY: `len < capacity` guarantees there is a free slot before
            // `head`; `wrap_sub` with subtrahend 1 keeps the result in-bounds.
            self.head = unsafe { self.wrap_sub(self.head, 1) };
            // SAFETY: `head < capacity`, so the slot holds writable memory.
            unsafe { self.buf.ptr().add(self.head.as_index()) }
        };
        unsafe { dest.write(value) };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted self.len < self.capacity <= usize::MAX"
        )]
        {
            self.len += 1;
        }
        dest
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
    pub(super) unsafe fn copy(&mut self, src: WrappedIndex, dst: WrappedIndex, len: usize) {
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

    /// Copies a potentially wrapping block of memory `len` long from `src` to
    /// `dst`, handling all combinations of wrap-around and overlap correctly.
    ///
    /// Ported directly from std's `VecDeque::wrap_copy`. The invariant is that
    /// there is at most one continuous overlapping region between `src` and
    /// `dst`: `(abs(dst - src) + len) <= capacity()`.
    ///
    /// # Safety
    ///
    /// - `src` and `dst` must be valid `WrappedIndex` values (< capacity).
    /// - The ranges `[src, src+len)` and `[dst, dst+len)` must satisfy the
    ///   overlap constraint above (at most one overlapping region).
    /// - All destination slots must be initialized or about-to-be-initialized
    ///   (this is an overlapping `memcpy`, not `memmove`-free).
    #[inline]
    #[allow(clippy::arithmetic_side_effects, reason = "debug assertions only")]
    unsafe fn wrap_copy(&mut self, src: WrappedIndex, dst: WrappedIndex, len: usize) {
        debug_assert!(
            cmp::min(src.abs_diff(dst), self.capacity() - src.abs_diff(dst)) + len
                <= self.capacity(),
            "wrc dst={} src={} len={} cap={}",
            dst,
            src,
            len,
            self.capacity()
        );

        // If T is a ZST, don't do any copying.
        if size_of::<T>() == 0 || src == dst || len == 0 {
            return;
        }

        // Checks if the dst is after [src, src + len) slice.
        let dst_after_src = unsafe { self.wrap_sub(dst, src.as_index()) } < len;

        let src_pre_wrap_len = self.capacity() - src.as_index();
        let dst_pre_wrap_len = self.capacity() - dst.as_index();
        // This checks if the source slice wraps (not enough pre-wrap space for `len` elements)
        let src_wraps = src_pre_wrap_len < len;
        // This checks if the destination slice wraps (not enough pre-wrap space for `len` elements)
        let dst_wraps = dst_pre_wrap_len < len;

        match (dst_after_src, src_wraps, dst_wraps) {
            (_, false, false) => {
                // src doesn't wrap, dst doesn't wrap
                //
                //        S . . .
                // 1 [_ _ A A B B C C _]
                // 2 [_ _ A A A A B B _]
                //            D . . .
                //
                unsafe {
                    self.copy(src, dst, len);
                }
            }
            (false, false, true) => {
                // dst before src, src doesn't wrap, dst wraps
                //
                //    S . . .
                // 1 [A A B B _ _ _ C C]
                // 2 [A A B B _ _ _ A A]
                // 3 [B B B B _ _ _ A A]
                //    . .           D .
                //
                unsafe {
                    self.copy(src, dst, dst_pre_wrap_len);
                    self.copy(
                        src.add(dst_pre_wrap_len),
                        WrappedIndex::zero(),
                        len - dst_pre_wrap_len,
                    );
                }
            }
            (true, false, true) => {
                // src before dst, src doesn't wrap, dst wraps
                //
                //              S . . .
                // 1 [C C _ _ _ A A B B]
                // 2 [B B _ _ _ A A B B]
                // 3 [B B _ _ _ A A A A]
                //    . .           D .
                //
                unsafe {
                    self.copy(
                        src.add(dst_pre_wrap_len),
                        WrappedIndex::zero(),
                        len - dst_pre_wrap_len,
                    );
                    self.copy(src, dst, dst_pre_wrap_len);
                }
            }
            (false, true, false) => {
                // dst before src, src wraps, dst doesn't wrap
                //
                //    . .           S .
                // 1 [C C _ _ _ A A B B]
                // 2 [C C _ _ _ B B B B]
                // 3 [C C _ _ _ B B C C]
                //              D . . .
                //
                unsafe {
                    self.copy(src, dst, src_pre_wrap_len);
                    self.copy(
                        WrappedIndex::zero(),
                        dst.add(src_pre_wrap_len),
                        len - src_pre_wrap_len,
                    );
                }
            }
            (true, true, false) => {
                // src before dst, src wraps, dst doesn't wrap
                //
                //    . .           S .
                // 1 [A A B B _ _ _ C C]
                // 2 [A A A A _ _ _ C C]
                // 3 [C C A A _ _ _ C C]
                //    D . . .
                //
                unsafe {
                    self.copy(
                        WrappedIndex::zero(),
                        dst.add(src_pre_wrap_len),
                        len - src_pre_wrap_len,
                    );
                    self.copy(src, dst, src_pre_wrap_len);
                }
            }
            (false, true, true) => {
                // dst before src, src wraps, dst wraps
                //
                //    . . .         S .
                // 1 [A B C D _ E F G H]
                // 2 [A B C D _ E G H H]
                // 3 [A B C D _ E G H A]
                // 4 [B C C D _ E G H A]
                //    . .         D . .
                //
                debug_assert!(dst_pre_wrap_len > src_pre_wrap_len);
                let delta = dst_pre_wrap_len - src_pre_wrap_len;
                unsafe {
                    self.copy(src, dst, src_pre_wrap_len);
                    self.copy(WrappedIndex::zero(), dst.add(src_pre_wrap_len), delta);
                    self.copy(
                        WrappedIndex::from_arbitrary_number(delta),
                        WrappedIndex::zero(),
                        len - dst_pre_wrap_len,
                    );
                }
            }
            (true, true, true) => {
                // src before dst, src wraps, dst wraps
                //
                //    . .         S . .
                // 1 [A B C D _ E F G H]
                // 2 [A A B D _ E F G H]
                // 3 [H A B D _ E F G H]
                // 4 [H A B D _ E F F G]
                //    . . .         D .
                //
                debug_assert!(src_pre_wrap_len > dst_pre_wrap_len);
                let delta = src_pre_wrap_len - dst_pre_wrap_len;
                unsafe {
                    self.copy(
                        WrappedIndex::zero(),
                        WrappedIndex::from_arbitrary_number(delta),
                        len - src_pre_wrap_len,
                    );
                    self.copy(
                        WrappedIndex::from_arbitrary_number(self.capacity() - delta),
                        WrappedIndex::zero(),
                        delta,
                    );
                    self.copy(src, dst, dst_pre_wrap_len);
                }
            }
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

    // --- try_insert_within_capacity family -------------------------------------

    use super::TryInsertWithinCapacityError;

    #[test]
    fn insert_within_capacity_at_front_of_empty() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_insert_within_capacity(0, 42), Ok(()));
        assert_eq!(dq.len(), 1);
        assert_eq!(dq.front(), Some(&42));
    }

    #[test]
    fn insert_within_capacity_in_middle() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_insert_within_capacity(2, 99), Ok(()));
        assert_eq!(collect_into_array::<5>(&dq), Some([1, 2, 99, 3, 4]));
    }

    #[test]
    fn insert_within_capacity_at_back_equals_push_back() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        // Inserting at index == len should behave like push_back.
        assert_eq!(dq.try_insert_within_capacity(2, 3), Ok(()));
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn insert_within_capacity_at_front_shifts_all() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_insert_within_capacity(0, 0), Ok(()));
        assert_eq!(collect_into_array::<4>(&dq), Some([0, 1, 2, 3]));
    }

    #[test]
    fn insert_within_capacity_out_of_bounds() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        let err = match dq.try_insert_within_capacity(5, 99) {
            Err(e) => e,
            Ok(_) => panic!("expected out-of-bounds error"),
        };
        assert_eq!(err, TryInsertWithinCapacityError::OutOfBounds);
        assert_eq!(dq.len(), 1);
    }

    #[test]
    fn insert_within_capacity_full_buffer_rejects() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        // Full; inserting anywhere must fail with Full.
        let err = match dq.try_insert_within_capacity(0, 99) {
            Err(e) => e,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(err, TryInsertWithinCapacityError::Full { len: 2 });
        // Deque unchanged.
        assert_eq!(dq.len(), 2);
        assert_eq!(collect_into_array::<2>(&dq), Some([1, 2]));
    }

    #[test]
    fn insert_within_capacity_give_back_recovers_value_on_full() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        let (returned, err) = match dq.try_insert_within_capacity_give_back(0, 99) {
            Err(pair) => pair,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(returned, 99);
        assert_eq!(err, TryInsertWithinCapacityError::Full { len: 2 });
        assert_eq!(dq.len(), 2);
    }

    #[test]
    fn insert_within_capacity_give_back_recovers_value_on_oob() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        let (returned, err) = match dq.try_insert_within_capacity_give_back(5, 42) {
            Err(pair) => pair,
            Ok(_) => panic!("expected out-of-bounds error"),
        };
        assert_eq!(returned, 42);
        assert_eq!(err, TryInsertWithinCapacityError::OutOfBounds);
        assert!(dq.is_empty());
    }

    #[test]
    fn insert_mut_within_capacity_returns_reference() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(3), Ok(()));
        assert_eq!(dq.try_insert_mut_within_capacity(1, 2), Ok(&mut 2));
        *dq.get_mut(1).unwrap() += 10;
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 12, 3]));
    }

    #[test]
    fn insert_within_capacity_across_wrap_boundary() {
        // Build a wrapped state: push backs then fronts to wrap head past 0.
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(3), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(5), Ok(()));
        // Now: [5, 4, 1, 2, 3], len=5, cap=6, one spare slot.
        // Head has wrapped; inserting in the middle exercises the shift.
        assert_eq!(dq.try_insert_within_capacity(2, 99), Ok(()));
        assert_eq!(dq.len(), 6);
        assert_eq!(collect_into_array::<6>(&dq), Some([5, 4, 99, 1, 2, 3]));
    }

    #[test]
    fn insert_within_capacity_head_shift_wraps_both_src_and_dst() {
        // Force the head-shift branch of `insert_within_cap` into the
        // `(dst_after_src=false, src_wraps=true, dst_wraps=true)` case of
        // `wrap_copy`, where both the source block and the destination block
        // cross the physical end of the buffer.
        //
        // The branch is selected when `k = len - index` is NOT less than
        // `index`, i.e. `index <= len / 2`. A head shift copies `index` elements 
        // from `src = head` to `dst = head - 1`. Since `dst` has exactly one 
        // more slot of pre-wrap room than `src`, to make both wrap, 
        // with `r = capacity - head`, we need `index >= r + 2`, plus 
        // `index <= len - index` and `index < len`. 
        
        // With capacity 8, head 7 (`r = 1`), len 6, and
        // index 3 all constraints hold: `3 <= 6 - 3`, src spans slots 7,0,1
        // (wraps, room 1 < 3) and dst spans slots 6,7,0 (wraps, room 2 < 3);
        // the forward distance from src to dst is 7 >= 3, so
        // `dst_after_src` is false.
        //
        // Reaching head=7, len=6 (used slots 7,0,1,2,3,4) using only
        // in-capacity operations: push backs 1,2 (head=0, back=2), one
        // front-insert of 10 (head retreats 0 -> 7, back still 2), then pushes
        // back 20,30,40 into slots 2,3,4. Head never moves again.
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        assert_eq!(dq.try_insert_within_capacity(0, 10), Ok(()));
        for v in [20, 30, 40] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.len(), 6);
        assert_eq!(collect_into_array::<6>(&dq), Some([10, 1, 2, 20, 30, 40]));
        // Probe: index 3, k = len - index = 3, not less than index ->
        // head-shift branch, wrap_copy(src=7, dst=6, len=3) — both blocks
        // wrap.
        assert_eq!(dq.try_insert_within_capacity(3, 99), Ok(()));
        assert_eq!(dq.len(), 7);
        assert_eq!(collect_into_array::<7>(&dq), Some([10, 1, 2, 99, 20, 30, 40]));
    }
}
