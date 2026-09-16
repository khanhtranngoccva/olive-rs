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
use olive_core::{ptr, slice};

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
    /// On failure the deque is left unchanged and `value` is dropped. Use
    /// [`Self::try_push_back_within_capacity_give_back`] to recover the value.
    ///
    /// # Errors
    ///
    /// Returns [`TryPushWithinCapacityError`] if `len == capacity`.
    #[inline]
    pub fn try_push_back_within_capacity(
        &mut self,
        value: T,
    ) -> Result<(), TryPushWithinCapacityError> {
        self.try_push_back_mut_within_capacity_give_back(value)
            .map(|_| ())
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_push_back_within_capacity`], but returns the value
    /// back on failure.
    ///
    /// This is the canonical implementation for back pushes; all other
    /// within-capacity back-push variants delegate here.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryPushWithinCapacityError)` if `len == capacity`.
    #[inline]
    pub fn try_push_back_within_capacity_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, TryPushWithinCapacityError)> {
        self.try_push_back_mut_within_capacity_give_back(value)
            .map(|_| ())
    }

    /// Prepends an element to the front of the deque without attempting to
    /// grow the buffer. Succeeds only if there is already spare capacity.
    ///
    /// On failure the deque is left unchanged and `value` is dropped. Use
    /// [`Self::try_push_front_within_capacity_give_back`] to recover the value.
    ///
    /// # Errors
    ///
    /// Returns [`TryPushWithinCapacityError`] if `len == capacity`.
    #[inline]
    pub fn try_push_front_within_capacity(
        &mut self,
        value: T,
    ) -> Result<(), TryPushWithinCapacityError> {
        self.try_push_front_mut_within_capacity_give_back(value)
            .map(|_| ())
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_push_front_within_capacity`], but returns the value
    /// back on failure.
    ///
    /// This is the canonical implementation for front pushes; all other
    /// within-capacity front-push variants delegate here.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryPushWithinCapacityError)` if `len == capacity`.
    #[inline]
    pub fn try_push_front_within_capacity_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, TryPushWithinCapacityError)> {
        self.try_push_front_mut_within_capacity_give_back(value)
            .map(|_| ())
    }

    /// Appends an element to the back of the deque without attempting to grow
    /// the buffer, returning a mutable reference to it. Succeeds only if there
    /// is already spare capacity.
    ///
    /// On failure the deque is left unchanged and `value` is dropped. Use
    /// [`Self::try_push_back_mut_within_capacity_give_back`] to recover the
    /// value.
    ///
    /// # Errors
    ///
    /// Returns [`TryPushWithinCapacityError`] if `len == capacity`.
    #[inline]
    pub fn try_push_back_mut_within_capacity(
        &mut self,
        value: T,
    ) -> Result<&mut T, TryPushWithinCapacityError> {
        self.try_push_back_mut_within_capacity_give_back(value)
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_push_back_mut_within_capacity`], but returns the value
    /// back on failure.
    ///
    /// This is the canonical implementation for mutable-reference back pushes;
    /// all other mutable back-push variants delegate here.
    ///
    /// # Errors
    ///
    /// Returns `(&mut T, (T, TryPushWithinCapacityError))` on failure.
    #[inline]
    pub fn try_push_back_mut_within_capacity_give_back(
        &mut self,
        value: T,
    ) -> Result<&mut T, (T, TryPushWithinCapacityError)> {
        let cap = self.capacity();
        if self.len >= cap {
            return Err((value, TryPushWithinCapacityError { len: self.len }));
        }
        // SAFETY: `len < capacity`, so the slot computed below is in-bounds.
        let ptr = unsafe { self.push_back_within_cap(value) };
        // SAFETY: `ptr` points to the freshly-written, in-bounds slot.
        Ok(unsafe { &mut *ptr })
    }

    /// Prepends an element to the front of the deque without attempting to
    /// grow the buffer, returning a mutable reference to it. Succeeds only if
    /// there is already spare capacity.
    ///
    /// On failure the deque is left unchanged and `value` is dropped. Use
    /// [`Self::try_push_front_mut_within_capacity_give_back`] to recover the
    /// value.
    ///
    /// # Errors
    ///
    /// Returns [`TryPushWithinCapacityError`] if `len == capacity`.
    #[inline]
    pub fn try_push_front_mut_within_capacity(
        &mut self,
        value: T,
    ) -> Result<&mut T, TryPushWithinCapacityError> {
        self.try_push_front_mut_within_capacity_give_back(value)
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_push_front_mut_within_capacity`], but returns the
    /// value back on failure.
    ///
    /// This is the canonical implementation for mutable-reference front
    /// pushes; all other mutable front-push variants delegate here.
    ///
    /// # Errors
    ///
    /// Returns `(&mut T, (T, TryPushWithinCapacityError))` on failure.
    #[inline]
    pub fn try_push_front_mut_within_capacity_give_back(
        &mut self,
        value: T,
    ) -> Result<&mut T, (T, TryPushWithinCapacityError)> {
        let cap = self.capacity();
        if self.len >= cap {
            return Err((value, TryPushWithinCapacityError { len: self.len }));
        }
        // SAFETY: `len < capacity`, so retreating `head` stays in-bounds.
        let ptr = unsafe { self.push_front_within_cap(value) };
        // SAFETY: `ptr` points to the freshly-written, in-bounds slot.
        Ok(unsafe { &mut *ptr })
    }

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
    pub(super) unsafe fn copy_nonoverlapping(
        &mut self,
        src: WrappedIndex,
        dst: WrappedIndex,
        len: usize,
    ) {
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
    pub(super) unsafe fn wrap_copy(&mut self, src: WrappedIndex, dst: WrappedIndex, len: usize) {
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

    /// Rearranges the internal storage of this deque so it is one contiguous
    /// slice, which is then returned.
    ///
    /// This method does not allocate and does not change the order of the
    /// inserted elements. As it returns a mutable slice, this can be used to
    /// sort a deque.
    ///
    /// Once the internal storage is contiguous, the [`as_slices`] and
    /// [`as_mut_slices`] methods will return the entire contents of the
    /// deque in a single slice.
    ///
    /// [`as_slices`]: VecDeque::as_slices
    /// [`as_mut_slices`]: VecDeque::as_mut_slices
    pub fn make_contiguous(&mut self) -> &mut [T] {
        if size_of::<T>() == 0 {
            self.head = WrappedIndex::zero();
        }

        if self.is_contiguous() {
            // SAFETY: head < capacity, head + len <= capacity
            unsafe {
                return slice::from_raw_parts_mut(
                    self.buf.ptr().add(self.head.as_index()),
                    self.len,
                );
            }
        }

        let &mut Self { head, len, .. } = self;
        let ptr = self.buf.ptr();
        let cap = self.capacity();

        #[allow(clippy::arithmetic_side_effects, reason = "invariant: len <= cap")]
        let free = cap - len;
        #[allow(clippy::arithmetic_side_effects, reason = "invariant: head < cap")]
        let head_len = cap - head.as_index();

        // tail <= head < capacity
        // head cannot be <= capacity, because we know that VecDeque is non-empty, since it is not
        // contiguous at this point
        #[allow(clippy::arithmetic_side_effects, reason = "invariant: head_len <= len")]
        let tail = WrappedIndex::from_arbitrary_number(len - head_len);
        let tail_len = tail.as_index();

        if free >= head_len {
            // there is enough free space to copy the head in one go,
            // this means that we first shift the tail backwards, and then
            // copy the head to the correct position.
            //
            // from: DEFGH....ABC
            // to:   ABCDEFGH....
            // SAFETY: head_len + tail_len = len <= capacity, head + head_len = head + cap - head = cap
            unsafe {
                self.copy(
                    WrappedIndex::zero(),
                    WrappedIndex::from_arbitrary_number(head_len),
                    tail_len,
                );
                // ...DEFGH.ABC
                self.copy_nonoverlapping(head, WrappedIndex::zero(), head_len);
                // ABCDEFGH....
            }

            self.head = WrappedIndex::zero();
        } else if free >= tail_len {
            // there is enough free space to copy the tail in one go,
            // this means that we first shift the head forwards, and then
            // copy the tail to the correct position.
            //
            // from: FGH....ABCDE
            // to:   ...ABCDEFGH.
            // SAFETY: head + head_len == head + cap - head == cap, tail + head_len == tail_len + head_len == len <= capacity
            // tail_len == len - head <= len <= capacity
            // tail + tail_len + head_len == tail_len + tail_len + head_len == len + tail_len <= len + free == capacity
            unsafe {
                self.copy(head, tail, head_len);
                // FGHABCDE....
                self.copy_nonoverlapping(WrappedIndex::zero(), tail.add(head_len), tail_len);
                // ...ABCDEFGH.
            }

            self.head = tail;
        } else {
            // `free` is smaller than both `head_len` and `tail_len`.
            // the general algorithm for this first moves the slices
            // right next to each other and then uses `slice::rotate`
            // to rotate them into place:
            //
            // initially:   HIJK..ABCDEFG
            // step 1:      ..HIJKABCDEFG
            // step 2:      ..ABCDEFGHIJK
            //
            // or:
            //
            // initially:   FGHIJK..ABCDE
            // step 1:      FGHIJKABCDE..
            // step 2:      ABCDEFGHIJK..

            // pick the shorter of the 2 slices to reduce the amount
            // of memory that needs to be moved around.
            if head_len > tail_len {
                // tail is shorter, so:
                //  1. copy tail forwards
                //  2. rotate used part of the buffer
                //  3. update head to point to the new beginning (which is just `free`)

                // SAFETY: tail_len + free <= len + free == capacity,
                // noncontiguous => len > 0 => free < capacity
                unsafe {
                    // if there is no free space in the buffer, then the slices are already
                    // right next to each other and we don't need to move any memory.
                    if free != 0 {
                        // because we only move the tail forward as much as there's free space
                        // behind it, we don't overwrite any elements of the head slice, and
                        // the slices end up right next to each other.
                        self.copy(
                            WrappedIndex::zero(),
                            WrappedIndex::from_arbitrary_number(free),
                            tail_len,
                        );
                    }

                    // We just copied the tail right next to the head slice,
                    // so all of the elements in the range are initialized
                    let slice = &mut *self.buffer_range(free..self.capacity());

                    // because the deque wasn't contiguous, we know that `tail_len < self.len == slice.len()`,
                    // so this will never panic.
                    slice.rotate_left(tail_len);

                    // the used part of the buffer now is `free..self.capacity()`, so set
                    // `head` to the beginning of that range.
                    self.head = WrappedIndex::from_arbitrary_number(free);
                }
            } else {
                // head is shorter so:
                //  1. copy head backwards
                //  2. rotate used part of the buffer
                //  3. update head to point to the new beginning (which is the beginning of the buffer)

                // SAFETY: head + head_len == capacity, tail_len + head_len == len <= capacity,
                unsafe {
                    // if there is no free space in the buffer, then the slices are already
                    // right next to each other and we don't need to move any memory.
                    if free != 0 {
                        // copy the head slice to lie right behind the tail slice.
                        self.copy(
                            self.head,
                            WrappedIndex::from_arbitrary_number(tail_len),
                            head_len,
                        );
                    }

                    // because we copied the head slice so that both slices lie right
                    // next to each other, all the elements in the range are initialized.
                    let slice = &mut *self.buffer_range(0..self.len);

                    // because the deque wasn't contiguous, we know that `head_len < self.len == slice.len()`
                    // so this will never panic.
                    slice.rotate_right(head_len);

                    // the used part of the buffer now is `0..self.len`, so set
                    // `head` to the beginning of that range.
                    self.head = WrappedIndex::zero();
                }
            }
        }

        // SAFETY: the slice is newly made contiguous.
        unsafe { slice::from_raw_parts_mut(ptr.add(self.head.as_index()), self.len) }
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
// Removal (never allocates)
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Removes all elements from the deque, dropping them in place.
    ///
    /// Removal never allocates, so this cannot fail. The buffer's capacity is
    /// left untouched.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Removes the last element from the deque and returns it, or `None` if
    /// the deque is empty.
    ///
    /// Removal never allocates, so this cannot fail.
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            // SAFETY: asserted `self.len > 0`.
            #[allow(clippy::arithmetic_side_effects, reason = "asserted self.len > 0")]
            {
                self.len -= 1;
            }
            // SAFETY: we read the slot at logical index `new_len` (the old
            // back), which lies inside the initialized region `[0..old_len)`.
            Some(unsafe {
                let idx = self.to_wrapped_index(self.len);
                ptr::read(self.buf.ptr().add(idx.as_index()))
            })
        }
    }

    /// Removes the first element from the deque and returns it, or `None` if
    /// the deque is empty.
    ///
    /// Removal never allocates, so this cannot fail.
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            // SAFETY: `self.head < capacity` (or both zero for an empty
            // buffer, guarded above).
            let new_head = unsafe { self.wrap_add(self.head, 1) };
            // SAFETY: asserted `self.len > 0`.
            #[allow(clippy::arithmetic_side_effects, reason = "asserted self.len > 0")]
            {
                self.len -= 1;
            }
            // SAFETY: `head` still points at the (now removed) front slot,
            // which holds an initialized element. Read it out before
            // advancing the head.
            let value = unsafe { ptr::read(self.buf.ptr().add(self.head.as_index())) };
            self.head = new_head;
            Some(value)
        }
    }

    /// Shortens the deque, keeping only the first `new_len` elements and
    /// dropping the rest. If `new_len` is greater than or equal to the current
    /// length, nothing happens.
    ///
    /// Truncation never allocates, so this cannot fail.
    pub fn truncate(&mut self, new_len: usize) {
        // Truncating to a length >= current length is a no-op (matches std).
        if new_len >= self.len {
            return;
        }
        let old_len = self.len;
        // Resolve the logical tail range `[new_len..old_len)` into one or two
        // contiguous physical runs before mutating anything. The range lies
        // entirely within `[0..old_len)`, so it is always resolvable.
        let (a_range, b_range) = self
            .try_slice_ranges(new_len..old_len, old_len)
            .expect("tail range is resolvable");
        // Build raw fat-slice pointers for both runs.
        // SAFETY: `try_slice_ranges` returns valid ranges into the physical
        // buffer over initialized elements.
        let a_slice = unsafe { self.buffer_range(a_range) };
        let b_slice = unsafe { self.buffer_range(b_range) };
        // Shrink the logical length *before* running destructors so that, if a
        // destructor panics, unwinding sees a length that already excludes the
        // dropped tail and cannot double-free it. (A second panic during unwind
        // aborts, per Rust's rules.)
        self.len = new_len;
        // Panic guard for the second run: if dropping the first run panics,
        // this guard ensures the second run is still cleaned up during unwind
        // rather than leaked. On the happy path the guard simply drops `b`
        // when it goes out of scope after `a` has been dropped.
        struct SecondRunGuard<'a, T> {
            ptr: *mut [T],
            _marker: core::marker::PhantomData<&'a mut [T]>,
        }
        impl<T> Drop for SecondRunGuard<'_, T> {
            fn drop(&mut self) {
                // SAFETY: the pointer was built from `buffer_range` over
                // initialized, in-bounds slots and has not been consumed yet.
                unsafe { ptr::drop_in_place(&mut *self.ptr) };
            }
        }
        let _guard = SecondRunGuard {
            ptr: b_slice,
            _marker: core::marker::PhantomData,
        };
        // SAFETY: `a_slice` covers disjoint, fully-initialized slots; dropping
        // it as a fat slice runs every element's destructor exactly once.
        unsafe { ptr::drop_in_place(&mut *a_slice) };
        // If we reach here, dropping `a` did not panic. The guard will drop
        // `b` when `_guard` goes out of scope below.
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

    // --- try_push_{back,front}_{mut}_within_capacity give_back family ------------------

    #[test]
    fn push_back_within_capacity_give_back_recovers_value_on_full() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        let (returned, err) = match dq.try_push_back_within_capacity_give_back(3) {
            Err(pair) => pair,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(returned, 3);
        assert_eq!(err.len, 2);
        // Deque unchanged.
        assert_eq!(dq.len(), 2);
        assert_eq!(collect_into_array::<2>(&dq), Some([1, 2]));
    }

    #[test]
    fn push_front_within_capacity_give_back_recovers_value_on_full() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        let (returned, err) = match dq.try_push_front_within_capacity_give_back(0) {
            Err(pair) => pair,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(returned, 0);
        assert_eq!(err.len, 2);
        // Deque unchanged.
        assert_eq!(dq.len(), 2);
        assert_eq!(collect_into_array::<2>(&dq), Some([1, 2]));
    }

    #[test]
    fn push_within_capacity_give_back_succeeds_when_space_available() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity_give_back(1), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity_give_back(0), Ok(()));
        assert_eq!(collect_into_array::<2>(&dq), Some([0, 1]));
    }

    // --- try_push_{back,front}_mut_within_capacity family -----------------------

    #[test]
    fn push_back_mut_within_capacity_returns_reference() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [10, 20, 30] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        // Push a new element at the back and mutate it through the returned
        // reference; the pre-existing elements must be untouched.
        let slot = match dq.try_push_back_mut_within_capacity(99) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *slot *= 2;
        assert_eq!(collect_into_array::<4>(&dq), Some([10, 20, 30, 198]));
    }

    #[test]
    fn push_front_mut_within_capacity_returns_reference() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [10, 20, 30] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        // Prepend a new element and mutate it through the returned reference;
        // the pre-existing elements must be untouched.
        let slot = match dq.try_push_front_mut_within_capacity(-5) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *slot *= 3;
        assert_eq!(collect_into_array::<4>(&dq), Some([-15, 10, 20, 30]));
    }

    #[test]
    fn push_back_mut_within_capacity_full_rejects_and_gives_back() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        let (returned, err) = match dq.try_push_back_mut_within_capacity_give_back(3) {
            Err(pair) => pair,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(returned, 3);
        assert_eq!(err.len, 2);
        // Deque unchanged.
        assert_eq!(dq.len(), 2);
        assert_eq!(collect_into_array::<2>(&dq), Some([1, 2]));
    }

    #[test]
    fn push_front_mut_within_capacity_full_rejects_and_gives_back() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back_within_capacity(1), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(2), Ok(()));
        let (returned, err) = match dq.try_push_front_mut_within_capacity_give_back(0) {
            Err(pair) => pair,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(returned, 0);
        assert_eq!(err.len, 2);
        // Deque unchanged.
        assert_eq!(dq.len(), 2);
        assert_eq!(collect_into_array::<2>(&dq), Some([1, 2]));
    }

    #[test]
    fn push_back_mut_within_capacity_drops_value_on_plain_failure() {
        use crate::test_helpers::DropCounter;
        use std::sync::Arc;

        let counter = Arc::new(DropCounter::new());
        let mut dq: VecDeque<Tracked> = VecDeque::try_with_capacity(1).expect("allocation ok");
        assert_eq!(
            dq.try_push_back_within_capacity(Tracked(0, counter.clone())),
            Ok(())
        );
        // Buffer is full; the plain (non-give-back) variant drops the rejected value.
        let err = match dq.try_push_back_mut_within_capacity(Tracked(99, counter.clone())) {
            Err(e) => e,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(err.len, 1);
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn push_front_mut_within_capacity_drops_value_on_plain_failure() {
        use crate::test_helpers::DropCounter;
        use std::sync::Arc;

        let counter = Arc::new(DropCounter::new());
        let mut dq: VecDeque<Tracked> = VecDeque::try_with_capacity(1).expect("allocation ok");
        assert_eq!(
            dq.try_push_back_within_capacity(Tracked(0, counter.clone())),
            Ok(())
        );
        // Buffer is full; the plain (non-give-back) variant drops the rejected value.
        let err = match dq.try_push_front_mut_within_capacity(Tracked(99, counter.clone())) {
            Err(e) => e,
            Ok(_) => panic!("expected full-buffer error"),
        };
        assert_eq!(err.len, 1);
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn push_mut_within_capacity_zst() {
        let mut dq: VecDeque<()> = VecDeque::new();
        assert_eq!(dq.try_push_back_mut_within_capacity(()), Ok(&mut ()));
        assert_eq!(dq.try_push_front_mut_within_capacity(()), Ok(&mut ()));
        assert_eq!(dq.try_push_back_mut_within_capacity(()), Ok(&mut ()));
        assert_eq!(dq.len(), 3);
    }

    #[test]
    fn push_back_mut_within_capacity_across_wrap_boundary() {
        // Build a wrapped state: [5, 4, 1, 2, 3], len=5, cap=6.
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(5), Ok(()));
        // One spare slot remains; push back through the returned reference and
        // verify the five pre-existing elements are untouched.
        let slot = match dq.try_push_back_mut_within_capacity(7) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *slot *= 10;
        assert_eq!(collect_into_array::<6>(&dq), Some([5, 4, 1, 2, 3, 70]));
    }

    #[test]
    fn push_front_mut_within_capacity_across_wrap_boundary() {
        // Build a wrapped state: [5, 4, 1, 2, 3], len=5, cap=6.
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(5), Ok(()));
        // One spare slot remains; prepend through the returned reference and
        // verify the five pre-existing elements are untouched.
        let slot = match dq.try_push_front_mut_within_capacity(-7) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *slot *= 10;
        assert_eq!(collect_into_array::<6>(&dq), Some([-70, 5, 4, 1, 2, 3]));
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
        for v in [1, 3, 5] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        // Insert in the middle and mutate through the returned reference; the
        // surrounding elements must be untouched.
        let slot = match dq.try_insert_mut_within_capacity(1, 2) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *slot += 10;
        assert_eq!(collect_into_array::<4>(&dq), Some([1, 12, 3, 5]));
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
        // `index`, i.e. tail >= head.

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
        assert_eq!(
            collect_into_array::<7>(&dq),
            Some([10, 1, 2, 99, 20, 30, 40])
        );
    }

    // --- pop_back / pop_front ---------------------------------------------------

    #[test]
    fn pop_back_returns_none_when_empty() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        assert_eq!(dq.pop_back(), None);
        assert!(dq.is_empty());
    }

    #[test]
    fn pop_front_returns_none_when_empty() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        assert_eq!(dq.pop_front(), None);
        assert!(dq.is_empty());
    }

    #[test]
    fn pop_back_removes_last_in_order() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.pop_back(), Some(4));
        assert_eq!(dq.pop_back(), Some(3));
        assert_eq!(dq.pop_back(), Some(2));
        assert_eq!(dq.pop_back(), Some(1));
        assert_eq!(dq.pop_back(), None);
        assert!(dq.is_empty());
    }

    #[test]
    fn pop_front_removes_first_in_order() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.pop_front(), Some(1));
        assert_eq!(dq.pop_front(), Some(2));
        assert_eq!(dq.pop_front(), Some(3));
        assert_eq!(dq.pop_front(), Some(4));
        assert_eq!(dq.pop_front(), None);
        assert!(dq.is_empty());
    }

    #[test]
    fn pop_back_across_wrap_boundary() {
        // Build a wrapped state: [5, 4, 1, 2, 3] with head retreated past 0.
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(5), Ok(()));
        assert_eq!(collect_into_array::<5>(&dq), Some([5, 4, 1, 2, 3]));
        // Popping from the back walks down through the first physical segment.
        assert_eq!(dq.pop_back(), Some(3));
        assert_eq!(dq.pop_back(), Some(2));
        assert_eq!(dq.pop_back(), Some(1));
        assert_eq!(collect_into_array::<2>(&dq), Some([5, 4]));
    }

    #[test]
    fn pop_front_across_wrap_boundary() {
        // Build a wrapped state: [5, 4, 1, 2, 3] with head retreated past 0.
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(5), Ok(()));
        assert_eq!(collect_into_array::<5>(&dq), Some([5, 4, 1, 2, 3]));
        // Popping from the front advances `head` forward, eventually wrapping
        // around to slot 0 and beyond.
        assert_eq!(dq.pop_front(), Some(5));
        assert_eq!(dq.pop_front(), Some(4));
        assert_eq!(dq.pop_front(), Some(1));
        assert_eq!(collect_into_array::<2>(&dq), Some([2, 3]));
    }

    #[test]
    fn alternating_pops_drain_correctly() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4, 5, 6] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        // Pop alternately from both ends; verify each value lands correctly.
        assert_eq!(dq.pop_front(), Some(1));
        assert_eq!(dq.pop_back(), Some(6));
        assert_eq!(dq.pop_front(), Some(2));
        assert_eq!(dq.pop_back(), Some(5));
        assert_eq!(dq.pop_front(), Some(3));
        assert_eq!(dq.pop_back(), Some(4));
        assert!(dq.is_empty());
    }

    #[test]
    fn zst_pop_round_trip() {
        let mut dq: VecDeque<()> = VecDeque::new();
        assert_eq!(dq.pop_back(), None);
        assert_eq!(dq.pop_front(), None);
        assert_eq!(dq.try_push_back_within_capacity(()), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(()), Ok(()));
        assert_eq!(dq.len(), 2);
        assert_eq!(dq.pop_back(), Some(()));
        assert_eq!(dq.pop_front(), Some(()));
        assert!(dq.is_empty());
    }

    // --- truncate / clear -------------------------------------------------------

    #[test]
    fn truncate_to_zero_clears() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        dq.truncate(0);
        assert!(dq.is_empty());
        // Capacity is preserved by truncation (matches std semantics).
        assert!(dq.capacity() >= 3);
    }

    #[test]
    fn truncate_geq_len_is_noop() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        dq.truncate(3);
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
        dq.truncate(100);
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn truncate_keeps_prefix_and_drops_tail() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        dq.truncate(3);
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn truncate_across_wrap_boundary() {
        // Wrapped state: [5, 4, 1, 2, 3], len=5, cap=6.
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(5), Ok(()));
        assert_eq!(collect_into_array::<5>(&dq), Some([5, 4, 1, 2, 3]));
        // Keep only the first two logical elements; the dropped tail spans the
        // wrap boundary.
        dq.truncate(2);
        assert_eq!(dq.len(), 2);
        assert_eq!(collect_into_array::<2>(&dq), Some([5, 4]));
    }

    #[test]
    fn truncate_then_reuse_buffer() {
        // After truncating, pushing again must reuse the same buffer without
        // corrupting the surviving prefix.
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        dq.truncate(2);
        assert_eq!(collect_into_array::<2>(&dq), Some([1, 2]));
        assert_eq!(dq.try_push_back_within_capacity(99), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(-1), Ok(()));
        assert_eq!(collect_into_array::<4>(&dq), Some([-1, 1, 2, 99]));
    }

    #[test]
    fn clear_empties_but_preserves_capacity() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let cap_before = dq.capacity();
        dq.clear();
        assert!(dq.is_empty());
        assert_eq!(dq.capacity(), cap_before);
        // The deque remains usable after clearing.
        assert_eq!(dq.try_push_back_within_capacity(42), Ok(()));
        assert_eq!(dq.back(), Some(&42));
    }

    #[test]
    fn clear_on_wrapped_state() {
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(5), Ok(()));
        dq.clear();
        assert!(dq.is_empty());
        assert_eq!(dq.try_push_back_within_capacity(7), Ok(()));
        assert_eq!(dq.front(), Some(&7));
    }

    #[test]
    fn clear_zst() {
        let mut dq: VecDeque<()> = VecDeque::new();
        for _ in 0..10 {
            assert_eq!(dq.try_push_back_within_capacity(()), Ok(()));
        }
        assert_eq!(dq.len(), 10);
        dq.clear();
        assert!(dq.is_empty());
    }

    // --- destructor accounting --------------------------------------------------

    use crate::test_helpers::DropCounter;
    use std::sync::Arc;

    #[allow(dead_code)]
    struct Tracked(u32, Arc<DropCounter>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.1.record_drop();
        }
    }

    #[test]
    // FIXME: Use a ledger
    fn pop_drops_removed_element_exactly_once() {
        let counter = Arc::new(DropCounter::new());
        let mut dq: VecDeque<Tracked> = VecDeque::try_with_capacity(8).expect("allocation ok");
        for i in 0..4u32 {
            assert_eq!(
                dq.try_push_back_within_capacity(Tracked(i, counter.clone())),
                Ok(())
            );
        }
        assert_eq!(counter.get(), 0);
        assert_eq!(dq.pop_back().map(|t| t.0), Some(3));
        assert_eq!(counter.get(), 1);
        assert_eq!(dq.pop_front().map(|t| t.0), Some(0));
        assert_eq!(counter.get(), 2);
        // Two survivors remain in the deque.
        assert_eq!(dq.len(), 2);
        drop(dq);
        // Total of four drops: every element exactly once.
        assert_eq!(counter.get(), 4);
    }

    #[test]
    // FIXME: Use a ledger
    fn truncate_drops_only_the_removed_tail() {
        let counter = Arc::new(DropCounter::new());
        let mut dq: VecDeque<Tracked> = VecDeque::try_with_capacity(8).expect("allocation ok");
        for i in 0..5u32 {
            assert_eq!(
                dq.try_push_back_within_capacity(Tracked(i, counter.clone())),
                Ok(())
            );
        }
        assert_eq!(counter.get(), 0);
        dq.truncate(2);
        // Exactly three elements (indices 2, 3, 4) were dropped.
        assert_eq!(counter.get(), 3);
        assert_eq!(dq.len(), 2);
        drop(dq);
        // Two more survivors drop on final cleanup -> total five.
        assert_eq!(counter.get(), 5);
    }

    #[test]
    // FIXME: Use a ledger
    fn clear_drops_everything() {
        let counter = Arc::new(DropCounter::new());
        let mut dq: VecDeque<Tracked> = VecDeque::try_with_capacity(8).expect("allocation ok");
        for i in 0..5u32 {
            assert_eq!(
                dq.try_push_back_within_capacity(Tracked(i, counter.clone())),
                Ok(())
            );
        }
        assert_eq!(counter.get(), 0);
        dq.clear();
        assert_eq!(counter.get(), 5);
        assert!(dq.is_empty());
        drop(dq);
        // Nothing left to drop.
        assert_eq!(counter.get(), 5);
    }
}
