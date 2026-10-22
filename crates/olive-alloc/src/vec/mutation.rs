use super::Vec;
use core::fmt;
use core::ptr;
use core::slice;
use olive_core::alloc::Allocator;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

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
}
