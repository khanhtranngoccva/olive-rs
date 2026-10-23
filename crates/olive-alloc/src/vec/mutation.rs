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
    pub fn retain<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|i| predicate(i));
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

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_core::try_traits::TryFromIterator;
    use olive_test_helpers::{CloneBudget, FlakyCloneTrackedItem, Ledger, PanicArmer, TrackedItem};
    use std::sync::Arc;

    #[test]
    fn clear_keeps_capacity() {
        let mut v = Vec::new();
        for i in 0..5 {
            v.try_push(i).unwrap();
        }
        let cap_before = v.capacity();
        v.clear();
        assert!(v.is_empty());
        assert_eq!(v.capacity(), cap_before);
    }

    // FIXME: use the existing PanickyDrop
    #[test]
    fn clear_panicking_drop_is_safe() {
        let ledger = Arc::new(Ledger::new());
        let armer = Arc::new(PanicArmer::new());

        /// Panics exactly once, on the first drop that fires while armed.
        struct ClearPanicDrop {
            id: u32,
            ledger: Arc<Ledger>,
            armer: Arc<PanicArmer>,
        }
        impl Drop for ClearPanicDrop {
            fn drop(&mut self) {
                self.ledger.unregister(self.id);
                if self.armer.fetch_and_disarm() {
                    panic!("forced panic in drop");
                }
            }
        }

        armer.arm();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut v = Vec::new();
            for i in 0..4u32 {
                v.try_push(ClearPanicDrop {
                    id: i,
                    ledger: ledger.clone(),
                    armer: armer.clone(),
                })
                .unwrap();
            }
            // `clear` drops every element; the first drop to fire panics,
            // which also disarms the armer for every later drop.
            v.clear();
        }));

        assert!(result.is_err(), "expected a drop to panic during clear");

        // Every element must have been dropped exactly once and nothing may
        // have been dropped twice: the shrinking length keeps the drop path
        // safe even in the face of the panic.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..4));
    }

    #[test]
    fn pop_if_matches() {
        let mut v = Vec::new();
        for i in [1, 2, 3, 4] {
            v.try_push(i).unwrap();
        }
        let popped = v.pop_if(|x| *x % 2 == 0);
        assert_eq!(popped, Some(4));
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn pop_if_no_match() {
        let mut v = Vec::new();
        for i in [1, 3, 5] {
            v.try_push(i).unwrap();
        }
        let popped = v.pop_if(|x| *x % 2 == 0);
        assert_eq!(popped, None);
        assert_eq!(v.as_slice(), &[1, 3, 5]);
    }

    #[test]
    fn pop_if_empty() {
        let mut v: Vec<i32> = Vec::new();
        let popped = v.pop_if(|_| true);
        assert_eq!(popped, None);
    }

    #[test]
    fn swap_and_reverse() {
        let mut v = Vec::new();
        for i in 0..4 {
            v.try_push(i).unwrap();
        }
        v.try_swap(0, 3).unwrap();
        assert_eq!(v.as_slice(), &[3, 1, 2, 0]);
        v.reverse();
        assert_eq!(v.as_slice(), &[0, 2, 1, 3]);
    }

    #[test]
    fn try_swap_out_of_bounds_reports_error() {
        let mut v = Vec::new();
        for i in 0..3 {
            v.try_push(i).unwrap();
        }
        let err = v.try_swap(0, 3).unwrap_err();
        assert_eq!(err.index, 3);
        assert_eq!(err.len, 3);
        // Nothing was swapped on failure.
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn sort_orders_elements() {
        let mut v = Vec::new();
        for x in [3, 1, 4, 1, 5, 9, 2, 6] {
            v.try_push(x).unwrap();
        }
        v.sort();
        assert_eq!(v.as_slice(), &[1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn truncate_drops_tail() {
        let mut v = Vec::new();
        for i in 0..5 {
            v.try_push(i).unwrap();
        }
        v.truncate(2);
        assert_eq!(v.as_slice(), &[0, 1]);
        // Capacity is preserved by truncate.
        assert!(v.capacity() >= 5);
    }

    #[test]
    fn truncate_drop_runs_each_element_once() {
        let ledger = Arc::new(Ledger::new());
        let budget = CloneBudget::new(u32::MAX);
        let mut v: Vec<FlakyCloneTrackedItem> = Vec::new();
        for i in 0..5u32 {
            v.try_push(FlakyCloneTrackedItem::with_id(&ledger, i, budget.share()))
                .unwrap();
        }
        // truncate(3) destroys the tail (ids 3 and 4).
        v.truncate(3);
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        // Drops the remaining IDs
        drop(v);
        // All five dropped exactly once; no leaks, no double-frees.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    /// Regression test: if a destructor in the truncated tail panics, `truncate`
    /// must have already shrunk `len` before dropping, so unwinding the Vec cannot
    /// drop (double-free) those elements again.
    #[test]
    fn truncate_panicking_drop_is_safe() {
        let ledger = Arc::new(Ledger::new());
        let armer = Arc::new(PanicArmer::new());

        /// Panics exactly once, the first time it runs while armed; records its id.
        struct PanicDrop {
            id: u32,
            ledger: Arc<Ledger>,
            armer: Arc<PanicArmer>,
        }
        impl Drop for PanicDrop {
            fn drop(&mut self) {
                self.ledger.unregister(self.id);
                if self.armer.fetch_and_disarm() {
                    panic!("forced panic in drop");
                }
            }
        }

        armer.arm();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let l = ledger.clone();
            let a = armer.clone();
            move || {
                let mut v = Vec::new();
                for i in 0..4u32 {
                    l.register(i);
                    v.try_push(PanicDrop {
                        id: i,
                        ledger: l.clone(),
                        armer: a.clone(),
                    })
                    .unwrap();
                }
                // Truncate off the last two (ids 2 and 3); the first one dropped
                // (id 3) panics, which disarms the armer for any later drops.
                v.truncate(2);
            }
        }));

        assert!(result.is_err(), "expected the drop to panic");

        // Every element must be accounted for properly.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..4));
    }

    #[test]
    fn set_len_after_manual_write() {
        let mut v: Vec<i32> = Vec::try_with_capacity(4).unwrap();
        // SAFETY: we write valid values into allocated-but-uninit slots.
        unsafe {
            v.as_mut_ptr().add(0).write(10);
            v.as_mut_ptr().add(1).write(20);
            v.as_mut_ptr().add(2).write(30);
            v.set_len(3);
        }
        assert_eq!(v.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn swap_remove_replaces_with_last() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let removed = v.try_swap_remove(1).unwrap();
        assert_eq!(removed, 1);
        // Last element (4) moved to index 1.
        assert_eq!(v.as_slice(), &[0, 4, 2, 3]);
    }

    #[test]
    fn swap_remove_last_element() {
        let mut v = Vec::new();
        v.try_push(10).unwrap();
        v.try_push(20).unwrap();
        let removed = v.try_swap_remove(1).unwrap();
        assert_eq!(removed, 20);
        assert_eq!(v.as_slice(), &[10]);
    }

    #[test]
    fn try_swap_remove_out_of_bounds() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        let err = v.try_swap_remove(5).unwrap_err();
        assert_eq!(err.index, 5);
        assert_eq!(err.len, 1);
    }

    #[test]
    fn try_remove_shifts_left() {
        let mut v = Vec::new();
        for i in 0..5 {
            v.try_push(i).unwrap();
        }
        assert_eq!(v.try_remove(0).unwrap(), 0);
        assert_eq!(v.try_remove(2).unwrap(), 3);
        assert_eq!(v.as_slice(), &[1, 2, 4]);
    }

    #[test]
    fn try_remove_out_of_bounds() {
        let mut v = Vec::new();
        v.try_push(7).unwrap();
        let err = v.try_remove(3).unwrap_err();
        assert_eq!(err.index, 3);
        assert_eq!(err.len, 1);
    }

    #[test]
    fn retain_filters_in_place() {
        let mut v = Vec::new();
        for i in 0..6 {
            v.try_push(i).unwrap();
        }
        v.retain(|x| *x % 2 == 0);
        assert_eq!(v.as_slice(), &[0, 2, 4]);
    }

    #[test]
    fn retain_mut_filters_in_place() {
        let mut v = Vec::new();
        for i in 0..6i32 {
            v.try_push(i).unwrap();
        }
        v.retain_mut(|x| {
            if *x % 2 == 0 {
                *x *= 2;
                true
            } else {
                false
            }
        });
        assert_eq!(v.as_slice(), &[0, 4, 8]);
    }

    #[test]
    fn retain_mut_all_removed() {
        let mut v = Vec::new();
        for i in 0..3i32 {
            v.try_push(i).unwrap();
        }
        v.retain_mut(|_| false);
        assert!(v.is_empty());
    }

    #[test]
    fn retain_mut_none_removed() {
        let mut v = Vec::new();
        for i in 0..3i32 {
            v.try_push(i).unwrap();
        }
        v.retain_mut(|_| true);
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn dedup_by_key_consecutive() {
        let mut v = Vec::new();
        for i in [10, 20, 21, 30, 20] {
            v.try_push(i).unwrap();
        }
        v.dedup_by_key(|&mut x| x / 10);
        assert_eq!(v.as_slice(), &[10, 20, 30, 20]);
    }

    #[test]
    fn dedup_by_sorted_duplicates() {
        let mut v = Vec::new();
        for i in [1, 1, 2, 3, 3, 3, 4] {
            v.try_push(i).unwrap();
        }
        v.dedup_by_key(|&mut x| x);
        assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn dedup_by_single_element() {
        let mut v = Vec::new();
        v.try_push(42).unwrap();
        v.dedup_by_key(|&mut x| x);
        assert_eq!(v.as_slice(), &[42]);
    }

    #[test]
    fn dedup_by_custom_predicate() {
        let mut v: Vec<std::string::String> = Vec::new();
        for s in ["foo", "bar", "Bar", "baz", "bar"] {
            v.try_push(std::string::String::from(s)).unwrap();
        }
        v.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
        assert_eq!(
            v.as_slice(),
            &[
                std::string::String::from("foo"),
                std::string::String::from("bar"),
                std::string::String::from("baz"),
                std::string::String::from("bar")
            ]
        );
    }

    /// Regression test: `dedup_by` must leave the buffer in a consistent state if
    /// the predicate panics mid-loop. Before the drop-guard fix, a panic after some
    /// duplicates had been dropped would leak the tail and/or double-free the
    /// already-dropped slots when the Vec was unwound.
    #[test]
    fn dedup_by_panic_is_safe() {
        let ledger = Arc::new(Ledger::new());

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            || {
                let mut v: Vec<TrackedItem<i32>> = Vec::try_from_iter([
                    TrackedItem::construct(&ledger, 1),
                    TrackedItem::construct(&ledger, 1),
                    TrackedItem::construct(&ledger, 2),
                    TrackedItem::construct(&ledger, 3),
                ])
                .unwrap();
                // Panic on the third predicate call, mid-gap-fill, so some duplicate
                // elements have already been destroyed by the time we unwind.
                let mut calls = 0usize;
                v.dedup_by(|a, b| {
                    calls += 1;
                    if calls == 3 {
                        panic!("forced panic mid-dedup");
                    }
                    a.id() == b.id()
                });
            }
        }));

        assert!(result.is_err(), "expected the predicate to panic");
        // After the catch, the partially-deduplicated Vec has been unwound.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..4));
    }

    #[test]
    fn try_push_within_capacity_success() {
        let mut v: Vec<i32> = Vec::try_with_capacity(4).unwrap();
        v.try_push_within_capacity(1).unwrap();
        v.try_push_within_capacity(2).unwrap();
        v.try_push_within_capacity(3).unwrap();
        v.try_push_within_capacity(4).unwrap();
        assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn try_push_within_capacity_full() {
        let mut v: Vec<i32> = Vec::try_with_capacity(2).unwrap();
        v.try_push_within_capacity(1).unwrap();
        v.try_push_within_capacity(2).unwrap();
        let err = v.try_push_within_capacity(3).unwrap_err();
        assert_eq!(err.len, 2);
    }
}
