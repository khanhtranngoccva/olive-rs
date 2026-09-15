//! The borrowing iterator produced by `VecDeque::try_drain`.
//!
//! This borrows the deque and yields a *range* of its drained elements one at
//! a time.
//!
//! The implementation uses a verbatim copy of the std for stability due to a lack
//! of overflow risk.
use super::VecDeque;
use super::wrapped_index::WrappedIndex;
use crate::alloc::Global;
use core::fmt;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::size_of;
use core::mem::{self};
use core::ops::RangeBounds;
use core::ptr;
use core::ptr::NonNull;
use olive_core::alloc::Allocator;
use olive_core::slice::{TrySliceRangeError, try_range};

/// A draining iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`drain`] method on [`VecDeque`]. See its
/// documentation for more.
pub struct Drain<'a, T: 'a, A: Allocator = Global> {
    // We can't just use a &mut VecDeque<T, A>, as that would make Drain invariant over T
    // and we want it to be covariant instead
    pub(super) deque: NonNull<VecDeque<T, A>>,
    // drain_start is stored in deque.len
    pub(super) drain_len: usize,
    // index into the logical array, not the physical one (always lies in [0..deque.len))
    pub(super) idx: usize,
    // number of elements after the drained range
    pub(super) tail_len: usize,
    pub(super) remaining: usize,
    // Needed to make Drain covariant over T
    _marker: PhantomData<&'a T>,
}

impl<'a, T, A: Allocator> Drain<'a, T, A> {
    pub(super) unsafe fn new(
        deque: &'a mut VecDeque<T, A>,
        drain_start: usize,
        drain_len: usize,
    ) -> Self {
        // This is a guard against mem::forget.
        let orig_len = mem::replace(&mut deque.len, drain_start);
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "drain_start + drain_len == drain_end <= orig_len"
        )]
        let tail_len = orig_len - drain_start - drain_len;
        Drain {
            deque: NonNull::from(deque),
            drain_len,
            idx: drain_start,
            tail_len,
            remaining: drain_len,
            _marker: PhantomData,
        }
    }

    // Only returns pointers to the slices, as that's all we need
    // to drop them. May only be called if `self.remaining != 0`.
    pub(super) unsafe fn as_slices(&self) -> (*mut [T], *mut [T]) {
        // ignore-tidy-undocumented-unsafe
        unsafe {
            let deque = self.deque.as_ref();

            // We know that `self.idx + self.remaining <= deque.len <= usize::MAX`, so this won't overflow.
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "self.idx + self.remaining <= deque.len <= usize::MAX"
            )]
            let logical_remaining_range = self.idx..self.idx + self.remaining;

            // SAFETY: `logical_remaining_range` represents the
            // range into the logical buffer of elements that
            // haven't been drained yet, so they're all initialized,
            // and `slice::range(start..end, end) == start..end`,
            // so the preconditions for `slice_ranges` are met.
            let (a_range, b_range) = deque
                .try_slice_ranges(logical_remaining_range.clone(), logical_remaining_range.end)
                .expect("logical remaining range is within bounds");
            (deque.buffer_range(a_range), deque.buffer_range(b_range))
        }
    }
}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for Drain<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Drain")
            .field(&self.drain_len)
            .field(&self.idx)
            .field(&self.tail_len)
            .field(&self.remaining)
            .finish()
    }
}

unsafe impl<T: Sync, A: Allocator + Sync> Sync for Drain<'_, T, A> {}
unsafe impl<T: Send, A: Allocator + Send> Send for Drain<'_, T, A> {}

impl<T, A: Allocator> Drop for Drain<'_, T, A> {
    fn drop(&mut self) {
        struct DropGuard<'r, 'a, T, A: Allocator>(&'r mut Drain<'a, T, A>);

        let guard = DropGuard(self);

        if mem::needs_drop::<T>() && guard.0.remaining != 0 {
            // SAFETY: We just checked that `self.remaining != 0`.
            let (front, back) = unsafe { guard.0.as_slices() };
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted front + back = remaining (as_slices invariant) => front <= remaining,
                idx + front <= idx + remaining == len <= usize::MAX"
            )]
            {
                guard.0.idx += front.len();
                guard.0.remaining -= front.len();
            }
            // SAFETY: This can't have been dropped before since
            // `idx` & `remaining` track what's been dropped.
            unsafe { ptr::drop_in_place(front) };
            guard.0.remaining = 0;
            // SAFETY: Ditto.
            unsafe { ptr::drop_in_place(back) };
        }

        // Dropping `guard` handles moving the remaining elements into place.
        impl<T, A: Allocator> Drop for DropGuard<'_, '_, T, A> {
            #[inline]
            fn drop(&mut self) {
                if mem::needs_drop::<T>() && self.0.remaining != 0 {
                    // SAFETY: We just checked that `self.remaining != 0`.
                    unsafe {
                        let (front, back) = self.0.as_slices();
                        ptr::drop_in_place(front);
                        ptr::drop_in_place(back);
                    }
                }

                // At this point, all drained items should be completely dropped as if they
                // never existed.

                // ignore-tidy-undocumented-unsafe
                let source_deque = unsafe { self.0.deque.as_mut() };

                let drain_len = self.0.drain_len;
                let head_len = source_deque.len; // #elements in front of the drain
                let tail_len = self.0.tail_len; // #elements behind the drain
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "asserted head_len + tail_len <= orig_len <= capacity"
                )]
                let new_len = head_len + tail_len;

                if size_of::<T>() == 0 {
                    // no need to copy around any memory if T is a ZST
                    source_deque.len = new_len;
                    return;
                }

                // Next, we will fill the hole left by the drain with as few writes as possible.
                // The code below handles the following control flow and reduces the amount of
                // branches under the assumption that `head_len == 0 || tail_len == 0`, i.e.
                // draining at the front or at the back of the dequeue is especially common.
                //
                // H = "head index" = `deque.head`
                // h = elements in front of the drain
                // d = elements in the drain
                // t = elements behind the drain
                //
                // Note that the buffer may wrap at any point and the wrapping is handled by
                // `wrap_copy` and `to_physical_idx`.
                //
                // Case 1: if `head_len == 0 && tail_len == 0`
                // Everything was drained, reset the head index back to 0.
                //             H
                // [ . . . . . d d d d . . . . . ]
                //   H
                // [ . . . . . . . . . . . . . . ]
                //
                // Case 2: else if `tail_len == 0`
                // Don't move data or the head index.
                //         H
                // [ . . . h h h h d d d d . . . ]
                //         H
                // [ . . . h h h h . . . . . . . ]
                //
                // Case 3: else if `head_len == 0`
                // (`tail_len` != 0, `head_len` < `tail_len`)
                // Don't move data, but move the head index.
                //         H
                // [ . . . d d d d t t t t . . . ]
                //                 H
                // [ . . . . . . . t t t t . . . ]
                //
                // Case 4: else if `tail_len <= head_len`
                // Move tail data, but not the head index.
                //       H
                // [ . . h h h h d d d d t t . . ]
                //       H
                // [ . . h h h h t t . . . . . . ]
                //
                // Case 5: else
                // Move head data and the head index.
                //       H
                // [ . . h h d d d d t t t t . . ]
                //               H
                // [ . . . . . . h h t t t t . . ]

                // When draining at the front (`.drain(..n)`) or at the back (`.drain(n..)`),
                // we don't need to copy any data. The number of elements copied would be 0.
                // This branch *cannot* be executed if either condition fails due to possibly of
                // breaking < capacity invariant.
                if head_len != 0 && tail_len != 0 {
                    join_head_and_tail_wrapping(source_deque, drain_len, head_len, tail_len);
                    // Marking this function as cold helps LLVM to eliminate it entirely if
                    // this branch is never taken.
                    // We use `#[cold]` instead of `#[inline(never)]`, because inlining this
                    // function into the general case (`.drain(n..m)`) is fine.
                    // See `tests/codegen-llvm/vecdeque-drain.rs` for a test.
                    #[cold]
                    fn join_head_and_tail_wrapping<T, A: Allocator>(
                        source_deque: &mut VecDeque<T, A>,
                        drain_len: usize,
                        head_len: usize,
                        tail_len: usize,
                    ) {
                        // Pick whether to move the head or the tail here.
                        let (src, dst, len);
                        if head_len < tail_len {
                            src = source_deque.head;
                            // SAFETY: head_len >= 1, tail_len >= 1, drain_len < capacity
                            dst = unsafe { source_deque.to_wrapped_index(drain_len) };
                            len = head_len;
                        } else {
                            // `head_len + drain_len` is the logical index of the
                            // first tail element, which is always `< capacity`.
                            #[allow(
                                clippy::arithmetic_side_effects,
                                reason = "asserted tail_len >= 1 => head_len < capacity, 
                                tail_len < capacity, head_len + tail_len < capacity"
                            )]
                            let src_idx = head_len + drain_len;
                            // SAFETY: Specified in above clippy lint.
                            src = unsafe { source_deque.to_wrapped_index(src_idx) };
                            // SAFETY: Specified in above clippy lint.
                            dst = unsafe { source_deque.to_wrapped_index(head_len) };
                            len = tail_len;
                        };

                        // ignore-tidy-undocumented-unsafe
                        unsafe {
                            source_deque.wrap_copy(src, dst, len);
                        }
                    }
                }

                if new_len == 0 {
                    // Special case: If the entire deque was drained, reset the head back to 0,
                    // like `.clear()` does.
                    source_deque.head = WrappedIndex::zero();
                } else if head_len < tail_len {
                    // If we moved the head above, then we need to adjust the head index here.
                    // source_deque has length head_len
                    // SAFETY: tail_len > 0 => drain_len < len <= capacity
                    source_deque.head = unsafe { source_deque.to_wrapped_index(drain_len) };
                }
                source_deque.len = new_len;
            }
        }
    }
}

impl<T, A: Allocator> Iterator for Drain<'_, T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        // SAFETY: idx == len (possibly == capacity) only if self.remaining == 0 (eliminated above)
        // so idx < capacity
        let wrapped_idx = unsafe { self.deque.as_ref().to_wrapped_index(self.idx) };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "idx < capacity and remaining > 0, so both stay in bounds"
        )]
        {
            self.idx += 1;
            self.remaining -= 1;
        }
        // ignore-tidy-undocumented-unsafe
        Some(unsafe { ptr::read(self.deque.as_mut().buf.ptr().add(wrapped_idx.as_index())) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining;
        (len, Some(len))
    }
}

impl<T, A: Allocator> DoubleEndedIterator for Drain<'_, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        // `remaining > 0` is guaranteed by the guard above, so the decrement
        // cannot underflow. The sum `idx + remaining` is a logical index into
        // `[drain_start..orig_len]`, always `< capacity`.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "remaining > 0 and idx + remaining < capacity"
        )]
        {
            self.remaining -= 1;
            let back_idx = self.idx + self.remaining;
            // SAFETY: idx + old_remaining may be == len == capacity, idx + old_remaining - 1 < len
            let wrapped_idx = unsafe { self.deque.as_ref().to_wrapped_index(back_idx) };
            // ignore-tidy-undocumented-unsafe
            Some(unsafe { ptr::read(self.deque.as_mut().buf.ptr().add(wrapped_idx.as_index())) })
        }
    }
}

impl<T, A: Allocator> ExactSizeIterator for Drain<'_, T, A> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<T, A: Allocator> FusedIterator for Drain<'_, T, A> {}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Removes a range of elements from the deque and returns them as an
    /// iterator, shifting later elements forward to fill the gap.
    ///
    /// This is the fallible-port analogue of `std`'s `VecDeque::drain`. The
    /// only failure mode is validating the requested range: iteration itself
    /// never allocates (it merely destroys and shifts elements), so once the
    /// drainer is constructed it cannot fail.
    ///
    /// # Errors
    ///
    /// Returns [`TrySliceRangeError`] if the resolved range is out of bounds
    /// or reversed.
    ///
    /// # Leaking
    ///
    /// If [`mem::forget`](core::mem::forget) is called on the returned
    /// iterator, the deque is left in an inconsistent state: the drained range
    /// is excluded from its length but the tail has not been shifted.
    pub fn try_drain<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<Drain<'_, T, A>, TrySliceRangeError> {
        let len = self.len();
        let r = try_range(range, ..len)?;
        let start = r.start;
        let end = r.end;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start <= end (guaranteed by try_range)"
        )]
        let count = end - start;

        // SAFETY: `try_range` guarantees `start <= end <= len`, so
        // `drain_start < len` when `count > 0`, and all slots in
        // `[0..len)` are initialized. The constructor caps `deque.len`
        // to `drain_start` internally.
        let drain = unsafe { Drain::new(self, start, count) };
        Ok(drain)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use crate::collections::vec_deque::VecDeque;

    /// Collect the deque's logical contents into a `std::vec::Vec`.
    fn collect_logical(dq: &VecDeque<i32>) -> std::vec::Vec<i32> {
        let (a, b) = dq.as_slices();
        a.iter().chain(b.iter()).copied().collect()
    }

    #[test]
    fn drain_full_range_non_wrapped() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let drained: std::vec::Vec<i32> = dq.try_drain(..).unwrap().collect();
        assert_eq!(drained, std::vec![1, 2, 3, 4, 5]);
        assert_eq!(dq.head.as_index(), 0);
        assert!(dq.is_empty());
    }

    #[test]
    fn drain_middle_range_non_wrapped() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [0, 1, 2, 3, 4] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let drained: std::vec::Vec<i32> = dq.try_drain(1..3).unwrap().collect();
        assert_eq!(drained, std::vec![1, 2]);
        assert_eq!(collect_logical(&dq), std::vec![0, 3, 4]);
    }

    #[test]
    fn drain_front_range() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let drained: std::vec::Vec<i32> = dq.try_drain(..2).unwrap().collect();
        assert_eq!(drained, std::vec![1, 2]);
        assert_eq!(collect_logical(&dq), std::vec![3, 4, 5]);
    }

    #[test]
    fn drain_back_range() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let drained: std::vec::Vec<i32> = dq.try_drain(3..).unwrap().collect();
        assert_eq!(drained, std::vec![4, 5]);
        assert_eq!(collect_logical(&dq), std::vec![1, 2, 3]);
    }

    #[test]
    fn drain_single_element() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let drained: std::vec::Vec<i32> = dq.try_drain(1..2).unwrap().collect();
        assert_eq!(drained, std::vec![2]);
        assert_eq!(collect_logical(&dq), std::vec![1, 3]);
    }

    #[test]
    fn drain_empty_range_is_noop() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let drained: std::vec::Vec<i32> = dq.try_drain(1..1).unwrap().collect();
        assert!(drained.is_empty());
        assert_eq!(collect_logical(&dq), std::vec![1, 2, 3]);
    }

    #[test]
    fn drain_empty_range_in_empty_deque() {
        let mut dq = VecDeque::<i32>::new();
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let drained: std::vec::Vec<i32> = dq.try_drain(0..0).unwrap().collect();
        assert!(drained.is_empty());
        assert_eq!(collect_logical(&dq), std::vec![]);
    }

    #[test]
    fn drain_wrapped_buffer() {
        // Build a wrapped state: push backs then fronts to wrap head past 0.
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(5), Ok(()));
        // Logical: [5, 4, 1, 2, 3], head has wrapped.
        assert_eq!(collect_logical(&dq), std::vec![5, 4, 1, 2, 3]);

        // Drain the middle three: logical [1..4) = [4, 1, 2].
        let drained: std::vec::Vec<i32> = dq.try_drain(1..4).unwrap().collect();
        assert_eq!(drained, std::vec![4, 1, 2]);
        // Remaining: [5, 3].
        assert_eq!(collect_logical(&dq), std::vec![5, 3]);
    }

    #[test]
    fn drain_entire_wrapped_buffer() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("ok");
        for v in [1, 2] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_front_within_capacity(3), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        // Logical: [4, 3, 1, 2], full buffer, head wrapped.
        assert_eq!(collect_logical(&dq), std::vec![4, 3, 1, 2]);

        let drained: std::vec::Vec<i32> = dq.try_drain(..).unwrap().collect();
        assert_eq!(drained, std::vec![4, 3, 1, 2]);
        assert!(dq.is_empty());
    }

    #[test]
    // FIXME: should use ledger
    fn drain_partial_consumption_drops_rest() {
        // Use a type with observable drop behavior.
        #[derive(Debug)]
        struct Dropped(std::sync::Arc<std::cell::Cell<u32>>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let counter = std::sync::Arc::new(std::cell::Cell::new(0u32));
        let mut dq: VecDeque<Dropped> = VecDeque::try_with_capacity(8).expect("ok");
        for _i in 0..5 {
            let d = Dropped(counter.clone());
            assert_eq!(dq.try_push_back_within_capacity(d), Ok(()));
        }
        assert_eq!(counter.get(), 0);

        // Drain all 5 but only consume 2.
        let mut drain = dq.try_drain(..).unwrap();
        assert!(drain.next().is_some());
        assert!(drain.next().is_some());
        // Drop the drainer with 3 unconsumed elements.
        drop(drain);
        // All 5 should be dropped: 2 moved out (dropped when the Option is
        // discarded) + 3 destroyed by the drainer's Drop.
        assert_eq!(counter.get(), 5);
        assert!(dq.is_empty());
    }

    #[test]
    fn drain_next_back_works() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let drained: std::vec::Vec<i32> = {
            let mut drain = dq.try_drain(..).unwrap();
            let out = std::vec![
                drain.next_back().unwrap(),
                drain.next_back().unwrap(),
                drain.next().unwrap(),
                drain.next().unwrap(),
                drain.next_back().unwrap()
            ];
            assert_eq!(drain.next(), None);
            out
        };
        // Order consumed: 5, 4, 1, 2, 3 — the full sequence in mixed order.
        assert_eq!(drained, std::vec![5, 4, 1, 2, 3]);
        assert!(dq.is_empty());
    }

    #[test]
    fn drain_zst() {
        let mut dq: VecDeque<()> = VecDeque::new();
        for _ in 0..5 {
            assert_eq!(dq.try_push_back_within_capacity(()), Ok(()));
        }
        let drained: std::vec::Vec<()> = dq.try_drain(1..4).unwrap().collect();
        assert_eq!(drained.len(), 3);
        assert_eq!(dq.len(), 2);
    }

    #[test]
    fn drain_out_of_bounds_returns_error() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert!(dq.try_drain(0..10).is_err());
        // Deque unchanged.
        assert_eq!(collect_logical(&dq), std::vec![1, 2, 3]);
    }

    #[test]
    fn drain_exact_size_iterator() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3, 4] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let mut drain = dq.try_drain(1..3).unwrap();
        assert_eq!(drain.len(), 2);
        assert_ne!(drain.len(), 0);
        assert_eq!(drain.size_hint(), (2, Some(2)));
        drain.next();
        assert_eq!(drain.len(), 1);
        drain.next();
        assert_eq!(drain.len(), 0);
    }

    #[test]
    fn drain_multiple_sequential() {
        let mut dq = VecDeque::<i32>::try_with_capacity(16).expect("ok");
        for v in 0..10 {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        // Drain [2..5)
        let d1: std::vec::Vec<i32> = dq.try_drain(2..5).unwrap().collect();
        assert_eq!(d1, std::vec![2, 3, 4]);
        assert_eq!(collect_logical(&dq), std::vec![0, 1, 5, 6, 7, 8, 9]);
        // Drain [0..2)
        let d2: std::vec::Vec<i32> = dq.try_drain(0..2).unwrap().collect();
        assert_eq!(d2, std::vec![0, 1]);
        assert_eq!(collect_logical(&dq), std::vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn drain_after_pops_creates_gap() {
        // Simulate a scenario where pops from the front leave a gap.
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3, 4, 5, 6] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        // Pop two from the front: [3, 4, 5, 6], head advanced.
        assert_eq!(dq.pop_front(), Some(1));
        assert_eq!(dq.pop_front(), Some(2));
        // Now drain the middle: logical [1..3) = [4, 5].
        let drained: std::vec::Vec<i32> = dq.try_drain(1..3).unwrap().collect();
        assert_eq!(drained, std::vec![4, 5]);
        assert_eq!(collect_logical(&dq), std::vec![3, 6]);
    }

    #[test]
    fn drain_debug_impl() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let dbg = {
            let drain = dq.try_drain(..).unwrap();
            std::format!("{:?}", drain)
        };
        assert!(dbg.contains("Drain"));
    }
}
