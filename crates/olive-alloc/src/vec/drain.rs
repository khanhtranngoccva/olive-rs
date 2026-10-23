//! The borrowing iterator produced by `Vec::try_drain`.
//!
//! This borrows the vector and yields a *range* of its drained elements one at a time.

use super::Vec;
use crate::alloc::Global;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ptr;
use olive_core::alloc::Allocator;
use olive_core::slice::{self, TrySliceRangeError};

/// An iterator that drains a *range* of a vector, yielding owned elements while
/// removing them from the vector.
///
/// This `struct` is created by the
/// [`try_drain`](super::Vec::try_drain) method on
/// [`Vec`](super::Vec).
///
/// # Example
///
/// ```
/// use olive_alloc::vec::Vec;
/// let mut v = Vec::new();
/// for i in 0..5 { v.try_push(i).unwrap(); }
/// // Drain the middle two elements.
/// let drained: std::vec::Vec<i32> = v.try_drain(1..3).unwrap().collect();
/// assert_eq!(drained, std::vec![1, 2]);
/// // The remaining elements are compacted forward.
/// assert_eq!(v.as_slice(), &[0, 3, 4]);
/// ```
pub struct Drain<'a, T, A: Allocator = Global> {
    /// Number of elements already yielded from the front (via `next`). Together
    /// with `taken_back` this partitions the drain range into three regions:
    /// `[start + consumed, start + count - taken_back)` is still live.
    consumed: usize,
    /// Number of elements already yielded from the back (via `next_back`).
    taken_back: usize,
    /// Offset (in elements) of the drain range's start from the vector's base
    /// pointer. Because `try_drain` caps the vector's length to `start` at
    /// construction, this also equals the vec's length for the whole lifetime
    /// of the drainer — which is what keeps the drained hole invisible to the
    /// vec's own drop path.
    original_start: usize,
    /// Total number of elements in the resolved drain range (`end - start`).
    /// The entire range is removed when this iterator is dropped, regardless of
    /// how many elements were actually yielded.
    original_count: usize,
    /// The vector's length *before* it was capped down to `original_start`.
    /// Retained because capping makes the surviving tail's extent no longer
    /// derivable from `vec.len()` alone; `drop` uses this to relocate that
    /// tail, which sits at offsets `[original_start + original_count,
    /// original_len)`.
    original_len: usize,
    /// Mutable pointer back to the owning `Vec`, used to read element slots and
    /// to compact the surviving tail into place when this iterator is dropped.
    /// Derived from the enclosing `&mut self` borrow, so it carries mutable
    /// provenance (keeps Miri's Stacked Borrows model happy). We store a raw
    /// pointer rather than a `&'a mut Vec` so the `Drop` impl can reach through
    /// it without fighting the borrow checker over reborrows, and so the
    /// iterator's exclusive window over the buffer stays explicit.
    vec: *mut Vec<T, A>,
    _marker: PhantomData<&'a ()>,
}

// SAFETY: mirroring `Vec`, the iterator hands out owned values only and holds a
// unique mutable view of the buffer for its lifetime. Sound iff `T` and `A` are
// `Send`/`Sync`.
unsafe impl<T: Send, A: Allocator + Send> Send for Drain<'_, T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for Drain<'_, T, A> {}

impl<T, A: Allocator> Drain<'_, T, A> {
    /// Constructs a drainer from the raw parts of a borrowed `Vec`.
    ///
    /// No pointer arithmetic is performed here: the caller passes the already-
    /// validated integer bounds and the owning `Vec`, and all element pointers
    /// are derived lazily (and only while a live element provably exists).
    ///
    /// # Safety
    ///
    /// `count == end - start` must be the full resolved range length and
    /// `start + count <= original_len`; the first `original_len` slots of the
    /// `Vec` pointed to by `vec` must be initialized.
    ///
    /// The caller is expected to have already capped `vec`'s length down
    /// to `start` before calling this, so that the drained hole is excluded
    /// from the vector's live contents.
    #[inline]
    pub(super) const unsafe fn new_from_parts(
        start: usize,
        count: usize,
        original_len: usize,
        vec: *mut Vec<T, A>,
    ) -> Self {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "precondition needs to be met by caller"
        )]
        {
            debug_assert!(start + count <= original_len);
        }
        debug_assert!(unsafe { &*vec }.len() == start);
        Self {
            consumed: 0,
            taken_back: 0,
            original_start: start,
            original_count: count,
            original_len,
            vec,
            _marker: PhantomData,
        }
    }

    /// Number of elements still held in the live window.
    #[inline]
    const fn remaining(&self) -> usize {
        // `consumed + taken_back` can never exceed `original_count`: each yield
        // decrements the live window by exactly one and stops at zero.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "consumed + taken_back <= original_count"
        )]
        let taken = { self.consumed + self.taken_back };
        #[allow(clippy::arithmetic_side_effects, reason = "taken <= original_count")]
        {
            self.original_count - taken
        }
    }

    /// Returns the number of elements still held by this iterator.
    #[inline]
    pub const fn len(&self) -> usize {
        self.remaining()
    }

    /// Returns `true` if all elements have been yielded.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Views the remaining (not-yet-yielded) elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // The computed front pointer is invalid only when it lands one-past-the-
        // end of the buffer: `front_offset() == original_len`. When that case happens,
        // the fallback must be used. This fallback behavior is triggered sparingly
        // for conformance with std. (this also happens to work with ZSTs)
        if self.front_offset() == self.original_len {
            return unsafe { slice::from_raw_parts(self.base_ptr(), 0) };
        }
        let rem = self.remaining();
        // SAFETY: `front_offset() < original_len`;
        // the front of the live window addresses a real, initialized slot (or a
        // valid dangling-aligned ZST address), and `[front, front + rem)` lies
        // entirely within the initialized region of the buffer.
        unsafe {
            let vec = &*self.vec;
            let front = vec.as_ptr().add(self.front_offset());
            slice::from_raw_parts(front, rem)
        }
    }

    /// Base pointer of the owning vector's buffer. Only ever combined with an
    /// offset that provably lands on a live slot (guarded by `remaining > 0`)
    /// or with length zero, so no OOB pointer is formed.
    #[inline]
    fn base_ptr(&self) -> *const T {
        // SAFETY: `self.vec` points to the live `Vec` borrowed exclusively by
        // this drainer for `'a`.
        unsafe { (*self.vec).as_ptr() }
    }

    /// Element offset of the current front of the live window.
    #[inline]
    fn front_offset(&self) -> usize {
        #[allow(clippy::arithmetic_side_effects, reason = "consumed <= original_count")]
        {
            self.original_start + self.consumed
        }
    }

    /// Element offset of the current back of the live window (the last live
    /// element is at `back_offset() - 1`).
    #[inline]
    fn back_offset(&self) -> usize {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start + count == end <= original_len"
        )]
        let end = { self.original_start + self.original_count };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "taken_back <= original_count"
        )]
        {
            end - self.taken_back
        }
    }
}

impl<T, A: Allocator> Iterator for Drain<'_, T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining() == 0 {
            return None;
        }
        // Record the front offset *before* advancing; it addresses a live,
        // initialized slot because we just confirmed `remaining > 0`.
        let offset = self.front_offset();
        #[allow(clippy::arithmetic_side_effects, reason = "consumed < original_count")]
        {
            self.consumed += 1;
        }
        // SAFETY: `offset` is within the initialized region and the value is now
        // ours to move out. Reading a ZST from its (possibly dangling-aligned)
        // pointer is well defined. The borrow is exclusive for `'a`.
        Some(unsafe {
            let vec = &mut *self.vec;
            ptr::read(vec.as_mut_ptr().add(offset))
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining(), Some(self.remaining()))
    }
}

impl<T, A: Allocator> ExactSizeIterator for Drain<'_, T, A> {}

impl<T, A: Allocator> FusedIterator for Drain<'_, T, A> {}

impl<T, A: Allocator> DoubleEndedIterator for Drain<'_, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.remaining() == 0 {
            return None;
        }
        // The last live element sits just below the current back boundary.
        // Compute its offset *before* recording the take so it addresses a
        // live, initialized slot.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "remaining > 0 implies back_offset >= original_start + 1"
        )]
        let offset = { self.back_offset() - 1 };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "taken_back < original_count"
        )]
        {
            self.taken_back += 1;
        }
        // SAFETY: as in `next`, `offset` is within the initialized region and
        // the value becomes the caller's. The borrow is exclusive for `'a`.
        Some(unsafe {
            let vec = &mut *self.vec;
            ptr::read(vec.as_mut_ptr().add(offset))
        })
    }
}

impl<T, A: Allocator> Drop for Drain<'_, T, A> {
    fn drop(&mut self) {
        let vec = unsafe { &mut *self.vec };

        let original_start = self.original_start;
        let original_count = self.original_count;
        let original_len = self.original_len;
        let consumed = self.consumed;
        let remaining = self.remaining();

        // Buffer layout (offsets from the base pointer):
        //   [prefix:           0..original_start)                              <- in vec.len(), survives
        //   [yielded left:     original_start..original_start+cons)            <- moved out (uninit bits)
        //   [hole:             original_start+cons..original_end-taken_back)   <- never yielded, must drop
        //   [yielded right:    original_end-taken_back..original_end)          <- moved out (uninit bits)
        //   [suffix:           original_end..original_len)                     <- beyond vec.len(), shifts left
        // where cons + taken_back + remaining = original_count
        // and original_end = original_start + original_count.
        //
        // Steps:
        // 1. Destroy the unconsumed ("hole") elements.
        // 2. Shift the suffix left by `original_count` to close the full gap.
        // 3. Extend the length past the prefix to include the compacted suffix.
        //
        // Only step 1 can panic: `drop_in_place` over a fat slice runs every
        // `T` destructor in the hole, whereas step 2 (`ptr::copy`) is a pure
        // bitwise move that drops nothing and step 3 (`set_len`) just writes a
        // field. Because the compiler lowers a fat-slice drop to a per-element
        // sequence inside one function body, if one destructor panics mid-hole
        // the unwinder still runs the remaining destructors as landing pads on
        // the same frame before leaving it — so no hole element is leaked. What
        // the unwind does NOT do is run any code placed after the
        // `drop_in_place` call in this same `Drop` body; that is why the
        // compaction (steps 2 and 3) lives in a separate guard whose `Drop`
        // runs unconditionally. Arming the guard *before* step 1 guarantees the
        // compaction runs both on the happy path (guard falls out of scope
        // normally) and on unwind (the guard's `Drop` runs during stack
        // teardown).
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "original_count <= original_len"
        )]
        let final_len = original_len - original_count;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start + count == end <= original_len"
        )]
        let suffix_src_offset = original_start + original_count;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "suffix_src_offset <= original_len"
        )]
        let suffix_len = original_len - suffix_src_offset;
        let base = vec.as_mut_ptr();
        let _compact_guard = DrainCompactGuard {
            vec,
            base,
            original_start,
            suffix_src_offset,
            suffix_len,
            final_len,
        };

        // Step 1: destroy the unconsumed remainder of the drain range.
        if remaining > 0 {
            #[allow(clippy::arithmetic_side_effects, reason = "consumed <= original_count")]
            let hole_offset = original_start + consumed;
            unsafe {
                let p = base.add(hole_offset);
                ptr::drop_in_place(slice::from_raw_parts_mut(p, remaining));
            }
        }

        // Steps 2 (shift) and 3 (length restore) are performed by the guard's
        // `Drop`, which runs unconditionally — on the happy path when the guard
        // falls out of scope, and on unwind if step 1 panicked. See
        // `DrainCompactGuard` below.
    }
}

/// Finishes a drain's compaction — shifting the surviving suffix left and
/// restoring the vector's length — even if the preceding element destruction
/// unwinds.
struct DrainCompactGuard<'a, T, A: Allocator> {
    /// Exclusive reference to the owning vector being compacted.
    vec: &'a mut Vec<T, A>,
    /// Base of the vector's buffer; offsets below are relative to it.
    base: *mut T,
    /// Offset where the drained range began; the suffix lands here after the
    /// leftward shift.
    original_start: usize,
    /// Offset where the suffix currently begins: `original_start + count`.
    suffix_src_offset: usize,
    /// Number of suffix elements to relocate.
    suffix_len: usize,
    /// Length to restore: `original_len - original_count`, i.e. prefix plus the
    /// compacted suffix. Always `<= capacity()`.
    final_len: usize,
}

impl<T, A: Allocator> Drop for DrainCompactGuard<'_, T, A> {
    fn drop(&mut self) {
        // Step 2: shift the suffix left by the drained count.
        // SAFETY: both pointers are within the allocation and aligned;
        // `ptr::copy` correctly handles the overlapping case (leftward shift).
        // For ZSTs this is a no-op. OOB pointer is prevented with a suffix_len
        // guard.
        if self.suffix_len > 0 && size_of::<T>() != 0 {
            unsafe {
                let dst = self.base.add(self.original_start);
                let src = self.base.add(self.suffix_src_offset);
                ptr::copy(src, dst, self.suffix_len);
            }
        }

        // Step 3: extend the length to include the compacted suffix.
        // SAFETY: every slot in `[0..final_len)` is initialized (the
        // untouched prefix plus the relocated suffix) and `final_len <=
        // capacity()`, satisfying `set_len`'s preconditions.
        unsafe {
            self.vec.set_len(self.final_len);
        }
    }
}

impl<T, A: Allocator> Vec<T, A> {
    /// Removes a range of elements from the vector and returns them as an
    /// iterator, shifting later elements to the left to fill the gap.
    ///
    /// This is the fallible-port analogue of `std`'s `Vec::drain`. The only
    /// failure mode is validating the requested range: iteration itself never
    /// allocates (it merely destroys and shifts elements), so once the drainer
    /// is constructed it cannot fail.
    ///
    /// # Errors
    ///
    /// Returns [`TrySliceRangeError`] if the resolved range is out of bounds or
    /// reversed.
    ///
    /// # Leaking
    ///
    /// If [`mem::forget`](core::mem::forget) is called, all elements from the
    /// start of `range` to the end of the vector, except the extracted elements,
    /// are leaked.
    pub fn try_drain<R: core::ops::RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<Drain<'_, T, A>, TrySliceRangeError> {
        let len = self.len();
        let r = slice::try_range(range, ..len)?;
        let start = r.start;
        let end = r.end;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start <= end (guaranteed by try_range)"
        )]
        let count = end - start;
        // Cap the vector's logical length down to `start` *before* handing out
        // the drainer. Once `next()` has yielded an element via `ptr::read`,
        // that slot is uninitialized; if the vec still counted it in its length,
        // a later drop of the vec (e.g. after the drainer is forgotten) would
        // drop uninitialized memory. Capping excludes the whole drained range
        // from the vec's live contents up front, so the vec can never touch the
        // hole. This mirrors std's internal bookkeeping.
        //
        // SAFETY: `start <= len <= capacity()`, and every slot in `[0..start)`
        // is an initialized value, so capping to `start` preserves the invariant
        // that all live slots are valid.
        unsafe {
            self.set_len(start);
        }
        // SAFETY: `count == end - start` is the full resolved range length and
        // `start + count == end <= len`; the first `len` slots are initialized.
        // The vector already had its length capped
        let drain = unsafe { Drain::new_from_parts(start, count, len, &raw mut *self) };
        Ok(drain)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use olive_test_helpers::{CloneBudget, Counter, FlakyCloneTrackedItem, Ledger, PanicArmer};
    use std::sync::Arc;

    #[test]
    fn drain_as_slice() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let mut d = v.try_drain(1..4).unwrap();
        assert_eq!(d.as_slice(), &[1, 2, 3]);
        let _ = d.next();
        assert_eq!(d.as_slice(), &[2, 3]);
    }

    #[test]
    fn drain_from_front() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let drained: std::vec::Vec<i32> = v.try_drain(..2).unwrap().collect();
        assert_eq!(drained, std::vec![0, 1]);
        assert_eq!(v.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn drain_at_end() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let drained: std::vec::Vec<i32> = v.try_drain(3..).unwrap().collect();
        assert_eq!(drained, std::vec![3, 4]);
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn drain_entire_vec() {
        let mut v = Vec::new();
        for i in 0..4i32 {
            v.try_push(i).unwrap();
        }
        let drained: std::vec::Vec<i32> = v.try_drain(..).unwrap().collect();
        assert_eq!(drained, std::vec![0, 1, 2, 3]);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn drain_middle_range() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let drained: std::vec::Vec<i32> = v.try_drain(1..3).unwrap().collect();
        assert_eq!(drained, std::vec![1, 2]);
        assert_eq!(v.as_slice(), &[0, 3, 4]);
    }

    #[test]
    fn drain_single_element() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let drained: std::vec::Vec<i32> = v.try_drain(2..3).unwrap().collect();
        assert_eq!(drained, std::vec![2]);
        assert_eq!(v.as_slice(), &[0, 1, 3, 4]);
    }

    #[test]
    fn drain_empty_range() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let drained: std::vec::Vec<i32> = v.try_drain(2..2).unwrap().collect();
        assert!(drained.is_empty());
        assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn drain_out_of_bounds_errors() {
        let mut v = Vec::new();
        for i in 0..3i32 {
            v.try_push(i).unwrap();
        }
        assert!(v.try_drain(3..5).is_err());
        // Reversed range must be rejected (lint suppressed: this is the point of the test).
        #[allow(
            clippy::reversed_empty_ranges,
            reason = "intentionally testing a reversed range"
        )]
        {
            assert!(v.try_drain(2..1).is_err());
        }
        assert!(v.try_drain(..100).is_err());
        // Vec is unmodified by failed drains.
        assert_eq!(v.as_slice(), &[0, 1, 2]);
        // Valid: empty range (does not mutate).
        let d = v.try_drain(1..1).unwrap();
        assert_eq!(d.len(), 0);
        drop(d);
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn drain_double_ended_mixed() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let mut d = v.try_drain(1..4).unwrap();
        assert_eq!(d.next_back(), Some(3));
        assert_eq!(d.next(), Some(1));
        drop(d);
        // Elements 1 and 3 yielded, element 2 destroyed. Range [1..4) fully removed.
        assert_eq!(v.as_slice(), &[0, 4]);
    }

    #[test]
    fn drain_zst() {
        let mut v: Vec<()> = Vec::new();
        for _ in 0..5 {
            v.try_push(()).unwrap();
        }
        let count = v.try_drain(1..3).unwrap().count();
        assert_eq!(count, 2);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn drain_zst_len_size_hint_is_empty() {
        let mut v: Vec<()> = Vec::new();
        for _ in 0..5 {
            v.try_push(()).unwrap();
        }
        // Drain the middle range [1, 4): three elements outstanding.
        let mut d = v.try_drain(1..4).unwrap();
        assert_eq!(d.len(), 3);
        assert_eq!(d.size_hint(), (3, Some(3)));
        assert_eq!(d.as_slice().len(), 3);

        // Yield one from each end: one remains, and the slice length tracks it.
        assert_eq!(d.next(), Some(()));
        assert_eq!(d.next_back(), Some(()));
        assert_eq!(d.len(), 1);
        assert_eq!(d.size_hint(), (1, Some(1)));
        assert_eq!(d.as_slice().len(), 1);

        // Exhaustion: all views agree on 0.
        assert_eq!(d.next(), Some(()));
        assert_eq!(d.len(), 0);
        assert_eq!(d.as_slice().len(), 0);
        assert!(d.is_empty());
        drop(d);
    }

    #[test]
    fn drain_len_size_hint_is_empty() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let mut d = v.try_drain(1..4).unwrap();
        assert_eq!(d.len(), 3);
        assert_eq!(d.size_hint(), (3, Some(3)));
        assert!(!d.is_empty());
        assert_eq!(d.next(), Some(1));
        assert_eq!(d.len(), 2);
        assert_eq!(d.size_hint(), (2, Some(2)));
        assert_eq!(d.next_back(), Some(3));
        assert_eq!(d.len(), 1);
        assert_eq!(d.size_hint(), (1, Some(1)));
        assert_eq!(d.next(), Some(2));
        assert_eq!(d.len(), 0);
        assert_eq!(d.size_hint(), (0, Some(0)));
        assert!(d.is_empty());
        assert_eq!(d.next(), None);
    }

    #[test]
    fn drain_partial_consume_drops_rest_of_range() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let mut d = v.try_drain(1..3).unwrap();
        assert_eq!(d.next(), Some(1));
        drop(d);
        // Element 1 yielded, element 2 destroyed. Entire range removed.
        assert_eq!(v.as_slice(), &[0, 3, 4]);
    }

    #[test]
    fn drain_no_consume_still_removes_range() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let d = v.try_drain(1..3).unwrap();
        drop(d);
        assert_eq!(v.as_slice(), &[0, 3, 4]);
    }

    #[test]
    fn drain_fully_collected_compacts_and_drops_once() {
        let counter = Arc::new(Counter::new());

        #[allow(dead_code)]
        struct Tracked(u32, Arc<Counter>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.1.record();
            }
        }

        let mut v: Vec<Tracked> = Vec::new();
        for i in 0..5u32 {
            v.try_push(Tracked(i, counter.clone())).unwrap();
        }
        // Collect the whole drained range; the temporary Drain drops at end of stmt
        // and compacts the vec back to [0, 3, 4].
        let collected: std::vec::Vec<Tracked> = v.try_drain(1..3).unwrap().collect();
        assert_eq!(collected[0].0, 1);
        assert_eq!(collected[1].0, 2);
        assert_eq!(v.len(), 3);
        assert_eq!(v.as_slice()[0].0, 0);
        assert_eq!(v.as_slice()[1].0, 3);
        assert_eq!(v.as_slice()[2].0, 4);
        // Nothing dropped yet: 1,2 alive in `collected`, 0,3,4 alive in v.
        assert_eq!(counter.get(), 0);
        drop(collected);
        // 1, 2 drop now.
        assert_eq!(counter.get(), 2);
        drop(v);
        // 0, 3, 4 drop as well -> total 5. Every element dropped exactly once.
        assert_eq!(counter.get(), 5);
    }

    /// Regression: `Drain::drop` used to skip the step-1 destruction of the
    /// unconsumed remainder whenever `size_of::<T>() == 0`, so a partially
    /// consumed drain over a ZST with a `Drop` impl leaked destructor calls.
    /// A ZST still owns a logical value, and every unconsumed element must run
    /// its destructor exactly once.
    #[test]
    fn drain_partial_consume_of_zst_runs_destructors() {
        use std::cell::Cell;
        use std::thread_local;

        thread_local! {
            static DROPS: Cell<usize> = const { Cell::new(0) };
        }

        struct ZstWithDrop;
        impl Drop for ZstWithDrop {
            fn drop(&mut self) {
                DROPS.set(DROPS.get() + 1);
            }
        }

        let mut v: Vec<ZstWithDrop> = Vec::new();
        for _ in 0..5 {
            v.try_push(ZstWithDrop).unwrap();
        }

        let mut d = v.try_drain(1..3).unwrap();
        // Consume exactly one of the two drained elements.
        let taken = d.next().expect("drained element");
        drop(taken);
        // Dropping the drain must destroy the one unconsumed element.
        drop(d);

        // Both elements of the drained range have now been destroyed.
        assert_eq!(DROPS.get(), 2);
        // The three survivors are untouched.
        assert_eq!(v.len(), 3);
        drop(v);
        assert_eq!(DROPS.get(), 5);
    }

    // Regression test for the stored-head-pointer OOB bug (see agents/BUGBOT.md):
    // once a drain is fully consumed, an advancing head pointer might sit at
    // `base + end`. This is OOB relative to the allocation and a hard
    // overflow on small pointer-width targets. The redesigned `Drain` tracks
    // position by integer offset instead.
    #[test]
    fn drain_exhausted_tail_query_is_safe() {
        let mut v = Vec::new();
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let mut d = v.try_drain(3..5).unwrap();
        // Consume everything; the "head" would now be at base + 5 (past the end).
        assert_eq!(d.next(), Some(3));
        assert_eq!(d.next(), Some(4));
        assert_eq!(d.next(), None);
        // Querying the exhausted drainer must not form an OOB pointer.
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        assert_eq!(d.as_slice(), &[] as &[i32]);
        drop(d);
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    /// Regression: a partially-consumed drain's `Drop` must destroy the hole at
    /// `original_start + consumed`, not `original_start + (original_count - remaining)`.
    ///
    /// The naive formula `start + (count - remaining)` derives the offset from
    /// `remaining` alone. It ignores the number of elements being taken from the back,
    /// thus double items that are taken out on there. It will also leak items on the
    /// left.
    ///
    /// This test exercises BOTH directions simultaneously: `next()` × 2 (front) and
    /// `next_back()` × 1 (back), so any formula that mishandles either side fails.
    #[test]
    fn drain_interleaved_drop_destroys_correct_hole() {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Single-threaded local sink that records drop.
        let sink: Rc<RefCell<std::vec::Vec<u32>>> = Rc::new(RefCell::new(std::vec::Vec::new()));

        struct Rec(u32, Option<Rc<RefCell<std::vec::Vec<u32>>>>);

        impl Rec {
            /// Detach the sink reference so this element's `Drop` will NOT record
            /// itself. Returns the payload for inspection.
            fn disarm(mut self) -> u32 {
                self.1.take(); // drop our Rc clone; destructor now sees None
                self.0
            }
        }

        impl Drop for Rec {
            fn drop(&mut self) {
                if let Some(s) = self.1.take() {
                    RefCell::<std::vec::Vec<u32>>::borrow_mut(&s).push(self.0);
                }
            }
        }

        let mut v: Vec<Rec> = Vec::new();
        for i in 0..6u32 {
            v.try_push(Rec(i, Some(Rc::clone(&sink)))).unwrap();
        }

        let mut d = v.try_drain(1..5).unwrap();
        // Disarm each yielded element: pull its payload out and detach its sink
        // clone, so only the drainer's own step-1 destruction reaches the sink.
        assert_eq!(d.next().expect("first drain element").disarm(), 1);
        assert_eq!(d.next().expect("second drain element").disarm(), 2);
        assert_eq!(d.next_back().expect("back drain element").disarm(), 4);
        drop(d);

        // The drainer must have destroyed exactly the single unconsumed hole
        // element, payload 3.
        let recorded = RefCell::<std::vec::Vec<u32>>::borrow(&sink);
        assert_eq!(*recorded, std::vec![3u32]);
        // Compaction removed the whole range: only payloads 0 and 5 survive.
        assert_eq!(v.len(), 2);
    }

    // Regression test for the hole-bookkeeping fix: `try_drain` caps the vector's
    // length to `start` at construction, so the drained range is a hole the vec no
    // longer counts. If a partially-consumed Drain is forgotten mid-iteration, the
    // vec must NOT try to drop the (partially uninitialized) hole — it only owns
    // the prefix. The unconsumed hole and the suffix leak, matching std's
    // documented behaviour for `mem::forget` on a `Drain`. Crucially there is no
    // double-drop and no UB from dropping uninitialized memory.
    //
    // We pull one element from each end (`next` and `next_back`) so both directions
    // of the drain are exercised before the forget. The ledger records each element
    // id so we can name precisely which ones dropped and which leaked.
    //
    // Ignored under Miri: `mem::forget` deliberately leaks four elements, which
    // Miri reports as unreachable-but-still-referenced memory at exit.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn drain_forget_mid_iteration_leaks_hole_not_prefix() {
        let ledger = Arc::new(Ledger::new());
        let budget = CloneBudget::new(u32::MAX);

        let mut v: Vec<FlakyCloneTrackedItem> = Vec::new();
        for i in 0..6u32 {
            v.try_push(FlakyCloneTrackedItem::with_id(&ledger, i, budget.share()))
                .unwrap();
        }

        // Drain [1..4): elements 1, 2, 3 live in the hole. Pull one from each end:
        // element 1 via `next`, element 3 via `next_back`, leaving element 2 as the
        // sole unconsumed hole member.
        let mut d = v.try_drain(1..4).unwrap();
        let front = d.next().expect("front drain element");
        assert_eq!(front.id(), 1);
        let back = d.next_back().expect("back drain element");
        assert_eq!(back.id(), 3);

        core::mem::forget(d);
        // Only the prefix (id 0) is owned by the vec now.
        assert!(ledger.drop_counts().is_empty());
        assert_eq!(v.len(), 1);
        drop(v);
        assert_eq!(ledger.live_ids(), [1, 2, 3, 4, 5]);
        assert_eq!(
            ledger.drop_count(0),
            1,
            "only the prefix may be dropped by the vec"
        );

        drop(front); // id 1
        drop(back); // id 3q
        // Final tally: ids 0, 1, 3 dropped exactly once. Ids 2, 4, 5 leaked (the
        // accepted cost of mem::forget). No double-free anywhere.
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.leaked_ids(), [2, 4, 5]);
        assert_eq!(ledger.drop_count(0), 1);
        assert_eq!(ledger.drop_count(1), 1);
        assert_eq!(ledger.drop_count(3), 1);
    }

    // Regression test for the drain compaction guard: dropping the center
    // slice withstands a panic, but compaction is not panic-safe by default
    // unless a guard is in place.
    #[test]
    fn drain_panic_in_destructor_still_compacts() {
        let armer = Arc::new(PanicArmer::new());
        let counter = Arc::new(Counter::new());

        #[allow(dead_code)]
        struct Panicky(u32, Arc<PanicArmer>, Arc<Counter>);
        impl Drop for Panicky {
            fn drop(&mut self) {
                self.2.record();
                if self.1.is_armed() && self.0 == 2 {
                    panic!("boom");
                }
            }
        }

        let mut v: Vec<Panicky> = Vec::new();
        for i in 0..6u32 {
            v.try_push(Panicky(i, armer.clone(), counter.clone()))
                .unwrap();
        }

        // Drain [1..4): elements 1, 2, 3 live in the hole; consume only element 1
        // so 2 and 3 remain to be destroyed in step 1. Arm the panic before the
        // drainer drops.
        let mut d = v.try_drain(1..4).unwrap();
        let taken = d.next().expect("first drain element");
        assert_eq!(taken.0, 1);
        armer.arm();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(d)));
        assert!(panicked.is_err(), "expected the destructor to panic");

        // The unwind ran the compaction guard: the suffix (4, 5) was shifted left
        // over the drained gap and the length restored past the prefix. So the vec
        // now holds exactly [0, 4, 5], not a stranded prefix of just [0].
        assert_eq!(v.len(), 3);
        assert_eq!(v.as_slice()[0].0, 0);
        assert_eq!(v.as_slice()[1].0, 4);
        assert_eq!(v.as_slice()[2].0, 5);

        // Both hole elements (2 and 3) were destroyed by the fat-slice teardown,
        // even though #2's destructor panicked.
        assert_eq!(
            counter.get(),
            2,
            "both hole elements must be dropped despite the panic"
        );

        // Disarm before dropping the rest so the panic isn't re-triggered.
        armer.disarm();
        drop(taken); // element 1 -> 3
        drop(v); // elements 0, 4, 5 -> 6 total
        assert_eq!(
            counter.get(),
            6,
            "every element dropped exactly once overall"
        );
    }

    // An empty range sitting exactly at the tail (`start == len`) would force
    // construction-time pointer arithmetic to `base.add(len)` and cause OOB
    // if pointer were used as state.
    #[test]
    fn drain_empty_range_at_tail_constructs_cleanly() {
        let mut v = Vec::new();
        for i in 0..4i32 {
            v.try_push(i).unwrap();
        }
        let d = v.try_drain(4..4).unwrap();
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        assert_eq!(d.as_slice(), &[] as &[i32]);
        drop(d);
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    }
}
