use super::BinaryHeap;
use super::poison::PoisonOnUnwind;
use core::{mem::ManuallyDrop, ptr};
use olive_core::alloc::Allocator;

/// Hole represents a hole in a slice i.e., an index without valid value
/// (because it was moved from or duplicated).
///
/// In drop, `Hole` will restore the slice by filling the hole
/// position with the value that was originally removed.
struct Hole<'a, T: 'a> {
    data: &'a mut [T],
    elt: ManuallyDrop<T>,
    pos: usize,
}

impl<'a, T> Hole<'a, T> {
    /// Creates a new `Hole` at index `pos`.
    ///
    /// Unsafe because `pos` must be within the data slice, and the
    /// Hole must be dropped.
    #[inline]
    unsafe fn new(data: &'a mut [T], pos: usize) -> Self {
        debug_assert!(pos < data.len());
        // SAFE: pos should be inside the slice
        let elt = unsafe { ptr::read(data.get_unchecked(pos)) };
        Hole {
            data,
            elt: ManuallyDrop::new(elt),
            pos,
        }
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    /// Move hole to new location
    ///
    /// Unsafe because index must be within the data slice and not equal to pos.
    #[inline]
    unsafe fn move_to(&mut self, index: usize) {
        debug_assert!(index != self.pos);
        debug_assert!(index < self.data.len());
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let index_ptr: *const _ = ptr.add(index);
            let hole_ptr = ptr.add(self.pos);
            ptr::copy_nonoverlapping(index_ptr, hole_ptr, 1);
        }
        self.pos = index;
    }
}

impl<T> Drop for Hole<'_, T> {
    #[inline]
    fn drop(&mut self) {
        // fill the hole again
        unsafe {
            let pos = self.pos;
            ptr::copy_nonoverlapping(&*self.elt, self.data.get_unchecked_mut(pos), 1);
        }
    }
}

impl<T: Ord, A: Allocator> BinaryHeap<T, A> {
    /// Given an element at a specific position (usually at the end of the binary heap), compute the
    /// location it is sifted up to.
    ///
    /// This function does not modify any data.
    ///
    /// # Panics
    /// - Panics if `pos >= self.len()`.
    pub(super) fn compute_sift_up(&self, start: usize, mut pos: usize) -> usize {
        let element = &self.data[pos];
        while pos > start {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted pos > start >= 0, so the decrement cannot underflow"
            )]
            let parent = (pos - 1) / 2;
            if element <= &self.data[parent] {
                break;
            }
            pos = parent;
        }
        pos
    }

    /// Given an element at a specific position, remove it, and put the element in its sifted up
    /// location.
    ///
    /// This function will not panic.
    ///
    /// # Safety
    /// - The `new_pos` argument must be computed by [`compute_sift_up`].
    /// - Caller must ensure `new_pos <= pos < self.len()`.
    /// - The heap's contents must not change after the call to [`compute_sift_up`].
    ///
    /// [`compute_sift_up`]: Self::compute_sift_up
    pub(super) unsafe fn finalize_sift_up(&mut self, pos: usize, new_pos: usize) {
        // Take out the value at `pos` and create a hole.
        // SAFETY: The caller guarantees that pos < self.len()
        let mut hole = unsafe { Hole::new(&mut self.data, pos) };

        while hole.pos() > new_pos {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted hole.pos() > new_pos >= 0, so the decrement cannot underflow"
            )]
            let parent = (hole.pos() - 1) / 2;
            // SAFETY: hole.pos() != parent, as above
            unsafe {
                hole.move_to(parent);
            };
        }

        debug_assert!(hole.pos() == new_pos);
    }

    /// Given an element at a specific position, attempt a sift up of the element up the binary heap.
    ///
    /// # Panics
    ///
    /// - Panics if the [`Ord`] implementation of the element panics. However, the operation is
    ///   guaranteed to be atomic, allowing operations to not break the heap's invariants.
    /// - Panics if `pos >= self.len()`.
    pub(super) unsafe fn sift_up(&mut self, start: usize, pos: usize) -> usize {
        let new_pos = self.compute_sift_up(start, pos);
        // SAFETY: compute_sift_up computed the value, which satisfies the required bounds
        unsafe { self.finalize_sift_up(pos, new_pos) };
        new_pos
    }

    /// Given an element at a specific position, compute the complete
    /// sift-down walk it performs over `data[pos..end]`. The resulting plan
    /// has at least 1 element, which corresponds to a no-op.
    ///
    /// This function does not modify any data.
    ///
    /// # Panics
    ///
    /// - Panics if the [`Ord`] implementation panics; no data has been modified at that point.
    /// - The caller must guarantee that `pos < end <= self.len()`.
    pub(super) fn compute_sift_down(&self, stack: &mut SiftStack, pos: usize, end: usize) -> Plan {
        // Panic assertions and element retrieval
        assert!(pos < end);
        assert!(end <= self.len());

        // SAFETY: assertions above.
        let element = unsafe { self.data.get_unchecked(pos) };

        // We record the beginning position.
        stack.begin(pos);

        // Loop invariant: child == 2 * pos + 1 for the current pos.

        // If computing child == 2 * pos + 1 ever overflows, then child == usize::MAX > end - 1 > end - 2,
        // so no further code is fired, and the stack is final.
        // If 2 * pos + 2 overflows but 2 * pos + 1 does not, then child == usize::MAX and the stack is also final.
        let mut child = pos.saturating_mul(2).saturating_add(1);

        // If end <= 2, then we have 2 elements and a tree without any node having 2 children, so
        // the code below cannot fire. It is guaranteed by the fact child >= 1 > end.saturating_sub(2) == 0.
        while child <= end.saturating_sub(2) {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted child + 1 < end <= len, so both the sibling index and the increment are in range"
            )]
            {
                child += (unsafe {
                    self.data.get_unchecked(child) <= self.data.get_unchecked(child + 1)
                }) as usize;
            }
            // If we are already in order, stop.
            if element >= unsafe { self.data.get_unchecked(child) } {
                return stack.drain();
            }
            stack.push_offset(child);
            child = child.saturating_mul(2).saturating_add(1);
        }

        // A lone left child at the end: move onto it only if it wins.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted pos < end, so end >= 1"
        )]
        if child == end - 1 && element < unsafe { self.data.get_unchecked(child) } {
            stack.push_offset(child);
        }

        stack.drain()
    }

    /// Commit a plan produced by [`compute_sift_down`]: the element at
    /// `pos` is moved down the recorded path to its final position.
    ///
    /// The commit invokes no user code and cannot panic.
    ///
    /// # Safety
    ///
    /// - The `plan` argument must be computed by [`compute_sift_down`] for this heap.
    /// - Caller must ensure `pos < self.len()`.
    /// - The heap's contents must not change after the call to [`compute_sift_down`].
    ///
    /// [`compute_sift_down`]: Self::compute_sift_down
    pub(super) unsafe fn finalize_sift_down(&mut self, pos: usize, plan: Plan) {
        debug_assert_eq!(plan.offsets().first(), Some(&pos));
        // SAFETY: The caller guarantees that `pos` still holds the
        // element the plan was computed for, and that the plan
        // describes a real walk over the untouched data.
        unsafe { plan.apply(self.data.as_mut_slice()) };
    }

    /// Take an element at `pos` and move it down the heap,
    /// while its children are larger.
    ///
    /// Returns the new position of the element.
    ///
    /// # Panics
    ///
    /// - Panics if the [`Ord`] implementation of the element panics. However, the operation is
    ///   guaranteed to be atomic, allowing operations to not break the heap's invariants.
    /// - Panics if `pos >= end` or `end > self.len()`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `pos < end <= self.len()`.
    pub(super) unsafe fn sift_down_range(&mut self, pos: usize, end: usize) -> usize {
        let mut stack = SiftStack::new();
        let plan = self.compute_sift_down(&mut stack, pos, end);
        let final_position = *plan
            .offsets()
            .last()
            .expect("sifting down plan should have a final position");
        unsafe { self.finalize_sift_down(pos, plan) };
        final_position
    }

    /// Same as [`sift_down_range`], but sifts to the end of the heap.
    ///
    /// # Panics
    ///
    /// - Panics if the [`Ord`] implementation of the element panics. However, the operation is
    ///   guaranteed to be atomic, allowing operations to not break the heap's invariants.
    /// - Panics if `pos >= self.len()`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `pos < self.len()`.
    ///
    /// [`sift_down_range`]: Self::sift_down_range
    pub(super) unsafe fn sift_down(&mut self, pos: usize) -> usize {
        let len = self.len();
        // SAFETY: pos < len is guaranteed by the caller and
        //  obviously len = self.len() <= self.len().
        unsafe { self.sift_down_range(pos, len) }
    }

    /// Rebuild assuming data[0..start] is still a proper heap.
    pub(super) fn rebuild_tail(&mut self, start: usize) {
        assert!(start <= self.len());
        if start == self.len() {
            return;
        }

        #[allow(
            clippy::arithmetic_side_effects,
            reason = "guarded: start < self.len(), so the decrement cannot underflow"
        )]
        let tail_len = self.len() - start;

        #[inline(always)]
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "leading_zeros is at most usize::BITS, and it equals usize::BITS only for x == 0, which the caller's tail_len >= 1 excludes"
        )]
        fn log2_fast(x: usize) -> usize {
            (usize::BITS - x.leading_zeros() - 1) as usize
        }

        // From std: `rebuild` takes O(self.len()) operations
        // and about 2 * self.len() comparisons in the worst case
        // while repeating `sift_up` takes O(tail_len * log(start)) operations
        // and about 1 * tail_len * log_2(start) comparisons in the worst case,
        // assuming start >= tail_len. For larger heaps, the crossover point
        // no longer follows this reasoning and was determined empirically.
        let better_to_rebuild = if start < tail_len {
            true
        } else if self.len() <= 2048 {
            2usize.saturating_mul(self.len()) < tail_len.saturating_mul(log2_fast(start))
        } else {
            2usize.saturating_mul(self.len()) < tail_len.saturating_mul(11)
        };

        if better_to_rebuild {
            self.rebuild();
            return;
        }

        for i in start..self.len() {
            // SAFETY: The index `i` is always less than self.len().
            unsafe { self.sift_up(0, i) };
        }
    }

    /// Restores the heap invariant over the whole buffer by running
    /// `sift_down` bottom-up.
    ///
    /// This is the documented recovery operation: it repairs a buffer a
    /// panic left permuted and lifts the quarantine — on success it
    /// clears the poison flag, so it is callable on a poisoned heap.
    ///
    /// # Panics
    /// - Panics if the [`Ord`] implementation panics. Re-running `rebuild` is
    ///   both the repair and the recovery.
    pub fn rebuild(&mut self) {
        let mut n = self.len() / 2;
        let mut quarantine = PoisonOnUnwind::new(self);
        while n > 0 {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the loop condition proves n > 0, so the decrement cannot underflow"
            )]
            {
                n -= 1;
            }
            // SAFETY: n starts from self.len() / 2 and goes down to 0.
            //  The only case when !(n < self.len()) is if
            //  self.len() == 0, but it's ruled out by the loop condition.
            unsafe { quarantine.sift_down(n) };
        }
        quarantine.defuse();
        quarantine.poisoned = false;
    }
}

/// The most positions any single sift walk can visit, including the initial position.
pub(super) const PATH_CAPACITY: usize = usize::BITS as usize + 1;

/// A fixed-capacity recorder for one sift-down walk.
///
/// The decision recorded at each level is which child won the sibling
/// comparison. That choice depends on the compared values, so it cannot
/// be re-derived arithmetically from the endpoints the way the up-sift's
/// ancestor chain can; storing it is what lets the down-sift be computed
/// entirely before anything is committed, and the committed phase then
/// consists of plain, panic-free moves.
#[derive(Clone, Copy, Debug)]
pub(super) struct SiftStack {
    offsets: [usize; PATH_CAPACITY],
    depth: usize,
}

impl SiftStack {
    /// Create a fresh, empty recorder. The empty state is not valid for the commit phase,
    /// and begin() must be invoked.
    #[inline]
    pub(super) const fn new() -> Self {
        Self {
            offsets: [0; PATH_CAPACITY],
            depth: 0,
        }
    }

    /// Start a new recording with the beginning offset.
    #[inline]
    pub(super) const fn begin(&mut self, offset: usize) {
        self.depth = 0;
        self.push_offset(offset);
    }

    /// Record the next position visited by the walk.
    #[inline]
    pub(super) const fn push_offset(&mut self, offset: usize) {
        self.offsets[self.depth] = offset;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "PATH_CAPACITY bounds any real walk and begin() resets the fill, so depth < PATH_CAPACITY holds here"
        )]
        {
            self.depth += 1;
        }
    }

    /// Finish the recording and take the resulting [`Plan`], resetting
    /// the fill for the next walk.
    #[inline]
    pub(super) const fn drain(&mut self) -> Plan {
        let plan = Plan {
            offsets: self.offsets,
            depth: self.depth,
        };
        *self = Self::new();
        plan
    }
}

/// The recorded path of one completed sift-down walk: the positions the
/// hole visited, outermost first.
///
/// The plan is pure data, captured before any element moves, so applying
/// it cannot invoke user code and cannot panic.
//
// FIXME: Given a starting position and a bit vector of exactly that usize::BITS
// (basically an usize) one can derive the positioning end-to-end (left/right).
#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    offsets: [usize; PATH_CAPACITY],
    depth: usize,
}

impl Plan {
    /// The recorded positions, in visit order.
    #[inline]
    pub(super) fn offsets(&self) -> &[usize] {
        &self.offsets[..self.depth]
    }

    /// Commit the walk: swap the element at `offsets[0]` down the
    /// recorded path until it rests at `offsets[depth - 1]`.
    ///
    /// # Safety
    ///
    /// The plan must be one drained from a recording of a real sift-down
    /// walk over a slice of the same length, and since then the slice
    /// must not have been modified at all.
    ///
    /// The plan must not be empty.
    #[inline]
    pub(super) unsafe fn apply<T>(self, data: &mut [T]) {
        debug_assert!(!self.offsets.is_empty());
        let mut hole = unsafe { Hole::new(data, self.offsets[0]) };
        for step in 1..self.depth {
            unsafe { hole.move_to(self.offsets[step]) };
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::collections::binary_heap::BinaryHeap;
    use olive_core::try_traits::try_from_iterator::TryFromIterator;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn heapify_fixes_an_unordered_vector() {
        // TryFromIterator is `rebuild` under the hood.
        let mut heap: BinaryHeap<i32> =
            BinaryHeap::try_from_iter([1, 5, 3, 9, 2]).expect("allocation ok");
        let mut drained = Vec::new();
        while let Some(item) = heap.pop() {
            drained.push(item);
        }
        assert_eq!(drained, vec![9, 5, 3, 2, 1]);
    }
}
