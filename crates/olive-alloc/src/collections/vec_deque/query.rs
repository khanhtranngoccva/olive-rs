use super::VecDeque;
use super::wrapped_index::WrappedIndex;
use core::mem::size_of;
use olive_core::alloc::Allocator;

// Internal block
impl<T, A: Allocator> VecDeque<T, A> {
    /// Returns the index in the underlying buffer for a given logical element
    /// index + addend.
    ///
    /// # Safety
    /// - Both indexes must be either less than the capacity,
    ///   or must be equal to 0 if the capacity is 0.
    #[inline]
    unsafe fn wrap_add(&self, idx: WrappedIndex, addend: usize) -> WrappedIndex {
        let idx = idx.as_index();
        let cap = self.capacity();
        debug_assert!(idx < cap || idx == 0);
        debug_assert!(addend < cap || addend == 0);

        if cap == 0 {
            return WrappedIndex::zero();
        }

        // Scenario: head = 14, addend = 14, cap = 15, usize::MAX = 15,
        // need carryover.
        // Wrapping add (modulo 16) loses information (happens in a ZST VecDeque)
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted addend < cap, everything-is-0 branch had a guard"
        )]
        let threshold = cap - addend;
        if idx >= threshold {
            // idx >= cap - addend => idx + addend >= cap, cap has overflowed.
            // cap <= idx + addend < 2 * cap => (idx + addend) % cap == idx + addend - cap
            // == idx - (cap - addend)
            #[allow(clippy::arithmetic_side_effects, reason = "asserted idx >= threshold")]
            return WrappedIndex::from_arbitrary_number(idx - threshold);
        } else {
            // idx < cap - addend => idx + addend < cap.
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted overflow being not possible: idx + addend < cap"
            )]
            return WrappedIndex::from_arbitrary_number(idx + addend);
        }
    }

    /// Converts a logical index of the VecDeque to a wrapped index used to access
    /// the buffer.
    #[inline]
    fn to_wrapped_index(&self, idx: usize) -> WrappedIndex {
        unsafe { self.wrap_add(self.head, idx) }
    }

    /// Returns the index in the underlying buffer for a given logical element
    /// index - subtrahend.
    ///
    /// # Safety
    /// - Both indexes must be either less than the capacity,
    ///   or must be equal to 0 if the capacity is 0.
    #[inline]
    unsafe fn wrap_sub(&self, idx: WrappedIndex, subtrahend: usize) -> WrappedIndex {
        let idx = idx.as_index();
        let cap = self.capacity();
        debug_assert!(idx < cap || idx == 0);
        debug_assert!(subtrahend < cap || subtrahend == 0);

        if cap == 0 {
            return WrappedIndex::zero();
        }

        if idx >= subtrahend {
            #[allow(clippy::arithmetic_side_effects, reason = "asserted subtrahend <= idx")]
            return WrappedIndex::from_arbitrary_number(idx - subtrahend);
        } else {
            // The modular result is idx + cap - subtrahend = cap - (subtrahend - idx).
            #[allow(clippy::arithmetic_side_effects, reason = "asserted idx < subtrahend")]
            let diff = subtrahend - idx;
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted diff <= subtrahend < cap"
            )]
            return WrappedIndex::from_arbitrary_number(cap - diff);
        }
    }
}

impl<T, A: Allocator> VecDeque<T, A> {
    /// Returns the number of elements the deque can hold without
    /// reallocating.
    pub fn capacity(&self) -> usize {
        if size_of::<T>() == 0 {
            usize::MAX
        } else {
            self.buf.capacity()
        }
    }
}
