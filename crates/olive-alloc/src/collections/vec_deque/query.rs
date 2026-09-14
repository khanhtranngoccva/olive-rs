use super::VecDeque;
use super::wrapped_index::WrappedIndex;
use core::mem::size_of;
use core::ops::Range;
use core::ops::RangeBounds;
use core::ptr;
use olive_core::alloc::Allocator;
use olive_core::slice::TrySliceRangeError;

// Internal block
impl<T, A: Allocator> VecDeque<T, A> {
    /// Returns the index in the underlying buffer for a given logical element
    /// index + addend.
    ///
    /// # Safety
    /// - Both indexes must be either less than the capacity,
    ///   or must be equal to 0 if the capacity is 0.
    #[inline]
    pub(super) unsafe fn wrap_add(&self, idx: WrappedIndex, addend: usize) -> WrappedIndex {
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
    ///
    /// # Safety
    /// - The index must be less than the buffer capacity, or 0 if the capacity
    ///   is 0.
    #[inline]
    pub(super) unsafe fn to_wrapped_index(&self, idx: usize) -> WrappedIndex {
        unsafe { self.wrap_add(self.head, idx) }
    }

    /// Returns the index in the underlying buffer for a given logical element
    /// index - subtrahend.
    ///
    /// # Safety
    /// - Both indexes must be either less than the capacity,
    ///   or must be equal to 0 if the capacity is 0.
    #[inline]
    pub(super) unsafe fn wrap_sub(&self, idx: WrappedIndex, subtrahend: usize) -> WrappedIndex {
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

    /// Returns a slice pointer into the buffer.
    /// `range` must lie inside `0..self.capacity()`.
    ///
    /// # Safety
    /// - The range must be within the bounds of the physical buffer, and every
    ///   element in it must be initialized.
    #[inline]
    pub(super) unsafe fn buffer_range(&self, range: Range<usize>) -> *mut [T] {
        // SAFETY: caller guarantees the range lies inside the buffer and that
        // its elements are initialized; `end >= start` by construction of a
        // valid `Range`, so the subtraction cannot underflow.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "`end >= start` for any valid `Range`, so no underflow"
        )]
        unsafe {
            ptr::slice_from_raw_parts_mut(self.buf.ptr().add(range.start), range.end - range.start)
        }
    }
}

impl<T, A: Allocator> VecDeque<T, A> {
    /// Given a range into the logical buffer of the deque, this function
    /// returns two ranges into the physical buffer that correspond to the
    /// given range. The `len` parameter should usually just be `self.len`;
    /// the reason it's passed explicitly is that if the deque is wrapped in
    /// a `Drain`, then `self.len` is not actually the length of the deque.
    ///
    /// # Errors
    ///
    /// Returns [`TrySliceRangeError`] if the range cannot be resolved against
    /// `len` — an excluded/inclusive edge overflows, the start exceeds the
    /// end, or the end exceeds `len`.
    pub(crate) fn try_slice_ranges<R>(
        &self,
        range: R,
        len: usize,
    ) -> Result<(Range<usize>, Range<usize>), TrySliceRangeError>
    where
        R: RangeBounds<usize>,
    {
        let Range { start, end } = olive_core::slice::try_range(range, ..len)?;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "try_range asserted start <= end"
        )]
        let len = end - start;

        if len == 0 {
            Ok((0..0, 0..0))
        } else {
            // The assertion above guarantees that `start <= end <= len`.
            // SAFETY: Because `len != 0`, we know that `start < end`, so `start < len`
            // and the indexing is valid.
            let wrapped_start = unsafe { self.to_wrapped_index(start) };

            // This subtraction can never overflow because `wrapped_start` is
            // at most `self.capacity()` (and if `self.capacity() != 0`, then
            // `wrapped_start` is strictly less than it).
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted wrapped_start <= capacity()"
            )]
            let head_len = self.capacity() - wrapped_start.as_index();

            if head_len >= len {
                // We know that `len + wrapped_start <= self.capacity() <=
                // usize::MAX`, so this addition can't overflow.
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "asserted wrapped_start + len <= capacity()"
                )]
                Ok((
                    wrapped_start.as_index()..wrapped_start.as_index() + len,
                    0..0,
                ))
            } else {
                // Can't overflow because of the if condition.
                #[allow(clippy::arithmetic_side_effects, reason = "asserted len > head_len")]
                let tail_len = len - head_len;
                Ok((wrapped_start.as_index()..self.capacity(), 0..tail_len))
            }
        }
    }

    /// Returns the number of elements the deque can hold without
    /// reallocating.
    pub fn capacity(&self) -> usize {
        if size_of::<T>() == 0 {
            usize::MAX
        } else {
            self.buf.capacity()
        }
    }

    /// Provides a reference to the element at the given index, or `None` if
    /// the index is out of bounds.
    ///
    /// Element at index 0 is the front of the queue.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: index < self.len <= capacity
            let idx = unsafe { self.to_wrapped_index(index) };
            // SAFETY: `index < self.len` implies the wrapped slot holds an
            // initialized element.
            Some(unsafe { &*self.buf.ptr().add(idx.as_index()) })
        } else {
            None
        }
    }

    /// Provides a mutable reference to the element at the given index, or
    /// `None` if the index is out of bounds.
    ///
    /// Element at index 0 is the front of the queue.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            // SAFETY: index < self.len <= capacity
            let idx = unsafe { self.to_wrapped_index(index) };
            // SAFETY: `index < self.len` implies the wrapped slot holds an
            // initialized element.
            Some(unsafe { &mut *self.buf.ptr().add(idx.as_index()) })
        } else {
            None
        }
    }

    /// Returns the number of elements in the deque.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the deque is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Provides a reference to the front element, or `None` if the deque is
    /// empty.
    pub fn front(&self) -> Option<&T> {
        self.get(0)
    }

    /// Provides a mutable reference to the front element, or `None` if the
    /// deque is empty.
    pub fn front_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    /// Provides a reference to the back element, or `None` if the deque is
    /// empty.
    pub fn back(&self) -> Option<&T> {
        self.get(self.len.wrapping_sub(1))
    }

    /// Provides a mutable reference to the back element, or `None` if the
    /// deque is empty.
    pub fn back_mut(&mut self) -> Option<&mut T> {
        self.get_mut(self.len.wrapping_sub(1))
    }

    /// Returns a pair of slices which contain, in order, the contents of the
    /// deque.
    ///
    /// If the elements do not wrap around the buffer, all of them will be in
    /// the first slice and the second slice will be empty.
    #[inline]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        // A full-range query against our own length can never fail to resolve.
        let (a_range, b_range) = self
            .try_slice_ranges(.., self.len)
            .expect("full range is resolvable");
        // SAFETY: `try_slice_ranges` always returns valid ranges into the
        // physical buffer over initialized elements.
        unsafe { (&*self.buffer_range(a_range), &*self.buffer_range(b_range)) }
    }

    /// Returns a pair of mutable slices which contain, in order, the contents
    /// of the deque.
    ///
    /// If the elements do not wrap around the buffer, all of them will be in
    /// the first slice and the second slice will be empty.
    #[inline]
    pub fn as_mut_slices(&mut self) -> (&mut [T], &mut [T]) {
        // A full-range query against our own length can never fail to resolve.
        let (a_range, b_range) = self
            .try_slice_ranges(.., self.len)
            .expect("full range is resolvable");
        // SAFETY: `try_slice_ranges` always returns valid ranges into the
        // physical buffer over initialized elements.
        unsafe {
            (
                &mut *self.buffer_range(a_range),
                &mut *self.buffer_range(b_range),
            )
        }
    }
}
