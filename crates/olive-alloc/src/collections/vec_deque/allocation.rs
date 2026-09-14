//! Allocation-backed mutation for [`VecDeque`].
//!
//! These methods may grow the underlying buffer, so they return a
//! [`Result`] carrying a [`TryReserveError`] instead of panicking on
//! out-of-memory. Each public push is paired with a `*_give_back` variant that
//! hands the uninserted value back to the caller on failure, mirroring the
//! convention established by `Vec::try_push_give_back`.

use super::VecDeque;
use super::wrapped_index::WrappedIndex;
use core::mem::size_of;
use core::ptr;
use olive_core::alloc::Allocator;
use olive_core::alloc_errors::{TryReserveError, TryReserveErrorKind};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned by fallible deque insert operations.
///
/// Used by [`VecDeque::try_insert`] and its variants: the operation can fail
/// either because growing the buffer failed or because the index was out of
/// bounds. In the give-back variant the value travels alongside this error as
/// a tuple: `Result<(), (T, TryVecDequeInsertError)>`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TryVecDequeInsertError {
    /// A capacity reservation failed.
    Reserve(TryReserveError),
    /// The provided index exceeded the deque's length.
    OutOfBounds,
}

impl core::fmt::Debug for TryVecDequeInsertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Reserve(e) => f
                .debug_tuple("TryVecDequeInsertError::Reserve")
                .field(e)
                .finish(),
            Self::OutOfBounds => f
                .debug_tuple("TryVecDequeInsertError::OutOfBounds")
                .finish(),
        }
    }
}

impl core::fmt::Display for TryVecDequeInsertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "deque insert failed: {e}"),
            Self::OutOfBounds => write!(f, "deque insert failed: index out of bounds"),
        }
    }
}

impl core::error::Error for TryVecDequeInsertError {}

// ---------------------------------------------------------------------------
// Reservation
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Ensures the deque has room for at least `additional` more elements.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the new capacity overflows or the
    /// allocation fails.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let new_cap = self
            .len
            .checked_add(additional)
            .ok_or(TryReserveErrorKind::CapacityOverflow)?;
        let old_cap = self.capacity();

        if new_cap > old_cap {
            self.buf.try_reserve(self.len, additional)?;
            // SAFETY: `old_cap` was the capacity before growth; `self.len <=
            // old_cap` held before the reserve.
            unsafe {
                self.handle_capacity_increase(old_cap);
            }
        }
        Ok(())
    }

    /// Ensures the deque has room for exactly `len + additional` elements.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the new capacity overflows or the
    /// allocation fails.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let new_cap = self
            .len
            .checked_add(additional)
            .ok_or(TryReserveErrorKind::CapacityOverflow)?;
        let old_cap = self.capacity();

        if new_cap > old_cap {
            self.buf.try_reserve_exact(self.len, additional)?;
            // SAFETY: same as `try_reserve`.
            unsafe {
                self.handle_capacity_increase(old_cap);
            }
        }
        Ok(())
    }

    /// Ensures the deque has room for at least `total` elements *in total*
    /// (an absolute target, not an increment).
    ///
    /// `total < len` results in a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the new capacity overflows or the
    /// allocation fails.
    pub fn try_reserve_total(&mut self, total: usize) -> Result<(), TryReserveError> {
        let additional = total.saturating_sub(self.len);
        self.try_reserve(additional)
    }
}

// ---------------------------------------------------------------------------
// Fallible pushes
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Appends an element to the back of the deque, growing the buffer if
    /// needed.
    ///
    /// On failure the deque is left unchanged and `value` is dropped. Use
    /// [`Self::try_push_back_give_back`] to recover the value instead.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails.
    pub fn try_push_back(&mut self, value: T) -> Result<(), TryReserveError> {
        self.try_push_back_give_back(value).map_err(|(_, e)| e)
    }

    /// Like [`Self::try_push_back`], but on failure returns the unappended
    /// `value` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryReserveError)` if growing the buffer fails.
    pub fn try_push_back_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)> {
        if self.len == self.capacity() {
            if let Err(e) = self.try_reserve(1) {
                return Err((value, e));
            }
        }
        // SAFETY: spare capacity exists (we grew if needed).
        unsafe { self.push_back_within_cap(value) };
        Ok(())
    }

    /// Prepends an element to the front of the deque, growing the buffer if
    /// needed.
    ///
    /// On failure the deque is left unchanged and `value` is dropped. Use
    /// [`Self::try_push_front_give_back`] to recover the value instead.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails.
    pub fn try_push_front(&mut self, value: T) -> Result<(), TryReserveError> {
        self.try_push_front_give_back(value).map_err(|(_, e)| e)
    }

    /// Like [`Self::try_push_front`], but on failure returns the unprepended
    /// `value` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryReserveError)` if growing the buffer fails.
    pub fn try_push_front_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)> {
        if self.len == self.capacity() {
            if let Err(e) = self.try_reserve(1) {
                return Err((value, e));
            }
        }
        // SAFETY: spare capacity exists (we grew if needed).
        unsafe { self.push_front_within_cap(value) };
        Ok(())
    }

    /// Appends an element to the back of the deque and returns a mutable
    /// reference to it.
    ///
    /// This is convenient when the element needs further initialization after
    /// insertion (e.g., setting fields on a struct).
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails.
    pub fn try_push_back_mut(&mut self, value: T) -> Result<&mut T, TryReserveError> {
        self.try_push_back_mut_give_back(value)
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_push_back_mut`], but on failure returns the unappended
    /// `value` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryReserveError)` if growing the buffer fails.
    pub fn try_push_back_mut_give_back(
        &mut self,
        value: T,
    ) -> Result<&mut T, (T, TryReserveError)> {
        if self.len == self.capacity() {
            if let Err(e) = self.try_reserve(1) {
                return Err((value, e));
            }
        }
        // SAFETY: spare capacity exists (we grew if needed).
        let ptr = unsafe { self.push_back_within_cap(value) };
        // SAFETY: `ptr` points to the freshly-written, in-bounds slot.
        Ok(unsafe { &mut *ptr })
    }

    /// Prepends an element to the front of the deque and returns a mutable
    /// reference to it.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails.
    pub fn try_push_front_mut(&mut self, value: T) -> Result<&mut T, TryReserveError> {
        self.try_push_front_mut_give_back(value)
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_push_front_mut`], but on failure returns the
    /// unprepended `value` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryReserveError)` if growing the buffer fails.
    pub fn try_push_front_mut_give_back(
        &mut self,
        value: T,
    ) -> Result<&mut T, (T, TryReserveError)> {
        if self.len == self.capacity() {
            if let Err(e) = self.try_reserve(1) {
                return Err((value, e));
            }
        }
        // SAFETY: spare capacity exists (we grew if needed).
        let ptr = unsafe { self.push_front_within_cap(value) };
        // SAFETY: `ptr` points to the freshly-written, in-bounds slot.
        Ok(unsafe { &mut *ptr })
    }
}

// ---------------------------------------------------------------------------
// Fallible insert
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Inserts an element at position `index`, shifting later elements toward
    /// the back.
    ///
    /// On failure the deque is left unchanged and `value` is dropped. Use
    /// [`Self::try_insert_give_back`] to recover the value instead.
    ///
    /// # Errors
    ///
    /// * [`TryVecDequeInsertError::OutOfBounds`] — `index > len`.
    /// * [`TryVecDequeInsertError::Reserve`] — growing the buffer failed.
    pub fn try_insert(&mut self, index: usize, value: T) -> Result<(), TryVecDequeInsertError> {
        self.try_insert_mut_give_back(index, value)
            .map(|_| ())
            .map_err(|(_returned, e)| e)
    }

    /// Like [`Self::try_insert`], but returns the value back on failure.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryVecDequeInsertError)` on failure.
    pub fn try_insert_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<(), (T, TryVecDequeInsertError)> {
        self.try_insert_mut_give_back(index, value).map(|_| ())
    }

    /// Inserts an element at position `index` and returns a mutable reference
    /// to it.
    ///
    /// # Errors
    ///
    /// * [`TryVecDequeInsertError::OutOfBounds`] — `index > len`.
    /// * [`TryVecDequeInsertError::Reserve`] — growing the buffer failed.
    pub fn try_insert_mut(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<&mut T, TryVecDequeInsertError> {
        self.try_insert_mut_give_back(index, value)
            .map_err(|(_returned, err)| err)
    }

    /// Like [`Self::try_insert_mut`], but returns the value back on failure.
    ///
    /// # Errors
    ///
    /// Returns `(T, TryVecDequeInsertError)` on failure.
    pub fn try_insert_mut_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<&mut T, (T, TryVecDequeInsertError)> {
        if index > self.len {
            return Err((value, TryVecDequeInsertError::OutOfBounds));
        }
        if self.len == self.capacity() {
            if let Err(e) = self.try_reserve(1) {
                return Err((value, TryVecDequeInsertError::Reserve(e)));
            }
        }
        // SAFETY: `index <= len` checked above; `len < capacity` after the
        // conditional reserve.
        let ptr = unsafe { self.insert_within_cap(index, value) };
        // SAFETY: `ptr` points to the freshly-written, in-bounds slot.
        Ok(unsafe { &mut *ptr })
    }
}

// ---------------------------------------------------------------------------
// Fallible bulk transfers
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Moves all elements from `other` to the back of `self`, in their
    /// original order. After the call, `other` is empty.
    ///
    /// This is equivalent to repeatedly calling [`Self::try_push_back`] for
    /// each element in `other`, but more efficient because it reserves space
    /// once instead of checking (and possibly growing) on every push.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails. On failure,
    /// `other` is left unchanged.
    pub fn try_append(&mut self, other: &mut Self) -> Result<(), TryReserveError> {
        let count = other.len;
        if count == 0 {
            return Ok(());
        }
        // Reserve enough room for all incoming elements. If this fails, we
        // haven't touched either deque yet.
        self.try_reserve(count)?;
        // Copy elements from `other` into `self`'s back slots, one by one.
        // We iterate over `other`'s logical order (front to back) and write
        // each into the next available back slot of `self`.
        //
        // SAFETY: we just reserved `count` slots, so there is room for all
        // elements. Each write lands in a previously-uninitialized slot.
        unsafe {
            for i in 0..count {
                // Read from `other` at logical index `i`.
                let src_idx = other.to_wrapped_index(i);
                let val_ptr = other.buf.ptr().add(src_idx.as_index());
                // Compute destination: the next back slot in `self`.
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "asserted self.len + i < self.len + count == cap"
                )]
                let dst_idx = self.to_wrapped_index(self.len + i);
                let dst_ptr = self.buf.ptr().add(dst_idx.as_index());
                // Move the element (no clone).
                ptr::copy_nonoverlapping(val_ptr, dst_ptr, 1);
            }
        }
        // Update lengths.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "reserved capacity guarantees no overflow"
        )]
        {
            self.len += count;
        }
        other.len = 0;
        // Reset `other`'s head to 0 since it's now empty.
        other.head = WrappedIndex::zero();
        Ok(())
    }

    /// Moves all elements from `other` to the front of `self`, preserving
    /// `other`'s internal order (its front element becomes adjacent to
    /// `self`'s old front). After the call, `other` is empty.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails. On failure,
    /// `other` is left unchanged.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "reserved capacity guarantees no overflow"
    )]
    pub fn try_prepend(&mut self, other: &mut Self) -> Result<(), TryReserveError> {
        let count = other.len;
        if count == 0 {
            return Ok(());
        }
        // Reserve enough room.
        self.try_reserve(count)?;

        #[allow(
            clippy::arithmetic_side_effects,
            reason = "`count` has just been reserved"
        )]
        if size_of::<T>() == 0 {
            // ZST: no physical memory to move; just adjust bookkeeping.
            self.len += count;
            other.len = 0;
            other.head = WrappedIndex::zero();
            return Ok(());
        }

        // Retreat `self.head` by `count` to make room at the front.
        // SAFETY: we just reserved `count` slots, so retreating `head` by
        // `count` stays within bounds.
        self.head = unsafe { self.wrap_sub(self.head, count) };

        // Now copy `other`'s elements into the newly-freed front region of
        // `self`. The first element of `other` (logical index 0) goes into
        // the new front of `self` (which is at the retreated `head`).
        //
        // SAFETY: all destination slots are within the freshly-reserved
        // capacity and were uninitialized before this operation.
        unsafe {
            for i in 0..count {
                let src_idx = other.to_wrapped_index(i);
                let val_ptr = other.buf.ptr().add(src_idx.as_index());
                let dst_idx = self.to_wrapped_index(i);
                let dst_ptr = self.buf.ptr().add(dst_idx.as_index());
                ptr::copy_nonoverlapping(val_ptr, dst_ptr, 1);
            }
        }

        self.len += count;
        other.len = 0;
        other.head = WrappedIndex::zero();

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::test_helpers::{BudgetedAlloc, FailAlloc};
    use core::mem::size_of;

    /// Concatenate the two halves of a deque into a fixed-size array for
    /// comparison. Returns `None` if the combined length exceeds `N`.
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

    /// Same as [`collect_into_array`] but for deques backed by an arbitrary
    /// allocator (used by OOM tests that need a non-global allocator).
    fn collect_any<const N: usize, A: Allocator>(dq: &VecDeque<i32, A>) -> Option<[i32; N]> {
        let (a, b) = dq.as_slices();
        let mut out: [i32; N] = [0; N];
        if a.len() + b.len() > N {
            return None;
        }
        out[..a.len()].copy_from_slice(a);
        out[a.len()..a.len() + b.len()].copy_from_slice(b);
        Some(out)
    }

    // --- try_reserve ----------------------------------------------------------

    #[test]
    fn reserve_grows_amortized() {
        let mut dq = VecDeque::<i32>::new();
        assert_eq!(dq.try_reserve(10), Ok(()));
        assert!(dq.capacity() >= 10);
        assert!(dq.is_empty());
    }

    #[test]
    fn reserve_is_a_no_op_when_already_enough() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        let cap_before = dq.capacity();
        assert_eq!(dq.try_reserve(4), Ok(()));
        // Amortized growth never shrinks; small requests keep the same buffer.
        assert_eq!(dq.capacity(), cap_before);
    }

    #[test]
    fn reserve_exact_requests_only_what_is_needed() {
        let mut dq = VecDeque::<i32>::new();
        assert_eq!(dq.try_reserve_exact(5), Ok(()));
        // Exact growth should land close to the request (allocator may round
        // up slightly, but must not overshoot dramatically).
        assert!(dq.capacity() >= 5 && dq.capacity() <= 8);
    }

    #[test]
    fn reserve_total_below_len_is_a_no_op() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        let cap_before = dq.capacity();
        assert_eq!(dq.try_reserve_total(3), Ok(()));
        assert_eq!(dq.capacity(), cap_before);
    }

    #[test]
    fn reserve_overflow_returns_capacity_overflow() {
        let mut dq: VecDeque<u8> = VecDeque::new();
        let err = match dq.try_reserve(usize::MAX) {
            Err(e) => e,
            Ok(_) => panic!("expected capacity overflow"),
        };
        assert!(err.is_capacity_overflow());
        // The deque survived the failed reservation.
        assert!(dq.is_empty());
    }

    #[test]
    fn reserve_exact_overflow_returns_capacity_overflow() {
        let mut dq: VecDeque<u8> = VecDeque::new();
        let err = match dq.try_reserve_exact(usize::MAX) {
            Err(e) => e,
            Ok(_) => panic!("expected capacity overflow"),
        };
        assert!(err.is_capacity_overflow());
        assert!(dq.is_empty());
    }

    #[test]
    fn reserve_total_above_len_grows() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        // Requesting a total of 10 should grow from cap 4 to at least 10.
        assert_eq!(dq.try_reserve_total(10), Ok(()));
        assert!(dq.capacity() >= 10);
        // Existing element is still intact.
        assert_eq!(dq.get(0), Some(&1));
    }

    #[test]
    fn reserve_preserves_elements_across_growth() {
        // Build a wrapped state: push back 3, push front 2 → head retreats,
        // elements span two physical segments.
        let mut dq = VecDeque::<i32>::try_with_capacity(5).expect("allocation ok");
        assert_eq!(dq.try_push_back(10), Ok(()));
        assert_eq!(dq.try_push_back(20), Ok(()));
        assert_eq!(dq.try_push_back(30), Ok(()));
        assert_eq!(dq.try_push_front(40), Ok(()));
        assert_eq!(dq.try_push_front(50), Ok(()));
        // Logical order: [50, 40, 10, 20, 30], len=5, cap=5 (full).
        // Now force a growth that triggers handle_capacity_increase.
        assert_eq!(dq.try_reserve(5), Ok(()));
        assert!(dq.capacity() > 5);
        // All five elements must survive in order.
        assert_eq!(collect_into_array::<5>(&dq), Some([50, 40, 10, 20, 30]));
    }

    #[test]
    fn reserve_oom_returns_alloc_error_kind() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let err = match dq.try_reserve(4) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        assert!(!err.is_capacity_overflow());
        assert!(dq.is_empty());
    }

    #[test]
    fn zst_reserve_never_allocates() {
        let mut dq: VecDeque<()> = VecDeque::new();
        // ZSTs report usize::MAX capacity, so any finite reservation is a
        // no-op that must succeed without touching the heap.
        assert_eq!(dq.try_reserve(1 << 40), Ok(()));
        assert_eq!(dq.try_reserve_exact(1 << 40), Ok(()));
        assert_eq!(dq.try_reserve_total(1 << 40), Ok(()));
        assert_eq!(dq.capacity(), usize::MAX);
    }

    // --- try_push_back / give_back ---------------------------------------------

    #[test]
    fn push_back_appends_and_grows_as_needed() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        // Buffer now full; next push forces a grow.
        assert_eq!(dq.try_push_back(3), Ok(()));
        assert!(dq.capacity() > 2);
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn push_back_into_fresh_deque_grows_from_zero() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        assert_eq!(dq.try_push_back(7), Ok(()));
        assert_eq!(dq.len(), 1);
        assert_eq!(dq.back(), Some(&7));
    }

    #[test]
    fn push_back_oom_leaves_deque_unchanged() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let err = match dq.try_push_back(1) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        assert!(dq.is_empty());
        assert_eq!(dq.capacity(), 0);
    }

    #[test]
    fn push_back_give_back_recovers_value_on_failure() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let (returned, err) = match dq.try_push_back_give_back(42) {
            Err(pair) => pair,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert_eq!(returned, 42);
        assert!(err.is_alloc());
        assert!(dq.is_empty());
    }

    #[test]
    fn push_back_give_back_succeeds_when_space_exists() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back_give_back(9), Ok(()));
        assert_eq!(dq.back(), Some(&9));
    }

    #[test]
    fn push_back_give_back_triggers_growth_then_succeeds() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        // Buffer full; give_back must grow internally and still succeed.
        assert_eq!(dq.try_push_back_give_back(3), Ok(()));
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    // --- try_push_front / give_back --------------------------------------------

    #[test]
    fn push_front_prepends_and_grows_as_needed() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_front(2), Ok(()));
        assert_eq!(dq.try_push_front(1), Ok(()));
        // Full; next push grows.
        assert_eq!(dq.try_push_front(0), Ok(()));
        assert!(dq.capacity() > 2);
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([0, 1, 2]));
    }

    #[test]
    fn push_front_oom_leaves_deque_unchanged() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let err = match dq.try_push_front(1) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        assert!(dq.is_empty());
    }

    #[test]
    fn push_front_give_back_recovers_value_on_failure() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let (returned, err) = match dq.try_push_front_give_back(42) {
            Err(pair) => pair,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert_eq!(returned, 42);
        assert!(err.is_alloc());
        assert!(dq.is_empty());
    }

    #[test]
    fn push_front_give_back_succeeds_when_space_exists() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_front_give_back(9), Ok(()));
        assert_eq!(dq.front(), Some(&9));
    }

    #[test]
    fn push_front_give_back_triggers_growth_then_succeeds() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_front(1), Ok(()));
        assert_eq!(dq.try_push_front(2), Ok(()));
        // Buffer full; give_back must grow internally and still succeed.
        assert_eq!(dq.try_push_front_give_back(0), Ok(()));
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([0, 2, 1]));
    }

    // --- try_push_back_mut / try_push_front_mut ------------------------------------

    #[test]
    fn push_back_mut_returns_reference_to_appended_element() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back_mut(5), Ok(&mut 5));
        assert_eq!(dq.try_push_back_mut(10), Ok(&mut 10));
        *dq.back_mut().unwrap() += 5;
        assert_eq!(dq.back(), Some(&15));
        assert_eq!(dq.len(), 2);
    }

    #[test]
    fn push_front_mut_returns_reference_to_prepended_element() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back_mut(5), Ok(&mut 5));
        assert_eq!(dq.try_push_front_mut(20), Ok(&mut 20));
        *dq.front_mut().unwrap() -= 7;
        assert_eq!(dq.front(), Some(&13));
        assert_eq!(dq.len(), 2);
    }

    #[test]
    fn push_back_mut_grows_when_full() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        // Full; next push must grow internally and still return a reference.
        assert_eq!(dq.try_push_back_mut(3), Ok(&mut 3));
        assert!(dq.capacity() > 2);
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn push_front_mut_grows_when_full() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        // Full; front push must grow internally and still return a reference.
        assert_eq!(dq.try_push_front_mut(0), Ok(&mut 0));
        assert!(dq.capacity() > 2);
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([0, 1, 2]));
    }

    #[test]
    fn push_back_mut_oom_leaves_deque_unchanged() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let err = match dq.try_push_back_mut(1) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        assert!(dq.is_empty());
        assert_eq!(dq.capacity(), 0);
    }

    #[test]
    fn push_back_mut_give_back_recovers_value_on_failure() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let (returned, err) = match dq.try_push_back_mut_give_back(42) {
            Err(pair) => pair,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert_eq!(returned, 42);
        assert!(err.is_alloc());
        assert!(dq.is_empty());
    }

    #[test]
    fn push_front_mut_oom_leaves_deque_unchanged() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let err = match dq.try_push_front_mut(1) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        assert!(dq.is_empty());
    }

    #[test]
    fn push_front_mut_give_back_recovers_value_on_failure() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let (returned, err) = match dq.try_push_front_mut_give_back(42) {
            Err(pair) => pair,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert_eq!(returned, 42);
        assert!(err.is_alloc());
        assert!(dq.is_empty());
    }

    #[test]
    fn push_mut_family_works_across_wrap_boundary() {
        // Build a full wrapped state: [3, 2, 1, 4] spanning two physical
        // segments (head retreated past 0 by the push_front).
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        assert_eq!(dq.try_push_back(3), Ok(()));
        assert_eq!(dq.try_push_front(4), Ok(()));
        assert_eq!(dq.len(), 4);
        // Deque is full; both directions must grow internally while the
        // elements are still wrapped across the buffer boundary.
        assert_eq!(dq.try_push_back_mut(9), Ok(&mut 9));
        assert!(dq.capacity() > 4);
        assert_eq!(dq.back(), Some(&9));
        assert_eq!(dq.try_push_front_mut(8), Ok(&mut 8));
        assert_eq!(dq.front(), Some(&8));
        assert_eq!(dq.len(), 6);
        assert_eq!(collect_into_array::<6>(&dq), Some([8, 4, 1, 2, 3, 9]));
    }

    // --- Mixed sequences ---------------------------------------------------------

    #[test]
    fn interleaved_growth_preserves_order() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        // Each push_front prepends, so later front-pushes end up more toward
        // the front. Trace:
        //   pb(5) -> [5]
        //   pf(4) -> [4, 5]
        //   pb(6) -> [4, 5, 6]
        //   pf(3) -> [3, 4, 5, 6]
        //   pb(7) -> [3, 4, 5, 6, 7]  (triggers growth 4->8)
        //   pf(2) -> [2, 3, 4, 5, 6, 7]
        //   pb(8) -> [2, 3, 4, 5, 6, 7, 8]
        //   pf(1) -> [1, 2, 3, 4, 5, 6, 7, 8]
        assert_eq!(dq.try_push_back(5), Ok(()));
        assert_eq!(dq.try_push_front(4), Ok(()));
        assert_eq!(dq.try_push_back(6), Ok(()));
        assert_eq!(dq.try_push_front(3), Ok(()));
        assert_eq!(dq.try_push_back(7), Ok(()));
        assert_eq!(dq.try_push_front(2), Ok(()));
        assert_eq!(dq.try_push_back(8), Ok(()));
        assert_eq!(dq.try_push_front(1), Ok(()));
        assert_eq!(dq.len(), 8);
        assert_eq!(collect_into_array::<8>(&dq), Some([1, 2, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn zst_pushes_always_succeed() {
        let mut dq: VecDeque<()> = VecDeque::new();
        for _ in 0..100 {
            assert_eq!(dq.try_push_back(()), Ok(()));
            assert_eq!(dq.try_push_front(()), Ok(()));
        }
        assert_eq!(dq.len(), 200);
        assert_eq!(size_of::<()>(), 0);
    }

    #[test]
    fn zst_pushes_report_max_capacity() {
        let mut dq: VecDeque<()> = VecDeque::new();
        assert_eq!(dq.capacity(), usize::MAX);
        assert_eq!(dq.try_push_back(()), Ok(()));
        assert_eq!(dq.try_push_front(()), Ok(()));
        // Capacity stays at max; no allocation ever happens.
        assert_eq!(dq.capacity(), usize::MAX);
        assert_eq!(dq.len(), 2);
    }

    #[test]
    fn many_pushes_stay_consistent_under_growth() {
        let mut dq: VecDeque<usize> = VecDeque::new();
        for i in 0..1000usize {
            assert_eq!(dq.try_push_back(i), Ok(()));
        }
        // Push front in descending order: 1999, 1998, ..., 1000.
        // Each push_front prepends, so the last one (1000) ends up at the front.
        // Final logical order: [1000, 1001, ..., 1999, 0, 1, ..., 999]
        for i in (0..1000usize).rev() {
            assert_eq!(dq.try_push_front(i + 1000), Ok(()));
        }
        assert_eq!(dq.len(), 2000);
        let (a, b) = dq.as_slices();
        assert_eq!(a.len() + b.len(), 2000);
        // Spot-check the boundaries.
        assert_eq!(dq.front(), Some(&1000));
        assert_eq!(dq.back(), Some(&999));
        // Verify a few interior elements straddling the growth boundary.
        assert_eq!(dq.get(500), Some(&1500));
        assert_eq!(dq.get(1000), Some(&0));
        assert_eq!(dq.get(1500), Some(&500));
    }

    // --- try_insert / try_insert_mut ---------------------------------------------

    #[test]
    fn insert_at_front_of_empty_deque() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        assert_eq!(dq.try_insert(0, 42), Ok(()));
        assert_eq!(dq.len(), 1);
        assert_eq!(dq.front(), Some(&42));
    }

    #[test]
    fn insert_at_back_equals_push_back() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        // Inserting at index == len should behave like push_back.
        assert_eq!(dq.try_insert(2, 3), Ok(()));
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn insert_in_middle_shifts_elements() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.try_insert(2, 99), Ok(()));
        assert_eq!(collect_into_array::<5>(&dq), Some([1, 2, 99, 3, 4]));
    }

    #[test]
    fn insert_at_front_with_other_elements() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.try_insert(0, 0), Ok(()));
        assert_eq!(collect_into_array::<4>(&dq), Some([0, 1, 2, 3]));
    }

    #[test]
    fn insert_out_of_bounds_returns_error() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        let err = match dq.try_insert(5, 99) {
            Err(e) => e,
            Ok(_) => panic!("expected out-of-bounds error"),
        };
        assert_eq!(err, TryVecDequeInsertError::OutOfBounds);
        assert_eq!(dq.len(), 1);
        assert_eq!(dq.front(), Some(&1));
    }

    #[test]
    fn insert_give_back_recovers_value_on_oom() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let (returned, err) = match dq.try_insert_give_back(0, 42) {
            Err(pair) => pair,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert_eq!(returned, 42);
        assert!(matches!(err, TryVecDequeInsertError::Reserve(_)));
        assert!(dq.is_empty());
    }

    #[test]
    fn insert_mut_returns_reference_to_inserted_element() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(3), Ok(()));
        assert_eq!(dq.try_insert_mut(1, 2), Ok(&mut 2));
        *dq.get_mut(1).unwrap() += 10;
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 12, 3]));
    }

    #[test]
    fn insert_mut_grow_when_full() {
        let mut dq = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(3), Ok(()));
        // Full; inserting in the middle must grow and still work.
        assert_eq!(dq.try_insert_mut(1, 2), Ok(&mut 2));
        assert!(dq.capacity() > 2);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn insert_across_wrap_boundary() {
        // Build a wrapped state: push backs then fronts to wrap head past 0.
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        assert_eq!(dq.try_push_back(3), Ok(()));
        assert_eq!(dq.try_push_front(4), Ok(()));
        // Now: [4, 1, 2, 3], full, head has wrapped.
        // Grow by inserting in the middle (forces reserve).
        assert_eq!(dq.try_insert(2, 99), Ok(()));
        assert_eq!(dq.len(), 5);
        assert_eq!(collect_into_array::<5>(&dq), Some([4, 1, 99, 2, 3]));
    }

    #[test]
    fn insert_zst_succeeds() {
        let mut dq: VecDeque<()> = VecDeque::new();
        assert_eq!(dq.try_push_back(()), Ok(()));
        assert_eq!(dq.try_push_back(()), Ok(()));
        assert_eq!(dq.try_insert(1, ()), Ok(()));
        assert_eq!(dq.len(), 3);
    }

    // --- try_append / try_prepend --------------------------------------------------

    #[test]
    fn append_moves_elements_to_back() {
        let mut a = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        let mut b = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(a.try_push_back(1), Ok(()));
        assert_eq!(a.try_push_back(2), Ok(()));
        assert_eq!(b.try_push_back(3), Ok(()));
        assert_eq!(b.try_push_back(4), Ok(()));
        assert_eq!(a.try_append(&mut b), Ok(()));
        assert_eq!(collect_into_array::<4>(&a), Some([1, 2, 3, 4]));
        assert!(b.is_empty());
    }

    #[test]
    fn append_empty_source_is_noop() {
        let mut a = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        let mut b: VecDeque<i32> = VecDeque::new();
        assert_eq!(a.try_push_back(1), Ok(()));
        assert_eq!(a.try_append(&mut b), Ok(()));
        assert_eq!(collect_into_array::<1>(&a), Some([1]));
    }

    #[test]
    fn append_grows_when_needed() {
        let mut a = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        let mut b = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(a.try_push_back(1), Ok(()));
        assert_eq!(a.try_push_back(2), Ok(()));
        for v in [3, 4, 5] {
            assert_eq!(b.try_push_back(v), Ok(()));
        }
        assert_eq!(a.try_append(&mut b), Ok(()));
        assert!(a.capacity() >= 5);
        assert_eq!(collect_into_array::<5>(&a), Some([1, 2, 3, 4, 5]));
        assert!(b.is_empty());
    }

    #[test]
    fn append_from_populated_to_full_target_grows() {
        // Target is full; appending must trigger growth internally.
        let mut a = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        let mut b = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(a.try_push_back(1), Ok(()));
        assert_eq!(a.try_push_back(2), Ok(()));
        assert_eq!(b.try_push_back(3), Ok(()));
        assert_eq!(b.try_push_back(4), Ok(()));
        assert_eq!(a.try_append(&mut b), Ok(()));
        assert!(a.capacity() >= 4);
        assert_eq!(collect_into_array::<4>(&a), Some([1, 2, 3, 4]));
        assert!(b.is_empty());
    }

    #[test]
    fn prepend_moves_elements_to_front() {
        let mut a = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        let mut b = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(a.try_push_back(3), Ok(()));
        assert_eq!(a.try_push_back(4), Ok(()));
        assert_eq!(b.try_push_back(1), Ok(()));
        assert_eq!(b.try_push_back(2), Ok(()));
        assert_eq!(a.try_prepend(&mut b), Ok(()));
        assert_eq!(collect_into_array::<4>(&a), Some([1, 2, 3, 4]));
        assert!(b.is_empty());
    }

    #[test]
    fn prepend_preserves_source_order() {
        let mut a = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        let mut b = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        assert_eq!(a.try_push_back(10), Ok(()));
        assert_eq!(b.try_push_back(1), Ok(()));
        assert_eq!(b.try_push_back(2), Ok(()));
        assert_eq!(b.try_push_back(3), Ok(()));
        assert_eq!(a.try_prepend(&mut b), Ok(()));
        assert_eq!(collect_into_array::<4>(&a), Some([1, 2, 3, 10]));
    }

    #[test]
    fn prepend_grows_when_needed() {
        let mut a = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        let mut b = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(a.try_push_back(3), Ok(()));
        assert_eq!(a.try_push_back(4), Ok(()));
        for v in [1, 2, 5] {
            assert_eq!(b.try_push_back(v), Ok(()));
        }
        assert_eq!(a.try_prepend(&mut b), Ok(()));
        assert!(a.capacity() >= 5);
        assert_eq!(collect_into_array::<5>(&a), Some([1, 2, 5, 3, 4]));
        assert!(b.is_empty());
    }

    #[test]
    fn prepend_from_populated_to_full_target_grows() {
        // Target is full; prepending must trigger growth internally.
        let mut a = VecDeque::<i32>::try_with_capacity(2).expect("allocation ok");
        let mut b = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        assert_eq!(a.try_push_back(3), Ok(()));
        assert_eq!(a.try_push_back(4), Ok(()));
        assert_eq!(b.try_push_back(1), Ok(()));
        assert_eq!(b.try_push_back(2), Ok(()));
        assert_eq!(a.try_prepend(&mut b), Ok(()));
        assert!(a.capacity() >= 4);
        assert_eq!(collect_into_array::<4>(&a), Some([1, 2, 3, 4]));
        assert!(b.is_empty());
    }

    /// Builds a populated 2-item deque whose next growth will hit OOM.
    fn oom_deque(starting: i32) -> VecDeque<i32, BudgetedAlloc> {
        let mut dq = VecDeque::<i32, _>::try_with_capacity_in(2, BudgetedAlloc::new(1))
            .expect("within budget");
        assert_eq!(dq.try_push_back(starting), Ok(()));
        assert_eq!(dq.try_push_back(starting + 1), Ok(()));
        dq
    }

    #[test]
    fn append_oom_leaves_both_deques_unchanged() {
        // `a`'s allocator has no budget left, so the reserve in `try_append`
        // must fail before any element moves.
        let mut a = oom_deque(1); // [1, 2]
        let mut b = oom_deque(3); // [3, 4]
        let err = match a.try_append(&mut b) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        // Neither deque may have been touched on failure.
        assert_eq!(collect_any::<2, BudgetedAlloc>(&a), Some([1, 2]));
        assert_eq!(collect_any::<2, BudgetedAlloc>(&b), Some([3, 4]));
    }

    #[test]
    fn prepend_oom_leaves_both_deques_unchanged() {
        let mut a = oom_deque(1); // [1, 2]
        let mut b = oom_deque(3); // [3, 4]
        let err = match a.try_prepend(&mut b) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        assert_eq!(collect_any::<2, BudgetedAlloc>(&a), Some([1, 2]));
        assert_eq!(collect_any::<2, BudgetedAlloc>(&b), Some([3, 4]));
    }

    #[test]
    fn append_with_wrapped_source_and_target() {
        // Build wrapped states in both deques so the transfer crosses physical
        // segment boundaries on both sides.
        let mut a = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        let mut b = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        // a: [3, 1, 2] with head retreated (wrapped layout).
        assert_eq!(a.try_push_back(1), Ok(()));
        assert_eq!(a.try_push_back(2), Ok(()));
        assert_eq!(a.try_push_front(3), Ok(()));
        // b: [5, 6, 7] with head retreated (wrapped layout).
        assert_eq!(b.try_push_back(6), Ok(()));
        assert_eq!(b.try_push_back(7), Ok(()));
        assert_eq!(b.try_push_front(5), Ok(()));
        assert_eq!(a.try_append(&mut b), Ok(()));
        assert_eq!(collect_into_array::<6>(&a), Some([3, 1, 2, 5, 6, 7]));
        assert!(b.is_empty());
    }

    #[test]
    fn prepend_with_wrapped_source_and_target() {
        let mut a = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        let mut b = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        // a: [3, 1, 2] wrapped.
        assert_eq!(a.try_push_back(1), Ok(()));
        assert_eq!(a.try_push_back(2), Ok(()));
        assert_eq!(a.try_push_front(3), Ok(()));
        // b: [5, 6, 7] wrapped.
        assert_eq!(b.try_push_back(6), Ok(()));
        assert_eq!(b.try_push_back(7), Ok(()));
        assert_eq!(b.try_push_front(5), Ok(()));
        assert_eq!(a.try_prepend(&mut b), Ok(()));
        // b's front (5) lands adjacent to a's old front (3).
        assert_eq!(collect_into_array::<6>(&a), Some([5, 6, 7, 3, 1, 2]));
        assert!(b.is_empty());
    }

    #[test]
    fn append_and_prepend_zst() {
        let mut a: VecDeque<()> = VecDeque::new();
        let mut b: VecDeque<()> = VecDeque::new();
        assert_eq!(a.try_push_back(()), Ok(()));
        assert_eq!(b.try_push_back(()), Ok(()));
        assert_eq!(b.try_push_back(()), Ok(()));
        assert_eq!(a.try_append(&mut b), Ok(()));
        assert_eq!(a.len(), 3);
        assert!(b.is_empty());

        let mut c: VecDeque<()> = VecDeque::new();
        assert_eq!(c.try_push_back(()), Ok(()));
        assert_eq!(a.try_prepend(&mut c), Ok(()));
        assert_eq!(a.len(), 4);
        assert!(c.is_empty());
    }
}
