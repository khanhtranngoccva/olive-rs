//! Allocation-backed mutation for [`VecDeque`].
//!
//! These methods may grow the underlying buffer, so they return a
//! [`Result`] carrying a [`TryReserveError`] instead of panicking on
//! out-of-memory. Each public push is paired with a `*_give_back` variant that
//! hands the uninserted value back to the caller on failure, mirroring the
//! convention established by `Vec::try_push_give_back`.

use super::VecDeque;
use crate::collections::vec_deque::mutation::TryPushWithinCapacityError;
use olive_core::alloc::Allocator;
use olive_core::alloc_errors::{TryReserveError, TryReserveErrorKind};

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
    pub fn try_push_back_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, TryReserveError)> {
        if self.len == self.capacity() {
            if let Err(e) = self.try_reserve(1) {
                return Err((value, e));
            }
        }
        match self.try_push_back_within_capacity(value) {
            Ok(()) => Ok(()),
            Err(TryPushWithinCapacityError { .. }) => {
                unreachable!("capacity was just secured above")
            }
        }
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
    pub fn try_push_front_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, TryReserveError)> {
        if self.len == self.capacity() {
            if let Err(e) = self.try_reserve(1) {
                return Err((value, e));
            }
        }
        match self.try_push_front_within_capacity(value) {
            Ok(()) => Ok(()),
            Err(TryPushWithinCapacityError { .. }) => {
                unreachable!("capacity was just secured above")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::test_helpers::FailAlloc;
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
}
