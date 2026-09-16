//! Trait implementations for [`VecDeque`].
//!
//! Covers the fallible construction and extension traits from `olive-core`:
//! [`TryClone`], [`TryExtend`], [`TryExtendFromSlice`], and
//! [`TryFromIterator`], plus a standard-library [`Debug`] impl.

use core::fmt;

use olive_core::alloc::{Allocator, AllocatorTryClone};
use olive_core::alloc_errors::TryReserveError;
use olive_core::recovery::{ResumableSource, Resume};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
use olive_core::try_traits::try_from_iterator::TryFromIterator;

use super::VecDeque;
use crate::alloc::Global;

/// Error returned by fallible deque operations that may both reserve capacity
/// and clone elements.
///
/// Covers [`TryClone`] on [`VecDeque`] — any operation whose failure modes are
/// limited to a capacity reservation ([`TryReserveError`]) or an element clone
/// failure ([`TryCloneError`]). Mirrors `Vec`'s `TryVecWithCloneError`.
#[derive(Clone, PartialEq, Eq)]
pub enum TryVecDequeWithCloneError {
    /// A capacity reservation on the deque failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
}

impl fmt::Debug for TryVecDequeWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .debug_tuple("TryVecDequeWithCloneError::Reserve")
                .field(e)
                .finish(),
            Self::Clone(e) => f
                .debug_tuple("TryVecDequeWithCloneError::Clone")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryVecDequeWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "deque operation failed: {e}"),
            Self::Clone(e) => write!(f, "deque operation failed: {e}"),
        }
    }
}

impl core::error::Error for TryVecDequeWithCloneError {}

impl From<TryReserveError> for TryVecDequeWithCloneError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryVecDequeWithCloneError {
    #[inline]
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl<T, A: Allocator> fmt::Debug for VecDeque<T, A>
where
    T: fmt::Debug,
{
    /// Formats the deque as a comma-separated list of its elements in logical
    /// order (front to back), regardless of how the circular buffer is laid
    /// out physically. An empty deque renders as `[]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// ---------------------------------------------------------------------------
// TryClone
// ---------------------------------------------------------------------------

impl<T: TryClone, A: AllocatorTryClone> TryClone for VecDeque<T, A> {
    /// Fallibly clone a deque, element by element via [`TryClone`].
    ///
    /// The backing allocator is cloned first so the result lives on an
    /// equivalent allocator (mirroring `Box`'s requirement that the clone stay
    /// on the same backing store). Capacity is reserved up front before any
    /// element work, so an allocation failure short-circuits early.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if cloning the allocator or reserving
    /// capacity fails, or if cloning an element fails.
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let alloc = self.buf.allocator().try_clone()?;
        let mut out =
            Self::try_with_capacity_in(self.len(), alloc).map_err(TryCloneError::Reserve)?;
        for elem in self.iter() {
            let cloned = elem.try_clone()?;
            // SAFETY: capacity was reserved above for every element.
            unsafe { out.push_back_within_cap(cloned) };
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TryFromIterator
// ---------------------------------------------------------------------------

impl<T> TryFromIterator<T> for VecDeque<T, Global> {
    type Error = TryReserveError;

    /// Fallibly collect an iterator into a deque on the default [`Global`]
    /// allocator. For a custom allocator use [`VecDeque::try_from_iter_in`].
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if a reservation fails.
    fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Self::Error> {
        Self::try_from_iter_in(iter, Global)
    }
}

impl<T, A: Allocator> VecDeque<T, A> {
    /// Fallibly collects an iterator into a [`VecDeque<T>`] on the given
    /// allocator.
    ///
    /// This is the allocator-aware backend for [`TryFromIterator`]. The size
    /// hint's upper bound (or lower bound when no upper is advertised) seeds a
    /// single best-effort batch reserve up front so that well-behaved iterators
    /// allocate only once in the best scenario.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving capacity for an incoming
    /// element fails. On failure the deque retains every element already
    /// consumed.
    pub fn try_from_iter_in<I: IntoIterator<Item = T>>(
        iter: I,
        alloc: A,
    ) -> Result<Self, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let mut deque = Self::new_in(alloc);
        // Best-effort batch reserve from the hint; ignore failures so a bogus
        // hint cannot abort collection before any element is seen.
        let _ = deque.try_reserve_total(upper.unwrap_or(lower));
        for item in iter {
            // The iterator may yield more elements than its hint promised.
            if deque.len == deque.capacity() {
                deque.try_reserve(1)?;
            }
            // SAFETY: a spare slot exists (we grew if needed).
            unsafe { deque.push_back_within_cap(item) };
        }
        Ok(deque)
    }
}

// ---------------------------------------------------------------------------
// TryExtend
// ---------------------------------------------------------------------------

impl<T, A: Allocator> TryExtend<T> for VecDeque<T, A> {
    type Error = TryReserveError;

    /// Fallibly extend the deque with all items produced by `source`, appending
    /// them to the back.
    ///
    /// The source is decomposed into an optional stranded head and a remainder
    /// iterator, and capacity can be reserved up front from the size hint's
    /// estimated total (best-effort — over-reserve failures are ignored).
    ///
    /// # Errors
    ///
    /// Returns `(Resume<S::Inner>, TryReserveError)` if a reservation or push
    /// fails. The [`Resume`] carries the stranded element (if any) alongside
    /// the unconsumed remainder so the caller can retry with a stable error
    /// type.
    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = T>,
    {
        let (head, mut inner, hint) = source.decompose_with_size_hint();
        // Ignore over-reserve failures; growth happens lazily below.
        let _ = self.try_reserve_total(hint.estimated_total());
        // Push the head first.
        if let Some(head) = head {
            if let Err((head, err)) = self.try_push_back_give_back(head) {
                return Err((Resume::new(head, inner), err));
            }
        }

        // Push the remainder. While we have spare capacity this is a cheap
        // within-capacity push; once capacity is exhausted (under-hinted or
        // OOM'd) grow one slot at a time, stranding the current element on
        // failure.
        while let Some(next) = inner.next() {
            if self.len == self.capacity() {
                if let Err(e) = self.try_reserve(1) {
                    return Err((Resume::new(next, inner), e));
                }
            }
            // SAFETY: a spare slot was just confirmed.
            unsafe { self.push_back_within_cap(next) };
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TryExtendFromSlice
// ---------------------------------------------------------------------------

impl<'s, T, A: Allocator> TryExtendFromSlice<'s, T> for VecDeque<T, A>
where
    T: TryClone,
{
    type Error = TryVecDequeWithCloneError;

    /// Fallibly extend the deque by cloning each element of `other` and
    /// appending it to the back.
    ///
    /// Capacity is reserved for the entire slice up front so that a mid-way
    /// clone failure does not leave the deque in a partially-grown state. On
    /// failure the error carries the unconsumed tail of `other` (starting at
    /// the first element whose clone failed) so the caller can retry once
    /// memory pressure has eased.
    ///
    /// # Errors
    ///
    /// Returns `(&'s [T], TryVecDequeWithCloneError)` if reserving capacity or
    /// cloning an element fails. The returned slice is the remainder beginning
    /// at the first failed element.
    fn try_extend_from_slice(&mut self, other: &'s [T]) -> Result<(), (&'s [T], Self::Error)> {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(|e| (other, TryVecDequeWithCloneError::Reserve(e)))?;
        let mut i = 0usize;
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    // SAFETY: capacity was reserved above for all of `other`.
                    unsafe { self.push_back_within_cap(cloned) };
                    #[allow(clippy::arithmetic_side_effects, reason = "i <= other.len()")]
                    {
                        i += 1;
                    }
                }
                Err(e) => return Err((&other[i..], TryVecDequeWithCloneError::Clone(e))),
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Front extension (inherent — deque-specific, no std counterpart)
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Fallibly extend the front of the deque with all items produced by
    /// `source`, consuming them in order and prepending each as it arrives —
    /// the same semantics as nightly std's `VecDeque::extend_front`. As a
    /// result the source's **last** element ends up at the very front, and the
    /// pre-existing elements shift toward the back.
    ///
    /// This is the front-facing analogue of [`TryExtend::try_extend`]. Because
    /// prepending is specific to deque semantics (there is no stable
    /// standard-library trait it parallels), it is an inherent method rather
    /// than a trait impl.
    ///
    /// Capacity can be reserved up front from the size hint's estimated total
    /// (best-effort — over-reserve failures are ignored); growth otherwise
    /// happens lazily.
    ///
    /// # Errors
    ///
    /// Returns `(Resume<S::Inner>, TryReserveError)` if a reservation or push
    /// fails. The [`Resume`] carries the stranded element (if any) alongside
    /// the unconsumed remainder so the caller can retry with a stable error
    /// type. Already-prepended elements remain committed at the front of the
    /// deque.
    pub fn try_extend_front<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, TryReserveError)>
    where
        S: ResumableSource<Item = T>,
    {
        let (head, mut inner, hint) = source.decompose_with_size_hint();
        // Ignore over-reserve failures; growth happens lazily below.
        let _ = self.try_reserve_total(hint.estimated_total());
        // Push the head first.
        if let Some(head) = head {
            if let Err((head, err)) = self.try_push_front_give_back(head) {
                return Err((Resume::new(head, inner), err));
            }
        }

        // Each subsequent item is prepended as it arrives (nightly std
        // semantics): the last-produced item ends up at the very front. While
        // we have spare capacity this is a cheap within-capacity push; once
        // capacity is exhausted grow one slot at a time, stranding the current
        // element on failure.
        while let Some(next) = inner.next() {
            if self.len == self.capacity() {
                if let Err(e) = self.try_reserve(1) {
                    return Err((Resume::new(next, inner), e));
                }
            }
            // SAFETY: a spare slot was just confirmed.
            unsafe { self.push_front_within_cap(next) };
        }
        Ok(())
    }

    /// Fallibly extend the front of the deque by cloning each element of
    /// `other` and prepending it as we walk forward — the same semantics as
    /// nightly std's `extend_front`. As a result the slice's **last** element
    /// ends up at the very front.
    ///
    /// This is the front-facing analogue of
    /// [`TryExtendFromSlice::try_extend_from_slice`]. Because prepending is
    /// specific to deque semantics, it is an inherent method rather than a
    /// trait impl.
    ///
    /// Capacity is reserved for the entire slice up front so that a mid-way
    /// clone failure does not leave the deque in a partially-grown state. On
    /// failure the error carries the unconsumed tail of `other` (starting at
    /// the first element whose clone failed) so the caller can retry once
    /// memory pressure has eased.
    ///
    /// # Errors
    ///
    /// Returns `(&'s [T], TryVecDequeWithCloneError)` if reserving capacity or
    /// cloning an element fails. The returned slice is the remainder beginning
    /// at the first failed element.
    pub fn try_extend_front_from_slice<'s>(
        &mut self,
        other: &'s [T],
    ) -> Result<(), (&'s [T], TryVecDequeWithCloneError)>
    where
        T: TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(|e| (other, TryVecDequeWithCloneError::Reserve(e)))?;
        // Each element is cloned and prepended as we walk forward, so the last
        // element ends up at the very front.
        let mut i = 0usize;
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    // SAFETY: capacity was reserved above for all of `other`.
                    unsafe { self.push_front_within_cap(cloned) };
                    #[allow(clippy::arithmetic_side_effects, reason = "i <= other.len()")]
                    {
                        i += 1;
                    }
                }
                Err(e) => return Err((&other[i..], TryVecDequeWithCloneError::Clone(e))),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::test_helpers::{CloneBudget, FailAlloc, FlakyClone, FlakyTrackedItem, Ledger};
    use std::format;
    use std::sync::Arc;

    // --- Debug ---------------------------------------------------------------

    #[test]
    fn debug_empty_deque() {
        let dq: VecDeque<i32> = VecDeque::new();
        assert_eq!(format!("{dq:?}"), "[]");
    }

    #[test]
    fn debug_nonempty_deque() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        for v in [1, 2, 3] {
            dq.try_push_back(v).unwrap();
        }
        assert_eq!(format!("{dq:?}"), "[1, 2, 3]");
    }

    #[test]
    fn debug_wrapped_deque_preserves_logical_order() {
        // Force a wrapped layout: fill cap-3, pop 2 from front, push 1 back.
        let mut dq = VecDeque::<i32>::try_with_capacity(3).expect("allocation ok");
        for v in [1, 2, 3] {
            dq.try_push_back(v).unwrap();
        }
        assert_eq!(dq.pop_front(), Some(1));
        assert_eq!(dq.pop_front(), Some(2));
        dq.try_push_back(4).unwrap();
        assert!(!dq.is_contiguous());
        // Logical order must be [3, 4] regardless of physical wrap.
        assert_eq!(format!("{dq:?}"), "[3, 4]");
    }

    // --- TryClone ------------------------------------------------------------

    #[test]
    fn try_clone_empty() {
        let dq: VecDeque<i32> = VecDeque::new();
        let cloned = dq.try_clone().expect("clone ok");
        assert!(cloned.is_empty());
    }

    #[test]
    fn try_clone_preserves_logical_order() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        for v in [5, 6, 7] {
            dq.try_push_back(v).unwrap();
        }
        dq.try_push_front(4).unwrap();
        let cloned = dq.try_clone().expect("clone ok");
        let (a, b) = cloned.as_slices();
        let mut all: std::vec::Vec<i32> = a.iter().copied().chain(b.iter().copied()).collect();
        all.sort_unstable_by_key(|&x| x);
        assert_eq!(all, [4, 5, 6, 7]);
        assert_eq!(cloned.front(), Some(&4));
        assert_eq!(cloned.back(), Some(&7));
    }

    #[test]
    fn try_clone_wrapped_state() {
        // Build a genuinely wrapped deque, then clone it.
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        for v in [1, 2, 3, 4] {
            dq.try_push_back(v).unwrap();
        }
        dq.pop_front();
        dq.pop_front();
        dq.try_push_back(5).unwrap();
        dq.try_push_back(6).unwrap();
        assert!(!dq.is_contiguous());
        let cloned = dq.try_clone().expect("clone ok");
        assert_eq!(cloned.len(), 4);
        assert_eq!(cloned.get(0), Some(&3));
        assert_eq!(cloned.get(1), Some(&4));
        assert_eq!(cloned.get(2), Some(&5));
        assert_eq!(cloned.get(3), Some(&6));
    }

    #[test]
    fn try_clone_empty_on_global_succeeds() {
        // Empty deque: no allocation needed, clone succeeds.
        let dq: VecDeque<i32> = VecDeque::new();
        let cloned = dq.try_clone().expect("clone ok");
        assert!(cloned.is_empty());
    }

    #[test]
    fn try_clone_reports_allocator_clone_failure() {
        // A flaky allocator with an exhausted clone budget must surface its
        // failure as a `TryCloneError` before any element work begins.
        use crate::test_helpers::{CloneBudget, FlakyCloneAlloc};
        use std::sync::Arc;
        let alloc = FlakyCloneAlloc::new(Arc::new(CloneBudget::new(0)));
        let dq: VecDeque<i32, FlakyCloneAlloc> = VecDeque::new_in(alloc);
        let err = dq.try_clone().expect_err("allocator clone must fail");
        assert!(matches!(err, TryCloneError::Other(_)));
    }

    #[test]
    fn try_clone_with_budgeted_allocator_succeeds_and_matches() {
        // With a sufficient clone budget the clone succeeds and preserves the
        // logical order of elements.
        use crate::test_helpers::{CloneBudget, FlakyCloneAlloc};
        use std::sync::Arc;
        let alloc = FlakyCloneAlloc::new(Arc::new(CloneBudget::new(16)));
        let mut dq: VecDeque<i32, FlakyCloneAlloc> = VecDeque::new_in(alloc);
        for v in [9, 8, 7] {
            dq.try_push_back(v).unwrap();
        }
        let cloned = dq.try_clone().expect("clone ok");
        assert_eq!(cloned.len(), 3);
        assert_eq!(cloned.front(), Some(&9));
        assert_eq!(cloned.back(), Some(&7));
    }

    #[test]
    fn try_clone_element_failure_propagates_and_drops_partial() {
        // A mid-clone element failure must surface as a `TryCloneError` and the
        // partially-built result deque must be dropped cleanly — each transient
        // clone exactly once, no leaks, no double-frees.
        let ledger = Arc::new(Ledger::new());
        // Budget of 2 allows exactly two successful clones; the third fails.
        let budget = Arc::new(CloneBudget::new(2));

        // Seed three elements into the source deque (ids 0, 1, 2).
        let mut dq: VecDeque<FlakyTrackedItem> = VecDeque::new();
        for _ in 0..3 {
            let id = ledger.allocate();
            ledger.register(id);
            dq.try_push_back(FlakyTrackedItem {
                id,
                ledger: ledger.clone(),
                inner: (*budget).share(),
            })
            .unwrap();
        }

        let err = match dq.try_clone() {
            Ok(_) => panic!("element clone must fail"),
            Err(e) => e,
        };
        assert!(matches!(err, TryCloneError::Other(_)));

        // The two successfully-cloned transients (ids 3 and 4) were dropped
        // exactly once when the half-built result deque was discarded.
        assert_eq!(ledger.total_allocated(), 5);
        assert_eq!(
            ledger.drop_count(3),
            1,
            "first transient clone dropped once"
        );
        assert_eq!(
            ledger.drop_count(4),
            1,
            "second transient clone dropped once"
        );
        assert_eq!(ledger.drop_count(5), 0, "no fifth id was ever minted");
        // Source elements are still alive; nothing leaked or double-freed.
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());

        // Tear down the source deque: all three seeds drop exactly once too.
        drop(dq);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    // --- TryFromIterator -------------------------------------------------------

    #[test]
    fn try_from_iter_collects_all_elements() {
        let dq = VecDeque::<i32>::try_from_iter(0..10).expect("collect ok");
        assert_eq!(dq.len(), 10);
        assert_eq!(dq.front(), Some(&0));
        assert_eq!(dq.back(), Some(&9));
    }

    #[test]
    fn try_from_iter_empty() {
        let dq = VecDeque::<i32>::try_from_iter(std::iter::empty()).expect("collect ok");
        assert!(dq.is_empty());
    }

    #[test]
    fn try_from_iter_underhinted_grows_as_needed() {
        // An iterator whose size hint underestimates the true count forces
        // mid-collection growth.
        struct Underhinted(core::ops::Range<i32>);
        impl Iterator for Underhinted {
            type Item = i32;
            fn next(&mut self) -> Option<i32> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(1))
            }
        }
        let items = Underhinted(0..8);
        let dq = VecDeque::<i32>::try_from_iter(items).expect("collect ok");
        assert_eq!(dq.len(), 8);
        assert_eq!(dq.front(), Some(&0));
        assert_eq!(dq.back(), Some(&7));
    }

    #[test]
    fn try_from_iter_overhint_falls_back_to_incremental_growth() {
        // An iterator that advertises a huge upper bound but yields only a few
        // elements. The upfront batch reserve is too large for the byte-capped
        // allocator and fails silently; collection then proceeds via
        // per-element reserves that each stay within the cap.
        use crate::test_helpers::allocators::ByteCapAlloc;

        struct Overhinted(core::ops::Range<i32>);
        impl Iterator for Overhinted {
            type Item = i32;
            fn next(&mut self) -> Option<i32> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                // Claim up to 10 000 elements; actually yield only 3.
                (0, Some(10_000))
            }
        }

        // Cap at 64 bytes: a bulk reserve for 10 000 × 4-byte i32s (~40 KB)
        // is rejected, but expanding to up to 64 bytes succeed.
        let alloc = ByteCapAlloc::new(64);
        let dq = VecDeque::<i32, _>::try_from_iter_in(Overhinted(0..3), alloc)
            .expect("incremental growth should succeed");
        assert_eq!(dq.len(), 3);
        assert_eq!(dq.front(), Some(&0));
        assert_eq!(dq.back(), Some(&2));
    }

    // --- TryExtend -------------------------------------------------------------

    #[test]
    fn try_extend_appends_to_back() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(100).unwrap();
        dq.try_extend(1..=3).expect("extend ok");
        assert_eq!(dq.len(), 4);
        assert_eq!(dq.front(), Some(&100));
        assert_eq!(dq.back(), Some(&3));
    }

    #[test]
    fn try_extend_empty_source_is_noop() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_extend(std::iter::empty()).expect("extend ok");
        assert_eq!(dq.len(), 1);
    }

    #[test]
    fn try_extend_retry_via_resume_preserves_items() {
        // Simulate a failure: extend with a source backed by a failing
        // allocator would strand items; here we verify the retry contract by
        // feeding a Resume directly.
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(0).unwrap();
        let resume = Resume::new(10, 11..13);
        dq.try_extend(resume).expect("extend ok");
        assert_eq!(dq.len(), 4);
        assert_eq!(dq.get(0), Some(&0));
        assert_eq!(dq.get(1), Some(&10));
        assert_eq!(dq.get(2), Some(&11));
        assert_eq!(dq.get(3), Some(&12));
    }

    #[test]
    fn try_extend_oom_strands_head_and_returns_remainder() {
        // A failing allocator cannot fund the initial reserve, but since we
        // ignore over-reserve failures the head push will fail instead.
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let (resume, err) = match dq.try_extend(1..=3) {
            Ok(_) => panic!("expected allocation failure"),
            Err(pair) => pair,
        };
        assert!(err.is_alloc());
        // The stranded head is the first element; the remainder holds the rest.
        assert_eq!(*resume.head().expect("stranded head present"), 1);
        let collected: std::vec::Vec<i32> = resume.into_remainder().collect();
        assert_eq!(collected, [2, 3]);
        // The deque itself is unchanged.
        assert!(dq.is_empty());
    }

    #[test]
    fn try_extend_overhint_falls_back_to_incremental_growth() {
        // An iterator that advertises a huge upper bound but yields only a few
        // elements. The upfront batch reserve is too large for the byte-capped
        // allocator and fails silently; extend then proceeds via per-element
        // reserves that each stay within the cap.
        use crate::test_helpers::allocators::ByteCapAlloc;

        #[derive(Debug)]
        struct Overhinted(core::ops::Range<i32>);
        impl Iterator for Overhinted {
            type Item = i32;
            fn next(&mut self) -> Option<i32> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(10_000))
            }
        }

        // Start with one element already in the deque.
        let alloc = ByteCapAlloc::new(64);
        let mut dq: VecDeque<i32, _> = VecDeque::new_in(alloc.clone());
        dq.try_push_back(-1).unwrap();

        dq.try_extend(Overhinted(0..3))
            .expect("incremental growth should succeed");
        assert_eq!(dq.len(), 4);
        assert_eq!(dq.front(), Some(&-1));
        assert_eq!(dq.back(), Some(&2));
    }

    #[test]
    fn try_extend_underhint_grows_mid_iteration() {
        // An iterator whose size hint severely underestimates the true count
        // forces mid-extension growth once the initially reserved capacity is
        // exhausted.
        #[derive(Debug)]
        struct Underhinted(core::ops::Range<i32>);
        impl Iterator for Underhinted {
            type Item = i32;
            fn next(&mut self) -> Option<i32> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(1))
            }
        }

        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(99).unwrap();
        dq.try_extend(Underhinted(0..8)).expect("extend ok");
        assert_eq!(dq.len(), 9);
        assert_eq!(dq.front(), Some(&99));
        assert_eq!(dq.back(), Some(&7));
    }

    // --- TryExtendFromSlice ------------------------------------------------------

    #[test]
    fn try_extend_from_slice_appends_clones() {
        // A type whose `TryClone` succeeds and records call count via a shared
        // counter, proving the clone path is exercised rather than a plain copy.
        use crate::test_helpers::CloneCounter;

        struct Cloned(i32, Arc<CloneCounter>);
        impl TryClone for Cloned {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                self.1.record_clone();
                Ok(Cloned(self.0, self.1.clone()))
            }
        }
        impl fmt::Debug for Cloned {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "Cloned({})", self.0)
            }
        }
        impl PartialEq for Cloned {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        let counter = CloneCounter::shared();
        let mut dq: VecDeque<Cloned> = VecDeque::new();
        let src = [
            Cloned(1, counter.clone()),
            Cloned(2, counter.clone()),
            Cloned(3, counter.clone()),
        ];
        dq.try_extend_from_slice(&src).expect("extend ok");
        assert_eq!(dq.len(), 3);
        assert_eq!(counter.get(), 3);
        assert_eq!(dq.get(0), Some(&Cloned(1, counter.clone())));
        assert_eq!(dq.get(2), Some(&Cloned(3, counter.clone())));
    }

    #[test]
    fn try_extend_from_slice_empty_is_noop() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_extend_from_slice(&[]).expect("extend ok");
        assert!(dq.is_empty());
    }

    #[test]
    fn try_extend_from_slice_reserve_failure_returns_whole_slice() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let src = [1, 2, 3];
        let (rest, err) = match dq.try_extend_from_slice(&src) {
            Ok(_) => panic!("expected allocation failure"),
            Err(pair) => pair,
        };
        assert!(matches!(err, TryVecDequeWithCloneError::Reserve(_)));
        // On a reserve failure nothing was consumed, so the whole slice comes
        // back.
        assert_eq!(rest, &src[..]);
        assert!(dq.is_empty());
    }

    #[test]
    fn try_extend_from_slice_clone_failure_returns_residual() {
        // Three source elements where the third one fails to clone. The first
        // two are committed to the deque; the residual slice begins at index 2
        // (the failing element).
        let mut dq: VecDeque<FlakyClone> = VecDeque::new();
        let src = [
            FlakyClone::new(2),           // count=0 → clones to count=1 ✓
            FlakyClone { count: 1, threshold: 2 }, // clones to count=2 ✓
            FlakyClone { count: 2, threshold: 2 }, // count >= threshold → ✗
        ];
        let (rest, err) = match dq.try_extend_from_slice(&src) {
            Ok(_) => panic!("expected clone failure"),
            Err(pair) => pair,
        };
        assert!(matches!(err, TryVecDequeWithCloneError::Clone(TryCloneError::Other(_))));
        // Residual starts at the first failed element (index 2).
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].count, 2);
        // The two successful clones were committed.
        assert_eq!(dq.len(), 2);
    }

    // --- try_extend_front ----------------------------------------------------

    #[test]
    fn try_extend_front_prepends_each_item_as_it_arrives() {
        // Each item is prepended as it arrives (nightly std semantics), so the
        // last-produced item ends up at the very front.
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(100).unwrap();
        dq.try_extend_front(1..=3).expect("extend front ok");
        assert_eq!(dq.len(), 4);
        assert_eq!(dq.front(), Some(&3));
        assert_eq!(dq.back(), Some(&100));
        assert_eq!(dq.get(1), Some(&2));
        assert_eq!(dq.get(2), Some(&1));
    }

    #[test]
    fn try_extend_front_empty_source_is_noop() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_extend_front(std::iter::empty()).expect("extend front ok");
        assert_eq!(dq.len(), 1);
        assert_eq!(dq.front(), Some(&1));
    }

    #[test]
    fn try_extend_front_retry_via_resume_preserves_items() {
        // Feed a Resume directly to verify the retry contract: the stranded
        // head is prepended first, then each remainder item as it arrives.
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(0).unwrap();
        let resume = Resume::new(10, 11..13);
        dq.try_extend_front(resume).expect("extend front ok");
        // Head 10 pushed first, then 11, 12 each prepended on top of it. The
        // remainder is 11..13 (two items), so final order: [12, 11, 10, 0].
        assert_eq!(dq.len(), 4);
        assert_eq!(dq.get(0), Some(&12));
        assert_eq!(dq.get(1), Some(&11));
        assert_eq!(dq.get(2), Some(&10));
        assert_eq!(dq.get(3), Some(&0));
    }

    #[test]
    fn try_extend_front_oom_strands_first_consumed_element() {
        // A failing allocator cannot fund any growth. The first item (1) is
        // attempted first and its reserve fails immediately. Nothing was
        // committed; the stranded element is 1 and the remainder holds [2, 3].
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let (resume, err) = match dq.try_extend_front(1..=3) {
            Ok(_) => panic!("expected allocation failure"),
            Err(pair) => pair,
        };
        assert!(err.is_alloc());
        assert_eq!(*resume.head().expect("stranded head present"), 1);
        let collected: std::vec::Vec<i32> = resume.into_remainder().collect();
        assert_eq!(collected, [2, 3]);
        // The deque itself is unchanged.
        assert!(dq.is_empty());
    }

    #[test]
    fn try_extend_front_overhint_falls_back_to_incremental_growth() {
        use crate::test_helpers::allocators::ByteCapAlloc;

        #[derive(Debug)]
        struct Overhinted(core::ops::Range<i32>);
        impl Iterator for Overhinted {
            type Item = i32;
            fn next(&mut self) -> Option<i32> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(10_000))
            }
        }

        let alloc = ByteCapAlloc::new(64);
        let mut dq: VecDeque<i32, _> = VecDeque::new_in(alloc.clone());
        dq.try_push_back(-1).unwrap();

        dq.try_extend_front(Overhinted(0..3))
            .expect("incremental growth should succeed");
        assert_eq!(dq.len(), 4);
        // Last item (2) ends up at the front; -1 stays at the back.
        assert_eq!(dq.front(), Some(&2));
        assert_eq!(dq.back(), Some(&-1));
    }

    #[test]
    fn try_extend_front_underhint_grows_mid_iteration() {
        #[derive(Debug)]
        struct Underhinted(core::ops::Range<i32>);
        impl Iterator for Underhinted {
            type Item = i32;
            fn next(&mut self) -> Option<i32> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(1))
            }
        }

        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(99).unwrap();
        dq.try_extend_front(Underhinted(0..8)).expect("extend front ok");
        assert_eq!(dq.len(), 9);
        // Last item (7) ends up at the front; 99 stays at the back.
        assert_eq!(dq.front(), Some(&7));
        assert_eq!(dq.back(), Some(&99));
    }

    #[test]
    fn try_extend_front_wrapped_buffer_preserves_logical_order() {
        // Force a wrapped layout the same way as the Debug/Clone tests: fill
        // the buffer, pop two from the front, push one more back so the new
        // element wraps around to slot 0 ahead of `head`.
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        for v in [1, 2, 3, 4] {
            dq.try_push_back(v).unwrap();
        }
        dq.pop_front();
        dq.pop_front();
        dq.try_push_back(5).unwrap();
        assert!(!dq.is_contiguous());
        dq.try_extend_front([7, 8]).expect("extend front ok");
        assert_eq!(dq.len(), 5);
        // 7 prepended first, then 8 on top → 8 is the new front.
        assert_eq!(dq.front(), Some(&8));
        assert_eq!(dq.back(), Some(&5));
        assert_eq!(dq.get(1), Some(&7));
        assert_eq!(dq.get(2), Some(&3));
        assert_eq!(dq.get(3), Some(&4));
    }

    // --- try_extend_front_from_slice ------------------------------------------

    #[test]
    fn try_extend_front_from_slice_prepends_clones_as_they_arrive() {
        use crate::test_helpers::CloneCounter;

        struct Cloned(i32, Arc<CloneCounter>);
        impl TryClone for Cloned {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                self.1.record_clone();
                Ok(Cloned(self.0, self.1.clone()))
            }
        }
        impl fmt::Debug for Cloned {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "Cloned({})", self.0)
            }
        }
        impl PartialEq for Cloned {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        let counter = CloneCounter::shared();
        let mut dq: VecDeque<Cloned> = VecDeque::new();
        let src = [
            Cloned(1, counter.clone()),
            Cloned(2, counter.clone()),
            Cloned(3, counter.clone()),
        ];
        dq.try_extend_front_from_slice(&src).expect("extend front ok");
        assert_eq!(dq.len(), 3);
        assert_eq!(counter.get(), 3);
        // Each clone prepended as it arrives → last element (3) at the front.
        assert_eq!(dq.get(0), Some(&Cloned(3, counter.clone())));
        assert_eq!(dq.get(1), Some(&Cloned(2, counter.clone())));
        assert_eq!(dq.get(2), Some(&Cloned(1, counter.clone())));
    }

    #[test]
    fn try_extend_front_from_slice_empty_is_noop() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(5).unwrap();
        dq.try_extend_front_from_slice(&[]).expect("extend front ok");
        assert_eq!(dq.len(), 1);
        assert_eq!(dq.front(), Some(&5));
    }

    #[test]
    fn try_extend_front_from_slice_reserve_failure_returns_whole_slice() {
        let mut dq: VecDeque<i32, FailAlloc> = VecDeque::new_in(FailAlloc);
        let src = [1, 2, 3];
        let (rest, err) = match dq.try_extend_front_from_slice(&src) {
            Ok(_) => panic!("expected allocation failure"),
            Err(pair) => pair,
        };
        assert!(matches!(err, TryVecDequeWithCloneError::Reserve(_)));
        // On a reserve failure nothing was consumed, so the whole slice comes
        // back.
        assert_eq!(rest, &src[..]);
        assert!(dq.is_empty());
    }

    #[test]
    fn try_extend_front_from_slice_clone_failure_returns_tail_residual() {
        // Forward iteration: src[0] clones fine and is prepended, then src[1]
        // fails (count=1... wait, need to design carefully).
        // src[0]: count=0, threshold=2 → clones to count=1 ✓ (prepended)
        // src[1]: count=1, threshold=1 → count >= threshold → ✗
        // src[2]: never reached
        let mut dq: VecDeque<FlakyClone> = VecDeque::new();
        let src = [
            FlakyClone { count: 0, threshold: 2 },   // ✓ clones to count=1
            FlakyClone { count: 1, threshold: 1 },   // ✗ count >= threshold
            FlakyClone { count: 0, threshold: 5 },   // not reached
        ];
        let (rest, err) = match dq.try_extend_front_from_slice(&src) {
            Ok(_) => panic!("expected clone failure"),
            Err(pair) => pair,
        };
        assert!(matches!(err, TryVecDequeWithCloneError::Clone(TryCloneError::Other(_))));
        // Residual starts at the first failed element (index 1).
        assert_eq!(rest.len(), 2);
        assert_eq!(rest[0].count, 1);
        assert_eq!(rest[1].count, 0);
        // The one successful clone was committed at the front.
        assert_eq!(dq.len(), 1);
        assert_eq!(dq.get(0).unwrap().count, 1);
    }

    #[test]
    fn try_extend_front_from_slice_last_element_fails() {
        // All but the last element clone fine; the last one fails.
        // src[0]: count=0, threshold=3 → ✓ (count→1)
        // src[1]: count=0, threshold=3 → ✓ (count→1)
        // src[2]: count=3, threshold=3 → ✗
        let mut dq: VecDeque<FlakyClone> = VecDeque::new();
        let src = [
            FlakyClone { count: 0, threshold: 3 },   // ✓
            FlakyClone { count: 0, threshold: 3 },   // ✓
            FlakyClone { count: 3, threshold: 3 },   // ✗
        ];
        let (rest, err) = match dq.try_extend_front_from_slice(&src) {
            Ok(_) => panic!("expected clone failure"),
            Err(pair) => pair,
        };
        assert!(matches!(err, TryVecDequeWithCloneError::Clone(TryCloneError::Other(_))));
        // Residual is just the failing last element.
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].count, 3);
        // Two clones were committed, each prepended as it arrived:
        // src[0] pushed first, then src[1] on top → [clone(src[1]), clone(src[0])].
        assert_eq!(dq.len(), 2);
        assert_eq!(dq.get(0).unwrap().count, 1);
        assert_eq!(dq.get(1).unwrap().count, 1);
    }
}
