//! Iterator types for the [`Vec`].

use core::fmt;
use core::fmt::Debug;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::slice;

use crate::alloc::{Allocator, Global};
use crate::raw_vec::RawVec;
use crate::vec::Vec;
use olive_core::alloc::{AllocatorTryClone, AllocatorTryDefault};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

/// An iterator that moves out of a vector.
///
/// This `struct` is created by the [`into_iter`](super::Vec::into_iter) method
/// on [`Vec`](super::Vec) (provided by the [`IntoIterator`] trait).
///
/// # Example
///
/// ```
/// use olive_alloc::vec::Vec;
/// let v = {
///     let mut v = Vec::new();
///     for i in 0..3 { v.try_push(i).unwrap(); }
///     v
/// };
/// let collected: std::vec::Vec<i32> = v.into_iter().collect();
/// assert_eq!(collected, std::vec![0, 1, 2]);
/// ```
pub struct IntoIter<T, A: Allocator = Global> {
    /// Owns the backing allocation. Its `Drop` frees the block and drops the
    /// allocator. We read element pointers from it lazily (via `ptr()`) but
    /// never advance a stored head pointer.
    raw: RawVec<T, A>,
    /// Number of elements already yielded from the front (via `next`). Together
    /// with `taken_back` this partitions the buffer into three regions:
    /// `[consumed, len - taken_back)` is still live.
    consumed: usize,
    /// Number of elements already yielded from the back (via `next_back`).
    taken_back: usize,
    /// Total number of elements originally held (`len` at construction).
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: mirroring `Vec`, the iterator never hands out references that
// outlive the buffer, and moving it moves the whole allocation. Sound iff `T`
// and `A` are `Send`/`Sync`.
unsafe impl<T: Send, A: Allocator + Send> Send for IntoIter<T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for IntoIter<T, A> {}

impl<T, A: Allocator> IntoIter<T, A> {
    /// Constructs an iterator from the raw parts of a consumed `Vec`.
    ///
    /// # Safety
    ///
    /// `start` must be the base of a valid allocation of at least `cap`
    /// elements made with `alloc`, and the first `len` slots must be
    /// initialized. For zero-sized `T`, `start` may be a dangling aligned
    /// pointer and `cap` may be an arbitrary value (it is completely ignored).
    #[inline]
    pub(super) const unsafe fn new_from_parts(
        start: NonNull<T>,
        len: usize,
        cap: usize,
        alloc: A,
    ) -> Self {
        // SAFETY: preconditions passed through from the caller.
        Self {
            raw: unsafe { RawVec::from_nonnull_in(start, cap, alloc) },
            consumed: 0,
            taken_back: 0,
            len,
            _marker: PhantomData,
        }
    }

    /// Number of elements still held in the live window.
    #[inline]
    const fn remaining(&self) -> usize {
        // `consumed + taken_back` can never exceed `len`: each yield shrinks the
        // live window by exactly one and stops at zero.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "consumed + taken_back <= len"
        )]
        let taken = { self.consumed + self.taken_back };
        #[allow(clippy::arithmetic_side_effects, reason = "taken <= len")]
        {
            self.len - taken
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

    /// Views the remaining elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // The computed front pointer is invalid only when it lands one-past-the-
        // end of the buffer: `front_offset() == len`. When that case happens,
        // the fallback must be used. This fallback behavior is triggered
        // sparingly for conformance with std. (this also happens to work with ZSTs)
        if self.front_offset() == self.len {
            return unsafe { slice::from_raw_parts(self.raw.ptr(), 0) };
        }
        // SAFETY: `front_offset() < len`; the front of the live window addresses a
        // real, initialized slot (or a valid dangling-aligned ZST address),
        // and `[front, front + rem)` lies entirely within the initialized region
        // of the buffer.
        let rem = self.remaining();
        unsafe {
            let front = self.raw.ptr().add(self.front_offset());
            slice::from_raw_parts(front, rem)
        }
    }

    /// Element offset (from the buffer base) of the current front of the live
    /// window.
    #[inline]
    fn front_offset(&self) -> usize {
        self.consumed
    }

    /// Element offset (from the buffer base) just past the current back of the
    /// live window (the last live element is at `back_offset() - 1`).
    #[inline]
    fn back_offset(&self) -> usize {
        #[allow(clippy::arithmetic_side_effects, reason = "taken_back <= len")]
        {
            self.len - self.taken_back
        }
    }
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining() == 0 {
            return None;
        }
        // Record the front offset *before* advancing; it addresses a live,
        // initialized slot because we just confirmed `remaining > 0`.
        let offset = self.front_offset();
        #[allow(clippy::arithmetic_side_effects, reason = "consumed < len")]
        {
            self.consumed += 1;
        }
        // SAFETY: `offset` is within the initialized region and the value is now
        // ours to move out. Reading a ZST from its (possibly dangling-aligned)
        // pointer is well defined.
        Some(unsafe { ptr::read(self.raw.ptr().add(offset)) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining(), Some(self.remaining()))
    }
}

impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining()
    }
}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
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
            reason = "remaining > 0 implies back_offset >= 1"
        )]
        let offset = { self.back_offset() - 1 };
        #[allow(clippy::arithmetic_side_effects, reason = "taken_back < len")]
        {
            self.taken_back += 1;
        }
        // SAFETY: as in `next`, `offset` is within the initialized region and
        // the value becomes the caller's.
        Some(unsafe { ptr::read(self.raw.ptr().add(offset)) })
    }
}

impl<T, A: Allocator> Drop for IntoIter<T, A> {
    fn drop(&mut self) {
        // Destroy the unconsumed tail.
        // SAFETY: the addressed slots lie within the initialized region and hold
        // valid values; for ZSTs `drop_in_place` performs no memory access but
        // still runs any `Drop` glue. To avoid OOB overflow, only drop when rem > 0
        let rem = self.remaining();
        if rem > 0 {
            unsafe {
                let front = self.raw.ptr().add(self.consumed);
                ptr::drop_in_place(slice::from_raw_parts_mut(front, rem));
            }
        }
        // The owned `RawVec` field is dropped automatically after this body
        // returns, freeing the backing block and dropping the allocator. No
        // explicit free call or guard is needed: `RawVec::drop` handles both.
    }
}

impl<T: Debug, A: Allocator> Debug for IntoIter<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Match std: the remaining elements are wrapped in an `IntoIter(...)`
        // tuple, not printed as a bare list.
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

impl<T: TryClone, A: AllocatorTryClone> TryClone for IntoIter<T, A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let alloc = self.raw.allocator().try_clone()?;
        let mut out = Vec::<T, A>::try_with_capacity_in(self.as_slice().len(), alloc)?;
        for item in self.as_slice() {
            let cloned = item.try_clone()?;
            // SAFETY: capacity was reserved above for every element.
            unsafe { out.force_push(cloned) };
        }
        Ok(out.into_iter())
    }
}

impl<T> Default for IntoIter<T, Global> {
    #[inline]
    fn default() -> Self {
        Vec::new_in(Global).into_iter()
    }
}

impl<T, A: AllocatorTryDefault> TryDefault for IntoIter<T, A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Vec::new_in(A::try_default()?).into_iter())
    }
}

impl<T, A: Allocator> IntoIterator for Vec<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Take ownership of the raw parts so the iterator owns the allocation
        // directly rather than nesting a whole `Vec`. The `ManuallyDrop` inside
        // `into_parts_with_alloc` prevents the original `Vec` from freeing the
        // buffer; the iterator takes over responsibility for it.
        let (ptr, len, cap, alloc) = self.into_parts_with_allocator();
        // SAFETY: as above — the parts come straight off a valid `Vec`.
        unsafe { IntoIter::new_from_parts(ptr, len, cap, alloc) }
    }
}

impl<T, A: Allocator> Vec<T, A> {
    /// Returns an iterator over the vector's elements.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns an iterator over mutable references to the vector's elements.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}

// Borrowed iteration delegates to the slice iterators, mirroring `std`'s
// `IntoIterator` impls for `&Vec` and `&mut Vec`.
impl<'a, T, A: Allocator> IntoIterator for &'a Vec<T, A> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a mut Vec<T, A> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::vec::{IntoIter, Vec};
    use olive_core::try_traits::try_clone::TryClone;
    use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};
    use olive_test_helpers::{
        BudgetedAlloc, CloneBudget, Counter, DropCountingAlloc, FailDefaultAlloc, FlakyCloneAlloc,
        Ledger, TrackedItem,
    };
    use std::sync::Arc as StdArc;

    fn make_vec(alloc: BudgetedAlloc) -> Vec<u32, BudgetedAlloc> {
        let mut v = Vec::new_in(alloc);
        for x in [1, 2, 3, 4] {
            v.try_push(x).unwrap();
        }
        v
    }

    #[test]
    fn into_iter_yields_all() {
        let v = {
            let mut v = Vec::new();
            for i in 0..5 {
                v.try_push(i).unwrap();
            }
            v
        };
        let collected: std::vec::Vec<i32> = v.into_iter().collect();
        assert_eq!(collected, std::vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn into_iter_size_hint() {
        let v = {
            let mut v = Vec::new();
            for i in 0..5 {
                v.try_push(i).unwrap();
            }
            v
        };
        let mut it = v.into_iter();
        assert_eq!(it.size_hint(), (5, Some(5)));
        assert_eq!(it.len(), 5);
        let _ = it.next();
        assert_eq!(it.size_hint(), (4, Some(4)));
        assert_eq!(it.len(), 4);
    }

    #[test]
    fn into_iter_double_ended() {
        let v = {
            let mut v = Vec::new();
            for i in 0..5 {
                v.try_push(i).unwrap();
            }
            v
        };
        let mut it = v.into_iter();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next_back(), Some(4));
        assert_eq!(it.next(), Some(1));
        assert_eq!(it.next_back(), Some(3));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn into_iter_size_hint_double_ended() {
        let v = {
            let mut v = Vec::new();
            for i in 0..5 {
                v.try_push(i).unwrap();
            }
            v
        };
        let mut it = v.into_iter();
        assert_eq!(it.size_hint(), (5, Some(5)));
        assert_eq!(it.len(), 5);
        let _ = it.next();
        assert_eq!(it.size_hint(), (4, Some(4)));
        assert_eq!(it.len(), 4);
    }

    // Regression test for the stored-head-pointer OOB bug (see agents/BUGBOT.md):
    //
    // The redesigned iterator tracks position by integer offset,
    // so querying `as_slice`/`len` after exhaustion must be well defined and
    // return an empty slice rather than forming an OOB pointer.
    #[test]
    fn into_iter_exhausted_query_is_safe() {
        let v = {
            let mut v = Vec::new();
            for i in 0..4i32 {
                v.try_push(i).unwrap();
            }
            v
        };
        let mut it = v.into_iter();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), Some(1));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), None);
        // Querying the exhausted iterator must not form an OOB pointer.
        assert_eq!(it.len(), 0);
        assert!(it.is_empty());
        assert_eq!(it.as_slice(), &[] as &[i32]);
    }

    // Same concern reached from the rear: exhaust via `next_back` only, then query.
    // Guards `next_back`'s back-offset arithmetic against forming an OOB pointer.
    #[test]
    fn into_iter_exhausted_from_back_is_safe() {
        let v = {
            let mut v = Vec::new();
            for i in 0..3i32 {
                v.try_push(i).unwrap();
            }
            v
        };
        let mut it = v.into_iter();
        assert_eq!(it.next_back(), Some(2));
        assert_eq!(it.next_back(), Some(1));
        assert_eq!(it.next_back(), Some(0));
        assert_eq!(it.next_back(), None);
        assert_eq!(it.len(), 0);
        assert_eq!(it.as_slice(), &[] as &[i32]);
    }

    #[test]
    fn into_iter_zst_exhausted_is_safe() {
        let v = {
            let mut v: Vec<()> = Vec::new();
            for _ in 0..5 {
                v.try_push(()).unwrap();
            }
            v
        };
        let mut it = v.into_iter();
        // Five elements outstanding: all three views must agree on 5.
        assert_eq!(it.len(), 5);
        assert_eq!(it.size_hint(), (5, Some(5)));
        assert_eq!(it.as_slice().len(), 5);

        // Consume two from the front and one from the back: two remain. The slice
        // length must track down to 2, never snap to 0.
        assert_eq!(it.next(), Some(()));
        assert_eq!(it.next(), Some(()));
        assert_eq!(it.next_back(), Some(()));
        assert_eq!(it.len(), 2);
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert_eq!(it.as_slice().len(), 2);

        // Exhaustion: everything agrees on 0.
        assert_eq!(it.next(), Some(()));
        assert_eq!(it.next(), Some(()));
        assert_eq!(it.len(), 0);
        assert_eq!(it.as_slice(), &[] as &[()]);
        assert_eq!(it.as_slice().len(), 0);
    }

    /// Regression test: dropping an `IntoIter` partway through must drop only the
    /// not-yet-yielded chunk. Here we yield two of five, then drop the iterator;
    /// exactly three drops should occur (the unconsumed parts).
    #[test]
    fn into_iter_drop_mid_way_drops_only_tail() {
        let ledger = StdArc::new(Ledger::new());

        // Yield two elements, then drop the iterator while three remain.
        {
            let mut v: Vec<TrackedItem<()>> = Vec::new();
            for i in 0..5u32 {
                v.try_push(TrackedItem::with_id(&ledger, i, ())).unwrap();
            }
            let mut it = v.into_iter();
            let _ = it.next().unwrap(); // id 0
            let _ = it.next_back().unwrap(); // id 4
            // Dropping the iterator must destroy only the unconsumed segments (ids 1,2,3).
            drop(it);

            assert_eq!(ledger.live_ids(), [0, 4]);
            assert!(ledger.double_dropped().is_empty());
            assert_eq!(ledger.drop_count(1), 1);
            assert_eq!(ledger.drop_count(2), 1);
            assert_eq!(ledger.drop_count(3), 1);
        }

        // All five dropped exactly once: 3 (middle) + 2 (yielded). No leaks, no
        // double-frees.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    #[test]
    fn into_iter_drops_allocator_exactly_once() {
        let counter = StdArc::new(Counter::new());
        let alloc = DropCountingAlloc::new(counter.clone());
        let mut v: Vec<i32, DropCountingAlloc> = Vec::new_in(alloc);
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let mut it = v.into_iter();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), Some(1));
        // Drop with three elements still held; the owned RawVec (and hence the
        // allocator) must be released here.
        drop(it);

        // Exactly one allocator instance lived inside the iterator and was dropped
        // when the iterator was. No leak (count > 0), no double-free (count == 1).
        assert_eq!(counter.get(), 1);
    }

    /// Regression test: `Debug for IntoIter` must match std's format, which wraps
    /// the not-yet-yielded elements in an `IntoIter(...)` tuple rather than printing
    /// a bare list. Before the fix the impl printed `[0, 1, 2]`.
    #[test]
    fn into_iter_debug_matches_std_format() {
        let mut v = Vec::new();
        for i in 0..3 {
            v.try_push(i).unwrap();
        }
        let mut it = v.into_iter();
        assert_eq!(std::format!("{it:?}"), "IntoIter([0, 1, 2])");
        let _ = it.next();
        assert_eq!(std::format!("{it:?}"), "IntoIter([1, 2])");
        let _ = it.next_back();
        assert_eq!(std::format!("{it:?}"), "IntoIter([1])");
    }

    #[test]
    fn try_clone_preserves_remaining() {
        let mut it = make_vec(BudgetedAlloc::new(16)).into_iter();
        assert_eq!(it.next(), Some(1));
        let mut clone = it.try_clone().unwrap();
        assert_eq!(clone.next(), Some(2));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), Some(4));
        assert_eq!(it.next(), None);
        assert_eq!(clone.next(), Some(3));
        assert_eq!(clone.next(), Some(4));
        assert_eq!(clone.next(), None);
    }

    #[test]
    fn try_clone_fails_when_allocator_clone_fails() {
        let alloc = FlakyCloneAlloc::new(CloneBudget::new(0));
        let mut v = Vec::new_in(alloc);
        v.try_push(1).unwrap();
        let it = v.into_iter();
        assert!(it.try_clone().is_err());
    }

    #[test]
    fn try_clone_fails_when_budget_exhausted() {
        let alloc = BudgetedAlloc::new(16);
        let mut it = make_vec(alloc.clone()).into_iter();
        it.next();
        alloc.drain();
        // The remaining three elements cannot be re-allocated, so cloning the
        // owned iterator must fail.
        assert!(it.try_clone().is_err());
    }

    #[test]
    fn try_clone_neither_leaks_nor_double_drops() {
        let ledger = StdArc::new(Ledger::new());
        let mut v: Vec<TrackedItem<u32>, BudgetedAlloc> = Vec::new_in(BudgetedAlloc::new(16));
        for x in [10, 20, 30, 40] {
            v.try_push(TrackedItem::construct(&ledger, x)).unwrap();
        }

        let mut it = v.into_iter();
        // Yielding the front element drops it exactly once when the yielded
        // value is dropped.
        assert_eq!(it.next().unwrap().inner, 10);

        let mut clone = it.try_clone().unwrap();

        // Both iterators drain independently: each remaining element instance
        // must die exactly once. A clone that aliased the original buffer
        // would double-drop here; a clone that failed to take ownership would
        // leak.
        for _ in 0..3 {
            it.next();
        }
        assert!(it.next().is_none());
        for _ in 0..3 {
            clone.next();
        }
        assert!(clone.next().is_none());

        // 4 elements from the original, 3 cloned.
        assert_eq!(ledger.total_allocated(), 7);
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped ids: {:?}",
            ledger.double_dropped()
        );
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(
            ledger.all_dropped_once(0..ledger.total_allocated()),
            "drop counts: {:?}",
            ledger.drop_counts()
        );
    }

    #[test]
    fn try_clone_failure_drops_partially_cloned_elements() {
        let ledger = StdArc::new(Ledger::new());
        // Two successful element clones, then every further clone fails.
        let budget = CloneBudget::new(2);
        let mut v: Vec<TrackedItem<CloneBudget>, BudgetedAlloc> =
            Vec::new_in(BudgetedAlloc::new(16));
        for _ in 0..4 {
            v.try_push(TrackedItem::construct(&ledger, budget.share()))
                .unwrap();
        }

        let mut it = v.into_iter();
        let _ = it.next(); // the front element drops immediately

        // The clone loop succeeds for two of the three remaining elements and
        // then hits the exhausted budget. The partially built iterator must
        // drop what it already cloned on the way out.
        assert!(it.try_clone().is_err());

        // Exactly the three live originals remain.
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.live_ids(), [1, 2, 3], "only the originals live");

        // Draining the original cleans up the rest: every id minted during
        // the test died exactly once.
        for _ in 0..3 {
            it.next();
        }
        assert!(it.next().is_none());

        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped ids: {:?}",
            ledger.double_dropped()
        );
        assert!(
            ledger.all_dropped_once(0..ledger.total_allocated()),
            "drop counts: {:?}",
            ledger.drop_counts()
        );
    }

    #[test]
    fn default_and_try_default_are_empty() {
        let it: IntoIter<u32> = Default::default();
        assert_eq!(it.count(), 0);
        let it2: IntoIter<u32> = TryDefault::try_default().unwrap();
        assert_eq!(it2.count(), 0);
    }

    #[test]
    fn try_default_allocator_failure_propagates() {
        let e = <IntoIter<u32, FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }
}
