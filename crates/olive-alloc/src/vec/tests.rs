//! Tests for the `Vec` module.
//!
//! Kept in a separate file so the module root stays focused on the type and its
//! operations; see [`super`] for the main definitions.

extern crate std;

use core::ptr::NonNull;

use olive_core::try_traits::try_clone::TryCloneError;
use olive_core::try_traits::try_collect::TryCollect;
use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
use olive_core::try_traits::try_from_iterator::TryFromIterator;

use super::*;
use crate::alloc::{AllocError, Layout};
use std::format;

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

/// An allocator whose every allocation fails. Used to exercise OOM paths.
#[derive(Default)]
struct FailAlloc;

unsafe impl Allocator for FailAlloc {
    fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

/// A value whose clone succeeds until it has been cloned `fail_after` times,
/// then fails forever. Lets us simulate a mid-operation clone failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Flaky(u32);

impl TryClone for Flaky {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if self.0 >= 2 {
            Err(TryCloneError::Other("flaky"))
        } else {
            Ok(Flaky(self.0 + 1))
        }
    }
}

/// Atomic clone counter shared by all instances of [`CountingFlaky`]. Counts
/// down from a configured threshold; when it reaches zero, clones fail.
static CLONE_REMAINING: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// A unit value that counts every successful clone globally (via atomic).
/// Clones succeed while the remaining count is above zero; once exhausted, all
/// subsequent clones fail. This lets us simulate "the Nth clone in a batch
/// fails" without depending on per-instance state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CountingFlaky;

impl CountingFlaky {
    /// Sets the number of successful clones allowed before failure.
    fn set_remaining(n: u32) {
        CLONE_REMAINING.store(n, std::sync::atomic::Ordering::SeqCst);
    }
}

impl TryClone for CountingFlaky {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        use std::sync::atomic::Ordering;
        loop {
            let cur = CLONE_REMAINING.load(Ordering::SeqCst);
            if cur == 0 {
                return Err(TryCloneError::Other("counting flaky exhausted"));
            }
            match CLONE_REMAINING.compare_exchange_weak(
                cur,
                cur - 1,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(CountingFlaky),
                Err(_) => continue,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn new_is_empty() {
    let v: Vec<i32> = Vec::new();
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 0);
}

#[test]
fn default_matches_new() {
    let v: Vec<i32> = Default::default();
    assert!(v.is_empty());
}

#[test]
fn try_with_capacity_allocates_room() {
    let v = Vec::<i32>::try_with_capacity(10).expect("alloc ok");
    assert!(v.is_empty());
    assert!(v.capacity() >= 10);
}

#[test]
fn try_with_capacity_zero_is_fine() {
    let v = Vec::<i32>::try_with_capacity(0).expect("zero cap ok");
    assert!(v.is_empty());
}

#[test]
fn new_in_custom_allocator() {
    let v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
    assert!(v.is_empty());
    // No allocation happened yet, so this is still usable.
    assert_eq!(v.as_slice(), []);
}

#[test]
fn from_elem_clones_count_times() {
    let v = Vec::try_from_elem(&7i32, 4).expect("elem ok");
    assert_eq!(v.as_slice(), &[7, 7, 7, 7]);
}

#[test]
fn from_elem_zero_count_is_empty() {
    let v = Vec::try_from_elem(&7i32, 0).expect("empty ok");
    assert!(v.is_empty());
}

#[test]
fn from_slice_clones_elements() {
    let src = [1, 2, 3];
    let v = <Vec<i32>>::try_from(&src[..]).expect("slice ok");
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn try_from_facade_matches_infallible_clone() {
    use core::convert::TryFrom;
    let src: std::vec::Vec<u8> = std::vec![9, 8, 7];
    let cloned = <Vec<u8>>::try_from(src.as_slice()).unwrap();
    assert_eq!(cloned.as_slice(), &[9, 8, 7]);
}

#[test]
fn collect_uses_size_hint() {
    let v: Vec<i32> = (0..5).try_collect().expect("collect ok");
    assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4]);
}

#[test]
fn collect_grows_past_hint() {
    // An iterator that lies about its size hint (claims small, yields more).
    struct LyingIter {
        n: usize,
        yielded: usize,
    }
    impl Iterator for LyingIter {
        type Item = i32;
        fn next(&mut self) -> Option<i32> {
            if self.yielded < self.n {
                let v = self.yielded as i32;
                self.yielded += 1;
                Some(v)
            } else {
                None
            }
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, Some(0)) // Deliberately under-reports.
        }
    }
    let v: Vec<i32> = LyingIter { n: 10, yielded: 0 }
        .try_collect()
        .expect("grow ok");
    assert_eq!(
        v.as_slice(),
        std::vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].as_slice()
    );
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

#[test]
fn push_pop_roundtrip() {
    let mut v = Vec::new();
    v.try_push(1).expect("push ok");
    v.try_push(2).expect("push ok");
    assert_eq!(v.as_slice(), &[1, 2]);
    assert_eq!(v.pop(), Some(2));
    assert_eq!(v.pop(), Some(1));
    assert_eq!(v.pop(), None);
}

#[test]
fn push_grows_amortized() {
    let mut v = Vec::new();
    for i in 0..100 {
        v.try_push(i).expect("push ok");
    }
    assert_eq!(v.len(), 100);
    assert!(v.capacity() >= 100);
    assert_eq!(v.as_slice()[99], 99);
}

#[test]
fn try_push_mut_returns_reference_to_last() {
    let mut v = Vec::new();
    let r = v.try_push_mut(42i32).unwrap();
    *r += 8;
    assert_eq!(v.as_slice(), &[50]);

    // Push again; the reference from the first call is invalidated by borrow
    // checker (NLL ends it), so we can safely get a new one.
    let r2 = v.try_push_mut(100i32).unwrap();
    *r2 *= 2;
    assert_eq!(v.as_slice(), &[50, 200]);
}

#[test]
fn try_push_mut_fails_on_oom() {
    use crate::vec::tests::FailAlloc;
    let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
    // First push needs an allocation, which always fails.
    let err = v.try_push_mut(42).expect_err("should fail");
    assert!(err.is_alloc());
    assert!(v.is_empty());
}

#[test]
fn try_push_mut_give_back_returns_reference_on_success() {
    let mut v = Vec::new();
    let r = v.try_push_mut_give_back(10i32).unwrap();
    *r += 5;
    assert_eq!(v.as_slice(), &[15]);
}

#[test]
fn try_push_mut_give_back_returns_value_on_oom() {
    use crate::vec::tests::FailAlloc;
    let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
    let (returned, err) = v.try_push_mut_give_back(99).expect_err("should fail");
    assert_eq!(returned, 99);
    assert!(err.is_alloc());
    assert!(v.is_empty());
}

#[test]
fn insert_at_front_middle_end() {
    let mut v = Vec::new();
    v.try_push(1).unwrap();
    v.try_push(3).unwrap();
    v.try_insert(0, 0).unwrap();
    v.try_insert(2, 2).unwrap();
    v.try_insert(4, 4).unwrap();
    assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4]);
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

// ---------------------------------------------------------------------------
// Resize / extend-from-slice
// ---------------------------------------------------------------------------

#[test]
fn resize_grows_by_cloning() {
    let mut v = Vec::new();
    v.try_push(1).unwrap();
    v.try_resize(4, &9).expect("resize ok");
    assert_eq!(v.as_slice(), &[1, 9, 9, 9]);
}

#[test]
fn resize_shrinks() {
    let mut v = Vec::new();
    for i in 0..5 {
        v.try_push(i).unwrap();
    }
    v.try_resize(2, &0).expect("shrink ok");
    assert_eq!(v.as_slice(), &[0, 1]);
}

#[test]
fn resize_with_calls_closure() {
    let mut v = Vec::new();
    let mut counter = 0u32;
    v.try_resize_with(3, || -> Result<i32, ()> {
        counter += 1;
        Ok(counter as i32)
    })
    .expect("resize_with ok");
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn resize_with_closure_error_rolls_back() {
    let mut v = Vec::new();
    v.try_push(0).unwrap();
    let result: Result<(), TryVecWithClosureError<i32>> = v.try_resize_with(4, || Err(42));
    // The closure failed, so the vector must be rolled back to its original length.
    assert!(result.is_err());
    assert_eq!(v.as_slice(), &[0]);
}

#[test]
fn extend_from_slice_appends_clones() {
    let mut v = Vec::new();
    v.try_push(1).unwrap();
    v.try_extend_from_slice_with_rollback(&[2, 3])
        .expect("extend ok");
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn append_moves_all_elements() {
    let mut a = Vec::new();
    a.try_push(1).unwrap();
    let mut b = Vec::new();
    b.try_push(2).unwrap();
    b.try_push(3).unwrap();
    a.try_append(&mut b).expect("append ok");
    assert_eq!(a.as_slice(), &[1, 2, 3]);
    assert!(b.is_empty());
}

// ---------------------------------------------------------------------------
// Reserve / shrink
// ---------------------------------------------------------------------------

#[test]
fn shrink_to_fit_reduces_capacity() {
    let mut v = Vec::<i32>::try_with_capacity(100).expect("alloc ok");
    for i in 0..4 {
        v.try_push(i).unwrap();
    }
    let before = v.capacity();
    v.try_shrink_to_fit().expect("shrink ok");
    assert!(v.capacity() <= before);
    assert!(v.capacity() >= 4);
    assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
}

// ---------------------------------------------------------------------------
// Give-back variants
// ---------------------------------------------------------------------------

#[test]
fn push_give_back_returns_value_on_oom() {
    let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
    // First push needs an allocation, which always fails.
    let (returned, err) = v.try_push_give_back(42).expect_err("should fail");
    assert_eq!(returned, 42);
    assert!(err.is_alloc());
    assert!(v.is_empty());
}

#[test]
fn insert_give_back_out_of_bounds() {
    let mut v = Vec::new();
    v.try_push(1).unwrap();
    let (val, e) = v.try_insert_give_back(5, 99).expect_err("oob");
    assert_eq!(val, 99);
    assert!(matches!(e, TryVecInsertError::OutOfBounds));
    assert_eq!(v.as_slice(), &[1]);
}

#[test]
fn insert_give_back_oom_returns_value() {
    let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
    // Inserting at index 0 of an empty vec requires growth -> OOM.
    let (val, e) = v.try_insert_give_back(0, 7).expect_err("oom");
    assert_eq!(val, 7);
    assert!(matches!(e, TryVecInsertError::Reserve(_)));
    assert!(v.is_empty());
}

// ---------------------------------------------------------------------------
// Fallible constructors under OOM
// ---------------------------------------------------------------------------

#[test]
fn with_capacity_in_fail_alloc_reports_alloc() {
    let err: TryReserveError =
        Vec::<i32, FailAlloc>::try_with_capacity_in(8, FailAlloc).expect_err("should fail");
    assert!(err.is_alloc());
}

#[test]
fn with_capacity_overflow_detected() {
    let mut v: Vec<i32> = Vec::new();
    let err = v.try_reserve(usize::MAX).expect_err("should overflow");
    assert!(err.is_capacity_overflow());
}

#[test]
fn from_elem_in_fail_alloc_reports_reserve() {
    let e = Vec::try_from_elem_in(&1i32, 4, FailAlloc).expect_err("should fail");
    assert!(matches!(e, TryVecWithCloneError::Reserve(r) if r.is_alloc()));
}

#[test]
fn from_slice_in_fail_alloc_reports_reserve() {
    let e = Vec::try_from_slice_in(&[1, 2], FailAlloc).expect_err("should fail");
    assert!(matches!(e, TryVecWithCloneError::Reserve(_)));
}

#[test]
fn from_iter_in_fail_alloc_reports_reserve() {
    let e = Vec::<i32, FailAlloc>::try_from_iter_in(0..3, FailAlloc).expect_err("should fail");
    assert!(e.is_alloc());
}

// ---------------------------------------------------------------------------
// Clone-failure rollback
// ---------------------------------------------------------------------------

#[test]
fn resize_rollbacks_partial_on_clone_failure() {
    // Allow exactly 2 successful clones; the 3rd will fail.
    CountingFlaky::set_remaining(2);
    let mut v: Vec<CountingFlaky> = Vec::new();
    v.try_push(CountingFlaky).unwrap();
    // Growing from len 1 to len 4 requires 3 clones of the source value.
    // First 2 succeed, 3rd fails → rollback to original length.
    let e = v
        .try_resize(4, &CountingFlaky)
        .expect_err("clone should fail");
    assert!(matches!(e, TryVecWithCloneError::Clone(_)));
    // Rolled back to original length.
    assert_eq!(v.len(), 1);
    assert_eq!(v.as_slice()[0], CountingFlaky);
}

#[test]
fn extend_from_slice_rolls_back_on_clone_failure() {
    let mut v: Vec<Flaky> = Vec::new();
    v.try_push(Flaky(0)).unwrap();
    // Build a source slice whose third element fails to clone.
    let mut fv: Vec<Flaky> = Vec::new();
    fv.try_push(Flaky(0)).unwrap();
    fv.try_push(Flaky(1)).unwrap();
    fv.try_push(Flaky(2)).unwrap(); // This one will fail to clone.
    let e = v
        .try_extend_from_slice_with_rollback(fv.as_slice())
        .expect_err("clone fail");
    assert!(matches!(e, TryVecWithCloneError::Clone(_)));
    // Only the original element remains.
    assert_eq!(v.len(), 1);
    assert_eq!(v.as_slice()[0], Flaky(0));
}

// ---------------------------------------------------------------------------
// TryClone on Vec
// ---------------------------------------------------------------------------

#[test]
fn try_clone_vec_deep_copies() {
    let mut v: Vec<i32> = Vec::new();
    for i in 0..5 {
        v.try_push(i).unwrap();
    }
    let c = v.try_clone().expect("clone ok");
    assert_eq!(c.as_slice(), &[0, 1, 2, 3, 4]);
    // Independent buffer.
    assert_ne!(v.as_ptr(), c.as_ptr());
}

#[test]
fn try_clone_empty_vec() {
    let v: Vec<i32> = Vec::new();
    let c = v.try_clone().expect("clone ok");
    assert!(c.is_empty());
}

// ---------------------------------------------------------------------------
// TryExtend / TryExtendFromSlice traits
// ---------------------------------------------------------------------------

#[test]
fn try_extend_trait_success() {
    let mut v: Vec<i32> = Vec::new();
    v.try_extend(0..3).expect("extend ok");
    assert_eq!(v.as_slice(), &[0, 1, 2]);
}

#[test]
fn try_extend_trait_oom_carries_resume() {
    let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
    let (resume, err) = v.try_extend(0..3).expect_err("oom");
    assert!(err.is_alloc());
    // Nothing was committed.
    assert!(v.is_empty());
    // The resume carries the unconsumed remainder.
    let remaining: std::vec::Vec<i32> = resume.into_remainder().collect();
    assert!(!remaining.is_empty());
}

#[test]
fn try_extend_retry_recovers() {
    // Simulate: first attempt fails partway, retry with the resume completes.
    // We can't force a real mid-way OOM easily, so verify the resume round-trips
    // through a second call on a healthy allocator.
    let mut v: Vec<i32> = Vec::new();
    let start = Resume::from_remainder(0..4);
    v.try_extend(start).expect("retry ok");
    assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
}

#[test]
fn try_extend_from_slice_trait_success() {
    let mut v: Vec<i32> = Vec::new();
    v.try_extend_from_slice(&[7, 8]).expect("ok");
    assert_eq!(v.as_slice(), &[7, 8]);
}

#[test]
fn try_extend_from_slice_trait_returns_remainder_on_clone_fail() {
    let mut v: Vec<Flaky> = Vec::new();
    let mut src: Vec<Flaky> = Vec::new();
    src.try_push(Flaky(0)).unwrap();
    src.try_push(Flaky(1)).unwrap();
    src.try_push(Flaky(2)).unwrap(); // fails to clone
    let (rest, e) = v
        .try_extend_from_slice(src.as_slice())
        .expect_err("clone fail");
    assert!(matches!(e, TryCloneError::Other(_)));
    // Remainder begins at the failing element.
    assert_eq!(rest.len(), 1);
    // Nothing committed before the failure either (first two succeeded though).
    // Actually the first two DID commit; only the third failed. Verify len==2.
    assert_eq!(v.len(), 2);
}

// ---------------------------------------------------------------------------
// TryFromIterator
// ---------------------------------------------------------------------------

#[test]
fn try_from_iterator_collects() {
    let v: Vec<i32> = TryFromIterator::try_from_iter(0..4).expect("iter ok");
    assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
}

// ---------------------------------------------------------------------------
// IntoIterator
// ---------------------------------------------------------------------------

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
    let _ = it.next();
    assert_eq!(it.size_hint(), (4, Some(4)));
}

/// Regression test: dropping an `IntoIter` partway through must drop only the
/// not-yet-yielded tail. Before the fix, `next()` moved elements out with
/// `ptr::read` but left `vec.len` unchanged, so the iterator's `Drop` (via
/// `Vec`) re-dropped every already-yielded element — a double-free. Here we
/// yield two of five, then drop the iterator; exactly three drops should occur
/// (the unconsumed tail), and the two yielded values are dropped when their
/// bindings go out of scope at the end of the block.
#[test]
fn into_iter_drop_mid_way_drops_only_tail() {
    static DROPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    struct CountDrop;
    impl Drop for CountDrop {
        fn drop(&mut self) {
            DROPPED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    DROPPED.store(0, std::sync::atomic::Ordering::SeqCst);

    // Yield two elements, then drop the iterator while three remain.
    {
        let mut v = Vec::new();
        for _ in 0..5 {
            v.try_push(CountDrop).unwrap();
        }
        let mut it = v.into_iter();
        let a = it.next().unwrap();
        let b = it.next().unwrap();
        drop(it); // drops the remaining 3
        drop(a); // +1
        drop(b); // +1
    }
    // 3 (tail) + 2 (yielded) = 5 total, each exactly once.
    assert_eq!(DROPPED.load(std::sync::atomic::Ordering::SeqCst), 5);
}

// Regression test for the stored-head-pointer OOB bug (see agents/BUGBOT.md):
// once an `IntoIter` is fully consumed, an advancing head pointer would sit at
// `base + len` — one-past-the-end, OOB relative to the allocation and a hard
// overflow on small pointer-width targets. The redesigned iterator tracks
// position by integer offset, so querying `as_slice`/`len` after exhaustion
// must be well defined and return an empty slice rather than forming an OOB
// pointer.
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

// Zero-sized types have a *dangling* base pointer; the old design advanced that
// pointer on every `next`, which is UB even when never dereferenced. Position
// must be carried entirely by integers so no pointer arithmetic touches the
// dangling base. This also exercises `as_slice` returning an empty slice over
// the dangling base without faulting.
#[test]
fn into_iter_zst_exhausted_is_safe() {
    let v = {
        let mut v: Vec<()> = Vec::new();
        for _ in 0..4 {
            v.try_push(()).unwrap();
        }
        v
    };
    let mut it = v.into_iter();
    assert_eq!(it.next(), Some(()));
    assert_eq!(it.next(), Some(()));
    assert_eq!(it.next_back(), Some(()));
    assert_eq!(it.next_back(), Some(()));
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
    assert_eq!(it.as_slice(), &[] as &[()]);
}

// A zero-sized type's buffer is dangling, so `as_slice` cannot point into real
// storage. It therefore returns a slice over the dangling base — but that
// slice's *length* must still agree with the iterator's own accounting
// (`len()` / `size_hint().0`). std's `IntoIter::as_slice` returns
// `from_raw_parts(ptr, self.len())`, i.e. the remaining count, unconditionally;
// collapsing the length to 0 for ZSTs would desynchronize the three views of
// "how many elements are left". This locks in that consistency while elements
// are still outstanding (not merely after exhaustion).
#[test]
fn into_iter_zst_as_slice_length_matches_len() {
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
    assert_eq!(it.as_slice().len(), 0);
}

// The iterator owns its backing allocation (and thus its allocator) directly.
// Dropping an unconsumed `IntoIter` must therefore drop the allocator exactly
// once — neither leaking it nor double-dropping it. We wrap `Global` in a
// counting allocator whose `Drop` records how many times it was destroyed,
// then assert the count is exactly one after the iterator goes out of scope.
#[test]
fn into_iter_drops_allocator_exactly_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ALLOC_DROPS: AtomicUsize = AtomicUsize::new(0);

    /// A pass-through allocator that counts how many instances are dropped.
    /// Not `Copy`: it carries a `Drop` impl, so each clone is a distinct
    /// instance that must itself be destroyed (and counted).
    #[derive(Clone)]
    struct CountingAlloc;

    unsafe impl Allocator for CountingAlloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            Global.allocate(layout)
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    impl Drop for CountingAlloc {
        fn drop(&mut self) {
            ALLOC_DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    ALLOC_DROPS.store(0, Ordering::SeqCst);

    // Build a vec on the counting allocator, consume partway, then drop the
    // iterator with elements still outstanding.
    {
        let mut v: Vec<i32, CountingAlloc> = Vec::new_in(CountingAlloc);
        for i in 0..5i32 {
            v.try_push(i).unwrap();
        }
        let mut it = v.into_iter();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), Some(1));
        // Drop with three elements still held; the owned RawVec (and hence the
        // allocator) must be released here.
        drop(it);
    }

    // Exactly one allocator instance lived inside the iterator and was dropped
    // when the iterator was. No leak (count > 0), no double-free (count == 1).
    assert_eq!(ALLOC_DROPS.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// Deref / Index / Display-ish
// ---------------------------------------------------------------------------

#[test]
fn deref_as_slice() {
    let mut v = Vec::new();
    v.try_push(1).unwrap();
    v.try_push(2).unwrap();
    let s: &[i32] = &v;
    assert_eq!(s, &[1, 2]);
}

#[test]
fn index_get_and_set() {
    let mut v = Vec::new();
    for i in 0..3 {
        v.try_push(i).unwrap();
    }
    assert_eq!(v[1], 1);
    v[1] = 99;
    assert_eq!(v.as_slice(), &[0, 99, 2]);
}

#[test]
fn range_full_index() {
    let mut v = Vec::new();
    v.try_push(1).unwrap();
    v.try_push(2).unwrap();
    let all: &[i32] = &v[..];
    assert_eq!(all, &[1, 2]);
}

#[test]
fn debug_format_lists_elements() {
    let mut v = Vec::new();
    v.try_push(1).unwrap();
    v.try_push(2).unwrap();
    let s = format!("{v:?}");
    assert_eq!(s, "[1, 2]");
}

// ---------------------------------------------------------------------------
// into_boxed_slice / raw parts
// ---------------------------------------------------------------------------

#[test]
fn into_boxed_slice_preserves_contents() {
    let v = {
        let mut v: Vec<i32> = Vec::new();
        for i in 0..5 {
            v.try_push(i).unwrap();
        }
        v
    };
    let boxed = v.try_into_boxed_slice().expect("box ok");
    assert_eq!(boxed.as_ref(), [0, 1, 2, 3, 4]);
}

#[test]
fn into_boxed_slice_empty() {
    let v: Vec<i32> = Vec::new();
    let boxed = v.try_into_boxed_slice().expect("box ok");
    assert!(boxed.is_empty());
}

#[test]
fn into_boxed_slice_give_back_on_oom() {
    let v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
    // Empty vec: shrink-to-fit on an empty buffer shouldn't need to allocate,
    // but FailAlloc makes any realloc fail. Either way, no data is lost.
    match v.try_into_boxed_slice_give_back() {
        Ok(boxed) => assert!(boxed.is_empty()),
        Err((vec_back, e)) => {
            assert!(e.is_alloc());
            assert!(vec_back.is_empty());
        }
    }
}

// ---------------------------------------------------------------------------
// try_into_array / try_into_array_give_back
// ---------------------------------------------------------------------------

#[test]
fn try_into_array_exact_length() {
    let mut v = Vec::new();
    for i in 0..4u8 {
        v.try_push(i).unwrap();
    }
    let arr: Box<[u8; 4]> = v.try_into_array().expect("should succeed");
    assert_eq!(arr.as_ref(), &[0, 1, 2, 3]);
}

#[test]
fn try_into_array_wrong_length_too_few() {
    let mut v = Vec::new();
    v.try_push(1u8).unwrap();
    v.try_push(2).unwrap();
    let err = v.try_into_array::<4>().unwrap_err();
    match err {
        TryVecIntoArrayError::LengthMismatch { expected, actual } => {
            assert_eq!(expected, 4);
            assert_eq!(actual, 2);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn try_into_array_wrong_length_too_many() {
    let mut v = Vec::new();
    for i in 0..6u8 {
        v.try_push(i).unwrap();
    }
    let err = v.try_into_array::<3>().unwrap_err();
    match err {
        TryVecIntoArrayError::LengthMismatch { expected, actual } => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 6);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn try_into_array_zero_size() {
    let v: Vec<u32> = Vec::new();
    let arr: Box<[u32; 0]> = v.try_into_array().expect("empty array ok");
    assert_eq!(arr.len(), 0);
}

#[test]
fn try_into_array_with_excess_capacity() {
    // Push enough elements to trigger growth, then pop down to exactly N.
    let mut v = Vec::new();
    for i in 0..10u8 {
        v.try_push(i).unwrap();
    }
    // Pop back to 3 — capacity stays high.
    while v.len() > 3 {
        v.pop();
    }
    assert!(v.capacity() > 3);
    let arr: Box<[u8; 3]> = v.try_into_array().expect("shrink + convert");
    assert_eq!(arr.as_ref(), &[0, 1, 2]);
}

#[test]
fn try_into_array_give_back_length_mismatch() {
    let mut v = Vec::new();
    v.try_push(99u8).unwrap();
    let (vec_back, err) = v.try_into_array_give_back::<5>().unwrap_err();
    assert_eq!(vec_back.len(), 1);
    assert_eq!(vec_back[0], 99);
    match err {
        TryVecIntoArrayError::LengthMismatch { expected, actual } => {
            assert_eq!(expected, 5);
            assert_eq!(actual, 1);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn try_into_array_give_back_success() {
    let mut v = Vec::new();
    for i in ['a', 'b', 'c'] {
        v.try_push(i).unwrap();
    }
    let arr: Box<[char; 3]> = v.try_into_array_give_back().expect("ok");
    assert_eq!(arr.as_ref(), &['a', 'b', 'c']);
}

#[test]
fn try_into_array_error_display() {
    let e = TryVecIntoArrayError::LengthMismatch {
        expected: 4,
        actual: 2,
    };
    let msg = format!("{e}");
    assert!(msg.contains("[T; 4]"));
    assert!(msg.contains("2"));

    let e2 = TryVecIntoArrayError::Shrink(TryReserveError::new_capacity_overflow());
    let msg2 = format!("{e2}");
    assert!(msg2.contains("shrink"));
}

#[test]
fn into_raw_parts_roundtrip() {
    let v = {
        let mut v = Vec::new();
        for i in 0..4 {
            v.try_push(i * 10).unwrap();
        }
        v
    };
    let (base, len, cap) = v.into_raw_parts();
    assert_eq!(len, 4);
    assert!(cap >= 4);
    // Read back through the raw pointer.
    let mut recovered = Vec::new();
    for i in 0..len {
        recovered.try_push(unsafe { base.add(i).read() }).unwrap();
    }
    assert_eq!(recovered.as_slice(), &[0, 10, 20, 30]);
    // Reassemble into a Vec so its Drop frees the buffer with the right layout.
    // SAFETY: `base` was allocated by the global allocator for `cap` elements;
    // first `len` are initialized. This is exactly what `from_raw_parts`
    // reconstructs.
    let _rebuilt = unsafe { Vec::<i32>::from_raw_parts(base, len, cap) };
}

// ---------------------------------------------------------------------------
// Drop correctness
// ---------------------------------------------------------------------------

#[test]
fn drop_runs_each_element_once() {
    static DROPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    struct CountDrop;
    impl Drop for CountDrop {
        fn drop(&mut self) {
            DROPPED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    DROPPED.store(0, std::sync::atomic::Ordering::SeqCst);
    {
        let mut v = Vec::new();
        for _ in 0..5 {
            v.try_push(CountDrop).unwrap();
        }
        v.truncate(3); // drops 2
    } // drops remaining 3
    assert_eq!(DROPPED.load(std::sync::atomic::Ordering::SeqCst), 5);
}

/// Regression test: `dedup_by` must leave the buffer in a consistent state if
/// the predicate panics mid-loop. Before the drop-guard fix, a panic after some
/// duplicates had been dropped would leak the tail and/or double-free the
/// already-dropped slots when the Vec was unwound.
#[test]
fn dedup_by_panic_is_safe() {
    static DROPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    struct CountDrop(u8);
    impl Drop for CountDrop {
        fn drop(&mut self) {
            DROPPED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    DROPPED.store(0, std::sync::atomic::Ordering::SeqCst);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut v = Vec::new();
        for i in 0..6u8 {
            v.try_push(CountDrop(i)).unwrap();
        }
        // Panic on the comparison involving element index 4 (the 5th element),
        // which is mid-gap-fill so some duplicates have already been dropped.
        let mut calls = 0usize;
        v.dedup_by(|a, b| {
            calls += 1;
            if calls == 3 {
                panic!("forced panic mid-dedup");
            }
            a.0 == b.0
        });
    }));

    assert!(result.is_err(), "expected the predicate to panic");
    // Whatever subset survived, every surviving element must be dropped exactly
    // once when the Vec is unwound — no leaks, no double-frees. The exact count
    // depends on how far the loop got before the panic; we only require that it
    // is positive (at least one element remained) and <= 6 (no double-count).
    let dropped = DROPPED.load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        (1..=6).contains(&dropped),
        "unexpected drop count: {dropped}"
    );
}

/// Regression test: if a destructor in the truncated tail panics, `truncate`
/// must have already shrunk `len` before dropping, so unwinding the Vec cannot
/// drop (double-free) those elements again. Before the fix, `truncate` dropped
/// each element *then* decremented, so a panicking drop left the length still
/// counting the element and the unwind would free it twice.
#[test]
fn truncate_panicking_drop_is_safe() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    static PANIC_ONCE: AtomicBool = AtomicBool::new(true);

    /// Counts its drop; panics exactly once, the first time it runs.
    struct PanicDrop;
    impl Drop for PanicDrop {
        fn drop(&mut self) {
            DROPPED.fetch_add(1, Ordering::SeqCst);
            if PANIC_ONCE.swap(false, Ordering::SeqCst) {
                panic!("forced panic in drop");
            }
        }
    }

    DROPPED.store(0, Ordering::SeqCst);
    PANIC_ONCE.store(true, Ordering::SeqCst);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut v = Vec::new();
        for _ in 0..4 {
            v.try_push(PanicDrop).unwrap();
        }
        // Truncate off the last two; the first one dropped will panic.
        v.truncate(2);
    }));

    assert!(result.is_err(), "expected the drop to panic");
    // The two truncated elements were dropped during `truncate` (one panicked),
    // and the two surviving elements are dropped on unwind. Total must be
    // exactly 4 — no double-free of the panicked element, no leak.
    let dropped = DROPPED.load(Ordering::SeqCst);
    assert_eq!(dropped, 4, "expected exactly 4 drops, got {dropped}");
}

// ---------------------------------------------------------------------------
// Error display
// ---------------------------------------------------------------------------

#[test]
fn error_types_display() {
    let re = TryReserveError::new_capacity_overflow();
    assert!(format!("{re}").contains("overflow"));
    let al = TryReserveError::new_alloc(Layout::new::<u8>());
    assert!(format!("{al}").contains("allocation"));

    let ve = TryVecInsertError::OutOfBounds;
    assert!(format!("{ve}").contains("out of bounds"));

    let wce = TryVecWithCloneError::Clone(TryCloneError::Other("boom"));
    assert!(format!("{wce}").contains("boom"));

    let rme = TryVecRemoveError { index: 5, len: 3 };
    assert!(format!("{rme}").contains("5"));
    assert!(format!("{rme}").contains("3"));

    let pwce = TryPushWithinCapacityError { len: 4 };
    assert!(format!("{pwce}").contains("full"));
    assert!(format!("{pwce}").contains("4"));
}

// ---------------------------------------------------------------------------
// set_len / swap_remove / try_swap_remove
// ---------------------------------------------------------------------------

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
fn try_swap_remove_success() {
    let mut v = Vec::new();
    for i in 0..4i32 {
        v.try_push(i).unwrap();
    }
    let val = v.try_swap_remove(2).expect("in bounds");
    assert_eq!(val, 2);
    assert_eq!(v.len(), 3);
}

#[test]
fn try_swap_remove_out_of_bounds() {
    let mut v = Vec::new();
    v.try_push(1).unwrap();
    let err = v.try_swap_remove(5).unwrap_err();
    assert_eq!(err.index, 5);
    assert_eq!(err.len, 1);
}

// ---------------------------------------------------------------------------
// try_insert_mut / try_insert_mut_give_back
// ---------------------------------------------------------------------------

#[test]
fn try_insert_mut_returns_reference() {
    let mut v = Vec::new();
    v.try_push(1).unwrap();
    v.try_push(3).unwrap();
    {
        let slot: &mut i32 = v.try_insert_mut(1, 99).unwrap();
        *slot = 2;
    }
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn try_insert_mut_give_back_on_oom() {
    let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
    // Inserting at index 0 of an empty vec requires growth -> OOM.
    let (val, err) = v.try_insert_mut_give_back(0, 42).unwrap_err();
    assert_eq!(val, 42);
    match err {
        TryVecInsertError::Reserve(_) => {}
        other => panic!("expected Reserve, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// try_remove
// ---------------------------------------------------------------------------

#[test]
fn try_remove_success() {
    let mut v = Vec::new();
    for i in 0..4i32 {
        v.try_push(i).unwrap();
    }
    let val = v.try_remove(1).expect("in bounds");
    assert_eq!(val, 1);
    assert_eq!(v.as_slice(), &[0, 2, 3]);
}

#[test]
fn try_remove_out_of_bounds() {
    let mut v = Vec::new();
    v.try_push(7).unwrap();
    let err = v.try_remove(3).unwrap_err();
    assert_eq!(err.index, 3);
    assert_eq!(err.len, 1);
}

// ---------------------------------------------------------------------------
// retain_mut
// ---------------------------------------------------------------------------

#[test]
fn retain_mut_keeps_matching() {
    let mut v = Vec::new();
    for i in 0..6i32 {
        v.try_push(i).unwrap();
    }
    v.retain_mut(|x| *x % 2 == 0);
    assert_eq!(v.as_slice(), &[0, 2, 4]);
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

// ---------------------------------------------------------------------------
// dedup_by_key / dedup_by
// ---------------------------------------------------------------------------

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

#[test]
fn dedup_by_single_element() {
    let mut v = Vec::new();
    v.try_push(42).unwrap();
    v.dedup_by_key(|&mut x| x);
    assert_eq!(v.as_slice(), &[42]);
}

// ---------------------------------------------------------------------------
// try_push_within_capacity
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// pop_if
// ---------------------------------------------------------------------------

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

// ─── try_drain ────────────────────────────────────────────────────────────────

#[test]
fn drain_middle_range_full_collect() {
    let mut v = Vec::new();
    for i in 0..5i32 {
        v.try_push(i).unwrap();
    }
    let drained: std::vec::Vec<i32> = v.try_drain(1..3).unwrap().collect();
    assert_eq!(drained, std::vec![1, 2]);
    assert_eq!(v.as_slice(), &[0, 3, 4]);
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
fn drain_len_and_is_empty() {
    let mut v = Vec::new();
    for i in 0..5i32 {
        v.try_push(i).unwrap();
    }
    let mut d = v.try_drain(1..4).unwrap();
    assert_eq!(d.len(), 3);
    assert!(!d.is_empty());
    assert_eq!(d.next(), Some(1));
    assert_eq!(d.len(), 2);
    assert_eq!(d.next(), Some(2));
    assert_eq!(d.len(), 1);
    assert_eq!(d.next(), Some(3));
    assert_eq!(d.len(), 0);
    assert!(d.is_empty());
    assert_eq!(d.next(), None);
}

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

// Mirror of `into_iter_zst_as_slice_length_matches_len` for `Drain`. A ZST
// buffer is dangling, so `as_slice` returns a slice over the base pointer — but
// its length must still agree with `len()` / `size_hint().0`. std's `Drain`
// delegates to an inner `slice::Iter`, whose `as_slice().len()` always equals
// its remaining count; reporting a different length would desynchronize the
// three views while elements are still outstanding. (While the drainer holds
// the mutable borrow we cannot read `v.len()`, so consistency is asserted
// entirely through the drainer's own views.)
#[test]
fn drain_zst_as_slice_length_matches_len() {
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
    drop(d);
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

// Regression test for the stored-head-pointer OOB bug (see agents/BUGBOT.md):
// once a drain is fully consumed, an advancing head pointer would sit at
// `base + end`. When the drained range ends at the tail of the buffer that
// address is one-past-the-end — OOB relative to the allocation and a hard
// overflow on small pointer-width targets. The redesigned `Drain` tracks
// position by integer offset, so querying `as_slice`/`len` after exhaustion
// must be well defined and return an empty slice rather than forming an OOB
// pointer.
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

// An empty range sitting exactly at the tail (`start == len`) previously forced
// construction-time pointer arithmetic to `base.add(len)` — already OOB before
// any iteration. It must construct cleanly and leave the vec untouched.
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

// Same OOB-head concern but reached via the back: exhaust from the rear of a
// range that touches the tail, then query. Guards `next_back`'s offset math.
#[test]
fn drain_exhausted_from_back_at_tail_is_safe() {
    let mut v = Vec::new();
    for i in 0..6i32 {
        v.try_push(i).unwrap();
    }
    let mut d = v.try_drain(4..6).unwrap();
    assert_eq!(d.next_back(), Some(5));
    assert_eq!(d.next_back(), Some(4));
    assert_eq!(d.next_back(), None);
    assert_eq!(d.len(), 0);
    assert_eq!(d.as_slice(), &[] as &[i32]);
    drop(d);
    assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
}

// Regression test for the hole-bookkeeping fix: `try_drain` caps the vector's
// length to `start` at construction, so the drained range is a hole the vec no
// longer counts. If a partially-consumed Drain is forgotten mid-iteration, the
// vec must NOT try to drop the (partially uninitialized) hole — it only owns
// the prefix. The unconsumed hole and the suffix leak, matching std's
// documented behaviour for `mem::forget` on a `Drain`. Crucially there is no
// double-drop and no UB from dropping uninitialized memory.
#[test]
fn drain_forget_mid_iteration_leaks_hole_not_prefix() {
    static DROP_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    #[allow(dead_code)]
    struct Tracked(u32);
    impl Drop for Tracked {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    DROP_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut v: Vec<Tracked> = Vec::new();
    for i in 0..6u32 {
        v.try_push(Tracked(i)).unwrap();
    }
    // Drain [1..4): elements 1, 2, 3 live in the hole. Consume only element 1.
    // (The vec is capped to len == 1 at this point, but we can't observe that
    // while `d` holds the mutable borrow; the drop-count assertion below proves
    // the prefix is the only part the vec still owns.)
    let mut d = v.try_drain(1..4).unwrap();
    // Take exactly one element out of the range; forget the rest of the drainer.
    let taken = d.next().expect("first drain element");
    assert_eq!(taken.0, 1);
    core::mem::forget(d);
    // The vec still holds only the prefix (element 0). Dropping it drops
    // exactly one element. Element 1 was moved out (caller-owned). Elements
    // 2, 3 (unconsumed hole) and 4, 5 (suffix) were abandoned by both owners
    // and leak — the accepted cost of `mem::forget`.
    drop(v);
    drop(taken);
    // Total drops: 1 (prefix, from vec) + 1 (`taken`) = 2. Four elements leak.
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::Relaxed), 2);
}

// A fully-consumed drain has no unconsumed hole, so forgetting it after
// collecting still leaves the vec holding its prefix while the (already-shown)
// suffix leaks. Here we verify the *normal* path instead: a fully-collected
// drain compacts the vec correctly and drops every element exactly once.
#[test]
fn drain_fully_collected_compacts_and_drops_once() {
    static DROP_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    #[allow(dead_code)]
    struct Tracked(u32);
    impl Drop for Tracked {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    DROP_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut v: Vec<Tracked> = Vec::new();
    for i in 0..5u32 {
        v.try_push(Tracked(i)).unwrap();
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
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::Relaxed), 0);
    drop(collected);
    // 1, 2 drop now.
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::Relaxed), 2);
    drop(v);
    // 0, 3, 4 drop as well -> total 5. Every element dropped exactly once.
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::Relaxed), 5);
}

// Regression test for the drain compaction guard: step 1 of a drain's `Drop`
// destroys the unconsumed hole via `drop_in_place`, which runs `T` destructors
// and can therefore panic. If it panics mid-way, the remaining destructions are
// abandoned (≈ `mem::forget` on the rest of the hole), but the compaction —
// shifting the suffix left and restoring the length — must STILL run so the vec
// is not stranded at its capped `original_start` with an orphaned suffix. This
// verifies that by making a destructor panic and asserting the recovered vec's
// length and contents are coherent.
#[test]
fn drain_panic_in_destructor_still_compacts() {
    static PANIC_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    #[allow(dead_code)]
    struct Panicky(u32);
    impl Drop for Panicky {
        fn drop(&mut self) {
            if PANIC_ARMED.load(std::sync::atomic::Ordering::SeqCst) && self.0 == 2 {
                panic!("boom");
            }
        }
    }
    let mut v: Vec<Panicky> = Vec::new();
    for i in 0..6u32 {
        v.try_push(Panicky(i)).unwrap();
    }
    // Drain [1..4): elements 1, 2, 3 live in the hole; consume only element 1
    // so 2 and 3 remain to be destroyed in step 1. Arm the panic before the
    // drainer drops.
    let mut d = v.try_drain(1..4).unwrap();
    let taken = d.next().expect("first drain element");
    assert_eq!(taken.0, 1);
    PANIC_ARMED.store(true, std::sync::atomic::Ordering::SeqCst);
    let panicked = std::panic::catch_unwind(|| drop(d));
    assert!(panicked.is_err(), "expected the destructor to panic");
    // The unwind ran the compaction guard: the suffix (4, 5) was shifted left
    // over the drained gap and the length restored past the prefix. So the vec
    // now holds exactly [0, 4, 5], not a stranded prefix of just [0].
    assert_eq!(v.len(), 3);
    assert_eq!(v.as_slice()[0].0, 0);
    assert_eq!(v.as_slice()[1].0, 4);
    assert_eq!(v.as_slice()[2].0, 5);
    // Element 2's destructor fired (and panicked); 3's was abandoned by the
    // unwind (leaked, matching `mem::forget`). Disarm before dropping the rest
    // so the catch above isn't re-triggered.
    PANIC_ARMED.store(false, std::sync::atomic::Ordering::SeqCst);
    drop(taken);
    drop(v);
}

// ─── try_split_off ────────────────────────────────────────────────────────

#[test]
fn split_off_middle() {
    let mut v = Vec::new();
    for i in 0..5i32 {
        v.try_push(i).unwrap();
    }
    let right = v.try_split_off(2).unwrap();
    assert_eq!(v.as_slice(), &[0, 1]);
    assert_eq!(right.as_slice(), &[2, 3, 4]);
}

#[test]
fn split_off_at_zero() {
    let mut v = Vec::new();
    for i in 0..3i32 {
        v.try_push(i).unwrap();
    }
    let right = v.try_split_off(0).unwrap();
    assert!(v.is_empty());
    assert_eq!(right.as_slice(), &[0, 1, 2]);
}

#[test]
fn split_off_at_len_returns_empty() {
    let mut v = Vec::new();
    for i in 0..3i32 {
        v.try_push(i).unwrap();
    }
    let right = v.try_split_off(3).unwrap();
    assert_eq!(v.as_slice(), &[0, 1, 2]);
    assert!(right.is_empty());
}

#[test]
fn split_off_on_empty_vec() {
    let mut v: Vec<i32> = Vec::new();
    let right = v.try_split_off(0).unwrap();
    assert!(v.is_empty());
    assert!(right.is_empty());
}

#[test]
fn split_off_out_of_bounds_errors() {
    let mut v = Vec::new();
    for i in 0..3i32 {
        v.try_push(i).unwrap();
    }
    let err = v.try_split_off(4).unwrap_err();
    assert!(matches!(
        err,
        TryVecSplitOffError::OutOfBounds { index: 4, len: 3 }
    ));
    // vec unchanged on error
    assert_eq!(v.as_slice(), &[0, 1, 2]);
}

#[test]
fn split_off_moves_all_elements_exactly_once() {
    // Use a type whose Drop can be counted to verify no double-free or leak.
    static DROP_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    #[allow(dead_code)]
    struct Tracked(u32);
    impl Drop for Tracked {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    DROP_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut v: Vec<Tracked> = Vec::new();
    for i in 0..6u32 {
        v.try_push(Tracked(i)).unwrap();
    }
    let right = v.try_split_off(3).unwrap();
    assert_eq!(v.len(), 3);
    assert_eq!(right.len(), 3);
    // No drops yet — all 6 elements are alive across both vectors.
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::Relaxed), 0);
    drop(v);
    drop(right);
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::Relaxed), 6);
}

#[test]
fn split_off_zst() {
    #[derive(Debug, PartialEq)]
    struct Unit;
    let mut v: Vec<Unit> = Vec::new();
    for _ in 0..3 {
        v.try_push(Unit).unwrap();
    }
    let right = v.try_split_off(1).unwrap();
    assert_eq!(v.len(), 1);
    assert_eq!(right.len(), 2);
}

// ─── try_extend_from_within ──────────────────────────────────────────────

#[test]
fn extend_from_within_basic() {
    let mut v = Vec::new();
    for i in [1, 2, 3, 4, 5] {
        v.try_push(i).unwrap();
    }
    v.try_extend_from_within(1..3).unwrap();
    // Appended clones of elements at indices 1 and 2 (values 2, 3)
    assert_eq!(v.as_slice(), &[1, 2, 3, 4, 5, 2, 3]);
}

#[test]
fn extend_from_within_entire_vec() {
    let mut v = Vec::new();
    for i in [10, 20, 30] {
        v.try_push(i).unwrap();
    }
    v.try_extend_from_within(..).unwrap();
    assert_eq!(v.as_slice(), &[10, 20, 30, 10, 20, 30]);
}

#[test]
fn extend_from_within_empty_range_is_noop() {
    let mut v = Vec::new();
    for i in [1, 2, 3] {
        v.try_push(i).unwrap();
    }
    v.try_extend_from_within(1..1).unwrap();
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn extend_from_within_out_of_bounds_errors() {
    let mut v = Vec::new();
    for i in [1, 2, 3] {
        v.try_push(i).unwrap();
    }
    let err = v.try_extend_from_within(1..5).unwrap_err();
    assert!(matches!(
        err,
        TryVecExtendFromWithinError::InvalidRange { .. }
    ));
    // vec unchanged on error
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn extend_from_within_reversed_range_errors() {
    let mut v = Vec::new();
    for i in [1, 2, 3] {
        v.try_push(i).unwrap();
    }
    // Reversed range must be rejected (lint suppressed: this is the point of the test).
    #[allow(
        clippy::reversed_empty_ranges,
        reason = "intentionally testing a reversed range"
    )]
    let err = v.try_extend_from_within(2..1).unwrap_err();
    assert!(matches!(
        err,
        TryVecExtendFromWithinError::InvalidRange { .. }
    ));
}

#[test]
fn extend_from_within_with_open_ranges() {
    let mut v = Vec::new();
    for i in [0, 1, 2, 3, 4] {
        v.try_push(i).unwrap();
    }
    // `..2` means 0..2 → appends [0, 1]
    v.try_extend_from_within(..2).unwrap();
    assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4, 0, 1]);
    // Fresh vec for second assertion to avoid confusion with grown length.
    let mut w = Vec::new();
    for i in [0, 1, 2, 3, 4] {
        w.try_push(i).unwrap();
    }
    // `3..` means 3..5 → appends [3, 4]
    w.try_extend_from_within(3..).unwrap();
    assert_eq!(w.as_slice(), &[0, 1, 2, 3, 4, 3, 4]);
}

#[test]
fn extend_from_within_grows_capacity() {
    let mut v: Vec<i32> = Vec::new();
    v.try_push(99).unwrap();
    // Extend with itself: [99] → [99, 99]
    v.try_extend_from_within(..).unwrap();
    assert_eq!(v.as_slice(), &[99, 99]);
    // Again: [99, 99] → [99, 99, 99, 99]
    v.try_extend_from_within(..).unwrap();
    assert_eq!(v.as_slice(), &[99, 99, 99, 99]);
}
