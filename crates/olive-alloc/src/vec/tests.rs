//! Tests for the `Vec` module.
//!
//! Kept in a separate file so the module root stays focused on the type and its
//! operations; see [`super`] for the main definitions.

extern crate std;

use core::alloc::Layout;

use olive_core::TryClone;
use olive_core::try_traits::try_clone::TryCloneError;
use olive_core::try_traits::try_collect::TryCollect;
use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
use olive_core::try_traits::try_from_iterator::TryFromIterator;

use super::*;
use crate::borrow::Cow;
use crate::test_helpers::{
    CloneBudget, DropCounter, DropCountingAlloc, FailAlloc, FlakyClone, FlakyTrackedItem, Ledger,
    PanicArmer,
};
use std::format;
use std::sync::Arc;

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
    let ledger = Arc::new(Ledger::new());
    // Budget of 2: the standalone clone below consumes 1, leaving 1 for the
    // resize loop — so the loop appends one element then fails on the next.
    let budget = Arc::new(CloneBudget::new(2));

    // Seed one live payload (id 0).
    let seed_id = ledger.allocate();
    ledger.register(seed_id);
    let mut v: Vec<FlakyTrackedItem> = Vec::new();
    v.try_push(FlakyTrackedItem {
        id: seed_id,
        ledger: ledger.clone(),
        inner: (*budget).share(),
    })
    .unwrap();

    // Clone the seed out so we can hand it to try_resize by reference without
    // fighting the &mut self borrow. This is a real clone (id 1) that lives in
    // `src` until end of scope.
    let src = v.as_slice()[0].try_clone().expect("seed clone ok");

    // Growing from len 1 to len 4 needs 3 more clones. With 1 unit of budget
    // left, the first append succeeds (id 2), the second fails → rollback drops
    // the single transient clone (id 2) and restores len == 1.
    let e = v.try_resize(4, &src).expect_err("clone should fail");
    assert!(matches!(e, TryVecWithCloneError::Clone(_)));

    // Rolled back to original length: only the seed survives in the vec.
    assert_eq!(v.len(), 1);
    // The lone transient clone (id 2) was destroyed during rollback. The only
    // ids still alive are 0 (seed, in v) and 1 (src, a local) — both expected.
    // No double-free, and exactly the transient clone has been dropped so far.
    assert_eq!(ledger.live_ids(), [0, 1]);
    assert!(ledger.double_dropped().is_empty());
    assert_eq!(
        ledger.drop_count(2),
        1,
        "the one appended clone must be dropped once"
    );
    assert_eq!(ledger.total_allocated(), 3);

    // Tear down: `src` (id 1) and the vec's seed (id 0) drop → all three gone.
    drop(src);
    drop(v);
    assert!(ledger.leaked_ids().is_empty());
    assert!(ledger.all_dropped_once(0..3));
}

#[test]
fn extend_from_slice_rolls_back_on_clone_failure() {
    let ledger = Arc::new(Ledger::new());
    let budget = Arc::new(CloneBudget::new(2));

    // One live seed in the destination (id 0).
    let seed_id = ledger.allocate();
    ledger.register(seed_id);
    let mut v: Vec<FlakyTrackedItem> = Vec::new();
    v.try_push(FlakyTrackedItem {
        id: seed_id,
        ledger: ledger.clone(),
        inner: (*budget).share(),
    })
    .unwrap();

    // Source slice of three payloads (ids 1, 2, 3). Extending clones them in
    // order; the budget allows exactly two successful clones (of ids 1 and 2),
    // then the third clone (of id 3) fails → rollback discards the two appends.
    let mut fv: Vec<FlakyTrackedItem> = Vec::new();
    for _ in 0..3 {
        let id = ledger.allocate();
        ledger.register(id);
        fv.try_push(FlakyTrackedItem {
            id,
            ledger: ledger.clone(),
            inner: (*budget).share(),
        })
        .unwrap();
    }

    let e = v
        .try_extend_from_slice_with_rollback(fv.as_slice())
        .expect_err("clone fail");
    assert!(matches!(e, TryVecWithCloneError::Clone(_)));

    // Destination rolled back to just its seed.
    assert_eq!(v.len(), 1);
    // The two transient copies of source ids 1 and 2 were created as NEW ids
    // (4 and 5) and dropped during rollback. Still alive at this point: id 0
    // (seed, in v) and ids 1, 2, 3 (the sources, in fv). No double-free, and
    // exactly the two transients have been dropped.
    assert_eq!(ledger.live_ids(), [0, 1, 2, 3]);
    assert!(ledger.double_dropped().is_empty());
    assert_eq!(ledger.total_allocated(), 6);
    assert_eq!(
        ledger.drop_count(4),
        1,
        "transient copy of source[0] dropped once"
    );
    assert_eq!(
        ledger.drop_count(5),
        1,
        "transient copy of source[1] dropped once"
    );

    // Tear down both vecs: seed (0) + sources (1,2,3) all drop → 5 more, total 7.
    drop(fv);
    drop(v);
    assert!(ledger.leaked_ids().is_empty());
    assert!(ledger.all_dropped_once(0..6));
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

    let alloc = ByteCapAlloc::new(64);
    let mut v: Vec<i32, _> = Vec::new_in(alloc.clone());
    v.try_push(-1).unwrap();

    v.try_extend(Overhinted(0..3))
        .expect("incremental growth should succeed");
    assert_eq!(v.as_slice(), &[-1, 0, 1, 2]);
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

    let mut v: Vec<i32> = Vec::new();
    v.try_push(99).unwrap();
    v.try_extend(Underhinted(0..8)).expect("extend ok");
    assert_eq!(v.as_slice(), &[99, 0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn try_extend_from_slice_trait_success() {
    let mut v: Vec<i32> = Vec::new();
    v.try_extend_from_slice(&[7, 8]).expect("ok");
    assert_eq!(v.as_slice(), &[7, 8]);
}

#[test]
fn try_reserve_adaptive_falls_back_to_exact_when_amortized_refused() {
    use crate::test_helpers::allocators::ByteCapAlloc;
    let mut v: Vec<i32, ByteCapAlloc> =
        Vec::try_with_capacity_in(15, ByteCapAlloc::new(64)).expect("should create vector");
    v.try_extend(0..15).expect("capacity is sufficient");
    v.try_reserve_adaptive(1)
        .expect("exact reserve should succeed");
    assert!(v.capacity() >= 16);
}

#[test]
fn try_reserve_adaptive_succeeds_via_amortized_path_when_affordable() {
    // With a generous cap the first (amortized) attempt succeeds outright, so no
    // fallback is needed and the result still holds the requested room.
    let mut v: Vec<i32> = Vec::new();
    v.try_reserve_adaptive(16).expect("affordable reserve");
    assert!(v.capacity() >= 16);
}

#[test]
fn try_extend_from_slice_uses_adaptive_reserve_under_byte_cap() {
    use crate::test_helpers::allocators::ByteCapAlloc;
    let mut v: Vec<i32, ByteCapAlloc> =
        Vec::try_with_capacity_in(15, ByteCapAlloc::new(64)).expect("should create vector");
    v.try_extend(0..15).expect("capacity is sufficient");
    v.try_extend_from_slice(&[15])
        .expect("adaptive reserve should absorb the over-provisioned request");
    assert_eq!(v.len(), 16);
    assert_eq!(v.last(), Some(&15));
}

#[test]
fn try_extend_from_slice_trait_returns_remainder_on_clone_fail() {
    let mut v: Vec<FlakyClone> = Vec::new();
    let mut src: Vec<FlakyClone> = Vec::new();
    src.try_push(FlakyClone::new(2)).unwrap();
    src.try_push(FlakyClone {
        count: 1,
        threshold: 2,
    })
    .unwrap();
    src.try_push(FlakyClone {
        count: 2,
        threshold: 2,
    })
    .unwrap(); // fails to clone
    let (rest, e) = v
        .try_extend_from_slice(src.as_slice())
        .expect_err("clone fail");
    assert!(matches!(
        e,
        TryVecWithCloneError::Clone(TryCloneError::Other(_))
    ));
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
/// bindings go out of scope at the end of the block. The ledger records which
/// specific ids were dropped so we can prove no id is ever dropped twice.
#[test]
fn into_iter_drop_mid_way_drops_only_tail() {
    let ledger = Arc::new(Ledger::new());
    let budget = Arc::new(CloneBudget::new(u32::MAX));

    // Yield two elements, then drop the iterator while three remain.
    {
        let mut v: Vec<FlakyTrackedItem> = Vec::new();
        for i in 0..5u32 {
            ledger.register(i);
            v.try_push(FlakyTrackedItem {
                id: i,
                ledger: ledger.clone(),
                inner: (*budget).share(),
            })
            .unwrap();
        }
        let mut it = v.into_iter();
        let a = it.next().unwrap(); // id 0
        let b = it.next().unwrap(); // id 1
        // Dropping the iterator must destroy only the unconsumed tail (ids 2,3,4).
        drop(it);
        // At this point only ids 0 and 1 (held in a/b) remain alive.
        assert_eq!(ledger.live_ids(), [0, 1]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.drop_count(2), 1);
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        drop(a); // id 0
        drop(b); // id 1
    }
    // All five dropped exactly once: 3 (tail) + 2 (yielded). No leaks, no
    // double-frees.
    assert!(ledger.leaked_ids().is_empty());
    assert!(ledger.all_dropped_once(0..5));
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
    let counter = Arc::new(DropCounter::new());
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
    let ledger = Arc::new(Ledger::new());
    let budget = Arc::new(CloneBudget::new(u32::MAX));
    let mut v: Vec<FlakyTrackedItem> = Vec::new();
    for i in 0..5u32 {
        ledger.register(i);
        v.try_push(FlakyTrackedItem {
            id: i,
            ledger: ledger.clone(),
            inner: (*budget).share(),
        })
        .unwrap();
    }
    // truncate(3) destroys the tail (ids 3 and 4).
    v.truncate(3);
    assert_eq!(ledger.live_ids(), [0, 1, 2]);
    assert_eq!(ledger.drop_count(3), 1);
    assert_eq!(ledger.drop_count(4), 1);
    // Drops the remaining IDs
    drop(v);
    // All five dropped exactly once; no leaks, no double-frees.
    assert!(ledger.leaked_ids().is_empty());
    assert!(ledger.all_dropped_once(0..5));
}

/// Regression test: `dedup_by` must leave the buffer in a consistent state if
/// the predicate panics mid-loop. Before the drop-guard fix, a panic after some
/// duplicates had been dropped would leak the tail and/or double-free the
/// already-dropped slots when the Vec was unwound. The ledger records each
/// element's id so we can prove, regardless of where the panic lands, that every
/// element is dropped exactly once (duplicates destroyed during dedup plus the
/// survivors destroyed on unwind) with neither leak nor double-free.
#[test]
fn dedup_by_panic_is_safe() {
    let ledger = Arc::new(Ledger::new());
    let budget = Arc::new(CloneBudget::new(u32::MAX));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let l = ledger.clone();
        let b = budget.clone();
        move || {
            let mut v: Vec<FlakyTrackedItem> = Vec::new();
            for i in 0..6u8 {
                l.register(i as u32);
                v.try_push(FlakyTrackedItem {
                    id: i as u32,
                    ledger: l.clone(),
                    inner: (*b).share(),
                })
                .unwrap();
            }
            // Panic on the third predicate call, mid-gap-fill, so some duplicate
            // elements have already been destroyed by the time we unwind.
            let mut calls = 0usize;
            v.dedup_by(|a, b| {
                calls += 1;
                if calls == 3 {
                    panic!("forced panic mid-dedup");
                }
                a.id == b.id
            });
        }
    }));

    assert!(result.is_err(), "expected the predicate to panic");
    // After the catch, the partially-deduplicated Vec has been unwound. Every
    // one of the six elements must have been destroyed exactly once — whether
    // it died as a duplicate inside dedup or as a survivor on unwind. No leaks,
    // no double-frees.
    assert!(ledger.leaked_ids().is_empty());
    assert!(ledger.double_dropped().is_empty());
    assert!(ledger.all_dropped_once(0..6));
}

/// Regression test: if a destructor in the truncated tail panics, `truncate`
/// must have already shrunk `len` before dropping, so unwinding the Vec cannot
/// drop (double-free) those elements again. Before the fix, `truncate` dropped
/// each element *then* decremented, so a panicking drop left the length still
/// counting the element and the unwind would free it twice. The ledger tracks
/// each element's id so we can prove every one is destroyed exactly once even
/// though one destructor panics mid-truncation.
#[test]
fn truncate_panicking_drop_is_safe() {
    let ledger = Arc::new(Ledger::new());
    let armer = Arc::new(PanicArmer::new());

    /// Panics exactly once, the first time it runs while armed; records its id.
    struct PanicDrop {
        id: u32,
        ledger: Arc<Ledger>,
        armer: Arc<PanicArmer>,
    }
    impl Drop for PanicDrop {
        fn drop(&mut self) {
            self.ledger.unregister(self.id);
            if self.armer.is_armed() {
                self.armer.disarm();
                panic!("forced panic in drop");
            }
        }
    }

    armer.arm();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let l = ledger.clone();
        let a = armer.clone();
        move || {
            let mut v = Vec::new();
            for i in 0..4u32 {
                l.register(i);
                v.try_push(PanicDrop {
                    id: i,
                    ledger: l.clone(),
                    armer: a.clone(),
                })
                .unwrap();
            }
            // Truncate off the last two (ids 2 and 3); the first one dropped
            // (id 3) panics, which disarms the armer for any later drops.
            v.truncate(2);
        }
    }));

    assert!(result.is_err(), "expected the drop to panic");
    // The two truncated elements were dropped during `truncate` (id 3 panicked),
    // and the two survivors (ids 0 and 1) are dropped on unwind. Every element
    // destroyed exactly once — no double-free of the panicked element, no leak.
    assert!(ledger.leaked_ids().is_empty());
    assert!(ledger.double_dropped().is_empty());
    assert!(ledger.all_dropped_once(0..4));
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

/// Regression: a partially-consumed drain's `Drop` must destroy the hole at
/// `original_start + consumed`, not `original_start + (original_count - remaining)`.
///
/// The naive formula `start + (count - remaining)` derives the offset from
/// `remaining` alone, implicitly assuming the hole always starts at index 0
/// within the drained range. It ignores how many were consumed from the FRONT
/// (`consumed`) — equivalently, it assumes the right edge of the hole is always
/// at `start + count`, ignoring items taken from the RIGHT (`taken_back`).
/// Whenever `taken_back` is nonzero, the computed offset
/// lands outside the true hole `[start+consumed .. start+count-taken_back)`:
/// it drops an already-yielded (moved-out) element at the right and
/// skips a still-live one at the left.
///
/// This test exercises BOTH directions simultaneously: `next()` × 2 (front) and
/// `next_back()` × 1 (back), so any formula that mishandles either side fails.
/// Each payload records itself in a shared sink when dropped, letting us assert
/// exactly which elements the drainer destroyed versus handed to the caller.
#[test]
fn drain_interleaved_drop_destroys_correct_hole() {
    use std::cell::RefCell;
    use std::rc::Rc;

    // Single-threaded local sink. Every `Rec` holds an `Option<Rc<_>>` clone;
    // `disarm()` pulls that clone out so the destructor won't record the
    // element. Because the reference is simply moved (not forgotten), every
    // strong ref is eventually released by a normal drop — nothing leaks.
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
                // Call the *inherent* `RefCell::borrow_mut` by fully-qualified
                // path rather than method sugar. Sugar on `s.borrow_mut()` walks
                // the deref chain into `Vec<u32>` and hits an ambiguous
                // `BorrowMut` target (the blanket `BorrowMut<T> for T` vs. our
                // new `Vec: BorrowMut<[T]>`). Naming the concrete `&RefCell`
                // receiver picks the inherent method with no trait lookup.
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
    // element, payload 3. Nothing else in the drained range may be touched.
    // Same inherent-method trick as above: call `RefCell::borrow` by fully-
    // qualified path on the concrete `&RefCell` receiver to dodge the ambiguous
    // `core::borrow::Borrow` trait resolution.
    let recorded = RefCell::<std::vec::Vec<u32>>::borrow(&sink);
    assert_eq!(*recorded, std::vec![3u32]);
    // Compaction removed the whole range: only payloads 0 and 5 survive.
    assert_eq!(v.len(), 2);
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
    let budget = Arc::new(CloneBudget::new(u32::MAX));

    let mut v: Vec<FlakyTrackedItem> = Vec::new();
    for i in 0..6u32 {
        ledger.register(i);
        v.try_push(FlakyTrackedItem {
            id: i,
            ledger: ledger.clone(),
            inner: (*budget).share(),
        })
        .unwrap();
    }
    // Drain [1..4): elements 1, 2, 3 live in the hole. Pull one from each end:
    // element 1 via `next`, element 3 via `next_back`, leaving element 2 as the
    // sole unconsumed hole member. (The vec is capped to len == 1 here, but we
    // can't observe that while `d` borrows it; the ledger assertions below prove
    // the prefix is the only part the vec still owns.)
    let mut d = v.try_drain(1..4).unwrap();
    let front = d.next().expect("front drain element");
    assert_eq!(front.id, 1);
    let back = d.next_back().expect("back drain element");
    assert_eq!(back.id, 3);
    // Forget the rest of the drainer: its remaining hole (element 2) and the
    // suffix (4, 5) are abandoned by both owners.
    core::mem::forget(d);
    // Only the prefix (id 0) is owned by the vec now; ids 1 and 3 are caller-
    // owned locals. Nothing has dropped yet.
    assert!(ledger.drop_counts().is_empty());
    drop(v); // drops id 0 (the prefix) — and nothing else
    assert_eq!(ledger.live_ids(), [1, 2, 3, 4, 5]);
    assert_eq!(
        ledger.drop_count(0),
        1,
        "only the prefix may be dropped by the vec"
    );
    drop(front); // id 1
    drop(back); // id 3
    // Final tally: ids 0, 1, 3 dropped exactly once. Ids 2, 4, 5 leaked (the
    // accepted cost of mem::forget). No double-free anywhere.
    assert!(ledger.double_dropped().is_empty());
    assert_eq!(ledger.leaked_ids(), [2, 4, 5]);
    assert_eq!(ledger.drop_count(0), 1);
    assert_eq!(ledger.drop_count(1), 1);
    assert_eq!(ledger.drop_count(3), 1);
}

// A fully-consumed drain has no unconsumed hole, so forgetting it after
// collecting still leaves the vec holding its prefix while the (already-shown)
// suffix leaks. Here we verify the *normal* path instead: a fully-collected
// drain compacts the vec correctly and drops every element exactly once.
#[test]
fn drain_fully_collected_compacts_and_drops_once() {
    let counter = Arc::new(DropCounter::new());

    #[allow(dead_code)]
    struct Tracked(u32, Arc<DropCounter>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.1.record_drop();
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

// Regression test for the drain compaction guard: step 1 of a drain's `Drop`
// destroys the unconsumed hole with a single fat-slice `drop_in_place`, which
// runs every `T` destructor in the hole and can therefore panic. Because the
// compiler lowers a fat-slice drop to a per-element sequence inside ONE function
// body, the unwinder runs each *remaining* destructor as an unwind landing pad
// on that same frame before leaving it — so even when one destructor panics
// mid-hole, the other hole elements ARE still dropped (verified below by the
// drop counter). What the unwind does NOT do is run code placed after the
// `drop_in_place` call in the same `Drop` body, which is why the compaction
// (shifting the suffix left and restoring the length) lives in a separate guard
// whose `Drop` runs unconditionally. This test asserts both properties: the
// recovered vec's length/contents are coherent AND every hole element was
// dropped exactly once.
#[test]
fn drain_panic_in_destructor_still_compacts() {
    let armer = Arc::new(PanicArmer::new());
    let counter = Arc::new(DropCounter::new());

    #[allow(dead_code)]
    struct Panicky(u32, Arc<PanicArmer>, Arc<DropCounter>);
    impl Drop for Panicky {
        fn drop(&mut self) {
            self.2.record_drop();
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
    // even though #2's destructor panicked: the unwinder ran #3's destructor as
    // a landing pad on the same frame. So the counter already reflects 2 drops
    // from the hole, plus nothing else yet (element 1 is held in `taken`, and
    // 0, 4, 5 are still alive in `v`).
    assert_eq!(
        counter.get(),
        2,
        "both hole elements must be dropped despite the panic"
    );
    // Disarm before dropping the rest so the catch above isn't re-triggered.
    armer.disarm();
    drop(taken); // element 1 -> 3
    drop(v); // elements 0, 4, 5 -> 6 total
    assert_eq!(
        counter.get(),
        6,
        "every element dropped exactly once overall"
    );
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
    // Track each element by id so we can prove the split moved ownership without
    // dropping anything, and that each element is destroyed exactly once when
    // its owning vec finally goes away — no double-free, no leak.
    let ledger = Arc::new(Ledger::new());
    let budget = Arc::new(CloneBudget::new(u32::MAX));

    let mut v: Vec<FlakyTrackedItem> = Vec::new();
    for i in 0..6u32 {
        ledger.register(i);
        v.try_push(FlakyTrackedItem {
            id: i,
            ledger: ledger.clone(),
            inner: (*budget).share(),
        })
        .unwrap();
    }
    let right = v.try_split_off(3).unwrap();
    assert_eq!(v.len(), 3);
    assert_eq!(right.len(), 3);
    // The split only moved pointers; nothing was dropped. All six ids still live.
    assert!(ledger.drop_counts().is_empty());
    assert_eq!(ledger.live_ids(), [0, 1, 2, 3, 4, 5]);
    drop(v); // drops ids 0, 1, 2
    assert_eq!(ledger.live_ids(), [3, 4, 5]);
    assert!(ledger.double_dropped().is_empty());
    drop(right); // drops ids 3, 4, 5
    // Every element dropped exactly once; nothing leaked.
    assert!(ledger.leaked_ids().is_empty());
    assert!(ledger.all_dropped_once(0..6));
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

/// Regression test: if `try_clone` panics mid-loop (after some elements have
/// already been appended), the rollback guard must truncate the vector back to
/// its original length so every appended element is dropped exactly once when
/// the Vec unwinds — no leaks, no double-frees. Before the guard fix, a panic
/// after a partial append would leak the already-cloned tail.
///
/// Uses the shared [`Ledger`] helper to track individual payload ids, enabling
/// precise verification of both leaks and double-frees.
#[test]
fn extend_from_within_panic_is_safe() {
    /// A tracked item whose `try_clone` panics once it has produced its third
    /// successful clone overall (id >= 5 means ids 4 and 5 were the two prior
    /// successes), driving the loop past at least one successful append before
    /// unwinding. The panicked clone undoes its own registration so no payload
    /// survives that call.
    struct PanickingItem {
        pub id: u32,
        pub ledger: Arc<Ledger>,
        pub budget: Arc<CloneBudget>,
    }

    impl Drop for PanickingItem {
        fn drop(&mut self) {
            self.ledger.unregister(self.id);
        }
    }

    impl TryClone for PanickingItem {
        fn try_clone(&self) -> Result<Self, TryCloneError> {
            if !self.budget.try_consume() {
                return Err(TryCloneError::Other("budget exhausted"));
            }
            let id = self.ledger.allocate();
            self.ledger.register(id);
            if id >= 5 {
                // Undo the registration: no item survives this call.
                self.ledger.unregister(id);
                panic!("forced panic in try_clone");
            }
            Ok(PanickingItem {
                id,
                ledger: self.ledger.clone(),
                budget: self.budget.clone(),
            })
        }
    }

    let ledger = Arc::new(Ledger::new());
    let budget = Arc::new(CloneBudget::new(u32::MAX));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let l = ledger.clone();
        let b = budget.clone();
        move || {
            let mut v: Vec<PanickingItem> = Vec::new();
            // Push 4 initial items (ids 0–3).
            for _ in 0..4 {
                let id = l.allocate();
                l.register(id);
                v.try_push(PanickingItem {
                    id,
                    ledger: l.clone(),
                    budget: b.clone(),
                })
                .unwrap();
            }
            // Range 0..4: clones source elements 0–3. Clone calls produce ids
            // 4, 5, 6, …. Id 4 succeeds (append #1), id 5 succeeds (append
            // #2), id 6 triggers the panic. Two appends are outstanding when
            // we unwind through the rollback guard.
            let _ = v.try_extend_from_within(0..4);
        }
    }));

    assert!(result.is_err(), "expected try_clone to panic");

    // After the catch, all Payloads constructed inside the closure have been
    // destroyed. The counter-based ledger verifies all three invariants at once:
    // no leaks (nothing still live), no double-frees (no id dropped twice), and
    // an exact total of 6 drops (4 originals + 2 successful clones; the two that
    // aborted in `try_clone` never became Payloads).
    assert!(ledger.leaked_ids().is_empty());
    assert!(ledger.double_dropped().is_empty());
    // Exactly 6 ids were ever allocated (4 originals + 2 successful clones); the
    // third clone call panicked before allocating an id, so no Payload survived it.
    assert_eq!(ledger.total_allocated(), 6);
    // Every one of those 6 dropped exactly once.
    assert!(ledger.all_dropped_once(0..6));
}

// ---------------------------------------------------------------------------
// Conversion trait impls: TryFrom<[T; N]>, From<Box<[T], A>>, TryFrom<Cow<'b, [T]>>,
// TryFromIterator, TryFrom<&[T]>
// ---------------------------------------------------------------------------

/// Builds an olive `Vec` from literals without relying on a `vec!` macro (std's
/// would produce std's `Vec`, and no `vec!` shim exists in this crate).
fn mk_vec(items: &[i32]) -> Vec<i32, Global> {
    let mut v = Vec::new();
    for x in items {
        v.try_push(*x).unwrap();
    }
    v
}

#[test]
fn try_from_array_moves_elements_without_cloning() {
    use core::cell::Cell;
    struct Counting(u32, Cell<u32>);
    impl TryClone for Counting {
        fn try_clone(&self) -> Result<Self, TryCloneError> {
            self.1.set(self.1.get() + 1);
            Ok(Self(self.0, Cell::new(self.1.get())))
        }
    }
    let clones = Cell::new(0);
    let arr = [Counting(1, clones.clone()), Counting(2, clones.clone())];
    let v: Vec<Counting, Global> = Vec::try_from(arr).unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(v[0].0, 1);
    assert_eq!(v[1].0, 2);
    // Elements were moved, not cloned.
    assert_eq!(clones.get(), 0);
}

#[test]
fn try_from_empty_array_is_empty() {
    let v: Vec<i32, Global> = Vec::try_from([]).unwrap();
    assert!(v.is_empty());
    // An empty vector of a non-ZST never allocates, so it reports no usable
    // slots (`usize::MAX` is reserved for zero-sized elements).
    assert_eq!(v.capacity(), 0);
}

#[test]
fn try_from_zst_array_no_alloc() {
    #[derive(TryClone, PartialEq, Copy, Clone)]
    struct Zst;
    let v: Vec<Zst, Global> = Vec::try_from([Zst; 5]).unwrap();
    assert_eq!(v.len(), 5);
    // ZST elements occupy no memory and never require a real allocation, so
    // the reported capacity is `usize::MAX` (matching std's convention).
    assert_eq!(v.capacity(), usize::MAX);
}

#[test]
fn from_boxed_slice_preserves_contents_and_capacity() {
    // Build a vector with exactly `len` of capacity (no growth slack), so the
    // boxed slice carries a tight allocation and the round-trip is exact.
    let mut src = Vec::new();
    src.try_reserve_exact(3).unwrap();
    for x in [10, 20, 30] {
        src.try_push(x).unwrap();
    }
    let boxed: Box<[i32]> = src.try_into_boxed_slice().unwrap();
    let out: Vec<i32> = Vec::from(boxed);
    assert_eq!(out.as_slice(), &[10, 20, 30]);
    // The buffer was taken over untouched, so the reported capacity matches
    // the original allocation exactly.
    assert_eq!(out.capacity(), 3);
}

#[test]
fn from_boxed_empty_slice() {
    let src: Vec<u8> = Vec::new();
    let boxed: Box<[u8]> = src.try_into_boxed_slice().unwrap();
    let v: Vec<u8> = Vec::from(boxed);
    assert!(v.is_empty());
    // The empty box carried no real allocation, so the reconstructed vector
    // of a non-ZST reports zero usable slots.
    assert_eq!(v.capacity(), 0);
}

#[test]
fn try_from_cow_borrowed_clones_elements() {
    let src: &[i32] = &[11, 22];
    let cow = Cow::Borrowed(src);
    let v: Vec<i32> = Vec::try_from(cow).unwrap();
    assert_eq!(v.as_slice(), &[11, 22]);
}

#[test]
fn try_from_cow_borrowed_fails_when_element_clone_fails() {
    // The Borrowed arm delegates to `try_from_slice_in`, which clones each
    // element via `TryClone`. A `FlakyClone` with threshold 0 fails its first
    // clone attempt, so the very first element cannot be copied into the new
    // vector and the conversion surfaces a `Clone` error.
    let items = [FlakyClone::new(0), FlakyClone::new(0)];
    let cow = Cow::Borrowed(items.as_slice());
    match Vec::try_from(cow).expect_err("first element clone should fail") {
        TryVecWithCloneError::Clone(TryCloneError::Other(_)) => {}
        other => panic!("expected a clone failure, got {other:?}"),
    }
}

#[test]
fn try_from_cow_owned_passes_through() {
    let mut owned: Vec<i32> = Vec::new();
    for x in [7, 8, 9] {
        owned.try_push(x).unwrap();
    }
    // Give the source vec surplus capacity so we can verify it's preserved.
    owned.try_reserve(5).unwrap();
    let src_cap = owned.capacity();
    assert!(src_cap > 3);

    let cow = Cow::Owned(owned);
    let v: Vec<i32> = Vec::try_from(cow).unwrap();
    assert_eq!(v.as_slice(), &[7, 8, 9]);
    // The Owned arm transfers the buffer wholesale — capacity is preserved.
    assert_eq!(v.capacity(), src_cap);
}

#[test]
fn try_from_cow_borrowed_fails_when_reservation_fails() {
    // Drive the Borrowed arm through a FlakyCloneAlloc whose underlying
    // allocator always fails, so the reservation in try_from_slice_in errors.
    // Since TryFrom<Cow> is scoped to Global, we verify the same code path
    // via try_from_slice_in directly (the Borrowed arm delegates to it).
    let src: &[i32] = &[1, 2, 3];
    let res: Result<Vec<i32, FailAlloc>, _> = Vec::try_from_slice_in(src, FailAlloc);
    match res.expect_err("reservation should fail") {
        TryVecWithCloneError::Reserve(r) => assert!(r.is_alloc()),
        other => panic!("expected a reserve failure, got {other:?}"),
    }
}

#[test]
fn try_from_iter_builds_vec_on_target_allocator() {
    let v: Vec<i32, Global> = (1..=4).try_collect().unwrap();
    assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn try_from_iter_fails_when_allocation_fails() {
    use olive_core::alloc_errors::TryReserveErrorKind;
    // The `Global`-scoped trait impl can't target a failing allocator, so drive
    // the same code path through the `_in` seam with one that does.
    let res: Result<Vec<i32, FailAlloc>, _> = Vec::try_from_iter_in(1..=4, FailAlloc);
    let r = res.expect_err("allocation should fail");
    assert!(matches!(r.kind(), TryReserveErrorKind::AllocError { .. }))
}

/// An iterator whose `size_hint` advertises a far larger upper bound than the
/// number of elements it actually yields — the canonical "over-reporting" case.
struct OverhintedIter {
    remaining: usize,
    advertised_upper: usize,
}

impl Iterator for OverhintedIter {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let value = (self.advertised_upper - self.remaining) as i32;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.remaining.min(self.advertised_upper),
            Some(self.advertised_upper),
        )
    }
}

#[test]
fn try_from_iter_absorbs_overhint_and_grows_incrementally() {
    use crate::test_helpers::allocators::ByteCapAlloc;
    // The iterator advertises an upper bound of 1_000 but yields only 4
    // elements. The upfront best-effort reserve for 1_000 slots (~4 KB) is
    // rejected by the byte-capped allocator; `try_from_iter_in` absorbs that
    // failure and falls back to incremental per-element growth, where each
    // small reserve stays within the cap. Collection therefore succeeds with
    // all four real elements — the bogus hint did not abort it.
    let iter = OverhintedIter {
        remaining: 4,
        advertised_upper: 1_000,
    };
    let v: Vec<i32, ByteCapAlloc> = Vec::try_from_iter_in(iter, ByteCapAlloc::new(64))
        .expect("over-hint should be absorbed via incremental growth");
    assert_eq!(v.as_slice(), &[997, 998, 999, 1_000]);
}

#[test]
fn try_from_iter_still_fails_on_genuine_oom_under_byte_cap() {
    use crate::test_helpers::allocators::ByteCapAlloc;
    // Contrast to the absorption case above: the iterator's hint is honest
    // (it really does yield 20 elements), so the upfront reserve is legitimate
    // yet exceeds the 64-byte cap, and incremental growth cannot recover either
    // — reaching 20 i32s requires a buffer of at least 80 bytes, beyond the
    // cap. A genuine OOM is surfaced, not swallowed by the over-hint fallback.
    let iter = std::iter::repeat(1i32).take(20);
    let res: Result<Vec<i32, ByteCapAlloc>, _> = Vec::try_from_iter_in(iter, ByteCapAlloc::new(64));
    let e = res.expect_err("genuine OOM should surface");
    assert!(e.is_alloc())
}

#[test]
fn try_from_borrowed_slice_clones_elements() {
    let src: &[i32] = &[10, 20, 30];
    let v: Vec<i32, Global> = Vec::try_from(src).unwrap();
    assert_eq!(v.as_slice(), &[10, 20, 30]);
}

#[test]
fn try_from_borrowed_slice_fails_when_reservation_fails() {
    // The `Global`-scoped trait impl can't target a failing allocator, so drive
    // the same code path through the `_in` seam with one that does.
    let src: &[i32] = &[1, 2, 3];
    let res: Result<Vec<i32, FailAlloc>, _> = Vec::try_from_slice_in(src, FailAlloc);
    match res.expect_err("reservation should fail") {
        TryVecWithCloneError::Reserve(r) => {
            assert!(r.is_alloc());
        }
        other => panic!("expected a reserve failure, got {other:?}"),
    }
}

#[test]
fn try_from_borrowed_slice_fails_when_element_clone_fails() {
    // `FlakyClone` with threshold 0 fails its first clone attempt, so the very
    // first element in the slice cannot be copied into the new vector.
    let items = [FlakyClone::new(0), FlakyClone::new(0)];
    let res: Result<Vec<FlakyClone, Global>, _> = Vec::try_from(&items[..]);
    match res.expect_err("first element clone should fail") {
        TryVecWithCloneError::Clone(TryCloneError::Other(_)) => {}
        other => panic!("expected a clone failure, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Comparison trait impls: PartialEq, Eq, PartialOrd, Ord, Hash
// ---------------------------------------------------------------------------

// FIXME: should use PartialEq and Eq to avoid conflicts
#[test]
fn vec_partial_eq_same_len_equal() {
    let a = mk_vec(&[1, 2, 3]);
    let b = mk_vec(&[1, 2, 3]);
    assert_eq!(a, b);
}

#[test]
fn vec_partial_eq_differs_at_element() {
    let a = mk_vec(&[1, 2, 3]);
    let b = mk_vec(&[1, 2, 4]);
    assert_ne!(a, b);
}

#[test]
fn vec_partial_eq_different_lengths() {
    let a = mk_vec(&[1, 2, 3]);
    let b = mk_vec(&[1, 2]);
    assert_ne!(a, b);
}

#[test]
fn vec_partial_eq_cross_type_vs_slice() {
    let v = mk_vec(&[1, 2, 3]);
    let s: &[i32] = &[1, 2, 3];
    let s_mut: &mut [i32] = &mut [1, 2, 3];
    assert_eq!(v, s);
    assert_eq!(v, s_mut);
}

#[test]
fn vec_partial_eq_cross_type_vs_array() {
    let v = mk_vec(&[1, 2, 3]);
    let mut a: [i32; 3] = [1, 2, 3];
    assert_eq!(v, a);
    assert_eq!(v, &a);
    assert_eq!(v, &mut a);
}

#[test]
fn vec_ord_less_than_by_prefix_then_length() {
    let a = mk_vec(&[1, 2]);
    let b = mk_vec(&[1, 2, 3]);
    assert!(a < b);
    assert!(b > a);
}

#[test]
fn vec_ord_lexicographic() {
    let a = mk_vec(&[1, 2, 3]);
    let b = mk_vec(&[1, 3, 2]);
    assert!(a < b);
    assert!(b > a);
}

#[test]
fn vec_hash_consistent_for_equal_vectors() {
    use core::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let a = mk_vec(&[1, 2, 3]);
    let b = mk_vec(&[1, 2, 3]);

    let mut ha = DefaultHasher::new();
    let mut hb = DefaultHasher::new();
    a.hash(&mut ha);
    b.hash(&mut hb);
    assert_eq!(ha.finish(), hb.finish());
}

#[test]
fn vec_hash_differs_for_different_vectors() {
    use core::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let a = mk_vec(&[1, 2, 3]);
    let b = mk_vec(&[1, 2, 4]);

    let mut ha = DefaultHasher::new();
    let mut hb = DefaultHasher::new();
    a.hash(&mut ha);
    b.hash(&mut hb);
    assert_ne!(ha.finish(), hb.finish());
}
