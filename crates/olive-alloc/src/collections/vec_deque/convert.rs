//! Zero-allocation conversions for [`VecDeque`].
use core::mem::ManuallyDrop;
use core::ptr;

use super::VecDeque;
use super::wrapped_index::WrappedIndex;
use crate::alloc::Allocator;
use crate::raw_vec::RawVec;
use crate::vec::Vec;

// ---------------------------------------------------------------------------
// VecDeque -> Vec
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Consumes this deque and returns its contents as a [`Vec`] on the same
    /// allocator.
    ///
    /// The conversion moves the existing buffer and does not allocate.
    /// The resulting `Vec` reports the same capacity the deque had.
    ///
    /// # Allocator transfer
    ///
    /// The deque's allocator handle is transferred verbatim into the returned
    /// vector. Transferring between different buffers using convenience methods
    /// is deliberately not supported.
    ///
    /// This is also available via [`From<VecDeque<T, A>> for Vec<T, A>`].
    pub fn into_vec(mut self) -> Vec<T, A> {
        self.make_contiguous();

        unsafe {
            let me = ManuallyDrop::new(self);
            let buf = me.buf.ptr();
            let len = me.len();
            let cap = me.capacity();
            let alloc = ptr::read(me.allocator());

            if !me.head.is_zero() {
                ptr::copy(buf.add(me.head.as_index()), buf, len);
            }
            Vec::from_raw_parts_in(buf, len, cap, alloc)
        }
    }
}

impl<T, A: Allocator> From<VecDeque<T, A>> for Vec<T, A> {
    /// Converts a [`VecDeque`] into a [`Vec`], moving the buffer without
    /// allocating. See [`VecDeque::into_vec`].
    fn from(deque: VecDeque<T, A>) -> Self {
        deque.into_vec()
    }
}

// ---------------------------------------------------------------------------
// Vec -> VecDeque
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Consumes this vector and returns its contents as a [`VecDeque`] on the
    /// same allocator.
    ///
    /// The conversion does not allocate.
    ///
    /// # Allocator transfer
    ///
    /// The vector's allocator handle is transferred verbatim into the returned
    /// deque. Transferring between different buffers using convenience methods
    /// is deliberately not supported.
    ///
    /// This is also available via [`From<Vec<T, A>> for VecDeque<T, A>`].
    pub fn into_vecdeque(self) -> VecDeque<T, A> {
        // Decompose by value: `into_raw_parts_with_alloc` wraps `self` in
        // `ManuallyDrop`, so nothing here double-frees. The buffer and the
        // allocator handle are handed straight to the reconstructed deque.
        //
        // SAFETY: we consume `self`; the pointer is handed back to us.
        let (ptr, len, cap, alloc) = unsafe { self.into_raw_parts_with_alloc() };

        // SAFETY: a `Vec`'s buffer is a linear run starting at `ptr`, so a
        // deque with `head == 0` and `len` elements reads exactly those slots.
        // `len <= cap` is the vector invariant, and `ptr` was allocated by
        // `alloc` (or is dangling for an empty/ZST buffer).
        unsafe {
            let buf = RawVec::from_raw_parts_in(ptr, cap, alloc);
            VecDeque {
                buf,
                head: WrappedIndex::zero(),
                len,
            }
        }
    }
}

impl<T, A: Allocator> From<Vec<T, A>> for VecDeque<T, A> {
    /// Converts a [`Vec`] into a [`VecDeque`], reinterpreting the buffer
    /// without allocating. See [`Vec::into_vecdeque`].
    fn from(vec: Vec<T, A>) -> Self {
        vec.into_vecdeque()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::alloc::Global;
    use crate::test_helpers::{Ledger, TrackedItem};
    use std::sync::Arc;
    /// Helper: build a crate `Vec<i32>` on the global allocator holding `vals`.
    fn mk_vec(vals: &[i32]) -> Vec<i32, Global> {
        let mut v = Vec::new();
        for &x in vals {
            v.try_push(x).unwrap();
        }
        v
    }

    /// Builds a deque of `n` tracked items sharing one ledger, registering each
    /// id as live before pushing. Returns the ids in insertion order.
    fn make_tracked_deque(n: u32) -> (VecDeque<TrackedItem<()>>, Arc<Ledger>, std::vec::Vec<u32>) {
        let ledger = Arc::new(Ledger::new());
        let mut dq = VecDeque::new();
        let mut ids = std::vec::Vec::with_capacity(n as usize);
        for i in 0..n {
            let id = ledger.allocate();
            ledger.register(id);
            let item = TrackedItem {
                id,
                ledger: ledger.clone(),
                inner: (),
            };
            dq.try_push_back(item).expect("push ok");
            ids.push(i);
        }
        (dq, ledger, ids)
    }

    // --- VecDeque -> Vec ----------------------------------------------------

    #[test]
    fn into_vec_preserves_order_and_length() {
        let mut dq = VecDeque::new();
        for i in 0..5 {
            dq.try_push_back(i).unwrap();
        }
        let v = dq.into_vec();
        assert_eq!(v.len(), 5);
        assert_eq!(v[..], [0, 1, 2, 3, 4]);
    }

    #[test]
    fn into_vec_empty() {
        let dq: VecDeque<i32> = VecDeque::new();
        let v = dq.into_vec();
        assert!(v.is_empty());
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn into_vec_zst() {
        let mut dq: VecDeque<()> = VecDeque::new();
        for _ in 0..4 {
            dq.try_push_back(()).unwrap();
        }
        let v = dq.into_vec();
        assert_eq!(v.len(), 4);
        assert_eq!(v.capacity(), usize::MAX);
    }

    #[test]
    fn into_vec_preserves_capacity_when_head_already_at_front() {
        // Pushing only keeps the deque contiguous at slot 0, so the buffer is
        // handed over untouched and the exact capacity survives.
        let mut dq = VecDeque::new();
        for i in 0..8 {
            dq.try_push_back(i).unwrap();
        }
        let cap_before = dq.capacity();
        let v = dq.into_vec();
        assert_eq!(v.capacity(), cap_before);
        assert_eq!(v.len(), 8);
    }

    /// Builds a genuinely wrapped deque of tracked items sharing one ledger:
    /// allocates a fixed capacity of `n + 2`, fills `n` slots from the back,
    /// then pushes two fronts so the buffer is full and the logical run
    /// straddles the physical wrap point. Returns the ids in logical
    /// (front-to-back) order.
    fn make_wrapped_tracked_deque(
        n: u32,
    ) -> (VecDeque<TrackedItem<()>>, Arc<Ledger>, std::vec::Vec<u32>) {
        let ledger = Arc::new(Ledger::new());
        // Total live count will be n + 2, which equals the capacity, so the
        // final state is a full buffer with head pushed past slot 0.
        let mut dq =
            VecDeque::<TrackedItem<()>>::try_with_capacity(n as usize + 2).expect("allocation ok");

        // Fill n slots from the back.
        let mut ids = std::vec::Vec::with_capacity(n as usize);
        for i in 0..n {
            let id = ledger.allocate();
            ledger.register(id);
            let item = TrackedItem {
                id,
                ledger: ledger.clone(),
                inner: (),
            };
            dq.try_push_back_within_capacity(item)
                .expect("within-capacity push ok");
            ids.push(i);
        }

        // Push two more elements onto the front; each prepend lands ahead of
        // the previous one, wrapping around ahead of head and forcing a
        // non-contiguous layout. Final len == capacity. Because `push_front`
        // reverses insertion order relative to the deque's front, collect into
        // a reversed iterator so the returned ids read front-to-back.
        let mut front_ids_rev = std::vec::Vec::with_capacity(2);
        for _j in 0..2u32 {
            let id = ledger.allocate();
            ledger.register(id);
            let item = TrackedItem {
                id,
                ledger: ledger.clone(),
                inner: (),
            };
            dq.try_push_front_within_capacity(item)
                .expect("within-capacity push ok");
            front_ids_rev.push(id);
        }
        // Logical order is now [last_front_id, first_front_id, back_0, ..., back_{n-1}].
        let ordered = front_ids_rev.into_iter().rev().chain(ids).collect();
        (dq, ledger, ordered)
    }

    #[test]
    fn into_vec_shifts_wrapped_elements() {
        // Fixed-capacity deque filled, then pushed on the front so the live run
        // wraps the physical buffer — exactly the layout that must be shifted.
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("allocation ok");
        for v in [1, 2, 3] {
            dq.try_push_back_within_capacity(v).unwrap();
        }
        dq.try_push_front_within_capacity(4).unwrap();
        dq.try_push_front_within_capacity(5).unwrap();
        assert!(!dq.is_contiguous());

        let v = dq.into_vec();
        assert_eq!(v[..], [5, 4, 1, 2, 3]);
    }

    #[test]
    fn into_vec_drops_each_element_exactly_once() {
        let (dq, ledger, ids) = make_tracked_deque(6);
        drop(dq.into_vec());
        assert!(ledger.all_dropped_once(ids.iter().copied()));
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn into_vec_drops_wrapped_elements_exactly_once() {
        let (dq, ledger, ids) = make_wrapped_tracked_deque(6);
        assert!(!dq.is_contiguous());
        drop(dq.into_vec());
        // Every element — including those sitting on both sides of the physical
        // wrap point — must be dropped exactly once by the handoff.
        assert!(ledger.all_dropped_once(ids.iter().copied()));
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
    }

    // --- Vec -> VecDeque ----------------------------------------------------

    #[test]
    fn from_vec_preserves_order_and_length() {
        let v = mk_vec(&[9, 8, 7, 6]);
        let dq = VecDeque::from(v);
        assert_eq!(dq.len(), 4);
        assert_eq!(dq.front(), Some(&9));
        assert_eq!(dq.back(), Some(&6));
        let (a, b) = dq.as_slices();
        assert_eq!(a, &[9, 8, 7, 6]);
        assert!(b.is_empty());
    }

    #[test]
    fn from_vec_empty() {
        let v: Vec<i32, Global> = Vec::new();
        let dq = VecDeque::from(v);
        assert!(dq.is_empty());
        assert_eq!(dq.capacity(), 0);
    }

    #[test]
    fn from_vec_zst() {
        let mut v: Vec<(), Global> = Vec::new();
        for _ in 0..3 {
            v.try_push(()).unwrap();
        }
        let dq = VecDeque::from(v);
        assert_eq!(dq.len(), 3);
        assert_eq!(dq.capacity(), usize::MAX);
    }

    #[test]
    fn from_vec_preserves_capacity() {
        let mut v = mk_vec(&[1, 2, 3]);
        v.try_reserve(10).unwrap();
        let cap_before = v.capacity();
        let dq = VecDeque::from(v);
        assert_eq!(dq.capacity(), cap_before);
        assert_eq!(dq.len(), 3);
    }

    #[test]
    fn from_vec_drops_each_element_exactly_once() {
        let ledger = Arc::new(Ledger::new());
        let mut v: Vec<TrackedItem<()>, Global> = Vec::new();
        let mut ids = std::vec::Vec::new();
        for i in 0..5u32 {
            let id = ledger.allocate();
            ledger.register(id);
            v.try_push(TrackedItem {
                id,
                ledger: ledger.clone(),
                inner: (),
            })
            .unwrap();
            ids.push(i);
        }
        drop(VecDeque::from(v));
        assert!(ledger.all_dropped_once(ids.iter().copied()));
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
    }

    // --- Round trips --------------------------------------------------------

    #[test]
    fn round_trip_deque_to_vec_to_deque() {
        let mut dq = VecDeque::new();
        for i in 0..10 {
            dq.try_push_back(i).unwrap();
        }
        for _ in 0..4 {
            dq.pop_front().unwrap();
        }
        let expected: std::vec::Vec<i32> = (4..10).collect();

        let v = dq.into_vec();
        assert_eq!(v[..], expected[..]);
        let dq2 = VecDeque::from(v);
        let (a, b) = dq2.as_slices();
        let joined: std::vec::Vec<i32> = a.iter().chain(b.iter()).copied().collect();
        assert_eq!(joined, expected);
    }

    #[test]
    fn round_trip_vec_to_deque_to_vec() {
        let v = mk_vec(&[10, 20, 30, 40, 50]);
        let dq = VecDeque::from(v);
        let back = dq.into_vec();
        assert_eq!(back[..], [10, 20, 30, 40, 50]);
    }

    // --- Custom allocator ---------------------------------------------------

    #[test]
    fn custom_allocator_handoff_drops_handle_exactly_once() {
        let drops = Arc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::DropCountingAlloc::new(drops.clone());
        let mut dq: VecDeque<i32, _> = VecDeque::new_in(alloc);
        for i in 0..4 {
            dq.try_push_back(i).unwrap();
        }
        let v = dq.into_vec();
        assert_eq!(v[..], [0, 1, 2, 3]);
        drop(v);
        // The single allocator handle lived in the deque, moved into the vec,
        // and died with it — exactly once.
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn custom_allocator_reverse_handoff_drops_handle_exactly_once() {
        let drops = Arc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::DropCountingAlloc::new(drops.clone());
        let mut v: Vec<i32, _> = Vec::new_in(alloc);
        for i in 0..4 {
            v.try_push(i).unwrap();
        }
        let dq = VecDeque::from(v);
        assert_eq!(dq.len(), 4);
        drop(dq);
        assert_eq!(drops.get(), 1);
    }

    // --- Trait plumbing -----------------------------------------------------

    #[test]
    fn into_trait_impls_are_usable_via_into() {
        // Exercise the blanket `Into` derived from our `From` impls.
        let mut dq = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_push_back(2).unwrap();
        let v: Vec<i32, Global> = dq.into();
        assert_eq!(v[..], [1, 2]);

        let src = mk_vec(&[3, 4]);
        let dq2: VecDeque<i32, Global> = src.into();
        let (s1, _) = dq2.as_slices();
        assert_eq!(dq2.len(), 2);
        assert_eq!(s1, [3, 4]);
    }

    #[test]
    #[allow(
        clippy::unnecessary_fallible_conversions,
        reason = "this test intentionally exercises the fallible TryFrom/TryInto plumbing"
    )]
    fn try_from_impls_never_fail() {
        // Core's blanket `impl<T, U> TryFrom<U> for T where U: Into<T>` gives us
        // a `TryFrom` (and `TryInto`) for free from our `From` impls, with
        // `Infallible` as the error type. Exercise that plumbing in both
        // directions.
        use core::convert::{Infallible, TryFrom, TryInto};

        let mut dq = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_push_back(2).unwrap();
        let v: Result<Vec<i32, Global>, Infallible> = Vec::try_from(dq);
        assert_eq!(v.expect("infallible")[..], [1, 2]);

        let src = mk_vec(&[3, 4]);
        let dq2 = VecDeque::try_from(src).expect("infallible");
        assert_eq!(dq2.len(), 2);

        // The blanket `TryInto` should also be reachable and error-free.
        let back: Result<Vec<i32, Global>, Infallible> = dq2.try_into();
        assert_eq!(back.expect("infallible")[..], [3, 4]);
    }
}
