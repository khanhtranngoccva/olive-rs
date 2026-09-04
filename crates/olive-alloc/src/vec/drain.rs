//! The borrowing iterator produced by `Vec::try_drain`.
//!
//! This borrows the vector and yields a *range* of its drained elements one at a time.
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ptr;
use core::slice;

use crate::alloc::{Allocator, Global};

/// An iterator that drains a *range* of a vector, yielding owned elements while
/// removing them from the vector.
///
/// This `struct` is created by the
/// [`try_drain`](super::Vec::try_drain) method on
/// [`Vec`](super::Vec).
///
/// # Example
///
/// ```
/// use olive_alloc::vec::Vec;
/// let mut v = Vec::new();
/// for i in 0..5 { v.try_push(i).unwrap(); }
/// // Drain the middle two elements.
/// let drained: std::vec::Vec<i32> = v.try_drain(1..3).unwrap().collect();
/// assert_eq!(drained, std::vec![1, 2]);
/// // The remaining elements are compacted forward.
/// assert_eq!(v.as_slice(), &[0, 3, 4]);
/// ```
pub struct Drain<'a, T, A: Allocator = Global> {
    /// Number of elements already yielded from the front (via `next`). Together
    /// with `taken_back` this partitions the drain range into three regions:
    /// `[start + consumed, start + count - taken_back)` is still live.
    consumed: usize,
    /// Number of elements already yielded from the back (via `next_back`).
    taken_back: usize,
    /// Offset (in elements) of the drain range's start from the vector's base
    /// pointer. Because `try_drain` caps the vector's length to `start` at
    /// construction, this also equals the vec's length for the whole lifetime
    /// of the drainer — which is what keeps the drained hole invisible to the
    /// vec's own drop path.
    original_start: usize,
    /// Total number of elements in the resolved drain range (`end - start`).
    /// The entire range is removed when this iterator is dropped, regardless of
    /// how many elements were actually yielded.
    original_count: usize,
    /// The vector's length *before* it was capped down to `original_start`.
    /// Retained because capping makes the surviving tail's extent no longer
    /// derivable from `vec.len()` alone; `drop` uses this to relocate that
    /// tail, which sits at offsets `[original_start + original_count,
    /// original_len)`.
    original_len: usize,
    /// Mutable pointer back to the owning `Vec`, used to read element slots and
    /// to compact the surviving tail into place when this iterator is dropped.
    /// Derived from the enclosing `&mut self` borrow, so it carries mutable
    /// provenance (keeps Miri's Stacked Borrows model happy). We store a raw
    /// pointer rather than a `&'a mut Vec` so the `Drop` impl can reach through
    /// it without fighting the borrow checker over reborrows, and so the
    /// iterator's exclusive window over the buffer stays explicit.
    vec: *mut super::Vec<T, A>,
    _marker: PhantomData<&'a ()>,
}

// SAFETY: mirroring `Vec`, the iterator hands out owned values only and holds a
// unique mutable view of the buffer for its lifetime. Sound iff `T` and `A` are
// `Send`/`Sync`.
unsafe impl<T: Send, A: Allocator + Send> Send for Drain<'_, T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for Drain<'_, T, A> {}

impl<T, A: Allocator> Drain<'_, T, A> {
    /// Constructs a drainer from the raw parts of a borrowed `Vec`.
    ///
    /// No pointer arithmetic is performed here: the caller passes the already-
    /// validated integer bounds and the owning `Vec`, and all element pointers
    /// are derived lazily (and only while a live element provably exists).
    ///
    /// # Safety
    ///
    /// `count == end - start` must be the full resolved range length and
    /// `start + count <= original_len`; the first `original_len` slots of the
    /// `Vec` pointed to by `vec` must be initialized.
    ///
    /// The caller is expected to have already capped `vec`'s length down
    /// to `start` before calling this, so that the drained hole is excluded
    /// from the vector's live contents.
    #[inline]
    pub(super) const unsafe fn new_from_parts(
        start: usize,
        count: usize,
        original_len: usize,
        vec: *mut super::Vec<T, A>,
    ) -> Self {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "precondition needs to be met by caller"
        )]
        {
            debug_assert!(start + count <= original_len);
        }
        debug_assert!(unsafe { &*vec }.len() == start);
        Self {
            consumed: 0,
            taken_back: 0,
            original_start: start,
            original_count: count,
            original_len,
            vec,
            _marker: PhantomData,
        }
    }

    /// Number of elements still held in the live window.
    #[inline]
    const fn remaining(&self) -> usize {
        // `consumed + taken_back` can never exceed `original_count`: each yield
        // decrements the live window by exactly one and stops at zero.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "consumed + taken_back <= original_count"
        )]
        let taken = { self.consumed + self.taken_back };
        #[allow(clippy::arithmetic_side_effects, reason = "taken <= original_count")]
        {
            self.original_count - taken
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

    /// Views the remaining (not-yet-yielded) elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // The computed front pointer is invalid only when it lands one-past-the-
        // end of the buffer: `front_offset() == original_len`. When that case happens,
        // the fallback must be used. This fallback behavior is triggered sparingly
        // for conformance with std. (this also happens to work with ZSTs)
        if self.front_offset() == self.original_len {
            return unsafe { slice::from_raw_parts(self.base_ptr(), 0) };
        }
        let rem = self.remaining();
        // SAFETY: `front_offset() < original_len`;
        // the front of the live window addresses a real, initialized slot (or a
        // valid dangling-aligned ZST address), and `[front, front + rem)` lies
        // entirely within the initialized region of the buffer.
        unsafe {
            let vec = &*self.vec;
            let front = vec.as_ptr().add(self.front_offset());
            slice::from_raw_parts(front, rem)
        }
    }

    /// Base pointer of the owning vector's buffer. Only ever combined with an
    /// offset that provably lands on a live slot (guarded by `remaining > 0`)
    /// or with length zero, so no OOB pointer is formed.
    #[inline]
    fn base_ptr(&self) -> *const T {
        // SAFETY: `self.vec` points to the live `Vec` borrowed exclusively by
        // this drainer for `'a`.
        unsafe { (*self.vec).as_ptr() }
    }

    /// Element offset of the current front of the live window.
    #[inline]
    fn front_offset(&self) -> usize {
        #[allow(clippy::arithmetic_side_effects, reason = "consumed <= original_count")]
        {
            self.original_start + self.consumed
        }
    }

    /// Element offset of the current back of the live window (the last live
    /// element is at `back_offset() - 1`).
    #[inline]
    fn back_offset(&self) -> usize {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start + count == end <= original_len"
        )]
        let end = { self.original_start + self.original_count };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "taken_back <= original_count"
        )]
        {
            end - self.taken_back
        }
    }
}

impl<T, A: Allocator> Iterator for Drain<'_, T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining() == 0 {
            return None;
        }
        // Record the front offset *before* advancing; it addresses a live,
        // initialized slot because we just confirmed `remaining > 0`.
        let offset = self.front_offset();
        #[allow(clippy::arithmetic_side_effects, reason = "consumed < original_count")]
        {
            self.consumed += 1;
        }
        // SAFETY: `offset` is within the initialized region and the value is now
        // ours to move out. Reading a ZST from its (possibly dangling-aligned)
        // pointer is well defined. The borrow is exclusive for `'a`.
        Some(unsafe {
            let vec = &mut *self.vec;
            ptr::read(vec.as_mut_ptr().add(offset))
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining(), Some(self.remaining()))
    }
}

impl<T, A: Allocator> ExactSizeIterator for Drain<'_, T, A> {}

impl<T, A: Allocator> FusedIterator for Drain<'_, T, A> {}

impl<T, A: Allocator> DoubleEndedIterator for Drain<'_, T, A> {
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
            reason = "remaining > 0 implies back_offset >= original_start + 1"
        )]
        let offset = { self.back_offset() - 1 };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "taken_back < original_count"
        )]
        {
            self.taken_back += 1;
        }
        // SAFETY: as in `next`, `offset` is within the initialized region and
        // the value becomes the caller's. The borrow is exclusive for `'a`.
        Some(unsafe {
            let vec = &mut *self.vec;
            ptr::read(vec.as_mut_ptr().add(offset))
        })
    }
}

impl<T, A: Allocator> Drop for Drain<'_, T, A> {
    fn drop(&mut self) {
        let vec = unsafe { &mut *self.vec };

        let original_start = self.original_start;
        let original_count = self.original_count;
        let original_len = self.original_len;
        let consumed = self.consumed;
        let remaining = self.remaining();

        // Buffer layout (offsets from the base pointer):
        //   [prefix:           0..original_start)                              <- in vec.len(), survives
        //   [yielded left:     original_start..original_start+cons)            <- moved out (uninit bits)
        //   [hole:             original_start+cons..original_end-taken_back)   <- never yielded, must drop
        //   [yielded right:    original_end-taken_back..original_end)          <- moved out (uninit bits)
        //   [suffix:           original_end..original_len)                     <- beyond vec.len(), shifts left
        // where cons + taken_back + remaining = original_count
        // and original_end = original_start + original_count.
        //
        // Steps:
        // 1. Destroy the unconsumed ("hole") elements.
        // 2. Shift the suffix left by `original_count` to close the full gap.
        // 3. Extend the length past the prefix to include the compacted suffix.
        //
        // Only step 1 can panic: `drop_in_place` over a fat slice runs every
        // `T` destructor in the hole, whereas step 2 (`ptr::copy`) is a pure
        // bitwise move that drops nothing and step 3 (`set_len`) just writes a
        // field. Because the compiler lowers a fat-slice drop to a per-element
        // sequence inside one function body, if one destructor panics mid-hole
        // the unwinder still runs the remaining destructors as landing pads on
        // the same frame before leaving it — so no hole element is leaked. What
        // the unwind does NOT do is run any code placed after the
        // `drop_in_place` call in this same `Drop` body; that is why the
        // compaction (steps 2 and 3) lives in a separate guard whose `Drop`
        // runs unconditionally. Arming the guard *before* step 1 guarantees the
        // compaction runs both on the happy path (guard falls out of scope
        // normally) and on unwind (the guard's `Drop` runs during stack
        // teardown).
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "original_count <= original_len"
        )]
        let final_len = original_len - original_count;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "start + count == end <= original_len"
        )]
        let suffix_src_offset = original_start + original_count;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "suffix_src_offset <= original_len"
        )]
        let suffix_len = original_len - suffix_src_offset;
        let base = vec.as_mut_ptr();
        let _compact_guard = DrainCompactGuard {
            vec,
            base,
            original_start,
            suffix_src_offset,
            suffix_len,
            final_len,
        };

        // Step 1: destroy the unconsumed remainder of the drain range.
        if remaining > 0 && size_of::<T>() != 0 {
            #[allow(clippy::arithmetic_side_effects, reason = "consumed <= original_count")]
            let hole_offset = original_start + consumed;
            unsafe {
                let p = base.add(hole_offset);
                ptr::drop_in_place(slice::from_raw_parts_mut(p, remaining));
            }
        }

        // Steps 2 (shift) and 3 (length restore) are performed by the guard's
        // `Drop`, which runs unconditionally — on the happy path when the guard
        // falls out of scope, and on unwind if step 1 panicked. See
        // `DrainCompactGuard` below.
    }
}

/// Finishes a drain's compaction — shifting the surviving suffix left and
/// restoring the vector's length — even if the preceding element destruction
/// unwinds.
struct DrainCompactGuard<'a, T, A: Allocator> {
    /// Exclusive reference to the owning vector being compacted.
    vec: &'a mut super::Vec<T, A>,
    /// Base of the vector's buffer; offsets below are relative to it.
    base: *mut T,
    /// Offset where the drained range began; the suffix lands here after the
    /// leftward shift.
    original_start: usize,
    /// Offset where the suffix currently begins: `original_start + count`.
    suffix_src_offset: usize,
    /// Number of suffix elements to relocate.
    suffix_len: usize,
    /// Length to restore: `original_len - original_count`, i.e. prefix plus the
    /// compacted suffix. Always `<= capacity()`.
    final_len: usize,
}

impl<T, A: Allocator> Drop for DrainCompactGuard<'_, T, A> {
    fn drop(&mut self) {
        // Step 2: shift the suffix left by the drained count.
        // SAFETY: both pointers are within the allocation and aligned;
        // `ptr::copy` correctly handles the overlapping case (leftward shift).
        // For ZSTs this is a no-op. OOB pointer is prevented with a suffix_len
        // guard.
        if self.suffix_len > 0 && size_of::<T>() != 0 {
            unsafe {
                let dst = self.base.add(self.original_start);
                let src = self.base.add(self.suffix_src_offset);
                ptr::copy(src, dst, self.suffix_len);
            }
        }

        // Step 3: extend the length to include the compacted suffix.
        // SAFETY: every slot in `[0..final_len)` is initialized (the
        // untouched prefix plus the relocated suffix) and `final_len <=
        // capacity()`, satisfying `set_len`'s preconditions.
        unsafe {
            self.vec.set_len(self.final_len);
        }
    }
}
