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
use olive_core::try_traits::try_clone::TryClone;

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

/// Error returned by fallible deque operations that invoke a user-supplied
/// fallible closure, such as [`VecDeque::try_resize_with`].
///
/// The closure may fail with any error type `E`, and the capacity reservation
/// itself may also fail independently. Mirrors `Vec`'s `TryVecWithClosureError`.
#[derive(Clone, PartialEq, Eq)]
pub enum TryVecDequeWithClosureError<E> {
    /// A capacity reservation on the deque failed (overflow or OOM).
    Reserve(TryReserveError),
    /// The closure returned an error.
    Closure(E),
}

impl<E: core::fmt::Debug> core::fmt::Debug for TryVecDequeWithClosureError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Reserve(e) => f
                .debug_tuple("TryVecDequeWithClosureError::Reserve")
                .field(e)
                .finish(),
            Self::Closure(e) => f
                .debug_tuple("TryVecDequeWithClosureError::Closure")
                .field(e)
                .finish(),
        }
    }
}

impl<E: core::fmt::Display> core::fmt::Display for TryVecDequeWithClosureError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "deque operation failed: {e}"),
            Self::Closure(e) => write!(f, "deque operation failed: {e}"),
        }
    }
}

impl<E: core::error::Error> core::error::Error for TryVecDequeWithClosureError<E> {}

impl<E> From<TryReserveError> for TryVecDequeWithClosureError<E> {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

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

impl<T, A: Allocator> VecDeque<T, A> {
    /// Shrinks the capacity of the deque as much as possible.
    ///
    /// It will drop down as close as possible to the length but the allocator
    /// may still inform the deque that there is space for a few more elements.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the reallocation fails. On failure or panic  
    /// the deque is restored to a consistent state at its original (larger)
    /// capacity.
    pub fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError> {
        self.try_shrink_to(0)
    }

    /// Shrinks the capacity of the deque with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the reallocation fails. On failure the
    /// deque is restored to a consistent state at its original (larger)
    /// capacity.
    pub fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError> {
        let target_cap = min_capacity.max(self.len);

        // never shrink ZSTs
        if size_of::<T>() == 0 || self.capacity() <= target_cap {
            return Ok(());
        }

        // There are three cases of interest:
        //   All elements are out of desired bounds
        //   Elements are contiguous, and tail is out of desired bounds
        //   Elements are discontiguous
        //
        // At all other times, element positions are unaffected.

        // `head` and `len` are at most `isize::MAX` and `target_cap <
        // self.capacity()`, so nothing can overflow.
        let old_head = self.head.as_index();
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "head + len < 2 * capacity <= 2 * isize::MAX == usize::MAX - 1"
        )]
        let tail_outside = (target_cap + 1..=self.capacity()).contains(&(old_head + self.len));

        if self.len == 0 {
            self.head = WrappedIndex::zero();
        } else if old_head >= target_cap && tail_outside {
            // Head and tail are both out of bounds, so copy all of them to the
            // front.
            //
            //  H := head
            //  L := last element
            //                    H           L
            //   [. . . . . . . . o o o o o o o . ]
            //    H           L
            //   [o o o o o o o . ]
            unsafe {
                // nonoverlapping because `head >= target_cap >= self.len`.
                self.copy_nonoverlapping(
                    WrappedIndex::from_arbitrary_number(old_head),
                    WrappedIndex::zero(),
                    self.len,
                );
            }
            self.head = WrappedIndex::zero();
        } else if old_head < target_cap && tail_outside {
            // Head is in bounds, tail is out of bounds.
            // Copy the overflowing part to the beginning of the
            // buffer. This won't overlap because `target_cap >= self.len`.
            //
            //  H := head
            //  L := last element
            //          H           L
            //   [. . . o o o o o o o . . . . . . ]
            //      L   H
            //   [o o . o o o o o ]
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "head + len < 2 * capacity and tail_outside implies head + len > target_cap"
            )]
            let len = old_head + self.len - target_cap;
            unsafe {
                self.copy_nonoverlapping(
                    WrappedIndex::from_arbitrary_number(target_cap),
                    WrappedIndex::zero(),
                    len,
                );
            }
        } else if !self.is_contiguous() {
            // The head slice is at least partially out of bounds, tail is in
            // bounds.
            // Copy the head backwards so it lines up with the target capacity.
            // This won't overlap because `target_cap >= self.len`.
            //
            //  H := head
            //  L := last element
            //            L                   H
            //   [o o o o o . . . . . . . . . o o ]
            //            L   H
            //   [o o o o o . o o ]
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "head < capacity and head_len <= len <= target_cap"
            )]
            let (head_len, new_head) = {
                let head_len = self.capacity() - old_head;
                (head_len, target_cap - head_len)
            };
            unsafe {
                // can't use `copy_nonoverlapping()` here because the new and
                // old regions for the head might overlap.
                self.copy(
                    WrappedIndex::from_arbitrary_number(old_head),
                    WrappedIndex::from_arbitrary_number(new_head),
                    head_len,
                );
            }
            self.head = WrappedIndex::from_arbitrary_number(new_head);
        }

        // The compaction above moved elements out of the region that survives
        // the shrink, but if the reallocation fails — since
        // `Allocator::shrink` may panic or fail on memory exhaustion — the
        // deque must be restored to a consistent layout for its *original*
        // capacity before we return. This mirrors std's drop-guard +
        // `abort_shrink` pair (std #123369): the guard fires exactly when the
        // shrink did not complete.
        struct Guard<'a, T, A: Allocator> {
            deque: &'a mut VecDeque<T, A>,
            old_head: usize,
            target_cap: usize,
        }

        impl<T, A: Allocator> Drop for Guard<'_, T, A> {
            #[cold]
            fn drop(&mut self) {
                unsafe {
                    // SAFETY: this only runs if `try_shrink_to_fit` returned
                    // without completing the shrink (error or abort unwind),
                    // which is precisely when `abort_shrink` is safe to call.
                    self.deque.abort_shrink(self.old_head, self.target_cap)
                }
            }
        }

        let guard = Guard {
            deque: self,
            old_head,
            target_cap,
        };

        guard.deque.buf.try_shrink_to_fit(target_cap)?;

        // Don't drop the guard if we didn't unwind.
        core::mem::forget(guard);

        debug_assert!(self.head.as_index() < self.capacity() || self.capacity() == 0);
        debug_assert!(self.len <= self.capacity());
        Ok(())
    }

    /// Reverts the deque back into a consistent state in case
    /// [`Self::try_shrink_to`] failed.
    ///
    /// This is necessary to prevent UB if the backing allocator returns an
    /// error from `shrink` and the caller subsequently fails, panics:
    /// the compaction performed by `try_shrink_to` has
    /// already relocated elements for the *new* capacity, so the deque must be
    /// re-laid-out for the *old* capacity it still holds.
    ///
    /// `old_head` refers to the head index before `try_shrink_to` was called.
    /// `target_cap` is the capacity that it was trying to shrink to.
    ///
    /// # Safety
    ///
    /// Must only be called after `try_shrink_to`'s compaction ran but its
    /// reallocation did not complete, i.e. the buffer still has its original
    /// (pre-shrink) capacity.
    unsafe fn abort_shrink(&mut self, old_head: usize, target_cap: usize) {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted: len <= target_cap, caller precondition"
        )]
        if self.head.as_index() <= target_cap - self.len {
            // The deque's buffer is contiguous, so no need to copy anything
            // around.
            return;
        }

        // `try_shrink_to` already copied the head to fit into the new
        // capacity, so this won't overflow.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted: head < target_cap, caller precondition"
        )]
        let head_len = target_cap - self.head.as_index();
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "invariant: head_len <= len (buffer is valid for target_cap)"
        )]
        let tail_len = self.len - head_len;

        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted target_cap <= capacity, caller precondition"
        )]
        if tail_len <= core::cmp::min(head_len, self.capacity() - target_cap) {
            // There's enough spare capacity to copy the tail to the back
            // (because `tail_len < self.capacity() - target_cap`), and copying
            // the tail should be cheaper than copying the head (because
            // `tail_len <= head_len`).
            unsafe {
                // The old tail and the new tail can't overlap because the head
                // slice lies between them. The head slice ends at
                // `target_cap`, so that's where we copy to.
                self.copy_nonoverlapping(
                    WrappedIndex::zero(),
                    WrappedIndex::from_arbitrary_number(target_cap),
                    tail_len,
                );
            }
        } else {
            // Either there's not enough spare capacity to make the deque
            // contiguous, or the head is shorter than the tail (and therefore
            // hopefully cheaper to copy).
            unsafe {
                // The old and the new head slice can overlap, so we can't use
                // `copy_nonoverlapping` here.
                self.copy(
                    self.head,
                    WrappedIndex::from_arbitrary_number(old_head),
                    head_len,
                );
            }
            self.head = WrappedIndex::from_arbitrary_number(old_head);
        }
    }
}

// ---------------------------------------------------------------------------
// Fallible resize
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Resizes the deque so its length becomes `new_len`, filling any new slots
    /// with clones of `value`.
    ///
    /// If `new_len` is greater than the current length, the deque is extended by
    /// cloning `value` via [`TryClone`]. If smaller, it is truncated.
    ///
    /// # Errors
    ///
    /// Returns [`super::TryVecDequeWithCloneError`] on a reservation or
    /// clone failure. On a mid-loop clone failure the deque is rolled back to
    /// its original length so no partially-produced elements remain.
    // FIXME: Make TryVecDequeWithCloneError be in the common file
    pub fn try_resize(
        &mut self,
        new_len: usize,
        value: &T,
    ) -> Result<(), super::traits::TryVecDequeWithCloneError>
    where
        T: TryClone,
    {
        use super::traits::TryVecDequeWithCloneError;
        let current = self.len;
        if new_len <= current {
            self.truncate(new_len);
            return Ok(());
        }
        #[allow(clippy::arithmetic_side_effects, reason = "asserted new_len > current")]
        let extra = new_len - current;
        self.try_reserve(extra)?;
        // SAFETY: the guard is a local that drops before this function returns,
        // so `self` outlives it.
        let guard = unsafe { self.truncate_back_guard() };
        for _ in 0..extra {
            match value.try_clone() {
                Ok(cloned) => {
                    // SAFETY: capacity was reserved above for all `extra`.
                    unsafe { self.push_back_within_cap(cloned) };
                }
                Err(e) => {
                    return Err(TryVecDequeWithCloneError::Clone(e));
                }
            }
        }
        core::mem::forget(guard);
        Ok(())
    }

    /// Resizes the deque so its length becomes `new_len`, producing new
    /// elements with the fallible closure `f`.
    ///
    /// The closure is invoked only after capacity is secured. If it returns an
    /// error, the deque is truncated back to its original length so no
    /// partially-produced elements remain.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecDequeWithClosureError<E>`] if either the capacity
    /// reservation fails or the closure returns `Err(e)`.
    pub fn try_resize_with<E, F>(
        &mut self,
        new_len: usize,
        mut f: F,
    ) -> Result<(), TryVecDequeWithClosureError<E>>
    where
        F: FnMut() -> Result<T, E>,
    {
        let current = self.len;
        if new_len <= current {
            self.truncate(new_len);
            return Ok(());
        }
        #[allow(clippy::arithmetic_side_effects, reason = "asserted new_len > current")]
        let extra = new_len - current;
        self.try_reserve(extra)
            .map_err(TryVecDequeWithClosureError::Reserve)?;
        // SAFETY: the guard is a local that drops before this function returns,
        // so `self` outlives it.
        let guard = unsafe { self.truncate_back_guard() };
        for _ in 0..extra {
            match f() {
                Ok(item) => {
                    // SAFETY: capacity was reserved above for all `extra`.
                    unsafe { self.push_back_within_cap(item) };
                }
                Err(e) => {
                    // Guard drops here and truncates back to `current`.
                    return Err(TryVecDequeWithClosureError::Closure(e));
                }
            }
        }
        // Success: defuse the guard so it doesn't truncate the new elements.
        core::mem::forget(guard);
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
    use crate::collections::vec_deque::traits::TryVecDequeWithCloneError;
    use crate::test_helpers::{BudgetedAlloc, FailAlloc};
    use core::mem::size_of;
    use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

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

    // --- try_shrink_to / try_shrink_to_fit -------------------------------------

    #[test]
    fn shrink_to_reduces_capacity_and_preserves_elements() {
        let mut dq = VecDeque::<i32>::new();
        assert_eq!(dq.try_reserve_exact(8), Ok(()));
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.capacity(), 8);
        assert_eq!(dq.try_shrink_to(3), Ok(()));
        assert!(dq.capacity() < 8);
        assert!(dq.capacity() >= 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn shrink_to_below_len_clamps_to_len() {
        let mut dq = VecDeque::<i32>::new();
        assert_eq!(dq.try_reserve_exact(8), Ok(()));
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        // Requesting fewer than `len` clamps to `len`; all elements survive.
        assert_eq!(dq.try_shrink_to(1), Ok(()));
        assert_eq!(dq.len(), 3);
        assert!(dq.capacity() < 8);
        assert!(dq.capacity() >= 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn shrink_to_is_a_no_op_when_already_small_enough() {
        // Fill the deque to exactly its capacity so that any min_capacity
        // request yields target == len == capacity → no shrink needed.
        let mut dq = VecDeque::<i32>::new();
        assert_eq!(dq.try_reserve_exact(4), Ok(()));
        for v in [1, 2, 3, 4] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.capacity(), 4);
        assert_eq!(dq.len(), 4);
        // target = max(4, 2) = 4 == capacity → no-op.
        assert_eq!(dq.try_shrink_to(2), Ok(()));
        assert_eq!(dq.capacity(), 4);
        assert_eq!(collect_into_array::<4>(&dq), Some([1, 2, 3, 4]));
    }

    #[test]
    fn shrink_deeply_wrapped_state() {
        // Pin capacity to exactly 5, fill it, then pop from the front to
        // advance `head` into a wrapped position.
        let mut dq = VecDeque::<i32>::new();
        assert_eq!(dq.try_reserve_exact(5), Ok(()));
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.capacity(), 5);
        // Pop 4 from the front; `head` advances to slot 4, leaving [5] at
        // physical slot 4 — a non-zero head with a single element.
        assert_eq!(dq.pop_front(), Some(1));
        assert_eq!(dq.pop_front(), Some(2));
        assert_eq!(dq.pop_front(), Some(3));
        assert_eq!(dq.pop_front(), Some(4));
        assert_eq!(dq.len(), 1);
        assert_eq!(dq.front(), Some(&5));
        // Shrink to fit; the lone element must survive compaction + realloc.
        assert_eq!(dq.try_shrink_to_fit(), Ok(()));
        assert_eq!(dq.len(), 1);
        assert_eq!(dq.front(), Some(&5));
        assert_eq!(dq.back(), Some(&5));
    }

    #[test]
    fn shrink_to_zero_clamps_to_len() {
        // `try_shrink_to(0)` clamps to `len` (max(3, 0) = 3), so the buffer
        // shrinks to fit exactly 3 elements rather than deallocating entirely.
        let mut dq = VecDeque::<i32>::new();
        assert_eq!(dq.try_reserve_exact(8), Ok(()));
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert!(!dq.is_empty());
        assert_eq!(dq.try_shrink_to(0), Ok(()));
        assert_eq!(dq.len(), 3);
        // Capacity should now be close to 3 (may slightly exceed due to
        // allocator granularity, but must be well under the original 8).
        assert!(dq.capacity() < 8);
        assert!(dq.capacity() >= 3);
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn shrink_empty_deque_to_zero_deallocates() {
        // An empty deque has len=0, so try_shrink_to(0) targets 0 and
        // deallocates the buffer entirely.
        let mut dq = VecDeque::<i32>::new();
        assert_eq!(dq.try_reserve_exact(8), Ok(()));
        assert_eq!(dq.capacity(), 8);
        assert!(dq.is_empty());
        assert_eq!(dq.try_shrink_to(0), Ok(()));
        assert_eq!(dq.capacity(), 0);
        assert!(dq.is_empty());
    }

    #[test]
    fn shrink_to_fit_matches_length() {
        let mut dq = VecDeque::<i32>::new();
        assert_eq!(dq.try_reserve_exact(16), Ok(()));
        for v in [7, 8] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.capacity(), 16);
        assert_eq!(dq.try_shrink_to_fit(), Ok(()));
        assert!(dq.capacity() < 16);
        assert!(dq.capacity() >= 2);
        assert_eq!(collect_into_array::<2>(&dq), Some([7, 8]));
    }

    #[test]
    fn shrink_zst_is_a_noop() {
        let mut dq: VecDeque<()> = VecDeque::new();
        for _ in 0..10 {
            assert_eq!(dq.try_push_back(()), Ok(()));
        }
        // ZSTs never hold physical memory; shrinking is a harmless no-op.
        assert_eq!(dq.try_shrink_to(1), Ok(()));
        assert_eq!(dq.try_shrink_to_fit(), Ok(()));
        assert_eq!(dq.len(), 10);
        assert_eq!(dq.capacity(), usize::MAX);
    }

    #[test]
    fn shrink_oom_leaves_deque_usable() {
        // BudgetedAlloc's single allocation was spent by `try_with_capacity_in`,
        // so the reallocation inside `try_shrink_to` must fail. The deque must
        // remain fully usable at its original (larger) capacity afterwards.
        let mut dq = VecDeque::<i32, _>::try_with_capacity_in(4, BudgetedAlloc::new(1))
            .expect("within budget");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        let err = match dq.try_shrink_to(2) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        // Elements intact, length unchanged, and the buffer kept its old size.
        assert_eq!(dq.len(), 3);
        assert_eq!(collect_any::<3, BudgetedAlloc>(&dq), Some([1, 2, 3]));
        assert!(dq.capacity() == 4);
    }

    #[test]
    fn shrink_then_push_again_works() {
        // Exercise the full lifecycle: grow, shrink, then grow again on top of
        // the compacted buffer.
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.try_shrink_to(2), Ok(()));
        // Push more onto the shrunk buffer; it should grow as needed.
        for v in [5, 6, 7] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.len(), 7);
        assert_eq!(collect_into_array::<7>(&dq), Some([1, 2, 3, 4, 5, 6, 7]));
    }

    #[test]
    fn shrink_failure_restores_wrapped_layout() {
        // Build a genuinely wrapped deque under an allocator that funds only
        // the initial buffer, so every subsequent reallocation fails.
        //
        // Sequence: fill cap-5 buffer → pop 3 → push 1 (wraps around).
        // Result: head=3, len=3, cap=5, elements [4,5,6] at slots 3,4,0.
        // `try_shrink_to(3)` targets cap=3 (= max(3, len)), forcing compaction
        // of the wrapped element before the (failing) reallocation.
        let mut dq = VecDeque::<i32, _>::try_with_capacity_in(5, BudgetedAlloc::new(1))
            .expect("initial allocation within budget");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        // Pop 3 from the front: head advances to slot 3, len drops to 2.
        for _ in 0..3 {
            assert!(dq.pop_front().is_some());
        }
        // Push one more: wraps around to slot 0. Now head=3, len=3, cap=5.
        assert_eq!(dq.try_push_back(6), Ok(()));
        assert_eq!(dq.len(), 3);
        assert!(!dq.is_contiguous());

        // Shrinking to 3 (== len) forces compaction of the wrapped element
        // into the front of the buffer *before* the (failing) reallocation.
        // The drop guard must undo that compaction against the original buffer.
        let err = match dq.try_shrink_to(3) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());

        // The deque must be fully consistent again: same elements, same order,
        // same capacity — ready for another operation. Note: `abort_shrink`
        // restores a *valid* layout, not necessarily the exact pre-shrink
        // physical arrangement; contiguity may differ.
        assert_eq!(dq.len(), 3);
        assert_eq!(dq.capacity(), 5);
        assert_eq!(collect_any::<3, BudgetedAlloc>(&dq), Some([4, 5, 6]));
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
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [10, 20, 30] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        // Append a new element and mutate it through the returned reference;
        // the pre-existing elements must be untouched.
        let slot = match dq.try_push_back_mut(99) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *slot *= 2;
        assert_eq!(collect_into_array::<4>(&dq), Some([10, 20, 30, 198]));
    }

    #[test]
    fn push_front_mut_returns_reference_to_prepended_element() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [10, 20, 30] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        // Prepend a new element and mutate it through the returned reference;
        // the pre-existing elements must be untouched.
        let slot = match dq.try_push_front_mut(-5) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *slot *= 3;
        assert_eq!(collect_into_array::<4>(&dq), Some([-15, 10, 20, 30]));
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
        // Build a full wrapped state: [4, 1, 2, 3] spanning two physical
        // segments (head retreated past 0 by the push_front).
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.try_push_front(4), Ok(()));
        assert_eq!(dq.len(), 4);
        // Deque is full; both directions must grow internally while the
        // elements are still wrapped across the buffer boundary. Mutate each
        // new element through its returned reference; the four pre-existing
        // elements must be untouched.
        let back_slot = match dq.try_push_back_mut(9) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *back_slot += 1;
        // Drop the mutable borrow before growing again from the front.
        let front_slot = match dq.try_push_front_mut(8) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *front_slot -= 1;
        assert!(dq.capacity() > 4);
        assert_eq!(dq.len(), 6);
        assert_eq!(collect_into_array::<6>(&dq), Some([7, 4, 1, 2, 3, 10]));
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
        for v in [1, 3, 5] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        // Insert in the middle and mutate through the returned reference; the
        // surrounding elements must be untouched.
        let slot = match dq.try_insert_mut(1, 2) {
            Ok(r) => r,
            Err(_) => panic!("expected success"),
        };
        *slot += 10;
        assert_eq!(collect_into_array::<4>(&dq), Some([1, 12, 3, 5]));
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

    // --- try_resize ------------------------------------------------------------

    #[test]
    fn try_resize_shrink_truncates() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.try_resize(3, &99), Ok(()));
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
        assert_eq!(dq.len(), 3);
    }

    #[test]
    fn try_resize_same_length_is_noop() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        assert_eq!(dq.try_resize(3, &99), Ok(()));
        assert_eq!(collect_into_array::<3>(&dq), Some([1, 2, 3]));
    }

    #[test]
    fn try_resize_grow_appends_clones_within_capacity() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        assert_eq!(dq.try_resize(5, &42), Ok(()));
        assert_eq!(collect_into_array::<5>(&dq), Some([1, 2, 42, 42, 42]));
    }

    #[test]
    fn try_resize_grow_from_empty() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        assert_eq!(dq.try_resize(4, &7), Ok(()));
        assert_eq!(collect_into_array::<4>(&dq), Some([7, 7, 7, 7]));
    }

    #[test]
    fn try_resize_oom_leaves_original_intact() {
        // Budget of 1 spends on the initial allocation; any growth must fail.
        let mut dq = VecDeque::<i32, _>::try_with_capacity_in(2, BudgetedAlloc::new(1))
            .expect("initial allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        let res = dq.try_resize(5, &9);
        assert!(matches!(res, Err(TryVecDequeWithCloneError::Reserve(_))));
        // Original elements survive unchanged.
        assert_eq!(collect_any::<2, BudgetedAlloc>(&dq), Some([1, 2]));
        assert_eq!(dq.len(), 2);
    }

    /// A value whose `try_clone` always fails, used to exercise the resize
    /// rollback path deterministically.
    #[derive(Debug, Clone, Copy)]
    #[allow(unused)]
    struct FailingClone(i32);

    impl TryClone for FailingClone {
        fn try_clone(&self) -> Result<Self, TryCloneError> {
            Err(TryCloneError::Other("always fails"))
        }
    }

    #[test]
    fn try_resize_clone_failure_rolls_back_to_original() {
        let mut dq: VecDeque<FailingClone> = VecDeque::new();
        assert_eq!(dq.try_push_back(FailingClone(1)), Ok(()));
        assert_eq!(dq.try_push_back(FailingClone(2)), Ok(()));
        let res = dq.try_resize(5, &FailingClone(9));
        assert!(matches!(res, Err(TryVecDequeWithCloneError::Clone(_))));
        // Rolled back to the original length of 2.
        assert_eq!(dq.len(), 2);
    }

    // --- try_resize_with -------------------------------------------------------

    #[test]
    fn try_resize_with_shrink_truncates_without_calling_closure() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back(v), Ok(()));
        }
        let mut calls = 0usize;
        let res = dq.try_resize_with(2, || {
            calls += 1;
            Ok::<i32, ()>(99)
        });
        assert_eq!(res, Ok(()));
        assert_eq!(calls, 0, "closure must not run when shrinking");
        assert_eq!(collect_into_array::<2>(&dq), Some([1, 2]));
    }

    #[test]
    fn try_resize_with_grow_invokes_closure_per_slot() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        let mut counter = 1i32;
        let res = dq.try_resize_with(4, || {
            counter += 1;
            Ok::<i32, ()>(counter)
        });
        assert_eq!(res, Ok(()));
        // New slots are filled in order: 2, 3, 4.
        assert_eq!(collect_into_array::<4>(&dq), Some([1, 2, 3, 4]));
    }

    #[test]
    fn try_resize_with_oom_returns_reserve_error() {
        let mut dq = VecDeque::<i32, _>::try_with_capacity_in(2, BudgetedAlloc::new(1))
            .expect("initial allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        let res = dq.try_resize_with(6, || Ok::<i32, ()>(0));
        assert!(matches!(res, Err(TryVecDequeWithClosureError::Reserve(_))));
        assert_eq!(collect_any::<2, BudgetedAlloc>(&dq), Some([1, 2]));
        assert_eq!(dq.len(), 2);
    }

    #[test]
    fn try_resize_with_closure_error_rolls_back() {
        let mut dq = VecDeque::<i32>::try_with_capacity(8).expect("allocation ok");
        assert_eq!(dq.try_push_back(1), Ok(()));
        assert_eq!(dq.try_push_back(2), Ok(()));
        let mut seen = 0usize;
        let res = dq.try_resize_with(5, || {
            seen += 1;
            if seen == 2 {
                Err("boom")
            } else {
                Ok(seen as i32)
            }
        });
        assert!(matches!(res, Err(TryVecDequeWithClosureError::Closure(_))));
        // Rolled back to the original length of 2.
        assert_eq!(dq.len(), 2);
        assert_eq!(collect_into_array::<2>(&dq), Some([1, 2]));
    }

    #[test]
    fn try_resize_zst() {
        let mut dq: VecDeque<()> = VecDeque::new();
        assert_eq!(dq.try_resize(3, &()), Ok(()));
        assert_eq!(dq.len(), 3);
        assert_eq!(dq.try_resize(1, &()), Ok(()));
        assert_eq!(dq.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Panic-safety regression tests
    // -----------------------------------------------------------------------

    use crate::test_helpers::{CloneBudget, Ledger, PanicArmer};
    use std::sync::Arc;

    /// A tracked item whose `try_clone` is gated by a shared [`CloneBudget`]
    /// and whose destructor panics exactly once (while the armer is armed).
    /// This lets us deterministically trigger a mid-resize clone failure AND
    /// a panicking destructor during the subsequent rollback.
    struct FlakyPanicTracked {
        id: u32,
        ledger: Arc<Ledger>,
        armer: Arc<PanicArmer>,
        budget: CloneBudget,
    }

    impl Drop for FlakyPanicTracked {
        fn drop(&mut self) {
            self.ledger.unregister(self.id);
            if self.armer.is_armed() {
                self.armer.disarm();
                panic!("forced panic in drop");
            }
        }
    }

    impl TryClone for FlakyPanicTracked {
        fn try_clone(&self) -> Result<Self, TryCloneError> {
            if !self.budget.try_consume() {
                return Err(TryCloneError::Other("budget exhausted"));
            }
            let id = self.ledger.allocate();
            self.ledger.register(id);
            Ok(FlakyPanicTracked {
                id,
                ledger: self.ledger.clone(),
                armer: self.armer.clone(),
                budget: self.budget.share(),
            })
        }
    }

    /// If a destructor in the rolled-back tail panics during `try_resize`'s 
    /// truncation, no element may be double-freed or leaked.
    #[test]
    fn try_resize_rollback_panicking_drop_is_safe() {
        let ledger = Arc::new(Ledger::new());
        let armer = Arc::new(PanicArmer::new());

        armer.arm();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let l = ledger.clone();
            let a = armer.clone();
            move || {
                let budget = CloneBudget::new(2); // allow exactly 2 clones
                let mut dq: VecDeque<FlakyPanicTracked> = VecDeque::new();
                // Seed 2 elements.
                for _ in 0..2 {
                    let id = l.allocate();
                    l.register(id);
                    dq.try_push_back(FlakyPanicTracked {
                        id,
                        ledger: l.clone(),
                        armer: a.clone(),
                        budget: budget.share(),
                    })
                    .unwrap();
                }
                // Source: its own registered id; the *clones* drawn from it
                // allocate fresh ids beyond the seeds'.
                let source_id = l.allocate();
                l.register(source_id);
                let source = FlakyPanicTracked {
                    id: source_id,
                    ledger: l.clone(),
                    armer: a.clone(),
                    budget: budget.share(),
                };
                // Resize from 2 → 5: needs 3 clones, only 2 succeed.
                // The 3rd clone fails → guard truncates back to len 2,
                // dropping the 2 cloned elements. First drop panics.
                let _ = dq.try_resize(5, &source);
            }
        }));

        // A panic should have occurred (either from the drop or propagated).
        assert!(result.is_err(), "expected a panic from the drop");
        // All registered elements (seed ids 0,1,2 + cloned ids 3,4) must each
        // be dropped exactly once. No double-frees, no leaks.
        // All registered elements (seed ids 0,1,2 + cloned ids 3,4) must each
        // be dropped exactly once. No double-frees, no leaks.
        assert!(ledger.double_dropped().is_empty(), "double-free detected");
        assert!(ledger.leaked_ids().is_empty(), "leak detected");
    }

    /// Regression test: if the closure in `try_resize_with` panics after some
    /// elements have been pushed, the `TruncateBackGuard` rolls the deque back
    /// to its original length.
    #[test]
    fn try_resize_with_closure_panic_is_safe() {
        use crate::test_helpers::TrackedItem;

        let ledger = Arc::new(Ledger::new());

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let l = ledger.clone();
            move || {
                let mut dq: VecDeque<TrackedItem<()>> = VecDeque::new();
                // Seed one element (id 0).
                let seed_id = l.allocate();
                l.register(seed_id);
                dq.try_push_back(TrackedItem {
                    id: seed_id,
                    ledger: l.clone(),
                    inner: (),
                })
                .unwrap();

                // Resize to 4: the closure succeeds twice (minting ids 1 and 2),
                // then panics on the third call. The guard rolls the two pushed
                // elements back out.
                let mut calls = 0usize;
                let _: Result<(), TryVecDequeWithClosureError<&str>> =
                    dq.try_resize_with(4, || {
                        calls += 1;
                        if calls == 3 {
                            panic!("forced panic mid-resize_with");
                        }
                        let id = l.allocate();
                        l.register(id);
                        Ok(TrackedItem {
                            id,
                            ledger: l.clone(),
                            inner: (),
                        })
                    });
            }
        }));

        assert!(result.is_err(), "expected the closure to panic");
        // All three minted ids (the seed plus the two from the closure) must
        // each be dropped exactly once — no double-free, no leak.
        assert!(ledger.double_dropped().is_empty(), "double-free detected");
        assert!(ledger.leaked_ids().is_empty(), "leak detected");
        assert!(ledger.all_dropped_once(0..3u32));
    }

    /// Regression test: `try_shrink_to_fit` performs raw-pointer compaction
    /// (no destructors run during the move). If the shrink reallocation
    /// itself fails, the guard restores a consistent layout via `abort_shrink`.
    #[test]
    fn try_shrink_to_fit_failure_restores_wrapped_state() {
        use crate::test_helpers::BudgetedAlloc;

        // Budget of 1: the single `try_reserve` below consumes it to create
        // the initial buffer. All subsequent pushes/pops fit within that
        // capacity without allocating. The shrink's internal `allocate` then
        // finds the budget exhausted and fails.
        let alloc = BudgetedAlloc::new(1);
        let mut dq: VecDeque<i32, BudgetedAlloc> = VecDeque::new_in(alloc);

        // Allocate the initial buffer (consumes the only unit of budget).
        dq.try_reserve(8).unwrap();
        assert_eq!(dq.capacity(), 8);

        // Fill the whole buffer: slots 0..8 hold [1..=8].
        for v in 1..=8i32 {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        // Pop four from the front: advances head to 4, leaving [5,6,7,8] in
        // slots 4..8.
        for expected in 1..=4i32 {
            assert_eq!(dq.pop_front(), Some(expected));
        }
        // Push two more: they wrap around into slots 0,1. Now the live
        // elements [5,6,7,8,9,10] occupy slots 4,5,6,7,0,1 — a wrapped layout
        // (head=4, len=6, head+len=10 > capacity=8).
        assert_eq!(dq.try_push_back_within_capacity(9), Ok(()));
        assert_eq!(dq.try_push_back_within_capacity(10), Ok(()));
        assert_eq!(dq.len(), 6);
        // Confirm the wrapped layout: `as_slices` returns two non-empty halves
        // when elements straddle the buffer boundary.
        let (front, back) = dq.as_slices();
        assert_eq!(front, &[5, 6, 7, 8]);
        assert_eq!(back, &[9, 10]);

        // Shrink-to-fit will attempt a reallocation which fails because the
        // budget is exhausted. The deque must be restored to its exact
        // pre-shrink wrapped state.
        let res = dq.try_shrink_to_fit();
        assert!(res.is_err(), "expected shrink to fail due to OOM");

        // Length and capacity are unchanged, and the full logical element
        // order is preserved across the buffer boundary. `abort_shrink` may
        // have re-laid-out the elements contiguously (it copies the cheaper
        // side), so we concatenate both halves rather than asserting a
        // specific split.
        assert_eq!(dq.len(), 6);
        assert_eq!(dq.capacity(), 8);
        let (front, back) = dq.as_slices();
        let mut restored = front.iter().chain(back.iter()).copied();
        assert_eq!(restored.next(), Some(5));
        assert_eq!(restored.next(), Some(6));
        assert_eq!(restored.next(), Some(7));
        assert_eq!(restored.next(), Some(8));
        assert_eq!(restored.next(), Some(9));
        assert_eq!(restored.next(), Some(10));
        assert_eq!(restored.next(), None);
    }
}
