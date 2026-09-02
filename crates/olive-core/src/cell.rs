//! Re-exports of [`core::cell`] types and foundational-trait impls
//! for these types.
//!
//! These are the fallible analogues of the standard library's infallible-by-
//! assumption traits applied to interior-mutability cells:
//!
//! * [`TryClone`] — a fallible analogue of [`core::clone::Clone`]. Cloning a
//!   cell reads its current value and clones it into a fresh, independent cell.
//!
//! Both cells live in `core`, so they ride along on this crate's glob re-export
//! of `core::*`; only their Olive trait impls are added here.
//!
//! # Semantics
//!
//! * **Independent copies.** Like std's `Clone` for `Cell`/`RefCell`, cloning
//!   produces a brand-new cell holding a clone of the *current* value. The new
//!   cell shares no state with the original; mutating one never affects the
//!   other.
//! * **Failure leaves `self` untouched.** A failed clone (e.g. the inner value
//!   ran out of memory) returns an error and does not disturb the source cell.
//! * **Borrow-checker independence.** Reading a `RefCell`'s value via
//!   [`RefCell::try_borrow`] can fail if the cell is already mutably borrowed;
//!   that failure surfaces as [`TryCloneError::Other`] rather than a panic, so a
//!   contended borrow degrades gracefully.

// Re-export the whole `core::cell` surface so this shadowing module stays a
// drop-in superset.
pub use core::cell::*;

use crate::try_traits::try_clone::{TryClone, TryCloneError};

// FIXME: missing UnsafeCell, BorrowError, BorrowMutError

// ---------------------------------------------------------------------------
// Cell
// ---------------------------------------------------------------------------

// Bounds are intentionally lax - since Copy never fails because it is meant 
// to represent bitwise copies, framework invariant is still honored. 
impl<T: Copy> TryClone for Cell<T> {
    /// Fallibly clone a [`Cell`], producing an independent cell holding a clone of
    /// the current value.
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let value = self.get();
        Ok(Cell::new(value))
    }
}

// ---------------------------------------------------------------------------
// RefCell
// ---------------------------------------------------------------------------

impl<T: TryClone> TryClone for RefCell<T> {
    /// Fallibly clone a [`RefCell`], producing an independent cell holding a clone
    /// of the current value.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError::Other`] if the cell is currently mutably borrowed
    /// (so its value cannot be read), or propagates the inner value's [`TryCloneError`]
    /// if cloning the value fails. This generally should not trigger, however.
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // `try_borrow` returns `Err(BorrowError)` when the cell is already
        // mutably borrowed. Surface that contention as a logic-level failure
        // rather than panicking.
        let Ok(guard) = self.try_borrow() else {
            return Err(TryCloneError::Other("RefCell is already mutably borrowed"));
        };
        let value = (*guard).try_clone()?;
        // Drop the borrow before constructing the new cell so we don't hold a
        // shared borrow across the (potentially allocating) construction.
        drop(guard);
        Ok(RefCell::new(value))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    /// A small non-`Copy` type implementing [`TryClone`] infallibly, used to
    /// exercise the `RefCell` borrow-then-clone path (which works for non-copy
    /// payloads because `RefCell` borrows its value).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Payload(u32);
    impl TryClone for Payload {
        fn try_clone(&self) -> Result<Self, TryCloneError> {
            Ok(*self)
        }
    }

    #[test]
    fn cell_try_clone_produces_independent_copy() {
        let c = Cell::new(7u32);
        let cloned = c.try_clone().unwrap();
        assert_eq!(cloned.get(), 7);
        // Mutating the clone must not affect the original.
        cloned.set(9);
        assert_eq!(c.get(), 7);
        assert_eq!(cloned.get(), 9);
    }

    #[test]
    fn refcell_try_clone_produces_independent_copy() {
        let r = RefCell::new(Payload(1));
        let cloned = r.try_clone().unwrap();
        assert_eq!(*cloned.borrow(), Payload(1));
        *cloned.borrow_mut() = Payload(2);
        assert_eq!(*r.borrow(), Payload(1));
        assert_eq!(*cloned.borrow(), Payload(2));
    }

    #[test]
    fn refcell_try_clone_fails_when_mutably_borrowed() {
        let r = RefCell::new(42u8);
        let _mut_guard = r.try_borrow_mut().unwrap();
        // While mutably borrowed, a clone must fail cleanly, not panic.
        let res = r.try_clone();
        assert!(res.is_err());
        // The source cell is undisturbed.
        drop(_mut_guard);
        assert_eq!(*r.borrow(), 42);
    }

    #[test]
    fn failing_inner_refcell_leaves_source_untouched() {
        struct Failing;
        impl TryClone for Failing {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                Err(TryCloneError::Other("always fails"))
            }
        }
        let r = RefCell::new(Failing);
        let res = r.try_clone();
        assert!(res.is_err());
        // The source cell is undisturbed: it can still be borrowed and read.
        let _still_readable = r.try_borrow().expect("source cell must remain borrowable");
    }
}
