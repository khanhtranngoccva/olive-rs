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
use crate::try_traits::try_default::{TryDefault, TryDefaultError};

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

// An empty `Cell` holds its payload inline with no allocation; when the payload
// has a canonical default, construction is infallible.
impl<T: TryDefault> TryDefault for Cell<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Cell::new(T::try_default()?))
    }
}

// ---------------------------------------------------------------------------
// LazyCell
// ---------------------------------------------------------------------------

// A `LazyCell` defers its payload until first dereference, so a fallible
// initializer is a natural fit: unlike [`Cell`] or [`RefCell`], whose payloads
// are constructed eagerly at build time, here construction only *stores* the
// closure and evaluation happens later. That lets us bound on [`TryDefault`]
// rather than the infallible [`Default`]: a `T` whose canonical value may fail
// to construct can still be lazily initialized, surfacing the failure at first
// access instead of at build time. The stored closure returns
// `Ok(T)` directly, so the cell resolves to `T` with no extra wrapping.
impl<T: TryDefault> TryDefault for LazyCell<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(LazyCell::new(|| {
            T::try_default().expect("lazy default failed")
        }))
    }
}

// ---------------------------------------------------------------------------
// OnceCell
// ---------------------------------------------------------------------------

// Cloning an `OnceCell` reads its current state (initialized or not) and
// produces an independent cell in the same state. If initialized, the inner
// value is cloned via `TryClone`; if empty, the clone is also empty.
impl<T: TryClone> TryClone for OnceCell<T> {
    /// Fallibly clone an [`OnceCell`], preserving whether it is initialized.
    ///
    /// # Errors
    ///
    /// Propagates the inner value's [`TryCloneError`] if the cell is initialized
    /// and cloning its payload fails. An empty cell clones without error.
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let out = Self::new();
        if let Some(value) = self.get() {
            let cloned = value.try_clone()?;
            // `set` returns `Err` only if already initialized, which cannot
            // happen on a freshly-constructed cell.
            out.set(cloned).map_err(|_| {
                TryCloneError::Other("OnceCell::try_clone: internal set failed unexpectedly")
            })?;
        }
        Ok(out)
    }
}

// An empty `OnceCell` carries no payload and performs no allocation, so its
// default construction is infallible — mirroring std's `OnceCell::new`.
impl<T> TryDefault for OnceCell<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new())
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

// A `RefCell` with a default payload performs no allocation at construction
// time — the value is stored inline. The fallible channel exists for symmetry
// with other container types whose "empty" state may still allocate.
impl<T: TryDefault> TryDefault for RefCell<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(RefCell::new(T::try_default()?))
    }
}

// ---------------------------------------------------------------------------
// UnsafeCell
// ---------------------------------------------------------------------------
//

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

    #[test]
    fn cell_try_default_produces_default_payload() {
        let c = Cell::<u32>::try_default().unwrap();
        assert_eq!(c.get(), 0);
        let s = Cell::<bool>::try_default().unwrap();
        assert!(!s.get());
    }

    #[test]
    fn lazy_cell_try_default_defers_evaluation() {
        let lazy = LazyCell::<u32>::try_default().unwrap();
        // The lazy cell is constructed without evaluating; first deref triggers
        // the initializer which yields `T::default()`.
        assert_eq!(*lazy, 0);
    }

    #[test]
    fn once_cell_try_clone_empty() {
        let oc: OnceCell<u32> = OnceCell::new();
        let cloned = oc.try_clone().unwrap();
        assert!(cloned.get().is_none());
    }

    #[test]
    fn once_cell_try_clone_initialized() {
        let oc = OnceCell::new();
        oc.set(42).unwrap();
        let cloned = oc.try_clone().unwrap();
        assert_eq!(cloned.get(), Some(&42));
        // Independent: mutating the clone doesn't affect the original.
        // (OnceCell values are immutable once set, so independence is
        // structural — they're separate allocations.)
        assert_eq!(oc.get(), Some(&42));
    }

    #[test]
    fn once_cell_try_default_is_empty() {
        let oc: OnceCell<u32> = OnceCell::try_default().unwrap();
        assert!(oc.get().is_none());
    }

    #[test]
    fn refcell_try_default_wraps_default_payload() {
        let r = RefCell::<u64>::try_default().unwrap();
        assert_eq!(*r.borrow(), 0);
    }
}
