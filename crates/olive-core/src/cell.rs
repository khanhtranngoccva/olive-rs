//! Re-exports of [`core::cell`] types and foundational-trait impls
//! for these types.
//!
//! The covered cells include [`Cell`], [`RefCell`], and
//! [`UnsafeCell`].
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

// Copy bound here is a std bound.
impl<T: Copy> TryClone for Cell<T> {
    /// Infallibly clone a [`Cell`], producing an independent cell holding a clone of
    /// the current value.
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let value = self.get();
        Ok(Cell::new(value))
    }
}

// A `Cell` holds its payload inline with no allocation.
impl<T: TryDefault> TryDefault for Cell<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Cell::new(T::try_default()?))
    }
}

// ---------------------------------------------------------------------------
// LazyCell
// ---------------------------------------------------------------------------

// NOTE: [`TryDefault`] is deliberately NOT implemented for [`LazyCell`].
//
// This cell type is built with the infallible lazy pattern in mind,
// so it is unsuitable for the fallible pattern.

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
// default construction is infallible.
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
    /// if cloning the value fails.
    ///
    /// This generally should not trigger, however (as an error here is usually
    /// indicative of a bug).
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

// A `RefCell` holds its payload inline with no allocation.
impl<T: TryDefault> TryDefault for RefCell<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(RefCell::new(T::try_default()?))
    }
}

// ---------------------------------------------------------------------------
// UnsafeCell
// ---------------------------------------------------------------------------

// An `UnsafeCell` holds its payload inline with no allocation.
impl<T: TryDefault> TryDefault for UnsafeCell<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(UnsafeCell::new(T::try_default()?))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::testing::{MaybeUnclonable, NoDefault};

    // -----------------------------------------------------------------------
    // Cell
    // -----------------------------------------------------------------------

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
    fn cell_try_clone_copies_nested_copy_struct() {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        struct Point {
            x: u8,
            y: u8,
        }
        let c = Cell::new(Point { x: 1, y: 2 });
        let cloned = c.try_clone().unwrap();
        assert_eq!(cloned.get(), Point { x: 1, y: 2 });
        // Mutating the clone must not affect the original.
        cloned.set(Point { x: 3, y: 4 });
        assert_eq!(c.get(), Point { x: 1, y: 2 });
    }

    #[test]
    fn cell_try_default_produces_default_payload() {
        let c = Cell::<u32>::try_default().unwrap();
        assert_eq!(c.get(), 0);
        let s = Cell::<bool>::try_default().unwrap();
        assert!(!s.get());
    }

    #[test]
    fn cell_try_default_propagates_inner_failure() {
        assert!(matches!(
            Cell::<NoDefault<u8>>::try_default(),
            Err(TryDefaultError::Other("no canonical value"))
        ));
    }

    // -----------------------------------------------------------------------
    // OnceCell
    // -----------------------------------------------------------------------

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
        let mut cloned = oc.try_clone().unwrap();
        assert_eq!(cloned.get(), Some(&42));
        *cloned.get_mut().unwrap() = 43;
        assert_eq!(cloned.get(), Some(&43));
        assert_eq!(oc.get(), Some(&42));
    }

    #[test]
    fn once_cell_try_clone_propagates_failure_and_leaves_source_intact() {
        let oc = OnceCell::new();
        oc.set(MaybeUnclonable::unclonable(1u32)).unwrap();
        assert!(matches!(
            oc.try_clone(),
            Err(TryCloneError::Other("unclonable"))
        ));
        assert_eq!(oc.get().unwrap().inner, 1);
    }

    #[test]
    fn once_cell_try_default_is_empty() {
        let oc: OnceCell<u32> = OnceCell::try_default().unwrap();
        assert!(oc.get().is_none());
    }

    #[test]
    fn once_cell_try_default_takes_no_payload_bounds() {
        // This struct has no TryDefault implementation, which stresses
        // the ability to invoke TryDefault.
        struct Plain;
        let oc = OnceCell::<Plain>::try_default().unwrap();
        assert!(oc.get().is_none());
    }

    // -----------------------------------------------------------------------
    // RefCell
    // -----------------------------------------------------------------------

    #[test]
    fn refcell_try_clone_produces_independent_copy() {
        let r = RefCell::new(1u32);
        let cloned = r.try_clone().unwrap();
        assert_eq!(*cloned.borrow(), 1);
        *cloned.borrow_mut() = 2;
        assert_eq!(*r.borrow(), 1);
        assert_eq!(*cloned.borrow(), 2);
    }

    #[test]
    fn refcell_try_clone_succeeds_while_shared_borrowed() {
        let r = RefCell::new(1u32);
        let shared = r.borrow();
        let cloned = r
            .try_clone()
            .expect("a shared borrow must not block cloning");
        assert_eq!(*cloned.borrow(), 1);
        drop(shared);
    }

    #[test]
    fn refcell_try_clone_contention_reports_other() {
        let r = RefCell::new(42u8);
        let _mut_guard = r.try_borrow_mut().unwrap();
        // The contention failure must be the documented `Other` variant with
        // the documented message, not some other error shape.
        assert!(matches!(
            r.try_clone(),
            Err(TryCloneError::Other("RefCell is already mutably borrowed"))
        ));
    }

    #[test]
    fn failing_inner_refcell_leaves_source_untouched() {
        let r = RefCell::new(MaybeUnclonable::unclonable(1u32));
        assert!(matches!(
            r.try_clone(),
            Err(TryCloneError::Other("unclonable"))
        ));
        // The source cell is undisturbed: it can still be borrowed and read.
        let still_readable = r.try_borrow().expect("source cell must remain borrowable");
        drop(still_readable);
        r.borrow_mut().inner = 2;
        assert_eq!(r.borrow().inner, 2);
    }

    #[test]
    fn refcell_try_default_wraps_default_payload() {
        let r = RefCell::<u64>::try_default().unwrap();
        assert_eq!(*r.borrow(), 0);
    }

    #[test]
    fn refcell_try_default_propagates_inner_failure() {
        assert!(matches!(
            RefCell::<NoDefault<u8>>::try_default(),
            Err(TryDefaultError::Other("no canonical value"))
        ));
    }

    // -----------------------------------------------------------------------
    // UnsafeCell
    // -----------------------------------------------------------------------

    #[test]
    fn unsafe_cell_try_default_wraps_default_payload() {
        let c = UnsafeCell::<u32>::try_default().unwrap();
        // Reading through `get` is safe here: no aliasing references exist.
        assert_eq!(unsafe { *c.get() }, 0);
        let b = UnsafeCell::<bool>::try_default().unwrap();
        assert!(!unsafe { *b.get() });
    }

    #[test]
    fn unsafe_cell_try_default_propagates_inner_failure() {
        assert!(matches!(
            UnsafeCell::<NoDefault<u8>>::try_default(),
            Err(TryDefaultError::Other("no canonical value"))
        ));
    }
}
