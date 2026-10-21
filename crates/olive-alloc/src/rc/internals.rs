use super::TryRcOutOfBoundsError;
use core::cell::Cell;

// ---------------------------------------------------------------------------
// Shared internals
// ---------------------------------------------------------------------------

/// Internal representation of an `Rc` allocation.
///
/// This mirrors the std internal layout: two `usize` counters followed by the
/// payload, allocated as a single block. The `align(2)` attribute ensures that
/// on exotic targets where `usize` has 1-byte alignment (e.g. AVR), the two
/// counters are at least 2-aligned so they cannot overlap with an odd-aligned
/// payload. On normal targets this is a no-op since `usize` is already
/// ≥ 2-aligned.
///
/// Using `#[repr(C)]` guarantees the field order is predictable: `strong` at
/// offset 0, `weak` at offset `size_of::<usize>()`, and `value` after that.
#[repr(C, align(2))]
pub(super) struct RcInner<T: ?Sized> {
    /// Number of strong (`Rc`) pointers currently alive. Zero means no strong
    /// references remain; the internal value may then be dropped even if
    /// weak refs exist.
    pub(super) strong: Cell<usize>,
    /// Number of weak (`Weak`) pointers currently alive. Includes the implicit
    /// weak reference held by every live `Rc`, so this is always at least the
    /// number of `Rc`s while any strong reference exists.
    pub(super) weak: Cell<usize>,
    /// The contained value.
    pub(super) value: T,
}

impl<T: ?Sized> RcInner<T> {
    /// Reads the current strong count.
    #[inline]
    pub(super) fn strong(&self) -> usize {
        self.strong.get()
    }

    /// Reads the current weak count.
    #[inline]
    pub(super) fn weak(&self) -> usize {
        self.weak.get()
    }

    /// Increments the strong count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already at
    /// `usize::MAX`.
    #[inline]
    pub(super) fn inc_strong(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.strong.get();
        match cur.checked_add(1) {
            Some(next) => {
                self.strong.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }

    /// Decrements the strong count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already zero
    /// (unbalanced decrement).
    #[inline]
    pub(super) fn dec_strong(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.strong.get();
        match cur.checked_sub(1) {
            Some(next) => {
                self.strong.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }

    /// Increments the weak count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already at
    /// `usize::MAX`.
    #[inline]
    pub(super) fn inc_weak(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.weak.get();
        match cur.checked_add(1) {
            Some(next) => {
                self.weak.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }

    /// Decrements the weak count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already zero
    /// (unbalanced decrement).
    #[inline]
    pub(super) fn dec_weak(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.weak.get();
        match cur.checked_sub(1) {
            Some(next) => {
                self.weak.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }
}

/// Helper type to allow accessing the reference counts without
/// making any assertions about the data field.
///
/// When a `Weak` outlives all `Rc`s, the payload (`value`) has been dropped
/// in-place but the allocation remains alive (pinned by the weak count). A
/// `&RcInner<T>` covering the whole struct would assert validity of the
/// already-dropped payload. `WeakInner` holds only references to the two
/// counter cells, which are always valid while the allocation exists.
pub(super) struct WeakInner<'a> {
    pub(super) strong: &'a Cell<usize>,
    pub(super) weak: &'a Cell<usize>,
}

impl WeakInner<'_> {
    /// Reads the current strong count.
    #[inline]
    pub(super) fn strong(&self) -> usize {
        self.strong.get()
    }

    /// Reads the current weak count.
    #[inline]
    pub(super) fn weak(&self) -> usize {
        self.weak.get()
    }

    /// Increments the strong count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already at
    /// `usize::MAX`.
    #[inline]
    pub(super) fn inc_strong(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.strong.get();
        match cur.checked_add(1) {
            Some(next) => {
                self.strong.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }

    /// Decrements the weak count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already zero
    /// (unbalanced decrement).
    #[inline]
    pub(super) fn dec_weak(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.weak.get();
        match cur.checked_sub(1) {
            Some(next) => {
                self.weak.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }
}
