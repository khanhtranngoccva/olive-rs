//! Cursors for [`BTreeSet`].
//!
//! Thin facades over the map's cursors ([`super::super::cursor`]).
//!
//! A set cursor points at a gap between two values
//! and can navigate in either direction, peek at or remove the neighbours of
//! the gap, and insert a value into the gap.

use super::super::cursor::{
    Cursor as MapCursor, CursorMut as MapCursorMut, CursorMutKey as MapCursorMutKey,
    TryBTreeMapCursorInsertError as MapTryBTreeCursorInsertError,
};
use super::super::set_val::SetValZST;
use super::BTreeSet;
use crate::alloc::{AllocError, Allocator, Global};
use core::borrow::Borrow;
use core::fmt;
use core::ops::Bound;
use olive_core::try_traits::{TryClone, TryCloneError};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned by fallible `BTreeSet` cursor insertion operations.
///
/// Covers [`CursorMut::insert_after`], [`CursorMut::insert_before`],
/// [`CursorMutKey::insert_after`], [`CursorMutKey::insert_before`] and their
/// `_give_back` variants — any operation whose failure modes are limited to
/// an out-of-order value ([`Self::Unordered`]) or an allocation
/// ([`AllocError`]).
#[derive(Clone, PartialEq, Eq)]
pub enum TryBTreeSetCursorInsertError {
    /// The value is not properly ordered relative to its neighbors.
    Unordered,
    /// An allocation during tree restructuring failed.
    Alloc(AllocError),
}

impl fmt::Debug for TryBTreeSetCursorInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unordered => f
                .debug_tuple("TryBTreeSetCursorInsertError::Unordered")
                .finish(),
            Self::Alloc(e) => f
                .debug_tuple("TryBTreeSetCursorInsertError::Alloc")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryBTreeSetCursorInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unordered => write!(
                f,
                "BTreeSet cursor insertion failed: value is not properly ordered relative to its neighbors"
            ),
            Self::Alloc(_) => {
                write!(
                    f,
                    "BTreeSet cursor insertion failed: memory allocation failed"
                )
            }
        }
    }
}

impl core::error::Error for TryBTreeSetCursorInsertError {}

impl From<AllocError> for TryBTreeSetCursorInsertError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

impl From<MapTryBTreeCursorInsertError> for TryBTreeSetCursorInsertError {
    #[inline]
    fn from(err: MapTryBTreeCursorInsertError) -> Self {
        match err {
            MapTryBTreeCursorInsertError::Unordered => Self::Unordered,
            MapTryBTreeCursorInsertError::Alloc(e) => Self::Alloc(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

/// A cursor over a [`BTreeSet`].
///
/// A `Cursor` points to a gap between two values, and is able to freely
/// navigate the set in either direction. It can also be used to look up the
/// values on either side of the gap.
///
/// A `Cursor` is obtained with [`BTreeSet::lower_bound`],
/// [`BTreeSet::upper_bound`] or [`CursorMut::as_cursor`].
///
/// To edit the set through a cursor, see [`CursorMut`] and [`CursorMutKey`].
pub struct Cursor<'a, T: 'a> {
    cursor: MapCursor<'a, T, SetValZST>,
}

impl<'a, T: 'a> Cursor<'a, T> {
    /// Advances the cursor to the next gap, returning the value that it
    /// moved over.
    ///
    /// If the cursor is already at the end of the set then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the value it returned.
    #[allow(clippy::should_implement_trait, reason = "cursors are not iterators")]
    pub fn next(&mut self) -> Option<&'a T> {
        self.cursor.next().map(|(k, _)| k)
    }

    /// Moves the cursor to the previous gap, returning the value that it
    /// moved over.
    ///
    /// If the cursor is already at the start of the set then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the value it returned.
    pub fn prev(&mut self) -> Option<&'a T> {
        self.cursor.prev().map(|(k, _)| k)
    }

    /// Returns the value that [`Cursor::next`] would move over, without
    /// moving the cursor.
    #[must_use]
    pub fn peek_next(&self) -> Option<&'a T> {
        self.cursor.peek_next().map(|(k, _)| k)
    }

    /// Returns the value that [`Cursor::prev`] would move over, without
    /// moving the cursor.
    #[must_use]
    pub fn peek_prev(&self) -> Option<&'a T> {
        self.cursor.peek_prev().map(|(k, _)| k)
    }
}

impl<T> Clone for Cursor<'_, T> {
    fn clone(&self) -> Self {
        Cursor {
            cursor: self.cursor.clone(),
        }
    }
}

impl<T> TryClone for Cursor<'_, T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Cloning is trivial.
        Ok(self.clone())
    }
}

impl<T> fmt::Debug for Cursor<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Cursor")
    }
}

// ---------------------------------------------------------------------------
// CursorMut
// ---------------------------------------------------------------------------

/// A cursor over a [`BTreeSet`] with exclusive access and ability to insert
/// and remove values, but without the ability to change the values.
///
/// A `CursorMut` can freely seek back-and-forth, and can safely modify the
/// set during traversal by inserting into and removing from the gap. The
/// values are only ever exposed by shared reference; to modify the values,
/// convert the cursor with [`CursorMut::with_mutable_key`] into a
/// [`CursorMutKey`].
///
/// A `CursorMut` is obtained with [`BTreeSet::lower_bound_mut`] or
/// [`BTreeSet::upper_bound_mut`].
pub struct CursorMut<'a, T: 'a, A: Allocator = Global> {
    cursor: MapCursorMut<'a, T, SetValZST, A>,
}

impl<'a, T: 'a, A: Allocator> CursorMut<'a, T, A> {
    /// Advances the cursor to the next gap, returning a shared reference to
    /// the value that it moved over.
    ///
    /// If the cursor is already at the end of the set then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the value it returned.
    #[allow(clippy::should_implement_trait, reason = "cursors are not iterators")]
    pub fn next(&mut self) -> Option<&'_ T> {
        self.cursor.next().map(|(k, _)| k)
    }

    /// Moves the cursor to the previous gap, returning a shared reference to
    /// the value that it moved over.
    ///
    /// If the cursor is already at the start of the set then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the value it returned.
    pub fn prev(&mut self) -> Option<&'_ T> {
        self.cursor.prev().map(|(k, _)| k)
    }

    /// Returns a shared reference to the value that [`CursorMut::next`]
    /// would move over, without moving the cursor.
    pub fn peek_next(&mut self) -> Option<&'_ T> {
        self.cursor.peek_next().map(|(k, _)| k)
    }

    /// Returns a shared reference to the value that [`CursorMut::prev`]
    /// would move over, without moving the cursor.
    pub fn peek_prev(&mut self) -> Option<&'_ T> {
        self.cursor.peek_prev().map(|(k, _)| k)
    }

    /// Returns a read-only cursor pointing to the same location as the
    /// `CursorMut`.
    ///
    /// The lifetime of the returned `Cursor` is bound to that of the
    /// `CursorMut`, which means it cannot outlive the `CursorMut` and that
    /// the `CursorMut` is frozen for the lifetime of the `Cursor`.
    #[must_use]
    pub fn as_cursor(&self) -> Cursor<'_, T> {
        Cursor {
            cursor: self.cursor.as_cursor(),
        }
    }

    /// Converts the cursor into a [`CursorMutKey`], which allows mutating
    /// the values of the set.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the set stays in a sorted state after
    /// having modified a value through the returned cursor, and the new
    /// value(s) replaced by the cursor must be unique.
    #[must_use]
    pub unsafe fn with_mutable_key(self) -> CursorMutKey<'a, T, A> {
        // SAFETY: precondition from the caller.
        CursorMutKey {
            cursor: unsafe { self.cursor.with_mutable_key() },
        }
    }
}

impl<T: Ord, A: Allocator> CursorMut<'_, T, A> {
    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to, similar to [`insert_after`], but the value is
    /// returned on error.
    ///
    /// # Errors
    ///
    /// If the value is not properly ordered relative to its neighbors, or on
    /// out-of-memory conditions or length overflows, this operation returns
    /// the value along with a [`TryBTreeSetCursorInsertError`] and leaves
    /// both the set and the cursor unchanged.
    ///
    /// [`insert_after`]: Self::insert_after
    pub fn insert_after_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, TryBTreeSetCursorInsertError)> {
        self.cursor
            .insert_after_give_back(value, SetValZST)
            .map_err(|(value, _, e)| (value, TryBTreeSetCursorInsertError::from(e)))
    }

    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted value.
    ///
    /// # Errors
    ///
    /// If the value is not properly ordered relative to its neighbors, or if
    /// the allocator cannot accommodate the insertion, the value is dropped
    /// and the set and cursor are left unchanged, and the
    /// [`TryBTreeSetCursorInsertError`] is returned.
    pub fn insert_after(&mut self, value: T) -> Result<(), TryBTreeSetCursorInsertError> {
        self.cursor
            .insert_after(value, SetValZST)
            .map_err(TryBTreeSetCursorInsertError::from)
    }

    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to, similar to [`insert_before`], but the value is
    /// returned on error.
    ///
    /// # Errors
    ///
    /// If the value is not properly ordered relative to its neighbors, or on
    /// out-of-memory conditions or length overflows, this operation returns
    /// the value along with a [`TryBTreeSetCursorInsertError`] and leaves
    /// both the set and the cursor unchanged.
    ///
    /// [`insert_before`]: Self::insert_before
    pub fn insert_before_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, TryBTreeSetCursorInsertError)> {
        self.cursor
            .insert_before_give_back(value, SetValZST)
            .map_err(|(value, _, e)| (value, TryBTreeSetCursorInsertError::from(e)))
    }

    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted value.
    ///
    /// # Errors
    ///
    /// If the value is not properly ordered relative to its neighbors, or if
    /// the allocator cannot accommodate the insertion, the value is dropped
    /// and the set and cursor are left unchanged, and the
    /// [`TryBTreeSetCursorInsertError`] is returned.
    pub fn insert_before(&mut self, value: T) -> Result<(), TryBTreeSetCursorInsertError> {
        self.cursor
            .insert_before(value, SetValZST)
            .map_err(TryBTreeSetCursorInsertError::from)
    }

    /// Similar to [`insert_after_unchecked`] but also returns the value on
    /// error.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns the value along with an [`AllocError`] and leaves both the
    /// set and the cursor unchanged.
    ///
    /// # Safety
    ///
    /// Please see [`insert_after_unchecked`] for the safety requirements.
    ///
    /// [`insert_after_unchecked`]: Self::insert_after_unchecked
    pub unsafe fn insert_after_unchecked_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, AllocError)> {
        // SAFETY: precondition from the caller.
        unsafe {
            self.cursor
                .insert_after_unchecked_give_back(value, SetValZST)
                .map_err(|(value, _, e)| (value, e))
        }
    }

    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted value.
    ///
    /// This is faster than [`insert_after`](Self::insert_after), but does not
    /// check that the value is properly ordered relative to its neighbors.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns an [`AllocError`] and leaves both the set and the cursor
    /// unchanged. The value is dropped.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The value of the newly inserted element must be unique in the set.
    /// * All values in the set must remain in sorted order.
    pub unsafe fn insert_after_unchecked(&mut self, value: T) -> Result<(), AllocError> {
        // SAFETY: precondition from the caller
        unsafe { self.cursor.insert_after_unchecked(value, SetValZST) }
    }

    /// Similar to [`insert_before_unchecked`] but also returns the value on
    /// error.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns the value along with an [`AllocError`] and leaves both the
    /// set and the cursor unchanged.
    ///
    /// # Safety
    ///
    /// Please see [`insert_before_unchecked`] for the safety requirements.
    ///
    /// [`insert_before_unchecked`]: Self::insert_before_unchecked
    pub unsafe fn insert_before_unchecked_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, AllocError)> {
        // SAFETY: precondition from the caller.
        unsafe {
            self.cursor
                .insert_before_unchecked_give_back(value, SetValZST)
                .map_err(|(value, _, e)| (value, e))
        }
    }

    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted value.
    ///
    /// This is faster than [`insert_before`](Self::insert_before), but does
    /// not check that the value is properly ordered relative to its
    /// neighbors.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns an [`AllocError`] and leaves both the set and the cursor
    /// unchanged. The value is dropped.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The value of the newly inserted element must be unique in the set.
    /// * All values in the set must remain in sorted order.
    pub unsafe fn insert_before_unchecked(&mut self, value: T) -> Result<(), AllocError> {
        // SAFETY: precondition from the caller
        unsafe { self.cursor.insert_before_unchecked(value, SetValZST) }
    }

    /// Removes the value that the cursor would return by [`next`](Self::next),
    /// if there is one.
    ///
    /// After the removal the cursor's position is unchanged.
    pub fn remove_next(&mut self) -> Option<T> {
        self.cursor.remove_next().map(|(k, _)| k)
    }

    /// Removes the value that the cursor would return by [`prev`](Self::prev),
    /// if there is one.
    ///
    /// After the removal the cursor's position is unchanged.
    pub fn remove_prev(&mut self) -> Option<T> {
        self.cursor.remove_prev().map(|(k, _)| k)
    }
}

impl<T, A: Allocator> fmt::Debug for CursorMut<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMut")
    }
}

// ---------------------------------------------------------------------------
// CursorMutKey
// ---------------------------------------------------------------------------

/// A cursor over a [`BTreeSet`] with exclusive access to the values.
///
/// A `CursorMutKey` points to a gap between two values, and is able to
/// navigate the set in either direction. It can also be used to look up the
/// values on either side of the gap, and to insert and remove values
/// adjacent to the gap. Unlike [`CursorMut`], it may also modify the values
/// of the set, which can break the sort order of the set.
///
/// A `CursorMutKey` is obtained from [`CursorMut::with_mutable_key`].
///
/// # Safety
///
/// You must ensure that the `BTreeSet` invariants are maintained.
/// Specifically:
///
/// * The value of the newly inserted element must be unique in the set.
/// * All values in the set must remain in sorted order.
pub struct CursorMutKey<'a, T: 'a, A: Allocator = Global> {
    cursor: MapCursorMutKey<'a, T, SetValZST, A>,
}

impl<T, A: Allocator> fmt::Debug for CursorMutKey<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMutKey")
    }
}

impl<'a, T: 'a, A: Allocator> CursorMutKey<'a, T, A> {
    /// Advances the cursor to the next gap, returning a mutable reference to
    /// the value that it moved over.
    ///
    /// If the cursor is already at the end of the set then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the value it returned.
    #[allow(clippy::should_implement_trait, reason = "cursors are not iterators")]
    pub fn next(&mut self) -> Option<&'_ mut T> {
        self.cursor.next().map(|(k, _)| k)
    }

    /// Moves the cursor to the previous gap, returning a mutable reference to
    /// the value that it moved over.
    ///
    /// If the cursor is already at the start of the set then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the value it returned.
    pub fn prev(&mut self) -> Option<&'_ mut T> {
        self.cursor.prev().map(|(k, _)| k)
    }

    /// Returns a mutable reference to the value that [`CursorMutKey::next`]
    /// would move over, without moving the cursor.
    pub fn peek_next(&mut self) -> Option<&'_ mut T> {
        self.cursor.peek_next().map(|(k, _)| k)
    }

    /// Returns a mutable reference to the value that [`CursorMutKey::prev`]
    /// would move over, without moving the cursor.
    pub fn peek_prev(&mut self) -> Option<&'_ mut T> {
        self.cursor.peek_prev().map(|(k, _)| k)
    }

    /// Converts the mutable cursor into an immutable cursor.
    ///
    /// This is useful for looking ahead without modifying the cursor itself.
    #[must_use]
    pub fn as_cursor(&self) -> Cursor<'_, T> {
        Cursor {
            cursor: self.cursor.as_cursor(),
        }
    }
}

impl<T: Ord, A: Allocator> CursorMutKey<'_, T, A> {
    /// Similar to [`insert_after_unchecked`] but also returns the value on
    /// error.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns the value along with an [`AllocError`](AllocError)
    /// and leaves both the set and the cursor unchanged.
    ///
    /// # Safety
    ///
    /// Please see [`insert_after_unchecked`] for the safety requirements.
    ///
    /// [`insert_after_unchecked`]: Self::insert_after_unchecked
    pub unsafe fn insert_after_unchecked_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, AllocError)> {
        // SAFETY: precondition from the caller.
        unsafe {
            self.cursor
                .insert_after_unchecked_give_back(value, SetValZST)
                .map_err(|(value, _, e)| (value, e))
        }
    }

    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted value.
    ///
    /// This is faster than [`insert_after`](Self::insert_after), but does not
    /// check that the value is properly ordered relative to its neighbors.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns an [`AllocError`](AllocError) and leaves both
    /// the set and the cursor unchanged. The value is dropped.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The value of the newly inserted element must be unique in the set.
    /// * All values in the set must remain in sorted order.
    pub unsafe fn insert_after_unchecked(&mut self, value: T) -> Result<(), AllocError> {
        // SAFETY: precondition from the caller
        unsafe { self.cursor.insert_after_unchecked(value, SetValZST) }
    }

    /// Similar to [`insert_before_unchecked`] but also returns the value on
    /// error.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns the value along with an [`AllocError`](AllocError)
    /// and leaves both the set and the cursor unchanged.
    ///
    /// # Safety
    ///
    /// Please see [`insert_before_unchecked`] for the safety requirements.
    ///
    /// [`insert_before_unchecked`]: Self::insert_before_unchecked
    pub unsafe fn insert_before_unchecked_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, AllocError)> {
        // SAFETY: precondition from the caller.
        unsafe {
            self.cursor
                .insert_before_unchecked_give_back(value, SetValZST)
                .map_err(|(value, _, e)| (value, e))
        }
    }

    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted value.
    ///
    /// This is faster than [`insert_before`](Self::insert_before), but does
    /// not check that the value is properly ordered relative to its
    /// neighbors.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns an [`AllocError`](AllocError) and leaves both
    /// the set and the cursor unchanged. The value is dropped.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The value of the newly inserted element must be unique in the set.
    /// * All values in the set must remain in sorted order.
    pub unsafe fn insert_before_unchecked(&mut self, value: T) -> Result<(), AllocError> {
        // SAFETY: precondition from the caller
        unsafe { self.cursor.insert_before_unchecked(value, SetValZST) }
    }

    /// Similar to [`insert_after`] but also returns the value on error.
    ///
    /// # Errors
    ///
    /// If the value is not properly ordered relative to its neighbors, or on
    /// out-of-memory conditions or length overflows, this operation returns
    /// the value along with a [`TryBTreeSetCursorInsertError`] and leaves
    /// both the set and the cursor unchanged.
    ///
    /// [`insert_after`]: Self::insert_after
    pub fn insert_after_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, TryBTreeSetCursorInsertError)> {
        self.cursor
            .insert_after_give_back(value, SetValZST)
            .map_err(|(value, _, e)| (value, TryBTreeSetCursorInsertError::from(e)))
    }

    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted value.
    ///
    /// This is safer than
    /// [`insert_after_unchecked`](Self::insert_after_unchecked), but runs
    /// additional checks.
    ///
    /// # Errors
    ///
    /// If the value is not properly ordered relative to its neighbors, or if
    /// the allocator cannot accommodate the insertion, the value is dropped
    /// and the set and cursor are left unchanged, and the
    /// [`TryBTreeSetCursorInsertError`] is returned.
    pub fn insert_after(&mut self, value: T) -> Result<(), TryBTreeSetCursorInsertError> {
        self.cursor
            .insert_after(value, SetValZST)
            .map_err(TryBTreeSetCursorInsertError::from)
    }

    /// Similar to [`insert_before`] but also returns the value on error.
    ///
    /// # Errors
    ///
    /// If the value is not properly ordered relative to its neighbors, or on
    /// out-of-memory conditions or length overflows, this operation returns
    /// the value along with a [`TryBTreeSetCursorInsertError`] and leaves
    /// both the set and the cursor unchanged.
    ///
    /// [`insert_before`]: Self::insert_before
    pub fn insert_before_give_back(
        &mut self,
        value: T,
    ) -> Result<(), (T, TryBTreeSetCursorInsertError)> {
        self.cursor
            .insert_before_give_back(value, SetValZST)
            .map_err(|(value, _, e)| (value, TryBTreeSetCursorInsertError::from(e)))
    }

    /// Inserts a new value into the set in the gap that the cursor is
    /// currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted value.
    ///
    /// This is safer than
    /// [`insert_before_unchecked`](Self::insert_before_unchecked), but runs
    /// additional checks.
    ///
    /// # Errors
    ///
    /// If the value is not properly ordered relative to its neighbors, or if
    /// the allocator cannot accommodate the insertion, the value is dropped
    /// and the set and cursor are left unchanged, and the
    /// [`TryBTreeSetCursorInsertError`] is returned.
    pub fn insert_before(&mut self, value: T) -> Result<(), TryBTreeSetCursorInsertError> {
        self.cursor
            .insert_before(value, SetValZST)
            .map_err(TryBTreeSetCursorInsertError::from)
    }

    /// Removes the value that the cursor would return by [`next`](Self::next),
    /// if there is one.
    ///
    /// After the removal the cursor's position is unchanged.
    pub fn remove_next(&mut self) -> Option<T> {
        self.cursor.remove_next().map(|(k, _)| k)
    }

    /// Removes the value that the cursor would return by [`prev`](Self::prev),
    /// if there is one.
    ///
    /// After the removal the cursor's position is unchanged.
    pub fn remove_prev(&mut self) -> Option<T> {
        self.cursor.remove_prev().map(|(k, _)| k)
    }
}

// ---------------------------------------------------------------------------
// BTreeSet cursor constructors
// ---------------------------------------------------------------------------

impl<T, A: Allocator> BTreeSet<T, A> {
    /// Returns a [`Cursor`] pointing at the gap before the smallest value
    /// greater than or equal to the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest value greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest value greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest value in the set.
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Cursor {
            cursor: self.map.lower_bound(bound),
        }
    }

    /// Returns a [`CursorMut`] pointing at the gap before the smallest value
    /// greater than or equal to the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest value greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest value greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest value in the set.
    pub fn lower_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, A>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        CursorMut {
            cursor: self.map.lower_bound_mut(bound),
        }
    }

    /// Returns a [`Cursor`] pointing at the gap after the greatest value
    /// smaller than or equal to the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest value smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest value smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest value in the set.
    pub fn upper_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Cursor {
            cursor: self.map.upper_bound(bound),
        }
    }

    /// Returns a [`CursorMut`] pointing at the gap after the greatest value
    /// smaller than or equal to the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest value smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest value smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest value in the set.
    pub fn upper_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, A>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        CursorMut {
            cursor: self.map.upper_bound_mut(bound),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;

    use core::ops::Bound;

    use olive_core::alloc::AllocError;
    use olive_core::try_traits::TryClone;
    use olive_test_helpers::BudgetedAlloc;

    use super::super::BTreeSet;
    use super::{Cursor, CursorMut, CursorMutKey, TryBTreeSetCursorInsertError};
    use crate::alloc::Global;

    fn make_set() -> BTreeSet<i32, Global> {
        let mut set = BTreeSet::new_in(Global);
        for i in 1..=4 {
            set.try_insert(i).unwrap();
        }
        set
    }

    #[test]
    fn navigation_forward() {
        let set = make_set();
        let mut cursor = set.lower_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_prev(), None);
        let mut seen = std::vec::Vec::new();
        while let Some(v) = cursor.next() {
            seen.push(*v);
        }
        assert_eq!(seen, std::vec![1, 2, 3, 4]);
        assert_eq!(cursor.peek_next(), None);
        assert_eq!(cursor.next(), None);
    }

    #[test]
    fn navigation_backward() {
        let set = make_set();
        let mut cursor = set.upper_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_next(), None);
        let mut seen = std::vec::Vec::new();
        while let Some(v) = cursor.prev() {
            seen.push(*v);
        }
        assert_eq!(seen, std::vec![4, 3, 2, 1]);
        assert_eq!(cursor.peek_prev(), None);
        assert_eq!(cursor.prev(), None);
    }

    #[test]
    fn lower_upper_bounds() {
        let set = make_set();

        let cursor = set.lower_bound(Bound::Included(&2));
        assert_eq!(cursor.peek_prev(), Some(&1));
        assert_eq!(cursor.peek_next(), Some(&2));

        let cursor = set.lower_bound(Bound::Excluded(&2));
        assert_eq!(cursor.peek_prev(), Some(&2));
        assert_eq!(cursor.peek_next(), Some(&3));

        let cursor = set.lower_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_prev(), None);
        assert_eq!(cursor.peek_next(), Some(&1));

        let cursor = set.upper_bound(Bound::Included(&3));
        assert_eq!(cursor.peek_prev(), Some(&3));
        assert_eq!(cursor.peek_next(), Some(&4));

        let cursor = set.upper_bound(Bound::Excluded(&3));
        assert_eq!(cursor.peek_prev(), Some(&2));
        assert_eq!(cursor.peek_next(), Some(&3));

        let cursor = set.upper_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_prev(), Some(&4));
        assert_eq!(cursor.peek_next(), None);
    }

    #[test]
    fn bounds_on_empty_set() {
        let set: BTreeSet<i32, Global> = BTreeSet::new_in(Global);
        let mut cursor = set.lower_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_prev(), None);
        assert_eq!(cursor.peek_next(), None);
        assert_eq!(cursor.next(), None);
        let mut cursor = set.upper_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_prev(), None);
        assert_eq!(cursor.peek_next(), None);
        assert_eq!(cursor.prev(), None);
    }

    #[test]
    fn cursor_is_cloneable() {
        let set = make_set();
        let mut cursor = set.lower_bound(Bound::Included(&2));
        assert_eq!(cursor.next(), Some(&2));

        let snapshot = cursor.clone();
        let mut snapshot_2 = cursor.try_clone().unwrap();

        assert_eq!(cursor.next(), Some(&3));
        // The snapshot kept the position it had when cloned.
        assert_eq!(snapshot.peek_prev(), Some(&2));
        assert_eq!(snapshot.peek_next(), Some(&3));
        assert_eq!(snapshot_2.peek_prev(), Some(&2));
        assert_eq!(snapshot_2.peek_next(), Some(&3));

        // The clones should not affect the old cursor.
        assert_eq!(snapshot_2.next(), Some(&3));
        // The old cursor already moved past 3, so 4 is next.
        assert_eq!(cursor.peek_next(), Some(&4));
    }

    #[test]
    fn cursor_mut_navigation_and_peek() {
        let mut set = make_set();
        let mut cursor: CursorMut<i32, Global> = set.lower_bound_mut(Bound::Unbounded);
        assert_eq!(cursor.next(), Some(&1));
        assert_eq!(cursor.peek_next(), Some(&2));
        // Peeking does not move the cursor.
        assert_eq!(cursor.next(), Some(&2));
        assert_eq!(cursor.prev(), Some(&2));
        assert_eq!(cursor.peek_prev(), Some(&1));
    }

    #[test]
    fn insert_and_give_back() {
        let mut set = make_set();
        {
            let mut cursor: CursorMut<i32, Global> = set.lower_bound_mut(Bound::Unbounded);
            // insert_after with the gap before 1 is an ordering error.
            match cursor.insert_after_give_back(5) {
                Err((v, TryBTreeSetCursorInsertError::Unordered)) => assert_eq!(v, 5),
                _ => panic!("out-of-order insertion should fail"),
            }
            match cursor.insert_before_give_back(5) {
                Err((v, TryBTreeSetCursorInsertError::Unordered)) => assert_eq!(v, 5),
                _ => panic!("out-of-order insertion should fail"),
            }
            // In-order insertion succeeds, and the non-give-back wrappers
            // behave identically on the happy path. Insertion places the
            // value directly before the gap and leaves the cursor at the
            // same gap.
            cursor.insert_before(0).unwrap();
            assert_eq!(cursor.peek_prev(), Some(&0));
            assert_eq!(cursor.next(), Some(&1));
            // Append past the end of the set from the end gap.
            while cursor.next().is_some() {}
            cursor.insert_before_give_back(100).unwrap();
            assert_eq!(cursor.peek_prev(), Some(&100));
            assert_eq!(cursor.peek_next(), None);
        }
        assert_eq!(set.len(), 6);
        assert_eq!(set.get(&0), Some(&0));
        assert_eq!(set.get(&100), Some(&100));
        let collected: std::vec::Vec<i32> = set.iter().copied().collect();
        assert_eq!(collected, std::vec![0, 1, 2, 3, 4, 100]);
    }

    #[test]
    fn checked_insert_alloc_failure() {
        let alloc = BudgetedAlloc::new(1 << 20);
        let mut set: BTreeSet<i32, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        // Fill the root leaf completely, so that the next insertion has to
        // split it and therefore really allocates.
        for i in 1..=11 {
            set.try_insert(i).unwrap();
        }

        {
            let mut cursor: CursorMutKey<i32, BudgetedAlloc> =
                unsafe { set.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            // Exhaust the budget so any further allocation fails. The value
            // is properly ordered here, so the failure must be reported as
            // an allocation error, not an ordering error.
            alloc.drain();
            match cursor.insert_after_give_back(12) {
                Err((v, TryBTreeSetCursorInsertError::Alloc(AllocError))) => assert_eq!(v, 12),
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            match cursor.insert_before_give_back(12) {
                Err((v, TryBTreeSetCursorInsertError::Alloc(AllocError))) => assert_eq!(v, 12),
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            match cursor.insert_after(12) {
                Err(TryBTreeSetCursorInsertError::Alloc(AllocError)) => {}
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            match cursor.insert_before(12) {
                Err(TryBTreeSetCursorInsertError::Alloc(AllocError)) => {}
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            assert_eq!(cursor.peek_prev(), Some(&mut 11));
        }
        assert_eq!(set.len(), 11);

        // After the budget is restored, the same insertion succeeds.
        alloc.set_budget(1 << 20);
        {
            let mut cursor: CursorMutKey<i32, BudgetedAlloc> =
                unsafe { set.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            cursor.insert_after(12).unwrap();
        }
        assert_eq!(set.len(), 12);
        assert_eq!(set.get(&12), Some(&12));
    }

    #[test]
    fn cursor_mut_insert_unchecked_happy_path() {
        let mut set = make_set();
        {
            let mut cursor: CursorMut<i32, Global> = set.lower_bound_mut(Bound::Unbounded);
            // SAFETY: prepending before the smallest element keeps the set
            // sorted.
            unsafe { cursor.insert_after_unchecked(0).unwrap() };
            // The cursor ends up at the gap before the inserted value.
            assert_eq!(cursor.peek_prev(), None);
            assert_eq!(cursor.next(), Some(&0));
        }
        {
            let mut cursor: CursorMut<i32, Global> = set.upper_bound_mut(Bound::Unbounded);
            // SAFETY: appending past the largest element keeps the set sorted.
            unsafe { cursor.insert_before_unchecked(5).unwrap() };
            assert_eq!(cursor.peek_prev(), Some(&5));
            assert_eq!(cursor.peek_next(), None);
        }
        assert_eq!(set.len(), 6);
        let collected: std::vec::Vec<i32> = set.iter().copied().collect();
        assert_eq!(collected, std::vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn cursor_mut_insert_unchecked_alloc_failure() {
        let alloc = BudgetedAlloc::new(1 << 20);
        let mut set: BTreeSet<i32, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        // Fill the root leaf completely, so that the next insertion has to
        // split it and therefore really allocates.
        for i in 1..=11 {
            set.try_insert(i).unwrap();
        }

        {
            let mut cursor: CursorMut<i32, BudgetedAlloc> = set.upper_bound_mut(Bound::Unbounded);
            alloc.drain();
            // SAFETY: the value is appended past the largest element, so the
            // set stays sorted; the failure must be an allocation error.
            let result = unsafe { cursor.insert_before_unchecked_give_back(12) };
            match result {
                Err((v, AllocError)) => assert_eq!(v, 12),
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            // SAFETY: same ordering precondition as above.
            let result = unsafe { cursor.insert_before_unchecked(12) };
            match result {
                Err(AllocError) => {}
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            // SAFETY: same ordering precondition as above.
            let result = unsafe { cursor.insert_after_unchecked_give_back(12) };
            match result {
                Err((v, AllocError)) => assert_eq!(v, 12),
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            // SAFETY: same ordering precondition as above.
            let result = unsafe { cursor.insert_after_unchecked(12) };
            match result {
                Err(AllocError) => {}
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            assert_eq!(cursor.peek_prev(), Some(&11));
        }
        assert_eq!(set.len(), 11);

        // After the budget is restored, the same insertion succeeds.
        alloc.set_budget(1 << 20);
        {
            let mut cursor: CursorMut<i32, BudgetedAlloc> = set.upper_bound_mut(Bound::Unbounded);
            // SAFETY: appending past the largest element keeps the set sorted.
            unsafe { cursor.insert_before_unchecked(12).unwrap() };
        }
        assert_eq!(set.len(), 12);
        assert_eq!(set.get(&12), Some(&12));
    }

    #[test]
    fn remove_next_and_prev() {
        let mut set = make_set();

        {
            let mut cursor: CursorMut<i32, Global> = set.lower_bound_mut(Bound::Unbounded);
            assert_eq!(cursor.remove_next(), Some(1));
            // The cursor stays in place, now facing the element that followed.
            assert_eq!(cursor.peek_prev(), None);
            assert_eq!(cursor.peek_next(), Some(&2));
        }
        assert_eq!(set.len(), 3);

        {
            let mut cursor: CursorMut<i32, Global> = set.upper_bound_mut(Bound::Unbounded);
            assert_eq!(cursor.remove_next(), None);
            cursor.prev();
            assert_eq!(cursor.remove_prev(), Some(3));
            assert_eq!(cursor.peek_prev(), Some(&2));
            assert_eq!(cursor.peek_next(), Some(&4));
        }
        assert_eq!(set.len(), 2);

        {
            let mut cursor: CursorMut<i32, Global> = set.lower_bound_mut(Bound::Unbounded);
            assert_eq!(cursor.remove_prev(), None);
        }
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn remove_across_internal_levels() {
        let mut set = BTreeSet::new_in(Global);
        for i in 0..=20 {
            set.try_insert(i).unwrap();
        }
        // 21 elements force the root to split into an internal node.
        assert!(set.len() > 11);

        {
            let mut cursor: CursorMutKey<i32, Global> =
                unsafe { set.lower_bound_mut(Bound::Unbounded).with_mutable_key() };
            let mut seen = std::vec::Vec::new();
            while let Some(v) = cursor.remove_next() {
                seen.push(v);
            }
            assert_eq!(seen.len(), 21);
        }
        assert_eq!(set.len(), 0);
        // The set must still work after the internal root level was popped.
        set.try_insert(7).unwrap();
        assert_eq!(set.get(&7), Some(&7));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn cursor_mut_as_cursor() {
        let mut set = make_set();
        let mut cursor_mut = set.lower_bound_mut(Bound::Unbounded);
        let mut cursor: Cursor<i32> = cursor_mut.as_cursor();
        assert_eq!(cursor.next(), Some(&1));
        // The mutable cursor and its immutable view agree on position.
        assert_eq!(cursor_mut.peek_next(), Some(&1));
    }

    #[test]
    fn cursor_mut_key_mutates_element() {
        let mut set = make_set();
        // SAFETY: the test restores a unique, sorted state before the
        // cursor goes out of scope.
        {
            let mut cursor: CursorMutKey<i32, Global> =
                unsafe { set.upper_bound_mut(Bound::Included(&4)).with_mutable_key() };
            let last = cursor.peek_prev().unwrap();
            assert_eq!(last, &mut 4);
            *last = 9;
        }
        let collected: std::vec::Vec<i32> = set.iter().copied().collect();
        assert_eq!(collected, std::vec![1, 2, 3, 9]);
    }

    #[test]
    fn insert_unchecked_happy_path() {
        let mut set = make_set();
        {
            let mut cursor: CursorMutKey<i32, Global> =
                unsafe { set.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            // SAFETY: appending past the largest element keeps the set sorted.
            unsafe { cursor.insert_before_unchecked(5).unwrap() };
            // SAFETY: the cursor stayed at the end gap, still ordered.
            unsafe { cursor.insert_before_unchecked(6).unwrap() };
        }
        assert_eq!(set.len(), 6);
        let collected: std::vec::Vec<i32> = set.iter().copied().collect();
        assert_eq!(collected, std::vec![1, 2, 3, 4, 5, 6]);
    }
}
