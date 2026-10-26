//! Cursors for the fallible [`BTreeMap`].
//!
//! A fallible port of the gap-based cursors from the unstable
//! features of the std library. Allocation failures during cursor edits are
//! reported through [`AllocError`] or the [`TryBTreeMapCursorInsertError`]
//! superset instead of panicking, and inserting an unordered key is reported
//! through the same superset.
//!
//! A cursor is a pointer to a gap between two elements. A cursor can visit
//! every gap in the map, and can be used to edit a gap or the elements on
//! either side of it.

use core::borrow::Borrow;
use core::fmt;
use core::ops::Bound;

use olive_core::try_traits::{TryClone, TryCloneError};

use crate::alloc::{AllocError, Allocator, Global};
use crate::collections::btree::entry;

use super::borrow::DormantMutRef;
use super::map::BTreeMap;
use super::node::{self, Handle, NodeRef, marker};
use super::search::SearchBound;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned by fallible `BTreeMap` cursor insertion operations.
///
/// Covers [`CursorMutKey::insert_after`], [`CursorMutKey::insert_before`]
/// and their `_give_back` variants — any operation whose failure modes are
/// limited to an out-of-order key ([`Self::Unordered`]) or an allocation
/// ([`AllocError`](crate::alloc::AllocError)).
#[derive(Clone, PartialEq, Eq)]
pub enum TryBTreeMapCursorInsertError {
    /// The key is not properly ordered relative to its neighbors.
    Unordered,
    /// An allocation during tree restructuring failed.
    Alloc(AllocError),
}

impl fmt::Debug for TryBTreeMapCursorInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unordered => f
                .debug_tuple("TryBTreeMapCursorInsertError::Unordered")
                .finish(),
            Self::Alloc(e) => f
                .debug_tuple("TryBTreeMapCursorInsertError::Alloc")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryBTreeMapCursorInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unordered => write!(
                f,
                "BTreeMap cursor insertion failed: key is not properly ordered relative to its neighbors"
            ),
            Self::Alloc(_) => {
                write!(
                    f,
                    "BTreeMap cursor insertion failed: memory allocation failed"
                )
            }
        }
    }
}

impl core::error::Error for TryBTreeMapCursorInsertError {}

impl From<AllocError> for TryBTreeMapCursorInsertError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

/// A cursor over a [`BTreeMap`].
///
/// A `Cursor` points to a gap between two elements, and is able to freely
/// navigate the map in either direction. It can also be used to look up the
/// keys and values on either side of the gap.
///
/// A `Cursor` is obtained with [`BTreeMap::lower_bound`],
/// [`BTreeMap::upper_bound`], [`CursorMut::as_cursor`].
///
/// To edit the map through a cursor, see [`CursorMut`] and [`CursorMutKey`].
pub struct Cursor<'a, K: 'a, V: 'a> {
    // If current is None then it means the tree has not been allocated yet.
    current: Option<Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge>>,
    root: Option<&'a node::Root<K, V>>,
}

impl<'a, K, V> Cursor<'a, K, V> {
    /// Advances the cursor to the next gap, returning the key and value of
    /// the element that it moved over.
    ///
    /// If the cursor is already at the end of the map then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the element it returned.
    #[allow(clippy::should_implement_trait, reason = "cursors are not iterators")]
    pub fn next(&mut self) -> Option<(&'a K, &'a V)> {
        let current = self.current.take()?;
        match current.next_kv() {
            Ok(kv) => {
                let result = kv.into_kv();
                self.current = Some(kv.next_leaf_edge());
                Some(result)
            }
            Err(root) => {
                self.current = Some(root.last_leaf_edge());
                None
            }
        }
    }

    /// Moves the cursor to the previous gap, returning the key and value of
    /// the element that it moved over.
    ///
    /// If the cursor is already at the start of the map then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the element it returned.
    pub fn prev(&mut self) -> Option<(&'a K, &'a V)> {
        let current = self.current.take()?;
        match current.next_back_kv() {
            Ok(kv) => {
                let result = kv.into_kv();
                self.current = Some(kv.next_back_leaf_edge());
                Some(result)
            }
            Err(root) => {
                self.current = Some(root.first_leaf_edge());
                None
            }
        }
    }

    /// Returns the key and value of the element that [`Cursor::next`] would
    /// move over, without moving the cursor.
    pub fn peek_next(&self) -> Option<(&'a K, &'a V)> {
        self.clone().next()
    }

    /// Returns the key and value of the element that [`Cursor::prev`] would
    /// move over, without moving the cursor.
    pub fn peek_prev(&self) -> Option<(&'a K, &'a V)> {
        self.clone().prev()
    }
}

impl<K, V> Clone for Cursor<'_, K, V> {
    fn clone(&self) -> Self {
        // A Cursor is a data-only edge handle to node in a BTreeMap and
        // we can just clone that pointer by copying it.
        let Cursor { current, root } = *self;
        Cursor { current, root }
    }
}

impl<K, V> TryClone for Cursor<'_, K, V> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(self.clone())
    }
}

impl<K, V> fmt::Debug for Cursor<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Cursor")
    }
}

/// A cursor over a [`BTreeMap`] with exclusive access to the values, but
/// without the ability to change the keys.
///
/// A `CursorMut` points to a gap between two elements, and is able to freely
/// navigate the map in either direction. It can also be used to look up the
/// keys and values on either side of the gap, and to insert and remove
/// elements adjacent to the gap.
///
/// A `CursorMut` is obtained with [`BTreeMap::lower_bound_mut`] or
/// [`BTreeMap::upper_bound_mut`].
pub struct CursorMut<'a, K: 'a, V: 'a, A: Allocator = Global> {
    inner: CursorMutKey<'a, K, V, A>,
}

impl<'a, K, V, A: Allocator> CursorMut<'a, K, V, A> {
    /// Advances the cursor to the next gap, returning a mutable reference to
    /// the key and value of the element that it moved over.
    ///
    /// If the cursor is already at the end of the map then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the element it returned.
    ///
    /// The returned references can be used to modify the value.
    #[allow(clippy::should_implement_trait, reason = "cursors are not iterators")]
    pub fn next(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.inner.next()?;
        Some((&*k, v))
    }

    /// Moves the cursor to the previous gap, returning a mutable reference to
    /// the value of the element that it moved over.
    ///
    /// If the cursor is already at the start of the map then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the element it returned.
    ///
    /// The returned references can be used to modify the value.
    pub fn prev(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.inner.prev()?;
        Some((&*k, v))
    }

    /// Returns a reference to the key and a mutable reference to the value of
    /// the element that [`CursorMut::next`] would move over, without moving
    /// the cursor.
    pub fn peek_next(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.inner.peek_next()?;
        Some((&*k, v))
    }

    /// Returns a reference to the key and a mutable reference to the value of
    /// the element that [`CursorMut::prev`] would move over, without moving
    /// the cursor.
    pub fn peek_prev(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.inner.peek_prev()?;
        Some((&*k, v))
    }

    /// Converts the mutable cursor into an immutable cursor.
    ///
    /// This is useful for looking ahead without modifying the cursor itself.
    pub fn as_cursor(&self) -> Cursor<'_, K, V> {
        self.inner.as_cursor()
    }

    /// Converts the cursor into a [`CursorMutKey`], which allows mutating
    /// the keys of the map.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the tree stays in a sorted state after
    /// having modified a key through the returned cursor, and the new
    /// key must be unique.
    pub unsafe fn with_mutable_key(self) -> CursorMutKey<'a, K, V, A> {
        self.inner
    }
}

impl<K, V, A: Allocator> fmt::Debug for CursorMut<'_, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMut")
    }
}

/// A cursor over a [`BTreeMap`] with exclusive access to both the keys and
/// the values.
///
/// A `CursorMutKey` points to a gap between two elements, and is able to
/// navigate the map in either direction. It can also be used to look up the
/// keys and values on either side of the gap, and to insert and remove
/// elements adjacent to the gap. Unlike [`CursorMut`], it may also modify
/// the keys of the map, which can break the sort order of the map.
///
/// A `CursorMutKey` is obtained from [`CursorMut::with_mutable_key`].
///
/// # Safety
///
/// You must ensure that the `BTreeMap` invariants are maintained.
/// Specifically:
///
/// * The key of the newly inserted element must be unique in the tree.
/// * All keys in the tree must remain in sorted order.
pub struct CursorMutKey<'a, K: 'a, V: 'a, A: Allocator = Global> {
    // If current is None then it means the tree has not been allocated yet.
    current: Option<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>>,
    root: DormantMutRef<'a, Option<node::Root<K, V>>>,
    length: &'a mut usize,
    alloc: &'a mut A,
}

impl<K, V, A: Allocator> fmt::Debug for CursorMutKey<'_, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMutKey")
    }
}

impl<K, V, A: Allocator> CursorMutKey<'_, K, V, A> {
    /// Advances the cursor to the next gap, returning a mutable reference to
    /// the key and value of the element that it moved over.
    ///
    /// If the cursor is already at the end of the map then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the element it returned.
    #[allow(clippy::should_implement_trait, reason = "cursors are not iterators")]
    pub fn next(&mut self) -> Option<(&mut K, &mut V)> {
        let current = self.current.take()?;
        match current.next_kv() {
            Ok(mut kv) => {
                let (k, v) = unsafe { kv.reborrow_mut().into_kv_mut() };
                let (k, v) = (k as *mut _, v as *mut _);
                self.current = Some(kv.next_leaf_edge());
                // SAFETY: The dormant references returned by `into_kv_mut` point
                // to a live element of the tree, which cannot be invalidated by
                // moving the cursor.
                Some(unsafe { (&mut *k, &mut *v) })
            }
            Err(root) => {
                self.current = Some(root.last_leaf_edge());
                None
            }
        }
    }

    /// Moves the cursor to the previous gap, returning a mutable reference to
    /// the key and value of the element that it moved over.
    ///
    /// If the cursor is already at the start of the map then `None` is
    /// returned and the cursor is not moved. Otherwise, the cursor is moved
    /// to the gap on the other side of the element it returned.
    pub fn prev(&mut self) -> Option<(&mut K, &mut V)> {
        let current = self.current.take()?;
        match current.next_back_kv() {
            Ok(mut kv) => {
                let (k, v) = unsafe { kv.reborrow_mut().into_kv_mut() };
                let (k, v) = (k as *mut _, v as *mut _);
                self.current = Some(kv.next_back_leaf_edge());
                // SAFETY: The dormant references returned by `into_kv_mut` point
                // to a live element of the tree, which cannot be invalidated by
                // moving the cursor.
                Some(unsafe { (&mut *k, &mut *v) })
            }
            Err(root) => {
                self.current = Some(root.first_leaf_edge());
                None
            }
        }
    }

    /// Returns a mutable reference to the key and value of the element that
    /// [`CursorMutKey::next`] would move over, without moving the cursor.
    pub fn peek_next(&mut self) -> Option<(&mut K, &mut V)> {
        let current = self.current.as_mut()?;
        let kv = unsafe { current.reborrow_mut() }
            .next_kv()
            .ok()?
            .into_kv_mut();
        Some(kv)
    }

    /// Returns a mutable reference to the key and value of the element that
    /// [`CursorMutKey::prev`] would move over, without moving the cursor.
    pub fn peek_prev(&mut self) -> Option<(&mut K, &mut V)> {
        let current = self.current.as_mut()?;
        let kv = unsafe { current.reborrow_mut() }
            .next_back_kv()
            .ok()?
            .into_kv_mut();
        Some(kv)
    }

    /// Converts the mutable cursor into an immutable cursor.
    ///
    /// This is useful for looking ahead without modifying the cursor itself.
    pub fn as_cursor(&self) -> Cursor<'_, K, V> {
        Cursor {
            // SAFETY: The tree is immutable while the cursor exists.
            root: unsafe { self.root.reborrow_shared().as_ref() },
            current: self.current.as_ref().map(|current| current.reborrow()),
        }
    }
}

impl<K: Ord, V, A: Allocator> CursorMutKey<'_, K, V, A> {
    /// Similar to [`insert_after_unchecked`] but also returns the key and
    /// value on error.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns the key and value along with an [`AllocError`] and
    /// leaves both the map and the cursor unchanged.
    ///
    /// # Safety
    ///
    /// Please see [`insert_after_unchecked`] for the safety requirements.
    ///
    /// [`insert_after_unchecked`]: Self::insert_after_unchecked
    pub unsafe fn insert_after_unchecked_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), (K, V, AllocError)> {
        if *self.length == usize::MAX {
            return Err((key, value, AllocError));
        }
        // SAFETY: The handle, root, allocator and length all come from the same
        // map, and the caller has upheld the map invariants.
        let handle = self.current.take();
        match unsafe {
            entry::do_two_phase(
                handle,
                &mut self.root,
                key,
                value,
                &*self.alloc,
                &mut *self.length,
            )
        } {
            Ok(new) => {
                self.current = Some(new.left_edge());
                Ok(())
            }
            Err((key, value, e, restored)) => {
                // The tree is untouched; put the cursor back where it was.
                self.current = restored;
                Err((key, value, e))
            }
        }
    }

    /// Inserts a new key-value pair into the map in the gap that the cursor
    /// is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns an [`AllocError`] and leaves both the map and the
    /// cursor unchanged. The key and value are dropped.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeMap` invariants are maintained.
    /// Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All keys in the tree must remain in sorted order.
    pub unsafe fn insert_after_unchecked(&mut self, key: K, value: V) -> Result<(), AllocError> {
        // SAFETY: precondition from the caller.
        unsafe {
            self.insert_after_unchecked_give_back(key, value)
                .map_err(|(_k, _v, e)| e)
        }
    }

    /// Similar to [`insert_before_unchecked`] but also returns the key and
    /// value on error.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns the key and value along with an [`AllocError`] and
    /// leaves both the map and the cursor unchanged.
    ///
    /// # Safety
    ///
    /// Please see [`insert_before_unchecked`] for the safety requirements.
    ///
    /// [`insert_before_unchecked`]: Self::insert_before_unchecked
    pub unsafe fn insert_before_unchecked_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), (K, V, AllocError)> {
        if *self.length == usize::MAX {
            return Err((key, value, AllocError));
        }
        let handle = self.current.take();
        // SAFETY: The handle, root, allocator and length all come from the same
        // map, and the caller has upheld the map invariants.
        match unsafe {
            entry::do_two_phase(
                handle,
                &mut self.root,
                key,
                value,
                &*self.alloc,
                &mut *self.length,
            )
        } {
            Ok(new) => {
                self.current = Some(new.right_edge());
                Ok(())
            }
            Err((key, value, e, restored)) => {
                // The tree is untouched; put the cursor back where it was.
                self.current = restored;
                Err((key, value, e))
            }
        }
    }

    /// Inserts a new key-value pair into the map in the gap that the cursor
    /// is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// # Errors
    ///
    /// On out-of-memory conditions or length overflows, this operation
    /// returns an [`AllocError`] and leaves both the map and the
    /// cursor unchanged. The key and value are dropped.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeMap` invariants are maintained.
    /// Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All keys in the tree must remain in sorted order.
    pub unsafe fn insert_before_unchecked(&mut self, key: K, value: V) -> Result<(), AllocError> {
        // SAFETY: precondition from the caller.
        unsafe {
            self.insert_before_unchecked_give_back(key, value)
                .map_err(|(_k, _v, e)| e)
        }
    }
}

impl<K: Ord, V, A: Allocator> CursorMutKey<'_, K, V, A> {
    /// Similar to [`insert_after`] but also returns the key and value on
    /// error.
    ///
    /// # Errors
    ///
    /// If the key is not properly ordered relative to its neighbors, or on
    /// out-of-memory conditions or length overflows, this operation returns
    /// the key and value along with a [`TryBTreeMapCursorInsertError`] and
    /// leaves both the map and the cursor unchanged.
    ///
    /// [`insert_after`]: Self::insert_after
    pub fn insert_after_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), (K, V, TryBTreeMapCursorInsertError)> {
        if let Some((prev, _)) = self.peek_prev() {
            if &key <= prev {
                return Err((key, value, TryBTreeMapCursorInsertError::Unordered));
            }
        }
        if let Some((next, _)) = self.peek_next() {
            if &key >= next {
                return Err((key, value, TryBTreeMapCursorInsertError::Unordered));
            }
        }
        // SAFETY: Ensured by checks above.
        unsafe { self.insert_after_unchecked_give_back(key, value) }
            .map_err(|(key, value, e)| (key, value, e.into()))
    }

    /// Inserts a new key-value pair into the map in the gap that the cursor
    /// is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// This is safer than [`insert_after_unchecked`](Self::insert_after_unchecked),
    /// but runs additional checks.
    ///
    /// # Errors
    ///
    /// If the key is not properly ordered relative to its neighbors, or if
    /// the allocator cannot accommodate the insertion, the key and value are
    /// dropped and the map and cursor are left unchanged, and the
    /// [`TryBTreeMapCursorInsertError`] is returned.
    pub fn insert_after(&mut self, key: K, value: V) -> Result<(), TryBTreeMapCursorInsertError> {
        self.insert_after_give_back(key, value)
            .map_err(|(_k, _v, e)| e)
    }

    /// Similar to [`insert_before`] but also returns the key and value on
    /// error.
    ///
    /// # Errors
    ///
    /// If the key is not properly ordered relative to its neighbors, or on
    /// out-of-memory conditions or length overflows, this operation returns
    /// the key and value along with a [`TryBTreeMapCursorInsertError`] and
    /// leaves both the map and the cursor unchanged.
    pub fn insert_before_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), (K, V, TryBTreeMapCursorInsertError)> {
        if let Some((prev, _)) = self.peek_prev() {
            if &key <= prev {
                return Err((key, value, TryBTreeMapCursorInsertError::Unordered));
            }
        }
        if let Some((next, _)) = self.peek_next() {
            if &key >= next {
                return Err((key, value, TryBTreeMapCursorInsertError::Unordered));
            }
        }
        // SAFETY: Ensured by checks above.
        unsafe { self.insert_before_unchecked_give_back(key, value) }
            .map_err(|(key, value, e)| (key, value, e.into()))
    }

    /// Inserts a new key-value pair into the map in the gap that the cursor
    /// is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// This is safer than [`insert_before_unchecked`](Self::insert_before_unchecked),
    /// but runs additional checks.
    ///
    /// # Errors
    ///
    /// If the key is not properly ordered relative to its neighbors, or if
    /// the allocator cannot accommodate the insertion, the key and value are
    /// dropped and the map and cursor are left unchanged, and the
    /// [`TryBTreeMapCursorInsertError`] is returned.
    pub fn insert_before(&mut self, key: K, value: V) -> Result<(), TryBTreeMapCursorInsertError> {
        self.insert_before_give_back(key, value)
            .map_err(|(_k, _v, e)| e)
    }

    /// Removes the key-value pair that the cursor would return by
    /// [`next`](Self::next), if there is one.
    ///
    /// After the removal the cursor's position is unchanged.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "length is at least 1 if there is a next element"
    )]
    pub fn remove_next(&mut self) -> Option<(K, V)> {
        let current = self.current.take()?;
        if current.reborrow().next_kv().is_err() {
            self.current = Some(current);
            return None;
        }

        let mut emptied_internal_root = false;
        let (kv, pos) = current
            .next_kv()
            .ok()?
            .remove_kv_tracking(|| emptied_internal_root = true, &*self.alloc);
        self.current = Some(pos);
        *self.length -= 1;

        if emptied_internal_root {
            let root = unsafe { self.root.reborrow().as_mut().unwrap() };
            root.pop_internal_level(&*self.alloc);
        }

        Some(kv)
    }

    /// Removes the key-value pair that the cursor would return by
    /// [`prev`](Self::prev), if there is one.
    ///
    /// After the removal the cursor's position is unchanged.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "length is at least 1 if there is a previous element"
    )]
    pub fn remove_prev(&mut self) -> Option<(K, V)> {
        let current = self.current.take()?;
        if current.reborrow().next_back_kv().is_err() {
            self.current = Some(current);
            return None;
        }

        let mut emptied_internal_root = false;
        let (kv, pos) = current
            .next_back_kv()
            .ok()?
            .remove_kv_tracking(|| emptied_internal_root = true, &*self.alloc);
        self.current = Some(pos);
        *self.length -= 1;

        if emptied_internal_root {
            let root = unsafe { self.root.reborrow().as_mut().unwrap() };
            root.pop_internal_level(&*self.alloc);
        }

        Some(kv)
    }
}

// ---------------------------------------------------------------------------
// BTreeMap cursor constructors
// ---------------------------------------------------------------------------

impl<K, V, A: Allocator> BTreeMap<K, V, A> {
    /// Returns a [`Cursor`] pointing at the gap before the smallest key
    /// greater than or equal to the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest key greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest key greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest key in the map.
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let root_node = match self.root.as_ref() {
            None => {
                return Cursor {
                    current: None,
                    root: None,
                };
            }
            Some(root) => root.reborrow(),
        };
        let edge = root_node.lower_bound(SearchBound::from_range(bound));
        Cursor {
            current: Some(edge),
            root: self.root.as_ref(),
        }
    }

    /// Returns a [`CursorMut`] pointing at the gap before the smallest key
    /// greater than or equal to the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest key greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest key greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest key in the map.
    pub fn lower_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, A>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let (root, dormant_root) = DormantMutRef::new(&mut self.root);
        let root_node = match root.as_mut() {
            None => {
                return CursorMut {
                    inner: CursorMutKey {
                        current: None,
                        root: dormant_root,
                        length: &mut self.length,
                        alloc: &mut *self.alloc,
                    },
                };
            }
            Some(root) => root.borrow_mut(),
        };
        let edge = root_node.lower_bound(SearchBound::from_range(bound));
        CursorMut {
            inner: CursorMutKey {
                current: Some(edge),
                root: dormant_root,
                length: &mut self.length,
                alloc: &mut *self.alloc,
            },
        }
    }

    /// Returns a [`Cursor`] pointing at the gap after the greatest key
    /// smaller than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest key smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest key smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest key in the map.
    pub fn upper_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let root_node = match self.root.as_ref() {
            None => {
                return Cursor {
                    current: None,
                    root: None,
                };
            }
            Some(root) => root.reborrow(),
        };
        let edge = root_node.upper_bound(SearchBound::from_range(bound));
        Cursor {
            current: Some(edge),
            root: self.root.as_ref(),
        }
    }

    /// Returns a [`CursorMut`] pointing at the gap after the greatest key
    /// smaller than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest key smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest key smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest key in the map.
    pub fn upper_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, A>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let (root, dormant_root) = DormantMutRef::new(&mut self.root);
        let root_node = match root.as_mut() {
            None => {
                return CursorMut {
                    inner: CursorMutKey {
                        current: None,
                        root: dormant_root,
                        length: &mut self.length,
                        alloc: &mut *self.alloc,
                    },
                };
            }
            Some(root) => root.borrow_mut(),
        };
        let edge = root_node.upper_bound(SearchBound::from_range(bound));
        CursorMut {
            inner: CursorMutKey {
                current: Some(edge),
                root: dormant_root,
                length: &mut self.length,
                alloc: &mut *self.alloc,
            },
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

    use super::super::map::BTreeMap;
    use super::{Cursor, CursorMutKey, TryBTreeMapCursorInsertError};
    use crate::alloc::Global;

    fn make_map() -> BTreeMap<i32, u32, Global> {
        let mut map = BTreeMap::new_in(Global);
        for i in 1..=4 {
            map.try_insert(i, i as u32 * 10).unwrap();
        }
        map
    }

    #[test]
    fn navigation_forward() {
        let map = make_map();
        let mut cursor = map.lower_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_prev(), None);
        let mut seen = std::vec::Vec::new();
        while let Some((k, v)) = cursor.next() {
            seen.push((*k, *v));
        }
        assert_eq!(seen, std::vec![(1, 10), (2, 20), (3, 30), (4, 40)]);
        assert_eq!(cursor.peek_next(), None);
        assert_eq!(cursor.next(), None);
    }

    #[test]
    fn navigation_backward() {
        let map = make_map();
        let mut cursor = map.upper_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_next(), None);
        let mut seen = std::vec::Vec::new();
        while let Some((k, v)) = cursor.prev() {
            seen.push((*k, *v));
        }
        assert_eq!(seen, std::vec![(4, 40), (3, 30), (2, 20), (1, 10)]);
        assert_eq!(cursor.peek_prev(), None);
        assert_eq!(cursor.prev(), None);
    }

    #[test]
    fn lower_upper_bounds() {
        let map = make_map();

        let cursor = map.lower_bound(Bound::Included(&2));
        assert_eq!(cursor.peek_prev(), Some((&1, &10)));
        assert_eq!(cursor.peek_next(), Some((&2, &20)));

        let cursor = map.lower_bound(Bound::Excluded(&2));
        assert_eq!(cursor.peek_prev(), Some((&2, &20)));
        assert_eq!(cursor.peek_next(), Some((&3, &30)));

        let cursor = map.lower_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_prev(), None);
        assert_eq!(cursor.peek_next(), Some((&1, &10)));

        let cursor = map.upper_bound(Bound::Included(&3));
        assert_eq!(cursor.peek_prev(), Some((&3, &30)));
        assert_eq!(cursor.peek_next(), Some((&4, &40)));

        let cursor = map.upper_bound(Bound::Excluded(&3));
        assert_eq!(cursor.peek_prev(), Some((&2, &20)));
        assert_eq!(cursor.peek_next(), Some((&3, &30)));

        let cursor = map.upper_bound(Bound::Unbounded);
        assert_eq!(cursor.peek_prev(), Some((&4, &40)));
        assert_eq!(cursor.peek_next(), None);
    }

    #[test]
    fn cursor_clone_and_debug() {
        let map = make_map();
        let cursor = map.lower_bound(Bound::Included(&2));
        let mut cloned = cursor.clone();
        assert_eq!(cloned.peek_next(), Some((&2, &20)));
        assert_eq!(cloned.next(), Some((&2, &20)));
        let mut cloned = cursor.try_clone().unwrap();
        assert_eq!(cloned.peek_next(), Some((&2, &20)));
        assert_eq!(cloned.next(), Some((&2, &20)));
        // The original cursor is unaffected by consuming the clone.
        assert_eq!(cursor.peek_next(), Some((&2, &20)));

        let debug = std::format!("{cursor:?}");
        assert!(debug.contains("Cursor"));
    }

    #[test]
    fn cursor_mut_values() {
        let mut map = make_map();
        let mut cursor = map.lower_bound_mut(Bound::Unbounded);
        while let Some((_, v)) = cursor.next() {
            *v += 1;
        }
        assert_eq!(map.get(&1), Some(&11));
        assert_eq!(map.get(&4), Some(&41));
    }

    #[test]
    fn insert_unchecked_success() {
        let mut map = BTreeMap::new_in(Global);

        // Empty map: insertion at the very start.
        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.lower_bound_mut(Bound::Unbounded).with_mutable_key() };
            unsafe { cursor.insert_after_unchecked(2, 20) }.unwrap();
            // The cursor now points at the gap before the new element.
            assert_eq!(cursor.peek_prev(), None);
            assert_eq!(cursor.peek_next(), Some((&mut 2, &mut 20)));

            unsafe { cursor.insert_before_unchecked(1, 10) }.unwrap();
            // The cursor now points at the gap after the new element.
            assert_eq!(cursor.peek_prev(), Some((&mut 1, &mut 10)));
            assert_eq!(cursor.peek_next(), Some((&mut 2, &mut 20)));

            assert_eq!(cursor.next(), Some((&mut 2, &mut 20)));
            unsafe { cursor.insert_after_unchecked(3, 30) }.unwrap();
            assert_eq!(cursor.peek_prev(), Some((&mut 2, &mut 20)));
            assert_eq!(cursor.peek_next(), Some((&mut 3, &mut 30)));
        }
        assert_eq!(map.len(), 3);
        let collected: std::vec::Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(collected, std::vec![(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn insert_unchecked_give_back_success() {
        let mut map = BTreeMap::new_in(Global);

        // Empty map: insertion at the very start.
        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.lower_bound_mut(Bound::Unbounded).with_mutable_key() };
            unsafe { cursor.insert_after_unchecked_give_back(2, 20) }.unwrap();
            // The cursor now points at the gap before the new element.
            assert_eq!(cursor.peek_prev(), None);
            assert_eq!(cursor.peek_next(), Some((&mut 2, &mut 20)));

            unsafe { cursor.insert_before_unchecked_give_back(1, 10) }.unwrap();
            // The cursor now points at the gap after the new element.
            assert_eq!(cursor.peek_prev(), Some((&mut 1, &mut 10)));
            assert_eq!(cursor.peek_next(), Some((&mut 2, &mut 20)));

            assert_eq!(cursor.next(), Some((&mut 2, &mut 20)));
            unsafe { cursor.insert_after_unchecked_give_back(3, 30) }.unwrap();
            assert_eq!(cursor.peek_prev(), Some((&mut 2, &mut 20)));
            assert_eq!(cursor.peek_next(), Some((&mut 3, &mut 30)));
        }
        assert_eq!(map.len(), 3);
        let collected: std::vec::Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(collected, std::vec![(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn insert_unchecked_alloc_failure() {
        let alloc = BudgetedAlloc::new(1 << 20);
        let mut map: BTreeMap<i32, u32, BudgetedAlloc> = BTreeMap::new_in(alloc.clone());
        // Fill the root leaf completely, so that the next insertion has to
        // split it and therefore really allocates.
        for i in 1..=11 {
            map.try_insert(i, i as u32 * 10).unwrap();
        }

        {
            let mut cursor: CursorMutKey<i32, u32, BudgetedAlloc> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            // Exhaust the budget so any further allocation fails.
            alloc.drain();
            assert_eq!(
                unsafe { cursor.insert_after_unchecked(12, 120) },
                Err(AllocError)
            );
            assert_eq!(
                unsafe { cursor.insert_before_unchecked(12, 120) },
                Err(AllocError)
            );
            // The cursor is left where it was, and the map is unchanged.
            assert_eq!(cursor.peek_prev(), Some((&mut 11, &mut 110)));
        }
        assert_eq!(map.len(), 11);

        // After the budget is restored, the same insertion succeeds.
        alloc.set_budget(1 << 20);
        {
            let mut cursor: CursorMutKey<i32, u32, BudgetedAlloc> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            unsafe { cursor.insert_after_unchecked(12, 120) }.unwrap();
        }
        assert_eq!(map.len(), 12);
        assert_eq!(map.get(&12), Some(&120));
    }

    #[test]
    fn insert_unchecked_give_back_alloc_failure() {
        let alloc = BudgetedAlloc::new(1 << 20);
        let mut map: BTreeMap<i32, u32, BudgetedAlloc> = BTreeMap::new_in(alloc.clone());
        // Fill the root leaf completely, so that the next insertion has to
        // split it and therefore really allocates.
        for i in 1..=11 {
            map.try_insert(i, i as u32 * 10).unwrap();
        }

        {
            let mut cursor: CursorMutKey<i32, u32, BudgetedAlloc> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            // Exhaust the budget so any further allocation fails.
            alloc.drain();
            match unsafe { cursor.insert_after_unchecked_give_back(12, 120) } {
                Err((k, v, AllocError)) => {
                    assert_eq!((k, v), (12, 120));
                }
                Ok(()) => panic!("insertion should fail when the budget is exhausted"),
            }
            match unsafe { cursor.insert_before_unchecked_give_back(12, 120) } {
                Err((k, v, AllocError)) => {
                    assert_eq!((k, v), (12, 120));
                }
                Ok(()) => panic!("insertion should fail when the budget is exhausted"),
            }
            assert_eq!(cursor.peek_prev(), Some((&mut 11, &mut 110)));
        }
        assert_eq!(map.len(), 11);

        // After the budget is restored, the same insertion succeeds.
        alloc.set_budget(1 << 20);
        {
            let mut cursor: CursorMutKey<i32, u32, BudgetedAlloc> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            unsafe { cursor.insert_after_unchecked_give_back(12, 120) }.unwrap();
        }
        assert_eq!(map.len(), 12);
        assert_eq!(map.get(&12), Some(&120));
    }

    #[test]
    fn checked_insert_success() {
        let mut map = make_map();

        // Gap before 1: key 0 fits.
        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.lower_bound_mut(Bound::Included(&1)).with_mutable_key() };
            cursor.insert_after(0, 5).unwrap();
            assert_eq!(cursor.peek_prev(), None);
            assert_eq!(cursor.peek_next(), Some((&mut 0, &mut 5)));
        }
        assert_eq!(map.len(), 5);

        // Gap after 4: key 9 fits.
        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            cursor.insert_before(9, 90).unwrap();
            assert_eq!(cursor.peek_prev(), Some((&mut 9, &mut 90)));
            assert_eq!(cursor.peek_next(), None);
        }
        assert_eq!(map.len(), 6);
        let collected: std::vec::Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(
            collected,
            std::vec![(0, 5), (1, 10), (2, 20), (3, 30), (4, 40), (9, 90)]
        );
    }

    #[test]
    fn checked_insert_unordered() {
        let mut map = make_map();

        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.lower_bound_mut(Bound::Included(&2)).with_mutable_key() };
            assert_eq!(
                cursor.insert_after(1, 0),
                Err(TryBTreeMapCursorInsertError::Unordered)
            );
            assert_eq!(
                cursor.insert_before(2, 0),
                Err(TryBTreeMapCursorInsertError::Unordered)
            );
        }
        assert_eq!(map.len(), 4);

        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            // The key must be greater than the previous element (4) and
            // strictly less than the next one, which does not exist here.
            assert_eq!(
                cursor.insert_before(4, 0),
                Err(TryBTreeMapCursorInsertError::Unordered)
            );
            assert_eq!(
                cursor.insert_after(3, 0),
                Err(TryBTreeMapCursorInsertError::Unordered)
            );
        }
        assert_eq!(map.len(), 4);
    }

    #[test]
    fn checked_insert_alloc_failure() {
        let alloc = BudgetedAlloc::new(1 << 20);
        let mut map: BTreeMap<i32, u32, BudgetedAlloc> = BTreeMap::new_in(alloc.clone());
        // Fill the root leaf completely, so that the next insertion has to
        // split it and therefore really allocates.
        for i in 1..=11 {
            map.try_insert(i, i as u32 * 10).unwrap();
        }

        {
            let mut cursor: CursorMutKey<i32, u32, BudgetedAlloc> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            // Exhaust the budget so any further allocation fails. The key is
            // properly ordered here, so the failure must be reported as an
            // allocation error, not an ordering error.
            alloc.drain();
            assert_eq!(
                cursor.insert_after(12, 120),
                Err(TryBTreeMapCursorInsertError::Alloc(AllocError))
            );
            assert_eq!(
                cursor.insert_before(12, 120),
                Err(TryBTreeMapCursorInsertError::Alloc(AllocError))
            );
            // The cursor is left where it was, and the map is unchanged.
            assert_eq!(cursor.peek_prev(), Some((&mut 11, &mut 110)));
        }
        assert_eq!(map.len(), 11);

        // After the budget is restored, the same insertion succeeds.
        alloc.set_budget(1 << 20);
        {
            let mut cursor: CursorMutKey<i32, u32, BudgetedAlloc> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            cursor.insert_before(12, 120).unwrap();
        }
        assert_eq!(map.len(), 12);
        assert_eq!(map.get(&12), Some(&120));
    }

    #[test]
    fn checked_insert_give_back_unordered() {
        let mut map = make_map(); // {1: 10, 2: 20, 3: 30, 4: 40}
        {
            let mut cursor: CursorMutKey<i32, u32> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            // An out-of-order key is rejected and handed back.
            match cursor.insert_after_give_back(2, 200) {
                Err((k, v, TryBTreeMapCursorInsertError::Unordered)) => {
                    assert_eq!((k, v), (2, 200));
                }
                _ => panic!("out-of-order insertion should fail"),
            }
            match cursor.insert_before_give_back(1, 100) {
                Err((k, v, TryBTreeMapCursorInsertError::Unordered)) => {
                    assert_eq!((k, v), (1, 100));
                }
                _ => panic!("out-of-order insertion should fail"),
            }
            // In-order insertion succeeds, and the non-give-back wrappers
            // behave identically on the happy path.
            cursor.insert_after_give_back(5, 50).unwrap();
        }
        {
            let mut cursor: CursorMutKey<i32, u32> =
                unsafe { map.lower_bound_mut(Bound::Unbounded).with_mutable_key() };
            cursor.insert_before(0, 0).unwrap();
        }
        assert_eq!(map.len(), 6);
        assert_eq!(map.get(&5), Some(&50));
        assert_eq!(map.get(&0), Some(&0));
    }

    #[test]
    fn checked_insert_give_back_alloc_failure() {
        let alloc = BudgetedAlloc::new(1 << 20);
        let mut map: BTreeMap<i32, u32, BudgetedAlloc> = BTreeMap::new_in(alloc.clone());
        // Fill the root leaf completely, so that the next insertion has to
        // split it and therefore really allocates.
        for i in 1..=11 {
            map.try_insert(i, i as u32 * 10).unwrap();
        }

        {
            let mut cursor: CursorMutKey<i32, u32, BudgetedAlloc> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            // Exhaust the budget so any further allocation fails. The key is
            // properly ordered here, so the failure must be reported as an
            // allocation error, not an ordering error.
            alloc.drain();
            match cursor.insert_after_give_back(12, 120) {
                Err((k, v, TryBTreeMapCursorInsertError::Alloc(AllocError))) => {
                    assert_eq!((k, v), (12, 120));
                }
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            match cursor.insert_before_give_back(12, 120) {
                Err((k, v, TryBTreeMapCursorInsertError::Alloc(AllocError))) => {
                    assert_eq!((k, v), (12, 120));
                }
                _ => panic!("insertion should fail when the budget is exhausted"),
            }
            // The cursor is left where it was, and the map is unchanged.
            assert_eq!(cursor.peek_prev(), Some((&mut 11, &mut 110)));
        }
        assert_eq!(map.len(), 11);

        // After the budget is restored, the same insertion succeeds.
        alloc.set_budget(1 << 20);
        {
            let mut cursor: CursorMutKey<i32, u32, BudgetedAlloc> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            cursor.insert_after_give_back(12, 120).unwrap();
        }
        assert_eq!(map.len(), 12);
        assert_eq!(map.get(&12), Some(&120));
    }

    #[test]
    fn remove_next_and_prev() {
        let mut map = make_map();

        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.lower_bound_mut(Bound::Unbounded).with_mutable_key() };
            assert_eq!(cursor.remove_next(), Some((1, 10)));
            // The cursor stays in place, now facing the element that followed.
            assert_eq!(cursor.peek_prev(), None);
            assert_eq!(cursor.peek_next(), Some((&mut 2, &mut 20)));
        }
        assert_eq!(map.len(), 3);

        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.upper_bound_mut(Bound::Unbounded).with_mutable_key() };
            assert_eq!(cursor.remove_next(), None);
            cursor.prev();
            assert_eq!(cursor.remove_prev(), Some((3, 30)));
            assert_eq!(cursor.peek_prev(), Some((&mut 2, &mut 20)));
            assert_eq!(cursor.peek_next(), Some((&mut 4, &mut 40)));
        }
        assert_eq!(map.len(), 2);

        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.lower_bound_mut(Bound::Unbounded).with_mutable_key() };
            assert_eq!(cursor.remove_prev(), None);
        }
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn remove_across_internal_levels() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..=20 {
            map.try_insert(i, i as u32).unwrap();
        }
        // 21 elements force the root to split into an internal node.
        assert!(map.len() > 11);

        {
            let mut cursor: CursorMutKey<i32, u32, Global> =
                unsafe { map.lower_bound_mut(Bound::Unbounded).with_mutable_key() };
            let mut seen = std::vec::Vec::new();
            while let Some((k, v)) = cursor.remove_next() {
                seen.push((k, v));
            }
            assert_eq!(seen.len(), 21);
        }
        assert_eq!(map.len(), 0);
        // The map must still work after the internal root level was popped.
        map.try_insert(7, 70).unwrap();
        assert_eq!(map.get(&7), Some(&70));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn cursor_mut_as_cursor() {
        let mut map = make_map();
        let mut cursor_mut = map.lower_bound_mut(Bound::Unbounded);
        let mut cursor: Cursor<i32, u32> = cursor_mut.as_cursor();
        assert_eq!(cursor.next(), Some((&1, &10)));
        // The mutable cursor and its immutable view agree on position.
        assert_eq!(cursor_mut.peek_next(), Some((&1, &mut 10)));
    }
}
