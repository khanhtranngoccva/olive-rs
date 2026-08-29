//! Iterator recovery for fallible extension operations.
//!
//! When a [`TryExtend::try_extend`](crate::try_traits::TryExtend::try_extend)-style
//! operation fails, elements from the iterator may have been consumed but not yet
//! committed to the collection. This module provides [`Resume`] so that callers can
//! re-package a stranded element alongside the remainder and pass it back in.
//!
//! ## Stable types across retries
//!
//! The trait [`ResumableSource`] separates the *source* of items (which may carry
//! a head) from the *inner iterator* that produces them:
//!
//! - A plain `IntoIterator` implements [`ResumableSource`] trivially — no head,
//!   inner type is its own `IntoIter`.
//! - A [`Resume<I>`] carries an optional head and wraps the same inner iterator
//!   `I`. Its `Inner` associated type is also `I`.
//!
//! Because `try_extend` accepts anything implementing [`ResumableSource`], both
//! raw sources and [`Resume`] wrappers satisfy the same bound, and the error
//! type is always parameterized over the stable inner iterator:
//!
//! ```text
//! First call:  try_extend(range)     -> Err(Resume<Range<i32>>)
//! Retry:      try_extend(resume)     -> Err(Resume<Range<i32>>)
//! Third call: same shape again        -> Err(Resume<Range<i32>>)
//! ```
//!
//! No new generic parameters are introduced on retry — the type never grows.
//!
//! Two scenarios produce a [`Resume`]:
//!
//! 1. **Initial reserve failure** — no elements were consumed. The [`Resume`]
//!    has no head; only the full remainder iterator is present.
//!
//! 2. **Mid-iteration failure** — one element was popped but could not be
//!    inserted. The [`Resume`] holds that stranded element as the head, plus the
//!    unconsumed remainder.

use core::fmt;

/// Trait for fallible iterators that may stall on allocation errors.
///
/// By default, iterators that implement this trait *stall*: when an internal
/// allocation fails they emit an `Err` item and hold the pending work so that
/// retrying [`Iterator::next`] re-attempts the failed operation. This preserves
/// correctness — no data is silently lost.
///
/// Callers that prefer progress over completeness can opt out:
///
/// - **Automatic**: call [`with_auto_unstall`](Stall::with_auto_unstall) to get
///   back the same iterator configured so that each error is emitted once and
///   the pending item is automatically discarded. Composable since it consumes
///   and returns `Self`.
/// - **Manual**: call [`unstall`](Stall::unstall) after seeing an `Err` to
///   discard the stalled item and move on.
///
/// Stalling is the default because it never loses data. Skipping is opt-in.
pub trait Stall: Iterator {
    /// Consume a pending/stalled item, if any, so that iteration can proceed past
    /// the point of failure.
    ///
    /// Returns `true` if there was a pending item that was discarded.
    fn unstall(&mut self) -> bool;

    /// Return `self` with automatic unstalling toggled.
    ///
    /// When `auto` is `true`, each error is emitted once and the pending item is
    /// automatically discarded so iteration continues. When `auto` is `false`
    /// (the default), errors are repeated until the underlying operation succeeds
    /// or [`Self::unstall`] is called manually.
    #[must_use]
    fn with_auto_unstall(self, auto: bool) -> Self
    where
        Self: Sized,
    {
        let mut s = self;
        s.set_auto_unstall(auto);
        s
    }

    /// Set whether the iterator automatically discards pending items after
    /// emitting an error.
    fn set_auto_unstall(&mut self, auto: bool);
}

/// A source of items that decomposes into an optional leading element and an
/// inner iterator.
///
/// Any `IntoIterator` implements this via blanket — no head, inner is itself.
/// [`Resume<I>`] implements it explicitly, carrying an optional head while still
/// exposing the same inner iterator `I`.
pub trait ResumableSource {
    /// The item type produced by this source.
    type Item;
    /// The stable inner iterator type. For a plain iterator this is `Self`; for
    /// a [`Resume<I>`] this is `I`.
    type Inner: Iterator<Item = Self::Item>;

    /// Decompose into an optional leading element and the inner iterator.
    fn safe_into_iter(self) -> (Option<Self::Item>, Self::Inner);
}

impl<I: IntoIterator> ResumableSource for I {
    type Item = I::Item;
    type Inner = I::IntoIter;

    #[inline]
    fn safe_into_iter(self) -> (Option<Self::Item>, Self::Inner) {
        (None, self.into_iter())
    }
}

/// Wraps an optional stranded element alongside a remainder iterator, allowing
/// the caller to pass both back into a fallible extend operation.
///
/// Constructed either directly by the caller or returned inside the error of a
/// failed `try_extend`. Implements [`ResumableSource`] with `Inner = I`, so
/// passing it back into `try_extend` preserves the same error type.
///
/// # Example
///
/// ```rust,ignore
/// use olive_core::recovery::Resume;
///
/// let mut vec = Vec::<i32>::new();
/// let items = 0..10_000;
///
/// // First call — `remaining` is Range<i32>.
/// let remaining = match vec.try_extend(items) {
///     Ok(()) => return,
///     Err((resume, _err)) => resume.into_remainder(),
/// };
///
/// // Retry — construct a Resume from whatever we have left.
/// let remaining = match vec.try_extend(Resume::from_remainder(remaining)) {
///     Ok(()) => return,
///     Err((resume, _err)) => resume.into_remainder(),
/// };
/// ```
pub struct Resume<I>
where
    I: Iterator,
{
    head: Option<I::Item>,
    remainder: I,
}

impl<I> Resume<I>
where
    I: Iterator,
{
    /// Create a [`Resume`] with a stranded element and the remainder.
    pub const fn new(head: I::Item, remainder: I) -> Self {
        Self {
            head: Some(head),
            remainder,
        }
    }

    /// Create a [`Resume`] with no stranded element — only the remainder.
    pub const fn from_remainder(remainder: I) -> Self {
        Self {
            head: None,
            remainder,
        }
    }

    /// Returns `true` if there is a stranded head element.
    #[inline]
    pub const fn has_head(&self) -> bool {
        self.head.is_some()
    }

    /// Returns a reference to the stranded element, or `None`.
    #[inline]
    pub fn head(&self) -> Option<&I::Item> {
        self.head.as_ref()
    }

    /// Returns a mutable reference to the stranded element, or `None`.
    #[inline]
    pub fn head_mut(&mut self) -> Option<&mut I::Item> {
        self.head.as_mut()
    }

    /// Returns a reference to the remainder iterator.
    #[inline]
    pub fn remainder(&self) -> &I {
        &self.remainder
    }

    /// Returns a mutable reference to the remainder iterator.
    #[inline]
    pub fn remainder_mut(&mut self) -> &mut I {
        &mut self.remainder
    }

    /// Consumes this value, returning the remainder iterator. Drops the head.
    #[inline]
    pub fn into_remainder(self) -> I {
        self.remainder
    }

    /// Consumes this value, returning both parts.
    #[inline]
    pub fn into_parts(self) -> (Option<I::Item>, I) {
        (self.head, self.remainder)
    }
}

impl<I> ResumableSource for Resume<I>
where
    I: Iterator,
{
    type Item = I::Item;
    type Inner = I;

    #[inline]
    fn safe_into_iter(self) -> (Option<Self::Item>, Self::Inner) {
        (self.head, self.remainder)
    }
}

impl<I> fmt::Debug for Resume<I>
where
    I: Iterator + fmt::Debug,
    I::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Resume")
            .field("head", &self.head)
            .field("remainder", &self.remainder)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use core::ops::Range;
    use std::format;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn resume_with_head() {
        let r = Resume::new(42, 3..6);
        assert!(r.has_head());
        assert_eq!(*r.head().unwrap(), 42);
        assert_eq!(r.remainder().size_hint(), (3, Some(3)));
    }

    #[test]
    fn resume_without_head() {
        let r = Resume::from_remainder(0..5);
        assert!(!r.has_head());
        assert!(r.head().is_none());
        assert_eq!(r.remainder().size_hint(), (5, Some(5)));
    }

    #[test]
    fn safe_into_iter_yields_head_then_remainder() {
        let r = Resume::new(0, 1..4);
        let (head, mut iter) = r.safe_into_iter();
        assert_eq!(head, Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_remainder_drops_head() {
        let r = Resume::new(7, 1..4);
        let rem = r.into_remainder();
        let v: Vec<i32> = rem.collect();
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn into_parts() {
        let r = Resume::new(99, 10..12);
        let (h, i) = r.into_parts();
        assert_eq!(h, Some(99));
        let v: Vec<i32> = i.collect();
        assert_eq!(v, vec![10, 11]);
    }

    #[test]
    fn stable_type_across_retries() {
        type Base = Range<i32>;

        let r1: Resume<Base> = Resume::new(0, 1..4);
        let (_head, inner): (_, Base) = r1.safe_into_iter();

        let r2: Resume<Base> = Resume::from_remainder(inner);
        let (_head2, mut inner2): (_, Base) = r2.safe_into_iter();

        // Still Base, never Resume<Resume<Base>>.
        assert_eq!(inner2.next(), Some(1));
    }

    #[test]
    fn blanket_source_for_range() {
        let range = 10..13;
        let (head, inner): (Option<i32>, _) = range.safe_into_iter();
        assert!(head.is_none());
        let v: Vec<i32> = inner.collect();
        assert_eq!(v, vec![10, 11, 12]);
    }

    #[test]
    fn debug_output() {
        let r = Resume::new(42, 1..3);
        let s = format!("{r:?}");
        assert!(s.contains("Resume"));
        assert!(s.contains("Some(42)"));
    }
}
