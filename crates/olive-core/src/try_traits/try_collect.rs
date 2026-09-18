//! [`TryCollect`] and [`TryCollectInto`]: fallible analogues of
//! [`Iterator::collect`] and [`Iterator::collect_into`].

use super::try_extend::TryExtend;
use super::try_from_iterator::TryFromIterator;
use crate::recovery::{ResumableSource, Resume};

/// A fallible analogue of [`Iterator::collect`].
///
/// Builds a value from an iterator, returning a [`Result`] if any step (typically
/// a capacity reservation) fails. Unlike [`FromIterator`], the builder does not
/// panic on out-of-memory.
pub trait TryCollect: Iterator + Sized {
    /// Fallibly collect an iterator into a target type.
    ///
    /// # Errors
    ///
    /// Returns an error defined by the implementation.
    fn try_collect<B>(self) -> Result<B, B::Error>
    where
        B: TryFromIterator<Self::Item>,
    {
        B::try_from_iter(self)
    }
}

/// A fallible analogue of [`Iterator::collect_into`].
///
/// Extends a value from an iterator or a [`Resume`] of an iterator, returning a [`Result`] if
/// any step (typically a capacity reservation) fails. Unlike [`Extend`], the builder does not
/// panic on out-of-memory.
pub trait TryCollectInto: ResumableSource + Sized {
    /// Fallibly extends a target type with contents from the iterator (or a [`Resume`] object).
    ///
    /// # Errors
    ///
    /// Returns a [`Resume`] object and an error defined by the implementation.
    fn try_collect_into<E>(
        self,
        collection: &mut E,
    ) -> Result<&mut E, (Resume<Self::Inner>, E::Error)>
    where
        E: TryExtend<Self::Item>,
    {
        collection.try_extend(self)?;
        Ok(collection)
    }
}

impl<T> TryCollect for T where T: Iterator + Sized {}
impl<T> TryCollectInto for T where T: ResumableSource + Sized {}
