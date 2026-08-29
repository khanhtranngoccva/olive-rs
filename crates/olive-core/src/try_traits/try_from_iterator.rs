//! [`TryFromIterator`]: a fallible analogue of [`core::iter::FromIterator`].

/// A fallible analogue of [`core::iter::FromIterator`].
///
/// Builds a value from an iterator, returning a [`Result`] if any step (typically
/// a capacity reservation) fails. Unlike [`FromIterator`], the builder does not
/// panic on out-of-memory.
pub trait TryFromIterator<Item>: Sized {
    /// The error type defined by the implementation.
    type Error;

    /// Build a value from `iter`, failing on allocation error.
    ///
    /// # Errors
    ///
    /// Returns the implementation-defined [`Self::Error`].
    fn try_from_iter<I: IntoIterator<Item = Item>>(iter: I) -> Result<Self, Self::Error>;
}
