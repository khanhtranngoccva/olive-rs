//! [`TryExtend`] and [`TryExtendFromSlice`]: fallible analogues of
//! [`core::iter::Extend`].

use crate::recovery::{ResumableSource, Resume};

/// Fallibly extend a collection from an iterator source.
///
/// This trait is declared **resumable** and **non-atomic**: on failure the error
/// carries a [`Resume`] wrapping the remainder of the source (plus any
/// consumed-but-uncommitted element), which can be passed straight back into
/// another `try_extend` call to retry. Because both plain iterators and
/// [`Resume`] wrappers implement [`ResumableSource`] with the same inner type,
/// the error type stays identical across retries — it never grows.
///
/// ```rust,ignore
/// let mut v: Vec<i32> = Vec::new();
/// let items = 0..10_000;
///
/// // First attempt — fails on OOM.
/// let remaining = match v.try_extend(items) {
///     Ok(()) => return,
///     Err((resume, _err)) => resume.into_remainder(),
/// };
///
/// // Retry with the remainder wrapped in a Resume.
/// let _remaining = match v.try_extend(Resume::from_remainder(remaining)) {
///     Ok(()) => return,
///     Err((resume, _err)) => resume.into_remainder(),
/// };
/// ```
pub trait TryExtend<Item>: Sized {
    /// The error returned on failure, paired with a [`Resume`] over the source's
    /// inner iterator so the caller can retry.
    type Error;

    /// Fallibly extend `self` with all items produced by `source`.
    ///
    /// # Errors
    /// - The [`Self::Error`] type defined by the implementation.
    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = Item>;
}

/// Fallibly extend a collection by cloning elements from a slice.
///
/// On failure the error is a tuple of the **remainder** (the unprocessed tail of
/// the input slice) and the underlying error. Callers can retry with just the
/// remainder once memory pressure has eased. A mid-way clone failure does not
/// trigger a rollback.
pub trait TryExtendFromSlice<Item>: Sized {
    /// The error type accompanying the remainder slice.
    type Error;

    /// Fallibly extend `self` by cloning each element of `other`.
    ///
    /// # Errors
    /// - The [`Self::Error`] type defined by the implementation.
    fn try_extend_from_slice<'s>(
        &mut self,
        other: &'s [Item],
    ) -> Result<(), (&'s [Item], Self::Error)>;
}
