//! [`TryDefault`]: a fallible analogue of [`core::default::Default`].

use crate::alloc::AllocError;
use crate::alloc_errors::TryReserveError;
use core::fmt;

/// Error returned when a fallible default construction fails.
///
/// Mirrors the shape of [`TryCloneError`](super::try_clone::TryCloneError):
/// a fixed set of variants covering every realistic failure mode for
/// constructing a default value, so that generic code and derive macros can
/// reason about the error uniformly without an associated type.
#[derive(Clone, PartialEq, Eq)]
pub enum TryDefaultError {
    /// A capacity reservation on a collection failed (overflow or OOM) during
    /// default construction.
    Reserve(TryReserveError),
    /// A single heap allocation failed (no reserve phase — e.g. an allocator
    /// that eagerly pools blocks).
    Alloc(AllocError),
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Debug for TryDefaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f.debug_tuple("TryDefaultError::Reserve").field(e).finish(),
            Self::Alloc(e) => f.debug_tuple("TryDefaultError::Alloc").field(e).finish(),
            Self::Other(msg) => f.debug_tuple("TryDefaultError::Other").field(msg).finish(),
        }
    }
}

impl fmt::Display for TryDefaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "default construction failed: {e}"),
            Self::Alloc(_) => write!(f, "default construction failed: memory allocation failed"),
            Self::Other(msg) => write!(f, "default construction failed: {msg}"),
        }
    }
}

impl core::error::Error for TryDefaultError {}

impl From<TryReserveError> for TryDefaultError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<AllocError> for TryDefaultError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

/// A fallible analogue of [`core::default::Default`].
///
/// Unlike [`Default`], whose implementations are required to be infallible (and
/// whose `Vec`-style constructors historically panic on allocation failure),
/// [`TryDefault`] returns a [`Result`] so that construction of an empty or
/// default value can fail gracefully when it must reserve capacity.
///
/// Most types implement this infallibly — an empty value needs no allocation at
/// all — but the error channel exists for the cases where "empty" still means
/// "allocate something" (e.g. a map that eagerly sizes its buckets, or an
/// allocator that pools blocks up front).
///
/// # Scope and limitations
///
/// This trait is intended for **plain-old-data (POD) and lightweight container**
/// types whose default construction is either trivially cheap or involves a
/// small, bounded allocation. It is *not* intended for:
///
/// - **Non-data / resource-holding types.** Their "default" would require opening a
///   file descriptor, binding a socket, or else — operations that are genuinely fallible
///   in ways unrelated to allocation and that belong in their own constructors.
///   You can add default creation behavior using idiomatic Rust approaches, including:
///     - Making the primary constructor accept a data-only configuration struct that
///       implements [`TryDefault`].
///     - You can also use the builder pattern instead. This is the approach that
///       `tokio` employs.
/// - **Types with no defaults.** Types having no meaningful default (e.g. [`core::num::NonZero`])
///   should omit this impl entirely rather than always fail.
/// 
/// These limitations allow the error mode of [`TryDefault`] to be the fixed 
/// [`TryDefaultError`] return type (rather than an associated type). This
/// enables uniform generic composition and future `#[derive(TryDefault)]`
/// support. In practice, default-construction failures are almost exclusively
/// allocation-related, which the three variants cover.
pub trait TryDefault: Sized {
    /// Construct the default value, failing instead of panicking if the
    /// construction requires a failed allocation.
    ///
    /// # Errors
    ///
    /// Returns [`TryDefaultError`] if a capacity reservation or allocation
    /// fails during construction.
    fn try_default() -> Result<Self, TryDefaultError>;
}

// Infallible defaults for primitive and marker types: building them performs no
// allocation whatsoever, so they never fail.
macro_rules! impl_try_default_infallible {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryDefault for $t {
                #[inline]
                fn try_default() -> Result<Self, TryDefaultError> {
                    Ok(Self::default())
                }
            }
        )*
    };
}

impl_try_default_infallible!(u8, u16, u32, u64, u128, usize);
impl_try_default_infallible!(i8, i16, i32, i64, i128, isize);
impl_try_default_infallible!(bool, char, (), f32, f64);

impl<T> TryDefault for Option<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn primitives_are_infallible() {
        assert_eq!(u32::try_default().unwrap(), 0);
        assert_eq!(isize::try_default().unwrap(), 0);
        assert_eq!(char::try_default().unwrap(), '\0');
        assert!(matches!(f64::try_default().unwrap(), 0.0));
        assert_eq!(Option::<i32>::try_default().unwrap(), None);
    }

    #[test]
    fn error_variants_are_distinct() {
        assert_ne!(
            TryDefaultError::Alloc(AllocError),
            TryDefaultError::Other("x")
        );
        assert_eq!(TryDefaultError::Other("a"), TryDefaultError::Other("a"));
    }

    #[test]
    fn from_alloc_error_works() {
        let err: TryDefaultError = AllocError.into();
        assert!(matches!(err, TryDefaultError::Alloc(_)));
    }

    #[test]
    fn from_reserve_error_works() {
        let reserve = TryReserveError::new_capacity_overflow();
        let err: TryDefaultError = reserve.into();
        assert!(matches!(err, TryDefaultError::Reserve(r) if r.is_capacity_overflow()));
    }
}
