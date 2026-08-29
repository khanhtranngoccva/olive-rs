//! [`TryClone`]: a fallible analogue of [`core::clone::Clone`].

use crate::alloc_errors::{AllocError, TryReserveError};
use core::fmt;

/// Error returned when a fallible clone operation fails.
#[derive(Clone, PartialEq, Eq)]
pub enum TryCloneError {
    /// A capacity reservation on a collection failed (overflow or OOM).
    Reserve(TryReserveError),
    /// A single heap allocation failed (no reserve phase — e.g. a leaf
    /// allocation such as a `Box`, `Arc`, or `Rc` node).
    Alloc(AllocError),
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Debug for TryCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f.debug_tuple("TryCloneError::Reserve").field(e).finish(),
            Self::Alloc(e) => f.debug_tuple("TryCloneError::Alloc").field(e).finish(),
            Self::Other(msg) => f.debug_tuple("TryCloneError::Other").field(msg).finish(),
        }
    }
}

impl fmt::Display for TryCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "clone failed: {e}"),
            Self::Alloc(_) => write!(f, "clone failed: memory allocation failed"),
            Self::Other(msg) => write!(f, "clone failed: {msg}"),
        }
    }
}

impl core::error::Error for TryCloneError {}

impl From<TryReserveError> for TryCloneError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<AllocError> for TryCloneError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

/// A fallible analogue of [`core::clone::Clone`].
///
/// Unlike [`Clone`], which panics on allocation failure, [`TryClone`] returns a
/// [`Result`] so callers can handle out-of-memory gracefully.
///
/// Implementors must ensure that `try_clone` never panics — inner values should
/// also be cloned via [`TryClone`] rather than [`Clone`].
///
/// # Laziness
///
/// If cloning requires allocating memory (e.g. growing a buffer), reserve the
/// backing storage **before** performing any logical work such as recursively
/// cloning inner fields if possible. This way an allocation failure
/// short-circuits early and avoids wasted computation or intermediate values.
pub trait TryClone: Sized {
    /// Attempt to clone `self`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if a capacity reservation or allocation fails.
    fn try_clone(&self) -> Result<Self, TryCloneError>;

    /// Fallibly overwrite `self` with a copy of `source`, mirroring
    /// [`core::clone::Clone::clone_from`] but returning a [`Result`] instead of
    /// panicking on allocation failure.
    ///
    /// The default implementation clones `source` into a fresh value and swaps it
    /// in via [`core::mem::replace`], so it never leaks or double-frees even if
    /// the clone fails midway — on error, `self` is left unchanged. Types that
    /// can reuse their existing backing storage (e.g. a `Vec` growing in place)
    /// should override this for efficiency.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if cloning `source` fails; on that path `self` is
    /// guaranteed to be unchanged.
    fn try_clone_from(&mut self, source: &Self) -> Result<(), TryCloneError> {
        let new_value = source.try_clone()?;
        // Swap in the fresh value, dropping the old one. On the error path above
        // we never reach here, so `self` is left untouched.
        drop(core::mem::replace(self, new_value));
        Ok(())
    }
}

// Infallible `Copy` primitives: cloning is a bit-for-bit copy with no allocation.
macro_rules! impl_try_clone_copy {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryClone for $t {
                #[inline]
                fn try_clone(&self) -> Result<Self, TryCloneError> {
                    Ok(*self)
                }
            }
        )*
    };
}

// TODO: need macro for tuple types and various other core types

impl_try_clone_copy!(u8, u16, u32, u64, u128, usize);
impl_try_clone_copy!(i8, i16, i32, i64, i128, isize);
impl_try_clone_copy!(f32, f64);
impl_try_clone_copy!(bool, char, ());

// Immutable references to slices / str are just pointer copies — no allocation.
impl<T> TryClone for &[T] {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

impl TryClone for &str {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

impl<T: TryClone> TryClone for Option<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            Some(v) => Ok(Some(v.try_clone()?)),
            None => Ok(None),
        }
    }
}

impl<T: TryClone, E: TryClone> TryClone for Result<T, E> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            Ok(v) => Ok(Ok(v.try_clone()?)),
            Err(e) => Ok(Err(e.try_clone()?)),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::format;

    #[test]
    fn try_clone_primitives() {
        assert_eq!((42u8).try_clone().unwrap(), 42);
        assert_eq!((-5i32).try_clone().unwrap(), -5);
        assert!(true.try_clone().unwrap());
        assert_eq!('x'.try_clone().unwrap(), 'x');
    }

    #[test]
    fn try_clone_option_and_result() {
        let o: Option<i32> = Some(7);
        assert_eq!(o.try_clone().unwrap(), Some(7));
        let r: Result<i32, bool> = Ok(9);
        assert_eq!(r.try_clone().unwrap(), Ok(9));
    }

    #[test]
    fn try_clone_from_overwrites_in_place() {
        let mut a: i32 = 1;
        let b: i32 = 42;
        a.try_clone_from(&b).unwrap();
        assert_eq!(a, 42);
    }

    #[test]
    fn try_clone_from_leaves_self_unchanged_on_error() {
        // A type whose clone always fails must leave `self` untouched.
        struct Failing;
        impl TryClone for Failing {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                Err(TryCloneError::Other("always fails"))
            }
        }
        let mut a = Failing;
        let b = Failing;
        let res = a.try_clone_from(&b);
        assert!(res.is_err());
        // `a` still exists and is unchanged (identity preserved by replace).
    }

    #[test]
    fn try_clone_error_display() {
        let e = TryCloneError::Other("demo");
        assert_eq!(format!("{e}"), "clone failed: demo");
        let r = TryCloneError::Reserve(TryReserveError::new_capacity_overflow());
        assert!(format!("{r}").starts_with("clone failed"));
    }
}
