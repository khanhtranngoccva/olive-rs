//! A container holding real data whose `TryDefault` always fails.

use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

/// A container holding a real `T` whose [`TryDefault`] always fails.
///
/// The impl is unconditional: the failure has nothing to do with `T`, so
/// `NoDefault<T>` works as a value type even when `T` itself is not
/// `TryDefault`.
#[derive(Debug)]
pub struct NoDefault<T>(pub T);

impl<T> NoDefault<T> {
    /// Wraps `inner` in a container that refuses to default-construct.
    pub const fn new(inner: T) -> Self {
        Self(inner)
    }
}

impl<T> TryDefault for NoDefault<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Err(TryDefaultError::Other("no canonical value"))
    }
}
