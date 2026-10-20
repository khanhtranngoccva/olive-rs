//! Re-exports of [`core::str`] types and foundational-trait impls
//! for these types.

// Re-export the whole `core::str` surface so this shadowing module stays a
// drop-in superset.
pub use core::str::*;

use crate::try_traits::{TryClone, TryCloneError};

impl TryClone for Utf8Error {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}
