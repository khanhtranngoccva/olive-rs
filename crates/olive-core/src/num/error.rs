//! Foundational-trait impls for the [`core::num`] integer-conversion error types.

use crate::try_traits::try_clone::{TryClone, TryCloneError};
use core::num::{IntErrorKind, ParseIntError, TryFromIntError};

// Cloning is trivial, it is a fieldless enum
impl TryClone for IntErrorKind {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(self.clone())
    }
}

// Cloning is trivial, it is a fieldless enum
impl TryClone for ParseIntError {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(self.clone())
    }
}

// Cloning is trivial, it is a fieldless enum
impl TryClone for TryFromIntError {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn int_error_kind_try_clone_preserves_variant() {
        let k = IntErrorKind::PosOverflow;
        assert_eq!(k.try_clone().unwrap(), k);
    }

    #[test]
    fn parse_int_error_try_clone_preserves_kind() {
        // An empty string parses to the `Empty` kind.
        let e = "".parse::<i32>().unwrap_err();
        let cloned = e.try_clone().unwrap();
        assert_eq!(*e.kind(), *cloned.kind());
    }

    #[test]
    fn try_from_int_error_try_clone_is_infallible() {
        let e = i8::try_from(300i16).unwrap_err();
        // Opaque ZST: just confirm the fallible clone succeeds without failing.
        assert!(e.try_clone().is_ok());
    }
}
