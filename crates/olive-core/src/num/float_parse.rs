//! Foundational-trait impls for floating point number parsing types,
//! for example, [`core::num::ParseFloatError`].

use crate::try_traits::try_clone::{TryClone, TryCloneError};
use core::num::ParseFloatError;

impl TryClone for ParseFloatError {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(self.clone())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn parse_float_error_try_clone_is_infallible() {
        // An invalid literal yields an error; confirm the fallible clone succeeds.
        let e = "abc".parse::<f64>().unwrap_err();
        assert!(e.try_clone().is_ok());

        // A different malformed literal exercises the same path.
        let e2 = "0x".parse::<f32>().unwrap_err();
        assert!(e2.try_clone().is_ok());
    }
}
