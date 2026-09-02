//! Foundational-trait impls for the [`core::num`] non-zero integer newtypes.

use core::num::{
    NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize, NonZeroU8,
    NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize,
};

// Each concrete-width newtype gets a `TryClone` impl via the macro below. A
// generic `NonZero<T>` impl is not possible: `NonZero<T>` is bounded on the
// perma-unstable sealed trait `ZeroablePrimitive`, which downstream crates
// cannot name or satisfy, so a blanket `impl<T> TryClone for NonZero<T>` would
// be rejected by the compiler. The 12 concrete widths cover all practical uses.
// None of them get a `TryDefault` — there is no canonical non-zero value to
// default to, so we omit the impl rather than always fail (consistent with the
// `UnsafeCell` stance).
use crate::try_traits::try_clone::{TryClone, TryCloneError};

macro_rules! impl_nonzero {
    ($($t:ty),* $(,)?) => {
        $(
            /// Cloning a non-zero integer is a bit-for-bit copy — no allocation,
            /// no failure.
            impl TryClone for $t {
                #[inline]
                fn try_clone(&self) -> Result<Self, TryCloneError> {
                    Ok(*self)
                }
            }
        )*
    };
}

impl_nonzero!(
    NonZeroI8,
    NonZeroI16,
    NonZeroI32,
    NonZeroI64,
    NonZeroI128,
    NonZeroIsize,
    NonZeroU8,
    NonZeroU16,
    NonZeroU32,
    NonZeroU64,
    NonZeroU128,
    NonZeroUsize,
);

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use core::num::NonZero;

    #[test]
    fn nonzero_try_clone_preserves_value() {
        let x = NonZeroU32::new(5).unwrap();
        assert_eq!(x.try_clone().unwrap(), x);

        let y = NonZeroI64::new(-7).unwrap();
        assert_eq!(y.try_clone().unwrap(), y);

        let z = NonZeroI128::new(1 << 60).unwrap();
        assert_eq!(z.try_clone().unwrap(), z);
    }

    #[test]
    fn nonzero_generic_form_try_clone_over_allowed_widths() {
        // The concrete-width newtypes (`NonZeroU32`, …) *are* `NonZero<{width}>`
        // — not aliases, but the identical type. So these tests exercise the
        // generic-form constructor path even though a blanket
        // `impl<T> TryClone for NonZero<T>` is impossible (sealed
        // `ZeroablePrimitive` bound). This confirms that user code writing
        // `NonZero::<u32>::new(..)` gets working `TryClone` behaviour.
        let x = NonZero::<u32>::new(5).unwrap();
        assert_eq!(x.try_clone().unwrap(), x);

        let y = NonZero::<i64>::new(-7).unwrap();
        assert_eq!(y.try_clone().unwrap(), y);

        let z = NonZero::<usize>::new(usize::MAX).unwrap();
        assert_eq!(z.try_clone().unwrap(), z);
    }
}
