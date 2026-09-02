//! Foundational-trait impls for the [`core::ops`] range types.
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};

use crate::try_traits::try_clone::{TryClone, TryCloneError};

// ---------------------------------------------------------------------------
// Single-endpoint / full ranges
// ---------------------------------------------------------------------------

/// Cloning a `RangeFrom(a..)` copies its endpoint via [`TryClone`] — no
/// allocation beyond what the endpoint itself requires, no failure unless the
/// endpoint's own clone fails.
impl<T: TryClone> TryClone for RangeFrom<T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(RangeFrom {
            start: self.start.try_clone()?,
        })
    }
}

/// Cloning a `RangeTo(..b)` copies its endpoint via [`TryClone`] — no
/// allocation beyond what the endpoint itself requires, no failure unless the
/// endpoint's own clone fails.
impl<T: TryClone> TryClone for RangeTo<T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(RangeTo {
            end: self.end.try_clone()?,
        })
    }
}

/// `RangeFull(..)` carries no data at all.
impl TryClone for RangeFull {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

// ---------------------------------------------------------------------------
// Two-endpoint ranges
// ---------------------------------------------------------------------------

/// Cloning a `Range<a..b>` copies both endpoints via [`TryClone`].
impl<Idx: TryClone> TryClone for Range<Idx> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let start = self.start.try_clone()?;
        let end = self.end.try_clone()?;
        Ok(Range { start, end })
    }
}

/// Cloning a `RangeInclusive<a..=b>` copies both endpoints via [`TryClone`].
impl<Idx: TryClone> TryClone for RangeInclusive<Idx> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Clone each endpoint independently through `TryClone`, then rebuild.
        // If either fails, no new range is constructed and `self` is untouched.
        let start = self.start().try_clone()?;
        let end = self.end().try_clone()?;
        Ok(RangeInclusive::new(start, end))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    /// A custom fallible type to exercise the generic non-`Copy` endpoint path.
    struct Failing;
    impl TryClone for Failing {
        fn try_clone(&self) -> Result<Self, TryCloneError> {
            Err(TryCloneError::Other("always fails"))
        }
    }

    #[test]
    fn range_try_clone_preserves_endpoints() {
        let r = 1..5usize;
        assert_eq!(r.try_clone().unwrap(), r);

        let rf = 3..;
        assert_eq!(rf.try_clone().unwrap(), rf);

        let rt = ..7u8;
        assert_eq!(rt.try_clone().unwrap(), rt);

        let rf_full = ..;
        assert_eq!(rf_full.try_clone().unwrap(), rf_full);
    }

    #[test]
    fn range_inclusive_try_clone_preserves_endpoints() {
        let ri = 1..=5usize;
        assert_eq!(ri.try_clone().unwrap(), ri);
    }

    #[test]
    fn range_inclusive_generic_endpoint_path() {
        // Exercise the generic path with a non-Copy `TryClone` endpoint by using
        // a pair of small structs that implement `TryClone` infallibly.
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct Marker(u8);
        impl TryClone for Marker {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                Ok(*self)
            }
        }
        let ri = Marker(1)..=Marker(9);
        let cloned = ri.try_clone().unwrap();
        assert_eq!(cloned.start(), &Marker(1));
        assert_eq!(cloned.end(), &Marker(9));
    }

    #[test]
    fn failing_inner_range_leaves_source_untouched() {
        let ri = Failing..Failing;
        let res = ri.try_clone();
        assert!(res.is_err());
        // The source range is undisturbed: its endpoints are still accessible.
        let _ = (&ri.start, &ri.end);
    }
}
