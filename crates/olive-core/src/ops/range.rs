//! Foundational-trait impls for the [`core::ops`] range types and `Bound`.
use core::ops::{Bound, Range, RangeFrom, RangeFull, RangeTo, RangeToInclusive};

use crate::try_traits::try_clone::{TryClone, TryCloneError};
use crate::try_traits::try_default::{TryDefault, TryDefaultError};

// ---------------------------------------------------------------------------
// Single-endpoint / full ranges
// ---------------------------------------------------------------------------

impl<T: TryClone> TryClone for RangeFrom<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(RangeFrom {
            start: self.start.try_clone()?,
        })
    }
}

impl<T: TryClone> TryClone for RangeTo<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(RangeTo {
            end: self.end.try_clone()?,
        })
    }
}

impl TryClone for RangeFull {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

impl TryDefault for RangeFull {
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(RangeFull)
    }
}

impl<T: TryClone> TryClone for RangeToInclusive<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(RangeToInclusive {
            end: self.end.try_clone()?,
        })
    }
}

// ---------------------------------------------------------------------------
// Two-endpoint ranges
// ---------------------------------------------------------------------------

impl<T: TryClone> TryClone for Range<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Range {
            start: self.start.try_clone()?,
            end: self.end.try_clone()?,
        })
    }
}

impl<Idx: TryDefault> TryDefault for Range<Idx> {
    fn try_default() -> Result<Self, TryDefaultError> {
        let start = Idx::try_default()?;
        let end = Idx::try_default()?;
        Ok(Range { start, end })
    }
}

impl<T: TryClone> TryClone for Bound<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            Bound::Included(t) => t.try_clone().map(Bound::Included),
            Bound::Excluded(t) => t.try_clone().map(Bound::Excluded),
            Bound::Unbounded => Ok(Bound::Unbounded),
        }
    }
}

// `RangeInclusive` deliberately has no `TryClone`. std's derived `Clone`
// copies the private `exhausted` flag alongside the endpoints, but we cannot
// do the same (at least without the dark layout magic).

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{MaybeUnclonable, NoDefault};

    #[test]
    fn range_from_try_clone_preserves_start() {
        let rf = MaybeUnclonable::clonable(42u8)..;
        assert_eq!(rf.try_clone().unwrap().start.inner, 42);
    }

    #[test]
    fn range_from_try_clone_propagates_failure() {
        let rf = MaybeUnclonable::unclonable(42u8)..;
        assert!(rf.try_clone().is_err());
    }

    #[test]
    fn range_to_try_clone_preserves_end() {
        let rt = ..MaybeUnclonable::clonable(42u8);
        assert_eq!(rt.try_clone().unwrap().end.inner, 42);
    }

    #[test]
    fn range_to_try_clone_propagates_failure() {
        let rt = ..MaybeUnclonable::unclonable(42u8);
        assert!(rt.try_clone().is_err());
    }

    #[test]
    fn range_full_try_clone_copies() {
        let rfull = RangeFull;
        assert_eq!(rfull.try_clone().unwrap(), rfull);
    }

    #[test]
    fn range_to_inclusive_try_clone_preserves_end() {
        let rt = ..=MaybeUnclonable::clonable(42u8);
        assert_eq!(rt.try_clone().unwrap().end.inner, 42);
    }

    #[test]
    fn range_to_inclusive_try_clone_propagates_failure() {
        let rt = ..=MaybeUnclonable::unclonable(42u8);
        assert!(rt.try_clone().is_err());
    }

    #[test]
    fn range_try_clone_preserves_endpoints() {
        let r = MaybeUnclonable::clonable(42u8)..MaybeUnclonable::clonable(43u8);
        let cloned = r.try_clone().unwrap();
        assert_eq!(cloned.start.inner, 42);
        assert_eq!(cloned.end.inner, 43);
    }

    #[test]
    fn range_try_clone_copies_generic_endpoints() {
        let r = MaybeUnclonable::clonable(1)..MaybeUnclonable::clonable(9);
        let cloned = r.try_clone().unwrap();
        assert_eq!(cloned.start.inner, 1);
        assert_eq!(cloned.end.inner, 9);
    }

    #[test]
    fn range_try_clone_propagates_failure() {
        let r = MaybeUnclonable::unclonable(1)..MaybeUnclonable::unclonable(9);
        assert!(r.try_clone().is_err());
    }

    #[test]
    fn bound_try_clone_preserves_variants() {
        assert!(matches!(
            Bound::Included(MaybeUnclonable::clonable(42u8))
                .try_clone()
                .unwrap(),
            Bound::Included(m) if m.inner == 42
        ));
        assert!(matches!(
            Bound::Excluded(MaybeUnclonable::clonable(42u8))
                .try_clone()
                .unwrap(),
            Bound::Excluded(m) if m.inner == 42
        ));
        assert!(matches!(
            Bound::<MaybeUnclonable<u8>>::Unbounded.try_clone().unwrap(),
            Bound::<MaybeUnclonable<u8>>::Unbounded
        ));
    }

    #[test]
    fn bound_try_clone_propagates_failure() {
        assert!(
            Bound::Included(MaybeUnclonable::unclonable(42))
                .try_clone()
                .is_err()
        );
        assert!(
            Bound::Excluded(MaybeUnclonable::unclonable(42))
                .try_clone()
                .is_err()
        );
    }

    #[test]
    fn range_full_try_default_is_infallible() {
        assert_eq!(RangeFull::try_default().unwrap(), RangeFull);
    }

    #[test]
    fn range_try_default_matches_default_endpoints() {
        let r: Range<u32> = Range::try_default().unwrap();
        assert_eq!(r, 0..0);
    }

    #[test]
    fn range_try_default_propagates_endpoint_failure() {
        assert!(Range::<NoDefault<u8>>::try_default().is_err());
    }
}
