//! Conversion between [`Arc`](super::Arc) and [`Weak`](super::Weak): downgrading a
//! strong reference to a weak one and upgrading a weak reference back to a strong
//! one.
//!
//! Both operations are fallible because they must clone the allocator handle
//! (via [`AllocatorTryClone`]) to attach to the resulting pointer, and the
//! accompanying counter mutation can report an out-of-bounds condition. Cloning
//! an allocator handle is a first-class fallible operation in this framework.

use core::fmt::{self, Display, Formatter};
use core::marker::PhantomData;
use core::sync::atomic::Ordering;

use crate::alloc::AllocError;
use olive_core::alloc::AllocatorTryClone;
use olive_core::try_traits::try_clone::TryCloneError;

use super::{Arc, Weak};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by fallible reference-counting conversions on [`Arc`] and
/// [`Weak`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryArcError {
    /// A counter increment would exceed `usize::MAX` or a decrement would
    /// underflow below zero.
    ///
    /// This can arise from a logic error (unbalanced inc/dec), adversarial
    /// misuse of the raw pointer APIs, or — through the safe API alone — from
    /// [`core::mem::forget`]ing enough `Arc`s that their skipped `Drop` leaves
    /// the strong count stranded near `usize::MAX`; any further increment then
    /// overflows. Such leaks are rare in practice but do make this variant
    /// reachable without undefined behavior.
    OutOfBounds,
    /// Cloning the allocator handle failed.
    CloneAlloc(TryCloneError),
}

impl From<TryCloneError> for TryArcError {
    #[inline]
    fn from(e: TryCloneError) -> Self {
        Self::CloneAlloc(e)
    }
}

impl From<AllocError> for TryArcError {
    #[inline]
    fn from(e: AllocError) -> Self {
        Self::CloneAlloc(TryCloneError::Alloc(e))
    }
}

impl Display for TryArcError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds => write!(f, "reference count out of bounds"),
            Self::CloneAlloc(e) => write!(f, "allocator clone failed: {e}"),
        }
    }
}

impl core::error::Error for TryArcError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::OutOfBounds => None,
            Self::CloneAlloc(e) => Some(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Downgrade: Arc → Weak (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: AllocatorTryClone> Arc<T, A> {
    /// Borrows an [`Arc`] as a [`Weak`] pointer.
    ///
    /// This does not increment the strong count, so the resulting `Weak` will
    /// not prevent the value from being dropped once all strong references are
    /// gone.
    ///
    /// The operation is fallible because it must clone the allocator handle
    /// (via [`AllocatorTryClone`]) to attach to the new `Weak`, and the weak
    /// counter mutation can report an out-of-bounds condition.
    ///
    /// # Errors
    ///
    /// Returns [`TryArcError::OutOfBounds`] if the weak count would overflow
    /// `usize`, or [`TryArcError::CloneAlloc`] if cloning the allocator handle
    /// fails.
    #[inline]
    pub fn try_downgrade(this: &Self) -> Result<Weak<T, A>, TryArcError> {
        let alloc = A::try_clone(&this.alloc)?;
        let inner = Self::inner(this);
        inner
            .weak
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| TryArcError::OutOfBounds)?;
        Ok(Weak {
            ptr: this.ptr,
            alloc,
            _marker: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Upgrade: Weak → Arc (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: AllocatorTryClone> Weak<T, A> {
    /// Attempts to upgrade the `Weak` reference to an [`Arc`].
    ///
    /// Returns `Ok(None)` if there are no strong (`Arc`) references left, in
    /// which case the value has been dropped.
    ///
    /// The operation is fallible because it must clone the allocator handle
    /// (via [`AllocatorTryClone`]) to attach to the new `Arc`, and the strong
    /// counter mutation can report an out-of-bounds condition.
    ///
    /// # Errors
    ///
    /// Returns [`TryArcError::OutOfBounds`] if the strong count would overflow
    /// `usize`, or [`TryArcError::CloneAlloc`] if cloning the allocator handle
    /// fails.
    #[inline]
    pub fn try_upgrade(&self) -> Result<Option<Arc<T, A>>, TryArcError> {
        // A dangling weak (from `Weak::new`) never referred to an allocation,
        // so it can never be upgraded. Checking this first also avoids ever
        // dereferencing the sentinel address below.
        let inner = match self.inner() {
            Some(i) => i,
            None => return Ok(None),
        };

        // Clone the allocator handle before touching any counter.
        let alloc = A::try_clone(&self.alloc)?;

        // Bump the strong count. Success uses `Acquire` so an upgrader is
        // synchronized-with a publisher that wrote the payload then bumped the
        // count with `Release`.
        match inner
            .strong
            .fetch_update(Ordering::Acquire, Ordering::Relaxed, |n| {
                if n != 0 { n.checked_add(1) } else { None }
            }) {
            Ok(_) => {}
            Err(current) => {
                return if current == 0 {
                    Ok(None)
                } else {
                    Err(TryArcError::OutOfBounds)
                };
            }
        }
        Ok(Some(Arc {
            ptr: self.ptr,
            alloc,
            _marker: PhantomData,
        }))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::Global;
    use crate::test_helpers::{CloneBudget, FlakyCloneAlloc};
    use std::fmt::Write as _;
    use std::rc::Rc;
    use std::string::String;

    /// Reads the payload behind an `Arc` without relying on `Deref` (which has
    /// not landed yet). Only valid while the payload is initialized.
    fn peek<T, A: AllocatorTryClone>(arc: &Arc<T, A>) -> &T {
        // SAFETY: the Arc owns a live allocation whose payload is initialized.
        #[allow(
            clippy::needless_borrow,
            reason = "Miri does not allow implicit autoref"
        )]
        unsafe {
            &(&*arc.ptr.as_ptr()).value
        }
    }

    // --- Happy paths --------------------------------------------------------

    #[test]
    fn downgrade_produces_live_weak_with_balanced_counts() {
        let arc = Arc::try_new(7i32).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        // Strong count is unchanged by a downgrade; the weak count gains one
        // explicit weak reference (the implicit one is excluded from the query).
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 1);
        // The weak sees the same payload and the same strong count.
        assert_eq!(weak.strong_count(), 1);
        assert_eq!(weak.weak_count(), 1);
        assert_eq!(Arc::as_ptr(&arc), weak.as_ptr());
    }

    #[test]
    fn upgrade_restores_strong_reference() {
        let arc = Arc::try_new(42u64).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        let upgraded = weak.try_upgrade().unwrap().expect("strong refs remain");
        // Upgrading adds exactly one strong owner.
        assert_eq!(Arc::strong_count(&upgraded), 2);
        assert_eq!(peek(&upgraded), &42);
        // The block is still shared: dropping the original leaves the upgraded
        // handle fully valid.
        drop(arc);
        assert_eq!(Arc::strong_count(&upgraded), 1);
        assert_eq!(peek(&upgraded), &42);
    }

    #[test]
    fn downgrade_then_drop_all_strongs_makes_upgrade_none() {
        let arc = Arc::try_new(1i32).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        // While a strong reference exists, upgrade succeeds.
        assert!(weak.try_upgrade().unwrap().is_some());
        drop(arc);
        // With no strong references left, the payload is dropped and upgrade
        // reports `None`; the weak still observes zero strongs.
        assert_eq!(weak.strong_count(), 0);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn multiple_upgrades_are_independent_owners() {
        let arc = Arc::try_new(9u8).unwrap();
        let w1 = Arc::try_downgrade(&arc).unwrap();
        let w2 = Arc::try_downgrade(&arc).unwrap();
        let a1 = w1.try_upgrade().unwrap().unwrap();
        let a2 = w2.try_upgrade().unwrap().unwrap();
        // Original + two upgrades = three strong owners.
        assert_eq!(Arc::strong_count(&a1), 3);
        assert_eq!(Arc::strong_count(&a2), 3);
        // All point at the same allocation.
        assert!(Arc::ptr_eq(&a1, &a2));
        assert!(Arc::ptr_eq(&a1, &arc));
    }

    // --- Dangling weak ------------------------------------------------------

    #[test]
    fn dangling_weak_upgrades_to_none() {
        // A weak from `Weak::new` never pointed at an allocation, so it can
        // never be upgraded — and doing so must not touch memory.
        let w: Weak<i32, Global> = Weak::new();
        assert!(w.try_upgrade().unwrap().is_none());
        // Unsized payloads take the identical path.
        let ws: Weak<str, Global> = Weak::new();
        assert!(ws.try_upgrade().unwrap().is_none());
        let wb: Weak<[u8], Global> = Weak::new();
        assert!(wb.try_upgrade().unwrap().is_none());
    }

    // --- Failure paths ------------------------------------------------------

    #[test]
    fn downgrade_alloc_clone_failure_leaves_counts_untouched() {
        // An allocator whose clone budget is exhausted up front forces the
        // `try_clone` to fail *before* any counter is bumped. The strong and
        // weak counts must therefore be exactly what they were before the call.
        let alloc = FlakyCloneAlloc::new(Rc::new(CloneBudget::new(0)));
        let arc = Arc::try_new_in(5i32, alloc).unwrap();
        let res = Arc::try_downgrade(&arc);
        assert!(matches!(res, Err(TryArcError::CloneAlloc(_))));
        // No weak reference was created, so the counts are unchanged.
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn upgrade_alloc_clone_failure_does_not_leak_a_strong_ref() {
        // Give the budget exactly one unit: the initial downgrade consumes it,
        // so the subsequent upgrade's allocator clone fails *before* the strong
        // count is bumped. The failed upgrade must therefore leave no extra
        // strong owner behind: after dropping `arc`, the strong count falls all
        // the way to zero (no leaked phantom ref keeping the payload alive).
        let alloc = FlakyCloneAlloc::new(Rc::new(CloneBudget::new(1)));
        let arc = Arc::try_new_in(5i32, alloc).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap(); // consumes the last budget unit
        let res = weak.try_upgrade();
        assert!(matches!(res, Err(TryArcError::CloneAlloc(_))));
        drop(arc);
        assert_eq!(weak.strong_count(), 0);
    }

    #[test]
    fn clone_budget_places_failure_deterministically() {
        // A fresh budget of 2 allows exactly two downgrades, then fails. This
        // sanity-checks the harness and shows the failure lands at a predictable
        // point in a sequence of conversions.
        let alloc = FlakyCloneAlloc::new(Rc::new(CloneBudget::new(2)));
        let arc = Arc::try_new_in(1i32, alloc).unwrap();
        assert!(Arc::try_downgrade(&arc).is_ok());
        assert!(Arc::try_downgrade(&arc).is_ok());
        assert!(Arc::try_downgrade(&arc).is_err());
    }

    #[test]
    fn error_variants_display_and_source() {
        let oob = TryArcError::OutOfBounds;
        let mut s = String::new();
        write!(&mut s, "{oob}").unwrap();
        assert_eq!(s, "reference count out of bounds");
        assert!(core::error::Error::source(&oob).is_none());

        let ca = TryArcError::CloneAlloc(TryCloneError::Other("boom"));
        let mut s = String::new();
        write!(&mut s, "{ca}").unwrap();
        assert_eq!(s, "allocator clone failed: clone failed: boom");
        assert!(core::error::Error::source(&ca).is_some());

        // `From<AllocError>` routes into the `CloneAlloc` arm.
        let from_alloc: TryArcError = AllocError.into();
        assert!(matches!(
            from_alloc,
            TryArcError::CloneAlloc(TryCloneError::Alloc(_))
        ));
    }
}
