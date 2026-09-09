//! Trait implementations for [`Arc`](super::Arc) and [`Weak`](super::Weak).
//!
//! Covers `Deref`, `TryClone`, `TryDefault`, `Default`, `Debug`, `Display`,
//! `AsRef`, `Borrow`, and `Pointer`.

use super::pointers;
use super::{Arc, Weak};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::ops::Deref;
use olive_core::alloc::Allocator;
use olive_core::alloc::AllocatorTryClone;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

// ---------------------------------------------------------------------------
// Deref (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Deref for Arc<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `ptr` is always a valid, aligned, non-null pointer to an
        // initialized `T`.
        unsafe { &*pointers::ptr_get_data(self.ptr.as_ptr()) }
    }
}

// ---------------------------------------------------------------------------
// Clone / TryClone (?Sized)
// ---------------------------------------------------------------------------

// The `TryClone` impl requires `A: AllocatorTryClone` (not merely
// `Allocator + Clone`) so that the cloned allocator handle is guaranteed to be
// equivalent to the original — a prerequisite for the refcount-bump clone to
// remain sound. Both the allocator clone and the counter bump are fallible;
// either failure surfaces as [`TryCloneError`] here.
impl<T: ?Sized, A: AllocatorTryClone> TryClone for Arc<T, A> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let alloc = A::try_clone(&self.alloc)?;
        let inner = self.inner();
        // Passing an existing reference between threads already requires synchronization.
        //
        // See [boost documentation][1] for details.
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        inner
            .strong
            .fetch_update(
                core::sync::atomic::Ordering::Relaxed,
                core::sync::atomic::Ordering::Relaxed,
                |n| n.checked_add(1),
            )
            .map_err(|_| TryCloneError::Other("strong count out of bounds"))?;
        Ok(Arc {
            ptr: self.ptr,
            alloc,
            _marker: core::marker::PhantomData,
        })
    }
}

impl<T: ?Sized, A: AllocatorTryClone> TryClone for Weak<T, A> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let alloc = A::try_clone(&self.alloc)?;
        // A dangling weak (from `Weak::new`) never referred to an allocation,
        // so there is no counter to bump — just duplicate the sentinel.
        if pointers::is_dangling_weak(self.ptr.as_ptr()) {
            return Ok(Weak {
                ptr: self.ptr,
                alloc,
                _marker: core::marker::PhantomData,
            });
        }
        let inner = match self.inner() {
            Some(i) => i,
            None => unreachable!("non-dangling weak must have a valid inner allocation"),
        };
        inner
            .weak
            .fetch_update(
                core::sync::atomic::Ordering::Relaxed,
                core::sync::atomic::Ordering::Relaxed,
                |n| n.checked_add(1),
            )
            .map_err(|_| TryCloneError::Other("weak count out of bounds"))?;
        Ok(Weak {
            ptr: self.ptr,
            alloc,
            _marker: core::marker::PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Pointer / AsRef / Borrow (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> fmt::Pointer for Arc<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&Self::as_ptr(self), f)
    }
}

impl<T: ?Sized, A: Allocator> AsRef<T> for Arc<T, A> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized, A: Allocator> Borrow<T> for Arc<T, A> {
    #[inline]
    fn borrow(&self) -> &T {
        self
    }
}

// ---------------------------------------------------------------------------
// Formatting (?Sized)
// ---------------------------------------------------------------------------

// Mirrors std: printing a `Weak` would require upgrading it, which needs a
// fallible allocator clone (`A: AllocatorTryClone`). Print only the marker
// instead.
impl<T: ?Sized, A: Allocator> Debug for Weak<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("(Weak)")
    }
}

// ---------------------------------------------------------------------------
// Default construction (sized)
// ---------------------------------------------------------------------------

impl<T: TryDefault, A: Allocator + TryDefault> TryDefault for Arc<T, A> {
    fn try_default() -> Result<Self, TryDefaultError> {
        let alloc = A::try_default()?;
        let uninit = Self::try_new_uninit_in(alloc).map_err(TryDefaultError::Alloc)?;
        let value = T::try_default()?;
        // SAFETY: we just initialized the Arc with strong == 1.
        Ok(unsafe { uninit.write(value) })
    }
}

// A dangling `Weak` is infallible by construction (no allocation involved), so
// the conventional `Default` succeeds and yields a weak that points at nothing.
impl<T: ?Sized, A: Allocator + Default> Default for Weak<T, A> {
    #[inline]
    fn default() -> Self {
        Self::new_in(A::default())
    }
}

impl<T: ?Sized, A: Allocator + TryDefault> TryDefault for Weak<T, A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new_in(A::try_default()?))
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
    use crate::test_helpers::allocators::FailDefaultAlloc;
    use crate::test_helpers::{CloneBudget, FailAlloc, FlakyCloneAlloc};
    use std::rc::Rc;
    use std::string::String;

    // --- Deref ---------------------------------------------------------------

    #[test]
    fn deref_yields_payload() {
        let arc = Arc::try_new(String::from("hello")).unwrap();
        assert_eq!(&*arc, "hello");
    }

    #[test]
    fn deref_works_for_unsized_slice() {
        let data = [1u8, 2, 3];
        let arc: Arc<[u8], Global> = Arc::try_from_slice(&data).unwrap();
        assert_eq!(&*arc, &[1u8, 2, 3]);
    }

    // --- TryClone ------------------------------------------------------------

    #[test]
    fn try_clone_shares_allocation() {
        let arc = Arc::try_new(42u32).unwrap();
        let arc2 = arc.try_clone().unwrap();
        assert_eq!(Arc::strong_count(&arc), 2);
        assert!(Arc::ptr_eq(&arc, &arc2));
        assert_eq!(*arc2, 42);
    }

    #[test]
    fn try_clone_alloc_failure_leaves_counts_untouched() {
        let alloc = FlakyCloneAlloc::new(Rc::new(CloneBudget::new(0)));
        let arc = Arc::try_new_in(5i32, alloc).unwrap();
        let res = arc.try_clone();
        assert!(res.is_err());
        // No strong reference was created, so the count is unchanged.
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    #[test]
    fn weak_try_clone_dangling_succeeds_without_counter() {
        let w: Weak<i32, Global> = Weak::new();
        let w2 = w.try_clone().unwrap();
        // Both are dangling sentinels.
        assert!(pointers::is_dangling_weak(w.ptr.as_ptr()));
        assert!(pointers::is_dangling_weak(w2.ptr.as_ptr()));
    }

    #[test]
    fn weak_try_clone_alloc_failure_leaves_counts_untouched() {
        // Budget of 1: the initial downgrade consumes it, so the subsequent
        // weak try_clone's allocator clone fails before bumping the weak count.
        let alloc = FlakyCloneAlloc::new(Rc::new(CloneBudget::new(1)));
        let arc = Arc::try_new_in(5i32, alloc).unwrap();
        let w = Arc::try_downgrade(&arc).unwrap(); // consumes the last budget unit
        let res = w.try_clone();
        assert!(res.is_err());
        // No additional weak reference was created.
        assert_eq!(Arc::weak_count(&arc), 1);
    }

    #[test]
    fn weak_try_clone_live_bumps_weak_count() {
        let arc = Arc::try_new(7i32).unwrap();
        let w = Arc::try_downgrade(&arc).unwrap();
        let w2 = w.try_clone().unwrap();
        assert_eq!(Arc::weak_count(&arc), 2);
        drop(w);
        assert_eq!(Arc::weak_count(&arc), 1);
        drop(w2);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    // --- Pointer / AsRef / Borrow ---------------------------------------------

    #[test]
    fn arc_pointer_formats_as_raw_ptr() {
        let arc = Arc::try_new(42u32).unwrap();
        let expected = std::format!("{:p}", Arc::as_ptr(&arc));
        let actual = std::format!("{:p}", arc);
        assert_eq!(expected, actual);
    }

    #[test]
    fn arc_as_ref_yields_payload() {
        let arc = Arc::try_new(String::from("hello")).unwrap();
        let s: &String = arc.as_ref();
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn arc_borrow_yields_payload() {
        let arc = Arc::try_new(String::from("world")).unwrap();
        let s: &String = Borrow::borrow(&arc);
        assert_eq!(s.as_str(), "world");
    }

    // --- Debug ----------------------------------------------------------------

    #[test]
    fn weak_debug_prints_marker() {
        let w: Weak<i32, Global> = Weak::new();
        assert_eq!(std::format!("{:?}", w), "(Weak)");
    }

    #[test]
    fn weak_debug_of_live_weak_also_prints_marker() {
        let arc = Arc::try_new(1i32).unwrap();
        let w = Arc::try_downgrade(&arc).unwrap();
        assert_eq!(std::format!("{:?}", w), "(Weak)");
    }

    // --- Default / TryDefault ------------------------------------------------

    #[test]
    fn weak_default_is_dangling() {
        let w: Weak<i32, Global> = Default::default();
        assert!(pointers::is_dangling_weak(w.ptr.as_ptr()));
        assert!(w.inner().is_none());
    }

    #[test]
    fn weak_try_default_global() {
        let w: Weak<u64, Global> = TryDefault::try_default().unwrap();
        assert!(pointers::is_dangling_weak(w.ptr.as_ptr()));
        assert!(w.inner().is_none());
    }

    #[test]
    fn arc_try_default_with_i32() {
        let arc: Arc<i32, Global> = TryDefault::try_default().unwrap();
        assert_eq!(*arc, 0);
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    #[test]
    fn arc_try_default_with_option() {
        let arc: Arc<Option<i32>, Global> = TryDefault::try_default().unwrap();
        assert_eq!(*arc, None);
    }

    // --- TryDefault failure modes --------------------------------------------

    #[test]
    fn arc_try_default_alloc_default_failure() {
        // The allocator's own `try_default` fails before any allocation is made.
        let res: Result<Arc<i32, FailDefaultAlloc>, _> = TryDefault::try_default();
        assert!(res.is_err());
    }

    #[test]
    fn arc_try_default_allocation_failure() {
        // The allocator default succeeds but the heap allocation fails.
        let res: Result<Arc<i32, FailAlloc>, _> = TryDefault::try_default();
        assert!(matches!(res, Err(TryDefaultError::Alloc(_))));
    }

    #[test]
    fn weak_try_default_alloc_default_failure() {
        // The allocator's own `try_default` fails; no allocation is attempted.
        let res: Result<Weak<i32, FailDefaultAlloc>, _> = TryDefault::try_default();
        assert!(res.is_err());
    }
}
