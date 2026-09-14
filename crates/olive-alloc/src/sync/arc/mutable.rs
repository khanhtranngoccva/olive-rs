//! Mutable-access methods for [`Arc`](super::Arc): `get_mut`,
//! `get_mut_unchecked`, and `try_make_mut`.
//!
//! These mirror std's `Arc::get_mut` / `Arc::make_mut` pair, adapted to
//! Olive's fully-fallible allocation model. The fallible clone-on-write path
//! is named `try_make_mut` (matching the `Rc::try_make_mut` precedent) and
//! returns `Result<&mut T, TryCloneError>`.

use core::mem::size_of_val;
use core::ptr;
use core::sync::atomic::Ordering;

use crate::alloc::{Allocator, AllocatorTryClone};
use olive_core::try_traits::try_clone::{TryCloneError, TryCloneToUninit};

use super::dst::UniqueArcUninit;
use super::{Arc, ArcInner, Weak};

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Gets a mutable reference to the contained value if this `Arc` is the
    /// sole strong reference **and** no [`Weak`] pointers to the same
    /// allocation exist. Returns `None` otherwise.
    ///
    /// This mirrors std's `Arc::get_mut`, which requires both that there are
    /// no other strong references *and* no weak references, because a live
    /// `Weak` could upgrade back into a second strong reference at any moment.
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        // SAFETY: ensured by the uniqueness check below.
        if Self::is_unique(this) {
            Some(unsafe { Self::get_mut_unchecked(this) })
        } else {
            None
        }
    }

    /// Gets a mutable reference to the contained value **without** checking
    /// the strong count.
    ///
    /// # Safety
    ///
    /// If any other `Arc` or `Weak` pointers to the same allocation exist,
    /// then they must not be dereferenced or have active borrows for the
    /// duration of the returned borrow, and their inner type must be exactly
    /// the same as the inner type of this `Arc` (including lifetimes).
    ///
    /// This is trivially the case if no such pointers exist, for example
    /// immediately after `Arc::new`.
    #[inline]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        // We are careful to *not* create a reference covering the "count" fields, as
        // this would alias with concurrent access to the reference counts (e.g. by `Weak`).
        unsafe { &mut (*this.ptr.as_ptr()).value }
    }

    /// Makes a mutable reference into the given `Arc`, disassociating other
    /// references by moving or cloning as needed.
    ///
    /// Three cases, mirroring std's `Arc::make_mut`:
    ///
    /// * **Shared** — more than one strong reference exists: the inner value is
    ///   cloned into a fresh allocation via [`TryCloneToUninit`], and this [`Arc`]
    ///   is replaced in place to point at the clone. The old shared allocation keeps
    ///   serving the remaining owners.
    /// * **Only weak refs remain** — exactly one strong reference (this one) but
    ///   some [`Weak`] pointers: the value is moved out of the old block into a
    ///   freshly allocated one, and the old block's counts are decremented, so its
    ///   payload is logically gone. The surviving `Weak`s no longer point to a live
    ///   value (their `upgrade` will now fail). No clone is performed.
    /// * **Unique** — no other [`Arc`] or [`Weak`] pointers exist: the payload is
    ///   accessed in place with no allocation.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if cloning the allocator handle, allocating the
    /// replacement block, or the clone itself fails.
    #[inline]
    pub fn try_make_mut(this: &mut Self) -> Result<&mut T, TryCloneError>
    where
        T: TryCloneToUninit,
        A: AllocatorTryClone,
    {
        let size_of_val = size_of_val::<T>(&**this);

        // Note that we hold both a strong reference and a weak reference.
        // Thus, releasing our strong reference only will not, by itself, cause
        // the memory to be deallocated.
        //
        // Use Acquire to ensure that we see any writes to `weak` that happen
        // before release writes (i.e., decrements) to `strong`. Since we hold a
        // weak count, there's no chance the ArcInner itself could be
        // deallocated.
        if this
            .inner()
            .strong
            .compare_exchange(1, 0, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Another strong pointer exists, so we must clone.
            let alloc = A::try_clone(&this.alloc)?;
            let new_arc = Arc::try_clone_from_ref_in(&**this, alloc)?;
            *this = new_arc;
        } else if this.inner().weak.load(Ordering::Relaxed) != 1 {
            // Relaxed suffices in the above because this is fundamentally an
            // optimization: we are always racing with weak pointers being
            // dropped. Worst case, we end up allocated a new Arc unnecessarily.

            // We removed the last strong ref, but there are additional weak
            // refs remaining. We'll move the contents to a new Arc, and
            // invalidate the other weak refs.

            // Note that it is not possible for the read of `weak` to yield
            // usize::MAX (i.e., locked), since the weak count can only be
            // locked by a thread with a strong reference.

            // Guard against errors while using the allocator.
            // If we fail before the Arc is overwritten, we expose a strong
            // count of 0, resulting in a UAF. Until the new Arc is written,
            // the old Arc must remain valid.
            struct Guard<'a, T: ?Sized> {
                inner: &'a ArcInner<T>,
            }
            impl<T: ?Sized> Drop for Guard<'_, T> {
                fn drop(&mut self) {
                    self.inner.strong.store(1, Ordering::Release);
                }
            }
            let guard = Guard {
                inner: this.inner(),
            };

            // Can just steal the data, all that's left is Weaks.
            // Note that this can fail (or panic) in two ways:
            // - The allocation can fail
            // - The allocator clone can fail
            let alloc = A::try_clone(&this.alloc)?;
            let in_progress: UniqueArcUninit<T, A> =
                UniqueArcUninit::try_new_for_value(&**this, alloc)?;

            // SAFETY: we have exclusive access to the payload (strong was
            // atomically set to 0 via the CAS above, and no other thread can
            // observe it until we restore or transfer ownership). Expressing
            // the move in terms of bytes handles `T: ?Sized` uniformly.
            unsafe {
                // Initialize `in_progress` with move of **this.
                // We have to express this in terms of bytes because `T: ?Sized`; there is no
                // operation that just copies (more precisely, moves) a value based on its `size_of_val()`.
                ptr::copy_nonoverlapping(
                    ptr::from_ref(&**this).cast::<u8>(),
                    in_progress.data_ptr().cast::<u8>(),
                    size_of_val,
                );

                // We are now safe from failures.
                core::mem::forget(guard);

                // Materialize our own implicit weak pointer, so that it can clean
                // up the ArcInner as needed.
                // Make sure the allocator is not leaked when the Arc is overwritten.
                // Only drop at the end of the scope to avoid panics.
                let _weak = Weak {
                    ptr: this.ptr,
                    alloc: ptr::read(&this.alloc),
                    _marker: core::marker::PhantomData,
                };

                ptr::write(this, in_progress.into_arc());
            }
        } else {
            // We were the sole reference of either kind; bump back up the
            // strong ref count.
            this.inner().strong.store(1, Ordering::Release);
        }

        // SAFETY: As with `get_mut()`, our reference was
        // either unique to begin with, or became one upon cloning the contents.
        Ok(unsafe { Self::get_mut_unchecked(this) })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::string::String;
    use crate::test_helpers::FlakyCloneAlloc;
    use crate::vec::Vec;
    use olive_core::try_traits::TryClone;
    use std::sync::Arc as StdArc;

    // --- get_mut -------------------------------------------------------------

    #[test]
    fn get_mut_returns_some_when_unique() {
        let mut arc = Arc::try_new(42i32).unwrap();
        let m: Option<&mut i32> = Arc::get_mut(&mut arc);
        assert!(m.is_some());
        *m.unwrap() = 99;
        assert_eq!(*arc, 99);
    }

    #[test]
    fn get_mut_returns_none_when_shared() {
        let arc = Arc::try_new(42i32).unwrap();
        let arc2 = arc.try_clone().unwrap();
        let mut arc = arc;
        assert!(Arc::get_mut(&mut arc).is_none());
        drop(arc2);
        // Now unique again.
        assert!(Arc::get_mut(&mut arc).is_some());
    }

    #[test]
    fn get_mut_returns_none_when_weak_exists() {
        let arc = Arc::try_new(42i32).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        let mut arc = arc;
        assert!(Arc::get_mut(&mut arc).is_none());
        drop(weak);
        assert!(Arc::get_mut(&mut arc).is_some());
    }

    // --- get_mut_unchecked ---------------------------------------------------

    #[test]
    fn get_mut_unchecked_mutates_payload() {
        let mut arc = Arc::try_new(String::try_from_str("hello").unwrap()).unwrap();
        // SAFETY: sole owner, no other refs.
        let s: &mut String = unsafe { Arc::get_mut_unchecked(&mut arc) };
        s.try_push_str(" world").unwrap();
        assert_eq!(&*arc, "hello world");
    }

    // --- try_make_mut: unique case -------------------------------------------

    #[test]
    fn try_make_mut_unique_no_allocation() {
        let mut arc = Arc::try_new(7i32).unwrap();
        let m: &mut i32 = Arc::try_make_mut(&mut arc).unwrap();
        *m = 42;
        assert_eq!(*arc, 42);
        // Still the same allocation (no clone occurred).
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    // --- try_make_mut: shared case (clone) ------------------------------------

    #[test]
    fn try_make_mut_shared_clones_into_fresh_block() {
        let v = Vec::try_from(&[1i32, 2, 3][..]).unwrap();
        let arc = Arc::try_new(v).unwrap();
        let arc2 = arc.try_clone().unwrap();
        let mut arc = arc;
        let m: &mut Vec<i32> = Arc::try_make_mut(&mut arc).unwrap();
        m.try_push(4).unwrap();
        // The original still sees the pre-mutation value.
        assert_eq!(arc2.len(), 3);
        assert_eq!(arc2[0], 1);
        assert_eq!(arc2[1], 2);
        assert_eq!(arc2[2], 3);
        // The mutated arc has the new value.
        assert_eq!(arc.len(), 4);
        assert_eq!(arc[3], 4);
        // They now point to different allocations.
        assert!(!Arc::ptr_eq(&arc, &arc2));
    }

    // --- try_make_mut: weak-only case (steal) --------------------------------

    #[test]
    fn try_make_mut_steals_when_only_weaks_remain() {
        let arc = Arc::try_new(String::try_from_str("original").unwrap()).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        let mut arc = arc;
        let m: &mut String = Arc::try_make_mut(&mut arc).unwrap();
        m.clear();
        m.try_push_str("moved").unwrap();
        assert_eq!(&*arc, "moved");
        // The weak can no longer upgrade (strong count on the old block is 0).
        assert!(weak.try_upgrade().unwrap().is_none());
        // The new arc is uniquely owned.
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn try_make_mut_steal_invalidates_multiple_weaks() {
        let arc = Arc::try_new(100u64).unwrap();
        let w1 = Arc::try_downgrade(&arc).unwrap();
        let w2 = Arc::try_downgrade(&arc).unwrap();
        let mut arc = arc;
        let m: &mut u64 = Arc::try_make_mut(&mut arc).unwrap();
        *m = 200;
        assert_eq!(*arc, 200);
        // Neither weak can upgrade anymore.
        assert!(w1.try_upgrade().unwrap().is_none());
        assert!(w2.try_upgrade().unwrap().is_none());
    }

    // --- try_make_mut: OOM paths ----------------------------------------------

    #[test]
    fn try_make_mut_shared_oom_on_alloc_clone() {
        // Budget of 1: try_clone consumes it, leaving 0 for try_make_mut's
        // internal allocator clone.
        let alloc = FlakyCloneAlloc::new(StdArc::new(crate::test_helpers::CloneBudget::new(1)));
        let arc = Arc::try_new_in(5i32, alloc).unwrap();
        let arc2 = arc.try_clone().unwrap(); // consumes the last budget unit
        let mut arc = arc;
        // The allocator clone inside try_make_mut will fail (budget exhausted).
        let res: Result<&mut i32, TryCloneError> = Arc::try_make_mut(&mut arc);
        assert!(res.is_err());
        // The arc is unchanged (still shared, still pointing at the original).
        assert_eq!(Arc::strong_count(&arc), 2);
        assert_eq!(*arc, 5);
        drop(arc2);
    }

    #[test]
    fn try_make_mut_shared_oom_on_arc_clone() {
        // Budget of 2: arc.try_clone() consumes 1, try_make_mut's internal
        // allocator clone consumes the last one, leaving 0 for the element
        // clones inside try_clone_from_ref_in.
        let ledger = StdArc::new(crate::test_helpers::Ledger::new());
        let budget = StdArc::new(crate::test_helpers::CloneBudget::new(2));
        let alloc = FlakyCloneAlloc::new(budget.clone());

        // Build a 2-element slice payload; each item shares the same budget.
        let mut src: Vec<crate::test_helpers::FlakyTrackedItem> = Vec::new();
        for _ in 0..2 {
            let id = ledger.allocate();
            ledger.register(id);
            src.try_push(crate::test_helpers::FlakyTrackedItem {
                id,
                ledger: ledger.clone(),
                inner: (*budget).share(),
            })
            .unwrap();
        }
        let arc = Arc::try_new_in(src, alloc).unwrap();
        let arc2 = arc.try_clone().unwrap(); // consumes 1 (budget: 2 -> 1)
        let mut arc = arc;

        // Allocator clone inside try_make_mut succeeds (budget: 1 -> 0),
        // but the first element clone in try_clone_to_uninit fails.
        let res: Result<&mut Vec<crate::test_helpers::FlakyTrackedItem>, TryCloneError> =
            Arc::try_make_mut(&mut arc);
        assert!(res.is_err());
        // The arc is unchanged (still shared, still pointing at the original).
        assert_eq!(Arc::strong_count(&arc), 2);
        assert_eq!(arc.len(), 2);
        drop(arc2);
    }

    #[test]
    fn try_make_mut_steal_oom_on_allocation() {
        let budget = StdArc::new(crate::test_helpers::CloneBudget::new(1));
        let flaky = FlakyCloneAlloc::new(budget.clone());
        let arc = Arc::try_new_in(99i32, flaky).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        let mut arc = arc;
        // Budget is now exhausted (downgrade consumed the last unit), so the
        // allocator clone inside try_make_mut's steal path will fail.
        let res: Result<&mut i32, TryCloneError> = Arc::try_make_mut(&mut arc);
        assert!(res.is_err());
        // The arc is unchanged.
        assert_eq!(*arc, 99);
        assert_eq!(Arc::strong_count(&arc), 1);
        drop(weak);
    }

    // --- Parity with std -------------------------------------------------------

    #[test]
    fn make_mut_parity_with_std_unique() {
        use std::sync as stock;
        let mut ours = Arc::try_new(10i32).unwrap();
        let mut theirs = stock::Arc::new(10i32);
        *Arc::try_make_mut(&mut ours).unwrap() = 20;
        *stock::Arc::make_mut(&mut theirs) = 20;
        assert_eq!(*ours, *theirs);
    }

    #[test]
    fn make_mut_parity_with_std_shared_tuple() {
        use std::sync as stock;
        // Use a tuple payload (implements both Clone and TryCloneToUninit).
        let ours = Arc::try_new((1i32, 2)).unwrap();
        let theirs = stock::Arc::new((1i32, 2));
        let ours2 = ours.try_clone().unwrap();
        let theirs2 = theirs.clone();
        let mut ours = ours;
        let mut theirs = theirs;
        *Arc::try_make_mut(&mut ours).unwrap() = (1, 99);
        *stock::Arc::make_mut(&mut theirs) = (1, 99);
        assert_eq!(*ours, *theirs);
        assert_eq!(*ours2, (1, 2));
        assert_eq!(*theirs2, (1, 2));
    }

    #[test]
    fn make_mut_parity_with_std_weak_steal() {
        use std::sync as stock;
        let ours = Arc::try_new(String::try_from_str("abc").unwrap()).unwrap();
        let theirs = stock::Arc::new(std::string::String::from("abc"));
        let our_weak = Arc::try_downgrade(&ours).unwrap();
        let their_weak = stock::Arc::downgrade(&theirs);
        let mut ours = ours;
        let mut theirs = theirs;
        Arc::try_make_mut(&mut ours).unwrap().clear();
        stock::Arc::make_mut(&mut theirs).clear();
        assert_eq!(&*ours, "");
        assert_eq!(&*theirs, "");
        assert!(our_weak.try_upgrade().unwrap().is_none());
        assert!(their_weak.upgrade().is_none());
    }

    // --- is_unique integration ------------------------------------------------

    #[test]
    fn is_unique_true_after_all_refs_dropped() {
        let arc = Arc::try_new(1i32).unwrap();
        let a2 = arc.try_clone().unwrap();
        let w = Arc::try_downgrade(&arc).unwrap();
        assert!(!Arc::is_unique(&arc));
        drop(a2);
        assert!(!Arc::is_unique(&arc));
        drop(w);
        assert!(Arc::is_unique(&arc));
    }

    #[test]
    fn is_unique_restores_weak_count_after_check() {
        // Single-threaded smoke test: the CAS lock/unlock cycle leaves the
        // weak count restored to 1 (the implicit weak ref).
        let arc = Arc::try_new(1i32).unwrap();
        assert!(Arc::is_unique(&arc));
        // After the check, the weak count must be back to 1.
        assert_eq!(arc.inner().weak.load(Ordering::Relaxed), 1);
    }
}
