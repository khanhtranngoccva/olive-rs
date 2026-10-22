use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ptr;
use core::sync::atomic::{self, Ordering};

use crate::alloc::Allocator;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

use super::{Arc, Weak};

impl<T, A: Allocator> Arc<T, A> {
    /// Consumes this `Arc`, attempting to take the value out, returning the
    /// owned `T`.
    ///
    /// Like std's `Arc::try_unwrap`, this succeeds whenever this handle is the
    /// last strong reference, even if `Weak` pointers are still alive.
    ///
    /// Unlike its [`Rc`](crate::rc::Rc) counterpart, the ownership check cannot
    /// be a plain counter load followed by a decrement: another thread could
    /// clone or drop a handle between the two steps. Instead, the check is a
    /// single atomic compare-and-exchange that transitions the strong count
    /// from 1 to 0, claiming sole ownership or failing outright. If the CAS
    /// fails, ownership is shared (or was just claimed elsewhere) and `this`
    /// is handed back untouched, with its count exactly as it was.
    ///
    /// # Errors
    ///
    /// Returns the input `Arc` if the value could not be unwrapped because
    /// ownership is shared.
    ///
    /// # Warning
    ///
    /// Like std's `Arc::try_unwrap`, discarding the `Err` handle immediately
    /// (as `Arc::try_unwrap(x).ok()` does) can drop the value even when a
    /// concurrent caller also observes `Err`, destroying the payload without
    /// either caller ever seeing it.
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // A single atomic transition from 1 to 0 is the whole ownership
        // check-and-claim. If the claiming succeeds, the payload is taken
        // by the caller's thread, so it does not have to broadcast its writes
        // via Release.
        if this
            .inner()
            .strong
            .compare_exchange(1, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            // Ownership is shared: hand the handle back with its counters
            // untouched (strong ≥ 2 or, transiently, mid-mutation elsewhere).
            return Err(this);
        }

        // This fence is needed ensure all writes from other threads before refcount
        // decrementing are received.
        atomic::fence(Ordering::Acquire);

        // Decompose the Arc into its members and prevent automatic dropping of
        // the payload.
        let me = ManuallyDrop::new(this);
        // SAFETY: we are the sole strong reference (the CAS reduced the count
        // to zero), so the payload is valid and exclusively ours. `ptr::read`
        // moves the `T` out of the allocation; any surviving weaks keep the
        // block itself alive but can no longer access the moved-out payload.
        let val = unsafe { ptr::read(&**me) };
        // SAFETY: moves the allocator handle out of the suppressed wrapper so
        // the teardown below can deallocate through it.
        let alloc = unsafe { ptr::read(&me.alloc) };

        // Drop a throwaway `Weak` standing in for this `Arc`'s implicit
        // strong-weak reference. With other weaks alive this only decrements
        // the count; when none remain it frees the block. The zeroed strong
        // count left by the CAS is exactly the state `Weak::drop`'s teardown
        // invariant expects.
        let _dummy_weak = Weak::<T, A> {
            ptr: me.ptr,
            alloc,
            _marker: PhantomData,
        };
        Ok(val)
    }

    /// Consumes this `Arc`, returning the owned value `T` if `this` is the
    /// last strong reference (live `Weak` pointers do not block this), or
    /// clone the reference otherwise.
    ///
    /// # Errors
    ///
    /// Returns the [`TryCloneError`] if the value could not be cloned in the
    /// fallback path.
    #[inline]
    pub fn unwrap_or_try_clone(this: Self) -> Result<T, TryCloneError>
    where
        T: TryClone,
    {
        Self::unwrap_or_try_clone_give_back(this).map_err(|(_self, e)| e)
    }

    /// Like [`Self::unwrap_or_try_clone`] but it also returns the original `this`
    /// on error.
    ///
    /// # Errors
    ///
    /// Returns the input `Arc` paired with [`TryCloneError`] if the value
    /// could not be cloned in the fallback path.
    #[inline]
    pub fn unwrap_or_try_clone_give_back(this: Self) -> Result<T, (Self, TryCloneError)>
    where
        T: TryClone,
    {
        match Self::try_unwrap(this) {
            Ok(s) => Ok(s),
            Err(arc) => (*arc).try_clone().map_err(|e| (arc, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::CountingAllocator;
    use olive_test_helpers::{Ledger, MaybeUnclonable, TrackedItem};
    use std::string::String;
    use std::sync as stock;

    // --- try_unwrap ---------------------------------------------------------

    #[test]
    fn try_unwrap_sole_owner_moves_out() {
        // Sole strong, no weaks: the payload is moved out and the block freed.
        let arc = Arc::try_new(String::from("hello")).unwrap();
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
        let val = Arc::try_unwrap(arc).unwrap();
        assert_eq!(val, "hello");
    }

    #[test]
    fn try_unwrap_shared_returns_handle_back() {
        // Shared ownership: cannot move out, so the whole input handle is
        // handed back as the Err variant and the allocation stays intact.
        let arc = Arc::try_new(42i32).unwrap();
        let arc2 = arc.try_clone().unwrap();
        let returned = Arc::try_unwrap(arc).unwrap_err();
        // The returned handle still points at the same live allocation.
        assert!(Arc::ptr_eq(&returned, &arc2));
        assert_eq!(*returned, 42);
        assert_eq!(Arc::strong_count(&returned), 2);
        // Dropping both handles frees the block exactly once.
        drop(returned);
        drop(arc2);
    }

    #[test]
    fn try_unwrap_with_live_weak_moves_value_out() {
        // Live weak references do not block a move-out (matching std): the
        // payload goes to the caller, the weak can no longer upgrade, and
        // the block is only freed once the weak itself is gone.
        let alloc = CountingAllocator::new();
        let arc = Arc::try_new_in(7u8, &alloc).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        let val = Arc::try_unwrap(arc).unwrap();
        assert_eq!(val, 7);
        assert!(weak.try_upgrade().unwrap().is_none());
        // The weak alone still pins the block; dropping it frees it.
        assert_eq!(alloc.deallocations(), 0);
        drop(weak);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_unwrap_preserves_drop_semantics() {
        let ledger = stock::Arc::new(Ledger::new());
        let arc = Arc::try_new(TrackedItem::construct(&ledger, ())).unwrap();
        let id = arc.id();
        let tracked = Arc::try_unwrap(arc).unwrap();
        // Still owned by the caller; nothing has been dropped yet.
        assert_eq!(ledger.live_ids(), [id]);
        drop(tracked);
        // Exactly one drop, on the moved-out value; nothing dropped twice and
        // nothing leaked inside the freed block.
        assert!(ledger.all_dropped_once([id]));
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn try_unwrap_leaves_no_strong_ref_behind_when_weak_survives() {
        // The CAS zeroes the strong count and the dummy weak tears the block
        // down, so no phantom strong owner may linger behind the moved-out
        // value: the surviving weak must see zero strongs.
        let arc = Arc::try_new(5i32).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        let val = Arc::try_unwrap(arc).unwrap();
        assert_eq!(val, 5);
        assert_eq!(weak.strong_count(), 0);
        drop(weak);
    }

    // --- Concurrency ---------------------------------------------------------

    #[test]
    fn concurrent_try_unwrap_shared_fails_atomically_for_both() {
        // Two threads race `try_unwrap` on an `Arc` with strong == 2. The
        // CAS admits no winner: each 1→0 transition fails while the other
        // thread still holds its handle, so both get `Err` back atomically
        // and both handles still see the payload. A load-then-decrement
        // check could instead let one thread observe a count of 1 and steal
        // the value out from under the other's live handle.
        let arc = Arc::try_new(41i32).unwrap();
        let arc2 = arc.try_clone().unwrap();
        // Hold both threads at a barrier so the race window is as tight as
        // the scheduler allows.
        let barrier = stock::Arc::new(stock::Barrier::new(2));
        let h1 = {
            let barrier = stock::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                Arc::try_unwrap(arc)
            })
        };
        let h2 = std::thread::spawn(move || {
            barrier.wait();
            Arc::try_unwrap(arc2)
        });
        let (r1, r2) = (h1.join().unwrap(), h2.join().unwrap());
        assert!(r1.is_err());
        assert!(r2.is_err());
        // Both handles are returned intact, still share the allocation and
        // still see the payload; dropping both tears down the block exactly
        // once (verified implicitly: no leak and no double-free abort).
        let l1 = r1.unwrap_err();
        let l2 = r2.unwrap_err();
        assert!(Arc::ptr_eq(&l1, &l2));
        assert_eq!(*l1, 41);
        assert_eq!(*l2, 41);
        drop(l1);
        drop(l2);
    }

    #[test]
    fn try_unwrap_discarded_err_handle_destroys_value_once() {
        let ledger = stock::Arc::new(Ledger::new());
        let arc = Arc::try_new(TrackedItem::construct(&ledger, ())).unwrap();
        let arc2 = arc.try_clone().unwrap();
        let id = arc.id();
        // The move-out fails (strong == 2) and the handle is discarded.
        assert!(Arc::try_unwrap(arc).ok().is_none());
        drop(arc2);
        assert!(ledger.all_dropped_once([id]));
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
    }

    // --- unwrap_or_try_clone ---------------------------------------------------

    #[test]
    fn unwrap_or_try_clone_sole_owner_moves_out() {
        // Sole owner: consumes the handle and moves the payload out directly.
        let arc = Arc::try_new(1234i32).unwrap();
        let val = Arc::unwrap_or_try_clone(arc).unwrap();
        assert_eq!(val, 1234);
    }

    #[test]
    fn unwrap_or_try_clone_with_live_weak_moves_value_out() {
        // A live weak does not force the clone fallback (matching std): the
        // value is moved out, the handle consumed, and the weak then goes
        // dangling because the last strong reference is gone.
        let arc = Arc::try_new(7u8).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        let val = Arc::unwrap_or_try_clone(arc).unwrap();
        assert_eq!(val, 7);
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
    }

    #[test]
    fn unwrap_or_try_clone_shared_clones_value() {
        // Shared ownership: cannot move out, so the value is cloned instead.
        let arc = Arc::try_new(99i32).unwrap();
        let arc2 = arc.try_clone().unwrap();
        let val = Arc::unwrap_or_try_clone(arc).unwrap();
        assert_eq!(val, 99);
        // The original allocation is untouched.
        assert_eq!(*arc2, 99);
        assert_eq!(Arc::strong_count(&arc2), 1);
        drop(arc2);
    }

    #[test]
    fn unwrap_or_try_clone_clone_failure() {
        // Shared ownership with an uncloneable payload: this variant must drop the handle.
        let arc = Arc::try_new(MaybeUnclonable::unclonable(13)).unwrap();
        let arc2 = arc.try_clone().unwrap();
        let _err = Arc::unwrap_or_try_clone(arc).unwrap_err();
        assert_eq!(arc2.inner, 13);
        assert_eq!(Arc::strong_count(&arc2), 1);
        drop(arc2);
    }

    #[test]
    fn unwrap_or_try_clone_give_back_sole_owner_moves_out() {
        // Sole owner: consumes the handle and moves the payload out directly.
        let arc = Arc::try_new(321i32).unwrap();
        let val = Arc::unwrap_or_try_clone_give_back(arc).unwrap();
        assert_eq!(val, 321);
    }

    #[test]
    fn unwrap_or_try_clone_give_back_shared_clones_value() {
        // Shared ownership: cannot move out, so the value is cloned instead.
        // The original allocation stays valid.
        let arc = Arc::try_new(88i32).unwrap();
        let arc2 = arc.try_clone().unwrap();
        let val = Arc::unwrap_or_try_clone_give_back(arc).unwrap();
        assert_eq!(val, 88);
        // The original allocation is untouched.
        assert_eq!(*arc2, 88);
        assert_eq!(Arc::strong_count(&arc2), 1);
        drop(arc2);
    }

    #[test]
    fn unwrap_or_try_clone_give_back_with_live_weak_moves_value_out() {
        // Same move-out through the give-back variant: a live weak does not
        // force the clone fallback, the value is moved out, the handle
        // consumed, and the weak goes dangling.
        let arc = Arc::try_new(7u8).unwrap();
        let weak = Arc::try_downgrade(&arc).unwrap();
        let val = Arc::unwrap_or_try_clone_give_back(arc).unwrap();
        assert_eq!(val, 7);
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
    }

    #[test]
    fn unwrap_or_try_clone_give_back_clone_failure_returns_handle() {
        // Shared ownership with an uncloneable payload: the give-back variant
        // must return the original handle paired with the error, with the
        // allocation intact and the strong count unchanged.
        let arc = Arc::try_new(MaybeUnclonable::unclonable(13)).unwrap();
        let arc2 = arc.try_clone().unwrap();
        let (returned, _err) = Arc::unwrap_or_try_clone_give_back(arc).unwrap_err();
        assert!(Arc::ptr_eq(&returned, &arc2));
        assert_eq!(returned.inner, 13);
        assert_eq!(Arc::strong_count(&returned), 2);
        drop(returned);
        drop(arc2);
    }
}
