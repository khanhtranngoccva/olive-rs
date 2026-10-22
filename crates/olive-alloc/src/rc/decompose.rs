use super::internals::RcInner;
use super::{Rc, Weak};
use crate::alloc::Allocator;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::Deref;
use olive_core::ptr::{self};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

impl<T, A: Allocator> Rc<T, A> {
    /// Consumes this `Rc`, attempting to take the value out, returning the
    /// owned `T`.
    ///
    /// Like std's `Rc::try_unwrap`, this succeeds whenever this handle is the
    /// last strong reference, even if `Weak` pointers are still alive.
    ///
    /// # Errors
    ///
    /// Returns the input `Rc` if the value could not be unwrapped because
    /// ownership is shared.
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // Move-out is safe whenever we are the unique strong owner: weak
        // references only pin the allocation, never the payload, and
        // `Weak::try_upgrade` refuses to hand out a new strong reference
        // once the strong count has reached zero.
        if Self::strong_count(&this) == 1 {
            // Suppress the handle's implicit drop so our manual teardown below
            // is the single owner of the strong reference.
            let me = ManuallyDrop::new(this);
            // SAFETY: we are the sole strong reference, so the payload is
            // valid and exclusively ours. `ptr::read` moves the `T` out of
            // the allocation; any surviving weaks keep the block itself
            // alive but can no longer access the moved-out payload.
            let val = unsafe { ptr::read(&**me) };
            let alloc = unsafe { ptr::read(&me.alloc) };
            let inner: &RcInner<T> = Rc::inner(me.deref());
            inner.dec_strong().expect("strong count underflow");
            // Drop a throwaway `Weak` standing in for this `Rc`'s implicit
            // weak reference. With other weaks alive this only decrements
            // the count; when none remain it frees the block.
            let _dummy_weak = Weak::<T, A> {
                ptr: me.ptr,
                alloc,
                _marker: PhantomData,
            };
            Ok(val)
        } else {
            Err(this)
        }
    }

    /// Consumes this `Rc`, returning the owned value `T` if `this` is the last
    /// strong reference (live `Weak` pointers do not block this), or clone
    /// the reference otherwise.
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
    /// Returns the input `Rc` paired with [`TryCloneError`] if the value
    /// could not be cloned in the fallback path.
    #[inline]
    pub fn unwrap_or_try_clone_give_back(this: Self) -> Result<T, (Self, TryCloneError)>
    where
        T: TryClone,
    {
        match Self::try_unwrap(this) {
            Ok(s) => Ok(s),
            Err(rc) => (*rc).try_clone().map_err(|e| (rc, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::CountingAllocator;
    use olive_core::try_traits::try_clone::TryClone;
    use std::string::String;

    #[test]
    fn try_unwrap_sole_owner_moves_out() {
        // Sole strong, no weaks: the payload is moved out and the block freed.
        let rc = Rc::try_new(String::from("hello")).unwrap();
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);
        let val = Rc::try_unwrap(rc).unwrap();
        assert_eq!(val, "hello");
    }

    #[test]
    fn try_unwrap_shared_returns_handle_back() {
        // Shared ownership: cannot move out, so the whole input handle is
        // handed back as the Err variant and the allocation stays intact.
        let rc = Rc::try_new(42i32).unwrap();
        let rc2 = rc.try_clone().unwrap();
        let returned = Rc::try_unwrap(rc).unwrap_err();
        // The returned handle still points at the same live allocation.
        assert!(Rc::ptr_eq(&returned, &rc2));
        assert_eq!(*returned, 42);
        assert_eq!(Rc::strong_count(&returned), 2);
        // Dropping both handles frees the block exactly once.
        drop(returned);
        drop(rc2);
    }

    #[test]
    fn try_unwrap_with_live_weak_moves_value_out() {
        // Live weak references do not block a move-out (matching std): the
        // payload goes to the caller, the weak can no longer upgrade, and
        // the block is only freed once the weak itself is gone.
        let alloc = CountingAllocator::new();
        let rc = Rc::try_new_in(7u8, &alloc).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let val = Rc::try_unwrap(rc).unwrap();
        assert_eq!(val, 7);
        assert!(weak.try_upgrade().unwrap().is_none());
        // The weak alone still pins the block; dropping it frees it.
        assert_eq!(alloc.deallocations(), 0);
        drop(weak);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_unwrap_preserves_drop_semantics() {
        use olive_test_helpers::{Ledger, TrackedItem};
        use std::sync::Arc;

        let ledger = Arc::new(Ledger::new());
        let rc = Rc::try_new(TrackedItem::construct(&ledger, ())).unwrap();
        let id = rc.id();
        let tracked = Rc::try_unwrap(rc).unwrap();
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
    fn unwrap_or_try_clone_sole_owner_moves_out() {
        // Sole owner: consumes the handle and moves the payload out directly.
        let rc = Rc::try_new(1234i32).unwrap();
        let val = Rc::unwrap_or_try_clone(rc).unwrap();
        assert_eq!(val, 1234);
    }

    #[test]
    fn unwrap_or_try_clone_with_live_weak_moves_value_out() {
        // A live weak does not force the clone fallback (matching std): the
        // value is moved out, the handle consumed, and the weak then goes
        // dangling because the last strong reference is gone.
        let rc = Rc::try_new(7u8).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let val = Rc::unwrap_or_try_clone(rc).unwrap();
        assert_eq!(val, 7);
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
    }

    #[test]
    fn unwrap_or_try_clone_give_back_sole_owner_moves_out() {
        // Sole owner: consumes the handle and moves the payload out directly.
        let rc = Rc::try_new(321i32).unwrap();
        let val = Rc::unwrap_or_try_clone_give_back(rc).unwrap();
        assert_eq!(val, 321);
    }

    #[test]
    fn unwrap_or_try_clone_shared_clones_value() {
        // Shared ownership: cannot move out, so the value is cloned instead.
        let rc = Rc::try_new(99i32).unwrap();
        let rc2 = rc.try_clone().unwrap();
        let val = Rc::unwrap_or_try_clone(rc).unwrap();
        assert_eq!(val, 99);
        // The original allocation is untouched.
        assert_eq!(*rc2, 99);
        assert_eq!(Rc::strong_count(&rc2), 1);
        drop(rc2);
    }

    #[test]
    fn unwrap_or_try_clone_give_back_shared_clones_value() {
        // Sole ownership: cannot move out, so the value is cloned instead.
        // Both handles remain valid and point at the same allocation.
        let rc = Rc::try_new(88i32).unwrap();
        let rc2 = rc.try_clone().unwrap();
        let val = Rc::unwrap_or_try_clone_give_back(rc).unwrap();
        assert_eq!(val, 88);
        // The original allocation is untouched.
        assert_eq!(*rc2, 88);
        assert_eq!(Rc::strong_count(&rc2), 1);
        drop(rc2);
    }

    #[test]
    fn unwrap_or_try_clone_give_back_with_live_weak_moves_value_out() {
        // Same move-out through the give-back variant: a live weak does not
        // force the clone fallback, the value is moved out, the handle
        // consumed, and the weak goes dangling.
        let rc = Rc::try_new(7u8).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let val = Rc::unwrap_or_try_clone_give_back(rc).unwrap();
        assert_eq!(val, 7);
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
    }
}
