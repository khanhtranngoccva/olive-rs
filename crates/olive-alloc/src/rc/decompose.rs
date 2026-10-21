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
    /// # Errors
    ///
    /// Returns the input `Rc` if the value could not be unwrapped because
    /// ownership is shared or weak references are still alive.
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // Move-out is safe only when we are the unique owner of the payload:
        // exactly one strong reference and no outstanding weak references.
        if Self::strong_count(&this) == 1 && Self::weak_count(&this) == 0 {
            // Suppress the handle's implicit drop so our manual teardown below
            // is the single owner of the strong reference.
            let me = ManuallyDrop::new(this);
            // SAFETY: we are the sole strong reference with no weaks, so the
            // payload is valid and exclusively ours. `ptr::read` moves the `T`
            // out of the allocation.
            let val = unsafe { ptr::read(&**me) };
            let alloc = unsafe { ptr::read(&me.alloc) };
            let inner: &RcInner<T> = Rc::inner(me.deref());
            inner.dec_strong().expect("strong count underflow");
            // Drop a throwaway `Weak` standing in for this `Rc`'s implicit weak
            // reference.
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
    /// strong reference, or clone the reference otherwise.
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
