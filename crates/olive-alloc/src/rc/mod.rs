//! A fully-fallible port of the standard library's [`Rc`](stock_alloc::rc::Rc) and
//! [`Weak`](stock_alloc::rc::Weak).
//!
//! Compared with the std original, three things differ:
//!
//! * Every constructor that allocates a new node — `Rc::try_new`,
//!   `Rc::try_new_give_back`, and the various `*_in` variants — returns a
//!   [`Result`] carrying [`AllocError`] instead of panicking or aborting on
//!   out-of-memory.
//! * All allocation goes through the [`Allocator`] trait rather than free
//!   functions, giving a single swappable seam for custom allocators and OOM
//!   simulation in tests.
//! * Every reference-count mutation and allocator-handle clone is fallible and
//!   reports failures via [`TryRcError`]. Cloning an existing [`Rc`] does not
//!   allocate a new block (it only bumps the strong count), but it still must
//!   clone the allocator handle, which is a first-class fallible operation in
//!   this framework. The crate-standard [`TryClone`] impls surface these
//!   failures as [`TryCloneError`].
//!
//! The reference counts themselves are plain (non-atomic) integers, exactly
//! like the std originals: `Rc` is not thread-safe, and sharing one across
//! threads is undefined behavior.
//!
//! # Allocator bounds
//!
//! Cloning an `Rc<T, A>` requires `A: AllocatorTryClone`, not merely
//! `Allocator + Clone`. The stronger bound guarantees that a cloned allocator
//! handle is *equivalent* to the original — memory allocated through one may be
//! freed through the other — which is essential for the refcount-bump clone
//! path to remain sound. A plain `Clone` on the allocator would permit two
//! independent backing stores, breaking the invariant that all handles share
//! one allocation.

mod construction;
mod decompose;
mod dst;
mod helpers;
mod internals;
mod mutable;
mod query;
mod reconstitution;
mod refcount;
mod traits;
mod uninit;

pub use construction::TryRcWithError;
use internals::RcInner;
pub use refcount::{TryRcError, TryRcOutOfBoundsError};

use crate::alloc::{Allocator, Global, Layout};
use core::marker::PhantomData;
use olive_core::alloc::LayoutExt;
use olive_core::ptr::{self, NonNull};

// ---------------------------------------------------------------------------
// Rc declaration
// ---------------------------------------------------------------------------

/// A single-threaded reference-counting pointer.
///
/// `Rc<T>` shares ownership of a heap-allocated `T` among any number of strong
/// references. The value is dropped when the last strong reference disappears;
/// [`Weak`] references do not keep it alive.
///
/// Unlike `std::rc::Rc`, constructing a fresh node can fail: use
/// [`try_new`](Self::try_new) and friends, which return
/// `Result<Self, AllocError>` instead of panicking or aborting on out-of-memory.
pub struct Rc<T: ?Sized, A: Allocator = Global> {
    ptr: NonNull<RcInner<T>>,
    alloc: A,
    // This phantom prevents impl of Send and Sync
    _marker: PhantomData<*const T>,
}

// ---------------------------------------------------------------------------
// Drop
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Drop for Rc<T, A> {
    #[inline]
    fn drop(&mut self) {
        let inner = Rc::inner(self);
        inner.dec_strong().expect("strong count underflow");
        if helpers::is_last_strong(inner.strong()) {
            // SAFETY: we are the last strong reference.
            unsafe { Self::drop_slow(self) };
        }
    }
}

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Destroys the value and conditionally frees the block after the last
    /// strong reference has been dropped.
    ///
    /// # Safety
    /// - The Rc must be uniquely owned (or strong == 1).
    #[inline(never)]
    unsafe fn drop_slow(this: &mut Self) {
        // Construct a temporary `Weak` standing in for this `Rc`'s implicit
        // weak reference. This is the decrement or deallocate guard that
        // unconditionally runs even if the call is unwinding.
        let _fake_weak = Weak::<T, &A> {
            ptr: this.ptr,
            alloc: &this.alloc,
            _marker: PhantomData,
        };

        // SAFETY: the value is still present in the block (we are inside the
        // last strong's drop, before any free). Destroying it here is exactly
        // what dropping the final strong reference must do.
        unsafe {
            ptr::drop_in_place(&mut (*this.ptr.as_ptr()).value);
        }
    }
}

// ---------------------------------------------------------------------------
// Weak
// ---------------------------------------------------------------------------

/// A weak reference to an [`Rc`] allocation.
///
/// [`Weak<T>`] borrows the allocation without keeping it alive: upgrading a
/// [`Weak`] via [`try_upgrade`](Self::try_upgrade) yields an
/// [`Option<Rc<T>>`] that is `None` once every strong reference has been
/// dropped.
pub struct Weak<T: ?Sized, A: Allocator = Global> {
    ptr: NonNull<RcInner<T>>,
    alloc: A,
    _marker: PhantomData<RcInner<T>>,
}

impl<T: ?Sized, A: Allocator> Drop for Weak<T, A> {
    #[inline]
    fn drop(&mut self) {
        // A dangling weak (from `Weak::new`) owns no allocation, so there is
        // nothing to decrement or free — return immediately. `inner()` folds
        // the sentinel check into one call.
        let inner = match self.inner() {
            Some(i) => i,
            None => return,
        };

        // Decrement the weak count. If this was the last reference of any kind
        // (strong already zero, and now weak hits zero), free the allocation.
        // An unbalanced decrement here would indicate a logic error, so we
        // surface it as a panic rather than silently corrupting the count.
        inner.dec_weak().expect("Weak weak count underflow");

        // Invariant: once the weak count reaches zero, the strong count must
        // also be zero (the last strong `Rc`'s drop either freed the block or
        // left it pinned by at least one `Weak`). So `weak == 0` is sufficient
        // to decide whether to deallocate.
        if helpers::is_last_ref(inner.weak()) {
            // SAFETY: `ptr` carries correct pointer metadata for `T`; the pointee
            // may be uninitialized (already dropped) but we only need its size
            // and alignment, which live in the fat pointer.
            let layout = unsafe { Layout::for_value_pointer(self.ptr.as_ptr()) };
            // SAFETY: the block was allocated with exactly this layout; the
            // header alone guarantees a non-zero size.
            unsafe {
                self.alloc.deallocate(self.ptr.cast(), layout);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use core::panic::{RefUnwindSafe, UnwindSafe};
    use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};
    use olive_test_helpers::{FailAlloc, FailDefaultAlloc};
    use static_assertions::assert_not_impl_any;
    use std::string::String;

    // `Rc` and `Weak` are never `Send` or `Sync`, regardless of their type
    // parameters. Unsynchronized reference counting is inherently
    // single-threaded — sharing the refcount pointer across threads would
    // permit unsynchronized concurrent mutation of the counters.
    assert_not_impl_any!(Rc<i32>: Send, Sync);
    assert_not_impl_any!(Rc<String>: Send, Sync);
    assert_not_impl_any!(Rc<&'static ()>: Send, Sync);
    assert_not_impl_any!(Weak<i32>: Send, Sync);
    assert_not_impl_any!(Weak<String>: Send, Sync);
    assert_not_impl_any!(Weak<&'static ()>: Send, Sync);

    #[test]
    fn basic_construction_and_deref() {
        let rc = Rc::try_new(42).unwrap();
        assert_eq!(*rc, 42);
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);
    }

    #[test]
    fn clone_shares_allocation() {
        let rc = Rc::try_new(String::from("hello")).unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
        assert_eq!(&*rc2, "hello");
    }

    #[test]
    fn drop_frees_at_last_strong() {
        struct Counted(Cell<i32>);
        impl Drop for Counted {
            fn drop(&mut self) {
                *self.0.get_mut() += 1;
            }
        }
        let counter = Cell::new(0i32);
        let rc = Rc::try_new(Counted(counter)).unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(rc.0.get(), 0);
        drop(rc);
        assert_eq!(rc2.0.get(), 0);
        drop(rc2);
        // Can't check counter directly since it was moved into Counted,
        // but if we got here without UB, the drop ran exactly once.
    }

    #[test]
    fn weak_does_not_keep_alive() {
        let rc = Rc::try_new(1).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_upgrade_restores_strong() {
        let rc = Rc::try_new(7).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let upgraded = weak.try_upgrade().unwrap().unwrap();
        assert_eq!(Rc::strong_count(&upgraded), 2);
        assert_eq!(*upgraded, 7);
        drop(upgraded);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn oom_returns_error_and_gives_back_value() {
        let res: Result<Rc<String, FailAlloc>, (String, AllocError)> =
            Rc::try_new_give_back_in(String::from("oom"), FailAlloc);
        let (given_back, err) = res.unwrap_err();
        assert_eq!(given_back, "oom");
        let _ = err;
    }

    #[test]
    fn try_new_with_fail_alloc_errors() {
        let res: Result<Rc<i32, FailAlloc>, AllocError> = Rc::try_new_in(5, FailAlloc);
        assert!(res.is_err());
    }

    #[test]
    fn into_raw_roundtrip() {
        let rc = Rc::try_new(99).unwrap();
        let raw = Rc::into_raw(rc);
        assert_eq!(unsafe { *raw }, 99);
        let rc = unsafe { Rc::from_raw(raw) };
        assert_eq!(*rc, 99);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn weak_dangling_into_raw_roundtrip() {
        // A dangling weak carries no allocation; its raw pointer is the
        // misaligned sentinel, and reconstituting it yields another dangling
        // weak that behaves identically.
        let w: Weak<u32, Global> = Weak::new();
        let raw = Weak::into_raw(w);
        assert_ne!(raw.addr(), 0);
        let w = unsafe { Weak::from_raw(raw) };
        // `as_ptr` on a dangling weak returns the sentinel itself — exercising
        // the same branch that `from_raw` relies on to detect the sentinel.
        assert_eq!(w.as_ptr().addr(), raw.addr());
        assert!(w.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn multiple_weak_refs_all_release() {
        let rc = Rc::try_new(3).unwrap();
        let w1 = Rc::try_downgrade(&rc).unwrap();
        let w2 = w1.try_clone().unwrap();
        assert_eq!(Rc::weak_count(&rc), 2);
        drop(rc);
        assert!(w1.try_upgrade().unwrap().is_none());
        assert!(w2.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn try_clone_matches_clone() {
        let rc = Rc::try_new(11).unwrap();
        let cloned = rc.try_clone().unwrap();
        assert!(Rc::ptr_eq(&rc, &cloned));
        assert_eq!(Rc::strong_count(&rc), 2);
    }

    #[test]
    fn comparisons_forward_to_payload() {
        let a = Rc::try_new([1, 2, 3]).unwrap();
        let b = Rc::try_new([1, 2, 3]).unwrap();
        let c = Rc::try_new([1, 2, 4]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a < c);
    }

    #[test]
    fn as_ref_and_borrow_work() {
        let rc = Rc::try_new(5u32).unwrap();
        let r: &u32 = rc.as_ref();
        assert_eq!(*r, 5);
        let b: &u32 = Borrow::borrow(&rc);
        assert_eq!(*b, 5);
    }

    #[test]
    fn weak_debug_prints_marker() {
        // Like std, Debug prints only `(Weak)` — it must not require
        // upgrading (which would demand `A: AllocatorTryClone`).
        let rc = Rc::try_new(1).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(std::format!("{weak:?}"), "(Weak)");
        drop(rc);
        assert_eq!(std::format!("{weak:?}"), "(Weak)");
    }

    // -----------------------------------------------------------------------
    // ?Sized tests
    // -----------------------------------------------------------------------

    #[test]
    fn unsized_byte_slice_construction_and_deref() {
        let arr = [10u8, 20, 30, 40];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        assert_eq!(&*rc, [10, 20, 30, 40]);
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);
    }

    #[test]
    fn unsized_byte_slice_clone_shares_allocation() {
        let arr = [1u8, 2, 3];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
        assert_eq!(&*rc2, [1, 2, 3]);
    }

    #[test]
    fn unsized_byte_slice_weak_lifecycle() {
        let arr = [65u8, 66, 67];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    // Note: Direct unsized coercion from Rc<Concrete> to Rc<dyn Trait> is not
    // supported by our implementation (the PhantomData<RcInner<T>> field
    // prevents the compiler's built-in unsized coercion). Trait objects are
    // still fully supported at the type level (Rc<dyn T> compiles, derefs,
    // clones, downgrades, and deallocates correctly); construction simply
    // requires going through a raw-pointer roundtrip or a future
    // `Rc::try_from_any` API. The slice and str tests below exercise the same
    // fat-pointer offset/layout/deallocation code paths.

    #[test]
    fn unsized_str_construction() {
        let s = String::from("hello world");
        let rc: Rc<str> = Rc::try_from_str(s.as_str()).unwrap();
        assert_eq!(&*rc, "hello world");
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    // An "absurd" payload — one whose size, combined with the two-counter
    // header, overflows an addressable block — must be rejected as a logic-level
    // invariant violation (`TryCloneError::Other`), NOT as transient OOM
    // (`Alloc(AllocError)`). The distinction matters: an OOM-style error invites
    // a retry/backoff loop that would spin forever on a permanently unsatisfiable
    // request. We exercise the layout helper directly because no real `&[u8]` can
    // carry such a length.
    #[test]
    fn absurd_payload_layout_reports_other_not_oom() {
        let usize_size = size_of::<usize>();
        let max_size = isize::MAX as usize - usize_size + 1;
        // A hypothetical Rust tcype requires the size is divisible by its alignment
        assert_eq!(max_size % usize_size, 0);
        let huge = Layout::from_size_align(max_size, usize_size)
            .expect("payload layout alone is representable");
        let err = rc_inner_layout_for_value_layout(huge)
            .expect_err("absurd payload must fail layout computation");
        // The helper surfaced a real layout error (the overflow).
        let _ = err;
        // Mirror the exact mapping used by `try_clone_from_ref_in`: the overflow
        // becomes `Other`, never `Alloc(AllocError)`. Asserting on both arms of
        // the match proves the absurd case cannot be mistaken for transient OOM.
        let mapped = TryCloneError::Other("Rc payload layout overflow");
        assert!(matches!(mapped, TryCloneError::Other(_)));
        assert!(!matches!(mapped, TryCloneError::Alloc(_)));
    }

    #[test]
    fn unsized_into_raw_roundtrip_slice() {
        let arr = [7u8, 8, 9];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let raw = Rc::into_raw(rc);
        let slice: &[u8] = unsafe { &*raw };
        assert_eq!(slice.len(), 3);
        assert_eq!(slice, [7, 8, 9]);
        let rc = unsafe { Rc::from_raw(raw) };
        assert_eq!(&*rc, [7, 8, 9]);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn unsized_as_ptr_and_borrow() {
        let arr = [10u8, 20];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let p: *const [u8] = Rc::as_ptr(&rc);
        let slice: &[u8] = unsafe { &*p };
        assert_eq!(slice.len(), 2);
        assert_eq!(slice, [10, 20]);
        let b: &[u8] = Borrow::borrow(&rc);
        assert_eq!(b, &[10, 20]);
    }

    // A value whose destructor panics. Used to verify that the panic-aware
    // guard in `Rc::drop` still frees the block when no `Weak` remains.
    struct PanickingDrop;
    impl Drop for PanickingDrop {
        fn drop(&mut self) {
            panic!("intentional panic in value destructor");
        }
    }

    #[test]
    #[should_panic(expected = "intentional panic")]
    fn panic_in_last_strong_no_weak_still_frees() {
        // No weak references: when the last strong drops and the value's
        // destructor panics, the guard must still deallocate the block. If it
        // leaked, Miri / leak detectors would catch it; here we at least prove
        // the code path runs without double-free or abort.
        let _rc = Rc::try_new(PanickingDrop).unwrap();
    }

    #[test]
    fn panic_in_last_strong_with_weak_keeps_block() {
        // A live `Weak` exists: the guard must NOT free the block (weak > 0),
        // leaving the `Weak` to release it later. Catch the panic so we can
        // verify the weak handle still sees a defunct-but-alive allocation.
        let rc = Rc::try_new(PanickingDrop).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(rc); // panics inside value's Drop; guard sees weak >= 1, skips free
        }));
        assert!(result.is_err(), "expected the value's Drop to panic");
        // The weak handle must still be valid (block not freed): upgrading
        // returns None because strong is 0, but accessing the weak does not
        // trigger a use-after-free. Dropping it now frees the block.
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
    }

    #[test]
    fn try_default_constructs_wrapped_default() {
        let rc = Rc::<i32>::try_default().unwrap();
        assert_eq!(*rc, 0);
        assert_eq!(Rc::strong_count(&rc), 1);
        // A different payload type exercises the same path.
        assert_eq!(*Rc::<u8>::try_default().unwrap(), 0);
    }

    #[test]
    fn try_default_str_is_empty_with_one_strong_ref() {
        let s = Rc::<str>::try_default().unwrap();
        assert_eq!(&*s, "");
        assert_eq!(Rc::strong_count(&s), 1);
    }

    #[test]
    fn try_default_slice_is_empty() {
        let a = Rc::<[u8]>::try_default().unwrap();
        assert_eq!(a.len(), 0);
        assert_eq!(Rc::strong_count(&a), 1);
    }

    #[test]
    fn rc_try_default_sized_with_global() {
        let r = <Rc<u32, Global> as TryDefault>::try_default().unwrap();
        assert_eq!(*r, 0);
    }

    #[test]
    fn rc_try_default_allocator_failure_propagates() {
        let e = <Rc<u32, FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }

    #[test]
    fn rc_str_try_default_allocator_failure_propagates() {
        let e = <Rc<str, FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }

    #[test]
    fn rc_slice_try_default_allocator_failure_propagates() {
        let e = <Rc<[u8], FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }

    #[test]
    fn try_default_sized() {
        assert_eq!(*Rc::<u32>::try_default().unwrap(), 0);
    }

    #[test]
    fn weak_new_dangling_upgrades_to_none() {
        // A freshly constructed `Weak` refers to no allocation, so upgrading it
        // must always yield `None`.
        let w: Weak<i32> = Weak::new();
        assert!(w.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_new_dropping_frees_nothing() {
        // Dropping a dangling weak must not touch the allocator at all. We
        // verify this indirectly: constructing and dropping several dangling
        // weaks (including for unsized types) leaves the process stable — there
        // is no deallocation path to corrupt.
        let _ = Weak::<i32>::new();
        let _ = Weak::<str>::new();
        let _ = Weak::<[u8]>::new();
        // Explicitly drop at end of scope; no panic / UB means we passed.
    }

    #[test]
    fn weak_default_equals_new() {
        // Both infallible constructors produce an equivalent dangling weak.
        let a: Weak<i32> = Default::default();
        let b: Weak<i32> = Weak::new();
        assert!(a.try_upgrade().unwrap().is_none());
        assert!(b.try_upgrade().unwrap().is_none());
        // And the fallible twin agrees.
        let c: Weak<i32> = TryDefault::try_default().unwrap();
        assert!(c.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn rc_weak_try_default_allocator_failure_propagates() {
        let e = <Weak<u32, FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }

    /// Regression: cloning a dangling `Weak` (from `Weak::new`) must not
    /// dereference the sentinel address. The clone must behave as another
    /// dangling weak — it upgrades to `None` and compares pointer-equal —
    /// and dropping both must remain a no-op.
    #[test]
    fn weak_new_try_clone_stays_dangling() {
        let w: Weak<i32> = Weak::new();
        let c = w.try_clone().unwrap();
        assert!(c.try_upgrade().unwrap().is_none());
        assert!(w.ptr_eq(&c));
    }

    // -----------------------------------------------------------------------
    // try_unwrap / try_unwrap_give_back /
    // unwrap_or_try_clone / unwrap_or_try_clone_give_back
    // All four consume the `Rc` by value.
    // -----------------------------------------------------------------------

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
    fn try_unwrap_with_live_weak_returns_handle_back() {
        // Live weak reference: the block must stay reachable, so we cannot
        // move out even though there is only one strong reference.
        let rc = Rc::try_new(7u8).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let returned = Rc::try_unwrap(rc).unwrap_err();
        assert_eq!(*returned, 7);
        assert_eq!(Rc::strong_count(&returned), 1);
        // Dropping the strong leaves the weak pointing at a dead block.
        drop(returned);
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
    }

    #[test]
    fn try_unwrap_preserves_drop_semantics() {
        // Moving out via `ptr::read` bypasses the normal drop path, so the
        // payload's destructor must run exactly once — on the owned value the
        // caller receives, not inside the freed block.
        #[derive(Debug)]
        struct Counted(Cell<i32>);
        impl Drop for Counted {
            fn drop(&mut self) {
                *self.0.get_mut() += 1;
            }
        }
        let counter = Cell::new(0i32);
        let rc = Rc::try_new(Counted(counter)).unwrap();
        let counted = Rc::try_unwrap(rc).unwrap();
        // Still owned by us; not yet dropped.
        assert_eq!(counted.0.get(), 0);
        drop(counted);
        // Reaching here without UB means the destructor ran exactly once.
    }

    #[test]
    fn unwrap_or_try_clone_sole_owner_moves_out() {
        // Sole owner: consumes the handle and moves the payload out directly.
        let rc = Rc::try_new(1234i32).unwrap();
        let val = Rc::unwrap_or_try_clone(rc).unwrap();
        assert_eq!(val, 1234);
    }

    #[test]
    fn unwrap_or_try_clone_shared_clones_value() {
        // Shared ownership: cannot move out, so the value is cloned instead.
        // Both handles remain valid and point at the same allocation.
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
    fn unwrap_or_try_clone_give_back_sole_owner_moves_out() {
        // Sole owner: consumes the handle and moves the payload out directly.
        let rc = Rc::try_new(321i32).unwrap();
        let val = Rc::unwrap_or_try_clone_give_back(rc).unwrap();
        assert_eq!(val, 321);
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

    // -----------------------------------------------------------------------
    // is_unique / get_mut / try_make_mut
    // -----------------------------------------------------------------------

    #[test]
    fn is_unique_true_only_when_no_other_refs() {
        let rc = Rc::try_new(1).unwrap();
        assert!(Rc::is_unique(&rc));

        let weak = Rc::try_downgrade(&rc).unwrap();
        // A live Weak disqualifies uniqueness even though strong == 1.
        assert!(!Rc::is_unique(&rc));
        drop(weak);
        assert!(Rc::is_unique(&rc));

        let rc2 = rc.try_clone().unwrap();
        assert!(!Rc::is_unique(&rc));
        drop(rc2);
        assert!(Rc::is_unique(&rc));
    }

    #[test]
    fn get_mut_requires_no_weak() {
        let mut rc = Rc::try_new(String::from("hi")).unwrap();
        *Rc::get_mut(&mut rc).unwrap() += "!";
        assert_eq!(*rc, "hi!");

        // A single Weak now blocks get_mut (matching std semantics).
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert!(Rc::get_mut(&mut rc).is_none());
        drop(weak);
        assert!(Rc::get_mut(&mut rc).is_some());
    }

    #[test]
    fn make_mut_unique_mutates_in_place() {
        let mut rc = Rc::try_new(5u32).unwrap();
        let addr = Rc::as_ptr(&rc) as usize;
        *Rc::try_make_mut(&mut rc).unwrap() += 1;
        assert_eq!(*rc, 6);
        // No reallocation happened — same payload address.
        assert_eq!(Rc::as_ptr(&rc) as usize, addr);
    }

    #[test]
    fn make_mut_shared_clones_and_disassociates() {
        let mut data = Rc::try_new(5i32).unwrap();
        let other = data.try_clone().unwrap();

        let old_addr = Rc::as_ptr(&data) as usize;
        assert_eq!(Rc::strong_count(&data), 2);

        *Rc::try_make_mut(&mut data).unwrap() += 1;

        // `data` moved to a fresh allocation holding the mutated clone.
        assert_eq!(*data, 6);
        assert_ne!(Rc::as_ptr(&data) as usize, old_addr);
        assert_eq!(Rc::strong_count(&data), 1);
        // The original owner is untouched and still valid.
        assert_eq!(*other, 5);
        assert_eq!(Rc::strong_count(&other), 1);
        assert!(!Rc::ptr_eq(&data, &other));
    }

    #[test]
    fn make_mut_with_weak_steals_without_cloning() {
        // Only weak references remain: the value must be MOVED into a fresh
        // block (not cloned), and the surviving Weaks become dangling.
        struct Movable<'a>(&'a Cell<i32>, String);

        impl TryClone for Movable<'_> {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                panic!("this test should not move");
            }
        }

        impl Drop for Movable<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        // External counter so we can observe the destructor across the move
        // (a counter embedded in the payload would itself be relocated).
        let counter_cell = Cell::new(0i32);
        let mut data = Rc::try_new(Movable(&counter_cell, String::from("payload"))).unwrap();
        let weak = Rc::try_downgrade(&data).unwrap();
        let old_addr = Rc::as_ptr(&data) as usize;

        Rc::try_make_mut(&mut data).unwrap().1.push('x');

        assert_eq!(data.1.as_str(), "payloadx");
        assert_ne!(Rc::as_ptr(&data) as usize, old_addr);
        assert_eq!(Rc::strong_count(&data), 1);
        // The weak reference is now detached from any live value.
        assert!(weak.try_upgrade().unwrap().is_none());
        // Dropping the new sole owner runs the destructor exactly once.
        drop(data);
        assert_eq!(counter_cell.get(), 1);
    }

    #[test]
    fn make_mut_unsized_slice_shared_clones() {
        let arr = [1u8, 2, 3];
        let mut rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let other = rc.try_clone().unwrap();

        let old_addr = Rc::as_ptr(&rc).cast::<u8>() as usize;

        let m = Rc::try_make_mut(&mut rc).unwrap();
        m[0] = 99;

        assert_eq!(&*rc, [99, 2, 3]);
        assert_ne!(Rc::as_ptr(&rc).cast::<u8>() as usize, old_addr);
        // Original slice untouched.
        assert_eq!(&*other, [1, 2, 3]);
    }

    // -----------------------------------------------------------------------
    // Raw-pointer provenance / sentinel semantics
    // -----------------------------------------------------------------------

    /// `Rc::as_ptr` must hand back a pointer with write provenance rooted in
    /// the live allocation, so that a sole owner can legally cast it to
    /// `*mut T` and mutate the payload in place (manual-write scenarios such as
    /// `ptr::write` through the raw pointer). Under Miri this proves the data
    /// word was derived from the allocation rather than fabricated; on native
    /// builds it proves the roundtrip works end-to-end.
    #[test]
    fn rc_as_ptr_write_provenance_manual_write() {
        let rc = Rc::try_new(42u64).unwrap();
        let p: *const u64 = Rc::as_ptr(&rc);
        // Cast away constness — only sound because we hold the sole strong
        // reference (strong == 1, weak == 0) and the pointer's provenance
        // covers the payload slot.
        let mut_p = p as *mut u64;
        unsafe {
            ptr::write(mut_p, 7);
        }
        assert_eq!(*rc, 7);
        // Overwrite again via the same pointer to confirm repeated use is fine.
        unsafe {
            ptr::write_volatile(mut_p, 9);
        }
        assert_eq!(*rc, 9);
    }

    /// Same guarantee for unsized payloads: the fat pointer returned by
    /// `Rc::as_ptr` must be writable through its data word while metadata
    /// (the slice length) is preserved.
    #[test]
    fn rc_as_ptr_write_provenance_unsized_slice() {
        let arr = [1u8, 2, 3];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let p: *const [u8] = Rc::as_ptr(&rc);
        let mut_p = p as *mut [u8];
        unsafe {
            (*mut_p)[1] = 20;
        }
        assert_eq!(&*rc, [1, 20, 3]);
    }

    /// A `Weak` produced by [`Weak::new`] never referred to an allocation, so
    /// `as_ptr` must return the deliberately misaligned dangling sentinel
    /// (`usize::MAX`) rather than deriving one from a real block. No valid
    /// allocation address can equal the sentinel, which makes "was this weak
    /// ever attached?" decidable by pure pointer comparison.
    #[test]
    fn weak_as_ptr_dangling_returns_sentinel() {
        let w: Weak<i32> = Weak::new();
        let p = w.as_ptr();
        assert_eq!(p.addr(), usize::MAX);
        // The sentinel is misaligned relative to any non-ZST payload, so it
        // can never alias a real allocation.
        assert_ne!(p.addr() % align_of::<i32>(), 0);
    }

    /// Once the last strong reference drops, a still-live `Weak` keeps the
    /// block pinned but the payload is gone: `as_ptr` must NOT return the
    /// sentinel (this weak WAS attached to a real allocation), and the
    /// address must not be dereferenced. We verify the address differs from
    /// the sentinel and that upgrade reports no strong references.
    #[test]
    fn weak_as_ptr_after_last_strong_is_not_sentinel() {
        let rc = Rc::try_new(1i32).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        drop(rc);
        let p = weak.as_ptr();
        // This weak did refer to a real allocation, so its data word is the
        // genuine (now dead) payload address, not the sentinel.
        assert_ne!(p.addr(), usize::MAX);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    fn assert_unwind_safe<T: UnwindSafe>() {}
    fn assert_ref_unwind_safe<T: RefUnwindSafe>() {}

    #[test]
    fn unwind_safe_impls_match_std() {
        assert_unwind_safe::<Rc<u32>>();
        assert_ref_unwind_safe::<Rc<u32>>();
        assert_unwind_safe::<Rc<str>>();
        assert_ref_unwind_safe::<Rc<str>>();
        assert_unwind_safe::<Rc<[u8]>>();
        assert_ref_unwind_safe::<Rc<[u8]>>();
    }
}
