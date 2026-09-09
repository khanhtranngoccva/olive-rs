//! Fallible node-construction methods for [`Arc`](super::Arc) and [`Weak`](super::Weak).
//!
//! # Invariants
//!
//! Every [`Arc`] constructor returns a node whose strong count is exactly 1
//! and whose weak count is exactly 1 (the implicit weak reference held by the sole
//! strong owner). The payload is either fully initialized (`try_new`,
//! `try_new_give_back`, `write`) or deliberately left uninitialized
//! (`try_new_uninit`). No constructor leaves a partially-initialized block
//! or leak on the failure path.

use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr;

use crate::alloc::{Allocator, Global, Layout};
use olive_core::ptr::NonNull;

use super::pointers::{dangling_inner_ptr, initialize_arcinner, ptr_get_data_mut};
use super::{Arc, ArcInner, Weak};

// ---------------------------------------------------------------------------
// Global construction block
// ---------------------------------------------------------------------------

impl<T> Arc<T, Global> {
    /// Allocates a new `Arc<T>` containing `x` on the global allocator.
    ///
    /// # Errors
    ///
    /// Returns [`crate::alloc::AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new(x: T) -> Result<Self, crate::alloc::AllocError> {
        Self::try_new_in(x, Global)
    }

    /// Like [`try_new`](Self::try_new), but on allocation failure returns the
    /// unallocated `x` back to the caller alongside the error, so the value is
    /// not dropped silently.
    ///
    /// # Errors
    ///
    /// Returns [`crate::alloc::AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new_give_back(x: T) -> Result<Self, (T, crate::alloc::AllocError)> {
        Self::try_new_give_back_in(x, Global)
    }

    /// Allocates a new `Arc<MaybeUninit<T>>` containing uninitialized memory on
    /// the global allocator.
    ///
    /// # Errors
    ///
    /// Returns [`crate::alloc::AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_uninit() -> Result<Arc<MaybeUninit<T>, Global>, crate::alloc::AllocError> {
        Self::try_new_uninit_in(Global)
    }

    /// Allocates a new `Arc<T>` with all bytes zeroed on the global allocator.
    ///
    /// This is useful for types where zeroed memory represents a valid value
    /// (e.g. integers, `bool`, enums without data). For arbitrary types, the
    /// result may be invalid; use with care.
    ///
    /// # Safety note
    ///
    /// Although this method is safe to call, the resulting `T` is only valid
    /// if zeroed memory is a valid representation of `T`. The caller is
    /// responsible for ensuring this invariant.
    ///
    /// # Errors
    ///
    /// Returns [`crate::alloc::AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed() -> Result<Self, crate::alloc::AllocError> {
        Self::try_new_zeroed_in(Global)
    }
}

// ---------------------------------------------------------------------------
// Weak construction block
// ---------------------------------------------------------------------------

impl<T: ?Sized> Weak<T, Global> {
    /// Creates a new dangling [`Weak`] that does not point to any allocation.
    ///
    /// A dangling `Weak` owns no memory and never keeps a value alive.
    /// Dropping it is a no-op.
    #[inline]
    pub fn new() -> Self {
        // No allocation occurs, so this cannot fail. The pointer is pinned to
        // the misaligned dangling sentinel, which `Drop` and every access path
        // detect before touching memory.
        Weak {
            ptr: dangling_inner_ptr(),
            alloc: Global,
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Like [`new`](Self::new), but parameterized over the choice of allocator.
    #[inline]
    pub fn new_in(alloc: A) -> Self {
        Weak {
            ptr: dangling_inner_ptr(),
            alloc,
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Generic construction block (sized)
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Arc<T, A> {
    /// Like [`try_new`](Self::try_new), but parameterized over the choice of
    /// allocator for the returned `Arc`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::alloc::AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_in(x: T, alloc: A) -> Result<Self, crate::alloc::AllocError> {
        let b = Self::try_new_uninit_in(alloc)?;
        // SAFETY: this pointer is newly initialized, strong == 1.
        Ok(unsafe { b.write(x) })
    }

    /// Like [`try_new_give_back`](Self::try_new_give_back), but parameterized
    /// over the choice of allocator. On allocation failure returns the
    /// unallocated `x` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns [`crate::alloc::AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_give_back_in(x: T, alloc: A) -> Result<Self, (T, crate::alloc::AllocError)> {
        match Self::try_new_uninit_in(alloc) {
            Ok(b) => {
                // SAFETY: this pointer is newly initialized, strong == 1.
                Ok(unsafe { b.write(x) })
            }
            Err(e) => Err((x, e)),
        }
    }

    /// Allocates a new `Arc<MaybeUninit<T>>` containing uninitialized memory,
    /// parameterized over the allocator.
    ///
    /// # Errors
    ///
    /// Returns [`crate::alloc::AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_uninit_in(alloc: A) -> Result<Arc<MaybeUninit<T>, A>, crate::alloc::AllocError> {
        let layout = Layout::new::<ArcInner<MaybeUninit<T>>>();
        let block = alloc.allocate(layout)?;
        // SAFETY: a successful allocation returns a non-null, aligned pointer to
        // a fresh block of at least `layout.size()` bytes.
        let ptr =
            unsafe { NonNull::new_unchecked(block.cast::<ArcInner<MaybeUninit<T>>>().as_ptr()) };
        // Seed the refcount headers to (1, 1); the payload stays uninitialized.
        // SAFETY: a successful allocation returns a live, aligned block whose
        // header fields are not yet initialized, and no other thread can observe
        // it before this function returns.
        unsafe { initialize_arcinner(ptr.as_ptr()) };
        Ok(Arc {
            ptr,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Allocates a new `Arc<T>` with all bytes zeroed, parameterized over the
    /// allocator.
    ///
    /// # Errors
    ///
    /// Returns [`crate::alloc::AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed_in(alloc: A) -> Result<Self, crate::alloc::AllocError> {
        let layout = Layout::new::<ArcInner<T>>();
        let block = alloc.allocate_zeroed(layout)?;
        // SAFETY: `allocate_zeroed` returns a valid, aligned, non-null pointer
        // to a freshly allocated block of at least `layout.size()` bytes.
        let ptr = unsafe { NonNull::new_unchecked(block.cast::<ArcInner<T>>().as_ptr()) };
        // Seed the refcount headers to (1, 1) on top of the zero-filled block.
        // SAFETY: a successful allocation returns a live, aligned block whose
        // header fields are not yet initialized as valid atomics, and no other
        // thread can observe it before this function returns.
        unsafe { initialize_arcinner(ptr.as_ptr()) };
        Ok(Arc {
            ptr,
            alloc,
            _marker: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Uninit initialization helper
// ---------------------------------------------------------------------------

impl<T: Sized, A: Allocator> Arc<MaybeUninit<T>, A> {
    /// Writes `val` into this `Arc`'s payload slot and returns the initialized
    /// `Arc<T, A>`.
    ///
    /// This is the bridge from an `Arc<MaybeUninit<T>>` produced by
    /// [`try_new_uninit`](Self::try_new_uninit) to a fully-initialized
    /// `Arc<T>` without reinterpreting through raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure this `Arc`'s strong count is exactly 1 (i.e. it
    /// was just constructed and never shared). Writing the payload requires
    /// exclusive access to the allocation.
    #[inline]
    pub unsafe fn write(self, val: T) -> Arc<T, A> {
        // SAFETY: the caller guarantees strong == 1, so we exclusively own the
        // payload slot; wrapping `val` in `MaybeUninit` matches the slot's type
        // and leaves it fully initialized in place.
        unsafe {
            ptr::write(ptr_get_data_mut(self.ptr.as_ptr()), MaybeUninit::new(val));
        }

        // SAFETY: the payload slot was just fully written above, so it is now a
        // valid `T`; the refcount header is untouched and still (1, 1).
        unsafe { self.assume_init() }
    }

    /// Reinterprets this `Arc<MaybeUninit<T>, A>` as an initialized `Arc<T, A>`.
    ///
    /// This performs no initialization of the payload — it only changes the
    /// static type of the backing allocation from `ArcInner<MaybeUninit<T>>` to
    /// `ArcInner<T>`.
    ///
    /// # Safety
    ///
    /// The payload slot must already contain a fully-initialized.
    #[inline]
    pub unsafe fn assume_init(self) -> Arc<T, A> {
        // Establish up front that the uninit and init allocations have identical
        // layouts, so the pointer cast below preserves size and alignment.
        debug_assert_eq!(
            Layout::new::<ArcInner<MaybeUninit<T>>>(),
            Layout::new::<ArcInner<T>>()
        );

        // Suppress the drop of `self` so the original `Arc<MaybeUninit<T>>`
        // does not decrement the refcount and free the allocation that the
        // returned `Arc<T>` inherits.
        let me = ManuallyDrop::new(self);
        let ptr = me.ptr;
        let alloc = unsafe { ptr::read(&me.alloc) };

        let new_ptr = ptr.cast::<ArcInner<T>>();
        Arc {
            ptr: new_ptr,
            alloc,
            _marker: PhantomData,
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
    use crate::alloc::AllocError;
    use crate::test_helpers::FailAlloc;
    use std::string::String;

    use crate::sync::arc::pointers::is_dangling_weak;

    /// Reads the payload behind an `Arc` without relying on `Deref` (which has
    /// not landed yet). Only valid while the payload is initialized and no
    /// other thread mutates it — both hold in these single-threaded tests.
    fn peek<T, A: Allocator>(arc: &Arc<T, A>) -> &T {
        // SAFETY: the Arc owns a live allocation whose payload is initialized.
        // Projects the `value` field out of the fat pointer, mirroring the
        // parent module's `ptr_get_data`.
        #[allow(
            clippy::needless_borrow,
            reason = "Miri does not allow implicit autoref"
        )]
        unsafe {
            &(&*arc.ptr.as_ptr()).value
        }
    }

    // --- Global construction ------------------------------------------------

    #[test]
    fn try_new_initializes_counters_and_payload() {
        let arc = Arc::try_new(42u32).unwrap();
        assert_eq!(peek(&arc), &42);
        // Fresh node: exactly one strong owner and the implicit weak ref.
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
        assert_eq!(arc.inner().weak(), 1);
    }

    #[test]
    fn try_new_give_back_success_preserves_value() {
        let arc = Arc::try_new_give_back(String::from("hi")).unwrap();
        assert_eq!(&**peek(&arc), "hi");
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn try_new_uninit_reports_one_strong_owner() {
        let arc: Arc<MaybeUninit<i32>, Global> = Arc::try_new_uninit().unwrap();
        // The uninit handle still carries a real strong reference to its block.
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn try_new_zeroed_yields_zeroed_payload() {
        let arc: Arc<u64, Global> = Arc::try_new_zeroed().unwrap();
        assert_eq!(peek(&arc), &0u64);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    // --- Uninit → init bridge ----------------------------------------------

    #[test]
    fn write_initializes_uninit_arc() {
        let uninit: Arc<MaybeUninit<String>, Global> = Arc::try_new_uninit().unwrap();
        let arc = unsafe { uninit.write(String::from("bridged")) };
        assert_eq!(&**peek(&arc), "bridged");
        // The bridge must preserve the exact refcount state of the source.
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn assume_init_reinterprets_without_touching_counters_or_payload() {
        // Build an uninit handle, then fill its payload slot directly (bypassing
        // `write`) so we can prove `assume_init` is a pure type reinterpretation:
        // it must not rewrite the payload nor disturb the refcount header.
        let uninit: Arc<MaybeUninit<[u8; 4]>, Global> = Arc::try_new_uninit().unwrap();
        let data = unsafe { ptr_get_data_mut(uninit.ptr.as_ptr()) };
        // SAFETY: exclusive access (strong == 1); writing a valid `[u8; 4]` makes
        // the slot initialized, satisfying `assume_init`'s precondition.
        unsafe { *data = MaybeUninit::new([1u8, 2, 3, 4]) };

        let arc = unsafe { uninit.assume_init() };
        // Payload bits survive verbatim — `assume_init` did not overwrite them.
        assert_eq!(peek(&arc), &[1u8, 2, 3, 4]);
        // Counters are untouched by the reinterpretation.
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
    }

    #[test]
    fn assume_init_and_write_agree_on_layout_and_state() {
        // Two handles built identically; one bridged via `write`, the other by
        // filling the slot then calling `assume_init`. Both must converge on the
        // same observable state, proving `write` composes over `assume_init`.
        let a = unsafe {
            let u: Arc<MaybeUninit<i64>, Global> = Arc::try_new_uninit().unwrap();
            u.write(99i64)
        };
        let b = unsafe {
            let u: Arc<MaybeUninit<i64>, Global> = Arc::try_new_uninit().unwrap();
            *ptr_get_data_mut(u.ptr.as_ptr()) = MaybeUninit::new(99i64);
            u.assume_init()
        };
        assert_eq!(peek(&a), peek(&b));
        assert_eq!(Arc::strong_count(&a), Arc::strong_count(&b));
        assert_eq!(Arc::weak_count(&a), Arc::weak_count(&b));
    }

    // --- Generic (allocator-parameterized) construction ---------------------

    #[test]
    fn try_new_in_with_custom_allocator() {
        let drops = std::rc::Rc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::LocalCountingAlloc::new(drops.clone());
        let arc = Arc::try_new_in(7i64, alloc).unwrap();
        assert_eq!(peek(&arc), &7);
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);
        drop(arc);
        // The allocator handle was consumed by the Arc and dropped with it.
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn try_new_give_back_in_oom_returns_value() {
        let res: Result<Arc<String, FailAlloc>, (String, AllocError)> =
            Arc::try_new_give_back_in(String::from("oom"), FailAlloc);
        match res {
            Err((given_back, _err)) => assert_eq!(given_back, "oom"),
            Ok(_) => panic!("expected allocation failure with FailAlloc"),
        }
    }

    #[test]
    fn try_new_in_oom_errors() {
        let res: Result<Arc<i32, FailAlloc>, AllocError> = Arc::try_new_in(5, FailAlloc);
        assert!(res.is_err());
    }

    #[test]
    fn try_new_uninit_in_oom_errors() {
        let res: Result<Arc<MaybeUninit<i32>, FailAlloc>, AllocError> =
            Arc::try_new_uninit_in(FailAlloc);
        assert!(res.is_err());
    }

    #[test]
    fn try_new_zeroed_in_oom_errors() {
        let res: Result<Arc<u8, FailAlloc>, AllocError> = Arc::try_new_zeroed_in(FailAlloc);
        assert!(res.is_err());
    }

    // --- Cross-checks -------------------------------------------------------

    #[test]
    fn global_and_generic_constructors_agree() {
        // Both routes allocate a fresh node with identical initial counters.
        let g = Arc::try_new(1u8).unwrap();
        assert_eq!(Arc::strong_count(&g), 1);
        assert_eq!(peek(&g), &1u8);
        let generic = Arc::try_new_in(1u8, Global).unwrap();
        assert_eq!(Arc::strong_count(&generic), 1);
        assert_eq!(peek(&generic), &1u8);
    }

    // --- Weak construction ---------------------------------------------------

    #[test]
    fn weak_new_is_dangling() {
        // A freshly constructed `Weak` refers to no allocation: its pointer is
        // the dangling sentinel and it owns no inner handle.
        let w: Weak<i32, Global> = Weak::new();
        assert!(is_dangling_weak(w.ptr.as_ptr()));
        assert!(w.inner().is_none());
    }

    #[test]
    fn weak_new_in_is_dangling_with_custom_allocator() {
        // The allocator-parameterized form is equally dangling, but must carry
        // the given allocator handle — which we observe being dropped exactly
        // once when the `Weak` itself is dropped.
        let drops = std::rc::Rc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::LocalCountingAlloc::new(drops.clone());
        let w: Weak<u64, _> = Weak::new_in(alloc);
        assert!(is_dangling_weak(w.ptr.as_ptr()));
        assert!(w.inner().is_none());
        drop(w);
        // The allocator handle was consumed by the Weak and dropped with it.
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn weak_new_dropping_frees_nothing() {
        // Dropping a dangling weak must not touch the allocator at all (there is
        // no deallocation path to corrupt). Constructing and dropping several —
        // including for unsized types — leaving the process stable proves this.
        let _ = Weak::<i32>::new();
        let _ = Weak::<str>::new();
        let _ = Weak::<[u8]>::new();
        // Explicitly dropped at end of scope; no leak or UB or panic means we passed.
    }

    #[test]
    fn weak_new_global_and_generic_agree() {
        // Both routes produce an equivalent dangling weak on the global
        // allocator.
        let g: Weak<i32, Global> = Weak::new();
        let generic: Weak<i32, Global> = Weak::new_in(Global);
        assert!(is_dangling_weak(g.ptr.as_ptr()));
        assert!(is_dangling_weak(generic.ptr.as_ptr()));
        assert!(g.inner().is_none());
        assert!(generic.inner().is_none());
    }
}
