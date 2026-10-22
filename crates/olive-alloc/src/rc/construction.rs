use core::cell::Cell;
use core::fmt::{self, Debug, Display, Formatter};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::pin::Pin;

use super::helpers;
use super::internals::RcInner;
use super::uninit::UniqueRcUninit;
use super::{Rc, Weak};
use crate::alloc::{AllocError, Allocator, Global, Layout, StaticAllocator};
use olive_core::alloc::AllocatorTryDefault;
use olive_core::ptr::{self, NonNull};
use olive_core::try_traits::{TryCloneError, TryDefault, TryDefaultError};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned by [`Rc::try_new_cyclic`] and [`Rc::try_new_cyclic_in`],
/// or any future error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryRcWithError<E> {
    /// Allocating the `RcInner<T>` block failed.
    Alloc(AllocError),
    /// The user callback returned an error, aborting construction. The
    /// partially-built allocation was released; no `Rc` was produced.
    Callback(E),
}

impl<E> From<AllocError> for TryRcWithError<E> {
    #[inline]
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl<E: Display> Display for TryRcWithError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => write!(f, "cyclic Rc allocation failed: {e}"),
            Self::Callback(e) => write!(f, "cyclic construction callback failed: {e}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for TryRcWithError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Alloc(e) => Some(e),
            Self::Callback(e) => Some(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Rc global construction block
// ---------------------------------------------------------------------------

impl<T> Rc<T, Global> {
    /// Allocates a new `Rc<T>` containing `x` on the global allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new(x: T) -> Result<Self, AllocError> {
        Self::try_new_in(x, Global)
    }

    /// Like [`try_new`](Self::try_new), but on allocation failure returns the
    /// unallocated `x` back to the caller alongside the error, so the value is
    /// not dropped silently.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new_give_back(x: T) -> Result<Self, (T, AllocError)> {
        match Self::try_new_uninit_in(Global) {
            Ok(b) => {
                // SAFETY: this pointer is newly initialized, strong == 1 and weak == 0
                // (excluding implicit weak ref).
                Ok(unsafe { b.write(x) })
            }
            Err(e) => Err((x, e)),
        }
    }

    /// Allocates a new `Rc<MaybeUninit<T>>` containing uninitialized memory on
    /// the global allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_uninit() -> Result<Rc<MaybeUninit<T>, Global>, AllocError> {
        Self::try_new_uninit_in(Global)
    }

    /// Allocates a new `Rc<MaybeUninit<T>>` with all bytes zeroed on
    /// the global allocator.
    ///
    /// This is useful for types where zeroed memory represents a valid value
    /// (e.g. integers, `bool`, enums without data), in which case a simple
    /// [`Self::assume_init`] is sufficient.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed() -> Result<Rc<MaybeUninit<T>>, AllocError> {
        Self::try_new_zeroed_in(Global)
    }

    /// Creates a cyclic `Rc` on the global allocator using a callback to wire
    /// up the cycle atomically.
    ///
    /// See [`try_new_cyclic_in`](Self::try_new_cyclic_in) for details.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcWithError<E>`]: either an [`AllocError`] if allocating
    /// the block fails, or the callback's own error `E`.
    #[inline]
    pub fn try_new_cyclic<E, F>(f: F) -> Result<Self, TryRcWithError<E>>
    where
        F: FnOnce(&Weak<T, Global>) -> Result<T, E>,
    {
        Self::try_new_cyclic_in(f, Global)
    }

    /// Allocates a new `Rc<T>` containing `x` and pins it in place, returning
    /// a `Pin<Rc<T>>`.
    ///
    /// If `T` does not implement [`Unpin`], then `*rc` will be pinned in memory
    /// and unable to be moved.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_pin(x: T) -> Result<Pin<Self>, AllocError> {
        let rc = Self::try_new(x)?;
        // SAFETY: a freshly allocated `Rc` owns its payload exclusively and the
        // global allocator never reclaims live memory except via an explicit
        // deallocation, so the pointee is stably located regardless of whether
        // `T: Unpin`.
        Ok(unsafe { Pin::new_unchecked(rc) })
    }

    /// Like [`try_pin`](Self::try_pin), but on allocation failure returns the
    /// unallocated `x` back to the caller alongside the error, so the value is
    /// not dropped silently.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_pin_give_back(x: T) -> Result<Pin<Self>, (T, AllocError)> {
        match Self::try_new_give_back(x) {
            Ok(rc) => {
                // SAFETY: as for [`try_pin`](Self::try_pin).
                Ok(unsafe { Pin::new_unchecked(rc) })
            }
            Err(given_back) => Err(given_back),
        }
    }
}

// ---------------------------------------------------------------------------
// Rc generic construction block (sized)
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Rc<T, A> {
    /// Like [`try_new`](Self::try_new), but parameterized over the choice of
    /// allocator for the returned `Rc`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_in(x: T, alloc: A) -> Result<Self, AllocError> {
        let b = Self::try_new_uninit_in(alloc)?;
        // SAFETY: this pointer is newly initialized, strong == 1 and weak == 0
        // (excluding implicit weak ref).
        Ok(unsafe { b.write(x) })
    }

    /// Like [`try_new_give_back`](Self::try_new_give_back), but parameterized
    /// over the choice of allocator. On allocation failure returns the
    /// unallocated `x` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_give_back_in(x: T, alloc: A) -> Result<Self, (T, AllocError)> {
        match Self::try_new_uninit_in(alloc) {
            Ok(b) => {
                // SAFETY: this pointer is newly initialized, strong == 1 and weak == 0
                // (excluding implicit weak ref).
                Ok(unsafe { b.write(x) })
            }
            Err(e) => Err((x, e)),
        }
    }

    /// Allocates a new `Rc<MaybeUninit<T>>` containing uninitialized memory.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_uninit_in(alloc: A) -> Result<Rc<MaybeUninit<T>, A>, AllocError> {
        let uninit = UniqueRcUninit::<MaybeUninit<T>, A>::try_new(alloc)?;
        // SAFETY: we are initializing a MaybeUninit.
        Ok(unsafe { uninit.into_rc() })
    }

    /// Allocates a new `Rc<MaybeUninit<T>>` with all bytes zeroed, parameterized over the
    /// allocator.
    ///
    /// This is useful for types where zeroed memory represents a valid value
    /// (e.g. integers, `bool`, enums without data), in which case a simple
    /// [`Self::assume_init`] is sufficient.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed_in(alloc: A) -> Result<Rc<MaybeUninit<T>, A>, AllocError> {
        let layout = Layout::new::<RcInner<T>>();
        let block = alloc.allocate_zeroed(layout)?;
        // SAFETY: `allocate_zeroed` returns a valid, aligned, non-null pointer
        // to a freshly allocated block of at least `layout.size()` bytes.
        let ptr =
            unsafe { NonNull::new_unchecked(block.cast::<RcInner<MaybeUninit<T>>>().as_ptr()) };
        // Initialize the refcount headers (the rest of the block is already
        // zeroed by the allocator).
        unsafe {
            let inner = ptr.as_ptr();
            ptr::write(&raw mut (*inner).strong, Cell::new(1));
            ptr::write(&raw mut (*inner).weak, Cell::new(1));
        }
        Ok(Rc {
            ptr,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Creates a cyclic `Rc` using a callback to wire up the cycle atomically, but it
    /// is generic over the allocator.
    ///
    /// ```ignore
    /// struct Node { name: String, parent: Option<Weak<Node>> }
    ///
    /// let node = Rc::try_new_cyclic_in(|weak| {
    ///     // Build the value around the back-reference.
    ///     Ok(Node { name: "root".into(), parent: Some(weak.try_clone().unwrap()) })
    /// }, Global).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TryRcWithError<E>`]: either an [`AllocError`] if allocating
    /// the block fails, or the callback's own error `E`.
    #[inline]
    pub fn try_new_cyclic_in<E, F>(f: F, alloc: A) -> Result<Self, TryRcWithError<E>>
    where
        F: FnOnce(&Weak<T, A>) -> Result<T, E>,
    {
        // Step 1: allocate the block directly with (strong=0, weak=1) and an
        // uninitialized payload.
        let layout = Layout::new::<RcInner<T>>();
        let block = alloc.allocate(layout).map_err(TryRcWithError::Alloc)?;
        // SAFETY: a successful allocation returns a non-null, aligned pointer.
        let inner = unsafe { NonNull::new_unchecked(block.cast::<RcInner<T>>().as_ptr()) };
        unsafe {
            let p = inner.as_ptr();
            ptr::write(&raw mut (*p).strong, Cell::new(0));
            ptr::write(&raw mut (*p).weak, Cell::new(1));
        }

        // Step 2: construct a plain Weak handle owning the weak=1 count.
        // From this point on, any failure or panic drops the Weak, bringing
        // weak to zero and freeing the block. The payload is still uninit,
        // so the block contains nothing that needs destructing.
        let weak = Weak {
            ptr: inner,
            alloc,
            _marker: PhantomData,
        };

        // Step 3: invoke the callback. It receives the Weak pinning the
        // (payload-uninit) block and returns the constructed value.
        let value = f(&weak).map_err(TryRcWithError::Callback)?;

        // Step 4: initialize the payload slot with the constructed value.
        // SAFETY: the block was allocated with exactly this layout and the
        // payload slot is currently uninitialized.
        unsafe {
            ptr::write(helpers::ptr_get_data_mut(inner.as_ptr()), value);
        }

        // Step 5: increment strong 0 → 1. Fresh allocation guarantees no
        // overflow, so we can safely expect.
        assert_eq!(unsafe { (*inner.as_ptr()).strong() }, 0);
        unsafe { (*inner.as_ptr()).inc_strong() }.expect("strong count is 0, cannot overflow");

        // Step 6: consume the Weak WITHOUT dropping it, so its weak count
        // becomes the implicit shared weak of the returned Rc. This yields
        // the standard (strong=1, weak=1) final state.
        let (rc_data_ptr, weak_alloc) = weak.into_raw_with_allocator();

        // SAFETY: `rc_data_ptr` was produced by `ptr_get_data` on our fresh,
        // fully-initialized block, and `weak_alloc` is the same allocator.
        Ok(unsafe { Rc::from_raw_in(rc_data_ptr, weak_alloc) })
    }

    /// Like [`try_pin`](Self::try_pin), but parameterized over the choice of
    /// allocator for the returned `Rc`.
    ///
    /// Requires `A: StaticAllocator` because a pinned pointee must remain at a
    /// stable address for its whole lifetime; only allocators that promise not
    /// to invalidate live memory without an explicit deallocation can back a
    /// pinned pointer.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_pin_in(x: T, alloc: A) -> Result<Pin<Self>, AllocError>
    where
        A: StaticAllocator,
    {
        let rc = Self::try_new_in(x, alloc)?;
        // SAFETY: a freshly allocated `Rc` owns its payload exclusively, and
        // `A: StaticAllocator` guarantees the backing memory stays valid until
        // an explicit deallocation, so the pointee is stably located regardless
        // of whether `T: Unpin`.
        Ok(unsafe { Pin::new_unchecked(rc) })
    }

    /// Like [`try_pin_give_back`](Self::try_pin_give_back), but parameterized
    /// over the choice of allocator. On allocation failure returns the
    /// unallocated `x` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_pin_give_back_in(x: T, alloc: A) -> Result<Pin<Self>, (T, AllocError)>
    where
        A: StaticAllocator,
    {
        match Self::try_new_give_back_in(x, alloc) {
            Ok(rc) => {
                // SAFETY: as for [`try_pin_in`](Self::try_pin_in).
                Ok(unsafe { Pin::new_unchecked(rc) })
            }
            Err(given_back) => Err(given_back),
        }
    }
}

// ---------------------------------------------------------------------------
// Weak global construction block (sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized> Weak<T, Global> {
    /// Creates a new dangling `Weak<T>` that does not point to any allocation.
    #[inline]
    pub fn new() -> Self {
        Weak {
            ptr: helpers::dangling_inner_ptr(),
            alloc: Global,
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Weak generic construction block (sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Creates a new dangling `Weak<T, A>` that does not point to any
    /// allocation, using the given allocator.
    ///
    /// This is the fallible analogue of std's `Weak::new_in`: because no
    /// allocation is performed, it cannot fail — the method exists purely for
    /// API symmetry with the `_in` family.
    #[inline]
    pub fn new_in(alloc: A) -> Self {
        Weak {
            ptr: helpers::dangling_inner_ptr(),
            alloc,
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Rc Default/TryDefault construction
// ---------------------------------------------------------------------------

impl<T: TryDefault, A: AllocatorTryDefault> TryDefault for Rc<T, A> {
    fn try_default() -> Result<Self, TryDefaultError> {
        let uninit = Self::try_new_uninit_in(A::try_default()?).map_err(TryDefaultError::Alloc)?;
        let value = T::try_default()?;
        // SAFETY: we just initialized the Rc with strong == 1 and weak == 0
        // (excluding the implicit ref).
        Ok(unsafe { uninit.write(value) })
    }
}

/// Converts a [`TryCloneError`] into a [`TryDefaultError`].
fn try_clone_error_to_default(e: TryCloneError) -> TryDefaultError {
    match e {
        TryCloneError::Reserve(r) => TryDefaultError::Reserve(r),
        TryCloneError::Alloc(a) => TryDefaultError::Alloc(a),
        TryCloneError::Other(m) => TryDefaultError::Other(m),
    }
}

impl<A: AllocatorTryDefault> TryDefault for Rc<str, A> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Self::try_from_str_in("", A::try_default()?).map_err(try_clone_error_to_default)
    }
}

impl<T, A: AllocatorTryDefault> TryDefault for Rc<[T], A> {
    fn try_default() -> Result<Self, TryDefaultError> {
        let empty: &[T] = &[];
        let uninit = UniqueRcUninit::<[T], _>::try_new_for_value(empty, A::try_default()?)
            .map_err(try_clone_error_to_default)?;
        // SAFETY: we are dealing with empty slices.
        Ok(unsafe { uninit.into_rc() })
    }
}

// ---------------------------------------------------------------------------
// Weak Default/TryDefault construction
// ---------------------------------------------------------------------------

impl<T: ?Sized> Default for Weak<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ?Sized, A: AllocatorTryDefault> TryDefault for Weak<T, A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new_in(A::try_default()?))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::CountingAllocator;
    use olive_core::try_traits::try_clone::TryClone;
    use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};
    use olive_test_helpers::{FailAlloc, FailDefaultAlloc};
    use std::string::String;

    struct CyclicNode {
        name: &'static str,
        parent: Option<Weak<CyclicNode>>,
    }

    #[test]
    fn basic_construction_and_deref() {
        let rc = Rc::try_new(42).unwrap();
        assert_eq!(*rc, 42);
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);
    }

    #[test]
    fn try_new_give_back_moves_value_in() {
        // The give-back variant takes ownership of the payload and hands
        // back a live handle when the allocation succeeds.
        let rc = Rc::try_new_give_back(String::from("moved in")).unwrap();
        assert_eq!(&*rc, "moved in");
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);
    }

    #[test]
    fn try_new_in_constructs_with_counting_allocator() {
        // The `_in` variants build with a caller-supplied allocator; the
        // counting allocator proves exactly one block is allocated for the
        // payload and released again on drop.
        let alloc = CountingAllocator::new();
        let rc: Rc<u32, &CountingAllocator> = Rc::try_new_in(42, &alloc).unwrap();
        assert_eq!(*rc, 42);
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(alloc.allocations(), 1);
        drop(rc);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_new_give_back_in_moves_value_in() {
        let alloc = CountingAllocator::new();
        let rc: Rc<String, &CountingAllocator> =
            Rc::try_new_give_back_in(String::from("given back"), &alloc).unwrap();
        assert_eq!(&*rc, "given back");
        assert_eq!(alloc.allocations(), 1);
        drop(rc);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_new_in_with_fail_alloc_errors() {
        let res: Result<Rc<i32, FailAlloc>, AllocError> = Rc::try_new_in(5, FailAlloc);
        assert!(res.is_err());
    }

    #[test]
    fn try_new_give_back_in_oom_returns_error_and_gives_back_value() {
        let res: Result<Rc<String, FailAlloc>, (String, AllocError)> =
            Rc::try_new_give_back_in(String::from("oom"), FailAlloc);
        let (given_back, err) = res.unwrap_err();
        assert_eq!(given_back, "oom");
        let _ = err;
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
        // Dropping a dangling weak must not touch the allocator at all: the
        // sentinel refers to no allocation, so construction and drop must
        // both produce zero allocation and deallocation traffic.
        let alloc = CountingAllocator::new();
        {
            let w: Weak<u32, &CountingAllocator> = Weak::new_in(&alloc);
            assert!(w.try_upgrade().unwrap().is_none());
            assert_eq!(alloc.allocations(), 0);
            assert_eq!(alloc.deallocations(), 0);
        }
        assert_eq!(alloc.allocations(), 0);
        assert_eq!(alloc.deallocations(), 0);
        // The Global dangling weak walks the same zero-traffic path,
        // including for unsized payloads.
        let _ = Weak::<i32>::new();
        let _ = Weak::<str>::new();
        let _ = Weak::<[u8]>::new();
    }

    #[test]
    fn try_new_uninit_write_and_assume_init() {
        let mut rc = Rc::<u64>::try_new_uninit().unwrap();
        *Rc::get_mut(&mut rc).unwrap() = MaybeUninit::new(123);
        // SAFETY: the payload slot was initialized above.
        let rc = unsafe { rc.assume_init() };
        assert_eq!(*rc, 123);
    }

    #[test]
    fn try_new_zeroed_reads_zero_then_mutates() {
        let rc = Rc::<u64>::try_new_zeroed().unwrap();
        // SAFETY: zeroed bytes are a valid `u64`.
        let mut rc = unsafe { rc.assume_init() };
        assert_eq!(*rc, 0);
        *Rc::get_mut(&mut rc).unwrap() = 55;
        assert_eq!(*rc, 55);
    }

    #[test]
    fn try_new_uninit_in_and_zeroed_in_with_counting_allocator() {
        let alloc = CountingAllocator::new();
        let mut rc: Rc<MaybeUninit<u32>, &CountingAllocator> =
            Rc::try_new_uninit_in(&alloc).unwrap();
        assert_eq!(alloc.allocations(), 1);
        Rc::get_mut(&mut rc).unwrap().write(7);
        // SAFETY: initialized above.
        let rc = unsafe { rc.assume_init() };
        assert_eq!(*rc, 7);
        drop(rc);
        assert_eq!(alloc.deallocations(), 1);
        let rc: Rc<MaybeUninit<u32>, &CountingAllocator> = Rc::try_new_zeroed_in(&alloc).unwrap();
        assert_eq!(alloc.allocations(), 2);
        // SAFETY: zeroed bytes are a valid `u32`.
        let rc = unsafe { rc.assume_init() };
        assert_eq!(*rc, 0);
        drop(rc);
        assert_eq!(alloc.deallocations(), 2);
    }

    #[test]
    fn try_new_cyclic_builds_back_reference() {
        let leaf = Rc::try_new_cyclic(|weak| -> Result<CyclicNode, core::convert::Infallible> {
            Ok(CyclicNode {
                name: "leaf",
                parent: Some(weak.try_clone().unwrap()),
            })
        })
        .unwrap();
        assert_eq!(leaf.name, "leaf");
        // The back-reference upgrades to the very same allocation.
        let parent = leaf
            .parent
            .as_ref()
            .unwrap()
            .try_upgrade()
            .unwrap()
            .unwrap();
        assert!(Rc::ptr_eq(&leaf, &parent));
        assert_eq!(Rc::strong_count(&leaf), 2);
        assert_eq!(Rc::weak_count(&leaf), 1);
    }

    #[test]
    fn try_new_cyclic_callback_error_releases_block() {
        let alloc = CountingAllocator::new();
        let res: Result<Rc<u32, &CountingAllocator>, TryRcWithError<&'static str>> =
            Rc::try_new_cyclic_in(|_weak| Err("no cycle today"), &alloc);
        assert!(matches!(
            res,
            Err(TryRcWithError::Callback("no cycle today"))
        ));
        // The half-built block was released; nothing is leaked.
        assert_eq!(alloc.allocations(), 1);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_new_cyclic_in_with_counting_allocator() {
        let alloc = CountingAllocator::new();
        // The payload cannot embed a `Weak<CyclicNodeIn, &CountingAllocator>`
        // (the allocator borrow is not nameable in a struct field), so the
        // back-reference is parked in a cell instead.
        let parked: Cell<Option<Weak<u32, &CountingAllocator>>> = Cell::new(None);
        let rc: Rc<u32, &CountingAllocator> = Rc::try_new_cyclic_in(
            |weak| -> Result<u32, core::convert::Infallible> {
                parked.set(Some(weak.try_clone().unwrap()));
                Ok(42)
            },
            &alloc,
        )
        .unwrap();
        assert_eq!(alloc.allocations(), 1);
        assert_eq!(*rc, 42);
        let upgraded = parked.take().unwrap().try_upgrade().unwrap().unwrap();
        assert!(Rc::ptr_eq(&rc, &upgraded));
        drop(upgraded);
        drop(rc);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_new_cyclic_in_alloc_failure_propagates() {
        let res: Result<Rc<u32, FailAlloc>, TryRcWithError<core::convert::Infallible>> =
            Rc::try_new_cyclic_in(|_weak| unreachable!(), FailAlloc);
        assert!(matches!(res, Err(TryRcWithError::Alloc(_))));
    }

    #[test]
    fn try_pin_returns_pinned_pointer() {
        let pinned = Rc::try_pin(21).unwrap();
        assert_eq!(*pinned, 21);
        // `Rc` is `Unpin`, so the pinned wrapper can be unwrapped and the
        // value mutated in place.
        let mut rc = Pin::into_inner(pinned);
        *Rc::get_mut(&mut rc).unwrap() = 22;
        assert_eq!(*rc, 22);
    }

    #[test]
    fn try_pin_in_oom() {
        let _err = Rc::<i32, FailAlloc>::try_pin_in(9, FailAlloc).unwrap_err();
    }

    #[test]
    fn try_pin_in_with_counting_allocator() {
        let alloc = CountingAllocator::new();
        let pinned = Rc::try_pin_in(5, &alloc).unwrap();
        assert_eq!(*pinned, 5);
        drop(pinned);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_pin_give_back_in_oom_returns_value() {
        let (given_back, _err) =
            Rc::<i32, FailAlloc>::try_pin_give_back_in(9, FailAlloc).unwrap_err();
        assert_eq!(given_back, 9);
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
    fn try_default_sized_with_global() {
        let r = <Rc<u32, Global> as TryDefault>::try_default().unwrap();
        assert_eq!(*r, 0);
    }

    #[test]
    fn try_default_allocation_failure_propagates() {
        // `FailAlloc`'s `TryDefault` succeeds, so the error here comes from
        // allocating the block itself, not from `A::try_default()`.
        let e = <Rc<u32, FailAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(e, TryDefaultError::Alloc(_)));
    }

    #[test]
    fn try_default_allocator_failure_propagates() {
        let e = <Rc<u32, FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }

    #[test]
    fn try_default_str_is_empty_with_one_strong_ref() {
        let s = Rc::<str>::try_default().unwrap();
        assert_eq!(&*s, "");
        assert_eq!(Rc::strong_count(&s), 1);
    }

    #[test]
    fn try_default_str_allocation_failure_propagates() {
        // The empty string still needs a block for the counters, which
        // `FailAlloc` refuses.
        let e = <Rc<str, FailAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(e, TryDefaultError::Alloc(_)));
    }

    #[test]
    fn try_default_str_allocator_failure_propagates() {
        let e = <Rc<str, FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }

    #[test]
    fn try_default_slice_is_empty() {
        let a = Rc::<[u8]>::try_default().unwrap();
        assert_eq!(a.len(), 0);
        assert_eq!(Rc::strong_count(&a), 1);
    }

    #[test]
    fn try_default_slice_allocation_failure_propagates() {
        // The empty slice still needs a block for the counters, which
        // `FailAlloc` refuses.
        let e = <Rc<[u8], FailAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(e, TryDefaultError::Alloc(_)));
    }

    #[test]
    fn try_default_slice_allocator_failure_propagates() {
        let e = <Rc<[u8], FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
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
}
