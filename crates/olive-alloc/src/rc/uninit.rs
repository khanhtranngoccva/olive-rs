use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};

use super::Rc;
use super::helpers;
use super::internals::RcInner;
use crate::alloc::{AllocError, Allocator, Layout};
use olive_core::ptr::PointerExt;
use olive_core::ptr::{self, NonNull};
use olive_core::try_traits::try_clone::TryCloneError;

// ---------------------------------------------------------------------------
// UniqueRcUninit — intermediate allocation handle
// ---------------------------------------------------------------------------

/// A uniquely-owned, freshly-allocated `RcInner<T>` block whose payload region
/// is still uninitialized.
///
/// This struct sits between "raw allocation" and "finished `Rc<T>`": it owns
/// the heap block, has already initialized the two reference-count headers to
/// `(strong = 1, weak = 1)`, but leaves the payload slot for the caller to fill
/// via [`data_ptr`](Self::data_ptr). Once the caller has written the payload,
/// calling [`into_rc`](Self::into_rc) consumes the handle and produces the
/// final `Rc<T, A>`.
///
/// If dropped without being converted (e.g. due to an early return or panic),
/// the block is deallocated automatically — no leaks.
///
/// There are two construction paths:
///
/// * [`try_new`](Self::try_new) — for **sized** `T`, where the layout is
///   derived from the type itself. Use this when constructing a value from
///   scratch (e.g. `Rc::try_new(x)`).
/// * [`try_new_for_value`](Self::try_new_for_value) — for **potentially
///   unsized** `T`, where the layout is derived from a reference's metadata
///   (slice length, vtable, etc.). Use this when cloning an existing value
///   into fresh storage (e.g. `Rc::try_clone_from_ref_in`).
pub(super) struct UniqueRcUninit<T: ?Sized, A: Allocator> {
    ptr: NonNull<RcInner<T>>,
    alloc: A,
    layout: Layout,
}

impl<T: Sized, A: Allocator> UniqueRcUninit<T, A> {
    /// Allocates a new `RcInner<T>` block for a **sized** type, initializing
    /// the refcount headers to `(1, 1)`.
    ///
    /// The layout is derived directly from `RcInner<T>` (no external reference
    /// needed), making this suitable for constructing values from thin air.
    ///
    /// The caller must subsequently write the payload through
    /// [`data_ptr`](Self::data_ptr) and then call [`into_rc`](Self::into_rc)
    /// to obtain the final `Rc`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub(super) fn try_new(alloc: A) -> Result<Self, AllocError> {
        let layout = Layout::new::<RcInner<T>>();
        let block = alloc.allocate(layout)?;
        let ptr = block.cast::<RcInner<T>>();

        // Initialize the refcount headers. The allocation is fresh (uninit),
        // so we write through raw pointers rather than forming references to
        // uninitialized `Cell` values.
        unsafe {
            let inner = ptr.as_ptr();
            ptr::write(&raw mut (*inner).strong, Cell::new(1));
            ptr::write(&raw mut (*inner).weak, Cell::new(1));
        }

        Ok(Self { ptr, alloc, layout })
    }
}

impl<T: ?Sized, A: Allocator> UniqueRcUninit<T, A> {
    /// Allocates a new `RcInner<T>` block for a **potentially unsized** type,
    /// deriving the layout from `src`'s metadata (slice length, vtable, etc.),
    /// and initializes the refcount headers to `(1, 1)`.
    ///
    /// This is the path used when **cloning** an existing value into fresh
    /// storage: the source reference provides the metadata that determines the
    /// block size and the fat-pointer width.
    ///
    /// The caller must subsequently write the payload through
    /// [`data_ptr`](Self::data_ptr) (typically via
    /// [`TryCloneToUninit::try_clone_to_uninit`]) and then call
    /// [`into_rc`](Self::into_rc) to obtain the final `Rc`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails, or
    /// [`TryCloneError::Other`] if the combined layout overflows.
    pub(super) fn try_new_for_value(src: &T, alloc: A) -> Result<Self, TryCloneError> {
        let value_layout = Layout::for_value(src);
        let (layout, _) = helpers::rc_inner_layout_for_value_layout(value_layout)
            .map_err(|_| TryCloneError::Other("Rc payload layout overflow"))?;

        let block = alloc.allocate(layout)?;
        let base: *mut u8 = block.cast::<u8>().as_ptr();

        // Graft the source's metadata (slice length / vtable / etc...) onto the
        // destination pointer so the resulting fat pointer carries the correct
        // info for the payload.
        let to_copy_metadata = ptr::from_ref(src) as *const RcInner<T>;
        let inner_fat = unsafe { base.cast_with_metadata(to_copy_metadata) };
        let ptr = unsafe { NonNull::new_unchecked(inner_fat) };

        // Initialize the refcount headers. The allocation is fresh (uninit),
        // so we write through raw pointers rather than forming references to
        // uninitialized `Cell` values. The `MaybeUninit` round-trip in
        // `cast_with_metadata` ensures these writes are observable and cannot
        // be elided by the optimizer.
        unsafe {
            let inner = ptr.as_ptr();
            ptr::write(&raw mut (*inner).strong, Cell::new(1));
            ptr::write(&raw mut (*inner).weak, Cell::new(1));
        }

        Ok(Self { ptr, alloc, layout })
    }
}

impl<T: ?Sized, A: Allocator> UniqueRcUninit<T, A> {
    /// Returns a raw mutable pointer to the payload region within the block.
    ///
    /// The pointed-to memory is uninitialized; the caller must write
    /// the payload before calling [`into_rc`](Self::into_rc). For sized `T`
    /// the caller can cast to `*mut T` and write directly; for unsized types
    /// the caller should use [`TryCloneToUninit::try_clone_to_uninit`] which
    /// knows how to interpret the target.
    #[inline]
    pub(super) fn data_ptr(&self) -> *mut u8 {
        // Project the `value` field out of the fat pointer to get the exact
        // address of the payload slot.
        unsafe { &raw mut (*self.ptr.as_ptr()).value as *mut u8 }
    }

    /// Consumes the handle and produces the finalized `Rc<T, A>`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the payload region (pointed to by
    /// [`data_ptr`](Self::data_ptr)) has been fully initialized with a valid
    /// `T` before calling this method.
    #[inline]
    pub(super) unsafe fn into_rc(self) -> Rc<T, A> {
        let me = ManuallyDrop::new(self);
        // SAFETY: we are transferring ownership of the block to the returned
        // `Rc`. Reading the fields out and wrapping ManuallyDrop on `self`
        // prevents its `Drop` impl from double-freeing the allocation.
        let ptr = unsafe { ptr::read(&me.ptr) };
        let alloc = unsafe { ptr::read(&me.alloc) };
        Rc {
            ptr,
            alloc,
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Allocator> Drop for UniqueRcUninit<T, A> {
    fn drop(&mut self) {
        // If we reach here, `into_rc` was never called (early return, panic,
        // etc.). Free the block. The header was initialized to (1, 1) so the
        // allocation is well-formed from the allocator's perspective.
        //
        // SAFETY: `self.ptr` was allocated by `self.alloc` with `self.layout`.
        unsafe {
            self.alloc.deallocate(self.ptr.cast(), self.layout);
        }
    }
}

// ---------------------------------------------------------------------------
// Uninit initialization helper
// ---------------------------------------------------------------------------

impl<T: Sized, A: Allocator> Rc<MaybeUninit<T>, A> {
    /// Writes `val` into the `Rc`'s slot and return the initialized `Rc`.
    ///
    /// # Safety
    ///
    /// The caller must ensure the `Rc`'s strong count is exactly 1 and
    /// weak count is 0 (excluding the implicit weak reference).
    #[inline]
    pub unsafe fn write(mut self, val: T) -> Rc<T, A> {
        // SAFETY: `inner` is a live allocation, caller ensures strong == 1 and weak == 0
        // (excluding implicit weak ref).
        let inner = unsafe { Rc::inner_mut(&mut self) };
        inner.value.write(val);
        // SAFETY: we just initialized the pointer.
        unsafe { self.assume_init() }
    }

    /// Reinterprets the `Rc<MaybeUninit<T>, A>` as an initialized `Rc<T, A>`.
    ///
    /// # Safety
    ///
    /// The payload slot must have been fully initialized.
    /// Calling this on uninitialized memory is UB.
    #[inline]
    pub unsafe fn assume_init(self) -> Rc<T, A> {
        let (ptr, alloc) = Rc::into_raw_with_allocator(self);
        // SAFETY: `Rc<MaybeUninit<T>, A>` and `Rc<T, A>` have identical layouts
        // (both contain `NonNull<RcInner<...>>`, `A`, and a zero-sized phantom);
        // the caller guarantees the payload slot is fully initialized. The casted Rc
        // uses the moved allocator.
        unsafe { Rc::from_raw_in(ptr as *const T, alloc) }
    }
}
