//! The global allocator port, mirroring the standard library's allocator surface.
//!

extern crate alloc as stock_alloc;
pub use core::alloc::{Layout, LayoutError};
use core::ptr;
pub use core::ptr::NonNull;
pub use olive_core::alloc::AllocError;
pub use olive_core::alloc::Allocator;
pub use olive_core::alloc::AllocatorTryClone;
use olive_core::alloc::LayoutExt;
pub use olive_core::alloc::StaticAllocator;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

// ──────────────────────────────────────────────────────────────────────────────
// Global + free functions
// ──────────────────────────────────────────────────────────────────────────────

/// The default allocator.
///
/// `Global` forwards to the free-standing functions in this module, which call
/// the compiler's `__rust_alloc` family of magic symbols. Those symbols resolve
/// at link time to whatever `#[global_allocator]` the final binary installs, so
/// a custom global allocator is honored transparently.
#[derive(Clone, Copy, Default)]
pub struct Global;

impl Global {
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn alloc_impl_runtime(layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
        match layout.size() {
            0 => Ok(hydrate(layout.dangling_pointer(), 0)),
            // SAFETY: `layout` is non-zero in size,
            #[allow(
                unused_qualifications,
                reason = "we are the alloc module and need a discriminator"
            )]
            size => unsafe {
                let raw_ptr = if zeroed {
                    stock_alloc::alloc::alloc_zeroed(layout)
                } else {
                    stock_alloc::alloc::alloc(layout)
                };
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                Ok(hydrate(ptr, size))
            },
        }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn deallocate_impl_runtime(ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY:
            // * We have checked that `layout` is non-zero in size.
            // * The caller is obligated to provide a layout that "fits", and in this case,
            //   "fit" always means a layout that is equal to the original, because our
            //   `allocate()`, `grow()`, and `shrink()` implementations never returns a larger
            //   allocation than requested.
            // * Other conditions must be upheld by the caller, as per `Allocator::deallocate()`'s
            //   safety documentation.
            #[allow(
                unused_qualifications,
                reason = "we are the alloc module and need a discriminator"
            )]
            unsafe {
                stock_alloc::alloc::dealloc(ptr.as_ptr(), layout)
            }
        }
    }

    // SAFETY: Same as `Allocator::grow`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn grow_impl_runtime(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        zeroed: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        match old_layout.size() {
            0 => self.alloc_impl(new_layout, zeroed),

            // SAFETY: `new_size` is non-zero as `old_size` is greater than or equal to `new_size`
            // as required by safety conditions. Other conditions must be upheld by the caller
            old_size if old_layout.align() == new_layout.align() => unsafe {
                let new_size = new_layout.size();

                // `realloc` probably checks for `new_size >= old_layout.size()` or something similar.
                core::hint::assert_unchecked(new_size >= old_layout.size());

                #[allow(
                    unused_qualifications,
                    reason = "we are the alloc module and need a discriminator"
                )]
                let raw_ptr = stock_alloc::alloc::realloc(ptr.as_ptr(), old_layout, new_size);
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                // SAFETY: new_size >= old_size
                if zeroed {
                    raw_ptr
                        .add(old_size)
                        .write_bytes(0, new_size.wrapping_sub(old_size));
                }
                Ok(hydrate(ptr, new_size))
            },

            // SAFETY: because `new_layout.size()` must be greater than or equal to `old_size`,
            // both the old and new memory allocation are valid for reads and writes for `old_size`
            // bytes. Also, because the old allocation wasn't yet deallocated, it cannot overlap
            // `new_ptr`. Thus, the call to `copy_nonoverlapping` is safe. The safety contract
            // for `dealloc` must be upheld by the caller.
            old_size => unsafe {
                let new_ptr = self.alloc_impl(new_layout, zeroed)?;
                ptr::copy_nonoverlapping(ptr.as_ptr(), base_ptr(new_ptr).as_ptr(), old_size);
                self.deallocate(ptr, old_layout);
                Ok(new_ptr)
            },
        }
    }

    // SAFETY: Same as `Allocator::grow`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn shrink_impl_runtime(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        _zeroed: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        match new_layout.size() {
            // SAFETY: conditions must be upheld by the caller
            0 => unsafe {
                self.deallocate(ptr, old_layout);
                Ok(hydrate(new_layout.dangling_pointer(), 0))
            },

            // SAFETY: `new_size` is non-zero. Other conditions must be upheld by the caller
            new_size if old_layout.align() == new_layout.align() => unsafe {
                // `realloc` probably checks for `new_size <= old_layout.size()` or something similar.
                core::hint::assert_unchecked(new_size <= old_layout.size());
                #[allow(
                    unused_qualifications,
                    reason = "we are the alloc module and need a discriminator"
                )]
                let raw_ptr = stock_alloc::alloc::realloc(ptr.as_ptr(), old_layout, new_size);
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                Ok(hydrate(ptr, new_size))
            },

            // SAFETY: because `new_size` must be smaller than or equal to `old_layout.size()`,
            // both the old and new memory allocation are valid for reads and writes for `new_size`
            // bytes. Also, because the old allocation wasn't yet deallocated, it cannot overlap
            // `new_ptr`. Thus, the call to `copy_nonoverlapping` is safe. The safety contract
            // for `dealloc` must be upheld by the caller.
            new_size => unsafe {
                let new_ptr = self.allocate(new_layout)?;
                ptr::copy_nonoverlapping(ptr.as_ptr(), base_ptr(new_ptr).as_ptr(), new_size);
                self.deallocate(ptr, old_layout);
                Ok(new_ptr)
            },
        }
    }

    // Methods below are meant to be dispatches to const or runtime, which is unstable.

    // SAFETY: Same as `Allocator::allocate`.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn alloc_impl(&self, layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
        Global::alloc_impl_runtime(layout, zeroed)
    }

    // SAFETY: Same as `Allocator::deallocate`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn deallocate_impl(&self, ptr: NonNull<u8>, layout: Layout) {
        Global::deallocate_impl_runtime(ptr, layout)
    }

    // SAFETY: Same as `Allocator::grow`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn grow_impl(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        zeroed: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.grow_impl_runtime(ptr, old_layout, new_layout, zeroed)
    }

    // SAFETY: Same as `Allocator::shrink`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn shrink_impl(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.shrink_impl_runtime(ptr, old_layout, new_layout, false)
    }
}

unsafe impl Allocator for Global {
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_impl(layout, false)
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_impl(layout, true)
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.deallocate_impl(ptr, layout) }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.grow_impl(ptr, old_layout, new_layout, false) }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.grow_impl(ptr, old_layout, new_layout, true) }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.shrink_impl(ptr, old_layout, new_layout) }
    }
}

// SAFETY: `Global` is a ZST that maintains no per-instance state and forwards every
// operation to the process-wide global allocator. Dropping it (a no-op) or letting its
// lifetime expire cannot invalidate any allocation; the only way memory is reclaimed is an
// explicit `deallocate`. This matches std's own `unsafe impl StaticAllocator for Global`.
unsafe impl StaticAllocator for Global {}

// `Global` is a stateless ZST: cloning it cannot fail, so its `TryClone` is
// infallible. This lets allocator-generic fallible ops (e.g. `Box::try_clone`)
// clone the backing allocator through the same `TryClone` seam as any other
// value, without assuming every allocator is `Copy`.
impl TryClone for Global {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

// SAFETY: `Global` is a stateless ZST forwarding every operation to the
// process-wide global allocator. Cloning it yields another handle to the very
// same backing store, so memory allocated through one handle is freely
// deallocatable through the other; moving or dropping a clone invalidates
// nothing. Equivalence therefore holds trivially.
unsafe impl AllocatorTryClone for Global {}

// `Global` carries no per-instance state and reserves nothing up front, so its
// default construction cannot fail.
impl TryDefault for Global {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Global)
    }
}

/// Extracts the base `NonNull<u8>` from a fat `NonNull<[u8]>`.
///
/// `NonNull::<[T]>::as_non_null_ptr` is not yet stable, so we recover the base
/// pointer through a raw-pointer cast.
const fn base_ptr(slice: NonNull<[u8]>) -> NonNull<u8> {
    slice.cast()
}

/// Hydrate a fat `NonNull<[u8]>` from a base `NonNull<u8>` and a slice length.
///
/// `NonNull::<T>::cast_slice` is not yet stable.
const fn hydrate(raw: NonNull<u8>, len: usize) -> NonNull<[u8]> {
    let slice = ptr::slice_from_raw_parts_mut(raw.as_ptr(), len);
    // SAFETY: raw is NonNull.
    unsafe { NonNull::new_unchecked(slice) }
}

/// Re-exports of the stock alloc crate.
pub use stock_alloc::alloc::{alloc, alloc_zeroed, dealloc, realloc};

#[cfg(test)]
mod test_allocators;

#[cfg(test)]
pub(crate) use test_allocators::CountingAllocator;
