//! The global allocator, built on [`allocator-api2`](https://docs.rs/allocator-api2).
//!
//! Rather than handrolling the default allocator's memory-management logic, Olive
//! delegates it to [`allocator_api2::alloc::Global`] and exposes its own [`Global`]
//! allocator.
//!
//! Because Rust's orphan rule forbids implementing Olive's own traits
//! ([`StaticAllocator`], [`AllocatorTryClone`], [`AllocatorTryDefault`]) on a foreign
//! type, [`Global`] is a thin local ZST that forwards every `Allocator` method straight
//! to [`allocator_api2::alloc::Global`] while carrying the Olive-specific trait impls.
//! Downstream code sees exactly one `Global`, as before.
pub use core::alloc::{Layout, LayoutError};
pub use core::ptr::NonNull;
pub use olive_core::alloc::AllocError;
pub use olive_core::alloc::Allocator;
pub use olive_core::alloc::AllocatorTryClone;
pub use olive_core::alloc::AllocatorTryDefault;
pub use olive_core::alloc::StaticAllocator;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

/// The default allocator.
///
/// Forwards every operation to [`allocator_api2::alloc::Global`], which in turn
/// calls the exposed `alloc::alloc` free functions. Those dispatch through the
/// installed `#[global_allocator]`, so a custom global allocator is honored
/// transparently.
#[derive(Clone, Copy, Default)]
pub struct Global;

unsafe impl Allocator for Global {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        allocator_api2::alloc::Global.allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        allocator_api2::alloc::Global.allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { allocator_api2::alloc::Global.deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { allocator_api2::alloc::Global.grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { allocator_api2::alloc::Global.grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { allocator_api2::alloc::Global.shrink(ptr, old_layout, new_layout) }
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

// SAFETY: `Global` is a stateless ZST over the process-wide global allocator. Every
// value produced by `try_default()` is the identical unit value, hence two independently
// constructed handles are trivially equivalent — memory allocated through one is
// deallocatable through the other, and dropping one invalidates nothing. Singleton-equivalence holds.
unsafe impl AllocatorTryDefault for Global {}

#[cfg(test)]
mod test_allocators;

#[cfg(test)]
pub(crate) use test_allocators::CountingAllocator;
