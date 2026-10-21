//! A fully-fallible port of the standard library's `Box`.
//!
//! Compared with the std original, three things differ:
//!
//! * Every constructor that allocates has a fallible `try_*` counterpart that
//!   returns [`Result`] carrying [`AllocError`] instead of aborting or panicking
//!   on OOM.
//! * All allocation goes through the [`Allocator`] trait rather than free
//!   functions, giving a single swappable seam for custom allocators and OOM
//!   simulation in tests.
//! * Compiler-internal attributes (`#[lang = "owned_box"]`) and unstable
//!   helpers (`into_unique`, `into_non_null_with_allocator`, `may_dangle`) are
//!   omitted; this is a pure type with no compiler magic.

use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr::{self, NonNull};

use crate::alloc::{Allocator, Global, Layout};

// ---------------------------------------------------------------------------
// Submodules
// ---------------------------------------------------------------------------

mod construction;
mod convert;
mod dst;
mod query;
mod reconstitution;
mod traits;
mod uninit;

// ---------------------------------------------------------------------------
// Box declaration
// ---------------------------------------------------------------------------

/// A pointer to heap-allocated data.
///
/// `Box<T>` points to data allocated on the heap via the [`Allocator`] trait.
/// It comes in two flavors:
///
/// * **Sized**: `Box<T>` where `T: Sized`. Owns exactly one `T`.
/// * **Unsized**: `Box<[T]>`, `Box<str>`, `Box<dyn Trait>`, etc. Points to an
///   arbitrary-length allocation described by a fat pointer.
///
/// Unlike `std::boxed::Box`, every allocating constructor here has a fallible
/// `try_*` variant returning `Result<Self, AllocError>`.
pub struct Box<T: ?Sized, A: Allocator = Global> {
    inner: NonNull<T>,
    alloc: A,
}

// ---------------------------------------------------------------------------
// Drop
// ---------------------------------------------------------------------------

/// Owns the raw pointer, layout, and allocator of a [`Box`] being dropped, and
/// frees them in its own `Drop`.
///
/// Dropping the contained value [ptr::drop_in_place](`ptr::drop_in_place`) can panic.
/// Without this guard, a panic would unwind past the deallocation line and leak the block.
/// Arming the guard *before* dropping the pointee guarantees the free runs both
/// on the happy path (guard falls out of scope normally) and on unwind (guard's
/// `Drop` runs during stack teardown). There is exactly one free site.
struct BoxDeallocGuard<'a, T: ?Sized, A: Allocator> {
    ptr: NonNull<u8>,
    layout: Option<Layout>,
    alloc: &'a A,
    _marker: PhantomData<T>,
}

impl<T: ?Sized, A: Allocator> Drop for BoxDeallocGuard<'_, T, A> {
    fn drop(&mut self) {
        // A `None` layout means the pointee was a ZST and nothing was
        // allocated, so there is nothing to free.
        if let Some(layout) = self.layout {
            // SAFETY: `ptr` denotes a live block previously allocated via
            // `*self.alloc` with `layout`; the pointee has already been
            // destroyed by the time this guard drops.
            unsafe { self.alloc.deallocate(self.ptr, layout) };
        }
    }
}

impl<T: ?Sized, A: Allocator> Drop for Box<T, A> {
    #[inline]
    fn drop(&mut self) {
        // Compute the layout from the fat pointer before dropping the value;
        // `Layout::for_value` needs the dynamic size (and vtable for trait
        // objects), which is carried by `self.inner`.
        let layout = {
            let val: &T = self.deref();
            Layout::for_value(val)
        };
        // Arm the deallocation guard *before* dropping the pointee. The guard
        // will free the block whether or not `drop_in_place` panics. ZSTs have
        // no backing allocation, hence `None`.
        let _dealloc_guard = BoxDeallocGuard {
            ptr: self.inner.cast(),
            layout: (layout.size() != 0).then_some(layout),
            alloc: &self.alloc,
            _marker: PhantomData::<T>,
        };
        // Drop the contained value in place. For a ZST (including a ZST trait
        // object) this is a no-op. Whether this succeeds or panics,
        // `_dealloc_guard` goes out of scope next and performs the single
        // deallocation.
        unsafe {
            ptr::drop_in_place(self.inner.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests;
