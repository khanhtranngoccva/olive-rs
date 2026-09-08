//! Memory allocation traits and errors, mirroring the standard library's allocator surface.
//!
//! This module gives every fallible data structure in the Olive stack a
//! canonical, trait-based way to allocate memory:
//!
//! * [`Layout`] — re-exported verbatim from the stable
//!   [`core::alloc::Layout`]. We borrow the API rather than reimplementing it.
//! * [`AllocError`] — the error type for failed allocations via [`Allocator`].
//!   Defined here because `core::alloc::AllocError` is still unstable
//!   (`allocator_api`); ours has the identical unit-struct shape.
//! * [`Allocator`] — the modern, `Result`-based allocator trait. The
//!   **canonical** interface: every Olive collection allocates through it.
//!
//! # How allocation reaches the global allocator
//!
//! This crate is `#![no_std]` but links the standard `alloc` crate (the same
//! arrangement the reference implementation uses). The free functions above
//! delegate to the **exposed** `alloc::alloc::{alloc, dealloc, realloc,
//! alloc_zeroed}` functions. Those internally dispatch through whatever
//! `#[global_allocator]` the final binary installs, so a custom global
//! allocator is honored transparently — no fragile `__rust_alloc` magic-symbol
//! plumbing required.
//!
//! We deliberately do *not* reimplement a heap or re-declare the compiler's
//! allocator symbols. Borrowing std's exposed allocator entry points means our
//! allocations participate in the same accounting, debug hooks, and OOM behavior
//! as the rest of the program, while still giving us a single, swappable seam.
//!
//! # Canonical implementation rule
//!
//! Per the project's discipline, every fallible container in `olive_alloc` must
//! perform its memory operations through the [`Allocator`] trait (with [`Global`]
//! as the default), rather than calling the free functions or `std::alloc`
//! directly. That keeps a single seam for swapping in custom allocators and for
//! simulating OOM in tests.

extern crate alloc;

use crate::mem::MaybeUninitUnsized;
use crate::try_traits::try_clone::TryClone;
use core::ptr::NonNull;

// Borrow the stable layout API from `core` instead of reimplementing it.
pub use crate::alloc_errors::AllocError;
pub use core::alloc::{Layout, LayoutError};

// ──────────────────────────────────────────────────────────────────────────────
// Allocator
// ──────────────────────────────────────────────────────────────────────────────

/// An implementation of `Allocator` can allocate, grow, shrink, and deallocate
/// arbitrary blocks of data described via [`Layout`].
///
/// `Allocator` is designed to be implemented on ZSTs, references, or smart
/// pointers, because having an allocator like `MyAlloc([u8; N])` cannot be moved
/// without updating the pointers to the allocated memory.
///
/// Unlike [`core::alloc::GlobalAlloc`], zero-sized allocations are allowed in
/// `Allocator`. If an underlying allocator does not support this (like jemalloc)
/// or returns a null pointer (such as `libc::malloc`), this must be caught by
/// the implementation.
///
/// ### Currently allocated memory
///
/// Some methods require that a memory block be *currently allocated* via an
/// allocator. This means that:
///
/// * the starting address was previously returned by [`allocate`][Allocator::allocate],
///   [`grow`][Allocator::grow], or [`shrink`][Allocator::shrink], and
/// * the memory block has not been subsequently deallocated, either directly via
///   [`deallocate`][Allocator::deallocate] or by being passed to a successful
///   `grow`/`shrink`. If `grow`/`shrink` returned `Err`, the passed pointer
///   remains valid.
///
/// ### Memory fitting
///
/// Some methods require that a layout *fit* a memory block: the block must be
/// allocated with the same alignment as [`layout.align()`][Layout::align], and
/// the provided [`layout.size()`][Layout::size] must fall in the range
/// `min ..= max`, where `min` is the size of the layout most recently used to
/// allocate the block, and `max` is the latest actual size returned from
/// `allocate`/`grow`/`shrink`.
///
/// # Safety
///
/// * Memory blocks returned from an allocator that are *currently allocated*
///   must point to valid memory and retain their validity while they are
///   currently allocated and the shorter of:
///   - the borrow-checker lifetime of the allocator type itself,
///   - as long as at least one of the instance and all of its clones has not
///     been dropped.
/// * Copying, cloning, or moving the allocator must not invalidate memory blocks
///   returned from this allocator. A copied/cloned allocator must behave like the
///   same allocator, and
/// * Any pointer to a memory block which is *currently allocated* may be passed
///   to any other method of the allocator.
pub unsafe trait Allocator {
    /// Attempts to allocate a block of memory.
    ///
    /// On success, returns a [`NonNull<[u8]>`](NonNull) meeting the size and alignment
    /// guarantees of `layout`. The returned block may have a larger size than
    /// specified by `layout.size()`, and may or may not have its contents
    /// initialized.
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or `layout`
    /// does not meet the allocator's size or alignment constraints.
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory *currently allocated* via this
    ///   allocator, and
    /// * `layout` must *meet the requirements* described in [Memory fitting](alloc::Allocator#memory-fitting).
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Behaves like `allocate`, but also ensures that the returned memory is
    /// zero-initialized.
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or `layout`
    /// does not meet the allocator's size or alignment constraints.
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        // SAFETY: `allocate` returns a valid memory block.
        unsafe { base_ptr(ptr).as_ptr().write_bytes(0, ptr.len()) }
        Ok(ptr)
    }

    /// Attempts to extend the memory block.
    ///
    /// Returns a new [`NonNull<[u8]>`][NonNull] containing a pointer and the actual size of the allocated
    /// memory. The pointer is suitable for holding data described by `new_layout`. To accomplish
    /// this, the allocator may extend the allocation referenced by `ptr` to fit the new layout.
    ///
    /// If this returns `Ok`, then ownership of the memory block referenced by `ptr` has been
    /// transferred to this allocator. Any access to the old `ptr` is Undefined Behavior, even if the
    /// allocation was grown in-place. The newly returned pointer is the only valid pointer
    /// for accessing this memory now.
    ///
    /// If this method returns `Err`, then ownership of the memory block has not been transferred to
    /// this allocator, and the contents of the memory block are unaltered.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    /// * `old_layout` must [*fit*] that block of memory (The `new_layout` argument need not fit it.).
    /// * `new_layout.size()` must be greater than or equal to `old_layout.size()`.
    ///
    /// Note that `new_layout.align()` need not be the same as `old_layout.align()`.
    ///
    /// [*currently allocated*]: alloc::alloc#currently-allocated-memory
    /// [*fit*]: #memory-fitting
    ///
    /// # Errors
    ///
    /// Returns `Err` if the new layout does not meet the allocator's size and alignment
    /// constraints of the allocator, or if growing otherwise fails.
    ///
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than panicking or
    /// aborting, but this is not a strict requirement. (Specifically: it is *legal* to implement
    /// this trait atop an underlying native allocation library that aborts on memory exhaustion.)
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;

        // SAFETY: because `new_layout.size()` must be greater than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid for reads and
        // writes for `old_layout.size()` bytes. Also, because the old allocation wasn't yet
        // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                base_ptr(new_ptr).as_ptr(),
                old_layout.size(),
            );
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    /// Behaves like `grow`, but also ensures that the new contents are set to zero before being
    /// returned.
    ///
    /// The memory block will contain the following contents after a successful call to
    /// `grow_zeroed`:
    ///   * Bytes `0..old_layout.size()` are preserved from the original allocation.
    ///   * Bytes `old_layout.size()..old_size` will either be preserved or zeroed, depending on
    ///     the allocator implementation. `old_size` refers to the size of the memory block prior
    ///     to the `grow_zeroed` call, which may be larger than the size that was originally
    ///     requested when it was allocated.
    ///   * Bytes `old_size..new_size` are zeroed. `new_size` refers to the size of the memory
    ///     block returned by the `grow_zeroed` call.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    /// * `old_layout` must [*fit*] that block of memory (The `new_layout` argument need not fit it.).
    /// * `new_layout.size()` must be greater than or equal to `old_layout.size()`.
    ///
    /// Note that `new_layout.align()` need not be the same as `old_layout.align()`.
    ///
    /// [*currently allocated*]: #currently-allocated-memory
    /// [*fit*]: #memory-fitting
    ///
    /// # Errors
    ///
    /// Returns `Err` if the new layout does not meet the allocator's size and alignment
    /// constraints of the allocator, or if growing otherwise fails.
    ///
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than panicking or
    /// aborting, but this is not a strict requirement. (Specifically: it is *legal* to implement
    /// this trait atop an underlying native allocation library that aborts on memory exhaustion.)
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate_zeroed(new_layout)?;

        // SAFETY: because `new_layout.size()` must be greater than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid for reads and
        // writes for `old_layout.size()` bytes. Also, because the old allocation wasn't yet
        // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                base_ptr(new_ptr).as_ptr(),
                old_layout.size(),
            );
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    /// Attempts to shrink the memory block.
    ///
    /// Returns a new [`NonNull<[u8]>`][NonNull] containing a pointer and the actual size of the allocated
    /// memory. The pointer is suitable for holding data described by `new_layout`. To accomplish
    /// this, the allocator may shrink the allocation referenced by `ptr` to fit the new layout.
    ///
    /// If this returns `Ok`, then ownership of the memory block referenced by `ptr` has been
    /// transferred to this allocator. Any access to the old `ptr` is Undefined Behavior, even if the
    /// allocation was shrunk in-place. The newly returned pointer is the only valid pointer
    /// for accessing this memory now.
    ///
    /// If this method returns `Err`, then ownership of the memory block has not been transferred to
    /// this allocator, and the contents of the memory block are unaltered.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    /// * `old_layout` must [*fit*] that block of memory (The `new_layout` argument need not fit it.).
    /// * `new_layout.size()` must be smaller than or equal to `old_layout.size()`.
    ///
    /// Note that `new_layout.align()` need not be the same as `old_layout.align()`.
    ///
    /// [*currently allocated*]: #currently-allocated-memory
    /// [*fit*]: #memory-fitting
    ///
    /// # Errors
    ///
    /// Returns `Err` if the new layout does not meet the allocator's size and alignment
    /// constraints of the allocator, or if shrinking otherwise fails.
    ///
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than panicking or
    /// aborting, but this is not a strict requirement. (Specifically: it is *legal* to implement
    /// this trait atop an underlying native allocation library that aborts on memory exhaustion.)
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;

        // SAFETY: because `new_layout.size()` must be lower than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid for reads and
        // writes for `new_layout.size()` bytes. Also, because the old allocation wasn't yet
        // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                base_ptr(new_ptr).as_ptr(),
                new_layout.size(),
            );
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    /// Returns a reference to `self` usable as an `Allocator`.
    ///
    /// Provided with a default implementation so that both `A` and `&A` can be
    /// used interchangeably as allocators.
    fn by_ref(&self) -> &Self {
        self
    }
}

/// Blanket impl so that a reference to any allocator is itself an allocator.
unsafe impl<A: Allocator + ?Sized> Allocator for &A {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forward the caller's guarantees to the inner allocator.
        unsafe { (**self).deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forward the caller's guarantees to the inner allocator.
        unsafe { (**self).grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forward the caller's guarantees to the inner allocator.
        unsafe { (**self).grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forward the caller's guarantees to the inner allocator.
        unsafe { (**self).shrink(ptr, old_layout, new_layout) }
    }

    #[inline]
    fn by_ref(&self) -> &Self {
        self
    }
}

/// Blanket impl so that a mutable reference to any allocator is itself an allocator.
unsafe impl<A: Allocator + ?Sized> Allocator for &mut A {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forward the caller's guarantees to the inner allocator.
        unsafe { (**self).deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forward the caller's guarantees to the inner allocator.
        unsafe { (**self).grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forward the caller's guarantees to the inner allocator.
        unsafe { (**self).grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forward the caller's guarantees to the inner allocator.
        unsafe { (**self).shrink(ptr, old_layout, new_layout) }
    }

    #[inline]
    fn by_ref(&self) -> &Self {
        self
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// StaticAllocator
// ──────────────────────────────────────────────────────────────────────────────

/// Marks that an allocator and its supertypes will never invalidate currently allocated
/// memory unless explicitly deallocated via a call to a deallocating method, even if
/// dropped or if the allocator's lifetime expires.
///
/// This is a necessity in conjunction with [`Pin`](core::pin::Pin), as only allocators that
/// promise memory is never reused without a destructor running may be used to back a pinned
/// pointer.
///
/// # Safety
///
/// Implementors must ensure that memory cannot be freed except via a call to
/// [`Allocator::deallocate`], and that subtype coercion preserves this invariant.
///
/// These requirements trivially apply to allocators that always maintain global state, such
/// as the standard library's `System` or `Global`. However, due to subtype coercion, it is
/// *not* sound to implement for an arbitrary `Allocator + 'static` due to
/// [edge-case interactions](https://github.com/rust-lang/rust/issues/157089) with `Pin::clone`.
/// Namely an impl of `StaticAllocator for MyAllocator + 'long` would guarantee that an impl of
/// `StaticAllocator for MyAllocator + 'short` is sound to write.
///
/// The following must thus be guaranteed:
/// - the `Drop` impl of the allocator does not invalidate any allocations;
/// - the allocator does not expose a safe API surface that allows invalidating
///   its allocations;
/// - the allocator's lifetime expiring does not invalidate any allocations;
/// - the above also hold for all equivalent allocators (see [`Allocator`] docs).
pub unsafe trait StaticAllocator: Allocator {}

// If an allocator is `StaticAllocator` all equivalent allocators must also uphold
// its semantics, and references are equivalent to the allocator they reference.
unsafe impl<A: StaticAllocator + ?Sized> StaticAllocator for &A {}

// ──────────────────────────────────────────────────────────────────────────────
// AllocatorTryClone
// ──────────────────────────────────────────────────────────────────────────────

/// Marks a type's [`TryClone`] implementation as sound with regard to
/// [`Allocator`] equivalence.
///
/// This is Olive's fallible analogue of std's unstable
/// [`AllocatorClone`](https://doc.rust-lang.org/beta/std/alloc/trait.AllocatorClone.html),
/// which is a marker over `Clone`. Because Olive routes every fallible op through
/// [`TryClone`], the marker here sits on `TryClone` instead: cloning an allocator
/// may itself allocate (e.g. an arena that pools blocks on the heap), so the
/// operation is fallible rather than infallible.
///
/// Implementors must ensure that, upon calling [`TryClone::try_clone`], the two
/// resulting handles are *equivalent*: memory allocated through one may be freed
/// through the other. Concretely, a `Box<T, A>` cloned via
/// [`TryClone`] relies on this guarantee — the clone must land on the *same*
/// backing store as the original, not on a freshly minted independent allocator.
/// Further, mutable accesses such as moving or dropping the allocator must not
/// invalidate its currently allocated blocks at least so long as clones exist.
///
/// Additionally, the bound that allocators do not unwind when (de)allocating
/// applies here too: cloning an allocator must not unwind either.
///
/// It must also be the case that types which are `AllocatorTryClone` are either
/// explicitly not copyable (such as by containing a `!Copy` field) or that
/// copying them also respects allocator equivalence as if it had been a clone.
///
/// # Safety
///
/// Implementors must uphold the equivalence and non-invalidation guarantees
/// described above for their [`TryClone::try_clone`] implementation.
pub unsafe trait AllocatorTryClone: Allocator + TryClone {}

unsafe impl<A: Allocator + ?Sized> AllocatorTryClone for &A {}

/// Extracts the base `NonNull<u8>` from a fat `NonNull<[u8]>`.
///
/// `NonNull::<T>::as_non_null_ptr` is not yet stable, so we recover the base
/// pointer through a raw-pointer cast.
const fn base_ptr(slice: NonNull<[u8]>) -> NonNull<u8> {
    slice.cast()
}

/// Extension methods for [`Layout`] to provide compatibility.
///
/// Because Rust traits cannot currently declare `const fn` members, the
/// const-evaluable variants are provided as free functions below
/// ([`layout_dangling_pointer`] and [`layout_padding_need_for`]). They accept
/// a `Layout` by value and can be called from `const` contexts.
pub trait LayoutExt {
    /// Creates a [`NonNull`] that is dangling, but well-aligned for this Layout.
    /// Note that the address of the returned pointer may potentially be that of a valid pointer,
    /// which means this must not be used as a "not yet initialized" sentinel value.
    ///
    /// Types that lazily allocate must track initialization by some other means.
    ///
    /// This is the MSRV-compatible equivalent of `Layout::dangling_ptr`.
    fn dangling_pointer(&self) -> NonNull<u8>;

    /// The number of padding bytes required after a block laid out with `self`
    /// before a field requiring alignment `align` may begin at an offset that is
    /// a multiple of `align`.
    ///
    /// This is the stable shim for [`Layout::padding_needed_for`].
    ///
    /// # Panics
    ///
    /// Panics if `align` is zero or not a power of two — i.e. not a valid
    /// alignment — mirroring the precondition every `Layout` constructor
    /// enforces.
    fn padding_need_for(&self, align: usize) -> usize;

    /// Computes the layout of the value pointed to by `ptr`, without requiring
    /// the pointee to be initialized.
    ///
    /// This is the stable but limited shim for [`Layout::for_value_raw`]
    /// (stabilized in 1.99). On stable, it delegates to [`Layout::for_value`]
    /// via an [`MaybeUninitUnsized`] reference formed from the pointer.
    ///
    /// This is meant to be [`Layout::for_value`] that is semantically sound for
    /// uninitialized pointers. For example, one can use this function to determine
    /// the layout of the memory block to deallocate. Due to safety limitations, it
    /// is not suitable for all use cases of [`Layout::for_value_raw`].
    ///
    /// # Safety
    ///
    /// - The pointer must be properly aligned, non-null and carry correct metadata
    ///   for the type `T` (slice length, vtable, etc.). If `T` is not a ZST, it must
    ///   also point to valid memory.
    /// - It does **not** need to point to initialized memory.
    /// - The memory must not be mutated during the call.
    /// - It must satisfy other requirements of [`Layout::for_value_raw`].
    unsafe fn for_value_pointer<T: ?Sized>(ptr: *const T) -> Layout;
}

impl LayoutExt for Layout {
    fn dangling_pointer(&self) -> NonNull<u8> {
        layout_dangling_pointer(*self)
    }

    fn padding_need_for(&self, align: usize) -> usize {
        layout_padding_need_for(*self, align)
    }

    #[inline]
    unsafe fn for_value_pointer<T: ?Sized>(ptr: *const T) -> Layout {
        #[cfg(unstable_features)]
        {
            // SAFETY: precondition from the caller.
            unsafe { Layout::for_value_raw(ptr) }
        }
        #[cfg(not(unstable_features))]
        {
            // SAFETY: precondition from the caller.
            unsafe { Layout::for_value(MaybeUninitUnsized::from_ptr(ptr)) }
        }
    }
}

/// Const-evaluable variant of [`LayoutExt::dangling_pointer`].
///
/// Creates a [`NonNull`] that is dangling, but well-aligned for `layout`.
/// Usable in `const` contexts where the trait method cannot be called.
#[must_use]
#[inline]
pub const fn layout_dangling_pointer(layout: Layout) -> NonNull<u8> {
    unsafe { NonNull::new_unchecked(core::ptr::without_provenance_mut(layout.align())) }
}

/// Const-evaluable variant of [`LayoutExt::padding_need_for`].
///
/// Returns the number of padding bytes required after a block laid out with
/// `layout` before a field requiring alignment `align` may begin at an offset
/// that is a multiple of `align`.
///
/// # Panics
///
/// Panics if `align` is zero or not a power of two.
#[must_use]
#[inline]
pub const fn layout_padding_need_for(layout: Layout, align: usize) -> usize {
    assert!(align.is_power_of_two(), "alignment is not a power of two");
    // layout.size % align, but since align is a power of 2,
    // binary AND with align - 1 is equivalent (strips non-modulo bytes).
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "alignment is positive due to being a power of two"
    )]
    let rem = layout.size() & (align - 1);
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "asserted rem < align (remainder"
    )]
    if rem == 0 { 0 } else { align - rem }
}

#[cfg(test)]
mod tests {
    //! Tests exercise the `Allocator` trait through `Global`, exercising the
    //! delegation to the standard `alloc` crate's exposed functions end-to-end.
    extern crate std;
    use super::*;

    #[test]
    fn layout_borrowed_from_core() {
        // Sanity: the borrowed core Layout behaves identically to std's.
        let l = Layout::array::<u128>(2).expect("valid");
        assert_eq!(l.size(), 32);
        assert_eq!(l.align(), 16);
        // `extend` returns `(Layout, offset)`.
        let (combined, offset) = l.extend(Layout::new::<u8>()).expect("extend ok");
        assert_eq!(combined.size(), 33);
        assert_eq!(offset, 32);
    }

    #[test]
    fn vec_still_works_through_global_path() {
        // Proves that delegating to the standard `alloc` crate's exposed
        // functions coexists cleanly with std's own allocations in the same
        // process (same global allocator, no double-linking surprises).
        let v: std::vec::Vec<i32> = (0..1000).collect();
        assert_eq!(v.iter().sum::<i32>(), (0..1000).sum::<i32>());
    }

    #[test]
    fn padding_need_for_rounds_up_to_alignment() {
        // A 16-byte header followed by an 8-aligned field needs no gap.
        let l16 = Layout::from_size_align(16, 16).expect("valid");
        assert_eq!(l16.padding_need_for(8), 0);
        // A 24-byte header followed by a 16-aligned field needs 8 bytes of pad.
        let l24 = Layout::from_size_align(24, 8).expect("valid");
        assert_eq!(l24.padding_need_for(16), 8);
        // Alignment smaller than or equal to the size remainder still rounds up.
        let l5 = Layout::from_size_align(5, 1).expect("valid");
        assert_eq!(l5.padding_need_for(8), 3);
    }

    #[test]
    fn padding_need_for_agrees_with_extend_offset() {
        // The helper must reproduce exactly the offset `Layout::extend` reports
        // for placing the second layout after the first — that is the invariant
        // `Rc::data_offset` relies on.
        let header = Layout::new::<[usize; 2]>();
        for align in [1usize, 2, 4, 8, 16, 32] {
            let value = Layout::from_size_align(7, align).expect("valid");
            let (_, offset) = header.extend(value).expect("extend ok");
            let expected = header.size() + header.padding_need_for(align);
            assert_eq!(offset, expected, "mismatch at alignment {align}");
        }
    }

    #[test]
    #[should_panic(expected = "not a power of two")]
    fn padding_need_for_rejects_invalid_alignment() {
        let l = Layout::new::<u8>();
        let _ = l.padding_need_for(3);
    }
}
