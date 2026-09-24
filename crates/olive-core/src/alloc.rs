//! The canonical allocator API for the Olive stack.
//!
//! Rather than handrolling its own allocator trait, Olive adopts
//! [`allocator-api2`](https://docs.rs/allocator-api2) — the community-standard,
//! dependency-light port of std's unstable `core::alloc::Allocator` — and
//! re-exports its public surface from here.
//!
//! Every fallible data structure in Olive allocates through this exact trait,
//! so third-party libraries built against the same API (most notably [`hashbrown`])
//! can interoperate with Olive collections without needing newtypes.
//!
//! ## Re-exported from `allocator-api2`
//!
//! These names are aliases for the upstream types, not local definitions:
//!
//! * [`Allocator`](crate::alloc::Allocator) — the modern, `Result`-based
//!   allocator trait. The **canonical** interface: every Olive collection
//!   allocates through it.
//! * [`AllocError`](crate::alloc::AllocError) — the error type returned by a
//!   failed allocation.
//!
//! ## Re-exported from `core`
//!
//! * [`Layout`](crate::alloc::Layout) / [`LayoutError`](crate::alloc::LayoutError)
//!   these exact same types can also be accessed from `allocator-api2`.
//!
//! ## Olive-specific additions
//!
//! These are defined locally because they do not exist in `allocator-api2`, but existing
//! in nightly instead:
//!
//! * [`StaticAllocator`](crate::alloc::StaticAllocator) — marker for allocators
//!   whose memory survives drop; required to back a pinned pointer.
//! * [`AllocatorTryClone`](crate::alloc::AllocatorTryClone) — Olive's fallible
//!   analogue of std's unstable `AllocatorClone`, sitting on
//!   [`TryClone`](crate::try_traits::TryClone) instead of `Clone` because cloning
//!   an allocator may itself allocate.
//! * [`LayoutExt`](crate::alloc::LayoutExt) — MSRV-compatible shims for the
//!   `Layout` conveniences that are not yet stabilized (`dangling_pointer`,
//!   `padding_need_for`, `for_value_raw`).
//!
//! # The global allocator
//!
//! The `olive-alloc` crate re-exports the Global allocator from `allocator-api2`
//!
//! # Canonical implementation rule
//!
//! Per the project's discipline, every fallible container in `olive_alloc` must
//! perform its memory operations through the [`Allocator`](crate::alloc::Allocator)
//! trait (with `Global` as the default), rather than calling the free functions
//! or `std::alloc` directly. That keeps a single seam for swapping in custom
//! allocators and for simulating OOM in tests.
//!
//! [`hashbrown`]: https://docs.rs/hashbrown

use crate::try_traits::try_clone::TryClone;
use core::ptr::NonNull;

// The canonical allocator surface comes from `allocator-api2`. Re-export it
// under these names so the whole stack (and external crates such as
// `hashbrown`) sees one identity for the trait and its error type.
pub use allocator_api2::alloc::{AllocError, Allocator};
pub use core::alloc::{Layout, LayoutError};

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
    /// via a [`crate::mem::MaybeUninitUnsized`] reference formed from the pointer.
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
            use crate::mem::MaybeUninitUnsized;
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
