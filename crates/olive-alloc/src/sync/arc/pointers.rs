//! Shared pointer and layout helpers for [`Arc`](super::Arc) internals.
//!
//! This module centralizes every helper that operates on the raw
//! `ArcInner<T>` block: computing payload offsets, projecting between inner
//! pointers and payload pointers, seeding refcount headers, and detecting the
//! dangling-weak sentinel. Child modules (`construction`, `dst`,
//! `reconstitution`, …) import from here rather than duplicating the logic.

use crate::alloc::{Layout, LayoutError};
use core::mem::MaybeUninit;
use olive_core::alloc::LayoutExt;
use olive_core::ptr::{self, NonNull};

use super::ArcInner;

// ---------------------------------------------------------------------------
// Layout computation
// ---------------------------------------------------------------------------

/// Computes the layout for an `ArcInner<T>` given the payload's layout.
///
/// Mirrors std's `arc_inner_layout_for_value_layout`: build the layout from the
/// concrete `ArcInner<()>` header type, extend it by the value layout, and pad
/// to the combined alignment. Returns the padded total layout and the byte
/// offset at which the payload begins within the block.
pub(crate) fn arc_inner_layout_for_value_layout(
    value_layout: Layout,
) -> Result<(Layout, usize), LayoutError> {
    // The `LayoutError` is unreachable without UB — see the identical proof in
    // `crate::rc::rc_inner_layout_for_value_layout`.
    let header = Layout::new::<ArcInner<()>>();
    let (extended, offset) = header.extend(value_layout)?;
    Ok((extended.pad_to_align(), offset))
}

// ---------------------------------------------------------------------------
// Pointer projection
// ---------------------------------------------------------------------------

/// Casts a pointer to `ArcInner<T>` to a pointer to the payload within it.
///
/// Implemented by projecting the `value` field out of the fat pointer rather
/// than by forming a reference to the whole struct, to prevent the pointer from
/// being tagged.
///
/// # Safety
///
/// - `p` must point to a valid `ArcInner<T>` allocation block.
/// - The reference count fields must be initialized.
/// - The `T` value does not have to be initialized.
#[inline]
pub(crate) unsafe fn ptr_get_data<T: ?Sized>(p: *const ArcInner<T>) -> *const T {
    unsafe { &raw const (*p).value }
}

/// Mutable counterpart of [`ptr_get_data`]: yields a `*mut T` pointing at the
/// payload slot without ever casting away constness from an immutable
/// helper's result.
///
/// Mutating callers must go through this rather than doing
/// `ptr_get_data(p) as *mut T`, which would silently launder a `*const T` into
/// a `*mut T` and hide the fact that the caller is asserting write access.
///
/// # Safety
///
/// - `p` must point to a valid `ArcInner<T>` allocation block.
/// - The reference count fields must be initialized.
/// - The `T` value does not have to be initialized.
/// - `p` must have strong == 1.
#[inline]
pub(crate) unsafe fn ptr_get_data_mut<T: ?Sized>(p: *mut ArcInner<T>) -> *mut T {
    unsafe { &raw mut (*p).value }
}

/// Computes the byte offset from the start of an `ArcInner<T>` allocation to
/// the beginning of its `value` field, given a fat pointer to the payload.
///
/// # Safety
///
/// - `p` must point to a valid, initialized payload within a live
///   `ArcInner<T>` allocation.
/// - The memory must not be mutated during the call.
#[inline]
pub(crate) unsafe fn data_offset<T: ?Sized>(p: *const T) -> usize {
    // SAFETY: precondition from the caller.
    let value_layout = unsafe { Layout::for_value_pointer(p) };
    // Overflow is impossible here — see the proof in
    // `arc_inner_layout_for_value_layout`.
    let (_, offset) = arc_inner_layout_for_value_layout(value_layout)
        .expect("Arc header/payload layout overflow");
    offset
}

/// Reverse of [`ptr_get_data`]: casts a payload pointer back to its enclosing
/// `ArcInner<T>`.
///
/// # Safety
///
/// The pointer must have been produced by [`ptr_get_data`] (or
/// [`super::Arc::as_ptr`]) on a valid `ArcInner<T>` allocation.
#[inline]
pub(crate) unsafe fn data_get_ptr<T: ?Sized>(p: *const T) -> *const ArcInner<T> {
    // SAFETY: the live ArcInner<T> ensures that the pointer to T is valid.
    let offset = unsafe { data_offset(p) };
    // SAFETY: subtracting the header offset lands at the start of the
    // `ArcInner` allocation, which is in-bounds.
    unsafe { p.byte_sub(offset) as *const ArcInner<T> }
}

// ---------------------------------------------------------------------------
// Refcount header initialization
// ---------------------------------------------------------------------------

/// Initializes the two reference-count headers of a freshly-allocated
/// `ArcInner<T>` block to `(strong = 1, weak = 1)`.
///
/// # Safety
///
/// - `p` must point to a valid, aligned `ArcInner<T>` allocation block whose
///   header fields have not yet been initialized.
#[inline]
pub(crate) unsafe fn initialize_arcinner<T: ?Sized>(p: *mut ArcInner<T>) {
    unsafe {
        ptr::write(
            &raw mut (*p).strong,
            core::sync::atomic::AtomicUsize::new(1),
        );
        ptr::write(&raw mut (*p).weak, core::sync::atomic::AtomicUsize::new(1));
    }
}

// ---------------------------------------------------------------------------
// Dangling-weak sentinel
// ---------------------------------------------------------------------------

/// The sentinel address of a "dangling" [`Weak`](super::Weak) — one that never
/// referred to a real allocation (see [`Weak::new`](super::Weak::new)).
///
/// This is deliberately **misaligned** ([`usize::MAX`] is odd, while `ArcInner`
/// requires at least 2-alignment). No valid allocation can ever occupy a
/// misaligned address, so comparing against this value reliably detects "was
/// this weak ever attached to anything?" without any possibility of collision
/// with a real pointer.
pub(crate) const DANGLING_WEAK_ADDR: usize = usize::MAX;

/// True if `p` points at the dangling sentinel produced by
/// [`Weak::new`](super::Weak::new).
#[inline]
pub(crate) fn is_dangling_weak<T: ?Sized>(p: *const ArcInner<T>) -> bool {
    p.addr() == DANGLING_WEAK_ADDR
}

/// Builds the dangling inner pointer stored by [`Weak::new`](super::Weak::new).
///
/// The address word is pinned to [`DANGLING_WEAK_ADDR`] (`usize::MAX`), which
/// is deliberately **misaligned** relative to `ArcInner`'s required alignment.
/// This guarantees the sentinel can never collide with a real allocation's
/// address.
#[inline]
pub(crate) const fn dangling_inner_ptr<T: ?Sized>() -> NonNull<ArcInner<T>> {
    // SAFETY: `DANGLING_WEAK_ADDR` is non-zero, satisfying `NonNull`'s
    // invariant. The address is intentionally misaligned — no valid
    // allocation could sit there. `Drop`, `upgrade`, and all other access
    // paths check `is_dangling_weak` before touching memory, so the pointer
    // is never dereferenced despite its misalignment.
    unsafe {
        let mut slot: MaybeUninit<*mut ArcInner<T>> = MaybeUninit::zeroed();
        let data_offset = const { ptr::address_word_offset::<T>() };
        *slot.as_mut_ptr().byte_add(data_offset).cast::<usize>() = DANGLING_WEAK_ADDR;
        NonNull::new_unchecked(slot.assume_init())
    }
}

// ---------------------------------------------------------------------------
// Refcount cap
// ---------------------------------------------------------------------------

/// The maximum reference count permitted for either the strong or weak
/// counter of an [`ArcInner`](super::ArcInner) allocation.
///
/// `usize::MAX` is reserved as a sentinel for temporarily "locking" the
/// weak count, preventing `Arc::downgrade` from racing to create new
/// `Weak` references. `Arc::is_unique` (which backs `Arc::get_mut`)
/// needs to observe both the strong and weak counts as indicating
/// uniqueness in one logical atomic step; since they live in separate
/// atomic words, it locks the weak count while reading the strong
/// count to keep the two reads consistent.
pub(crate) const MAX_REFCOUNT: usize = usize::MAX - 1;

// ---------------------------------------------------------------------------
// Counter predicates
// ---------------------------------------------------------------------------

/// Returns true when the given strong count indicates the last strong
/// reference has been released and the value should be destroyed.
#[inline]
pub(crate) fn is_last_strong(strong: usize) -> bool {
    strong == 0
}

/// A checked increment that does not exceed [`MAX_REFCOUNT`].
#[inline]
pub(crate) fn checked_increment(n: usize) -> Option<usize> {
    if n >= MAX_REFCOUNT {
        None
    } else {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "cannot overflow: guarded by the check above."
        )]
        {
            Some(n + 1)
        }
    }
}
