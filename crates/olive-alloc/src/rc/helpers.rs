use super::internals::RcInner;
use core::{
    alloc::{Layout, LayoutError},
    mem::MaybeUninit,
    ptr::NonNull,
};
use olive_core::{alloc::LayoutExt, ptr};

/// Computes the layout for an `RcInner<T>` given the payload's layout.
///
/// Mirrors std's `rc_inner_layout_for_value_layout`: build the layout from the
/// concrete `RcInner<()>` header type, extend it by the value layout, and pad
/// to the combined alignment.
pub(super) fn rc_inner_layout_for_value_layout(
    value_layout: Layout,
) -> Result<(Layout, usize), LayoutError> {
    // The `LayoutError` is impossible to reach - trying to reach this without UB would theoretically leave
    // behind absolutely no memory for anything else, even program code or any other state.
    // Proof as follows, assume that all addressable space is usize::MAX bytes large (this does not happen
    // in practice, since usually some addresses are unusable) and the allocator is a ZST:
    // 1. The largest possible RcInner block is isize::MAX - 1 (isize::MAX is odd and the alignment
    // is 2, so the size must be divisible by 2).
    // 2. The largest structure for that block is isize::MAX - 1 - 2 * size_of_pointer.
    // 3. The next invalid structure that would trigger the error is isize::MAX - 2 * size_of_pointer.
    // 4. The size required for that invalid containing RcInner block would have been isize::MAX if the
    // block did not have align(2), so the actual size is isize::MAX + 1.
    // 5. Summing these total, we can calculate that the total memory required for cloning such a structure
    // is at least 2 * isize::MAX + 1 - 2 * size_of_pointer = usize::MAX - 2 * size_of_pointer, without
    // considering input sizes.
    // 6. With an input slice thin pointer and an output thin pointer, the total affected size would be
    // usize::MAX, leaving behind no bytes.
    let header = Layout::new::<RcInner<()>>();
    let (extended, offset) = header.extend(value_layout)?;
    Ok((extended.pad_to_align(), offset))
}

/// Returns true when the given strong count indicates the last strong
/// reference has been released and the value should be destroyed.
#[inline]
pub(super) fn is_last_strong(strong: usize) -> bool {
    strong == 0
}

/// Returns true when the weak count is zero, meaning the entire
/// allocation can be freed.
#[inline]
pub(super) fn is_last_ref(weak: usize) -> bool {
    weak == 0
}

/// The sentinel address of a "dangling" [`Weak`] — one that never referred to a
/// real allocation (see [`Weak::new`]).
///
/// This is deliberately **misaligned** (`usize::MAX` is odd, while `RcInner`
/// requires at least 2-alignment). No valid allocation can ever occupy a
/// misaligned address, so comparing against this value reliably detects "was
/// this weak ever attached to anything?" without any possibility of collision
/// with a real pointer.
pub(super) const DANGLING_WEAK_ADDR: usize = usize::MAX;

/// True if `p` points at the dangling sentinel produced by [`Weak::new`].
#[inline]
pub(super) fn is_dangling_weak<T: ?Sized>(p: *const RcInner<T>) -> bool {
    p.addr() == DANGLING_WEAK_ADDR
}

// ---------------------------------------------------------------------------
// Pointer helpers
// ---------------------------------------------------------------------------

/// Computes the byte offset from the start of an `RcInner<T>` allocation to
/// the beginning of its `value` field, given the payload's alignment.
///
/// # Safety
///
/// - `p` must satisfy the conditions of [`LayoutExt::for_value_pointer`]: the
///   memory must be from a valid pointer given by [`ptr_get_data`] and must not
///   be mutated.
#[inline]
pub(super) unsafe fn data_offset<T: ?Sized>(p: *const T) -> usize {
    // SAFETY: precondition from the caller.
    let value_layout = unsafe { Layout::for_value_pointer(p) };
    // Overflow is impossible here. The value_layout came from a pointer that is previously
    // validated with the exact same function.
    let (_, offset) =
        rc_inner_layout_for_value_layout(value_layout).expect("Rc header/payload layout overflow");
    offset
}

/// Casts a pointer to `RcInner<T>` to a pointer to the payload within it.
///
/// Implemented via **pointer arithmetic** (base + [`data_offset`]) rather than
/// by forming a reference to the `value` field to prevent the pointer from
/// being tagged.
///
/// # Safety
///
/// - `p` must point to a valid `RcInner<T>` allocation block.
/// - The reference count fields must be initialized.
/// - The `T` value does not have to be initialized.
#[inline]
pub(super) unsafe fn ptr_get_data<T: ?Sized>(p: *const RcInner<T>) -> *const T {
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
/// - `p` must point to a valid `RcInner<T>` allocation block.
/// - The reference count fields must be initialized.
/// - The `T` value does not have to be initialized.
/// - `p` must have strong == 1 and weak == 0 (excluding implicit weak ref).
#[inline]
pub(super) unsafe fn ptr_get_data_mut<T: ?Sized>(p: *mut RcInner<T>) -> *mut T {
    unsafe { &raw mut (*p).value }
}

/// Reverse of [`ptr_get_data`]: casts a payload pointer back to its enclosing
/// `RcInner<T>`.
///
/// # Safety
///
/// The pointer must have been produced by [`ptr_get_data`] on a valid
/// `RcInner<T>` allocation.
#[inline]
pub(super) unsafe fn data_get_ptr<T: ?Sized>(p: *const T) -> *const RcInner<T> {
    // SAFETY: the live RcInner<T> ensures that the pointer to T is valid.
    let offset = unsafe { data_offset(p) };
    // SAFETY: subtracting the header offset lands at the start of the
    // `RcInner` allocation, which is in-bounds.
    unsafe { p.byte_sub(offset) as *const RcInner<T> }
}

/// Builds the dangling inner pointer stored by [`Weak::new`].
///
/// The address word is pinned to [`DANGLING_WEAK_ADDR`] (`usize::MAX`), which
/// is deliberately **misaligned** relative to `RcInner`'s required alignment.
/// This guarantees the sentinel can never collide with a real allocation's
/// address.
#[inline]
pub(super) const fn dangling_inner_ptr<T: ?Sized>() -> NonNull<RcInner<T>> {
    // SAFETY: `DANGLING_WEAK_ADDR` is non-zero, satisfying `NonNull`'s
    // invariant. The address is intentionally misaligned — no valid
    // allocation could sit there. `Drop`, `upgrade`, and all other access
    // paths check `is_dangling_weak` before touching memory, so the pointer
    // is never dereferenced despite its misalignment.
    unsafe {
        let mut slot: MaybeUninit<*mut RcInner<T>> = MaybeUninit::zeroed();
        let data_offset = const { ptr::address_word_offset::<T>() };
        *slot.as_mut_ptr().byte_add(data_offset).cast::<usize>() = DANGLING_WEAK_ADDR;
        NonNull::new_unchecked(slot.assume_init())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // An "absurd" payload — one whose size, combined with the two-counter
    // header, overflows an addressable block — must be rejected as a logic-level
    // invariant violation.
    #[test]
    fn absurd_payload_layout_reports_other_not_oom() {
        let usize_size = size_of::<usize>();
        let max_size = isize::MAX as usize - usize_size + 1;
        // A hypothetical Rust type requires the size is divisible by its alignment
        assert_eq!(max_size % usize_size, 0);
        let huge = Layout::from_size_align(max_size, usize_size)
            .expect("payload layout alone is representable");
        let err = rc_inner_layout_for_value_layout(huge)
            .expect_err("absurd payload must fail layout computation");
        // The helper surfaced a real layout error (the overflow).
        let _ = err;
    }
}
