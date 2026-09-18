//! A fully-fallible port of the standard library's [`Rc`](stock_alloc::rc::Rc) and
//! [`Weak`](stock_alloc::rc::Weak).
//!
//! Compared with the std original, three things differ:
//!
//! * Every constructor that allocates a new node — `Rc::try_new`,
//!   `Rc::try_new_give_back`, and the various `*_in` variants — returns a
//!   [`Result`] carrying [`AllocError`] instead of panicking or aborting on
//!   out-of-memory.
//! * All allocation goes through the [`Allocator`] trait rather than free
//!   functions, giving a single swappable seam for custom allocators and OOM
//!   simulation in tests.
//! * Every reference-count mutation and allocator-handle clone is fallible and
//!   reports failures via [`TryRcError`]. Cloning an existing [`Rc`] does not
//!   allocate a new block (it only bumps the strong count), but it still must
//!   clone the allocator handle, which is a first-class fallible operation in
//!   this framework. The crate-standard [`TryClone`] impls surface these
//!   failures as [`TryCloneError`].
//!
//! The reference counts themselves are plain (non-atomic) integers, exactly
//! like the std originals: `Rc` is not thread-safe, and sharing one across
//! threads is undefined behavior.
//!
//! # Allocator bounds
//!
//! Cloning an `Rc<T, A>` requires `A: AllocatorTryClone`, not merely
//! `Allocator + Clone`. The stronger bound guarantees that a cloned allocator
//! handle is *equivalent* to the original — memory allocated through one may be
//! freed through the other — which is essential for the refcount-bump clone
//! path to remain sound. A plain `Clone` on the allocator would permit two
//! independent backing stores, breaking the invariant that all handles share
//! one allocation.

use core::borrow::Borrow;
use core::cell::Cell;
use core::cmp::Ordering;
use core::default::Default;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::Deref;
use core::pin::Pin;

use crate::alloc::{AllocError, Allocator, Global, Layout, LayoutError, StaticAllocator};
use olive_core::alloc::{AllocatorTryClone, LayoutExt};
use olive_core::ptr::PointerExt;
use olive_core::ptr::{self, NonNull};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError, TryCloneToUninit};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

// All allocator bounds in this module use `AllocatorTryClone` (not merely
// `Allocator + Clone`) so that a cloned allocator handle is guaranteed to be
// equivalent to the original — a prerequisite for refcount-bump clones to
// remain sound. See the module-level docs above for details.

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by fallible reference-counting operations on [`Rc`] and
/// [`Weak`].
///
/// Every operation that mutates a refcount or clones an allocator handle can
/// fail with one of these variants. The caller decides whether to propagate
/// the error, panic via `.expect(...)` (not recommended unless the call is
/// not supposed to fail), or recover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryRcError {
    /// A counter increment would exceed `usize::MAX` or a decrement would
    /// underflow below zero.
    ///
    /// This can arise from a logic error (unbalanced inc/dec), adversarial
    /// misuse of the raw pointer APIs, or — through the safe API alone — from
    /// [`core::mem::forget`]ing enough `Rc`s that their skipped `Drop` leaves
    /// the strong count stranded near `usize::MAX`; any further increment then
    /// overflows. Such leaks are rare in practice but do make this variant
    /// reachable without undefined behavior.
    OutOfBounds,
    /// Cloning the allocator handle failed.
    CloneAlloc(TryCloneError),
}

impl From<TryCloneError> for TryRcError {
    #[inline]
    fn from(e: TryCloneError) -> Self {
        Self::CloneAlloc(e)
    }
}

impl Display for TryRcError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds => write!(f, "reference count out of bounds"),
            Self::CloneAlloc(e) => write!(f, "allocator clone failed: {e}"),
        }
    }
}

impl From<AllocError> for TryRcError {
    #[inline]
    fn from(e: AllocError) -> Self {
        Self::CloneAlloc(TryCloneError::Alloc(e))
    }
}

impl core::error::Error for TryRcError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::OutOfBounds => None,
            Self::CloneAlloc(e) => Some(e),
        }
    }
}

/// Error returned by fallible reference-count operations that only mutate a
/// counter.
///
/// A counter increment would exceed `usize::MAX` or a decrement would underflow
/// below zero. This can arise from a logic error (unbalanced inc/dec),
/// adversarial misuse of the raw pointer APIs, or — through the safe API alone —
/// from [`core::mem::forget`]ing enough `Rc`s that their skipped `Drop` strands
/// the strong count near `usize::MAX`, so a further increment overflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryRcOutOfBoundsError;

impl Display for TryRcOutOfBoundsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("reference count out of bounds")
    }
}

impl core::error::Error for TryRcOutOfBoundsError {}

impl From<TryRcOutOfBoundsError> for TryRcError {
    #[inline]
    fn from(_: TryRcOutOfBoundsError) -> Self {
        Self::OutOfBounds
    }
}

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
// Shared internals
// ---------------------------------------------------------------------------

/// Internal representation of an `Rc` allocation.
///
/// This mirrors the std internal layout: two `usize` counters followed by the
/// payload, allocated as a single block. The `align(2)` attribute ensures that
/// on exotic targets where `usize` has 1-byte alignment (e.g. AVR), the two
/// counters are at least 2-aligned so they cannot overlap with an odd-aligned
/// payload. On normal targets this is a no-op since `usize` is already
/// ≥ 2-aligned.
///
/// Using `#[repr(C)]` guarantees the field order is predictable: `strong` at
/// offset 0, `weak` at offset `size_of::<usize>()`, and `value` after that.
#[repr(C, align(2))]
pub(crate) struct RcInner<T: ?Sized> {
    /// Number of strong (`Rc`) pointers currently alive. Zero means no strong
    /// references remain; the internal value may then be dropped even if
    /// weak refs exist.
    strong: Cell<usize>,
    /// Number of weak (`Weak`) pointers currently alive. Includes the implicit
    /// weak reference held by every live `Rc`, so this is always at least the
    /// number of `Rc`s while any strong reference exists.
    weak: Cell<usize>,
    /// The contained value.
    value: T,
}

impl<T: ?Sized> RcInner<T> {
    /// Reads the current strong count.
    #[inline]
    pub(crate) fn strong(&self) -> usize {
        self.strong.get()
    }

    /// Reads the current weak count.
    #[inline]
    pub(crate) fn weak(&self) -> usize {
        self.weak.get()
    }

    /// Increments the strong count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already at
    /// `usize::MAX`.
    #[inline]
    pub(crate) fn inc_strong(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.strong.get();
        match cur.checked_add(1) {
            Some(next) => {
                self.strong.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }

    /// Decrements the strong count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already zero
    /// (unbalanced decrement).
    #[inline]
    pub(crate) fn dec_strong(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.strong.get();
        match cur.checked_sub(1) {
            Some(next) => {
                self.strong.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }

    /// Increments the weak count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already at
    /// `usize::MAX`.
    #[inline]
    pub(crate) fn inc_weak(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.weak.get();
        match cur.checked_add(1) {
            Some(next) => {
                self.weak.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }

    /// Decrements the weak count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already zero
    /// (unbalanced decrement).
    #[inline]
    pub(crate) fn dec_weak(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.weak.get();
        match cur.checked_sub(1) {
            Some(next) => {
                self.weak.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }
}

/// Helper type to allow accessing the reference counts without
/// making any assertions about the data field.
///
/// When a `Weak` outlives all `Rc`s, the payload (`value`) has been dropped
/// in-place but the allocation remains alive (pinned by the weak count). A
/// `&RcInner<T>` covering the whole struct would assert validity of the
/// already-dropped payload. `WeakInner` holds only references to the two
/// counter cells, which are always valid while the allocation exists.
pub(crate) struct WeakInner<'a> {
    strong: &'a Cell<usize>,
    weak: &'a Cell<usize>,
}

impl WeakInner<'_> {
    /// Reads the current strong count.
    #[inline]
    fn strong(&self) -> usize {
        self.strong.get()
    }

    /// Reads the current weak count.
    #[inline]
    fn weak(&self) -> usize {
        self.weak.get()
    }

    /// Increments the strong count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already at
    /// `usize::MAX`.
    #[inline]
    fn inc_strong(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.strong.get();
        match cur.checked_add(1) {
            Some(next) => {
                self.strong.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }

    /// Decrements the weak count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the count is already zero
    /// (unbalanced decrement).
    #[inline]
    fn dec_weak(&self) -> Result<(), TryRcOutOfBoundsError> {
        let cur = self.weak.get();
        match cur.checked_sub(1) {
            Some(next) => {
                self.weak.set(next);
                Ok(())
            }
            None => Err(TryRcOutOfBoundsError),
        }
    }
}

/// Computes the layout for an `RcInner<T>` given the payload's layout.
///
/// Mirrors std's `rc_inner_layout_for_value_layout`: build the layout from the
/// concrete `RcInner<()>` header type, extend it by the value layout, and pad
/// to the combined alignment.
fn rc_inner_layout_for_value_layout(value_layout: Layout) -> Result<(Layout, usize), LayoutError> {
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
fn is_last_strong(strong: usize) -> bool {
    strong == 0
}

/// Returns true when the weak count is zero, meaning the entire
/// allocation can be freed.
#[inline]
fn is_last_ref(weak: usize) -> bool {
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
const DANGLING_WEAK_ADDR: usize = usize::MAX;

/// True if `p` points at the dangling sentinel produced by [`Weak::new`].
#[inline]
fn is_dangling_weak<T: ?Sized>(p: *const RcInner<T>) -> bool {
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
unsafe fn data_offset<T: ?Sized>(p: *const T) -> usize {
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
unsafe fn ptr_get_data<T: ?Sized>(p: *const RcInner<T>) -> *const T {
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
unsafe fn ptr_get_data_mut<T: ?Sized>(p: *mut RcInner<T>) -> *mut T {
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
unsafe fn data_get_ptr<T: ?Sized>(p: *const T) -> *const RcInner<T> {
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
const fn dangling_inner_ptr<T: ?Sized>() -> NonNull<RcInner<T>> {
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
pub(crate) struct UniqueRcUninit<T: ?Sized, A: Allocator> {
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
    pub(crate) fn try_new(alloc: A) -> Result<Self, AllocError> {
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
    pub(crate) fn try_new_for_value(src: &T, alloc: A) -> Result<Self, TryCloneError> {
        let value_layout = Layout::for_value(src);
        let (layout, _) = rc_inner_layout_for_value_layout(value_layout)
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
    pub(crate) fn data_ptr(&self) -> *mut u8 {
        // Project the `value` field out of the fat pointer to get the exact
        // address of the payload slot.
        unsafe { &raw mut (*self.ptr.as_ptr()).value as *mut u8 }
    }

    /// Consumes the handle and produces the finalized `Rc<T, A>`.
    ///
    /// # Safety contract
    ///
    /// The caller guarantees that the payload region (pointed to by
    /// [`data_ptr`](Self::data_ptr)) has been fully initialized with a valid
    /// `T` before calling this method.
    #[inline]
    pub(crate) fn into_rc(self) -> Rc<T, A> {
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
// Rc declaration
// ---------------------------------------------------------------------------

/// A single-threaded reference-counting pointer.
///
/// `Rc<T>` shares ownership of a heap-allocated `T` among any number of strong
/// references. The value is dropped when the last strong reference disappears;
/// [`Weak`] references do not keep it alive.
///
/// Unlike `std::rc::Rc`, constructing a fresh node can fail: use
/// [`try_new`](Self::try_new) and friends, which return
/// `Result<Self, AllocError>` instead of panicking or aborting on out-of-memory.
pub struct Rc<T: ?Sized, A: Allocator = Global> {
    ptr: NonNull<RcInner<T>>,
    alloc: A,
    // This phantom prevents impl of Send and Sync
    _marker: PhantomData<*const T>,
}

// ---------------------------------------------------------------------------
// Global construction block
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

    /// Allocates a new `Rc<T>` with all bytes zeroed on the global allocator.
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
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed() -> Result<Self, AllocError> {
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
// Global reconstitution block
// ---------------------------------------------------------------------------

impl<T: ?Sized> Rc<T, Global> {
    /// Constructs a new `Rc<T>` from a raw pointer previously produced by
    /// [`Rc::into_raw`].
    ///
    /// # Safety
    ///
    /// * Creating a `Rc<T>` from a pointer other than one returned from
    ///   [`Rc::<T>::into_raw`](Rc::into_raw) or [`Rc::into_raw_with_allocator`](Rc::into_raw_with_allocator)
    ///   is undefined behavior.
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Rc<U>` was constructed
    ///   through `Rc<T>` and then converted to `Rc<U>` through an [unsized
    ///   coercion].
    /// * Note that if `U` or `U`'s data pointer is not `T` but has the same size
    ///   and alignment, this is basically like transmuting references of
    ///   different types. See [`mem::transmute`](core::mem::transmute) for more information
    ///   on what restrictions apply in this case.
    /// * The raw pointer must point to a block of memory allocated by the global allocator.
    /// * The user of [`Rc::from_raw`] has to make sure a specific value of `T` is only
    ///   dropped once.
    #[inline]
    pub unsafe fn from_raw(p: *const T) -> Self {
        // SAFETY: caller guarantees `p` derives from `into_raw`; converting it
        // back restores the exact reference counts the original had.
        unsafe {
            let inner = data_get_ptr(p);
            Rc {
                ptr: NonNull::new_unchecked(inner as *mut RcInner<T>),
                alloc: Global,
                _marker: PhantomData,
            }
        }
    }

    /// Converts an `Rc<T>` allocated using the global allocator into a raw pointer.
    ///
    /// The caller takes ownership of the reference count carried by the pointer
    /// and must eventually reconstruct an `Rc` from it (via
    /// [`from_raw`](Self::from_raw)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw(rc: Self) -> *const T {
        let me = ManuallyDrop::new(rc);
        let _alloc = unsafe { ptr::read(&me.alloc) };
        Rc::as_ptr(&me)
    }
}

// ---------------------------------------------------------------------------
// Generic construction block (sized)
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
        Ok(uninit.into_rc())
    }

    /// Allocates a new `Rc<T>` with all bytes zeroed, parameterized over the
    /// allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed_in(alloc: A) -> Result<Self, AllocError> {
        let layout = Layout::new::<RcInner<T>>();
        let block = alloc.allocate_zeroed(layout)?;
        // SAFETY: `allocate_zeroed` returns a valid, aligned, non-null pointer
        // to a freshly allocated block of at least `layout.size()` bytes.
        let ptr = unsafe { NonNull::new_unchecked(block.cast::<RcInner<T>>().as_ptr()) };
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
            ptr::write(ptr_get_data_mut(inner.as_ptr()), value);
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

    /// Consumes this `Rc`, attempting to take the value out, returning the
    /// owned `T`.
    ///
    /// # Errors
    ///
    /// Returns the input `Rc` if the value could not be unwrapped because
    /// ownership is shared or weak references are still alive.
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // Move-out is safe only when we are the unique owner of the payload:
        // exactly one strong reference and no outstanding weak references.
        if Self::strong_count(&this) == 1 && Self::weak_count(&this) == 0 {
            // Suppress the handle's implicit drop so our manual teardown below
            // is the single owner of the strong reference.
            let me = ManuallyDrop::new(this);
            // SAFETY: we are the sole strong reference with no weaks, so the
            // payload is valid and exclusively ours. `ptr::read` moves the `T`
            // out of the allocation.
            let val = unsafe { ptr::read(&**me) };
            let alloc = unsafe { ptr::read(&me.alloc) };
            let inner: &RcInner<T> = Rc::inner(me.deref());
            inner.dec_strong().expect("strong count underflow");
            // Drop a throwaway `Weak` standing in for this `Rc`'s implicit weak
            // reference.
            let _dummy_weak = Weak::<T, A> {
                ptr: me.ptr,
                alloc,
                _marker: PhantomData,
            };
            Ok(val)
        } else {
            Err(this)
        }
    }

    /// Consumes this `Rc`, returning the owned value `T` if `this` is the last
    /// strong reference, or clone the reference otherwise.
    ///
    /// # Errors
    ///
    /// Returns the [`TryCloneError`] if the value could not be cloned in the
    /// fallback path.
    #[inline]
    pub fn unwrap_or_try_clone(this: Self) -> Result<T, TryCloneError>
    where
        T: TryClone,
    {
        Self::unwrap_or_try_clone_give_back(this).map_err(|(_self, e)| e)
    }

    /// Like [`Self::unwrap_or_try_clone`] but it also returns the original `this`
    /// on error.
    ///
    /// # Errors
    ///
    /// Returns the input `Rc` paired with [`TryCloneError`] if the value
    /// could not be cloned in the fallback path.
    #[inline]
    pub fn unwrap_or_try_clone_give_back(this: Self) -> Result<T, (Self, TryCloneError)>
    where
        T: TryClone,
    {
        match Self::try_unwrap(this) {
            Ok(s) => Ok(s),
            Err(rc) => (*rc).try_clone().map_err(|e| (rc, e)),
        }
    }
}

// ---------------------------------------------------------------------------
// Generic pointer transformation block (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Like [`from_raw`](Self::from_raw), but parameterized over the choice of
    /// allocator.
    ///
    /// # Safety
    ///
    /// * Creating a [`Rc<T, A>`] from a pointer other than one returned from
    ///   [`Rc<U, A>::into_raw`](Rc::into_raw) or
    ///   [`Rc<U, A>::into_raw_with_allocator`](Rc::into_raw_with_allocator)
    ///   is undefined behavior.
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Rc<U, A>` was constructed
    ///   through `Rc<T, A>` and then converted to `Rc<U, A>` through an [unsized
    ///   coercion].
    /// * Note that if `U` or `U`'s data pointer is not `T` but has the same size
    ///   and alignment, this is basically like transmuting references of
    ///   different types. See [`mem::transmute`](core::mem::transmute) for
    ///   more information on what restrictions apply in this case.
    /// * The raw pointer must point to a block of memory allocated by `alloc`
    /// * The user of `from_raw` has to make sure a specific value of `T` is only
    ///   dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned [`Rc<T, A>`] is never accessed.
    #[inline]
    pub unsafe fn from_raw_in(p: *const T, alloc: A) -> Self {
        // SAFETY: caller guarantees validity.
        unsafe {
            let inner = data_get_ptr(p);
            Rc {
                ptr: NonNull::new_unchecked(inner as *mut RcInner<T>),
                alloc,
                _marker: PhantomData,
            }
        }
    }

    /// Converts an [`Rc<T, A>`] into a raw pointer, retaining its allocator.
    ///
    /// The caller takes ownership of both the allocation and the allocator and
    /// must eventually reconstruct an `Rc` from them (via
    /// [`from_raw_in`](Self::from_raw_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw_with_allocator(rc: Self) -> (*const T, A) {
        let me = ManuallyDrop::new(rc);
        // SAFETY: `rc`'s inner field is fully initialized
        let ptr = Rc::as_ptr(&me);
        let alloc = unsafe { ptr::read(&me.alloc) };
        (ptr, alloc)
    }
}

// ---------------------------------------------------------------------------
// Unsized construction via TryCloneToUninit
// ---------------------------------------------------------------------------

#[allow(private_bounds)]
impl<T: ?Sized + TryCloneToUninit> Rc<T, Global> {
    /// Clones a `&T` into a freshly allocated `Rc<T, Global>` for potentially
    /// unsized `T`.
    ///
    /// This is the fallible analogue of std's `Rc::from(&slice[..])`,
    /// `Rc::from("literal")`, etc. It works for any type implementing
    /// [`TryCloneToUninit`]: in practice that means sized types, `str`, and
    /// slices `[T]`. Trait-object payloads (`dyn Trait`) are not yet supported.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails, an element clone
    /// fails, or the payload layout is absurd (see
    /// [`try_clone_from_ref_in`](Self::try_clone_from_ref_in)).
    #[inline]
    pub fn try_clone_from_ref(src: &T) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, Global)
    }
}

#[allow(private_bounds)]
impl<T: ?Sized + TryCloneToUninit, A: Allocator> Rc<T, A> {
    /// Clones a `&T` into a freshly allocated `Rc<T, A>` for potentially
    /// unsized `T`, using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails, any element clone
    /// fails, or the payload layout is absurd (e.g. a slice whose length would
    /// overflow when combined with the header). The last case is reported as
    /// [`TryCloneError::Other`], not an OOM-style error, because it can never
    /// succeed on retry.
    #[inline]
    pub fn try_clone_from_ref_in(src: &T, alloc: A) -> Result<Self, TryCloneError> {
        let uninit = UniqueRcUninit::try_new_for_value(src, alloc)?;
        // SAFETY: `uninit.data_ptr()` points to the uninitialized payload slot
        // within a live allocation of the correct size and alignment.
        unsafe { <T as TryCloneToUninit>::try_clone_to_uninit(src, uninit.data_ptr()) }?;
        Ok(uninit.into_rc())
    }
}

// Convenience wrappers for common unsized types.
impl<T: TryClone, A: Allocator> Rc<[T], A> {
    /// Creates a new `Rc<[T]>` by cloning the slice of T items.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_slice_in(src: &[T], alloc: A) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, alloc)
    }
}

impl<T: TryClone> Rc<[T], Global> {
    /// Convenience wrapper around [`Self::try_from_slice_in`]
    /// using the global allocator.
    #[inline]
    pub fn try_from_slice(src: &[T]) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref(src)
    }
}

impl<A: Allocator> Rc<str, A> {
    /// Allocates a new `Rc<str>` by copying the UTF-8 bytes from `src` into
    /// fresh heap memory. Fallible analogue of std's `Rc::from("literal")`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_str_in(src: &str, alloc: A) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, alloc)
    }
}

impl Rc<str, Global> {
    /// Convenience wrapper around [`Self::try_from_str_in`] using
    /// the global allocator.
    #[inline]
    pub fn try_from_str(src: &str) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref(src)
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

// ---------------------------------------------------------------------------
// Query and mutation block (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Gets a shared raw pointer to the underlying data.
    ///
    /// The returned pointer carries write provenance derived from the live
    /// allocation, so it can be used in splitting and reconstitution
    /// operations (e.g. [`Self::into_raw_with_allocator`] / [`Self::into_raw`] /
    /// [`Self::from_raw_in`] / [`Self::from_raw`]).
    #[must_use]
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr: *mut RcInner<T> = this.ptr.as_ptr();
        unsafe { &raw mut (*ptr).value }
    }

    /// Returns a shared reference to the allocation's internal [`RcInner`]
    /// header (strong/weak counters and payload).
    #[inline]
    pub(crate) fn inner(this: &Self) -> &RcInner<T> {
        // SAFETY: an `Rc` always owns a live allocation whose header is valid
        // for reads while `self` exists.
        unsafe { &*this.ptr.as_ptr() }
    }

    /// Returns a **mutable** reference to the allocation's internal
    /// [`RcInner`] header.
    ///
    /// Prefer this over [`Self::inner`] when the caller intends to mutate the
    /// payload through the returned reference.
    ///
    /// # Safety
    ///
    /// - The pointer's strong count must be 1.
    #[inline]
    pub(crate) unsafe fn inner_mut(this: &mut Self) -> &mut RcInner<T> {
        // SAFETY: an `Rc` always owns a live allocation whose header is valid
        // for reads and writes while `this` exists; the `&mut self` receiver
        // guarantees exclusive access for the call's duration.
        unsafe { &mut *this.ptr.as_ptr() }
    }

    /// Gets a shared reference to the allocator backing this `Rc`.
    ///
    /// Implemented as an associated function (taking `&Self`) rather than an
    /// inherent method so that autoref-based deref coercion is not shadowed:
    /// a `T` whose own API exposes an `allocator` method remains reachable via
    /// `rc.as_ref().allocator()` / `(*rc).allocator()`.
    #[must_use]
    #[inline]
    pub const fn allocator(this: &Self) -> &A {
        &this.alloc
    }

    /// Determines if two `Rc` pointers point to the same allocation.
    ///
    /// This ignores the metadata comparison.
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::addr_eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Returns the number of strong [`Rc`] pointers to this allocation.
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        Self::inner(this).strong()
    }

    /// Returns the number of weak (`Weak`) pointers to this allocation,
    /// excluding the implicit weak reference held by each strong pointer.
    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        Self::inner(this).weak().saturating_sub(1)
    }

    /// Returns `true` if there are no other `Rc` or [`Weak`] pointers to this
    /// allocation.
    #[inline]
    pub fn is_unique(this: &Self) -> bool {
        Rc::weak_count(this) == 0 && Rc::strong_count(this) == 1
    }

    /// Gets a mutable reference to the contained value if this `Rc` is the
    /// sole strong reference **and** no [`Weak`] pointers to the same
    /// allocation exist. Returns `None` otherwise.
    ///
    /// This mirrors std's `Rc::get_mut`, which requires both that there are no
    /// other strong references *and* no weak references, because a live `Weak`
    /// could upgrade back into a second strong reference at any moment.
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        // SAFETY: ensured by the uniqueness check below.
        if Self::is_unique(this) {
            Some(unsafe { Self::get_mut_unchecked(this) })
        } else {
            None
        }
    }

    /// Gets a mutable reference to the contained value **without** checking
    /// the strong count.
    ///
    /// # Safety
    ///
    /// If any other Rc or Weak pointers to the same allocation exist,
    /// then they must not be dereferenced or have active borrows for the
    /// duration of the returned borrow, and their inner type must be exactly
    /// the same as the inner type of this Rc (including lifetimes).
    ///
    /// This is trivially the case if no such pointers exist, for example
    /// immediately after Rc::new.
    #[inline]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        let inner = unsafe { Self::inner_mut(this) };
        &mut inner.value
    }

    /// Makes a mutable reference into the given `Rc`, disassociating other
    /// references by moving or cloning as needed.
    ///
    /// Three cases, mirroring std's `Rc::make_mut`:
    ///
    /// * **Unique** — no other [`Rc`] or [`Weak`] pointers exist: the payload is
    ///   accessed in place with no allocation.
    /// * **Shared** — more than one strong reference exists: the inner value is
    ///   cloned into a fresh allocation via [`TryCloneToUninit`], and this [`Rc`]
    ///   is replaced in place to point at the clone. The old shared allocation keeps
    ///   serving the remaining owners.
    /// * **Only weak refs remain** — exactly one strong reference (this one) but
    ///   some [`Weak`] pointers: the value is moved out of the old block into a
    ///   freshly allocated one, and the old block's counts are decremented, so its
    ///   payload is logically gone. The surviving `Weak`s no longer points to a live
    ///   value (their `upgrade` will now fail). No clone is performed.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if cloning the allocator handle, allocating the
    /// replacement block, or the clone itself fails.
    #[inline]
    pub fn try_make_mut(this: &mut Self) -> Result<&mut T, TryCloneError>
    where
        T: TryCloneToUninit,
        A: AllocatorTryClone,
    {
        // Case 1: shared (more than one strong reference) — clone the payload
        // into a fresh block and reassign `this` to the clone.
        if Self::strong_count(this) != 1 {
            let alloc = A::try_clone(&this.alloc)?;
            let new_rc = Rc::try_clone_from_ref_in(&**this, alloc)?;
            *this = new_rc;
            // SAFETY: `new_rc` was just constructed with strong == 1 and weak == 0 (excluding the implicit ref),
            // so it is uniquely owned.
            return Ok(unsafe { Self::get_mut_unchecked(this) });
        }

        // Case 2: unique — mutate in place. We just ensured strong == 1 above.
        if Self::weak_count(this) == 0 {
            // SAFETY: strong == 1 and weak == 0 (excluding the implicit ref),
            // so we are the exclusive owner.
            return Ok(unsafe { Self::get_mut_unchecked(this) });
        }

        // Case 3: strong == 1 but weak > 0 — steal the data into a fresh block.
        let size_of_val = size_of_val::<T>(&**this);
        let alloc = A::try_clone(&this.alloc)?;
        let in_progress = UniqueRcUninit::try_new_for_value(&**this, alloc)?;
        // SAFETY: we own the only strong reference, so moving the payload out by
        // byte-copying `size_of_val` bytes is sound for any `T: ?Sized`. After
        // this copy the source slot is treated as moved-from; the counters are
        // adjusted immediately below so the old block never drops the payload.
        unsafe {
            // Almost all the block below all the way until drop() will not panic.
            ptr::copy_nonoverlapping(
                Self::as_ptr(this).cast::<u8>(),
                in_progress.data_ptr().cast::<u8>(),
                size_of_val,
            );

            // Leave the old block with 0 strong refs: the data has effectively
            // been moved to the new rc. Unreachable underflow cannot occur — we
            // hold a live strong reference, so the count is at least one.
            Self::inner(this)
                .dec_strong()
                .expect("strong count underflow");

            // Remove the implicit strong-held weak ref. Other `Weak`s remain and
            // are responsible for freeing the (now empty) block.
            Self::inner(this).dec_weak().expect("weak count underflow");

            // Last chance to not accidentally forget the allocator before we
            // overwrite `this`. The `_alloc` will be dropped last after the state has
            // committed to avoid panics.
            let alloc = ptr::read(&this.alloc);

            // Replace `this` with the freshly constructed `Rc` holding the moved
            // data. Writing over `this` using pointer syntax inhibits its `Drop` impl,
            // so the old Rc is not dropped.
            ptr::write(this, in_progress.into_rc());

            // Panics may occur here, but the state is already fully committed at this point.
            drop(alloc);
        }

        // SAFETY: after the move-out, `this.ptr` is the *only* pointer to the
        // new allocation (strong == 1, weak == 0), and we required the `Rc<T>`
        // itself to be `mut`, so this is the only possible reference to the
        // payload.
        Ok(unsafe { Self::get_mut_unchecked(this) })
    }

    /// Same as [`try_increment_strong_count`](Self::try_increment_strong_count),
    /// but parameterized over the allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the strong count would overflow
    /// `usize`.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Rc<T>::into_raw`] or
    ///   [`Rc<T>::into_raw_with_allocator`] and must satisfy the layout required by
    ///   [`Rc<T>::from_raw_in`].
    /// - `ptr` must point to a block allocated by the `alloc`.
    /// - The `Rc` must be valid - the strong count must not be 0.
    #[inline]
    pub unsafe fn try_increment_strong_count_in(
        ptr: *const T,
        alloc: &A,
    ) -> Result<(), TryRcOutOfBoundsError> {
        // NOTE: taking `alloc` by reference avoids paying for an allocator clone
        // that this operation does not need and reduces the caller need to clone 
        // the allocator. 
        // The reconstituted handle is wrapped in `ManuallyDrop` to prevent an 
        // unintentional refcount decrement.
        // The allocator reference also helps avoid allocator leaks.
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation backed by
        // `alloc`. Wrapping in `ManuallyDrop` prevents a panic during unwinding
        // from decrementing the refcount.
        let me = unsafe { ManuallyDrop::new(Rc::from_raw_in(ptr, &alloc)) };
        let inner = Rc::inner(&me);
        if inner.strong() == 0 {
            return Err(TryRcOutOfBoundsError);
        }
        inner.inc_strong()?;
        Ok(())
    }

    /// Same as [`try_decrement_strong_count`](Self::try_decrement_strong_count),
    /// but parameterized over the allocator so the correct deallocator is used
    /// when the last strong reference goes away.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the strong count is already zero
    /// (unbalanced decrement).
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Rc<T>::into_raw`] or
    ///   [`Rc<T>::into_raw_with_allocator`] and must satisfy the layout required by
    ///   [`Rc<T>::from_raw_in`].
    /// - `ptr` must point to a block allocated by the `alloc`.
    /// - The `Rc` must be valid - the strong count must not be 0.
    /// - This method can be used to free the [`Rc`] and its backing storage.
    #[inline]
    pub unsafe fn try_decrement_strong_count_in(
        ptr: *const T,
        alloc: &A,
    ) -> Result<(), TryRcOutOfBoundsError> {
        // NOTE: taking `alloc` by reference avoids paying for an allocator clone
        // that this operation does not need and reduces the caller need to clone 
        // the allocator. 
        // The reconstituted handle is wrapped in `ManuallyDrop` to prevent an 
        // unintentional refcount decrement.
        // The allocator reference also helps avoid allocator leaks.
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation backed by
        // `alloc`. Wrapping in `ManuallyDrop` prevents a panic during unwinding
        // from arbitrarily lowering the refcount.
        let mut me = unsafe { ManuallyDrop::new(Rc::from_raw_in(ptr, &alloc)) };
        let inner = Rc::inner(&me);
        inner.dec_strong()?;
        if is_last_strong(inner.strong()) {
            // SAFETY: we are the last strong reference.
            unsafe { Rc::drop_slow(&mut me) };
        }
        Ok(())
    }
}

impl<T: ?Sized> Rc<T, Global> {
    /// Increments the strong count of the allocation backing `ptr` without
    /// constructing an `Rc` wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the strong count would overflow
    /// `usize`.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Rc<T>::into_raw`] or
    ///   [`Rc<T>::into_raw_with_allocator`] and must satisfy the layout required by [`Rc<T>::from_raw`].
    /// - `ptr` must point to a block allocated by the global allocator.
    /// - The `Rc` must be valid - the strong count must not be 0.
    #[inline]
    pub unsafe fn try_increment_strong_count(ptr: *const T) -> Result<(), TryRcOutOfBoundsError> {
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation.
        unsafe { Self::try_increment_strong_count_in(ptr, &Global) }
    }

    /// Decrements the strong count of the allocation backing `ptr`. When the
    /// count reaches zero the value is dropped and the block is freed.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcOutOfBoundsError`] if the strong count is already zero
    /// (unbalanced decrement).
    ///
    /// # Safety
    ///
    /// - `ptr` must have been produced by [`Rc<T>::into_raw`] or
    ///   [`Rc<T>::into_raw_with_allocator`] and must satisfy the layout required by [`Rc<T>::from_raw`].
    /// - `ptr` must point to a block allocated by the global allocator.
    /// - The `Rc` must be valid - the strong count must not be 0.
    /// - This method can be called to release the Rc and backing storage, similar to
    ///   calling [`Rc<T>::from_raw`] and dropping the value.
    #[inline]
    pub unsafe fn try_decrement_strong_count(ptr: *const T) -> Result<(), TryRcOutOfBoundsError> {
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation backed by
        // the global allocator.
        unsafe { Self::try_decrement_strong_count_in(ptr, &Global) }
    }
}

// ---------------------------------------------------------------------------
// Conversion to Weak (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: AllocatorTryClone> Rc<T, A> {
    /// Borrows an [`Rc`] as a [`Weak`] pointer.
    ///
    /// This does not increment the strong count, so the resulting `Weak` will
    /// not prevent the value from being dropped once all strong references are
    /// gone.
    ///
    /// The operation is fallible because it must clone the allocator handle
    /// (via [`AllocatorTryClone`]) to attach to the new `Weak`, and counter
    /// mutation can report out-of-bounds conditions. Cloning an allocator
    /// handle is a first-class fallible operation in this framework.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if the weak count would overflow
    /// `usize`, or [`TryRcError::CloneAlloc`] if cloning the allocator handle
    /// fails.
    #[inline]
    pub fn try_downgrade(this: &Self) -> Result<Weak<T, A>, TryRcError> {
        let alloc = A::try_clone(&this.alloc)?;
        let inner = Self::inner(this);
        inner.inc_weak().map_err(TryRcError::from)?;
        Ok(Weak {
            ptr: this.ptr,
            alloc,
            _marker: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Deref (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Deref for Rc<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `ptr` is always a valid, aligned, non-null pointer to an
        // initialized `T`.
        unsafe { &*ptr_get_data(self.ptr.as_ptr()) }
    }
}

// ---------------------------------------------------------------------------
// Clone / TryClone (?Sized)
// ---------------------------------------------------------------------------

// The `TryClone` impl requires `A: AllocatorTryClone` (not merely
// `Allocator + Clone`) so that the cloned allocator handle is guaranteed to be
// equivalent to the original — a prerequisite for the refcount-bump clone to
// remain sound. Both the allocator clone and the counter bump are fallible;
// either failure surfaces as [`TryCloneError`] here.
impl<T: ?Sized, A: AllocatorTryClone> TryClone for Rc<T, A> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Bump the strong count and return a new handle sharing the same
        // allocation. The allocator is cloned via `AllocatorTryClone`.
        let alloc = A::try_clone(&self.alloc)?;
        let inner = Self::inner(self);
        inner
            .inc_strong()
            .map_err(|_| TryCloneError::Other("strong count out of bounds"))?;
        Ok(Rc {
            ptr: self.ptr,
            alloc,
            _marker: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Drop
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Drop for Rc<T, A> {
    #[inline]
    fn drop(&mut self) {
        // Decrement the strong count through an exclusive `&mut` re-borrow
        // (`inner_mut`) rather than aliasing a shared borrow against the write,
        // which Stacked Borrows would reject. The helper reports out-of-bounds
        // conditions, which are unreachable here: we own a live strong
        // reference, so the count is at least one and cannot underflow.
        let inner = Rc::inner(self);
        inner.dec_strong().expect("strong count underflow");
        if is_last_strong(inner.strong()) {
            // SAFETY: we are the last strong reference.
            unsafe { Self::drop_slow(self) };
        }
    }
}

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Destroys the value and conditionally frees the block after the last
    /// strong reference has been dropped.
    ///
    /// # Safety
    /// - The Rc must be uniquely owned (or strong == 1).
    #[inline(never)]
    unsafe fn drop_slow(this: &mut Self) {
        // Construct a temporary `Weak` standing in for this `Rc`'s implicit
        // weak reference. This is the decrement or deallocate guard that
        // unconditionally runs even if the call is unwinding.
        let _fake_weak = Weak::<T, &A> {
            ptr: this.ptr,
            alloc: &this.alloc,
            _marker: PhantomData,
        };

        // SAFETY: the value is still present in the block (we are inside the
        // last strong's drop, before any free). Destroying it here is exactly
        // what dropping the final strong reference must do.
        unsafe {
            ptr::drop_in_place(&mut (*this.ptr.as_ptr()).value);
        }
    }
}

// ---------------------------------------------------------------------------
// Formatting, comparison, hashing, borrowing (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> fmt::Pointer for Rc<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&Self::as_ptr(self), f)
    }
}

impl<T: Debug + ?Sized, A: Allocator> Debug for Rc<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T: Display + ?Sized, A: Allocator> Display for Rc<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<T: PartialEq + ?Sized, A: Allocator> PartialEq for Rc<T, A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: Eq + ?Sized, A: Allocator> Eq for Rc<T, A> {}

impl<T: PartialOrd + ?Sized, A: Allocator> PartialOrd for Rc<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: Ord + ?Sized, A: Allocator> Ord for Rc<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: Hash + ?Sized, A: Allocator> Hash for Rc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized, A: Allocator> AsRef<T> for Rc<T, A> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized, A: Allocator> Borrow<T> for Rc<T, A> {
    fn borrow(&self) -> &T {
        self
    }
}

// `Rc` hands out only shared access, so there is no `AsMut`/`BorrowMut` impl.

// Neither `Rc` nor `Weak` is `Send` or `Sync` under any circumstances — see
// the "Auto-trait" note further down. Compile-time assertions via
// `static_assertions::assert_impl_all!` live in the test module.

// ---------------------------------------------------------------------------
// Default construction (sized)
// ---------------------------------------------------------------------------

impl<T: TryDefault> TryDefault for Rc<T, Global> {
    fn try_default() -> Result<Self, TryDefaultError> {
        let uninit = Self::try_new_uninit().map_err(TryDefaultError::Alloc)?;
        let value = T::try_default()?;
        // SAFETY: we just initialized the Rc with strong == 1 and weak == 0
        // (excluding the implicit ref).
        Ok(unsafe { uninit.write(value) })
    }
}

// ---------------------------------------------------------------------------
// Weak
// ---------------------------------------------------------------------------

/// A weak reference to an [`Rc`] allocation.
///
/// [`Weak<T>`] borrows the allocation without keeping it alive: upgrading a
/// [`Weak`] via [`try_upgrade`](Self::try_upgrade) yields an
/// [`Option<Rc<T>>`] that is `None` once every strong reference has been
/// dropped.
pub struct Weak<T: ?Sized, A: Allocator = Global> {
    ptr: NonNull<RcInner<T>>,
    alloc: A,
    _marker: PhantomData<RcInner<T>>,
}

impl<T: ?Sized> Weak<T, Global> {
    /// Creates a new dangling `Weak<T>` that does not point to any allocation.
    #[inline]
    pub fn new() -> Self {
        Weak {
            ptr: dangling_inner_ptr(),
            alloc: Global,
            _marker: PhantomData,
        }
    }
}

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
            ptr: dangling_inner_ptr(),
            alloc,
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized> Weak<T, Global> {
    /// Creates a new `Weak` pointer from a raw pointer previously produced by
    /// [`Weak::into_raw`].
    ///
    /// # Safety
    ///
    /// The pointer must have originated from the [`Self::into_raw`] and
    /// must still own its potential weak reference, and must point to a block of memory
    /// allocated by global allocator.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`Self::into_raw`].
    ///
    /// This function is unsafe because improper use may lead to memory
    /// unsafety, even if the returned [`Weak<T>`] is never accessed.
    #[inline]
    pub unsafe fn from_raw(p: *const T) -> Self {
        // SAFETY: caller guarantees `p` derives from `Weak::into_raw`.
        unsafe {
            // A dangling weak's raw pointer is the sentinel itself (see
            // `Weak::as_ptr`), not a projected payload address.
            let inner = if is_dangling_weak(p as *const RcInner<T>) {
                p as *mut RcInner<T>
            } else {
                data_get_ptr(p) as *mut RcInner<T>
            };
            Weak {
                ptr: NonNull::new_unchecked(inner),
                alloc: Global,
                _marker: PhantomData,
            }
        }
    }

    /// Converts a `Weak<T>` into a raw pointer.
    ///
    /// The caller takes ownership of the weak reference carried by the pointer
    /// and must eventually reconstruct a `Weak` from it (via
    /// [`from_raw`](Self::from_raw)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw(self) -> *const T {
        let me = ManuallyDrop::new(self);
        let _alloc = unsafe { ptr::read(&me.alloc) };
        // SAFETY: the allocation and reference counts are valid
        Weak::as_ptr(&me)
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Like [`from_raw`](Self::from_raw), but parameterized over the choice of
    /// allocator.
    ///
    /// # Safety
    ///
    /// The pointer must have originated from the [`Self::into_raw_with_allocator`] and
    /// must still own its potential weak reference, and must point to a block of memory
    /// allocated by `alloc`.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`Self::into_raw_with_allocator`].
    ///
    /// This function is unsafe because improper use may lead to memory
    /// unsafety, even if the returned [`Weak<T, A>`] is never accessed.
    #[inline]
    pub unsafe fn from_raw_in(p: *const T, alloc: A) -> Self {
        // SAFETY: caller guarantees validity.
        unsafe {
            // Mirror `from_raw`: a dangling weak's sentinel is not a payload
            // address, so it must be cast directly instead of offset-adjusted.
            let inner = if is_dangling_weak(p as *const RcInner<T>) {
                p as *mut RcInner<T>
            } else {
                data_get_ptr(p) as *mut RcInner<T>
            };
            Weak {
                ptr: NonNull::new_unchecked(inner),
                alloc,
                _marker: PhantomData,
            }
        }
    }

    /// Converts a `Weak<T, A>` into a raw pointer, returning it along with
    /// its allocator.
    ///
    /// The caller takes ownership of both the allocation and the allocator and
    /// must eventually reconstruct a `Weak` from them (via
    /// [`from_raw_in`](Self::from_raw_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw_with_allocator(self) -> (*const T, A) {
        let me = ManuallyDrop::new(self);
        // SAFETY: the allocation and reference counts are valid
        let ptr = Weak::as_ptr(&me);
        let alloc = unsafe { ptr::read(&me.alloc) };
        (ptr, alloc)
    }

    /// Gets a shared raw pointer to the underlying `T`.
    ///
    /// The pointer may be dangling, or may be uninitialized if strong references
    /// have all vanished. In either case, it must not be dereferenced.
    ///
    /// A weak that never referred to an allocation (from [`Weak::new`]) yields
    /// the deliberately misaligned dangling sentinel address, which can never
    /// collide with a real payload address.
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        let ptr = self.ptr.as_ptr();

        if is_dangling_weak(ptr) {
            // If the pointer is dangling, we return the sentinel directly. This cannot be
            // a valid payload address, as the payload is at least as aligned as RcInner (usize).
            ptr as *const T
        } else {
            // SAFETY: if is_dangling returns false, then the pointer is dereferenceable.
            // The payload may be dropped at this point, and we have to maintain provenance,
            // so use raw pointer manipulation.
            unsafe { &raw mut (*ptr).value }
        }
    }

    /// Gets a shared reference to the allocator backing this `Weak`.
    #[must_use]
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Determines if two `Weak` pointers point to the same allocation.
    ///
    /// This method does not deal with fat pointer metadata.
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Returns a [`WeakInner`] handle to the allocation's reference-count
    /// cells, or `None` if this handle is dangling (constructed via
    /// [`Weak::new`]).
    ///
    /// We are careful to *not* create a reference covering the "data" field, as
    /// the field may have been dropped in-place (e.g., when the last `Rc` was
    /// dropped while this `Weak` still pins the allocation). Only the two
    /// counter cells are referenced, which remain valid for the lifetime of the
    /// allocation.
    #[inline]
    pub(crate) fn inner(&self) -> Option<WeakInner<'_>> {
        if is_dangling_weak(self.ptr.as_ptr()) {
            None
        } else {
            // SAFETY: a non-dangling `Weak` always owns a live allocation whose
            // counter cells are valid for reads/writes while `self` exists. The
            // payload (`value`) may have been dropped, but we never form a
            // reference to it here.
            Some(unsafe {
                let ptr = self.ptr.as_ptr();
                WeakInner {
                    strong: &(*ptr).strong,
                    weak: &(*ptr).weak,
                }
            })
        }
    }
}

impl<T: ?Sized, A: AllocatorTryClone> Weak<T, A> {
    /// Attempts to upgrade the `Weak` reference to an [`Rc`].
    ///
    /// Returns `Ok(None)` if there are no strong (`Rc`) references left, in
    /// which case the value has been dropped.
    ///
    /// The operation is fallible because it must clone the allocator handle
    /// (via [`AllocatorTryClone`]) to attach to the new `Rc`, and counter
    /// mutation can report out-of-bounds conditions. Cloning an allocator
    /// handle is a first-class fallible operation in this framework.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if a counter would overflow
    /// `usize`, or [`TryRcError::CloneAlloc`] if cloning the allocator handle
    /// fails.
    #[inline]
    pub fn try_upgrade(&self) -> Result<Option<Rc<T, A>>, TryRcError> {
        // A dangling weak (from `Weak::new`) never referred to an allocation,
        // so it can never be upgraded. Checking this first also avoids ever
        // dereferencing the sentinel address below.
        let inner = match self.inner() {
            Some(i) => i,
            None => return Ok(None),
        };
        if inner.strong() == 0 {
            return Ok(None);
        }

        let alloc = A::try_clone(&self.alloc)?;
        // Restore the strong count. The allocation is guaranteed to stay alive
        // because this `Weak` itself pins it. We only bump strong; the weak
        // count already accounts for this handle (it was incremented when the
        // `Weak` was created via `try_downgrade`).
        inner.inc_strong().map_err(TryRcError::from)?;
        Ok(Some(Rc {
            ptr: self.ptr,
            alloc,
            _marker: PhantomData,
        }))
    }
}

impl<T: ?Sized, A: AllocatorTryClone> TryClone for Weak<T, A> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let alloc = A::try_clone(&self.alloc)?;
        // SAFETY: bumping the weak count keeps the allocation alive for the new
        // handle; no allocation is involved.
        unsafe {
            let inner = self.ptr.as_ptr();
            (*inner)
                .inc_weak()
                .map_err(|_| TryCloneError::Other("weak count out of bounds"))?;
        }
        Ok(Weak {
            ptr: self.ptr,
            alloc,
            _marker: PhantomData,
        })
    }
}

impl<T: ?Sized, A: Allocator> Drop for Weak<T, A> {
    #[inline]
    fn drop(&mut self) {
        // A dangling weak (from `Weak::new`) owns no allocation, so there is
        // nothing to decrement or free — return immediately. `inner()` folds
        // the sentinel check into one call.
        let inner = match self.inner() {
            Some(i) => i,
            None => return,
        };

        // Decrement the weak count. If this was the last reference of any kind
        // (strong already zero, and now weak hits zero), free the allocation.
        // An unbalanced decrement here would indicate a logic error, so we
        // surface it as a panic rather than silently corrupting the count.
        inner.dec_weak().expect("Weak weak count underflow");

        // Invariant: once the weak count reaches zero, the strong count must
        // also be zero (the last strong `Rc`'s drop either freed the block or
        // left it pinned by at least one `Weak`). So `weak == 0` is sufficient
        // to decide whether to deallocate.
        if is_last_ref(inner.weak()) {
            // SAFETY: `ptr` carries correct pointer metadata for `T`; the pointee
            // may be uninitialized (already dropped) but we only need its size
            // and alignment, which live in the fat pointer.
            let layout = unsafe { Layout::for_value_pointer(self.ptr.as_ptr()) };
            // SAFETY: the block was allocated with exactly this layout; the
            // header alone guarantees a non-zero size.
            unsafe {
                self.alloc.deallocate(self.ptr.cast(), layout);
            }
        }
    }
}

// `Pointer` is intentionally implemented even though std's `Weak` does not
// derive it: it is useful for debugging (printing the raw address) and costs
// nothing. The impl simply forwards to the data pointer.
impl<T: ?Sized, A: Allocator> fmt::Pointer for Weak<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&Self::as_ptr(self), f)
    }
}

// Mirrors std: printing a `Weak` would require upgrading it, which needs a
// fallible allocator clone (`A: AllocatorTryClone`). Print only the marker
// instead.
impl<T: ?Sized, A: Allocator> Debug for Weak<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("(Weak)")
    }
}

impl<T: ?Sized, A: Allocator + Default> Default for Weak<T, A> {
    #[inline]
    fn default() -> Self {
        Self::new_in(A::default())
    }
}

impl<T: ?Sized> TryDefault for Weak<T, Global> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new())
    }
}

// NOTE: See the comment above the former "Auto-trait impls for Rc" section —
// `Weak` is likewise neither `Send` nor `Sync`. Assertions in the test module.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::test_helpers::FailAlloc;
    use static_assertions::assert_not_impl_any;
    use std::string::String;

    // `Rc` and `Weak` are never `Send` or `Sync`, regardless of their type
    // parameters. Unsynchronized reference counting is inherently
    // single-threaded — sharing the refcount pointer across threads would
    // permit unsynchronized concurrent mutation of the counters.
    assert_not_impl_any!(Rc<i32>: Send, Sync);
    assert_not_impl_any!(Rc<String>: Send, Sync);
    assert_not_impl_any!(Rc<&'static ()>: Send, Sync);
    assert_not_impl_any!(Weak<i32>: Send, Sync);
    assert_not_impl_any!(Weak<String>: Send, Sync);
    assert_not_impl_any!(Weak<&'static ()>: Send, Sync);

    #[test]
    fn basic_construction_and_deref() {
        let rc = Rc::try_new(42).unwrap();
        assert_eq!(*rc, 42);
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);
    }

    #[test]
    fn clone_shares_allocation() {
        let rc = Rc::try_new(String::from("hello")).unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
        assert_eq!(&*rc2, "hello");
    }

    #[test]
    fn drop_frees_at_last_strong() {
        struct Counted(Cell<i32>);
        impl Drop for Counted {
            fn drop(&mut self) {
                *self.0.get_mut() += 1;
            }
        }
        let counter = Cell::new(0i32);
        let rc = Rc::try_new(Counted(counter)).unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(rc.0.get(), 0);
        drop(rc);
        assert_eq!(rc2.0.get(), 0);
        drop(rc2);
        // Can't check counter directly since it was moved into Counted,
        // but if we got here without UB, the drop ran exactly once.
    }

    #[test]
    fn weak_does_not_keep_alive() {
        let rc = Rc::try_new(1).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_upgrade_restores_strong() {
        let rc = Rc::try_new(7).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let upgraded = weak.try_upgrade().unwrap().unwrap();
        assert_eq!(Rc::strong_count(&upgraded), 2);
        assert_eq!(*upgraded, 7);
        drop(upgraded);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn oom_returns_error_and_gives_back_value() {
        let res: Result<Rc<String, FailAlloc>, (String, AllocError)> =
            Rc::try_new_give_back_in(String::from("oom"), FailAlloc);
        let (given_back, err) = res.unwrap_err();
        assert_eq!(given_back, "oom");
        let _ = err;
    }

    #[test]
    fn try_new_with_fail_alloc_errors() {
        let res: Result<Rc<i32, FailAlloc>, AllocError> = Rc::try_new_in(5, FailAlloc);
        assert!(res.is_err());
    }

    #[test]
    fn into_raw_roundtrip() {
        let rc = Rc::try_new(99).unwrap();
        let raw = Rc::into_raw(rc);
        assert_eq!(unsafe { *raw }, 99);
        let rc = unsafe { Rc::from_raw(raw) };
        assert_eq!(*rc, 99);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn weak_dangling_into_raw_roundtrip() {
        // A dangling weak carries no allocation; its raw pointer is the
        // misaligned sentinel, and reconstituting it yields another dangling
        // weak that behaves identically.
        let w: Weak<u32, Global> = Weak::new();
        let raw = Weak::into_raw(w);
        assert_ne!(raw.addr(), 0);
        let w = unsafe { Weak::from_raw(raw) };
        // `as_ptr` on a dangling weak returns the sentinel itself — exercising
        // the same branch that `from_raw` relies on to detect the sentinel.
        assert_eq!(w.as_ptr().addr(), raw.addr());
        assert!(w.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn multiple_weak_refs_all_release() {
        let rc = Rc::try_new(3).unwrap();
        let w1 = Rc::try_downgrade(&rc).unwrap();
        let w2 = w1.try_clone().unwrap();
        assert_eq!(Rc::weak_count(&rc), 2);
        drop(rc);
        assert!(w1.try_upgrade().unwrap().is_none());
        assert!(w2.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn try_clone_matches_clone() {
        let rc = Rc::try_new(11).unwrap();
        let cloned = rc.try_clone().unwrap();
        assert!(Rc::ptr_eq(&rc, &cloned));
        assert_eq!(Rc::strong_count(&rc), 2);
    }

    #[test]
    fn comparisons_forward_to_payload() {
        let a = Rc::try_new([1, 2, 3]).unwrap();
        let b = Rc::try_new([1, 2, 3]).unwrap();
        let c = Rc::try_new([1, 2, 4]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a < c);
    }

    #[test]
    fn as_ref_and_borrow_work() {
        let rc = Rc::try_new(5u32).unwrap();
        let r: &u32 = rc.as_ref();
        assert_eq!(*r, 5);
        let b: &u32 = Borrow::borrow(&rc);
        assert_eq!(*b, 5);
    }

    #[test]
    fn weak_debug_prints_marker() {
        // Like std, Debug prints only `(Weak)` — it must not require
        // upgrading (which would demand `A: AllocatorTryClone`).
        let rc = Rc::try_new(1).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(std::format!("{weak:?}"), "(Weak)");
        drop(rc);
        assert_eq!(std::format!("{weak:?}"), "(Weak)");
    }

    // -----------------------------------------------------------------------
    // ?Sized tests
    // -----------------------------------------------------------------------

    #[test]
    fn unsized_byte_slice_construction_and_deref() {
        let arr = [10u8, 20, 30, 40];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        assert_eq!(&*rc, [10, 20, 30, 40]);
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);
    }

    #[test]
    fn unsized_byte_slice_clone_shares_allocation() {
        let arr = [1u8, 2, 3];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
        assert_eq!(&*rc2, [1, 2, 3]);
    }

    #[test]
    fn unsized_byte_slice_weak_lifecycle() {
        let arr = [65u8, 66, 67];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        drop(rc);
        assert!(weak.try_upgrade().unwrap().is_none());
    }

    // Note: Direct unsized coercion from Rc<Concrete> to Rc<dyn Trait> is not
    // supported by our implementation (the PhantomData<RcInner<T>> field
    // prevents the compiler's built-in unsized coercion). Trait objects are
    // still fully supported at the type level (Rc<dyn T> compiles, derefs,
    // clones, downgrades, and deallocates correctly); construction simply
    // requires going through a raw-pointer roundtrip or a future
    // `Rc::try_from_any` API. The slice and str tests below exercise the same
    // fat-pointer offset/layout/deallocation code paths.

    #[test]
    fn unsized_str_construction() {
        let s = String::from("hello world");
        let rc: Rc<str> = Rc::try_from_str(s.as_str()).unwrap();
        assert_eq!(&*rc, "hello world");
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    // An "absurd" payload — one whose size, combined with the two-counter
    // header, overflows an addressable block — must be rejected as a logic-level
    // invariant violation (`TryCloneError::Other`), NOT as transient OOM
    // (`Alloc(AllocError)`). The distinction matters: an OOM-style error invites
    // a retry/backoff loop that would spin forever on a permanently unsatisfiable
    // request. We exercise the layout helper directly because no real `&[u8]` can
    // carry such a length.
    #[test]
    fn absurd_payload_layout_reports_other_not_oom() {
        let usize_size = size_of::<usize>();
        let max_size = isize::MAX as usize - usize_size + 1;
        // A hypothetical Rust tcype requires the size is divisible by its alignment
        assert_eq!(max_size % usize_size, 0);
        let huge = Layout::from_size_align(max_size, usize_size)
            .expect("payload layout alone is representable");
        let err = rc_inner_layout_for_value_layout(huge)
            .expect_err("absurd payload must fail layout computation");
        // The helper surfaced a real layout error (the overflow).
        let _ = err;
        // Mirror the exact mapping used by `try_clone_from_ref_in`: the overflow
        // becomes `Other`, never `Alloc(AllocError)`. Asserting on both arms of
        // the match proves the absurd case cannot be mistaken for transient OOM.
        let mapped = TryCloneError::Other("Rc payload layout overflow");
        assert!(matches!(mapped, TryCloneError::Other(_)));
        assert!(!matches!(mapped, TryCloneError::Alloc(_)));
    }

    #[test]
    fn unsized_into_raw_roundtrip_slice() {
        let arr = [7u8, 8, 9];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let raw = Rc::into_raw(rc);
        let slice: &[u8] = unsafe { &*raw };
        assert_eq!(slice.len(), 3);
        assert_eq!(slice, [7, 8, 9]);
        let rc = unsafe { Rc::from_raw(raw) };
        assert_eq!(&*rc, [7, 8, 9]);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn unsized_as_ptr_and_borrow() {
        let arr = [10u8, 20];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let p: *const [u8] = Rc::as_ptr(&rc);
        let slice: &[u8] = unsafe { &*p };
        assert_eq!(slice.len(), 2);
        assert_eq!(slice, [10, 20]);
        let b: &[u8] = Borrow::borrow(&rc);
        assert_eq!(b, &[10, 20]);
    }

    // A value whose destructor panics. Used to verify that the panic-aware
    // guard in `Rc::drop` still frees the block when no `Weak` remains.
    struct PanickingDrop;
    impl Drop for PanickingDrop {
        fn drop(&mut self) {
            panic!("intentional panic in value destructor");
        }
    }

    #[test]
    #[should_panic(expected = "intentional panic")]
    fn panic_in_last_strong_no_weak_still_frees() {
        // No weak references: when the last strong drops and the value's
        // destructor panics, the guard must still deallocate the block. If it
        // leaked, Miri / leak detectors would catch it; here we at least prove
        // the code path runs without double-free or abort.
        let _rc = Rc::try_new(PanickingDrop).unwrap();
    }

    #[test]
    fn panic_in_last_strong_with_weak_keeps_block() {
        // A live `Weak` exists: the guard must NOT free the block (weak > 0),
        // leaving the `Weak` to release it later. Catch the panic so we can
        // verify the weak handle still sees a defunct-but-alive allocation.
        let rc = Rc::try_new(PanickingDrop).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(Rc::weak_count(&rc), 1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(rc); // panics inside value's Drop; guard sees weak >= 1, skips free
        }));
        assert!(result.is_err(), "expected the value's Drop to panic");
        // The weak handle must still be valid (block not freed): upgrading
        // returns None because strong is 0, but accessing the weak does not
        // trigger a use-after-free. Dropping it now frees the block.
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
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
    fn weak_new_dangling_upgrades_to_none() {
        // A freshly constructed `Weak` refers to no allocation, so upgrading it
        // must always yield `None`.
        let w: Weak<i32> = Weak::new();
        assert!(w.try_upgrade().unwrap().is_none());
    }

    #[test]
    fn weak_new_dropping_frees_nothing() {
        // Dropping a dangling weak must not touch the allocator at all. We
        // verify this indirectly: constructing and dropping several dangling
        // weaks (including for unsized types) leaves the process stable — there
        // is no deallocation path to corrupt.
        let _ = Weak::<i32>::new();
        let _ = Weak::<str>::new();
        let _ = Weak::<[u8]>::new();
        // Explicitly drop at end of scope; no panic / UB means we passed.
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

    // -----------------------------------------------------------------------
    // try_unwrap / try_unwrap_give_back /
    // unwrap_or_try_clone / unwrap_or_try_clone_give_back
    // All four consume the `Rc` by value.
    // -----------------------------------------------------------------------

    #[test]
    fn try_unwrap_sole_owner_moves_out() {
        // Sole strong, no weaks: the payload is moved out and the block freed.
        let rc = Rc::try_new(String::from("hello")).unwrap();
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);
        let val = Rc::try_unwrap(rc).unwrap();
        assert_eq!(val, "hello");
    }

    #[test]
    fn try_unwrap_shared_returns_handle_back() {
        // Shared ownership: cannot move out, so the whole input handle is
        // handed back as the Err variant and the allocation stays intact.
        let rc = Rc::try_new(42i32).unwrap();
        let rc2 = rc.try_clone().unwrap();
        let returned = Rc::try_unwrap(rc).unwrap_err();
        // The returned handle still points at the same live allocation.
        assert!(Rc::ptr_eq(&returned, &rc2));
        assert_eq!(*returned, 42);
        assert_eq!(Rc::strong_count(&returned), 2);
        // Dropping both handles frees the block exactly once.
        drop(returned);
        drop(rc2);
    }

    #[test]
    fn try_unwrap_with_live_weak_returns_handle_back() {
        // Live weak reference: the block must stay reachable, so we cannot
        // move out even though there is only one strong reference.
        let rc = Rc::try_new(7u8).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let returned = Rc::try_unwrap(rc).unwrap_err();
        assert_eq!(*returned, 7);
        assert_eq!(Rc::strong_count(&returned), 1);
        // Dropping the strong leaves the weak pointing at a dead block.
        drop(returned);
        assert!(weak.try_upgrade().unwrap().is_none());
        drop(weak);
    }

    #[test]
    fn try_unwrap_preserves_drop_semantics() {
        // Moving out via `ptr::read` bypasses the normal drop path, so the
        // payload's destructor must run exactly once — on the owned value the
        // caller receives, not inside the freed block.
        #[derive(Debug)]
        struct Counted(Cell<i32>);
        impl Drop for Counted {
            fn drop(&mut self) {
                *self.0.get_mut() += 1;
            }
        }
        let counter = Cell::new(0i32);
        let rc = Rc::try_new(Counted(counter)).unwrap();
        let counted = Rc::try_unwrap(rc).unwrap();
        // Still owned by us; not yet dropped.
        assert_eq!(counted.0.get(), 0);
        drop(counted);
        // Reaching here without UB means the destructor ran exactly once.
    }

    #[test]
    fn unwrap_or_try_clone_sole_owner_moves_out() {
        // Sole owner: consumes the handle and moves the payload out directly.
        let rc = Rc::try_new(1234i32).unwrap();
        let val = Rc::unwrap_or_try_clone(rc).unwrap();
        assert_eq!(val, 1234);
    }

    #[test]
    fn unwrap_or_try_clone_shared_clones_value() {
        // Shared ownership: cannot move out, so the value is cloned instead.
        // Both handles remain valid and point at the same allocation.
        let rc = Rc::try_new(99i32).unwrap();
        let rc2 = rc.try_clone().unwrap();
        let val = Rc::unwrap_or_try_clone(rc).unwrap();
        assert_eq!(val, 99);
        // The original allocation is untouched.
        assert_eq!(*rc2, 99);
        assert_eq!(Rc::strong_count(&rc2), 1);
        drop(rc2);
    }

    #[test]
    fn unwrap_or_try_clone_give_back_sole_owner_moves_out() {
        // Sole owner: consumes the handle and moves the payload out directly.
        let rc = Rc::try_new(321i32).unwrap();
        let val = Rc::unwrap_or_try_clone_give_back(rc).unwrap();
        assert_eq!(val, 321);
    }

    #[test]
    fn unwrap_or_try_clone_give_back_shared_clones_value() {
        // Sole ownership: cannot move out, so the value is cloned instead.
        // Both handles remain valid and point at the same allocation.
        let rc = Rc::try_new(88i32).unwrap();
        let rc2 = rc.try_clone().unwrap();
        let val = Rc::unwrap_or_try_clone_give_back(rc).unwrap();
        assert_eq!(val, 88);
        // The original allocation is untouched.
        assert_eq!(*rc2, 88);
        assert_eq!(Rc::strong_count(&rc2), 1);
        drop(rc2);
    }

    // -----------------------------------------------------------------------
    // is_unique / get_mut / try_make_mut
    // -----------------------------------------------------------------------

    #[test]
    fn is_unique_true_only_when_no_other_refs() {
        let rc = Rc::try_new(1).unwrap();
        assert!(Rc::is_unique(&rc));

        let weak = Rc::try_downgrade(&rc).unwrap();
        // A live Weak disqualifies uniqueness even though strong == 1.
        assert!(!Rc::is_unique(&rc));
        drop(weak);
        assert!(Rc::is_unique(&rc));

        let rc2 = rc.try_clone().unwrap();
        assert!(!Rc::is_unique(&rc));
        drop(rc2);
        assert!(Rc::is_unique(&rc));
    }

    #[test]
    fn get_mut_requires_no_weak() {
        let mut rc = Rc::try_new(String::from("hi")).unwrap();
        *Rc::get_mut(&mut rc).unwrap() += "!";
        assert_eq!(*rc, "hi!");

        // A single Weak now blocks get_mut (matching std semantics).
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert!(Rc::get_mut(&mut rc).is_none());
        drop(weak);
        assert!(Rc::get_mut(&mut rc).is_some());
    }

    #[test]
    fn make_mut_unique_mutates_in_place() {
        let mut rc = Rc::try_new(5u32).unwrap();
        let addr = Rc::as_ptr(&rc) as usize;
        *Rc::try_make_mut(&mut rc).unwrap() += 1;
        assert_eq!(*rc, 6);
        // No reallocation happened — same payload address.
        assert_eq!(Rc::as_ptr(&rc) as usize, addr);
    }

    #[test]
    fn make_mut_shared_clones_and_disassociates() {
        let mut data = Rc::try_new(5i32).unwrap();
        let other = data.try_clone().unwrap();

        let old_addr = Rc::as_ptr(&data) as usize;
        assert_eq!(Rc::strong_count(&data), 2);

        *Rc::try_make_mut(&mut data).unwrap() += 1;

        // `data` moved to a fresh allocation holding the mutated clone.
        assert_eq!(*data, 6);
        assert_ne!(Rc::as_ptr(&data) as usize, old_addr);
        assert_eq!(Rc::strong_count(&data), 1);
        // The original owner is untouched and still valid.
        assert_eq!(*other, 5);
        assert_eq!(Rc::strong_count(&other), 1);
        assert!(!Rc::ptr_eq(&data, &other));
    }

    #[test]
    fn make_mut_with_weak_steals_without_cloning() {
        // Only weak references remain: the value must be MOVED into a fresh
        // block (not cloned), and the surviving Weaks become dangling.
        struct Movable<'a>(&'a Cell<i32>, String);

        impl TryClone for Movable<'_> {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                panic!("this test should not move");
            }
        }

        impl Drop for Movable<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        // External counter so we can observe the destructor across the move
        // (a counter embedded in the payload would itself be relocated).
        let counter_cell = Cell::new(0i32);
        let mut data = Rc::try_new(Movable(&counter_cell, String::from("payload"))).unwrap();
        let weak = Rc::try_downgrade(&data).unwrap();
        let old_addr = Rc::as_ptr(&data) as usize;

        Rc::try_make_mut(&mut data).unwrap().1.push('x');

        assert_eq!(data.1.as_str(), "payloadx");
        assert_ne!(Rc::as_ptr(&data) as usize, old_addr);
        assert_eq!(Rc::strong_count(&data), 1);
        // The weak reference is now detached from any live value.
        assert!(weak.try_upgrade().unwrap().is_none());
        // Dropping the new sole owner runs the destructor exactly once.
        drop(data);
        assert_eq!(counter_cell.get(), 1);
    }

    #[test]
    fn make_mut_unsized_slice_shared_clones() {
        let arr = [1u8, 2, 3];
        let mut rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let other = rc.try_clone().unwrap();

        let old_addr = Rc::as_ptr(&rc).cast::<u8>() as usize;

        let m = Rc::try_make_mut(&mut rc).unwrap();
        m[0] = 99;

        assert_eq!(&*rc, [99, 2, 3]);
        assert_ne!(Rc::as_ptr(&rc).cast::<u8>() as usize, old_addr);
        // Original slice untouched.
        assert_eq!(&*other, [1, 2, 3]);
    }

    // -----------------------------------------------------------------------
    // Raw-pointer provenance / sentinel semantics
    // -----------------------------------------------------------------------

    /// `Rc::as_ptr` must hand back a pointer with write provenance rooted in
    /// the live allocation, so that a sole owner can legally cast it to
    /// `*mut T` and mutate the payload in place (manual-write scenarios such as
    /// `ptr::write` through the raw pointer). Under Miri this proves the data
    /// word was derived from the allocation rather than fabricated; on native
    /// builds it proves the roundtrip works end-to-end.
    #[test]
    fn rc_as_ptr_write_provenance_manual_write() {
        let rc = Rc::try_new(42u64).unwrap();
        let p: *const u64 = Rc::as_ptr(&rc);
        // Cast away constness — only sound because we hold the sole strong
        // reference (strong == 1, weak == 0) and the pointer's provenance
        // covers the payload slot.
        let mut_p = p as *mut u64;
        unsafe {
            ptr::write(mut_p, 7);
        }
        assert_eq!(*rc, 7);
        // Overwrite again via the same pointer to confirm repeated use is fine.
        unsafe {
            ptr::write_volatile(mut_p, 9);
        }
        assert_eq!(*rc, 9);
    }

    /// Same guarantee for unsized payloads: the fat pointer returned by
    /// `Rc::as_ptr` must be writable through its data word while metadata
    /// (the slice length) is preserved.
    #[test]
    fn rc_as_ptr_write_provenance_unsized_slice() {
        let arr = [1u8, 2, 3];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let p: *const [u8] = Rc::as_ptr(&rc);
        let mut_p = p as *mut [u8];
        unsafe {
            (*mut_p)[1] = 20;
        }
        assert_eq!(&*rc, [1, 20, 3]);
    }

    /// A `Weak` produced by [`Weak::new`] never referred to an allocation, so
    /// `as_ptr` must return the deliberately misaligned dangling sentinel
    /// (`usize::MAX`) rather than deriving one from a real block. No valid
    /// allocation address can equal the sentinel, which makes "was this weak
    /// ever attached?" decidable by pure pointer comparison.
    #[test]
    fn weak_as_ptr_dangling_returns_sentinel() {
        let w: Weak<i32> = Weak::new();
        let p = w.as_ptr();
        assert_eq!(p.addr(), usize::MAX);
        // The sentinel is misaligned relative to any non-ZST payload, so it
        // can never alias a real allocation.
        assert_ne!(p.addr() % align_of::<i32>(), 0);
    }

    /// Once the last strong reference drops, a still-live `Weak` keeps the
    /// block pinned but the payload is gone: `as_ptr` must NOT return the
    /// sentinel (this weak WAS attached to a real allocation), and the
    /// address must not be dereferenced. We verify the address differs from
    /// the sentinel and that upgrade reports no strong references.
    #[test]
    fn weak_as_ptr_after_last_strong_is_not_sentinel() {
        let rc = Rc::try_new(1i32).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        drop(rc);
        let p = weak.as_ptr();
        // This weak did refer to a real allocation, so its data word is the
        // genuine (now dead) payload address, not the sentinel.
        assert_ne!(p.addr(), usize::MAX);
        assert!(weak.try_upgrade().unwrap().is_none());
    }
}
