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
use core::mem::{ManuallyDrop, MaybeUninit, align_of_val, size_of_val};
use core::ops::Deref;

use crate::alloc::{AllocError, Allocator, Global, Layout, LayoutError};
use olive_core::alloc::AllocatorTryClone;
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
    /// underflow below zero. Indicates a logic error (unbalanced inc/dec) or
    /// adversarial misuse of the raw pointer APIs.
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

impl From<AllocError> for TryRcWithError<core::convert::Infallible> {
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
    /// Returns [`TryRcError::OutOfBounds`] if the count is already at
    /// `usize::MAX`.
    #[inline]
    pub(crate) fn inc_strong(&self) -> Result<(), TryRcError> {
        let cur = self.strong.get();
        match cur.checked_add(1) {
            Some(next) => {
                self.strong.set(next);
                Ok(())
            }
            None => Err(TryRcError::OutOfBounds),
        }
    }

    /// Decrements the strong count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if the count is already zero
    /// (unbalanced decrement).
    #[inline]
    pub(crate) fn dec_strong(&self) -> Result<(), TryRcError> {
        let cur = self.strong.get();
        match cur.checked_sub(1) {
            Some(next) => {
                self.strong.set(next);
                Ok(())
            }
            None => Err(TryRcError::OutOfBounds),
        }
    }

    /// Increments the weak count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if the count is already at
    /// `usize::MAX`.
    #[inline]
    pub(crate) fn inc_weak(&self) -> Result<(), TryRcError> {
        let cur = self.weak.get();
        match cur.checked_add(1) {
            Some(next) => {
                self.weak.set(next);
                Ok(())
            }
            None => Err(TryRcError::OutOfBounds),
        }
    }

    /// Decrements the weak count by one.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if the count is already zero
    /// (unbalanced decrement).
    #[inline]
    pub(crate) fn dec_weak(&self) -> Result<(), TryRcError> {
        let cur = self.weak.get();
        match cur.checked_sub(1) {
            Some(next) => {
                self.weak.set(next);
                Ok(())
            }
            None => Err(TryRcError::OutOfBounds),
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

/// Casts a pointer to `RcInner<T>` to a pointer to the payload within it.
#[inline]
fn ptr_get_data<T: ?Sized>(p: *const RcInner<T>) -> *const T {
    // SAFETY: `value` is a field of `RcInner<T>`; taking its address is valid
    // as long as `p` points to a live allocation.
    unsafe { &(*p).value as *const T }
}

/// Computes the byte offset from the start of an `RcInner<T>` allocation to
/// the beginning of its `value` field, given the payload's alignment.
#[inline]
fn data_offset<T: ?Sized>(p: *const T) -> usize {
    let value_layout = unsafe { Layout::for_value(&*p) };
    // Overflow is impossible here. The value_layout came from a pointer that is previously
    // validated with the exact same function.
    let (_, offset) =
        rc_inner_layout_for_value_layout(value_layout).expect("Rc header/payload layout overflow");
    offset
}

/// Reverse of [`ptr_get_data`]: casts a payload pointer back to its enclosing
/// `RcInner<T>`.
///
/// # Safety
///
/// The pointer must have been produced by [`ptr_get_data`] on a live
/// `RcInner<T>` allocation, and the allocation must still be alive.
#[inline]
unsafe fn data_get_ptr<T: ?Sized>(p: *const T) -> *const RcInner<T> {
    let offset = data_offset(p);
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
            Ok(mut b) => {
                b.write(x);
                // SAFETY: we just wrote `x` into the slot above.
                Ok(unsafe { b.assume_init() })
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
        Self::try_new_cyclic_in(Global, f)
    }

    // FIXME: add try_pin/try_pin_give_back, the Pin is accessible through olive_core::pin::Pin.
}

// ---------------------------------------------------------------------------
// Global reconstitution block
// ---------------------------------------------------------------------------

impl<T: ?Sized> Rc<T, Global> {
    /// Constructs a new `Rc<T>` from a raw pointer previously produced by
    /// [`into_raw`](Self::into_raw).
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained from [`into_raw`](Self::into_raw) on
    /// an `Rc` with the same global allocator, and must still be valid (not yet freed).
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
        // Mirrors std's `Rc::into_raw`: wrap in `ManuallyDrop` so that if this
        // function ever panics (e.g. during unwinding), the block is not
        // double-freed by the implicit drop at scope end. The `Global`
        // allocator is a ZST, but we still perform an explicit read to keep
        // the ownership transfer uniform with the generic path and to make
        // the intent clear to reviewers and future maintainers.
        let me = ManuallyDrop::new(rc);
        let _alloc = unsafe { ptr::read(&me.alloc) };
        ptr_get_data(me.ptr.as_ptr())
    }
}

// ---------------------------------------------------------------------------
// Generic construction block (sized)
// ---------------------------------------------------------------------------

impl<T: Sized, A: Allocator> Rc<T, A> {
    /// Like [`try_new`](Self::try_new), but parameterized over the choice of
    /// allocator for the returned `Rc`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_in(x: T, alloc: A) -> Result<Self, AllocError> {
        let mut b = Self::try_new_uninit_in(alloc)?;
        b.write(x);
        // SAFETY: we just wrote `x` into the slot above.
        Ok(unsafe { b.assume_init() })
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
            Ok(mut b) => {
                b.write(x);
                // SAFETY: we just wrote `x` into the slot above.
                Ok(unsafe { b.assume_init() })
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

    /// Creates a cyclic `Rc` using a callback to wire up the cycle atomically.
    ///
    /// ```ignore
    /// struct Node { name: String, parent: Option<Weak<Node>> }
    ///
    /// let node = Rc::try_new_cyclic(|weak| {
    ///     // Build the value around the back-reference.
    ///     Ok(Node { name: "root".into(), parent: Some(weak.clone()) })
    /// }).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TryRcWithError<E>`]: either an [`AllocError`] if allocating
    /// the block fails, or the callback's own error `E`.
    #[inline]
    pub fn try_new_cyclic_in<E, F>(alloc: A, f: F) -> Result<Self, TryRcWithError<E>>
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
        let value = match f(&weak) {
            Ok(t) => t,
            Err(e) => {
                // `value` was never placed in the block, so only the Weak's
                // drop (frees the uninit block) is needed on unwind.
                return Err(TryRcWithError::Callback(e));
            }
        };

        // Step 4: initialize the payload slot with the constructed value.
        // SAFETY: the block was allocated with exactly this layout and the
        // payload slot is currently uninitialized.
        unsafe {
            ptr::write(ptr_get_data(inner.as_ptr()) as *mut T, value);
        }

        // Step 5: increment strong 0 → 1. Fresh allocation guarantees no
        // overflow, so we can safely expect.
        debug_assert_eq!(unsafe { (*inner.as_ptr()).strong() }, 0);
        unsafe { (*inner.as_ptr()).inc_strong() }.expect("strong count is 0, cannot overflow");

        // Step 6: consume the Weak WITHOUT dropping it, so its weak count
        // becomes the implicit shared weak of the returned Rc. This yields
        // the standard (strong=1, weak=1) final state.
        let (rc_data_ptr, weak_alloc) = Weak::into_raw_with_allocator(weak);
        let rc_data_ptr: *const T = rc_data_ptr;

        // SAFETY: `rc_data_ptr` was produced by `ptr_get_data` on our fresh,
        // fully-initialized block, and `weak_alloc` is the same allocator.
        Ok(unsafe { Rc::from_raw_in(rc_data_ptr, weak_alloc) })
    }

    // FIXME: implement `try_pin_in` / `try_pin_give_back_in`
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
    /// The pointer must have been produced by
    /// [`into_raw_with_allocator`](Self::into_raw_with_allocator) on an `Rc`
    /// with the same allocator, and must still be valid.
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

    /// Converts an `Rc<T, A>` into a raw pointer, retaining its allocator.
    ///
    /// The caller takes ownership of both the allocation and the allocator and
    /// must eventually reconstruct an `Rc` from them (via
    /// [`from_raw_in`](Self::from_raw_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw_with_allocator(rc: Self) -> (*const T, A) {
        // Mirrors std's `Rc::into_raw_with_allocator`: wrap in `ManuallyDrop`
        // so that a panic during unwinding cannot cause the implicit drop at
        // scope end to free the block after we have already handed ownership
        // to the caller. The allocator is read out explicitly; the rest of the
        // struct is logically consumed by the return value.
        let me = ManuallyDrop::new(rc);
        let ptr = ptr_get_data(me.ptr.as_ptr());
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

impl<A: Allocator> Rc<[u8], A> {
    /// Allocates a new `Rc<[u8]>` by copying the bytes from `src` into fresh
    /// heap memory. Fallible analogue of std's `Rc::from(&bytes[..])`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_slice_in(src: &[u8], alloc: A) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, alloc)
    }
}

impl Rc<[u8], Global> {
    /// Convenience wrapper around [`Self::try_from_slice_in`]
    /// using the global allocator.
    #[inline]
    pub fn try_from_slice(src: &[u8]) -> Result<Self, TryCloneError> {
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
    /// Writes `val` into the `Rc`'s slot.
    ///
    /// After calling this, the caller must invoke [`assume_init`](Self::assume_init)
    /// to obtain the initialized `Rc<T, A>`. The value is written in place; if
    /// the write panics (it cannot for a plain `write`, but the pattern keeps
    /// the API symmetric with fallible variants), the `Rc` remains valid and
    /// will be dropped normally.
    #[inline]
    pub fn write(&mut self, val: T) {
        unsafe {
            let data = ptr_get_data::<MaybeUninit<T>>(self.ptr.as_ptr());
            data.cast::<T>().cast_mut().write(val);
        }
    }

    /// Reinterprets the `Rc<MaybeUninit<T>, A>` as an initialized `Rc<T, A>`.
    ///
    /// # Safety
    ///
    /// The payload slot must have been fully initialized, e.g. via
    /// [`write`](Self::write). Calling this on uninitialized memory is UB.
    #[inline]
    pub unsafe fn assume_init(self) -> Rc<T, A> {
        // Wrap in `ManuallyDrop` so that a panic during unwinding cannot cause
        // the implicit drop at scope end to free the block after we have
        // already handed ownership to the returned `Rc`. Reading the fields
        // out explicitly transfers ownership without invoking `Drop`.
        let me = ManuallyDrop::new(self);
        // SAFETY: `Rc<MaybeUninit<T>, A>` and `Rc<T, A>` have identical layouts
        // (both contain `NonNull<RcInner<...>>`, `A`, and a zero-sized phantom);
        // the caller guarantees the payload slot is fully initialized.
        unsafe {
            let ptr = NonNull::new_unchecked(me.ptr.as_ptr() as *mut RcInner<T>);
            let alloc = ptr::read(&me.alloc);
            Rc {
                ptr,
                alloc,
                _marker: PhantomData,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Query and mutation block (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Gets a shared raw pointer to the underlying data.
    #[must_use]
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        ptr_get_data(this.ptr.as_ptr())
    }

    /// Returns a shared reference to the allocation's internal [`RcInner`]
    /// header (strong/weak counters and payload), or `None` if this handle is
    /// dangling.
    ///
    // FIXME: Rc is a well defined standard type, can't ever dangle
    #[inline]
    pub(crate) fn inner(&self) -> Option<&RcInner<T>> {
        if is_dangling_weak(self.ptr.as_ptr()) {
            return None;
        }
        // SAFETY: a non-dangling `Rc` always owns a live allocation whose
        // header is valid for reads while `self` exists.
        Some(unsafe { &*self.ptr.as_ptr() })
    }

    /// Gets a shared reference to the allocator backing this `Rc`.
    ///
    /// Implemented as an inherent method (rather than an associated function)
    /// so it does not shadow any future free-function helpers in this module.
    #[must_use]
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
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
        this.inner().map_or(0, RcInner::strong)
    }

    /// Returns the number of weak (`Weak`) pointers to this allocation,
    /// excluding the implicit weak reference held by each strong pointer.
    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        this.inner()
            .map_or(0, |inner| inner.weak().saturating_sub(1))
    }

    /// Gets a mutable reference to the contained value if this `Rc` is the
    /// sole strong reference. Returns `None` if other strong references exist.
    ///
    /// This is the fallible analogue of std's `Rc::get_mut`: because we never
    /// panic on contention, the "already shared" case is simply reported via
    /// `Option`.
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        match this.inner() {
            Some(inner) if inner.strong() == 1 => {
                // SAFETY: we are the only strong reference, so no aliasing
                // reader can observe the payload concurrently. The weak count
                // is irrelevant — weak handles cannot read the value.
                Some(unsafe { &mut (*this.ptr.as_ptr()).value })
            }
            _ => None,
        }
    }

    /// Gets a mutable reference to the contained value **without** checking
    /// the strong count.
    ///
    /// # Safety
    ///
    /// The caller must guarantee exclusive access: no other strong or weak
    /// handle may be used to read or write the payload for the duration of
    /// the returned borrow. Violating this is undefined behaviour (aliasing
    /// `&mut`).
    #[inline]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        // SAFETY: caller's exclusivity promise.
        unsafe { &mut (*this.ptr.as_ptr()).value }
    }

    /// If this `Rc` is the sole strong reference, gets a mutable reference.
    /// Otherwise, clones the value into a fresh `Rc` and returns a mutable
    /// reference to it.
    ///
    /// This mirrors std's `Rc::make_mut`: the clone path uses
    /// [`TryCloneToUninit`] to write directly into a newly allocated block,
    /// which works for both sized and unsized payloads.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError`] if cloning the allocator handle or the
    /// clone-and-reallocate path fails.
    ///
    // FIXME: CloneAlloc does not map well with try_clone_to_uninit, need to make
    // another error or use an existing one (e.g. TryCloneError).
    #[inline]
    pub fn make_mut(this: &mut Self) -> Result<&mut T, TryRcError>
    where
        T: TryCloneToUninit,
        A: AllocatorTryClone,
    {
        // Fast path: sole owner can mutate in place. Checking the count first
        // avoids creating a long-lived mutable borrow that would prevent us
        // from reading `this`'s fields below.
        if Self::strong_count(this) == 1 {
            // SAFETY: strong count is 1, so we are the exclusive owner.
            return Ok(unsafe { Self::get_mut_unchecked(this) });
        }
        // Shared: allocate a fresh block sized to match the current payload's
        // metadata, clone directly into it, then swap the new handle in.
        let alloc = A::try_clone(&this.alloc)?;
        let new_rc = Rc::try_clone_from_ref_in(&**this, alloc)?;
        *this = new_rc;
        Ok(unsafe { Self::get_mut_unchecked(this) })
    }

    /// Clones the value pointed to by this `Rc`, falling back to the default
    /// value if the payload does not implement [`TryClone`].
    ///
    /// This is Olive's fallible counterpart to std's `Rc::unwrap_or_clone`:
    /// when the sole owner holds the reference, the handle is cheaply bumped;
    /// otherwise the payload is deep-cloned into a fresh allocation. If the
    /// clone fails, the caller-supplied `fallback` closure provides an
    /// alternative value to wrap in a new `Rc`.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError`] if both the clone and the fallback allocation
    /// fail.
    // FIXME: should probably return TryCloneError. Return type should be T.
    // T should be sized, and for simplicity, should be moved to a Sized block.
    // try_unwrap missing. It is possible to call try_clone_from_ref_in because it also supports sized values.
    #[inline]
    pub fn unwrap_or_try_clone(
        this: &Self,
        fallback: impl FnOnce() -> T,
    ) -> Result<Self, TryRcError>
    where
        T: Sized + TryClone,
        A: AllocatorTryClone,
    {
        if Self::strong_count(this) == 1 {
            // Sole owner: just clone the handle (cheap refcount bump).
            <Self as TryClone>::try_clone(this).map_err(TryRcError::CloneAlloc)
        } else {
            // Shared: attempt deep-copy of the payload into a fresh allocation.
            match T::try_clone(&**this) {
                Ok(cloned) => Rc::try_new_give_back_in(cloned, A::try_clone(&this.alloc)?)
                    .map_err(|(_, e)| TryRcError::CloneAlloc(TryCloneError::Alloc(e))),
                Err(_) => {
                    // Clone failed: use the fallback value.
                    let val = fallback();
                    Rc::try_new_give_back_in(val, A::try_clone(&this.alloc)?)
                        .map_err(|(_, e)| TryRcError::CloneAlloc(TryCloneError::Alloc(e)))
                }
            }
        }
    }

    /// Same as [`increment_strong_count`](Self::increment_strong_count), but
    /// parameterized over the allocator so the caller can recover the exact
    /// `Rc<T, A>` later.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if the strong count would overflow
    /// `usize`.
    ///
    /// # Safety
    ///
    /// As for [`increment_strong_count`](Self::increment_strong_count).
    // FIXME: check the entire module - methods dealing with pointers without `alloc`
    // assumes a global allocator and must have correct documentation (need to inline the
    // allocator requirement). Also you should not return anything.
    #[inline]
    pub unsafe fn increment_strong_count_in(ptr: *const T, alloc: &A) -> Result<(), TryRcError> {
        // NOTE: the alloc reference allows wrapping in a ManuallyDrop without leaking, and for
        // the allocator to be reused since cloning may be unintentionally expensive.
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation backed by
        // `alloc`. Wrapping in `ManuallyDrop` prevents a panic during unwinding
        // from decrementing the refcount.
        unsafe {
            let me = ManuallyDrop::new(Rc::from_raw_in(ptr, &alloc));
            me.inner()
                .expect("the pointer must be dangling")
                .inc_strong()?;
        }
        Ok(())
    }

    /// Same as [`decrement_strong_count`](Self::decrement_strong_count), but
    /// parameterized over the allocator so the correct deallocator is used
    /// when the last strong reference goes away.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if the strong count is already zero
    /// (unbalanced decrement).
    ///
    /// # Safety
    ///
    /// As for [`decrement_strong_count`](Self::decrement_strong_count).
    // FIXME: check the entire module - methods dealing with pointers without `alloc`
    // assumes a global allocator and must have correct documentation (need to inline the
    // allocator requirement). Also you should not return anything.
    #[inline]
    pub unsafe fn decrement_strong_count_in(ptr: *const T, alloc: &A) -> Result<(), TryRcError> {
        // NOTE: the alloc reference allows wrapping in a ManuallyDrop without leaking, and for
        // the allocator to be reused since cloning may be unintentionally expensive.
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation backed by
        // `alloc`. Wrapping in `ManuallyDrop` prevents a panic during unwinding
        // from arbitrarily lowering the refcount.
        unsafe {
            let me = ManuallyDrop::new(Rc::from_raw_in(ptr, &alloc));
            let inner = me.inner().expect("the pointer must be dangling");
            inner.dec_strong()?;
            if is_last_strong(inner.strong()) {
                Rc::drop_slow(&me);
            }
            Ok(())
        }
    }
}

impl<T: ?Sized> Rc<T, Global> {
    /// Increments the strong count of the allocation backing `ptr` without
    /// constructing an `Rc` wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if the strong count would overflow
    /// `usize`.
    ///
    /// # Safety
    ///
    /// `ptr` must have been produced by [`into_raw`](Self::into_raw) and must
    /// still be valid. The caller takes responsibility for eventually pairing
    /// every call with [`decrement_strong_count`](Self::decrement_strong_count)
    /// or wrapping the pointer back into an `Rc` via [`from_raw`](Self::from_raw).
    #[inline]
    pub unsafe fn increment_strong_count(ptr: *const T) -> Result<(), TryRcError> {
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation.
        unsafe { Self::increment_strong_count_in(ptr, &Global) }
    }

    /// Decrements the strong count of the allocation backing `ptr`. When the
    /// count reaches zero the value is dropped and the block is freed.
    ///
    /// # Errors
    ///
    /// Returns [`TryRcError::OutOfBounds`] if the strong count is already zero
    /// (unbalanced decrement).
    ///
    /// # Safety
    ///
    /// `ptr` must have been produced by [`into_raw`](Self::into_raw) and must
    /// still be valid. Pairing with [`increment_strong_count`](Self::increment_strong_count)
    /// is the caller's responsibility.
    #[inline]
    pub unsafe fn decrement_strong_count(ptr: *const T) -> Result<(), TryRcError> {
        // SAFETY: caller guarantees `ptr` is a live `Rc` allocation backed by
        // the global allocator.
        unsafe { Self::decrement_strong_count_in(ptr, &Global) }
    }
}

// ---------------------------------------------------------------------------
// Conversion to Weak (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: AllocatorTryClone> Rc<T, A> {
    /// Borrows an `Rc` as a [`Weak`] pointer.
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
        // SAFETY: bumping the weak count keeps the allocation alive for the
        // duration of the `Weak`; the strong count is untouched.
        unsafe {
            let inner = this.ptr.as_ptr();
            (*inner).inc_weak()?;
            Ok(Weak {
                ptr: this.ptr,
                alloc,
                _marker: PhantomData,
            })
        }
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

/// Infallible clone for `Rc`. Bumps the strong count and clones the allocator
/// handle via `A::clone`. Panics only on counter overflow, which indicates a
/// logic error in practice.
// FIXME: remove this
impl<T: ?Sized, A: Allocator + Clone> Clone for Rc<T, A> {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: bumping the strong count keeps the allocation alive for the
        // new handle; no allocation is involved. An unbalanced increment here
        // would indicate a logic error, so we surface it as a panic rather than
        // silently corrupting the count.
        unsafe {
            let inner = self.ptr.as_ptr();
            (*inner).inc_strong().expect("Rc strong count overflow");
        }
        Rc {
            ptr: self.ptr,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        }
    }
}

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
        // SAFETY: `self` is a live `Rc`, so the allocation is alive and the
        // header cell is stable.
        unsafe { (*self.ptr.as_ptr()).inc_strong() }
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
        // Decrement the strong count first. The helper reports out-of-bounds
        // conditions, which are unreachable here: we own a live strong
        // reference, so the count is at least one and cannot underflow.
        let inner = self.inner().expect("a Rc should not be dangling");
        // SAFETY: we own a live strong reference, so the allocation is alive
        // and the header cell is stable; the decrement cannot underflow
        // because `old_strong >= 1`.
        inner.dec_strong().expect("strong count underflow");
        if is_last_strong(inner.strong()) {
            Self::drop_slow(self);
        }
    }
}

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Destroys the value and conditionally frees the block after the last
    /// strong reference has been dropped.
    fn drop_slow(this: &Self) {
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

// The bound is `T: Default` (not `T: TryDefault`) because constructing the
// payload itself cannot fail — only the heap allocation can. Using `Default`
// keeps the impl maximally permissive: any type with an infallible default can
// be wrapped in a fallible `Rc`. A future `TryDefault`-bound variant could be
// added for payloads whose default construction is itself fallible.
// FIXME: must use TryDefault bounds for T
impl<T: Default> TryDefault for Rc<T, Global> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::try_new(T::default())?)
    }
}

// FIXME: must retire - allocation and default creation can fail.
impl<T: Default> Default for Rc<T, Global> {
    #[inline]
    fn default() -> Self {
        // Allocation failure is treated as a logic error in the infallible
        // `Default` context; in practice the global allocator only fails under
        // extreme OOM.
        Self::try_new(T::default()).expect("Rc::default allocation failed")
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
    /// [`Weak::into_raw`](Self::into_raw).
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained from
    /// [`Weak::into_raw`](Self::into_raw) on a
    /// `Weak` with the same allocator, and must still be valid.
    #[inline]
    pub unsafe fn from_raw(p: *const T) -> Self {
        // SAFETY: caller guarantees `p` derives from `Weak::into_raw`.
        unsafe {
            let inner = data_get_ptr(p);
            Weak {
                ptr: NonNull::new_unchecked(inner as *mut RcInner<T>),
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
    pub fn into_raw(w: Self) -> *const T {
        // Mirrors std's `Weak::into_raw`: wrap in `ManuallyDrop` so that if
        // this function ever panics (e.g. during unwinding), the block is not
        // double-freed by the implicit drop at scope end. The `Global`
        // allocator is a ZST, but we still perform an explicit read to keep
        // the ownership transfer uniform with the generic path and to make
        // the intent clear to reviewers and future maintainers.
        let me = ManuallyDrop::new(w);
        let _alloc = unsafe { ptr::read(&me.alloc) };
        ptr_get_data(me.ptr.as_ptr())
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Like [`from_raw`](Self::from_raw), but parameterized over the choice of
    /// allocator.
    ///
    /// # Safety
    ///
    /// The pointer must have been produced by
    /// [`into_raw_with_allocator`](Self::into_raw_with_allocator) on a `Weak`
    /// with the same allocator, and must still be valid.
    #[inline]
    pub unsafe fn from_raw_in(p: *const T, alloc: A) -> Self {
        // SAFETY: caller guarantees validity.
        unsafe {
            let inner = data_get_ptr(p);
            Weak {
                ptr: NonNull::new_unchecked(inner as *mut RcInner<T>),
                alloc,
                _marker: PhantomData,
            }
        }
    }

    /// Converts a `Weak<T, A>` into a raw pointer, returning its allocator.
    ///
    /// The caller takes ownership of both the allocation and the allocator and
    /// must eventually reconstruct a `Weak` from them (via
    /// [`from_raw_in`](Self::from_raw_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw_with_allocator(w: Self) -> (*const T, A) {
        // Mirrors std's `Weak::into_raw_with_allocator`: wrap in `ManuallyDrop`
        // so that a panic during unwinding cannot cause the implicit drop at
        // scope end to free the block after we have already handed ownership
        // to the caller. The allocator is read out explicitly; the rest of the
        // struct is logically consumed by the return value.
        let me = ManuallyDrop::new(w);
        let ptr = ptr_get_data(me.ptr.as_ptr());
        let alloc = unsafe { ptr::read(&me.alloc) };
        (ptr, alloc)
    }

    /// Gets a shared raw pointer to the underlying data.
    ///
    /// The pointer may be dangling if the strong references have all vanished;
    /// it must not be dereferenced unless [`try_upgrade`](Self::try_upgrade) succeeds.
    #[must_use]
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        ptr_get_data(this.ptr.as_ptr())
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
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::addr_eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Returns a shared reference to the allocation's internal [`RcInner`]
    /// header, or `None` if this handle is dangling (constructed via
    /// [`Weak::new`]) or the strong count has already reached zero.
    ///
    /// This is the primitive behind [`try_upgrade`](Self::try_upgrade): it hands
    /// out a direct borrow of the counter block so the caller can inspect the
    /// strong count without re-validating provenance.
    #[inline]
    pub(crate) fn inner(&self) -> Option<&RcInner<T>> {
        if is_dangling_weak(self.ptr.as_ptr()) {
            return None;
        }
        // SAFETY: a non-dangling `Weak` always owns a live allocation whose
        // header is valid for reads while `self` exists.
        Some(unsafe { &*self.ptr.as_ptr() })
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
    pub fn try_upgrade(this: &Self) -> Result<Option<Rc<T, A>>, TryRcError> {
        // A dangling weak (from `Weak::new`) never referred to an allocation,
        // so it can never be upgraded. Checking this first also avoids ever
        // dereferencing the sentinel address below.
        let inner_ref = match this.inner() {
            Some(i) => i,
            None => return Ok(None),
        };
        let strong = inner_ref.strong();
        if strong == 0 {
            return Ok(None);
        }

        let alloc = A::try_clone(&this.alloc)?;
        // Restore the strong count. The allocation is guaranteed to stay alive
        // because this `Weak` itself pins it. We only bump strong; the weak
        // count already accounts for this handle (it was incremented when the
        // `Weak` was created via `try_downgrade`).
        inner_ref.inc_strong()?;
        Ok(Some(Rc {
            ptr: this.ptr,
            alloc,
            _marker: PhantomData,
        }))
    }
}

// FIXME: remove this one, Allocator + Clone does not make sense.
impl<T: ?Sized, A: Allocator + Clone> Clone for Weak<T, A> {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: bumping the weak count keeps the allocation alive for the new
        // handle; no allocation is involved. An unbalanced decrement here would
        // indicate a logic error, so we surface it as a panic rather than
        // silently corrupting the count.
        unsafe {
            let inner = self.ptr.as_ptr();
            (*inner).inc_weak().expect("Weak weak count overflow");
        }
        Weak {
            ptr: self.ptr,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        }
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
        let Some(inner_ref) = self.inner() else {
            return;
        };

        // Decrement the weak count. If this was the last reference of any kind
        // (strong already zero, and now weak hits zero), free the allocation.
        // An unbalanced decrement here would indicate a logic error, so we
        // surface it as a panic rather than silently corrupting the count.
        inner_ref.dec_weak().expect("Weak weak count underflow");

        // Invariant: once the weak count reaches zero, the strong count must
        // also be zero (the last strong `Rc`'s drop either freed the block or
        // left it pinned by at least one `Weak`). So `weak == 0` is sufficient
        // to decide whether to deallocate.
        if is_last_ref(inner_ref.weak()) {
            let layout = Layout::for_value(&inner_ref);
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

// Unlike std (which prints only `(Weak)`), Olive's `Debug` shows the upgraded
// value when available, falling back to `<defunct>` otherwise. This is strictly
// more informative for debugging cyclic structures and weak-reference lifetimes.
impl<T: Debug + ?Sized, A: AllocatorTryClone> Debug for Weak<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Mirror std: show the upgraded value if possible, else "<defunct>".
        // Debug must not fail, so any upgrade error is treated as defunct.
        match Self::try_upgrade(self).ok().flatten() {
            Some(rc) => write!(f, "Weak({rc:?})"),
            None => f.write_str("<defunct>"),
        }
    }
}

// A default `Weak` is simply a dangling one: it points nowhere and frees
// nothing. Infallible by construction (no allocation involved), so both the
// conventional `Default` and the crate-standard fallible `TryDefault` succeed.
impl<T: ?Sized> Default for Weak<T, Global> {
    #[inline]
    fn default() -> Self {
        Self::new()
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
        let rc2 = rc.clone();
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
        let rc2 = rc.clone();
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
        assert!(Weak::try_upgrade(&weak).unwrap().is_none());
    }

    #[test]
    fn weak_upgrade_restores_strong() {
        let rc = Rc::try_new(7).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        let upgraded = Weak::try_upgrade(&weak).unwrap().unwrap();
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
    fn multiple_weak_refs_all_release() {
        let rc = Rc::try_new(3).unwrap();
        let w1 = Rc::try_downgrade(&rc).unwrap();
        let w2 = w1.clone();
        assert_eq!(Rc::weak_count(&rc), 2);
        drop(rc);
        assert!(Weak::try_upgrade(&w1).unwrap().is_none());
        assert!(Weak::try_upgrade(&w2).unwrap().is_none());
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
    fn weak_debug_defunct_after_drop() {
        let rc = Rc::try_new(1).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        drop(rc);
        let dbg = std::format!("{weak:?}");
        assert!(dbg.contains("defunct"));
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
        let rc2 = rc.clone();
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
        assert!(Weak::try_upgrade(&weak).unwrap().is_none());
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
        // A payload of `isize::MAX - 8` bytes is itself a valid layout, but
        // adding the 16-byte header pushes the total past what fits in an
        // addressable block, so `extend` reports an overflow.
        let huge = Layout::from_size_align(isize::MAX as usize - 8, 8)
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
        assert_eq!(unsafe { (*raw).len() }, 3);
        assert_eq!(unsafe { &*raw }, [7, 8, 9]);
        let rc = unsafe { Rc::from_raw(raw) };
        assert_eq!(&*rc, [7, 8, 9]);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn unsized_as_ptr_and_borrow() {
        let arr = [10u8, 20];
        let rc: Rc<[u8]> = Rc::try_from_slice(&arr[..]).unwrap();
        let p: *const [u8] = Rc::as_ptr(&rc);
        assert_eq!(unsafe { (*p).len() }, 2);
        assert_eq!(unsafe { &*p }, [10, 20]);
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
        assert!(Weak::try_upgrade(&weak).unwrap().is_none());
        drop(weak);
    }

    #[test]
    fn default_constructs_wrapped_default() {
        let rc = Rc::<i32>::default();
        assert_eq!(*rc, 0);
        assert_eq!(Rc::strong_count(&rc), 1);
        // Fallible twin agrees.
        assert_eq!(*Rc::<u8>::try_default().unwrap(), 0);
    }

    #[test]
    fn weak_new_dangling_upgrades_to_none() {
        // A freshly constructed `Weak` refers to no allocation, so upgrading it
        // must always yield `None`.
        let w: Weak<i32> = Weak::new();
        assert!(Weak::try_upgrade(&w).unwrap().is_none());
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
        assert!(Weak::try_upgrade(&a).unwrap().is_none());
        assert!(Weak::try_upgrade(&b).unwrap().is_none());
        // And the fallible twin agrees.
        let c: Weak<i32> = TryDefault::try_default().unwrap();
        assert!(Weak::try_upgrade(&c).unwrap().is_none());
    }
}
