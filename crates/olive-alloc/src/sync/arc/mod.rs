//! A fully-fallible port of the standard library's [`Arc`](stock_alloc::sync::Arc) and
//! [`Weak`](stock_alloc::sync::Weak).
//!
//! Compared with the std original, three things differ:
//!
//! * Every constructor that allocates a new node returns a [`Result`] carrying
//!   [`crate::alloc::AllocError`] instead of panicking or aborting on
//!   out-of-memory.
//! * All allocation goes through the [`Allocator`] trait rather than free
//!   functions, giving a single swappable seam for custom allocators and OOM
//!   simulation in tests.
//! * Every reference-count mutation and allocator-handle clone is fallible and
//!   reports failures via an error type. Cloning an existing [`Arc`] does not
//!   allocate a new block (it only bumps the strong count), but it still must
//!   clone the allocator handle, which is a first-class fallible operation in
//!   this framework. The crate-standard [`TryClone`](olive_core::try_traits::try_clone::TryClone)
//!   impls surface these failures as [`TryCloneError`](olive_core::try_traits::try_clone::TryCloneError).
//!
//! Unlike its single-threaded sibling [`crate::rc::Rc`], the reference counts
//! here are **atomic** integers: `Arc` is safe to share across threads, so every
//! counter read/write uses atomics with appropriate orderings. This makes
//! `Arc<T>` `Send + Sync` whenever `T` is `Send + Sync`.
//!
//! # Allocator bounds
//!
//! Cloning an `Arc<T, A>` requires `A: AllocatorTryClone`, not merely
//! `Allocator + Clone`. The stronger bound guarantees that a cloned allocator
//! handle is *equivalent* to the original — memory allocated through one may be
//! freed through the other — which is essential for the refcount-bump clone
//! path to remain sound. A plain `Clone` on the allocator would permit two
//! independent backing stores, breaking the invariant that all handles share
//! one allocation.
//!
//! # Status
//!
//! This module currently contains the struct declarations, the shared internal
//! header, and their `Drop` implementations. The fallible constructors (both
//! the global-allocator and allocator-generic forms) live in the child
//! [`construction`](self::construction) module, along with the uninit→init
//! bridge; the query methods (`as_ptr`, `allocator`, `ptr_eq`, and the
//! refcount reads) live in the child [`query`](self::query) module. `Deref`,
//! `Clone`/`TryClone`, weak-reference handling, and the raw-pointer
//! reconstitution methods will land in later incremental steps, each keeping
//! the tree compiling and tested.

use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr;
use core::sync::atomic::{
    self, AtomicUsize,
    Ordering::{Acquire, Release},
};

use crate::alloc::{Allocator, Global, Layout};
use olive_core::alloc::LayoutExt;
use olive_core::ptr::{self as ptr_ext, NonNull};

/// Fallible node-construction methods and the uninit→init bridge.
mod construction;
/// Query methods (`as_ptr`, `allocator`, `ptr_eq`, refcount reads).
mod query;

// ---------------------------------------------------------------------------
// Shared internals
// ---------------------------------------------------------------------------

/// Internal representation of an `Arc` allocation.
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
///
/// The counters are [`AtomicUsize`] because `Arc` is thread-safe: unlike
/// [`crate::rc::RcInner`]'s plain `Cell<usize>` fields, these must support
/// concurrent mutation from multiple threads.
#[repr(C, align(2))]
pub(crate) struct ArcInner<T: ?Sized> {
    /// Number of strong (`Arc`) pointers currently alive. Zero means no strong
    /// references remain; the internal value may then be dropped even if
    /// weak refs exist.
    strong: AtomicUsize,
    /// Number of weak (`Weak`) pointers currently alive. Includes the implicit
    /// weak reference held by every live `Arc`, so this is always at least the
    /// number of `Arc`s while any strong reference exists.
    weak: AtomicUsize,
    /// The contained value.
    value: T,
}

impl<T: ?Sized> ArcInner<T> {
    /// Reads the current strong count.
    ///
    /// Only a plain `Relaxed` load is provided as a convenience: every
    /// *mutation* of the counters uses an explicit atomic operation at its call
    /// site with the ordering that operation requires, mirroring the standard
    /// library. Folding an ordering into a wrapper would hide those decisions
    /// and make it easy to reuse the wrong one.
    #[inline]
    pub(crate) fn strong(&self) -> usize {
        self.strong.load(atomic::Ordering::Relaxed)
    }

    /// Reads the current weak count. See [`Self::strong`] for why this is a
    /// `Relaxed` load.
    #[inline]
    pub(crate) fn weak(&self) -> usize {
        self.weak.load(atomic::Ordering::Relaxed)
    }
}

/// Helper type allowing access to an allocation's reference-count cells without
/// making any assertions about the data field.
///
/// When a `Weak` outlives all `Arc`s, the payload (`value`) has been dropped
/// in-place but the allocation remains alive (pinned by the weak count). A
/// `&ArcInner<T>` covering the whole struct would assert validity of the
/// already-dropped payload. `WeakInner` holds only references to the two
/// counter cells, which are always valid while the allocation exists.
pub(crate) struct WeakInner<'a> {
    strong: &'a AtomicUsize,
    weak: &'a AtomicUsize,
}

impl WeakInner<'_> {
    /// Reads the current strong count.
    #[inline]
    fn strong(&self) -> usize {
        self.strong.load(atomic::Ordering::Relaxed)
    }

    /// Reads the current weak count.
    #[inline]
    fn weak(&self) -> usize {
        self.weak.load(atomic::Ordering::Relaxed)
    }
}

/// Returns true when the given strong count indicates the last strong
/// reference has been released and the value should be destroyed.
#[inline]
fn is_last_strong(strong: usize) -> bool {
    strong == 0
}

/// The sentinel address of a "dangling" [`Weak`] — one that never referred to a
/// real allocation (see [`Weak::new`]).
///
/// This is deliberately **misaligned** (`usize::MAX` is odd, while `ArcInner`
/// requires at least 2-alignment). No valid allocation can ever occupy a
/// misaligned address, so comparing against this value reliably detects "was
/// this weak ever attached to anything?" without any possibility of collision
/// with a real pointer.
const DANGLING_WEAK_ADDR: usize = usize::MAX;

/// True if `p` points at the dangling sentinel produced by [`Weak::new`].
#[inline]
fn is_dangling_weak<T: ?Sized>(p: *const ArcInner<T>) -> bool {
    p.addr() == DANGLING_WEAK_ADDR
}

// ---------------------------------------------------------------------------
// Pointer helpers
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
unsafe fn ptr_get_data<T: ?Sized>(p: *const ArcInner<T>) -> *const T {
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
unsafe fn ptr_get_data_mut<T: ?Sized>(p: *mut ArcInner<T>) -> *mut T {
    unsafe { &raw mut (*p).value }
}

/// Initializes the two reference-count headers of a freshly-allocated
/// `ArcInner<T>` block to `(strong = 1, weak = 1)`.
///
/// The allocation is fresh (uninitialized), so the counters are seeded with a
/// plain [`ptr::write`] of [`AtomicUsize::new(1)`] through a raw-mut reference
/// projected off each field — rather than calling `.store()` on an atomic that
/// has not yet been initialized, which would be undefined behavior. This mirrors
/// the standard library's approach for seeding `RcInner`/`ArcInner` headers.
///
/// # Safety
///
/// - `p` must point to a valid, aligned `ArcInner<T>` allocation block whose
///   header fields have not yet been initialized.
/// - No other thread may observe the block until this function returns; the
///   writes establish the initial counter state before the block escapes.
#[inline]
pub(super) unsafe fn initialize_arcinner<T: ?Sized>(p: *mut ArcInner<T>) {
    unsafe {
        ptr::write(&raw mut (*p).strong, AtomicUsize::new(1));
        ptr::write(&raw mut (*p).weak, AtomicUsize::new(1));
    }
}

/// Builds the dangling inner pointer stored by [`Weak::new`].
///
/// The address word is pinned to [`DANGLING_WEAK_ADDR`] (`usize::MAX`), which
/// is deliberately **misaligned** relative to `ArcInner`'s required alignment.
/// This guarantees the sentinel can never collide with a real allocation's
/// address.
#[inline]
const fn dangling_inner_ptr<T: ?Sized>() -> NonNull<ArcInner<T>> {
    // SAFETY: `DANGLING_WEAK_ADDR` is non-zero, satisfying `NonNull`'s
    // invariant. The address is intentionally misaligned — no valid
    // allocation could sit there. `Drop`, `upgrade`, and all other access
    // paths check `is_dangling_weak` before touching memory, so the pointer
    // is never dereferenced despite its misalignment.
    unsafe {
        let mut slot: MaybeUninit<*mut ArcInner<T>> = MaybeUninit::zeroed();
        let data_offset = const { ptr_ext::address_word_offset::<T>() };
        *slot.as_mut_ptr().byte_add(data_offset).cast::<usize>() = DANGLING_WEAK_ADDR;
        NonNull::new_unchecked(slot.assume_init())
    }
}

// ---------------------------------------------------------------------------
// Arc declaration
// ---------------------------------------------------------------------------

/// A thread-safe reference-counting pointer.
///
/// `Arc<T>` shares ownership of a heap-allocated `T` among any number of strong
/// references across threads. The value is dropped when the last strong
/// reference disappears; [`Weak`] references do not keep it alive.
///
/// Unlike `std::sync::Arc`, constructing a fresh node can fail: the upcoming
/// constructors return `Result<Self, AllocError>` instead of panicking or
/// aborting on out-of-memory.
pub struct Arc<T: ?Sized, A: Allocator = Global> {
    /// The backing pointer.
    ptr: NonNull<ArcInner<T>>,
    /// The allocator used to construct the pointer.
    alloc: A,
    // Prevents accidental auto-derived `Send`/`Sync`; the actual impls are
    // written explicitly below, gated on `T: Send + Sync` and `A: Send + Sync`.
    _marker: PhantomData<*const T>,
}

// ---------------------------------------------------------------------------
// Weak declaration
// ---------------------------------------------------------------------------

/// A weak reference to an [`Arc`] allocation.
///
/// [`Weak<T>`] borrows the allocation without keeping it alive: upgrading a
/// [`Weak`] yields an `Option<Arc<T>>` that is `None` once every strong
/// reference has been dropped.
pub struct Weak<T: ?Sized, A: Allocator = Global> {
    ptr: NonNull<ArcInner<T>>,
    alloc: A,
    _marker: PhantomData<ArcInner<T>>,
}

// ---------------------------------------------------------------------------
// Drop
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Drop for Arc<T, A> {
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the `fetch_sub` on the weak count performed by
        // the fake `Weak` created in `drop_slow`.
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return;
        }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data. Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data in another thread happens before decreasing
        // the reference count, which happens before this fence, which happens
        // before the deletion of the data.
        atomic::fence(Acquire);

        // SAFETY: we observed that we were the last strong reference, so
        // exclusive access to the payload is established.
        unsafe {
            Self::drop_slow(self);
        }
    }
}

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Returns a shared reference to the allocation's internal [`ArcInner`]
    /// header (strong/weak counters and payload).
    #[inline]
    pub(crate) fn inner(&self) -> &ArcInner<T> {
        // SAFETY: an `Arc` always owns a live allocation whose header is valid
        // for reads while `self` exists. Furthermore, we know that the
        // `ArcInner` structure itself is `Sync` if the inner data is `Sync` as
        // well, so we're ok loaning out an immutable pointer to these contents.
        unsafe { self.ptr.as_ref() }
    }

    /// Destroys the value and conditionally frees the block after the last
    /// strong reference has been dropped.
    ///
    /// # Safety
    /// - The Arc must be uniquely owned (or strong == 1).
    #[inline(never)]
    unsafe fn drop_slow(this: &mut Self) {
        // Construct a temporary `Weak` standing in for this `Arc`'s implicit
        // weak reference. This is the decrement-or-deallocate guard that
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
            ptr::drop_in_place(ptr_get_data_mut(this.ptr.as_ptr()));
        }
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Returns a [`WeakInner`] handle to the allocation's reference-count
    /// cells, or `None` if this handle is dangling (constructed via
    /// [`Weak::new`]).
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

impl<T: ?Sized, A: Allocator> Drop for Weak<T, A> {
    #[inline]
    fn drop(&mut self) {
        // A dangling weak (from `Weak::new`) owns no allocation, so there is
        // nothing to decrement or free — return immediately.
        let inner = match self.inner() {
            Some(inner) => inner,
            None => return,
        };

        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. If we
        // find out that we were the last weak pointer, then it is time to
        // deallocate the data entirely.
        if inner.weak.fetch_sub(1, Release) == 1 {
            // This fence pairs with the `Release` on the decrements above: it
            // orders our prior writes (including the counter) before the deallocation,
            // mirroring the standard library's `acquire!` macro.
            atomic::fence(Acquire);

            // Invariant: once the weak count reaches zero, the strong count
            // must also be zero (the last strong `Arc`'s drop either freed the
            // block or left it pinned by at least one `Weak`). So `weak == 0`
            // is sufficient to decide whether to deallocate.
            debug_assert!(is_last_strong(inner.strong()));

            // SAFETY: `self.ptr` carries correct pointer metadata for `T`; the
            // pointee may be uninitialized (already dropped) but we only need
            // its size and alignment, which live in the fat pointer. The block
            // was allocated with exactly this layout; the header alone
            // guarantees a non-zero size.
            let layout = unsafe { Layout::for_value_pointer(self.ptr.as_ptr()) };
            // SAFETY: the block was allocated by `self.alloc` with this layout,
            // and we are the last reference holding it.
            unsafe {
                self.alloc.deallocate(self.ptr.cast(), layout);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-trait impls
// ---------------------------------------------------------------------------

// `Arc<T, A>` is `Send`/`Sync` precisely when both the payload and the
// allocator are: sharing the pointer across threads is safe because the
// counters are atomic, and moving the handle is safe because neither the
// payload nor the allocator handle is invalidated by the move.
unsafe impl<T: ?Sized + Send + Sync, A: Allocator + Send + Sync> Send for Arc<T, A> {}
unsafe impl<T: ?Sized + Send + Sync, A: Allocator + Send + Sync> Sync for Arc<T, A> {}

unsafe impl<T: ?Sized + Send + Sync, A: Allocator + Send + Sync> Send for Weak<T, A> {}
unsafe impl<T: ?Sized + Send + Sync, A: Allocator + Send + Sync> Sync for Weak<T, A> {}
