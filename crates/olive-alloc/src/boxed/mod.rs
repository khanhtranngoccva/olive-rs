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
use core::mem::{self, ManuallyDrop, MaybeUninit, size_of};
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::ptr::{self, NonNull};

use crate::alloc::{AllocError, Allocator, Global, Layout, StaticAllocator};
use crate::raw_vec::RawVec;
use olive_core::alloc::LayoutExt;
use olive_core::alloc_errors::TryReserveError;
use olive_core::ptr::PointerExt;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError, TryCloneToUninit};

// ---------------------------------------------------------------------------
// Submodules
// ---------------------------------------------------------------------------

pub(crate) mod convert;

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
// Global construction block (sized T)
// ---------------------------------------------------------------------------

impl<T> Box<T> {
    /// Allocates memory on the heap and places `x` into it.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new(x: T) -> Result<Self, AllocError> {
        Self::try_new_give_back(x).map_err(|(_given_back, err)| err)
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
        match Self::try_new_uninit() {
            Ok(mut b) => {
                // Compiler optimization: do not monomorphize generic
                b.deref_mut().write(x);
                Ok(unsafe { mem::transmute::<Box<MaybeUninit<T>>, Self>(b) })
            }
            Err(e) => Err((x, e)),
        }
    }

    /// Allocates a new `Box<T>` containing `x` and pins it in place, returning
    /// a `Pin<Box<T>>`.
    ///
    /// If `T` does not implement `Unpin`, then `*boxed` will be pinned in memory
    /// and unable to be moved.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let p: core::pin::Pin<Box<i32>> = Box::try_pin(42).unwrap();
    /// assert_eq!(*p, 42);
    /// ```
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_pin(x: T) -> Result<Pin<Self>, AllocError> {
        let boxed = Self::try_new(x)?;
        // SAFETY: A freshly allocated box is not aliased anywhere else, so
        // it is safe to pin regardless of whether `T: Unpin`.
        Ok(unsafe { Pin::new_unchecked(boxed) })
    }

    /// Like [`try_pin`](Self::try_pin), but on allocation failure returns the
    /// unallocated `x` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_pin_give_back(x: T) -> Result<Pin<Self>, (T, AllocError)> {
        match Self::try_new_give_back(x) {
            Ok(boxed) => {
                // SAFETY: A freshly allocated box is not aliased anywhere else,
                // so it is safe to pin regardless of whether `T: Unpin`.
                Ok(unsafe { Pin::new_unchecked(boxed) })
            }
            Err(given_back) => Err(given_back),
        }
    }

    /// Constructs a `Box<MaybeUninit<T>>` filled with zeros.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed() -> Result<Box<MaybeUninit<T>>, AllocError> {
        // ZST optimization: no actual allocation needed.
        if size_of::<T>() == 0 {
            return Ok(Box {
                inner: NonNull::dangling(),
                alloc: Global,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = Global.allocate_zeroed(layout)?;

        Ok(Box {
            inner: ptr.cast(),
            alloc: Global,
        })
    }

    /// Constructs a `Box<MaybeUninit<T>>` containing uninitialized memory.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn try_new_uninit() -> Result<Box<MaybeUninit<T>>, AllocError> {
        // ZST optimization: no actual allocation needed.
        if size_of::<T>() == 0 {
            return Ok(Box {
                inner: NonNull::dangling(),
                alloc: Global,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = Global.allocate(layout)?;

        Ok(Box {
            inner: ptr.cast(),
            alloc: Global,
        })
    }
}

// ---------------------------------------------------------------------------
// Generic construction block (sized T)
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Box<T, A> {
    /// Like [`try_new`](Self::try_new), but parameterized over the choice of allocator for the
    /// returned `Box`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_in(x: T, alloc: A) -> Result<Self, AllocError> {
        let b = Self::try_new_uninit_in(alloc)?;
        Ok(b.write(x))
    }

    /// Like [`try_new_give_back`](Self::try_new_give_back), but parameterized over the choice of
    /// allocator for the returned `Box`. On allocation failure returns the
    /// unallocated `x` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_give_back_in(x: T, alloc: A) -> Result<Self, (T, AllocError)> {
        match Self::try_new_uninit_in(alloc) {
            Ok(b) => Ok(b.write(x)),
            Err(e) => Err((x, e)),
        }
    }

    /// Like [`try_new_zeroed`](Self::try_new_zeroed), but parameterized over the choice of allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed_in(alloc: A) -> Result<Box<MaybeUninit<T>, A>, AllocError> {
        // ZST optimization: no actual allocation needed.
        if size_of::<T>() == 0 {
            return Ok(Box {
                inner: NonNull::dangling(),
                alloc,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = alloc.allocate_zeroed(layout)?;

        Ok(Box {
            inner: ptr.cast(),
            alloc,
        })
    }

    /// Like [`try_new_uninit`](Self::try_new_uninit), but parameterized over the choice of allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_uninit_in(alloc: A) -> Result<Box<MaybeUninit<T>, A>, AllocError> {
        if size_of::<T>() == 0 {
            return Ok(Box {
                inner: NonNull::dangling(),
                alloc,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = alloc.allocate(layout)?;

        Ok(Box {
            inner: ptr.cast(),
            alloc,
        })
    }

    /// Like [`try_pin`](Self::try_pin), but parametrized over the choice of allocator.
    ///
    /// Allocates a new `Box<T>` containing `x` and pins it in place,
    /// returning a `Pin<Box<T>>`.
    ///
    /// If `T` does not implement `Unpin`, then `*boxed` will be pinned in
    /// memory and unable to be moved.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    /// use olive_alloc::alloc::Global;
    ///
    /// let p: core::pin::Pin<Box<i32, Global>> = Box::try_pin_in(42, Global).unwrap();
    /// assert_eq!(*p, 42);
    /// ```
    #[inline]
    pub fn try_pin_in(x: T, alloc: A) -> Result<Pin<Box<T, A>>, AllocError>
    where
        A: StaticAllocator,
    {
        let boxed = Self::try_new_in(x, alloc)?;
        // SAFETY: A freshly allocated box is not aliased anywhere else, and
        // `A: StaticAllocator` guarantees the backing memory stays valid, so
        // it is safe to pin regardless of whether `T: Unpin`.
        Ok(unsafe { Pin::new_unchecked(boxed) })
    }

    /// Like [`try_pin_give_back`](Self::try_pin_give_back), but parameterized over the choice of
    /// allocator for the returned `Box`. On allocation failure returns the
    /// unallocated `x` back to the caller alongside the error.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_pin_give_back_in(x: T, alloc: A) -> Result<Pin<Box<T, A>>, (T, AllocError)>
    where
        A: StaticAllocator,
    {
        match Self::try_new_give_back_in(x, alloc) {
            Ok(boxed) => {
                // SAFETY: A freshly allocated box is not aliased anywhere else, and
                // `A: StaticAllocator` guarantees the backing memory stays valid, so
                // it is safe to pin regardless of whether `T: Unpin`.
                Ok(unsafe { Pin::new_unchecked(boxed) })
            }
            Err(given_back) => Err(given_back),
        }
    }
}

// ---------------------------------------------------------------------------
// Box<MaybeUninit<T>, A> — initialization helpers
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Box<MaybeUninit<T>, A> {
    /// Writes `val` into the boxed slot and returns the initialized `Box<T, A>`.
    #[inline]
    pub fn write(mut self, val: T) -> Box<T, A> {
        self.deref_mut().write(val);
        unsafe { self.assume_init() }
    }

    /// Asserts that the contained value has been initialized and returns an
    /// owning `Box<T, A>`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the slot has been fully initialized.
    #[inline]
    pub unsafe fn assume_init(self) -> Box<T, A> {
        // SAFETY: `Box<T>` and `Box<MaybeUninit<T>>` have the same layout.
        unsafe { olive_core::mem::transmute_unchecked::<Self, Box<T, A>>(self) }
    }
}

// ---------------------------------------------------------------------------
// Global pointer transformation block
// ---------------------------------------------------------------------------

impl<T: ?Sized> Box<T, Global> {
    /// Reconstitutes a `Box<T, Global>` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must have been previously obtained from
    /// [`into_raw`](Self::into_raw) or equivalent, and must still be valid
    /// (not yet freed).
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let b: Box<i32> = Box::try_new(5).unwrap();
    /// let raw = Box::into_raw(b);
    /// let b: Box<i32> = unsafe { Box::from_raw(raw) };
    /// assert_eq!(*b, 5);
    /// ```
    #[inline]
    pub unsafe fn from_raw(p: *mut T) -> Self {
        // SAFETY: caller guarantees `p` is a valid, non-null, properly-aligned
        // allocation made by `Global`.
        unsafe {
            let inner = NonNull::new_unchecked(p);
            Box {
                inner,
                alloc: Global,
            }
        }
    }

    /// Reconstitutes a `Box<T, Global>` from a `NonNull<T>`.
    ///
    /// # Safety
    ///
    /// The `NonNull` must point to a valid, properly-aligned allocation made
    /// by `Global`.
    #[inline]
    pub unsafe fn from_non_null(nn: NonNull<T>) -> Self {
        // SAFETY: caller guarantees `nn` is a valid, non-null, properly-aligned
        // allocation made by `Global`. Constructing the box performs no
        // dereference; the safety contract is carried by the function signature.
        Self {
            inner: nn,
            alloc: Global,
        }
    }

    /// Converts a `Box<T, Global>` into a raw pointer.
    ///
    /// The caller takes ownership of the allocation and must eventually
    /// reconstruct a `Box` from it (via [`from_raw`](Self::from_raw)) or
    /// manually deallocate it.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let b: Box<i32> = Box::try_new(5).unwrap();
    /// let raw = Box::into_raw(b);
    /// assert_eq!(unsafe { *raw }, 5);
    /// let b: Box<i32> = unsafe { Box::from_raw(raw) };
    /// assert_eq!(*b, 5);
    /// ```
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw(b: Self) -> *mut T {
        // Avoid `into_raw_with_allocator` as that interacts poorly with Miri's Stacked Borrows.
        let mut b = ManuallyDrop::new(b);
        // We need to give Miri (specifically, Stacked Borrows) a chance to recognize this as a
        // safe-to-raw-pointer cast. To achieve this, we first create a mutable reference, and then
        // cast that to a raw pointer -- this cast is recognized by the aliasing model and leads to
        // a suitable retag.
        // It would be wrong for `into_raw_with_allocator` to do the same as that would induce
        // uniqueness assumptions (from the `&mut`) that we only want with the default allocator.
        (&mut **b) as *mut T
    }

    /// Converts a `Box<T, Global>` into a `NonNull<T>`.
    ///
    /// The caller takes ownership of the allocation and must eventually
    /// reconstruct a `Box` from it (via [`from_non_null`](Self::from_non_null))
    /// or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_non_null(b: Self) -> NonNull<T> {
        // As of August 2026, we cannot utilize `Box::leak`
        // because whether or not you can reconstruct the `Box`
        // later using `Box::from_raw` or `Box::from_non_null` is
        // an open question.
        // SAFETY: `Box` is guaranteed to be non-null.
        unsafe { NonNull::new_unchecked(Self::into_raw(b)) }
    }
}

// ---------------------------------------------------------------------------
// Generic pointer transformation block
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Box<T, A> {
    /// Reconstitutes a `Box<T, A>` from a raw pointer and an allocator.
    ///
    /// # Safety
    ///
    /// The pointer must have been produced by
    /// [`into_raw_with_allocator`](Self::into_raw_with_allocator) on a
    /// `Box<T, A>` with the same allocator, and must still be valid.
    #[inline]
    pub unsafe fn from_raw_in(p: *mut T, alloc: A) -> Self {
        // SAFETY: caller guarantees validity.
        unsafe {
            let inner = NonNull::new_unchecked(p);
            Self { inner, alloc }
        }
    }

    /// Reconstitutes a `Box<T, A>` from a `NonNull<T>` and an allocator.
    ///
    /// # Safety
    ///
    /// The `NonNull` must point to a valid, properly-aligned allocation made
    /// by `alloc`.
    #[inline]
    pub unsafe fn from_non_null_in(nn: NonNull<T>, alloc: A) -> Self {
        // SAFETY: caller guarantees `nn` is a valid, non-null, properly-aligned
        // allocation made by `alloc`. Constructing the box performs no
        // dereference; the safety contract is carried by the function signature.
        Self { inner: nn, alloc }
    }

    /// Converts a `Box<T, A>` into a raw pointer, retaining its allocator.
    ///
    /// The caller takes ownership of both the allocation and the allocator and
    /// must eventually reconstruct a `Box` from them (via
    /// [`from_raw_in`](Self::from_raw_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw_with_allocator(b: Self) -> (*mut T, A) {
        let mut b = ManuallyDrop::new(b);
        // We carefully get the raw pointer out in a way that Miri's aliasing model understands what
        // is happening: using the primitive "deref" of `Box`. In case `A` is *not* `Global`, we
        // want *no* aliasing requirements here!
        // In case `A` *is* `Global`, this does not quite have the right behavior; `into_raw`
        // works around that.
        let ptr = &raw mut **b;
        let alloc = unsafe { ptr::read(&b.alloc) };
        (ptr, alloc)
    }

    /// Converts a `Box<T, A>` into a `(NonNull<T>, A)` pair.
    ///
    /// The caller takes ownership of both the allocation and the allocator and
    /// must eventually reconstruct a `Box` from them (via
    /// [`from_non_null_in`](Self::from_non_null_in)) or manually deallocate it.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_non_null_with_allocator(b: Self) -> (NonNull<T>, A) {
        let (ptr, alloc) = Box::into_raw_with_allocator(b);
        // SAFETY: `Box` is guaranteed to be non-null.
        unsafe { (NonNull::new_unchecked(ptr), alloc) }
    }

    /// Gets a mutable raw pointer to the underlying data.
    #[must_use]
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_ptr()
    }

    /// Gets a shared raw pointer to the underlying data.
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    /// Gets a `NonNull<T>` pointing to the underlying data.
    #[must_use]
    #[inline]
    pub fn as_non_null(&self) -> NonNull<T> {
        self.inner
    }

    /// Gets a shared reference to the allocator backing this `Box`.
    #[must_use]
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Consumes and leaks the `Box`, returning a mutable reference,
    /// `&'a mut T`.
    ///
    /// Note that the type `T` must outlive the chosen lifetime `'a`. If the type
    /// has only static references, or none at all, then this may be chosen to be
    /// `'static`.
    ///
    /// This function is mainly useful for data that lives for the remainder of the program's life,
    /// i.e., memory that is meant to leak. If the memory should eventually be freed, prefer to use
    /// [`Box::into_raw`] or [`Box::into_non_null`] instead. Reconstructing ("unleaking") a `Box` from
    /// the mutable reference returned here (e.g. via [`Box::from_raw`]) is only possible if the
    /// allocator is `Global`, and even then it is a grey area (meaning it is possible under specific
    /// circumstances but many seemingly harmless ways of doing it are undefined behavior) and should
    /// be avoided.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::leak(b)` instead of `b.leak()`. This
    /// is so that there is no conflict with a method on the inner type.
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let s: &'static str = Box::leak(Box::try_new("world").unwrap());
    /// assert_eq!(s, "world");
    /// ```
    #[inline]
    pub fn leak<'a>(b: Self) -> &'a mut T
    where
        A: 'a,
    {
        let (ptr, alloc) = Box::into_raw_with_allocator(b);
        mem::forget(alloc);
        // SAFETY: `ptr` points to a valid, aligned allocation that is now
        // leaked (ownership transferred to the returned `'static` reference).
        unsafe { &mut *ptr }
    }

    /// Converts a `Box<T>` into a `Pin<Box<T>>`.
    ///
    /// This is also available via `From<Box<T>> for Pin<Box<T>>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let b = Box::try_new(7).unwrap();
    /// let p = Box::into_pin(b);
    /// assert_eq!(*p, 7);
    /// ```
    #[inline]
    pub fn into_pin(b: Self) -> Pin<Self>
    where
        A: StaticAllocator,
    {
        // SAFETY: The box was just consumed by value; no other reference
        // exists, and `A: StaticAllocator` guarantees the backing memory will not be
        // invalidated without an explicit deallocation, so pinning is sound.
        unsafe { Pin::new_unchecked(b) }
    }
}

// ---------------------------------------------------------------------------
// Drop
// ---------------------------------------------------------------------------

/// Owns the raw pointer, layout, and allocator of a [`Box`] being dropped, and
/// frees them in its own `Drop`.
///
/// Dropping the contained value (`ptr::drop_in_place`) can panic. Without this
/// guard, a panic would unwind past the deallocation line and leak the block.
/// Arming the guard *before* dropping the pointee guarantees the free runs both
/// on the happy path (guard falls out of scope normally) and on unwind (guard's
/// `Drop` runs during stack teardown). There is exactly one free site.
///
/// It touches only the pointer/layout/allocator — never the pointee itself — so
/// it is safe to use while dropping a `Pin<Box<T>>`: the pin invariant forbids
/// *moving* the pointee, and neither dropping it in place nor freeing its block
/// relocates it. This mirrors std's observable behavior, where a pinned box is
/// freed on a panicking pointee drop exactly like a plain one.
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

// ---------------------------------------------------------------------------
// Clone constructors (unsized T — slices, str, dyn)
// ---------------------------------------------------------------------------

impl<T: ?Sized + TryCloneToUninit> Box<T, Global> {
    /// Clones a `&T` into a freshly allocated `Box<T, Global>` for potentially
    /// unsized `T`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation or any element clone fails.
    #[inline]
    pub fn try_clone_from_ref(t: &T) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(t, Global)
    }
}

impl<T: ?Sized + TryCloneToUninit, A: Allocator> Box<T, A> {
    /// Clones a `&T` into a freshly allocated `Box<T, A>` for potentially
    /// unsized `T`, using the given allocator.
    ///
    /// Supported unsized targets are those with a [`TryCloneToUninit`] impl
    /// whose fat-pointer metadata is a plain length: slices (`[U]`), `str`,
    /// and `CStr`. The metadata is recovered from `src` and reattached to the
    /// fresh base via stable constructors (`slice_from_raw_parts_mut`,
    /// `str::from_utf8_unchecked`) — a Miri-clean path that derives provenance
    /// from the *destination* allocation.
    ///
    /// For `dyn Trait` targets, use [`Box::try_clone_from_ref_into_dyn`]
    /// instead, which takes the concrete type and preserves the vtable.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation or any element clone fails.
    #[inline]
    pub fn try_clone_from_ref_in(src: &T, alloc: A) -> Result<Self, TryCloneError> {
        struct DeallocDropGuard<'a, A: Allocator>(Layout, &'a A, NonNull<u8>);
        impl<A: Allocator> Drop for DeallocDropGuard<'_, A> {
            fn drop(&mut self) {
                let &mut DeallocDropGuard(layout, alloc, ptr) = self;
                // Safety: `ptr` was allocated by `*alloc` with layout `layout`
                unsafe {
                    alloc.deallocate(ptr, layout);
                }
            }
        }
        let layout = Layout::for_value::<T>(src);
        let (ptr, guard) = if layout.size() == 0 {
            (layout.dangling_pointer(), None)
        } else {
            // Safety: layout is non-zero-sized
            let ptr = alloc.allocate(layout)?.cast();
            (ptr, Some(DeallocDropGuard(layout, &alloc, ptr)))
        };
        let ptr = ptr.as_ptr();
        // Safety: `*ptr` is newly allocated, correctly aligned to `align_of_val(src)`,
        // and is valid for writes for `size_of_val(src)`.
        // If this panics, then `guard` will deallocate for us (if allocation occuured)
        unsafe {
            <T as TryCloneToUninit>::try_clone_to_uninit(src, ptr)?;
        }
        // Defuse the deallocate guard
        mem::forget(guard);
        // Safety: We just initialized `*ptr` as a clone of `src`. Relocate the
        // freshly allocated mutable base `ptr` onto `src`'s metadata (length /
        // vtable), yielding a valid `*mut T` fat pointer to the clone.
        Ok(unsafe { Box::from_raw_in(ptr.cast_with_metadata(src as *const T), alloc) })
    }
}

impl<T, A: Allocator> Box<[MaybeUninit<T>], A> {
    /// Asserts that all elements of the boxed slice have been initialized and
    /// returns an owning `Box<[T], A>`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that every element in the slice has been fully
    /// initialized.
    #[inline]
    pub unsafe fn assume_init(self) -> Box<[T], A> {
        // SAFETY: `[MaybeUninit<T>]` and `[T]` are layout-compatible; only the
        // logical initialization state differs. Preserving the fat pointer's
        // length via `into_raw_with_allocator` keeps the slice length intact.
        let (raw, alloc) = Box::into_raw_with_allocator(self);
        unsafe { Box::from_raw_in(raw as *mut [T], alloc) }
    }
}

impl<T, A: Allocator> Box<[T], A> {
    /// Creates a new empty boxed slice using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    // FIXME: remove, this method does not exist (slices do not have any concept of capacity)
    #[inline]
    pub fn try_new_empty_in(alloc: A) -> Result<Self, AllocError> {
        Self::try_with_capacity_in(0, alloc)
    }

    /// Creates a new boxed slice with the given capacity using the given
    /// allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    // FIXME: remove, this method does not exist (slices do not have any concept of capacity)
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, AllocError> {
        let buf = RawVec::<T, A>::try_with_capacity_in(capacity, alloc)
            .map_err(reserve_err_to_alloc_err)?;
        // SAFETY: freshly allocated buffer with exactly `capacity` elements.
        let boxed_uninit: Box<[MaybeUninit<T>], A> = unsafe { buf.into_box(capacity) };
        let (raw, alloc_out) = Box::into_non_null_with_allocator(boxed_uninit);
        let t_slice = {
            let base = raw.as_ptr().cast::<T>();
            ptr::slice_from_raw_parts_mut(base, capacity)
        };
        Ok(unsafe { Box::from_raw_in(t_slice, alloc_out) })
    }

    /// Creates a new boxed slice from a `&[T]` using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    // FIXME: remove this
    #[inline]
    pub fn try_from_slice_in(slice: &[T], alloc: A) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        let len = slice.len();
        let buf =
            RawVec::<T, A>::try_with_capacity_in(len, alloc).map_err(reserve_err_to_alloc_err)?;

        unsafe {
            let dst = buf.ptr();
            let src = slice.as_ptr();
            for i in 0..len {
                ptr::write(dst.add(i), (*src.add(i)).clone());
            }
        }

        // SAFETY: all `len` elements are now initialized.
        let boxed_uninit: Box<[MaybeUninit<T>], A> = unsafe { buf.into_box(len) };
        let (raw, alloc_out) = Box::into_non_null_with_allocator(boxed_uninit);
        let t_slice = {
            let base = raw.as_ptr().cast::<T>();
            ptr::slice_from_raw_parts_mut(base, len)
        };
        Ok(unsafe { Box::from_raw_in(t_slice, alloc_out) })
    }

    /// Fallibly clones a slice into a new boxed slice using the given allocator.
    ///
    /// Uses [`TryClone`] on each element rather than [`Clone`], so types that
    /// only implement the fallible trait can be boxed this way.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails, or propagates the first
    /// element-level [`TryCloneError`] encountered.
    // FIXME: use method 
    pub fn try_from_slice_try_clone_in(slice: &[T], alloc: A) -> Result<Self, TryCloneError>
    where
        T: TryClone,
    {
        let len = slice.len();
        let buf =
            RawVec::<T, A>::try_with_capacity_in(len, alloc).map_err(TryCloneError::Reserve)?;

        unsafe {
            let dst = buf.ptr();
            let src = slice.as_ptr();
            for i in 0..len {
                let cloned = (*src.add(i)).try_clone()?;
                ptr::write(dst.add(i), cloned);
            }
        }

        // SAFETY: all `len` elements are now initialized.
        let boxed_uninit: Box<[MaybeUninit<T>], A> = unsafe { buf.into_box(len) };
        let (raw, alloc_out) = Box::into_non_null_with_allocator(boxed_uninit);
        let t_slice = {
            let base = raw.as_ptr().cast::<T>();
            ptr::slice_from_raw_parts_mut(base, len)
        };
        Ok(unsafe { Box::from_raw_in(t_slice, alloc_out) })
    }
}

impl<T> Box<[T], Global> {
    /// Creates a new empty boxed slice.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_empty() -> Result<Self, AllocError> {
        Self::try_with_capacity(0)
    }

    /// Creates a new boxed slice with the given capacity (uninitialized).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, AllocError> {
        let buf = RawVec::<T>::try_with_capacity(capacity).map_err(reserve_err_to_alloc_err)?;
        // SAFETY: The buffer was just allocated with exactly `capacity`
        // elements. `into_box` wraps it as `Box<[MaybeUninit<T>]>`; we then
        // reinterpret the same memory as `Box<[T]>` since the layout is
        // identical. Contents remain uninitialized, matching std's behavior.
        let boxed_uninit: Box<[MaybeUninit<T>], Global> = unsafe { buf.into_box(capacity) };
        // Extract the raw fat pointer and allocator (consumes the box without
        // deallocating), then rebuild a `Box<[T]>` over the same allocation.
        let (raw, alloc) = Box::into_non_null_with_allocator(boxed_uninit);
        // SAFETY: the allocation holds `capacity` slots of size `T`;
        // reinterpreting `[MaybeUninit<T>]` as `[T]` is sound (same layout).
        let t_slice = {
            let base = raw.as_ptr().cast::<T>();
            ptr::slice_from_raw_parts_mut(base, capacity)
        };
        Ok(unsafe { Box::from_raw_in(t_slice, alloc) })
    }

    /// Creates a new boxed slice filled with zeros (for `Copy` types).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_zeroed(len: usize) -> Result<Self, AllocError>
    where
        T: Copy,
    {
        let buf = RawVec::<T>::try_with_capacity_zeroed(len).map_err(reserve_err_to_alloc_err)?;
        // SAFETY: buffer is zero-initialized; for `Copy` types this is a
        // valid (if unusual) value pattern.
        let boxed_uninit: Box<[MaybeUninit<T>], Global> = unsafe { buf.into_box(len) };
        let (raw, alloc) = Box::into_non_null_with_allocator(boxed_uninit);
        // SAFETY: all slots are zero-initialized and thus valid for `T`.
        let t_slice = {
            let base = raw.as_ptr().cast::<T>();
            ptr::slice_from_raw_parts_mut(base, len)
        };
        Ok(unsafe { Box::from_raw_in(t_slice, alloc) })
    }

    /// Creates a new boxed slice from a `&[T]`, cloning the elements.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_slice(slice: &[T]) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        let len = slice.len();
        let buf = RawVec::<T>::try_with_capacity(len).map_err(reserve_err_to_alloc_err)?;

        unsafe {
            let dst = buf.ptr();
            let src = slice.as_ptr();
            for i in 0..len {
                ptr::write(dst.add(i), (*src.add(i)).clone());
            }
        }

        // SAFETY: all `len` elements are now initialized.
        let boxed_uninit: Box<[MaybeUninit<T>], Global> = unsafe { buf.into_box(len) };
        let (raw, alloc) = Box::into_non_null_with_allocator(boxed_uninit);
        let t_slice = {
            let base = raw.as_ptr().cast::<T>();
            ptr::slice_from_raw_parts_mut(base, len)
        };
        Ok(unsafe { Box::from_raw_in(t_slice, alloc) })
    }

    /// Creates a new boxed slice from a `[T; N]` array, moving the elements.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_array<const N: usize>(array: [T; N]) -> Result<Self, AllocError> {
        let buf = RawVec::<T>::try_with_capacity(N).map_err(reserve_err_to_alloc_err)?;

        unsafe {
            let dst = buf.ptr();
            for i in 0..N {
                ptr::write(dst.add(i), ptr::read(array.as_ptr().add(i)));
            }
        }

        // SAFETY: all `N` elements are now initialized.
        let boxed_uninit: Box<[MaybeUninit<T>], Global> = unsafe { buf.into_box(N) };
        let (raw, alloc) = Box::into_non_null_with_allocator(boxed_uninit);
        let t_slice = {
            let base = raw.as_ptr().cast::<T>();
            ptr::slice_from_raw_parts_mut(base, N)
        };
        Ok(unsafe { Box::from_raw_in(t_slice, alloc) })
    }
}

// ---------------------------------------------------------------------------
// str-specific constructors: Box<str, A>
// ---------------------------------------------------------------------------

impl Box<str, Global> {
    /// Creates a new `Box<str>` from a `&str`, copying the bytes.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_str(s: &str) -> Result<Self, AllocError> {
        let bytes = s.as_bytes();
        let boxed_bytes: Box<[u8]> = Box::try_from_slice(bytes)?;
        // SAFETY: `s` is valid UTF-8, so the copied bytes are too.
        Ok(unsafe { from_boxed_utf8_unchecked(boxed_bytes) })
    }
}

impl<A: Allocator> Box<str, A> {
    /// Creates a new `Box<str>` from a `&str` using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_str_in(s: &str, alloc: A) -> Result<Self, AllocError> {
        let bytes = s.as_bytes();
        let boxed_bytes: Box<[u8], A> = Box::try_from_slice_in(bytes, alloc)?;
        // SAFETY: `s` is valid UTF-8.
        Ok(unsafe { from_boxed_utf8_unchecked(boxed_bytes) })
    }
}

/// Converts a `Box<[u8], A>` into a `Box<str, A>` without checking UTF-8 validity.
///
/// # Safety
///
/// The byte slice must be valid UTF-8.
pub unsafe fn from_boxed_utf8_unchecked<A: Allocator>(b: Box<[u8], A>) -> Box<str, A> {
    // SAFETY: caller guarantees the bytes are valid UTF-8. We extract the raw
    // fat pointer and allocator directly from the box's fields to avoid the
    // `Sized` requirement on `into_non_null_with_allocator`.
    unsafe {
        let inner = ptr::read(&b.inner);
        let alloc = ptr::read(&b.alloc);
        mem::forget(b);

        // The `NonNull<[u8]>` is a fat pointer with (ptr, len) metadata.
        // Coerce it to `NonNull<str>` which also has (ptr, len) metadata.
        // This is sound because `str` and `[u8]` have identical layout.
        let str_nn: NonNull<str> = mem::transmute(inner);
        Box::from_raw_in(str_nn.as_ptr(), alloc)
    }
}

// ---------------------------------------------------------------------------
// Helpers: error conversion and OOM handling
// ---------------------------------------------------------------------------

/// Converts a `TryReserveError` from `RawVec` into an `AllocError`.
pub(crate) fn reserve_err_to_alloc_err(_e: TryReserveError) -> AllocError {
    AllocError
}

#[cfg(test)]
mod tests;
