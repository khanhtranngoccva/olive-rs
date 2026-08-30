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

use core::any::Any;
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
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
        // FIXME:
        let me = ManuallyDrop::new(self);
        unsafe { mem::transmute_copy::<ManuallyDrop<Self>, Box<T, A>>(&me) }
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
        // Drop the contained value in place. For a ZST (including a ZST trait
        // object) this is a no-op, and there is no backing allocation to free.
        unsafe {
            ptr::drop_in_place(self.inner.as_ptr());
        }
        // Deallocate only if we actually allocated something. A zero-sized
        // value (e.g. a `dyn Trait` whose concrete type is a ZST) has no heap
        // block — its vtable lives in the fat pointer's metadata, not on the
        // heap — so deallocating a 0-byte "block" would corrupt the allocator.
        if layout.size() != 0 {
            unsafe {
                self.alloc.deallocate(self.inner.cast(), layout);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Construction (unsized T — slices, str, dyn)
// ---------------------------------------------------------------------------

impl<T: ?Sized + TryCloneToUninit> Box<T, Global> {
    /// Clones a `&T` into a freshly allocated `Box<T, Global>` for unsized `T`.
    ///
    /// For `T = [U]`, this clones the slice contents onto the heap.
    /// For `T = str`, this copies the string bytes onto the heap.
    /// For `T = CStr`, this copies the null-terminated byte string.
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
    /// Clones a `&T` into a freshly allocated `Box<T, A>` for unsized `T`,
    /// using the given allocator.
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

// ---------------------------------------------------------------------------
// Slice-specific constructors: Box<[T], A>
// ---------------------------------------------------------------------------

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
        let t_slice = unsafe {
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
        let t_slice = unsafe {
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
        let t_slice = unsafe {
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
        let t_slice = unsafe {
            let base = raw.as_ptr().cast::<T>();
            ptr::slice_from_raw_parts_mut(base, N)
        };
        Ok(unsafe { Box::from_raw_in(t_slice, alloc) })
    }
}

impl<T, A: Allocator> Box<[T], A> {
    /// Creates a new empty boxed slice using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
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
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, AllocError> {
        let buf = RawVec::<T, A>::try_with_capacity_in(capacity, alloc)
            .map_err(reserve_err_to_alloc_err)?;
        // SAFETY: freshly allocated buffer with exactly `capacity` elements.
        let boxed_uninit: Box<[MaybeUninit<T>], A> = unsafe { buf.into_box(capacity) };
        let (raw, alloc_out) = Box::into_non_null_with_allocator(boxed_uninit);
        let t_slice = unsafe {
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
        let t_slice = unsafe {
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
    pub fn try_from_slice_try_clone_in(slice: &[T], alloc: A) -> Result<Self, TryCloneError>
    where
        T: TryClone,
    {
        let len = slice.len();
        let buf = RawVec::<T, A>::try_with_capacity_in(len, alloc)
            .map_err(|e| TryCloneError::Reserve(e))?;

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
        let t_slice = unsafe {
            let base = raw.as_ptr().cast::<T>();
            ptr::slice_from_raw_parts_mut(base, len)
        };
        Ok(unsafe { Box::from_raw_in(t_slice, alloc_out) })
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
// Slice-specific accessors: Box<[T], A>
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Box<[T], A> {
    /// Gets the number of elements in the boxed slice.
    #[inline]
    pub fn len(&self) -> usize {
        (**self).len()
    }

    /// Returns `true` if the boxed slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets a shared reference to the inner slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &**self
    }

    /// Gets a mutable reference to the inner slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut **self
    }
}

// ---------------------------------------------------------------------------
// Str-specific accessors: Box<str, A>
// ---------------------------------------------------------------------------

impl<A: Allocator> Box<str, A> {
    /// Gets the length of the string in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        (**self).len()
    }

    /// Returns `true` if the string is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets a shared reference to the inner `str`.
    #[inline]
    pub fn as_str(&self) -> &str {
        &**self
    }
}

// ---------------------------------------------------------------------------
// Deref / DerefMut
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Deref for Box<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `inner` is always a valid, aligned, non-null pointer to an
        // initialized `T` (or a dangling pointer for ZSTs, which is fine for
        // `&T` since no read occurs).
        unsafe { self.inner.as_ref() }
    }
}

impl<T: ?Sized, A: Allocator> DerefMut for Box<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: same as Deref.
        unsafe { self.inner.as_mut() }
    }
}

// ---------------------------------------------------------------------------
// TryClone
// ---------------------------------------------------------------------------

impl<T: TryClone, A: Allocator + Clone> TryClone for Box<T, A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let cloned_inner = (**self).try_clone()?;
        Box::try_new_in(cloned_inner, self.alloc.clone()).map_err(TryCloneError::Alloc)
    }
}

impl<T: TryClone, A: Allocator + Clone> TryClone for Box<[T], A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let slice: &[T] = self;
        Box::try_from_slice_try_clone_in(slice, self.alloc.clone())
    }
}

impl<A: Allocator + Clone> TryClone for Box<str, A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let s: &str = self;
        Box::try_from_str_in(s, self.alloc.clone()).map_err(TryCloneError::Alloc)
    }
}

// ---------------------------------------------------------------------------
// Debug / Display
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> fmt::Pointer for Box<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.inner.as_ptr(), f)
    }
}

impl<T: Debug + ?Sized, A: Allocator> Debug for Box<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T: Display + ?Sized, A: Allocator> Display for Box<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

// ---------------------------------------------------------------------------
// PartialEq / Eq / PartialOrd / Ord / Hash
// ---------------------------------------------------------------------------

impl<T: PartialEq + ?Sized, A: Allocator> PartialEq for Box<T, A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: Eq + ?Sized, A: Allocator> Eq for Box<T, A> {}

impl<T: PartialOrd + ?Sized, A: Allocator> PartialOrd for Box<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: Ord + ?Sized, A: Allocator> Ord for Box<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: Hash + ?Sized, A: Allocator> Hash for Box<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

// ---------------------------------------------------------------------------
// AsRef / AsMut / Borrow / BorrowMut / ToOwned-style
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> AsRef<T> for Box<T, A> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized, A: Allocator> AsMut<T> for Box<T, A> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T: ?Sized, A: Allocator> Borrow<T> for Box<T, A> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized, A: Allocator> BorrowMut<T> for Box<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}

// ---------------------------------------------------------------------------
// From / Into conversions
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> From<Box<T, A>> for Pin<Box<T, A>>
where
    A: StaticAllocator,
{
    /// Converts a `Box<T>` into a `Pin<Box<T>>`. If `T` does not implement [`Unpin`], then
    /// `*boxed` will be pinned in memory and unable to be moved.
    ///
    /// This conversion does not allocate on the heap and happens in place.
    ///
    /// This is also available via [`Box::into_pin`].
    fn from(boxed: Box<T, A>) -> Self {
        Box::into_pin(boxed)
    }
}

impl<A: Allocator> From<Box<str, A>> for Box<[u8], A> {
    /// Converts a `Box<str>` into a `Box<[u8]>`.
    ///
    /// This conversion does not allocate on the heap and happens in place.
    fn from(s: Box<str, A>) -> Self {
        unsafe {
            // Extract fields directly to avoid the `Sized` requirement.
            let inner = ptr::read(&s.inner);
            let alloc = ptr::read(&s.alloc);
            mem::forget(s);

            // SAFETY: a `str` is just a sequence of `u8`s with the same layout.
            // Reinterpret the fat pointer (address + length metadata) as
            // `NonNull<[u8]>`, which has identical representation.
            let nn_bytes: NonNull<[u8]> = mem::transmute(inner);
            Box::from_raw_in(nn_bytes.as_ptr(), alloc)
        }
    }
}

// ---------------------------------------------------------------------------
// TryFrom: Box<[T]> → Box<[T; N]>
// ---------------------------------------------------------------------------

/// Casts a boxed slice to a boxed array.
///
/// # Safety
///
/// `boxed_slice.len()` must be exactly `N`.
unsafe fn boxed_slice_as_array_unchecked<T, A: Allocator, const N: usize>(
    boxed_slice: Box<[T], A>,
) -> Box<[T; N], A> {
    debug_assert_eq!(boxed_slice.len(), N);

    unsafe {
        // Extract fields directly to avoid the `Sized` requirement.
        let inner = ptr::read(&boxed_slice.inner);
        let alloc = ptr::read(&boxed_slice.alloc);
        mem::forget(boxed_slice);

        // SAFETY: Pointer and allocator came from an existing box,
        // and our safety condition requires that the length is exactly `N`.
        // A slice of length N has the same layout as an array [T; N]; drop
        // the (redundant) length metadata by re-pointing at `[T; N]`.
        let arr_ptr: *mut [T; N] = inner.as_ptr() as *mut u8 as *mut [T; N];
        Box::from_raw_in(arr_ptr, alloc)
    }
}

impl<T, const N: usize> TryFrom<Box<[T], Global>> for Box<[T; N], Global> {
    type Error = Box<[T], Global>;

    /// Attempts to convert a `Box<[T]>` into a `Box<[T; N]>`.
    ///
    /// The conversion occurs in-place and does not require a new memory
    /// allocation.
    ///
    /// # Errors
    ///
    /// Returns the old `Box<[T]>` in the `Err` variant if
    /// `boxed_slice.len()` does not equal `N`.
    fn try_from(boxed_slice: Box<[T], Global>) -> Result<Self, Self::Error> {
        if boxed_slice.len() == N {
            Ok(unsafe { boxed_slice_as_array_unchecked(boxed_slice) })
        } else {
            Err(boxed_slice)
        }
    }
}

// ---------------------------------------------------------------------------
// Any downcasting
// ---------------------------------------------------------------------------

impl<A: Allocator> Box<dyn Any, A> {
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Any>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                // Extract the base address from the fat pointer.
                let base_addr = self.inner.as_ptr() as *mut u8;
                let alloc = ptr::read(&self.alloc);
                mem::forget(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                // Casting the thin base address to `*mut T` reinterprets the
                // same address bits as a pointer to `T`, which is sound
                // because the allocation contains a valid `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Any + Send, A> {
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Any>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                let base_addr = self.inner.as_ptr() as *mut u8;
                let alloc = ptr::read(&self.alloc);
                mem::forget(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Any + Send + Sync, A> {
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Any>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                let base_addr = self.inner.as_ptr() as *mut u8;
                let alloc = ptr::read(&self.alloc);
                mem::forget(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Error, A> {
    /// Attempts to downcast the box to a concrete error type.
    pub fn downcast<T: Error + 'static>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                let base_addr = self.inner.as_ptr() as *mut u8;
                let alloc = ptr::read(&self.alloc);
                mem::forget(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Error + Send, A> {
    /// Attempts to downcast the box to a concrete error type.
    pub fn downcast<T: Error + 'static>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                let base_addr = self.inner.as_ptr() as *mut u8;
                let alloc = ptr::read(&self.alloc);
                mem::forget(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Allocator> Box<dyn Error + Send + Sync, A> {
    /// Attempts to downcast the box to a concrete error type.
    pub fn downcast<T: Error + 'static>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                let base_addr = self.inner.as_ptr() as *mut u8;
                let alloc = ptr::read(&self.alloc);
                mem::forget(self);
                // SAFETY: `is::<T>()` confirmed the dynamic type is `T`.
                let t_ptr: *mut T = base_addr.cast::<T>();
                Ok(Box::from_raw_in(t_ptr, alloc))
            }
        } else {
            Err(self)
        }
    }
}

// ---------------------------------------------------------------------------
// Iterator forwarding for Box<I, A> where I: Iterator + ?Sized
// ---------------------------------------------------------------------------

impl<I: Iterator + ?Sized, A: Allocator> Iterator for Box<I, A> {
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (**self).next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        (**self).nth(n)
    }
}

impl<I: DoubleEndedIterator + ?Sized, A: Allocator> DoubleEndedIterator for Box<I, A> {
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> {
        (**self).next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        (**self).nth_back(n)
    }
}

impl<I: core::iter::FusedIterator + ?Sized, A: Allocator> core::iter::FusedIterator for Box<I, A> {}

impl<I: ExactSizeIterator + ?Sized, A: Allocator> ExactSizeIterator for Box<I, A> {}

// ---------------------------------------------------------------------------
// Helpers: error conversion and OOM handling
// ---------------------------------------------------------------------------

/// Converts a `TryReserveError` from `RawVec` into an `AllocError`.
fn reserve_err_to_alloc_err(_e: TryReserveError) -> AllocError {
    AllocError
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use olive_core::TryClone;

    use super::*;
    use std::format;
    use std::string::String;
    use std::vec::Vec;

    // Local `vec!` shim: std's `vec!` macro isn't in scope without a prelude,
    // and importing it would collide with the `std::vec` module name.
    macro_rules! vec {
        ($($x:expr),* $(,)?) => {{
            let mut v = <std::vec::Vec<_>>::new();
            $(v.push($x);)*
            v
        }};
    }

    #[test]
    fn test_box_try_new_and_deref() {
        let b = Box::try_new(42i32).unwrap();
        assert_eq!(*b, 42);
    }

    #[test]
    fn test_box_try_new_ok() {
        let b = Box::try_new(99u64).unwrap();
        assert_eq!(*b, 99);
    }

    #[test]
    fn test_box_zst() {
        struct Zst;
        let b = Box::try_new(Zst).unwrap();
        // Should not allocate; should work fine.
        let _ = &b;
    }

    #[test]
    fn test_box_into_raw_from_raw_roundtrip() {
        let b = Box::try_new(String::from("hello")).unwrap();
        let raw: *mut String = Box::into_raw(b);
        assert_eq!(unsafe { &*raw }, "hello");
        // Reconstruct from the raw pointer using an explicit type ascription
        // so inference can resolve both `T` and `A`.
        let b: Box<String, Global> = unsafe { Box::from_raw(raw) };
        assert_eq!(&**b, "hello");
    }

    #[test]
    fn test_box_leak() {
        let b = Box::try_new(vec![1, 2, 3]).unwrap();
        let leaked: &'static mut Vec<i32> = Box::leak(b);
        assert_eq!(leaked, &vec![1, 2, 3]);
    }

    #[test]
    fn test_box_try_clone() {
        let b1 = Box::try_new(5u32).unwrap();
        let b2: Box<u32> = b1.try_clone().unwrap();
        assert_eq!(*b1, *b2);
        assert_ne!(Box::as_ptr(&b1), Box::as_ptr(&b2));
    }

    #[test]
    fn test_box_debug_display() {
        let b = Box::try_new(format_test_string()).unwrap();
        assert_eq!(format!("{b:?}"), "\"hi\"");
        assert_eq!(format!("{b}"), "hi");
    }

    fn format_test_string() -> String {
        String::from("hi")
    }

    #[test]
    fn test_box_equality_ordering_hash() {
        let a = Box::try_new(10i32).unwrap();
        let b = Box::try_new(10i32).unwrap();
        let c = Box::try_new(20i32).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a < c);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&10i32));
    }

    #[test]
    fn test_box_asref_asmut_borrow() {
        let mut b = Box::try_new(7u16).unwrap();
        let r: &u16 = b.as_ref();
        assert_eq!(*r, 7);
        let m: &mut u16 = b.as_mut();
        *m = 8;
        assert_eq!(*b, 8);
        let br: &u16 = (&b).borrow();
        assert_eq!(*br, 8);
        let bm: &mut u16 = (&mut b).borrow_mut();
        *bm = 9;
        assert_eq!(*b, 9);
    }

    #[test]
    fn test_box_try_pin() {
        let p: core::pin::Pin<Box<u8, Global>> =
            Box::<u8, Global>::try_pin_in(31u8, Global).unwrap();
        assert_eq!(*p, 31);
    }

    #[test]
    fn test_box_try_pin_global() {
        let p: core::pin::Pin<Box<u8>> = Box::try_pin(31u8).unwrap();
        assert_eq!(*p, 31);
    }

    #[test]
    fn test_box_try_new_give_back_success() {
        let b = Box::try_new_give_back(42u32).unwrap();
        assert_eq!(*b, 42);
    }

    #[test]
    fn test_box_try_pin_give_back_success() {
        let p: core::pin::Pin<Box<u32>> = Box::try_pin_give_back(7u32).unwrap();
        assert_eq!(*p, 7);
    }

    #[test]
    fn test_box_try_new_give_back_in_success() {
        let b = Box::<u32, Global>::try_new_give_back_in(42u32, Global).unwrap();
        assert_eq!(*b, 42);
    }

    #[test]
    fn test_box_try_pin_give_back_in_success() {
        let p: core::pin::Pin<Box<u32, Global>> =
            Box::<u32, Global>::try_pin_give_back_in(7u32, Global).unwrap();
        assert_eq!(*p, 7);
    }

    #[test]
    fn test_box_uninit_write_and_assume_init() {
        // write(): initialize via the consuming helper.
        let b: Box<MaybeUninit<u64>> = Box::try_new_uninit().unwrap();
        let init = b.write(99u64);
        assert_eq!(*init, 99);

        // assume_init(): manually fill then reinterpret.
        let mut u: Box<MaybeUninit<u64>> = Box::try_new_uninit().unwrap();
        unsafe { (*u).as_mut_ptr().cast::<u64>().write(1234u64) };
        let init = unsafe { u.assume_init() };
        assert_eq!(*init, 1234);
    }

    #[test]
    fn test_box_slice_try_from_slice() {
        let src = [1, 2, 3, 4, 5];
        let bs: Box<[i32]> = Box::try_from_slice(&src).unwrap();
        assert_eq!(bs.len(), 5);
        assert_eq!(bs[0], 1);
        assert_eq!(bs[4], 5);
    }

    #[test]
    fn test_box_slice_try_from_slice_chars() {
        let src = ['a', 'b', 'c'];
        let bs = Box::try_from_slice(&src).unwrap();
        assert_eq!(bs.len(), 3);
        assert_eq!(bs[1], 'b');
    }

    #[test]
    fn test_box_slice_try_from_array() {
        let bs: Box<[u8]> = Box::try_from_array([10, 20, 30]).unwrap();
        assert_eq!(bs.len(), 3);
        assert_eq!(bs[2], 30);
    }

    #[test]
    fn test_box_slice_try_with_capacity() {
        let bs: Box<[u8]> = Box::try_with_capacity(10).unwrap();
        assert_eq!(bs.len(), 10);
    }

    #[test]
    fn test_box_slice_try_new_empty() {
        let bs: Box<[i32]> = Box::try_new_empty().unwrap();
        assert!(bs.is_empty());
    }

    #[test]
    fn test_box_slice_try_clone() {
        let orig: Box<[i32]> = Box::try_from_slice(&[1, 2, 3]).unwrap();
        let cloned: Box<[i32]> = orig.try_clone().unwrap();
        assert_eq!(orig, cloned);
        assert_ne!(Box::as_ptr(&orig), Box::as_ptr(&cloned));
    }

    #[test]
    fn test_box_str_try_from_str() {
        let bs: Box<str> = Box::try_from_str("hello world").unwrap();
        assert_eq!(&*bs, "hello world");
        assert_eq!(bs.len(), 11);
    }

    #[test]
    fn test_box_str_try_clone() {
        let orig: Box<str> = Box::try_from_str("abc").unwrap();
        let cloned: Box<str> = orig.try_clone().unwrap();
        assert_eq!(orig, cloned);
    }

    #[test]
    fn test_box_str_to_bytes() {
        let bs: Box<str> = Box::try_from_str("xyz").unwrap();
        let bytes: Box<[u8]> = Box::from(bs);
        assert_eq!(&bytes[..], b"xyz");
    }

    #[test]
    fn test_box_dyn_trait_object() {
        trait Greet {
            fn greet(&self) -> String;
        }
        // Non-zero-sized payload so the boxed trait object owns real data on
        // the heap (the vtable itself lives in the fat pointer's metadata).
        struct Dog {
            tag: u64,
        };
        impl Greet for Dog {
            fn greet(&self) -> String {
                String::from("woof")
            }
        }
        // Box the concrete type, then unsize-coerce to a trait object through a
        // raw fat pointer. The concrete allocation carries the payload and the
        // coercion attaches the vtable — the sound way to get a `Box<dyn Trait>`.
        let boxed: Box<Dog, Global> = Box::try_new(Dog { tag: 1 }).unwrap();
        let raw: *mut Dog = Box::into_raw(boxed);
        let fat: *mut dyn Greet = raw as *mut dyn Greet;
        let b: Box<dyn Greet, Global> = unsafe { Box::from_raw_in(fat, Global) };
        assert_eq!(b.greet(), "woof");
    }

    #[test]
    fn test_box_dyn_zst_concrete() {
        trait Greet {
            fn greet(&self) -> String;
        }
        struct Empty; // ZST
        impl Greet for Empty {
            fn greet(&self) -> String {
                String::from("...")
            }
        }
        // A ZST behind a trait object has no data bytes on the heap; only the
        // vtable (in the fat-pointer metadata) distinguishes it. Boxing the
        // concrete value and unsize-coercing through a raw fat pointer yields a
        // working `Box<dyn Greet>` whose methods dispatch via the vtable.
        let boxed: Box<Empty, Global> = Box::try_new(Empty).unwrap();
        let raw: *mut Empty = Box::into_raw(boxed);
        let fat: *mut dyn Greet = raw as *mut dyn Greet;
        let b: Box<dyn Greet, Global> = unsafe { Box::from_raw_in(fat, Global) };
        assert_eq!(b.greet(), "...");
    }

    #[test]
    fn test_box_any_downcast() {
        // Build a `Box<dyn Any>` by boxing the concrete value and unsize-coercing
        // through a raw fat pointer (our `Box::try_new` is sized-only).
        let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
        let raw: *mut i32 = Box::into_raw(boxed_i32);
        let fat: *mut dyn Any = raw as *mut dyn Any;
        let b: Box<dyn Any, Global> = unsafe { Box::from_raw_in(fat, Global) };
        let recovered = b.downcast::<i32>().unwrap();
        assert_eq!(*recovered, 42);

        let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
        let raw: *mut i32 = Box::into_raw(boxed_i32);
        let fat: *mut dyn Any = raw as *mut dyn Any;
        let b: Box<dyn Any, Global> = unsafe { Box::from_raw_in(fat, Global) };
        assert!(b.downcast::<f64>().is_err());
    }

    #[test]
    fn test_box_error_downcast() {
        #[derive(Debug)]
        struct MyErr;
        impl fmt::Display for MyErr {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "my error")
            }
        }
        impl Error for MyErr {}

        // Box the concrete error and unsize-coerce through a raw fat pointer.
        let boxed: Box<MyErr, Global> = Box::try_new(MyErr).unwrap();
        let raw: *mut MyErr = Box::into_raw(boxed);
        let fat: *mut dyn Error = raw as *mut dyn Error;
        let e: Box<dyn Error, Global> = unsafe { Box::from_raw_in(fat, Global) };
        let recovered = e.downcast::<MyErr>().unwrap();
        let _ = recovered; // just verify it works
    }

    #[test]
    fn test_box_iterator_forwarding() {
        let v = vec![1, 2, 3, 4, 5];
        let iter: Box<std::slice::Iter<'_, i32>> = Box::try_new(v.iter()).unwrap();
        let collected: Vec<&i32> = iter.collect();
        assert_eq!(collected, vec![&1, &2, &3, &4, &5]);
    }

    #[test]
    fn test_box_double_ended_iterator() {
        let v = vec![1, 2, 3, 4, 5];
        let mut iter: Box<std::slice::Iter<'_, i32>> = Box::try_new(v.iter()).unwrap();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&5));
        assert_eq!(iter.next(), Some(&2));
    }

    #[test]
    fn test_box_custom_allocator_global() {
        // Verify that Box works with an explicit `Global` allocator; the
        // stored allocator must be the same unit struct we passed in.
        let b: Box<u32, Global> = Box::try_new_in(77u32, Global).unwrap();
        assert_eq!(*b, 77);
        assert!(matches!(b.allocator(), Global));
    }

    #[test]
    fn test_box_try_new_in() {
        let b = Box::try_new_in(55u16, Global).unwrap();
        assert_eq!(*b, 55);
    }

    #[test]
    fn test_box_slice_try_from() {
        let bs: Box<[i32]> = Box::try_from_slice(&[1, 2, 3]).unwrap();
        let arr: Box<[i32; 3]> = Box::try_from(bs).unwrap();
        assert_eq!(arr[0], 1);
        assert_eq!(arr[2], 3);

        let bs: Box<[i32]> = Box::try_from_slice(&[1, 2]).unwrap();
        let result: Result<Box<[i32; 3]>, Box<[i32]>> = Box::try_from(bs);
        assert!(result.is_err());
    }

    #[test]
    fn test_box_pointer_fmt() {
        let b = Box::try_new(0u8).unwrap();
        let _ = format!("{b:p}");
    }
}
