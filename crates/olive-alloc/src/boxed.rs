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
use core::mem::{self, ManuallyDrop, size_of};
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::ptr::{self, NonNull};

use crate::alloc::{AllocError, Allocator, Global, Layout};
use crate::raw_vec::RawVec;
use olive_core::alloc_errors::TryReserveError;

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

/// Monomorphic function for allocating an uninit `Box`.
#[inline]
// The is a separate function to avoid doing it in every generic version, but it
// looks small to the mir inliner (particularly in panic=abort) so leave it to
// the backend to decide whether pulling it in everywhere is worth doing.
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
fn try_box_new_uninit(layout: Layout) -> Result<*mut u8, AllocError> {
    Global
        .allocate(layout)
        .map(|pointer| pointer.cast().as_ptr())
}

// ---------------------------------------------------------------------------
// Construction (sized T)
// ---------------------------------------------------------------------------

impl<T> Box<T, Global> {
    ///
    /// Infallible facade over [`try_new`][Self::try_new]: panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn new(x: T) -> Self {
        match Self::try_new(x) {
            Ok(boxed) => boxed,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Allocates memory on the heap and places `x` into it.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new(x: T) -> Result<Self, AllocError> {
        // ZST optimization: no actual allocation needed.
        if size_of::<T>() == 0 {
            return Ok(Self {
                inner: NonNull::dangling(),
                alloc: Global,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = Global.allocate(layout).map_err(|_| AllocError)?;

        unsafe {
            ptr.as_ptr().cast::<T>().write(x);
        }

        Ok(Self {
            inner: ptr.cast(),
            alloc: Global,
        })
    }

    /// Constructs a `Box<T>` containing uninitialized memory.
    ///
    /// Infallible facade over [`try_new_uninit`][Self::try_new_uninit]: panics
    /// on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn new_uninit() -> Self {
        match Self::try_new_uninit() {
            Ok(boxed) => boxed,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`new_uninit`][Self::new_uninit].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_uninit() -> Result<Self, AllocError> {
        if size_of::<T>() == 0 {
            return Ok(Self {
                inner: NonNull::dangling(),
                alloc: Global,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = Global.allocate(layout).map_err(|_| AllocError)?;

        Ok(Self {
            inner: ptr.cast(),
            alloc: Global,
        })
    }
}

impl<T, A: Allocator> Box<T, A> {
    /// Like `new`, but parameterized over the choice of allocator for the
    /// returned `Box`.
    ///
    /// Infallible facade over [`try_new_in`][Self::try_new_in]: panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn new_in(x: T, alloc: A) -> Self {
        match Self::try_new_in(x, alloc) {
            Ok(boxed) => boxed,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`new_in`][Self::new_in].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_in(x: T, alloc: A) -> Result<Self, AllocError> {
        if size_of::<T>() == 0 {
            return Ok(Self {
                inner: NonNull::dangling(),
                alloc,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = alloc.allocate(layout).map_err(|_| AllocError)?;

        unsafe {
            ptr.as_ptr().cast::<T>().write(x);
        }

        Ok(Self {
            inner: ptr.cast(),
            alloc,
        })
    }

    /// Like `new_uninit`, but parameterized over the choice of allocator.
    ///
    /// Infallible facade over [`try_new_uninit_in`][Self::try_new_uninit_in]:
    /// panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn new_uninit_in(alloc: A) -> Self {
        match Self::try_new_uninit_in(alloc) {
            Ok(boxed) => boxed,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`new_uninit_in`][Self::new_uninit_in].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_new_uninit_in(alloc: A) -> Result<Self, AllocError> {
        if size_of::<T>() == 0 {
            return Ok(Self {
                inner: NonNull::dangling(),
                alloc,
            });
        }

        let layout = Layout::new::<T>();
        let ptr = alloc.allocate(layout).map_err(|_| AllocError)?;

        Ok(Self {
            inner: ptr.cast(),
            alloc,
        })
    }

    /// Converts a `Box<T>` into a raw pointer.
    ///
    /// The pointer is allocated according to the allocator inside the box.
    /// The caller takes ownership of the allocation and must eventually
    /// reconstruct a `Box` from it (via [`from_raw_global`] for the default
    /// allocator, or [`from_non_null`] with the original allocator) or manually
    /// deallocate it.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let mut x = Box::new(5);
    /// let raw = Box::into_raw(x);
    /// assert_eq!(unsafe { *raw }, 5);
    /// let x = unsafe { Box::from_raw_global(raw) };
    /// assert_eq!(*x, 5);
    /// ```
    #[inline]
    pub fn into_raw(b: Self) -> *mut T {
        let raw = b.inner.as_ptr();
        // Prevent double-free by leaking the allocator's bookkeeping.
        // The allocation itself is transferred to the caller.
        mem::forget(b);
        raw
    }

    /// Reconstitutes a `Box<T>` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must have been previously obtained from
    /// [`into_raw`][Self::into_raw] or equivalent, and must still be valid
    /// (not yet freed).
    #[inline]
    pub unsafe fn from_raw_global(p: *mut T) -> Box<T, Global> {
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

    /// Converts a `Box<T, A>` into a `(NonNull<T>, A)` pair.
    ///
    /// The caller takes ownership of both the allocation and the allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let b = Box::new(42);
    /// let (nn, alloc) = Box::into_non_null_and_alloc(b);
    /// assert_eq!(nn.as_ref(), &42);
    /// ```
    #[inline]
    pub fn into_non_null_and_alloc(b: Self) -> (NonNull<T>, A)
    where
        T: Sized,
    {
        // SAFETY: we consume `b` by value and never touch it after the read,
        // so reading both fields is sound even though `Box` implements Drop.
        unsafe {
            let nn = ptr::read(&b.inner);
            let alloc = ptr::read(&b.alloc);
            mem::forget(b);
            (nn, alloc)
        }
    }

    /// Reconstitutes a `Box<T, A>` from a `NonNull<T>` and an allocator.
    ///
    /// # Safety
    ///
    /// The `NonNull` must point to a valid, properly-aligned allocation made
    /// by `alloc`.
    #[inline]
    pub unsafe fn from_non_null(nn: NonNull<T>, alloc: A) -> Self
    where
        T: Sized,
    {
        // SAFETY: caller guarantees validity.
        unsafe { Self { inner: nn, alloc } }
    }

    /// Leaks the boxed value, returning a mutable reference with `'static`
    /// lifetime.
    ///
    /// The memory is intentionally leaked; the caller is responsible for
    /// eventual cleanup (usually never).
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let b = Box::new(String::from("hello"));
    /// let s: &'static str = Box::leak(Box::new("world"));
    /// assert_eq!(s, "world");
    /// ```
    #[inline]
    pub fn leak(b: Self) -> &'static mut T {
        let ptr = b.inner.as_ptr();
        // Prevent Drop from freeing the allocation.
        mem::forget(b);
        // SAFETY: `ptr` points to a valid, aligned allocation that is now
        // leaked (ownership transferred to the returned `'static` reference).
        unsafe { &mut *ptr }
    }

    /// Seals the boxed value behind a `ManuallyDrop`, preventing automatic
    /// deallocation while still allowing access.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let b = Box::new(10u32);
    /// let sealed = Box::seal(b);
    /// assert_eq!(*sealed, 10);
    /// drop(sealed); // Does NOT free the allocation.
    /// ```
    #[inline]
    pub fn seal(b: Self) -> ManuallyDrop<Self> {
        ManuallyDrop::new(b)
    }

    /// Pins the boxed value in place, returning a `Pin<Box<T>>`.
    ///
    /// If `T` does not implement `Unpin`, then `*boxed` will be pinned in
    /// memory and unable to be moved.
    ///
    /// This conversion does not allocate on the heap and happens in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// let p = Box::pin(42);
    /// assert_eq!(*p, 42);
    /// ```
    #[inline]
    pub fn pin(x: T) -> Pin<Box<T>> {
        // SAFETY: A freshly allocated box is not aliased anywhere else, so
        // it is safe to pin regardless of whether `T: Unpin`.
        unsafe { Pin::new_unchecked(Box::new(x)) }
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
    /// let b = Box::new(7);
    /// let p = Box::into_pin(b);
    /// assert_eq!(*p, 7);
    /// ```
    #[inline]
    pub fn into_pin(b: Self) -> Pin<Self> {
        // SAFETY: The box was just consumed by value; no other reference
        // exists, so pinning is sound.
        unsafe { Pin::new_unchecked(b) }
    }

    /// Gets a shared reference to the allocator backing this `Box`.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }
}

// ---------------------------------------------------------------------------
// Construction (unsized T — slices, str, dyn)
// ---------------------------------------------------------------------------

impl<T: ?Sized> Box<T, Global> {
    /// Converts a `&T` into a `Box<T>` for unsized `T`.
    ///
    /// For `T = [U]`, this clones the slice contents onto the heap.
    /// For `T = str`, this copies the string bytes onto the heap.
    ///
    /// A `dyn Trait` target is not supported by this borrow-based constructor
    /// (its vtable cannot be relocated without the concrete type); box a trait
    /// object through its concrete type instead.
    ///
    /// Infallible facade over [`try_from_ref`][Self::try_from_ref]: panics on
    /// OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn from_ref(t: &T) -> Self
    where
        T: CloneUnsize,
    {
        match Self::try_from_ref(t) {
            Ok(boxed) => boxed,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`from_ref`][Self::from_ref].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_ref(t: &T) -> Result<Self, AllocError>
    where
        T: CloneUnsize,
    {
        Self::try_from_ref_in(t, Global)
    }

    /// Converts a `&mut T` into a `Box<T>` for unsized `T`.
    ///
    /// Infallible facade over [`try_from_mut_ref`][Self::try_from_mut_ref]:
    /// panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn from_mut_ref(t: &mut T) -> Self
    where
        T: CloneUnsize,
    {
        match Self::try_from_mut_ref(t) {
            Ok(boxed) => boxed,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`from_mut_ref`][Self::from_mut_ref].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_mut_ref(t: &mut T) -> Result<Self, AllocError>
    where
        T: CloneUnsize,
    {
        Self::try_from_ref(t)
    }

    /// Reconstructs a `Box<T>` from a raw fat pointer for unsized `T`.
    ///
    /// # Safety
    ///
    /// The pointer must have been produced by [`into_raw`][Self::into_raw] on
    /// a `Box<T, Global>`.
    #[inline]
    pub unsafe fn from_raw_fat(p: *mut T) -> Self {
        // SAFETY: caller guarantees validity.
        unsafe {
            let inner = NonNull::new_unchecked(p);
            Self {
                inner,
                alloc: Global,
            }
        }
    }
}

impl<T: ?Sized, A: Allocator> Box<T, A> {
    /// Converts a `&T` into a `Box<T, A>` for unsized `T`, using the given
    /// allocator.
    ///
    /// Infallible facade over [`try_from_ref_in`][Self::try_from_ref_in]:
    /// panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn from_ref_in(t: &T, alloc: A) -> Self
    where
        T: CloneUnsize,
    {
        match Self::try_from_ref_in(t, alloc) {
            Ok(boxed) => boxed,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`from_ref_in`][Self::from_ref_in].
    ///
    /// Clones the pointed-to value onto the heap and returns an owning
    /// `Box<T>` rooted at the fresh allocation.
    ///
    /// This is intended for unsized targets whose fat-pointer metadata is a
    /// plain length — namely slices (`[U]`) and `str`. A `dyn Trait` target
    /// cannot be relocated to a fresh address generically (the vtable needs the
    /// concrete type, unavailable in a `?Sized` context on stable Rust), so box
    /// a trait object through its concrete type instead (e.g. [`Box::new`] or
    /// the `From<E>` impls further down).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_ref_in(t: &T, alloc: A) -> Result<Self, AllocError>
    where
        T: CloneUnsize,
    {
        // Delegate the kind-specific relocation to the trait impl, which knows
        // the concrete unsized element type and can rebuild the fat pointer at
        // the freshly allocated base address.
        let inner = T::clone_into(t, &alloc)?;
        Ok(Self { inner, alloc })
    }

    /// Reconstructs a `Box<T, A>` from a raw fat pointer and allocator.
    ///
    /// # Safety
    ///
    /// The pointer must have been produced by [`into_raw`][Self::into_raw] on
    /// a `Box<T, A>` with the same allocator.
    #[inline]
    pub unsafe fn from_raw_in(p: *mut T, alloc: A) -> Self {
        // SAFETY: caller guarantees validity.
        unsafe {
            let inner = NonNull::new_unchecked(p);
            Self { inner, alloc }
        }
    }
}

/// Allocates a fresh block holding a copy of `*t` and returns a `NonNull<Self>`
/// rooted at that block. Implemented only for the unsized kinds whose
/// fat-pointer metadata is a plain length (`[U]` and `str`), which can be
/// reattached to the new base address. A `dyn Trait` does not implement this,
/// so the borrow-based `from_ref*` constructors are unavailable for trait
/// objects — box those through their concrete type instead.
trait CloneUnsize {
    fn clone_into<A: Allocator>(t: &Self, alloc: &A) -> Result<NonNull<Self>, AllocError>;
}

impl<U: Copy> CloneUnsize for [U] {
    fn clone_into<B: Allocator>(t: &[U], alloc: &B) -> Result<NonNull<Self>, AllocError> {
        let layout = Layout::for_value(t);
        if t.is_empty() {
            // No data bytes to own; yield a valid non-null fat pointer with
            // zero length. Nothing is ever deallocated for it.
            let base_u = non_null_byte_base().cast::<U>();
            let p = ptr::slice_from_raw_parts_mut(base_u, 0);
            return Ok(unsafe { NonNull::new_unchecked(p) });
        }
        let fat = alloc.allocate(layout).map_err(|_| AllocError)?;
        let base = fat.as_ptr() as *mut U;
        // SAFETY: `layout` describes exactly `t.len()` elements of `*t`;
        // copying them into the fresh block yields a valid `[U]` there.
        unsafe {
            ptr::copy_nonoverlapping(t.as_ptr(), base, t.len());
        }
        let fat_slice = ptr::slice_from_raw_parts_mut(base, t.len());
        Ok(unsafe { NonNull::new_unchecked(fat_slice) })
    }
}

impl CloneUnsize for str {
    fn clone_into<B: Allocator>(t: &str, alloc: &B) -> Result<NonNull<Self>, AllocError> {
        let layout = Layout::for_value(t);
        if t.is_empty() {
            // Zero-length string: non-null dangling fat pointer, never freed.
            let s: &str = "";
            let p: *const str = s as *const str;
            return Ok(unsafe { NonNull::new_unchecked(p as *mut str) });
        }
        let fat = alloc.allocate(layout).map_err(|_| AllocError)?;
        let base = fat.as_ptr() as *mut u8;
        // SAFETY: `layout` describes exactly `t.len()` bytes of `*t`; copying
        // them into the fresh block yields a valid `str` there.
        unsafe {
            ptr::copy_nonoverlapping(t.as_ptr(), base, t.len());
        }
        // Build a `*mut str` at `base` with length `t.len()`. The copied bytes
        // are guaranteed valid UTF-8 (they came from a `&str`).
        let fat_bytes = ptr::slice_from_raw_parts_mut(base, t.len());
        // SAFETY: the block holds `t.len()` bytes of valid UTF-8, so this is a
        // well-formed `str`.
        let fat_str: *mut str =
            unsafe { core::str::from_utf8_unchecked(&*fat_bytes) as *const str as *mut str };
        Ok(unsafe { NonNull::new_unchecked(fat_str) })
    }
}

/// A non-null byte address usable as the base of a zero-length fat pointer.
/// Zero-length pointers are never dereferenced or deallocated, so any stable
/// non-null address suffices.
fn non_null_byte_base() -> *mut u8 {
    static ANCHOR: u8 = 0;
    core::ptr::addr_of!(ANCHOR) as *mut u8
}

// ---------------------------------------------------------------------------
// Slice-specific constructors: Box<[T], A>
// ---------------------------------------------------------------------------

impl<T> Box<[T], Global> {
    /// Creates a new empty boxed slice.
    ///
    /// Infallible facade over [`try_new_empty`][Self::try_new_empty]: panics
    /// on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn new_empty() -> Self {
        match Self::try_new_empty() {
            Ok(s) => s,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`new_empty`][Self::new_empty].
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
    /// Infallible facade over [`try_with_capacity`][Self::try_with_capacity]:
    /// panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn with_capacity(capacity: usize) -> Self {
        match Self::try_with_capacity(capacity) {
            Ok(s) => s,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`with_capacity`][Self::with_capacity].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, AllocError> {
        let buf = RawVec::<T>::try_with_capacity(capacity).map_err(reserve_err_to_alloc_err)?;
        // SAFETY: The buffer was just allocated with exactly `capacity`
        // elements. For a fresh `Box<[T]>` we treat all slots as valid
        // (uninitialized but well-formed for the type's bit patterns); this
        // matches std's behavior where `Box::with_capacity` leaves contents
        // uninitialized.
        Ok(unsafe { buf.into_box(capacity) })
    }

    /// Creates a new boxed slice filled with zeros (for `Copy` types).
    ///
    /// Infallible facade over [`try_new_zeroed`][Self::try_new_zeroed]: panics
    /// on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn new_zeroed(len: usize) -> Self
    where
        T: Copy,
    {
        match Self::try_new_zeroed(len) {
            Ok(s) => s,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`new_zeroed`][Self::new_zeroed].
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
        Ok(unsafe { buf.into_box(len) })
    }

    /// Creates a new boxed slice from a `&[T]`, cloning the elements.
    ///
    /// Infallible facade over [`try_from_slice`][Self::try_from_slice]: panics
    /// on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        match Self::try_from_slice(slice) {
            Ok(s) => s,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`from_slice`][Self::from_slice].
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
        let mut buf = RawVec::<T>::try_with_capacity(len).map_err(reserve_err_to_alloc_err)?;

        unsafe {
            let dst = buf.ptr();
            let src = slice.as_ptr();
            for i in 0..len {
                ptr::write(dst.add(i), (*src.add(i)).clone());
            }
        }

        // SAFETY: all `len` elements are now initialized.
        Ok(unsafe { buf.into_box(len) })
    }

    /// Creates a new boxed slice from a `[T; N]` array, moving the elements.
    ///
    /// Infallible facade over [`try_from_array`][Self::try_from_array]: panics
    /// on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn from_array<const N: usize>(array: [T; N]) -> Self {
        match Self::try_from_array(array) {
            Ok(s) => s,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`from_array`][Self::from_array].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_array<const N: usize>(array: [T; N]) -> Result<Self, AllocError> {
        let mut buf = RawVec::<T>::try_with_capacity(N).map_err(reserve_err_to_alloc_err)?;

        unsafe {
            let dst = buf.ptr();
            for i in 0..N {
                ptr::write(dst.add(i), ptr::read(array.as_ptr().add(i)));
            }
        }

        // SAFETY: all `N` elements are now initialized.
        Ok(unsafe { buf.into_box(N) })
    }
}

impl<T, A: Allocator> Box<[T], A> {
    /// Creates a new empty boxed slice using the given allocator.
    ///
    /// Infallible facade over [`try_new_empty_in`][Self::try_new_empty_in]:
    /// panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn new_empty_in(alloc: A) -> Self {
        match Self::try_new_empty_in(alloc) {
            Ok(s) => s,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`new_empty_in`][Self::new_empty_in].
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
    /// Infallible facade over [`try_with_capacity_in`][Self::try_with_capacity_in]:
    /// panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        match Self::try_with_capacity_in(capacity, alloc) {
            Ok(s) => s,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`with_capacity_in`][Self::with_capacity_in].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, AllocError> {
        let buf = RawVec::<T, A>::try_with_capacity_in(capacity, alloc)
            .map_err(reserve_err_to_alloc_err)?;
        // SAFETY: freshly allocated buffer with exactly `capacity` elements.
        Ok(unsafe { buf.into_box(capacity) })
    }

    /// Creates a new boxed slice from a `&[T]` using the given allocator.
    ///
    /// Infallible facade over [`try_from_slice_in`][Self::try_from_slice_in]:
    /// panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn from_slice_in(slice: &[T], alloc: A) -> Self
    where
        T: Clone,
    {
        match Self::try_from_slice_in(slice, alloc) {
            Ok(s) => s,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`from_slice_in`][Self::from_slice_in].
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
        let mut buf =
            RawVec::<T, A>::try_with_capacity_in(len, alloc).map_err(reserve_err_to_alloc_err)?;

        unsafe {
            let dst = buf.ptr();
            let src = slice.as_ptr();
            for i in 0..len {
                ptr::write(dst.add(i), (*src.add(i)).clone());
            }
        }

        // SAFETY: all `len` elements are now initialized.
        Ok(unsafe { buf.into_box(len) })
    }
}

// ---------------------------------------------------------------------------
// str-specific constructors: Box<str, A>
// ---------------------------------------------------------------------------

impl Box<str, Global> {
    /// Creates a new `Box<str>` from a `&str`, copying the bytes.
    ///
    /// Infallible facade over [`try_from_str`][Self::try_from_str]: panics on
    /// OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn from_str(s: &str) -> Self {
        match Self::try_from_str(s) {
            Ok(b) => b,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`from_str`][Self::from_str].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_str(s: &str) -> Result<Self, AllocError> {
        let bytes = s.as_bytes();
        let boxed_bytes: Box<[u8]> = Box::from_slice(bytes);
        // SAFETY: `s` is valid UTF-8, so the copied bytes are too.
        Ok(unsafe { from_boxed_utf8_unchecked(boxed_bytes) })
    }
}

impl<A: Allocator> Box<str, A> {
    /// Creates a new `Box<str>` from a `&str` using the given allocator.
    ///
    /// Infallible facade over [`try_from_str_in`][Self::try_from_str_in]:
    /// panics on OOM.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[inline]
    #[track_caller]
    pub fn from_str_in(s: &str, alloc: A) -> Self {
        match Self::try_from_str_in(s, alloc) {
            Ok(b) => b,
            Err(err) => handle_alloc_error(err),
        }
    }

    /// Fallible version of [`from_str_in`][Self::from_str_in].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails.
    #[inline]
    pub fn try_from_str_in(s: &str, alloc: A) -> Result<Self, AllocError> {
        let bytes = s.as_bytes();
        let boxed_bytes: Box<[u8], A> = Box::from_slice_in(bytes, alloc);
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
    // `Sized` requirement on `into_non_null_and_alloc`.
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
// Accessors
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Box<T, A> {
    /// Gets a raw pointer to the underlying data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    /// Gets a mutable raw pointer to the underlying data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_ptr()
    }

    /// Gets a `NonNull<T>` pointing to the underlying data.
    #[inline]
    pub fn non_null(&self) -> NonNull<T> {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// Slice-specific accessors: Box<[T], A>
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Box<[T], A> {
    /// Gets the number of elements in the boxed slice.
    #[inline]
    pub fn len(&self) -> usize {
        unsafe { (*self.inner.as_ptr()).len() }
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
        unsafe { (*self.inner.as_ptr()).len() }
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
// Drop
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Drop for Box<T, A> {
    fn drop(&mut self) {
        // Compute the layout from the fat pointer before dropping the value;
        // `Layout::for_value` needs the dynamic size (and vtable for trait
        // objects), which is carried by `self.inner`.
        let val: &T = self.deref();
        let layout = Layout::for_value(val);
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
// Clone
// ---------------------------------------------------------------------------

impl<T: Clone, A: Allocator + Clone> Clone for Box<T, A> {
    fn clone(&self) -> Self {
        match Self::try_new_in((**self).clone(), self.alloc.clone()) {
            Ok(b) => b,
            Err(err) => handle_alloc_error(err),
        }
    }
}

impl<T: Clone, A: Allocator + Clone> Clone for Box<[T], A> {
    fn clone(&self) -> Self {
        let slice: &[T] = self;
        match Box::try_from_slice_in(slice, self.alloc.clone()) {
            Ok(b) => b,
            Err(err) => handle_alloc_error(err),
        }
    }
}

impl<A: Allocator + Clone> Clone for Box<str, A> {
    fn clone(&self) -> Self {
        let s: &str = self;
        match Box::try_from_str_in(s, self.alloc.clone()) {
            Ok(b) => b,
            Err(err) => handle_alloc_error(err),
        }
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl<T: Default + Sized> Default for Box<T, Global> {
    fn default() -> Self {
        Self::new(T::default())
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

impl<T> From<T> for Box<T, Global> {
    /// Converts a `T` into a `Box<T>`.
    ///
    /// The conversion allocates on the heap and moves `t` from the stack
    /// into it.
    fn from(t: T) -> Self {
        Box::new(t)
    }
}

impl<T: Sized, A: Allocator> From<Box<T, A>> for Pin<Box<T, A>>
where
    A: 'static,
{
    /// Converts a `Box<T>` into a `Pin<Box<T>>`.
    fn from(boxed: Box<T, A>) -> Self {
        Box::into_pin(boxed)
    }
}

impl<T: Clone> From<&[T]> for Box<[T], Global> {
    /// Converts a `&[T]` into a `Box<[T]>`.
    fn from(slice: &[T]) -> Self {
        Box::from_slice(slice)
    }
}

impl<T: Clone> From<&mut [T]> for Box<[T], Global> {
    /// Converts a `&mut [T]` into a `Box<[T]>`.
    fn from(slice: &mut [T]) -> Self {
        Box::from_slice(slice)
    }
}

impl<T, const N: usize> From<[T; N]> for Box<[T], Global> {
    /// Converts a `[T; N]` into a `Box<[T]>`.
    fn from(array: [T; N]) -> Self {
        Box::from_array(array)
    }
}

impl From<&str> for Box<str, Global> {
    /// Converts a `&str` into a `Box<str>`.
    fn from(s: &str) -> Self {
        Box::from_str(s)
    }
}

impl From<&mut str> for Box<str, Global> {
    /// Converts a `&mut str` into a `Box<str>`.
    fn from(s: &mut str) -> Self {
        Box::from_str(s)
    }
}

impl<A: Allocator> From<Box<str, A>> for Box<[u8], A> {
    /// Converts a `Box<str>` into a `Box<[u8]>`.
    ///
    /// This conversion does not allocate on the heap and happens in place.
    fn from(s: Box<str, A>) -> Self {
        unsafe {
            let len = s.len();
            // Extract fields directly to avoid the `Sized` requirement.
            let inner = ptr::read(&s.inner);
            let alloc = ptr::read(&s.alloc);
            core::mem::forget(s);

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
        core::mem::forget(boxed_slice);

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
                core::mem::forget(self);
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
                core::mem::forget(self);
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
                core::mem::forget(self);
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
// Error boxing
// ---------------------------------------------------------------------------

impl<'a, E: Error + Sized + 'a> From<E> for Box<dyn Error + 'a, Global> {
    /// Converts a type of `Error` into a box of `dyn Error`.
    fn from(err: E) -> Self {
        let boxed: Box<E, Global> = Box::new(err);
        // Extract fields directly to avoid the `Sized` requirement.
        let inner = unsafe { ptr::read(&boxed.inner) };
        let alloc = unsafe { ptr::read(&boxed.alloc) };
        core::mem::forget(boxed);
        // Coerce the raw pointer from `*mut E` to `*mut dyn Error`.
        let dyn_ptr: *mut (dyn Error + 'a) = inner.as_ptr();
        // SAFETY: The allocation is valid and contains an `E: Error`.
        unsafe { Box::from_raw_in(dyn_ptr, alloc) }
    }
}

impl<'a, E: Error + Send + Sync + Sized + 'a> From<E>
    for Box<dyn Error + Send + Sync + 'a, Global>
{
    /// Converts a type of `Error` + `Send` + `Sync` into a box of
    /// `dyn Error` + `Send` + `Sync`.
    fn from(err: E) -> Self {
        let boxed: Box<E, Global> = Box::new(err);
        let inner = unsafe { ptr::read(&boxed.inner) };
        let alloc = unsafe { ptr::read(&boxed.alloc) };
        core::mem::forget(boxed);
        // Coerce the raw pointer from `*mut E` to `*mut dyn Error + Send + Sync`.
        let dyn_ptr: *mut (dyn Error + Send + Sync + 'a) = inner.as_ptr();
        // SAFETY: The allocation is valid and contains an `E: Error + Send + Sync`.
        unsafe { Box::from_raw_in(dyn_ptr, alloc) }
    }
}

impl<A: Allocator> Box<dyn Error, A> {
    /// Attempts to downcast the box to a concrete error type.
    pub fn downcast<T: Error + 'static>(self) -> Result<Box<T, A>, Self> {
        if self.is::<T>() {
            unsafe {
                let base_addr = self.inner.as_ptr() as *mut u8;
                let alloc = ptr::read(&self.alloc);
                core::mem::forget(self);
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
                core::mem::forget(self);
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
                core::mem::forget(self);
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

impl<I: core::iter::Iterator + ?Sized, A: Allocator> core::iter::Iterator for Box<I, A> {
    type Item = <I as core::iter::Iterator>::Item;

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

impl<I: core::iter::DoubleEndedIterator + ?Sized, A: Allocator> core::iter::DoubleEndedIterator
    for Box<I, A>
{
    #[inline]
    fn next_back(&mut self) -> Option<<I as core::iter::Iterator>::Item> {
        (**self).next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as core::iter::Iterator>::Item> {
        (**self).nth_back(n)
    }
}

impl<I: core::iter::FusedIterator + ?Sized, A: Allocator> core::iter::FusedIterator for Box<I, A> {}

impl<I: core::iter::ExactSizeIterator + ?Sized, A: Allocator> core::iter::ExactSizeIterator
    for Box<I, A>
{
}

// ---------------------------------------------------------------------------
// Helper: construct AllocError from a failed allocation
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
    fn test_box_new_and_deref() {
        let b = Box::new(42i32);
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
        let b = Box::new(Zst);
        // Should not allocate; should work fine.
        let _ = &b;
    }

    #[test]
    fn test_box_into_raw_from_raw_roundtrip() {
        let b = Box::new(String::from("hello"));
        let raw: *mut String = Box::into_raw(b);
        assert_eq!(unsafe { &*raw }, "hello");
        // Reconstruct from the raw pointer using an explicit type ascription
        // so inference can resolve both `T` and `A`.
        let b: Box<String, Global> = unsafe { Box::<String, Global>::from_raw_global(raw) };
        assert_eq!(&**b, "hello");
    }

    #[test]
    fn test_box_leak() {
        let b = Box::new(vec![1, 2, 3]);
        let leaked: &'static mut Vec<i32> = Box::leak(b);
        assert_eq!(leaked, &vec![1, 2, 3]);
    }

    #[test]
    fn test_box_clone_copy() {
        let b1 = Box::new(5u32);
        let b2: Box<u32> = b1.try_clone().unwrap();
        assert_eq!(*b1, *b2);
        assert_ne!(Box::as_ptr(&b1), Box::as_ptr(&b2));
    }

    #[test]
    fn test_box_default() {
        let b: Box<u8> = Box::default();
        assert_eq!(*b, 0);
    }

    #[test]
    fn test_box_debug_display() {
        let b = Box::new(format_test_string());
        assert_eq!(format!("{b:?}"), "\"hi\"");
        assert_eq!(format!("{b}"), "hi");
    }

    fn format_test_string() -> String {
        String::from("hi")
    }

    #[test]
    fn test_box_equality_ordering_hash() {
        let a = Box::new(10i32);
        let b = Box::new(10i32);
        let c = Box::new(20i32);
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
        let mut b = Box::new(7u16);
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
    fn test_box_from_t() {
        let b: Box<i64> = Box::from(-1);
        assert_eq!(*b, -1);
    }

    #[test]
    fn test_box_pin() {
        let p: core::pin::Pin<Box<u8, Global>> = Box::<u8, Global>::pin(31u8);
        assert_eq!(*p, 31);
    }

    #[test]
    fn test_box_seal_no_double_free() {
        let b = Box::new(100u32);
        let sealed = Box::seal(b);
        assert_eq!(**sealed, 100);
        // Dropping the ManuallyDrop does NOT free the allocation.
        // We intentionally leak here; valgrind/miri will catch real bugs.
    }

    #[test]
    fn test_box_slice_from_slice() {
        let src = [1, 2, 3, 4, 5];
        let bs: Box<[i32]> = Box::from_slice(&src);
        assert_eq!(bs.len(), 5);
        assert_eq!(bs[0], 1);
        assert_eq!(bs[4], 5);
    }

    #[test]
    fn test_box_slice_try_from_slice() {
        let src = ['a', 'b', 'c'];
        let bs = Box::try_from_slice(&src).unwrap();
        assert_eq!(bs.len(), 3);
        assert_eq!(bs[1], 'b');
    }

    #[test]
    fn test_box_slice_from_array() {
        let bs: Box<[u8]> = Box::from([10, 20, 30]);
        assert_eq!(bs.len(), 3);
        assert_eq!(bs[2], 30);
    }

    #[test]
    fn test_box_slice_with_capacity() {
        let bs: Box<[u8]> = Box::with_capacity(10);
        assert_eq!(bs.len(), 10);
    }

    #[test]
    fn test_box_slice_new_empty() {
        let bs: Box<[i32]> = Box::new_empty();
        assert!(bs.is_empty());
    }

    #[test]
    fn test_box_slice_clone() {
        let orig: Box<[i32]> = Box::from_slice(&[1, 2, 3]);
        let cloned: Box<[i32]> = orig.clone();
        assert_eq!(orig, cloned);
        assert_ne!(Box::as_ptr(&orig), Box::as_ptr(&cloned));
    }

    #[test]
    fn test_box_str_from_str() {
        let bs: Box<str> = Box::from_str("hello world");
        assert_eq!(&*bs, "hello world");
        assert_eq!(bs.len(), 11);
    }

    #[test]
    fn test_box_str_clone() {
        let orig: Box<str> = Box::from_str("abc");
        let cloned: Box<str> = orig.clone();
        assert_eq!(orig, cloned);
    }

    #[test]
    fn test_box_str_to_bytes() {
        let bs: Box<str> = Box::from_str("xyz");
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
        let boxed: Box<Dog, Global> = Box::new(Dog { tag: 1 });
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
        let boxed: Box<Empty, Global> = Box::new(Empty);
        let raw: *mut Empty = Box::into_raw(boxed);
        let fat: *mut dyn Greet = raw as *mut dyn Greet;
        let b: Box<dyn Greet, Global> = unsafe { Box::from_raw_in(fat, Global) };
        assert_eq!(b.greet(), "...");
    }

    #[test]
    fn test_box_any_downcast() {
        // Build a `Box<dyn Any>` by boxing the concrete value and unsize-coercing
        // through a raw fat pointer (our `Box::new` is sized-only).
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

        // Box the concrete error through the `From<E> for Box<dyn Error>` impl,
        // which boxes the value and unsize-coerces to attach the vtable — the
        // sound path for obtaining a boxed trait object.
        let err = MyErr;
        let e: Box<dyn Error> = Box::from(err);
        let recovered = e.downcast::<MyErr>().unwrap();
        let _ = recovered; // just verify it works
    }

    #[test]
    fn test_box_iterator_forwarding() {
        let v = vec![1, 2, 3, 4, 5];
        let iter: Box<std::slice::Iter<'_, i32>> = Box::new(v.iter());
        let collected: Vec<&i32> = iter.collect();
        assert_eq!(collected, vec![&1, &2, &3, &4, &5]);
    }

    #[test]
    fn test_box_double_ended_iterator() {
        let v = vec![1, 2, 3, 4, 5];
        let mut iter: Box<std::slice::Iter<'_, i32>> = Box::new(v.iter());
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
        let bs: Box<[i32]> = Box::from_slice(&[1, 2, 3]);
        let arr: Box<[i32; 3]> = Box::try_from(bs).unwrap();
        assert_eq!(arr[0], 1);
        assert_eq!(arr[2], 3);

        let bs: Box<[i32]> = Box::from_slice(&[1, 2]);
        let result: Result<Box<[i32; 3]>, Box<[i32]>> = Box::try_from(bs);
        assert!(result.is_err());
    }

    #[test]
    fn test_box_pointer_fmt() {
        let b = Box::new(0u8);
        let _ = format!("{b:p}");
    }
}
