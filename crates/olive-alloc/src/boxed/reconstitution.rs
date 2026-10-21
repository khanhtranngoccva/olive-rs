//! APIs that deal with [`Box`] conversion to and from raw pointers.

use super::Box;
use crate::alloc::Global;
use core::{
    mem::{self, ManuallyDrop},
    ptr::{self, NonNull},
};
use olive_core::alloc::Allocator;

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
        // We still want to extract the allocator for consistency even if it is a ZST.
        let _alloc = unsafe { ptr::read(&b.alloc) };
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
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::boxed::Box;
    ///
    /// // This test leaks and cannot be triggered under Miri.
    /// #[cfg(not(miri))]
    /// {
    ///     let s: &'static str = Box::leak(Box::try_new("world").unwrap());
    ///     assert_eq!(s, "world");
    /// }
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
}

#[cfg(test)]
mod tests {
    #![allow(clippy::vec_init_then_push)]

    extern crate std;

    use super::*;
    use std::string::String;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn test_box_into_raw_from_raw_roundtrip() {
        let b = Box::try_new(String::from("hello")).unwrap();
        let raw: *mut String = Box::into_raw(b);
        assert_eq!(unsafe { &*raw }, "hello");

        let b: Box<String, Global> = unsafe { Box::from_raw(raw) };
        assert_eq!(&**b, "hello");
    }

    #[test]
    #[cfg_attr(miri, ignore = "this test leaks memory")]
    fn test_box_leak() {
        let b = Box::try_new(vec![1, 2, 3]).unwrap();
        let leaked: &'static mut Vec<i32> = Box::leak(b);
        assert_eq!(leaked, &vec![1, 2, 3]);
    }

    #[test]
    fn test_box_dyn_trait_object() {
        trait Greet {
            fn greet(&self) -> String;
        }
        // Non-zero-sized payload so the boxed trait object owns real data on
        // the heap (the vtable itself lives in the fat pointer's metadata).
        #[allow(dead_code)]
        struct Dog {
            tag: u64,
        }
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
}
