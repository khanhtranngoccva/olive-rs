//! Conversions, deref forwarding, and trait impls for [`Box`].
//!
//! Split out of the main module; see the crate-level docs in [`super`] for the
//! overall design.
// FIXME: split into "traits.rs", "convert.rs" (conversion-only module)

use core::any::Any;
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::mem;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::ptr::{self, NonNull};

use crate::alloc::{Allocator, Global, StaticAllocator};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

use super::Box;

// ---------------------------------------------------------------------------
// Slice-specific accessors: Box<[T], A>
// ---------------------------------------------------------------------------
// FIXME: deref is enough, remove
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
        self
    }

    /// Gets a mutable reference to the inner slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

// ---------------------------------------------------------------------------
// Str-specific accessors: Box<str, A>
// ---------------------------------------------------------------------------

// FIXME: deref is enough, remove
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
        self
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

// FIXME: missing CStr implementation

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
        self
    }
}

impl<T: ?Sized, A: Allocator> AsMut<T> for Box<T, A> {
    fn as_mut(&mut self) -> &mut T {
        self
    }
}

impl<T: ?Sized, A: Allocator> Borrow<T> for Box<T, A> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: ?Sized, A: Allocator> BorrowMut<T> for Box<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        self
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


