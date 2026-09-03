//! Deref forwarding, and trait impls for [`Box`].

use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::ops::{Deref, DerefMut};

use crate::alloc::{Allocator, AllocatorTryClone};
use olive_core::try_traits::TryCloneToUninit;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

use super::Box;

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

// `Box::try_clone` must land on the *same* backing store as the original, so the
// allocator is required to be [`AllocatorTryClone`] — not merely `TryClone`, which
// could mint an independent handle. The guarantee is what makes it sound to free
// the clone's block through a freshly cloned allocator handle.
impl<T: ?Sized + TryCloneToUninit, A: AllocatorTryClone> TryClone for Box<T, A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let cloned_allocator = self.allocator().try_clone()?;
        Box::try_clone_from_ref_in(self, cloned_allocator)
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
