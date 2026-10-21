//! Trait implementations for the [`Rc`] and [`Weak`]

use super::helpers;
use super::{Rc, Weak};
use crate::alloc::Allocator;
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::ops::Deref;
use core::panic::{RefUnwindSafe, UnwindSafe};

// ---------------------------------------------------------------------------
// Rc Deref (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Deref for Rc<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `ptr` is always a valid, aligned, non-null pointer to an
        // initialized `T`.
        unsafe { &*helpers::ptr_get_data(self.ptr.as_ptr()) }
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

// Interior mutability inside `Rc` is only exercised through `&self` methods
// that can be re-entrantly observed, so unwind safety is granted conditionally
// on `T: RefUnwindSafe` exactly as std does for `std::rc::Rc`.
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Rc<T> {}
impl<T: ?Sized + RefUnwindSafe> RefUnwindSafe for Rc<T> {}

// ---------------------------------------------------------------------------
// Weak trait APIs
// ---------------------------------------------------------------------------

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
