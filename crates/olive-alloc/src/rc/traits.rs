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

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use core::panic::{RefUnwindSafe, UnwindSafe};
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
    fn as_ref_and_borrow_work() {
        let rc = Rc::try_new(5u32).unwrap();
        let r: &u32 = rc.as_ref();
        assert_eq!(*r, 5);
        let b: &u32 = Borrow::borrow(&rc);
        assert_eq!(*b, 5);
    }

    #[test]
    fn unwind_safe_impls_match_std() {
        assert_unwind_safe::<Rc<u32>>();
        assert_ref_unwind_safe::<Rc<u32>>();
        assert_unwind_safe::<Rc<str>>();
        assert_ref_unwind_safe::<Rc<str>>();
        assert_unwind_safe::<Rc<[u8]>>();
        assert_ref_unwind_safe::<Rc<[u8]>>();
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

    fn assert_unwind_safe<T: UnwindSafe>() {}
    fn assert_ref_unwind_safe<T: RefUnwindSafe>() {}

    #[test]
    fn debug_and_display_forward_to_payload() {
        let rc = Rc::try_new(String::from("payload")).unwrap();
        assert_eq!(std::format!("{rc:?}"), "\"payload\"");
        assert_eq!(std::format!("{rc}"), "payload");
    }

    #[test]
    fn ord_and_partial_ord_forward_to_payload() {
        let a = Rc::try_new(1).unwrap();
        let b = Rc::try_new(2).unwrap();
        assert_eq!(Ord::cmp(&a, &b), Ordering::Less);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
        // Equal payloads compare equal: the comparison impls are value-based,
        // not address-based.
        let b2 = Rc::try_new(2).unwrap();
        assert_eq!(b, b2);
        assert_eq!(b.cmp(&b2), Ordering::Equal);
    }

    #[test]
    fn hash_matches_payload_hash() {
        use std::collections::hash_map::DefaultHasher;

        fn hash_of<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        let rc = Rc::try_new(String::from("hashed")).unwrap();
        assert_eq!(hash_of(&rc), hash_of(&String::from("hashed")));
    }

    #[test]
    fn rc_pointer_fmt_prints_data_address() {
        let rc = Rc::try_new(1).unwrap();
        assert_eq!(
            std::format!("{rc:p}"),
            std::format!("{:p}", Rc::as_ptr(&rc))
        );
    }

    #[test]
    fn weak_pointer_fmt_prints_data_address() {
        // Unlike std, our `Weak` also implements `Pointer`.
        let rc = Rc::try_new(1).unwrap();
        let weak = Rc::try_downgrade(&rc).unwrap();
        assert_eq!(
            std::format!("{weak:p}"),
            std::format!("{:p}", weak.as_ptr())
        );
    }

    // FIXME: missing Pointer impl
}
