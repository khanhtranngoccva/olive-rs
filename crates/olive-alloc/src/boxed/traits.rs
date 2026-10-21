//! Deref forwarding, and trait impls for [`Box`].

use super::Box;
use crate::alloc::{Allocator, AllocatorTryClone};
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::ops::{Deref, DerefMut};
use olive_core::try_traits::TryCloneToUninit;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

// ---------------------------------------------------------------------------
// Send / Sync
// ---------------------------------------------------------------------------

unsafe impl<T: ?Sized + Send, A: Allocator + Send> Send for Box<T, A> {}
unsafe impl<T: ?Sized + Sync, A: Allocator + Sync> Sync for Box<T, A> {}

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
        let cloned_allocator = Box::allocator(self).try_clone()?;
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

    #[inline]
    fn last(self) -> Option<Self::Item> {
        // TODO: this item is not possible to fix on 1.85.
        // std uses a specialization to optimize the Box.
        #[inline]
        fn some<T>(_: Option<T>, x: T) -> Option<T> {
            Some(x)
        }

        self.fold(None, some)
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

impl<I: ExactSizeIterator + ?Sized, A: Allocator> ExactSizeIterator for Box<I, A> {
    fn len(&self) -> usize {
        (**self).len()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::alloc::Global;
    use olive_test_helpers::{CloneBudget, FlakyCloneAlloc};
    use std::format;
    use std::string::String;
    use std::vec;
    use std::vec::Vec;

    fn format_test_string() -> String {
        String::from("hi")
    }

    // Send / Sync.

    #[test]
    fn box_send_sync_bounds_hold() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Box<i32, Global>>();
        assert_sync::<Box<i32, Global>>();
        assert_send::<Box<str, Global>>();
        assert_sync::<Box<str, Global>>();
        assert_send::<Box<dyn Send, Global>>();
        assert_sync::<Box<dyn Sync, Global>>();
    }

    // TryClone.

    #[test]
    fn box_try_clone() {
        let b1 = Box::try_new(5u32).unwrap();
        let b2: Box<u32> = b1.try_clone().unwrap();
        assert_eq!(*b1, *b2);
        assert_ne!(Box::as_ptr(&b1), Box::as_ptr(&b2));
    }

    #[test]
    fn box_try_clone_allocator_failure() {
        // `Box::try_clone` first clones the allocator handle: when that fails
        // the error must propagate and no clone is produced.
        let b = Box::try_new_in(5u32, FlakyCloneAlloc::new(CloneBudget::new(0))).unwrap();
        let e = b.try_clone().unwrap_err();
        assert!(matches!(e, TryCloneError::Other("clone budget exhausted")));
    }

    // Pointer / Debug / Display.

    #[test]
    fn box_pointer_fmt() {
        let b = Box::try_new(0u8).unwrap();
        let _ = format!("{b:p}");
    }

    #[test]
    fn box_debug_display() {
        let b = Box::try_new(format_test_string()).unwrap();
        assert_eq!(format!("{b:?}"), "\"hi\"");
        assert_eq!(format!("{b}"), "hi");
    }

    #[test]
    fn box_equality_ordering_hash() {
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
    fn box_asref_asmut_borrow() {
        let mut b = Box::try_new(7u16).unwrap();
        let r: &u16 = b.as_ref();
        assert_eq!(*r, 7);
        let m: &mut u16 = b.as_mut();
        *m = 8;
        assert_eq!(*b, 8);
        let br: &u16 = b.borrow();
        assert_eq!(*br, 8);
        let bm: &mut u16 = b.borrow_mut();
        *bm = 9;
        assert_eq!(*b, 9);
    }

    #[test]
    fn box_iterator_forwarding() {
        let v = [1, 2, 3, 4, 5];
        let iter: Box<std::slice::Iter<'_, i32>> = Box::try_new(v.iter()).unwrap();
        let collected: Vec<&i32> = iter.collect();
        assert_eq!(collected, vec![&1, &2, &3, &4, &5]);

        let v = [1, 2, 3, 4, 5];
        let iter: Box<std::slice::Iter<'_, i32>> = Box::try_new(v.iter()).unwrap();
        assert_eq!(iter.last(), Some(&5));
        let iter: Box<std::slice::Iter<'_, i32>> = Box::try_new(v.iter()).unwrap();
        assert_eq!(iter.size_hint(), (5, Some(5)));
        assert_eq!(iter.len(), 5);
        assert_eq!(iter.count(), 5);
    }

    #[test]
    fn box_double_ended_iterator() {
        let v = [1, 2, 3, 4, 5];
        let mut iter: Box<std::slice::Iter<'_, i32>> = Box::try_new(v.iter()).unwrap();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&5));
        assert_eq!(iter.next(), Some(&2));
    }
}
