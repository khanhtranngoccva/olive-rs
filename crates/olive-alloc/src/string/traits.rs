//! Trait implementations for the [`String`] type, with exceptions for
//! comparison functions.

use super::String;
use core::{
    borrow::{Borrow, BorrowMut},
    fmt::{self, Write},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};
use olive_core::{
    alloc::{Allocator, AllocatorTryClone},
    alloc_errors::TryReserveError,
    recovery::{ResumableSource, Resume},
    try_traits::{TryClone, TryCloneError, TryExtend, TryFromIterator},
};

impl<A: Allocator> Deref for String<A> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        // SAFETY: the buffer always holds valid UTF-8 (every mutation goes
        // through a checked path).
        unsafe { core::str::from_utf8_unchecked(self.buf.as_slice()) }
    }
}

impl<A: Allocator> DerefMut for String<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut str {
        // SAFETY: the buffer always holds valid UTF-8 (every mutation goes
        // through a checked path).
        unsafe { core::str::from_utf8_unchecked_mut(self.buf.as_mut_slice()) }
    }
}

impl<A: Allocator> AsRef<str> for String<A> {
    #[inline]
    fn as_ref(&self) -> &str {
        self
    }
}

impl<A: Allocator> AsMut<str> for String<A> {
    #[inline]
    fn as_mut(&mut self) -> &mut str {
        self
    }
}

impl<A: Allocator> AsRef<[u8]> for String<A> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<A: Allocator> Borrow<str> for String<A> {
    #[inline]
    fn borrow(&self) -> &str {
        self
    }
}

impl<A: Allocator> BorrowMut<str> for String<A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut str {
        self
    }
}

impl<A: Allocator> fmt::Debug for String<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

impl<A: Allocator> fmt::Display for String<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

// Cross-type comparison impls (PartialEq, Eq, PartialOrd, Ord) live in cmp.rs.

impl<A: Allocator> Hash for String<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state)
    }
}

// SAFETY: `String` holds only `u8` bytes with no interior mutability or
// self-referential structure. Moving it leaves the heap buffer untouched, so
// `Send`/`Sync` follow from the buffer and allocator alone.
unsafe impl<A: Allocator + Send> Send for String<A> {}
unsafe impl<A: Allocator + Sync> Sync for String<A> {}

impl<A: AllocatorTryClone> TryClone for String<A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let alloc = self.buf.allocator().try_clone()?;
        let mut out = Self::try_with_capacity_in(self.len(), alloc)?;
        out.try_push_str(self.deref())
            .map_err(TryCloneError::Reserve)?;
        Ok(out)
    }
}

impl<A: Allocator> TryExtend<char> for String<A> {
    type Error = TryReserveError;

    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = char>,
    {
        let (head, mut inner, hint) = source.decompose_with_size_hint();
        // Ignore over-reserve.
        let _ = self.try_reserve_adaptive(hint.estimated_total());
        if let Some(head) = head {
            if let Err(e) = self.try_push(head) {
                return Err((Resume::new(head, inner), e));
            }
        }
        while let Some(next) = inner.next() {
            if let Err(e) = self.try_push(next) {
                return Err((Resume::new(next, inner), e));
            }
        }
        Ok(())
    }
}

impl<'a, A: Allocator> TryExtend<&'a str> for String<A> {
    type Error = TryReserveError;

    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = &'a str>,
    {
        let (head, mut inner) = source.decompose();
        // A string item has very unpredictable sizes, so we can't reserve.
        if let Some(head) = head {
            if let Err(e) = self.try_push_str(head) {
                return Err((Resume::new(head, inner), e));
            }
        }
        while let Some(next) = inner.next() {
            if let Err(e) = self.try_push_str(next) {
                return Err((Resume::new(next, inner), e));
            }
        }
        Ok(())
    }
}

impl TryFromIterator<char> for String {
    type Error = TryReserveError;

    fn try_from_iter<I: IntoIterator<Item = char>>(iter: I) -> Result<Self, Self::Error> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut out = Self::try_with_capacity(capacity)?;
        for c in iter {
            out.try_push(c)?;
        }
        Ok(out)
    }
}

impl<'a> TryFromIterator<&'a str> for String {
    type Error = TryReserveError;

    fn try_from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Result<Self, Self::Error> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut out = Self::try_with_capacity(capacity)?;
        for s in iter {
            out.try_push_str(s)?;
        }
        Ok(out)
    }
}

// FIXME: add try_from_iter or something like that

impl<A: Allocator> Write for String<A> {
    /// Makes [`String`] usable as a fallible formatting sink.
    ///
    /// This is the olive analogue of std's `impl fmt::Write for String`, except that
    /// a failed growth surfaces as [`fmt::Error`] rather than panicking. It is what
    /// lets [`TryToString`] render *any* [`Display`] value into an owned string via
    /// [`write_fmt`](fmt::Write::write_fmt): the formatter calls back into
    /// [`write_str`](Self::write_str) for each emitted fragment, and a reservation
    /// failure aborts the render cleanly instead of unwinding.
    fn write_str(&mut self, s: &str) -> fmt::Result {
        // Allocation failure maps to `fmt::Error`; the caller observes it as a
        // failed render and discards the partially-filled buffer.
        self.push_str_inner(s).map_err(|_| fmt::Error {})
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use olive_core::try_traits::try_collect::TryCollect;
    use olive_test_helpers::{BudgetedAlloc, FailAlloc};
    use std::format;

    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    // ── Deref / DerefMut ──────────────────────────────────────────────────────

    #[test]
    fn deref_coerces_to_str() {
        let s = mk("hello");
        let r: &str = &s;
        assert_eq!(r, "hello");
        assert_eq!(s.deref(), "hello");
    }

    #[test]
    fn deref_mut_coerces_to_mut_str() {
        let mut s = mk("abc");
        let m: &mut str = &mut s;
        m.make_ascii_uppercase();
        assert_eq!(&*s, "ABC");
    }

    // ── AsRef / AsMut ─────────────────────────────────────────────────────────

    #[test]
    fn as_ref_str_and_bytes() {
        let s = mk("héllo");
        let a: &str = s.as_ref();
        let b: &[u8] = s.as_ref();
        assert_eq!(a, "héllo");
        assert_eq!(b, "héllo".as_bytes());
    }

    #[test]
    fn as_mut_str_returns_correct_length_and_content() {
        let mut s = mk("abc");
        {
            let m: &mut str = s.as_mut();
            assert_eq!(m.len(), 3);
            assert_eq!(m, "abc");
            m.make_ascii_uppercase();
            assert_eq!(m, "ABC");
        }
        assert_eq!(&*s, "ABC");
    }

    #[test]
    fn as_mut_str_multibyte_len() {
        let mut s = mk("é🦊");
        let m: &mut str = s.as_mut();
        // 'é' is 2 bytes and the fox is 4, so the str view is 6 bytes long.
        assert_eq!(m.len(), 6);
    }

    // ── Borrow / BorrowMut ────────────────────────────────────────────────────

    #[test]
    fn borrow_matches_deref() {
        let s = mk("borrowed");
        let b: &str = s.borrow();
        assert_eq!(b, "borrowed");
    }

    #[test]
    fn borrow_mut_allows_in_place_mutation() {
        let mut s = mk("mutate");
        let b: &mut str = s.borrow_mut();
        b.make_ascii_uppercase();
        assert_eq!(&*s, "MUTATE");
    }

    // ── Debug / Display / Hash ────────────────────────────────────────────────

    #[test]
    fn display_debug_hash_ord() {
        let a = mk("apple");
        assert_eq!(format!("{a:?}"), "\"apple\"");
        assert_eq!(format!("{a}"), "apple");
        let mut m = std::collections::HashMap::new();
        m.insert(a.try_clone().unwrap(), 1);
        assert_eq!(m.get("apple"), Some(&1));
    }

    // ── TryClone ──────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_matches() {
        let s = mk("original");
        let c = s.try_clone().unwrap();
        assert_eq!(c.deref(), "original");
        assert_ne!(s.as_bytes().as_ptr(), c.as_bytes().as_ptr());
    }

    #[test]
    fn try_clone_oom_returns_reserve_error() {
        let alloc = BudgetedAlloc::new(1);
        let s: String<BudgetedAlloc> =
            String::try_from_str_in("original", alloc.try_clone().unwrap()).unwrap();

        // The clone needs a fresh allocation, which the drained budget
        // refuses.
        alloc.drain();
        let err = s.try_clone().unwrap_err();
        assert!(matches!(err, TryCloneError::Reserve(_)));
        // The original is untouched.
        assert_eq!(s.deref(), "original");
    }

    // ── TryExtend<char> ───────────────────────────────────────────────────────

    #[test]
    fn extend_chars_via_trait() {
        let mut s = String::new();
        s.try_extend(['a', 'b', 'c'].into_iter()).unwrap();
        assert_eq!(s.deref(), "abc");
    }

    #[test]
    fn extend_chars_oom_returns_resume_with_stranded_head() {
        let alloc = BudgetedAlloc::new(1);
        let mut s = String::try_with_capacity_in(3, alloc.try_clone().unwrap()).unwrap();
        s.try_push_str("abc").unwrap();

        // The string is full, so pushing must reserve first.
        alloc.drain();
        let Err((resume, err)) = s.try_extend(['x', 'y'].into_iter()) else {
            panic!("expected reservation failure");
        };
        assert!(err.is_alloc());
        // The failing element is stranded as the resume head; nothing was
        // inserted.
        assert!(resume.has_head());
        assert_eq!(*resume.head().unwrap(), 'x');
        assert_eq!(s.deref(), "abc");
    }

    // ── TryExtend<&str> ───────────────────────────────────────────────────────

    #[test]
    fn extend_strs_via_trait() {
        let mut s = mk("ab");
        s.try_extend(["cd", "ef"].into_iter()).unwrap();
        assert_eq!(s.deref(), "abcdef");
    }

    #[test]
    fn extend_strs_oom_returns_resume_with_stranded_head() {
        let alloc = BudgetedAlloc::new(1);
        let mut s = String::try_with_capacity_in(3, alloc.try_clone().unwrap()).unwrap();
        s.try_push_str("abc").unwrap();

        alloc.drain();
        let Err((resume, err)) = s.try_extend(["x", "y"].into_iter()) else {
            panic!("expected reservation failure");
        };
        assert!(err.is_alloc());
        assert!(resume.has_head());
        assert_eq!(*resume.head().unwrap(), "x");
        assert_eq!(s.deref(), "abc");
    }

    // ── TryFromIterator<char> ─────────────────────────────────────────────────

    #[test]
    fn collect_chars_into_string() {
        let s: String = ['x', 'y', 'z'].into_iter().try_collect().unwrap();
        assert_eq!(s.deref(), "xyz");
    }

    #[test]
    fn collect_chars_oom_on_capacity_overflow() {
        // An iterator hinting `usize::MAX` items makes the initial
        // reservation overflow, which is the deterministic OOM path for the
        // `Global`-only collect impls.
        let err = (0..usize::MAX)
            .map(|_| 'x')
            .try_collect::<String>()
            .unwrap_err();
        assert!(err.is_capacity_overflow());
    }

    // ── TryFromIterator<&str> ─────────────────────────────────────────────────

    #[test]
    fn collect_strs_into_string() {
        let s: String = ["ab", "cd"].into_iter().try_collect().unwrap();
        assert_eq!(s.deref(), "abcd");
    }

    #[test]
    fn collect_strs_oom_on_capacity_overflow() {
        let err = (0..usize::MAX)
            .map(|_| "x")
            .try_collect::<String>()
            .unwrap_err();
        assert!(err.is_capacity_overflow());
    }

    // ── fmt::Write ────────────────────────────────────────────────────────────

    #[test]
    fn write_str_and_write_fmt() {
        let mut s = String::new();
        s.write_str("direct").unwrap();
        write!(s, "-{}-{:?}", 42, "q").unwrap();
        assert_eq!(&*s, "direct-42-\"q\"");
    }

    #[test]
    fn write_failure_maps_to_fmt_error() {
        let mut s = String::<FailAlloc>::new_in(FailAlloc);
        let r = s.write_str("anything");
        assert!(r.is_err());
        // The render failed cleanly and the string stays empty.
        assert!(s.is_empty());
    }
}
