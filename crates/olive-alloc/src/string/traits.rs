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
