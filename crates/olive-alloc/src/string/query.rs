//! Implementations of [`String`] APIs that query string information.

use super::String;
use crate::vec::Vec;
use core::ops::{Deref, DerefMut};
use olive_core::alloc::Allocator;

impl<A: Allocator> String<A> {
    /// Returns a reference to the string's allocator.
    #[inline]
    pub const fn allocator(&self) -> &A {
        self.buf.allocator()
    }

    /// Returns the number of bytes the string can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Number of bytes in the string.
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if the string contains no bytes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Returns the string's contents as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.buf.as_slice()
    }

    /// Returns the string's contents as a `&str`.
    ///
    /// This is the explicit accessor counterpart to the implicit
    /// [`Deref`] coercion to `&str`.
    #[inline]
    pub fn as_str(&self) -> &str {
        self.deref()
    }

    /// Returns the string's contents as a `&mut str`.
    ///
    /// This is the explicit mutable-accessor counterpart to the implicit
    /// [`DerefMut`] coercion to `&mut str`.
    #[inline]
    pub fn as_mut_str(&mut self) -> &mut str {
        self.deref_mut()
    }

    /// Returns the string's backing byte buffer as a `&Vec<u8>`.
    ///
    /// This exposes the internal representation for callers that need to
    /// manipulate the bytes directly. Mutating the returned reference can
    /// violate the UTF-8 invariant; prefer the higher-level fallible
    /// mutators.
    #[inline]
    pub fn as_vec(&self) -> &Vec<u8, A> {
        &self.buf
    }

    /// Returns the string's backing byte buffer as a `&mut Vec<u8>`.
    ///
    /// # Safety
    ///
    /// Writing non-UTF-8 bytes through the returned reference violates the
    /// UTF-8 invariant relied upon by the rest of the library.
    #[inline]
    pub unsafe fn as_mut_vec(&mut self) -> &mut Vec<u8, A> {
        &mut self.buf
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Deref;

    use super::*;
    use crate::alloc::CountingAllocator;

    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    #[test]
    fn allocator_returns_backing_allocator() {
        let alloc = CountingAllocator::new();
        let s: String<&CountingAllocator> = String::try_from_str_in("abc", &alloc).unwrap();
        // The returned reference is the very instance the string was built
        // with.
        assert!(core::ptr::eq(*s.allocator(), &alloc));
    }

    #[test]
    fn capacity_grows_on_reserve() {
        let mut s = mk("abc");
        assert!(s.capacity() >= 3);
        s.try_reserve(64).unwrap();
        assert!(s.capacity() >= 67);
    }

    #[test]
    fn len_and_is_empty() {
        let mut s = String::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        s.try_push_str("héllo").unwrap();
        assert!(!s.is_empty());
        // "héllo" is 5 chars but 6 bytes.
        assert_eq!(s.len(), 6);
    }

    #[test]
    fn as_bytes_view() {
        let s = mk("é🦊");
        assert_eq!(s.as_bytes(), "é🦊".as_bytes());
        assert_eq!(s.as_bytes().len(), 6);
    }

    #[test]
    fn as_str_view() {
        let s = mk("query");
        assert_eq!(s.as_str(), "query");
        // Matches the Deref view.
        assert_eq!(s.deref(), s.as_str());
    }

    #[test]
    fn as_mut_str_mutates_in_place() {
        let mut s = mk("abc");
        s.as_mut_str().make_ascii_uppercase();
        assert_eq!(s.deref(), "ABC");
        assert_eq!(s.as_mut_str().len(), 3);
    }

    #[test]
    fn as_vec_exposes_backing_buffer() {
        let s = mk("vec");
        let v: &Vec<u8> = s.as_vec();
        assert_eq!(&v[..], b"vec");
        assert_eq!(v.len(), s.len());
    }

    #[test]
    fn as_mut_vec_allows_direct_writes() {
        let mut s = mk("abc");
        // SAFETY: writing ASCII bytes keeps the buffer valid UTF-8.
        let v = unsafe { s.as_mut_vec() };
        v[0] = b'A';
        assert_eq!(s.deref(), "Abc");
    }
}
