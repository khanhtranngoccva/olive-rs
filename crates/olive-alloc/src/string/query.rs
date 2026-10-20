//! Implementations of [`String`] APIs that query string information.

use super::String;
use crate::vec::Vec;
use core::ops::{Deref, DerefMut};
use olive_core::alloc::Allocator;

// FIXME: allocator() is missing

impl<A: Allocator> String<A> {
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
