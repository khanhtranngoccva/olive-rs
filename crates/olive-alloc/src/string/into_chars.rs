//! Owned character iterator produced by [`String::into_chars`](super::String::into_chars).
//!
//! Unlike [`core::str::Chars`], which borrows its source, [`IntoChars`] owns
//! the underlying bytes (held as a [`String`](super::String)), so it carries
//! no borrowed lifetime and drops cleanly when exhausted or discarded —
//! nothing is leaked.

use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

use super::String;
use crate::alloc::{Allocator, AllocatorTryClone, Global};
use core::fmt;
use core::iter::{DoubleEndedIterator, FusedIterator, Iterator};
use core::ptr;

/// An owning iterator over the characters of a consumed [`String`](super::String).
///
/// Constructed via [`String::into_chars`](super::String::into_chars); there is
/// no public constructor because the only sound way to build one is from a
/// valid UTF-8 string whose ownership we take.
pub struct IntoChars<A: Allocator = Global> {
    /// The full owned string, kept alive for the iterator's whole lifetime.
    s: String<A>,
    /// Byte offset of the next character to emit from the front. Always on a
    /// char boundary. Advances forward in [`Iterator::next`].
    pos: usize,
    /// Byte offset marking the exclusive end of the not-yet-emitted region.
    /// Always on a char boundary. Recedes backward in [`DoubleEndedIterator::next_back`].
    end: usize,
}

impl<A: Allocator> IntoChars<A> {
    /// Wraps an owned `String` in a fresh iterator spanning its whole length.
    pub(crate) fn new(s: String<A>) -> Self {
        let len = s.len();
        Self {
            s,
            pos: 0,
            end: len,
        }
    }

    /// Returns a reference to the not-yet-emitted portion of the string.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.s[self.pos..self.end]
    }

    /// Consumes the iterator, returning the not-yet-emitted remainder as a
    /// `String`. The remaining bytes are shifted left within the existing
    /// allocation so that no new allocation is needed.
    pub fn into_string(self) -> String<A> {
        let mut s = self.s;
        // Shift the unconsumed tail `[pos..end)` to the front `[0..remaining)`.
        #[allow(clippy::arithmetic_side_effects, reason = "pos <= end invariant")]
        let remaining = self.end - self.pos;
        if remaining > 0 && self.pos > 0 {
            let base = s.buf.as_mut_ptr();
            // SAFETY: both regions are within the allocation and overlap only
            // in the forward-copy direction (src starts after dst), so
            // `ptr::copy` handles it correctly.
            unsafe {
                ptr::copy(base.add(self.pos), base, remaining);
            }
        }
        // SAFETY: `remaining` bytes at offset 0 are valid UTF-8 (they were a
        // contiguous sub-slice of valid UTF-8, now shifted to the front).
        unsafe {
            s.buf.set_len(remaining);
        }
        s
    }
}

// Manual impl rather than `#[derive(TryClone)]`: the derive would emit an impl
// for every `A: Allocator`, but cloning the inner `String<A>` only succeeds when
// its allocator is itself fallibly cloneable (`A: AllocatorTryClone`). Gating the
// impl on that bound mirrors how `String<A>` implements `TryClone`.
impl<A: AllocatorTryClone> TryClone for IntoChars<A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Self {
            s: self.s.try_clone()?,
            pos: self.pos.try_clone()?,
            end: self.end.try_clone()?,
        })
    }
}

impl<A: Allocator> Iterator for IntoChars<A> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        if self.pos >= self.end {
            return None;
        }
        // SAFETY: `pos` is always on a char boundary (starts at 0, advances by
        // exactly one decoded char's width) and `self.s` is valid UTF-8, so the
        // leading char of `&s[pos..]` decodes cleanly.
        let ch = unsafe {
            self.s
                .as_str()
                .get_unchecked(self.pos..)
                .chars()
                .next()
                .unwrap_unchecked()
        };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "ch.len_utf8 <= remaining len"
        )]
        {
            self.pos += ch.len_utf8();
        }
        Some(ch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.pos);
        // Lower bound is number of chars rounded up, assuming that every char is 4 bytes wide.
        // Upper bound is assuming every character is 1 byte wide.
        (remaining.saturating_add(3) / 4, Some(remaining))
    }
}

impl<A: Allocator> DoubleEndedIterator for IntoChars<A> {
    fn next_back(&mut self) -> Option<char> {
        if self.pos >= self.end {
            return None;
        }
        // Decode the trailing character of the not-yet-emitted region.
        let tail = &self.s.as_str()[..self.end];
        let ch = tail.chars().last()?;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "ch.len_utf8 <= unconsumed len"
        )]
        {
            self.end -= ch.len_utf8();
        }
        Some(ch)
    }
}

// Once `pos == end` the iterator yields nothing further, so it is fused.
impl<A: Allocator> FusedIterator for IntoChars<A> {}

impl<A: Allocator> fmt::Debug for IntoChars<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoChars")
            .field("remaining", &self.as_str())
            .finish_non_exhaustive()
    }
}
