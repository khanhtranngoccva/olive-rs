//! Module that pertains to any functions of [`String`] that mutate,
//! but do not allocate.

use core::{fmt, ptr};

use super::String;
use olive_core::alloc::Allocator;

/// Error returned by [`String::try_truncate`].
///
/// Truncation is infallible with respect to allocation (it only shrinks), so
/// the sole failure mode is a malformed target length: either past the end of
/// the string, or landing in the middle of a multi-byte character.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TryStringTruncateError {
    /// The requested truncation offset.
    pub new_length: usize,
}

impl fmt::Debug for TryStringTruncateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TryStringTruncateError")
            .field("new_length", &self.new_length)
            .finish()
    }
}

impl fmt::Display for TryStringTruncateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot truncate to length {length}: not on a character boundary",
            length = self.new_length,
        )
    }
}

impl core::error::Error for TryStringTruncateError {}

/// Error returned by [`String::try_remove`].
///
/// Removal is infallible with respect to allocation, so the only failure mode
/// is an index that does not start a character.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TryStringRemoveError {
    /// The byte index that was attempted.
    pub index: usize,
    /// The string's current length at the time of the call.
    pub len: usize,
}

impl fmt::Debug for TryStringRemoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TryStringRemoveError")
            .field("index", &self.index)
            .field("len", &self.len)
            .finish()
    }
}

impl fmt::Display for TryStringRemoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot remove byte {index}: not at a character boundary (string length is {len})",
            index = self.index,
            len = self.len
        )
    }
}

impl core::error::Error for TryStringRemoveError {}

impl<A: Allocator> String<A> {
    /// Truncates the string to `new_len` bytes.
    ///
    /// If `new_len` is greater than or equal to the current length, this is a
    /// no-op (the string is already at most that long).
    ///
    /// # Errors
    ///
    /// Returns [`TryStringTruncateError`] if `new_len` lands in the middle of
    /// a multi-byte character.
    pub fn try_truncate(&mut self, new_len: usize) -> Result<(), TryStringTruncateError> {
        let len = self.len();
        if new_len >= len {
            return Ok(());
        }
        if !self.is_char_boundary(new_len) {
            return Err(TryStringTruncateError {
                new_length: new_len,
            });
        }
        self.buf.truncate(new_len);
        Ok(())
    }

    /// Removes the last character from the string and returns it, or `None` if
    /// the string is empty.
    pub fn pop(&mut self) -> Option<char> {
        // Decode the trailing character *before* shrinking so we can return it.
        let c = self.chars().next_back()?;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "c's UTF-8 width is always <= self.len()"
        )]
        {
            self.buf.truncate(self.len() - c.len_utf8());
        }
        Some(c)
    }

    /// Removes the character whose first byte is at `index`, shifting the
    /// following bytes left to close the gap.
    ///
    /// This is the fallible-port analogue of std's `String::remove`. It never
    /// allocates; it only fails if `index` does not start a character.
    ///
    /// # Errors
    ///
    /// Returns [`TryStringRemoveError`] if `index` is out of bounds or lands in
    /// the middle of a multi-byte character.
    pub fn try_remove(&mut self, index: usize) -> Result<char, TryStringRemoveError> {
        let len = self.len();
        if !self.is_char_boundary(index) || index >= len {
            return Err(TryStringRemoveError { index, len });
        }
        // Decode the char being removed *before* shifting so we can return it.
        let ch = self[index..]
            .chars()
            .next()
            .unwrap_or_else(|| unreachable!("char boundary checked"));
        let width = ch.len_utf8();
        // Shift `[index + width .. len]` down by `width` bytes and shrink.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "index + width <= len (valid char fits)"
        )]
        let remaining = len - (index + width);
        let ptr = self.buf.as_mut_ptr();
        // SAFETY: shifting left `remaining` bytes by `width`. Must not occur if remaining == 0 to avoid
        // OOB pointer overflow.
        unsafe {
            if remaining > 0 {
                #[allow(clippy::arithmetic_side_effects, reason = "index + width <= len")]
                {
                    ptr::copy(ptr.add(index + width), ptr.add(index), remaining);
                }
            }
        }
        #[allow(clippy::arithmetic_side_effects, reason = "width <= len")]
        unsafe {
            self.buf.set_len(len - width);
        }
        Ok(ch)
    }

    /// Retains only the characters for which `predicate` returns `true`,
    /// dropping the rest in place. No allocation occurs.
    ///
    /// # Panics
    ///
    /// If `predicate` panics, the string is sealed into a valid state: the
    /// unchecked tail is shifted left over any holes created by rejected
    /// characters, so the result contains all originally-kept characters
    /// followed by all not-yet-examined characters (in their original order).
    pub fn retain<F>(&mut self, mut predicate: F)
    where
        F: FnMut(char) -> bool,
    {
        let len = self.len();
        if len == 0 {
            return;
        }
        let base = self.buf.as_mut_ptr();

        // Scan forward to find the first character that should be removed.
        // Characters before it are all kept, so no critical section needed yet.
        let mut read_pos = 0usize;
        // Candidate char to be potentially removed.
        let mut candidate_char;
        loop {
            // SAFETY: read_pos < len (guarded below).
            #[allow(clippy::arithmetic_side_effects, reason = "read_pos < len")]
            let rest = unsafe { core::slice::from_raw_parts(base.add(read_pos), len - read_pos) };
            candidate_char = unsafe { core::str::from_utf8_unchecked(rest) }
                .chars()
                .next()
                .unwrap_or_else(|| unreachable!("non-empty slice has a leading char"));
            let w = candidate_char.len_utf8();
            if !predicate(candidate_char) {
                break;
            }
            #[allow(clippy::arithmetic_side_effects, reason = "read_pos + w <= len")]
            {
                read_pos += w;
            }
            if read_pos == len {
                // All characters kept; nothing to do.
                return;
            }
        }

        // Critical section: at least one character will be removed.
        // On unwind, we shift the unchecked tail left to seal gaps, then
        // restore the length. We track state in plain variables and use a
        // small RAII guard that borrows `self` exclusively.
        struct RetainGuard<'a, A: Allocator> {
            string: &'a mut String<A>,
            base: *mut u8,
            read: usize,
            write: usize,
            original_len: usize,
        }
        impl<A: Allocator> Drop for RetainGuard<'_, A> {
            #[cold]
            fn drop(&mut self) {
                #[allow(clippy::arithmetic_side_effects, reason = "read <= original_len")]
                let remaining = self.original_len - self.read;
                // SAFETY: Need to check to prevent OOB pointer.
                if remaining > 0 {
                    // SAFETY: The unchecked tail `[read..original_len)` consists
                    // of whole characters. Shifting it left to position `write`
                    // (also a char boundary) keeps the buffer valid UTF-8.
                    unsafe {
                        ptr::copy(
                            self.base.add(self.read),
                            self.base.add(self.write),
                            remaining,
                        );
                    }
                }
                // SAFETY: After filling holes, all bytes are contiguous and
                // valid UTF-8; the new length equals `write + remaining`.
                unsafe {
                    #[allow(
                        clippy::arithmetic_side_effects,
                        reason = "write + remaining <= original_len"
                    )]
                    {
                        self.string.buf.set_len(self.write + remaining);
                    }
                }
            }
        }

        let mut g = RetainGuard {
            string: self,
            base,
            read: read_pos,
            write: read_pos,
            original_len: len,
        };

        // Process the first rejected character: advance `read` past it.
        // We don't copy anything for it (it's dropped).
        let first_width = candidate_char.len_utf8();
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "read_pos + first_width <= len"
        )]
        {
            g.read += first_width;
        }

        // Continue scanning from after the first rejection.
        while g.read < g.original_len {
            // SAFETY: g.read < original_len.
            #[allow(clippy::arithmetic_side_effects, reason = "g.read < original_len")]
            let rest =
                unsafe { core::slice::from_raw_parts(base.add(g.read), g.original_len - g.read) };
            let ch = unsafe { core::str::from_utf8_unchecked(rest) }
                .chars()
                .next()
                .unwrap_or_else(|| unreachable!("non-empty slice has a leading char"));
            let w = ch.len_utf8();
            if predicate(ch) {
                if g.read != g.write {
                    // SAFETY: Both offsets are on char boundaries; copying `w`
                    // bytes from `[g.read..g.read+w)` to `[g.write..g.write+w)`
                    // is in-bounds and overlapping only forward.
                    unsafe {
                        ptr::copy(base.add(g.read), base.add(g.write), w);
                    }
                }
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "g.write + w <= original_len"
                )]
                {
                    g.write += w;
                }
            }
            #[allow(clippy::arithmetic_side_effects, reason = "g.read + w <= original_len")]
            {
                g.read += w;
            }
        }

        // Success path: commit the final length and forget the guard.
        // SAFETY: `g.write` is the total bytes of kept characters, all valid.
        unsafe { g.string.buf.set_len(g.write) };
        core::mem::forget(g);
    }

    /// Clears the string, retaining its allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::ops::Deref;

    use super::*;

    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    // ── try_truncate ──────────────────────────────────────────────────────────

    #[test]
    fn try_truncate_shrinks_len_not_cap() {
        let mut s = mk("hello world");
        s.try_reserve(32).unwrap();
        let expected_cap = s.capacity();
        s.try_truncate(5).unwrap();
        assert_eq!(s.deref(), "hello");
        assert!(s.capacity() == expected_cap);
    }

    #[test]
    fn try_truncate_rejects_mid_char() {
        let mut s = mk("é🦊");
        // é occupies bytes [0, 2); byte index 1 is inside it.
        let err = s.try_truncate(1).unwrap_err();
        assert_eq!(err.new_length, 1);
        assert_eq!(s.deref(), "é🦊");
    }

    #[test]
    fn try_truncate_at_or_beyond_len_is_noop() {
        let mut s = mk("hello");
        s.try_truncate(5).unwrap();
        assert_eq!(s.deref(), "hello");
        s.try_truncate(100).unwrap();
        assert_eq!(s.deref(), "hello");
    }

    // ── pop ───────────────────────────────────────────────────────────────────

    #[test]
    fn pop_removes_last_char() {
        let mut s = mk("aé🦊");
        assert_eq!(s.pop(), Some('🦊'));
        assert_eq!(s.pop(), Some('é'));
        assert_eq!(s.pop(), Some('a'));
        assert_eq!(s.pop(), None);
        assert!(s.is_empty());
    }

    // ── try_remove ────────────────────────────────────────────────────────────

    #[test]
    fn try_remove_ascii() {
        let mut s = mk("hello");
        assert_eq!(s.try_remove(0).unwrap(), 'h');
        assert_eq!(s.deref(), "ello");
    }

    #[test]
    fn try_remove_multibyte() {
        let mut s = mk("h🦊i");
        assert_eq!(s.try_remove(1).unwrap(), '🦊');
        assert_eq!(s.deref(), "hi");
    }

    #[test]
    fn try_remove_mid_char_fails() {
        let mut s = mk("é🦊");
        // é occupies bytes [0, 2); byte index 1 is inside it.
        let err = s.try_remove(1).unwrap_err();
        assert_eq!(err.index, 1);
        assert_eq!(err.len, 6);
        assert_eq!(s.deref(), "é🦊");
    }

    #[test]
    fn try_remove_out_of_bounds() {
        let mut s = mk("ab");
        let err = s.try_remove(5).unwrap_err();
        assert_eq!(err.index, 5);
        assert_eq!(err.len, 2);
        assert_eq!(s.deref(), "ab");
    }

    // ── retain ────────────────────────────────────────────────────────────────

    #[test]
    fn retain_keeps_matching() {
        let mut s = mk("a1b2c3");
        s.retain(|c| c.is_ascii_alphabetic());
        assert_eq!(s.deref(), "abc");
    }

    #[test]
    fn retain_multibyte() {
        let mut s = mk("aé🦊z");
        s.retain(|c| c != 'é');
        assert_eq!(s.deref(), "a🦊z");
    }

    #[test]
    fn retain_none_left() {
        let mut s = mk("123");
        s.retain(|_c| false);
        assert_eq!(s.deref(), "");
    }

    #[test]
    fn retain_predicate_panic_seals_valid_utf8() {
        let mut s = mk("aéb🦊d");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.retain(|c| {
                if c == 'b' {
                    panic!("predicate panicked");
                }
                if c == 'é' {
                    return false;
                }
                true
            })
        }));
        assert!(result.is_err());
        assert!(core::str::from_utf8(s.as_bytes()).is_ok());
        assert_eq!(s.deref(), "ab🦊d");
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn clear_empties_but_keeps_capacity() {
        let mut s = mk("hello");
        s.try_reserve(32).unwrap();
        let cap = s.capacity();
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.capacity(), cap);
        // The buffer must remain reusable after a clear.
        s.try_push_str("new").unwrap();
        assert_eq!(s.deref(), "new");
    }
}
