//! Implementations of [`String`] APIs that directly deal with encoding logic.

use core::{fmt, iter::FusedIterator};

use super::String;
use crate::{alloc::Global, vec::Vec};
use olive_core::{TryClone, alloc::Allocator, alloc_errors::TryReserveError};

/// Error returned by [`String::from_utf8`] and [`String::from_utf8_in`] when
/// the supplied bytes are not valid UTF-8.
#[derive(TryClone)]
pub struct FromUtf8Error<A: Allocator = Global> {
    pub(crate) error: core::str::Utf8Error,
    pub(crate) bytes: Vec<u8, A>,
}

// The allocator field is intentionally excluded from `Debug`: `Allocator` does
// not require it, and the useful payload (the offending bytes + validation
// error) is what a caller debugging a decode failure actually needs.
impl<A: Allocator> fmt::Debug for FromUtf8Error<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FromUtf8Error")
            .field("error", &self.error)
            .field("bytes", &self.bytes.as_slice())
            .finish()
    }
}

impl<A: Allocator> FromUtf8Error<A> {
    /// Returns the bytes that failed to validate as UTF-8.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8, A> {
        self.bytes
    }

    /// Returns the underlying UTF-8 validation error.
    #[inline]
    pub fn utf8_error(&self) -> &core::str::Utf8Error {
        &self.error
    }
}

impl<A: Allocator> fmt::Display for FromUtf8Error<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}

impl<A: Allocator> core::error::Error for FromUtf8Error<A> {}

/// Error returned by the lossless `try_from_utf16*` constructors when decoding
/// a UTF-16 buffer into a [`String`] fails.
///
/// Wraps the failure [`kind`](Self::kind): either the byte-input variant saw an
/// odd number of bytes, a lone surrogate was encountered mid-decode, or a
/// capacity reservation failed while growing the output buffer.
// The convention of this error is slightly different - it uses a nested struct for
// std parity, helpful for incremental porting.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TryFromUtf16Error {
    /// Which category of failure occurred.
    pub kind: TryFromUtf16ErrorKind,
}

impl TryFromUtf16Error {
    /// Returns `true` if the failure was an odd-length byte input.
    #[inline]
    #[must_use]
    pub const fn is_odd_bytes(&self) -> bool {
        matches!(self.kind, TryFromUtf16ErrorKind::OddBytes)
    }

    /// Returns `true` if the input contained an unpaired surrogate half.
    #[inline]
    #[must_use]
    pub const fn is_lone_surrogate(&self) -> bool {
        matches!(self.kind, TryFromUtf16ErrorKind::LoneSurrogate(_))
    }

    /// Returns `true` if a capacity reservation failed during conversion.
    #[inline]
    #[must_use]
    pub const fn is_reserve(&self) -> bool {
        matches!(self.kind, TryFromUtf16ErrorKind::Reserve(_))
    }
}

/// The category of failure behind a [`TryFromUtf16Error`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TryFromUtf16ErrorKind {
    /// The byte (`u8`)-slice input had an odd length and therefore could not be split
    /// into whole 2-byte code units.
    OddBytes,
    /// The decoded stream contained a high or low surrogate without its pair.
    /// Carries the zero-based **code unit index** (position in the sequence of
    /// `u16` values) where the unpaired surrogate was encountered. This is
    /// meaningful regardless of whether the input originated from a `&[u8]`
    /// (LE/BE byte slice) or a `&[u16]` slice — it always refers to the Nth
    /// 16-bit code unit in logical order.
    LoneSurrogate(usize),
    /// A capacity reservation on the destination buffer failed (overflow or OOM).
    Reserve(TryReserveError),
}

impl fmt::Debug for TryFromUtf16Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TryFromUtf16Error")
            .field("kind", &self.kind)
            .finish()
    }
}

impl fmt::Display for TryFromUtf16Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TryFromUtf16ErrorKind::OddBytes => write!(f, "input has an odd number of bytes"),
            TryFromUtf16ErrorKind::LoneSurrogate(idx) => {
                write!(f, "unpaired surrogate at code-unit index {idx}")
            }
            TryFromUtf16ErrorKind::Reserve(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for TryFromUtf16Error {}

impl From<TryReserveError> for TryFromUtf16Error {
    #[inline]
    fn from(e: TryReserveError) -> Self {
        Self {
            kind: TryFromUtf16ErrorKind::Reserve(e),
        }
    }
}

impl String {
    /// Validates that the specified byte slice is valid UTF-8, then creates a
    /// `String` from it, allocating through the global allocator.
    ///
    /// # Errors
    ///
    /// Returns a [`FromUtf8Error`] if the byte slice was not valid UTF-8.
    pub fn from_utf8(v: Vec<u8>) -> Result<String, FromUtf8Error> {
        match core::str::from_utf8(&v) {
            Ok(_) => Ok(String { buf: v }),
            Err(e) => Err(FromUtf8Error { error: e, bytes: v }),
        }
    }

    /// Attempts to convert an `u16` slice holding UTF-16-encoded text into a
    /// `String`, assuming native-endian code-unit order, allocating through the
    /// global allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryFromUtf16Error`] if the input contains an unpaired surrogate
    /// or reserving the buffer fails.
    pub fn try_from_utf16(input: &[u16]) -> Result<Self, TryFromUtf16Error> {
        Self::try_from_utf16_in(input, Global)
    }

    /// Attempts to convert a byte slice holding UTF-16BE-encoded text into a
    /// `String`, allocating through the global allocator.
    ///
    /// On big-endian targets this reduces to [`try_from_utf16`](Self::try_from_utf16)
    /// when the slice is 2-byte aligned; otherwise each pair of bytes is
    /// reinterpreted as a big-endian `u16`.
    ///
    /// # Errors
    ///
    /// Returns [`TryFromUtf16Error`] if the input has an odd number of bytes,
    /// contains an unpaired surrogate, or reserving the buffer fails.
    pub fn try_from_utf16be(input: &[u8]) -> Result<Self, TryFromUtf16Error> {
        Self::try_from_utf16be_in(input, Global)
    }

    /// Attempts to convert a byte slice holding UTF-16LE-encoded text into a
    /// `String`, allocating through the global allocator.
    ///
    /// On little-endian targets this reduces to [`try_from_utf16`](Self::try_from_utf16)
    /// when the slice is 2-byte aligned; otherwise each pair of bytes is
    /// reinterpreted as a little-endian `u16`.
    ///
    /// # Errors
    ///
    /// Returns [`TryFromUtf16Error`] if the input has an odd number of bytes,
    /// contains an unpaired surrogate, or reserving the buffer fails.
    pub fn try_from_utf16le(input: &[u8]) -> Result<Self, TryFromUtf16Error> {
        Self::try_from_utf16le_in(input, Global)
    }

    /// Attempts to convert a `u16` slice holding UTF-16-encoded text into a
    /// `String`, assuming native-endian code-unit order, replacing any invalid
    /// sequences with the Unicode replacement character (U+FFFD).
    ///
    /// Only a failed reservation can error; malformed input never does.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving the buffer fails.
    pub fn try_from_utf16_lossy(input: &[u16]) -> Result<Self, TryReserveError> {
        Self::try_from_utf16_lossy_in(input, Global)
    }

    /// Attempts to convert a byte slice holding UTF-16BE-encoded text into a
    /// `String`, replacing any invalid sequences with the Unicode replacement
    /// character (U+FFFD).
    ///
    /// Only a failed reservation can error; malformed input never does.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving the buffer fails.
    pub fn try_from_utf16be_lossy(input: &[u8]) -> Result<Self, TryReserveError> {
        Self::try_from_utf16be_lossy_in(input, Global)
    }

    /// Attempts to convert a byte slice holding UTF-16LE-encoded text into a
    /// `String`, replacing any invalid sequences with the Unicode replacement
    /// character (U+FFFD).
    ///
    /// Only a failed reservation can error; malformed input never does.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving the buffer fails.
    pub fn try_from_utf16le_lossy(input: &[u8]) -> Result<Self, TryReserveError> {
        Self::try_from_utf16le_lossy_in(input, Global)
    }
}

impl<A: Allocator> String<A> {
    /// Validates that the specified byte slice is valid UTF-8, then creates a
    /// `String` from it, retaining the buffer's existing allocator.
    ///
    /// # Errors
    ///
    /// Returns a [`FromUtf8Error`] if the byte slice was not valid UTF-8.
    pub fn from_utf8_in(v: Vec<u8, A>) -> Result<String<A>, FromUtf8Error<A>> {
        match core::str::from_utf8(&v) {
            Ok(_) => Ok(String { buf: v }),
            Err(e) => Err(FromUtf8Error { error: e, bytes: v }),
        }
    }

    /// Attempts to convert a `u16` slice holding UTF-16-encoded text into a
    /// `String`, allocating through `alloc`.
    ///
    /// # Errors
    ///
    /// Returns [`TryFromUtf16Error`] if the input has an odd number of bytes,
    /// contains an unpaired surrogate, or reserving the buffer fails.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn try_from_utf16_in(input: &[u16], alloc: A) -> Result<Self, TryFromUtf16Error> {
        // SAFETY: the iterator is a slice iterator that yields exactly
        // `input.len()` items, satisfying "at most count".
        unsafe { Self::try_from_codeunits_in(input.iter().copied(), input.len(), alloc) }
    }

    /// Attempts to convert a byte slice holding UTF-16LE-encoded text into a
    /// `String`, allocating through `alloc`.
    ///
    /// # Errors
    ///
    /// Returns [`TryFromUtf16Error`] if the input has an odd number of bytes,
    /// contains an unpaired surrogate, or reserving the buffer fails.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn try_from_utf16le_in(input: &[u8], alloc: A) -> Result<Self, TryFromUtf16Error> {
        if input.len() % 2 != 0 {
            return Err(TryFromUtf16Error {
                kind: TryFromUtf16ErrorKind::OddBytes,
            });
        }
        match (cfg!(target_endian = "little"), unsafe {
            input.align_to::<u16>()
        }) {
            (true, ([], v, [])) => Self::try_from_utf16_in(v, alloc),
            _ => {
                let (iter, count) =
                    convert_slice_to_u16_le_iter(input).map_err(|e| TryFromUtf16Error {
                        kind: TryFromUtf16ErrorKind::Reserve(e),
                    })?;
                // SAFETY: `convert_slice_to_u16_le_iter` guarantees the
                // iterator yields at most `count` items.
                unsafe { Self::try_from_codeunits_in(iter, count, alloc) }
            }
        }
    }

    /// Attempts to convert a byte slice holding UTF-16BE-encoded text into a
    /// `String`, allocating through `alloc`.
    ///
    /// # Errors
    ///
    /// Returns [`TryFromUtf16Error`] if the input has an odd number of bytes,
    /// contains an unpaired surrogate, or reserving the buffer fails.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn try_from_utf16be_in(input: &[u8], alloc: A) -> Result<Self, TryFromUtf16Error> {
        if input.len() % 2 != 0 {
            return Err(TryFromUtf16Error {
                kind: TryFromUtf16ErrorKind::OddBytes,
            });
        }
        match (cfg!(target_endian = "big"), unsafe {
            input.align_to::<u16>()
        }) {
            (true, ([], v, [])) => Self::try_from_utf16_in(v, alloc),
            _ => {
                let (iter, count) =
                    convert_slice_to_u16_be_iter(input).map_err(|e| TryFromUtf16Error {
                        kind: TryFromUtf16ErrorKind::Reserve(e),
                    })?;
                // SAFETY: `convert_slice_to_u16_be_iter` guarantees the
                // iterator yields at most `count` items.
                unsafe { Self::try_from_codeunits_in(iter, count, alloc) }
            }
        }
    }

    /// Attempts to convert a `u16` slice holding UTF-16-encoded text into a
    /// `String`.
    ///
    /// Only a failed reservation can error; malformed input never does.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving the buffer fails.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn try_from_utf16_lossy_in(input: &[u16], alloc: A) -> Result<Self, TryReserveError> {
        Self::try_from_codeunits_lossy_in(input.iter().cloned(), input.len(), alloc)
    }

    /// Attempts to convert a byte slice holding UTF-16LE-encoded text into a
    /// `String`, replacing any invalid sequences with the Unicode replacement
    /// character (U+FFFD), allocating through `alloc`.
    ///
    /// # Errors
    ///
    /// Returns [`TryFromUtf16Error`] if reserving the buffer fails.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn try_from_utf16le_lossy_in(input: &[u8], alloc: A) -> Result<Self, TryReserveError> {
        match (cfg!(target_endian = "little"), unsafe {
            input.align_to::<u16>()
        }) {
            (true, ([], v, [])) => Self::try_from_utf16_lossy_in(v, alloc),
            (true, ([], v, [_remainder])) => {
                // Reserve one more byte for the replacement
                let len_with_replacement = v
                    .len()
                    .checked_add(1)
                    .ok_or(TryReserveError::new_capacity_overflow())?;
                let mut s = Self::try_from_codeunits_lossy_in(
                    v.iter().cloned(),
                    len_with_replacement,
                    alloc,
                )?;
                s.try_push(char::REPLACEMENT_CHARACTER)?;
                Ok(s)
            }
            _ => {
                let (iter, count) = convert_slice_to_u16_le_iter(input)?;
                Self::try_from_codeunits_lossy_in(iter, count, alloc)
            }
        }
    }

    /// Attempts to convert a byte slice holding UTF-16BE-encoded text into a
    /// `String`, replacing any invalid sequences with the Unicode replacement
    /// character (U+FFFD), allocating through `alloc`.
    ///
    /// # Errors
    ///
    /// Returns [`TryFromUtf16Error`] if reserving the buffer fails.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn try_from_utf16be_lossy_in(input: &[u8], alloc: A) -> Result<Self, TryReserveError> {
        match (cfg!(target_endian = "big"), unsafe {
            input.align_to::<u16>()
        }) {
            (true, ([], v, [])) => Self::try_from_utf16_lossy_in(v, alloc),
            (true, ([], v, [_remainder])) => {
                // Reserve one more byte for the replacement
                let len_with_replacement = v
                    .len()
                    .checked_add(1)
                    .ok_or(TryReserveError::new_capacity_overflow())?;
                let mut s = Self::try_from_codeunits_lossy_in(
                    v.iter().cloned(),
                    len_with_replacement,
                    alloc,
                )?;
                s.try_push(char::REPLACEMENT_CHARACTER)?;
                Ok(s)
            }
            _ => {
                let (iter, count) = convert_slice_to_u16_be_iter(input)?;
                Self::try_from_codeunits_lossy_in(iter, count, alloc)
            }
        }
    }

    /// Core decoder over an iterator of native-endian `u16` code units.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the iterator yields **at most** `count`
    /// items. Fewer items are permitted (the iterator simply ends early); more
    /// would violate the capacity reservation made from `count` and may cause
    /// overflows.
    unsafe fn try_from_codeunits_in<I>(
        input: I,
        count: usize,
        alloc: A,
    ) -> Result<Self, TryFromUtf16Error>
    where
        I: IntoIterator<Item = u16>,
    {
        // Reserve ahead of time
        let cap = count
            .checked_mul(size_of::<char>())
            .ok_or(TryFromUtf16Error {
                kind: TryFromUtf16ErrorKind::Reserve(TryReserveError::new_capacity_overflow()),
            })?;
        let mut ret = String::try_with_capacity_in(cap, alloc).map_err(|e| TryFromUtf16Error {
            kind: TryFromUtf16ErrorKind::Reserve(e),
        })?;

        let mut cursor = 0usize;
        for decoded in char::decode_utf16(input.into_iter()) {
            let ch = decoded.map_err(|_| TryFromUtf16Error {
                kind: TryFromUtf16ErrorKind::LoneSurrogate(cursor),
            })?;
            #[allow(clippy::arithmetic_side_effects, reason = "bounded by count")]
            {
                cursor += ch.len_utf16();
            }
            ret.try_push(ch).map_err(|e| TryFromUtf16Error {
                kind: TryFromUtf16ErrorKind::Reserve(e),
            })?;
        }
        Ok(ret)
    }

    /// Lossy counterpart of [`try_from_codeunits_in`](Self::try_from_codeunits_in):
    /// undecodable units yield the replacement character instead of an error.
    fn try_from_codeunits_lossy_in<I>(
        input: I,
        count: usize,
        alloc: A,
    ) -> Result<Self, TryReserveError>
    where
        I: IntoIterator<Item = u16>,
    {
        // Reserve ahead of time
        let cap = count
            .checked_mul(size_of::<char>())
            .ok_or(TryReserveError::new_capacity_overflow())?;
        let mut ret = String::try_with_capacity_in(cap, alloc)?;
        for c in char::decode_utf16(input.into_iter()) {
            let c = c.unwrap_or(char::REPLACEMENT_CHARACTER);
            ret.try_push(c)?;
        }
        Ok(ret)
    }

    /// Interprets a raw byte buffer as a `String` without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the buffer pointed to by `ptr` must contain
    /// valid UTF-8.
    pub unsafe fn from_utf8_unchecked(buf: Vec<u8, A>) -> Self {
        debug_assert!(core::str::from_utf8(&buf).is_ok());
        Self { buf }
    }
}

/// Lossily converts a byte slice to a known-size u16 iterator using little-endian encoding,
/// adding a replacement character in place of the last malformed byte.
/// If the number of bytes in the slice is even, the conversion can be lossless, making the
/// function usable in lossless implementations.
fn convert_slice_to_u16_le_iter(
    input: &[u8],
) -> Result<(impl FusedIterator<Item = u16>, usize), TryReserveError> {
    // If `even_bytes` is false, then the last chunk is odd and malformed, and input.len() / 2
    // will round down, so adding 1 is necessary.
    let even_bytes = input.len() & 1 == 0;
    let len = (input.len() / 2)
        .checked_add(even_bytes as usize)
        .ok_or(TryReserveError::new_capacity_overflow())?;
    Ok((
        input.chunks(2).map(|chunk| {
            <[u8; 2]>::try_from(chunk)
                .map(u16::from_le_bytes)
                .unwrap_or(char::REPLACEMENT_CHARACTER as u16)
        }),
        len,
    ))
}

/// Lossily converts a byte slice to a known-size u16 iterator using big-endian encoding,
/// adding a replacement character in place of the last malformed byte.
/// If the number of bytes in the slice is even, the conversion can be lossless, making the
/// function usable in lossless implementations.
fn convert_slice_to_u16_be_iter(
    input: &[u8],
) -> Result<(impl FusedIterator<Item = u16>, usize), TryReserveError> {
    // If `even_bytes` is false, then the last chunk is odd and malformed, and input.len() / 2
    // will round down, so adding 1 is necessary.
    let even_bytes = input.len() & 1 == 0;
    let len = (input.len() / 2)
        .checked_add(even_bytes as usize)
        .ok_or(TryReserveError::new_capacity_overflow())?;
    Ok((
        input.chunks(2).map(|chunk| {
            <[u8; 2]>::try_from(chunk)
                .map(u16::from_be_bytes)
                .unwrap_or(char::REPLACEMENT_CHARACTER as u16)
        }),
        len,
    ))
}
