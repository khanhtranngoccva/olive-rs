//! A simple, unpacked error representation.
//!
//! This is the safe, boring variant of the payload storage: a plain newtype
//! over an enum, with every alternative held by value.
//!
//! It is ported from `repr_unpacked.rs` (nightly-era upstream), which
//! upstream uses on targets where a bit-packed representation would not work
//! or would bring no benefit.
//!
//! Once the basic structure is settled, a bit-packed variant
//! (upstream `repr_bitpacked.rs`) can land behind a
//! `cfg_attr(path = ...)` switch without touching any other file.

use super::{Custom, ErrorData, ErrorKind, RawOsError, SimpleMessage};

type Inner = ErrorData<Custom>;

/// The unpacked [`Error`](super::Error) payload, holding one of the four
/// error shapes directly.
pub(super) struct Repr(Inner);

impl Repr {
    #[inline]
    pub(super) fn new_custom(b: Custom) -> Self {
        Self(Inner::Custom(b))
    }

    #[inline]
    pub(super) fn new_os(code: RawOsError) -> Self {
        Self(Inner::Os(code))
    }

    #[inline]
    pub(super) fn new_simple(kind: ErrorKind) -> Self {
        Self(Inner::Simple(kind))
    }

    #[inline]
    pub(super) const fn new_simple_message(m: &'static SimpleMessage) -> Self {
        Self(Inner::SimpleMessage(m))
    }

    #[inline]
    pub(super) fn into_data(self) -> Inner {
        self.0
    }

    #[inline]
    pub(super) fn data(&self) -> ErrorData<&Custom> {
        match &self.0 {
            Inner::Os(c) => ErrorData::Os(*c),
            Inner::Simple(k) => ErrorData::Simple(*k),
            Inner::SimpleMessage(m) => ErrorData::SimpleMessage(m),
            Inner::Custom(m) => ErrorData::Custom(m),
        }
    }

    #[inline]
    pub(super) fn data_mut(&mut self) -> ErrorData<&mut Custom> {
        match &mut self.0 {
            Inner::Os(c) => ErrorData::Os(*c),
            Inner::Simple(k) => ErrorData::Simple(*k),
            Inner::SimpleMessage(m) => ErrorData::SimpleMessage(m),
            Inner::Custom(m) => ErrorData::Custom(m),
        }
    }
}
