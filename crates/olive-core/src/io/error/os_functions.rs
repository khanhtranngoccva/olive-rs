//! OS-function plumbing without atomics.
//!
//! Upstream `core` picks this module over `os_functions_atomic`
//! with `cfg_attr(path = ...)` on targets (or builds) that do not provide an
//! atomic pointer or when I/O statics are disabled.
//!
//! The registered [`OsFunctions`] vtable cannot be set with an atomic pointer
//! or declared in a variable (or both), so every dispatch goes through the
//! built-in default table and `set_functions` is a no-op.
//!
//! Olive mirrors that split verbatim.

use super::{ErrorKind, OsFunctions, RawOsError};
use core::fmt;

/// Sets the platform's I/O error machinery.
///
/// Without atomic support or no I/O static variable support
/// there is nowhere to install the table, so this
/// stores nothing and every dispatch keeps using the default table.
///
/// # Safety
///
/// The provided reference must point to data that is entirely constant; it must
/// not be created during runtime.
#[inline]
pub(super) unsafe fn set_functions(_f: &'static OsFunctions) {}

/// Formats the human-readable description of a raw OS error code.
#[inline]
pub(super) fn format_os_error(errno: RawOsError, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    let functions = OsFunctions::DEFAULT;
    (functions.format_os_error)(errno, fmt)
}

/// Decodes a raw OS error code into an [`ErrorKind`].
#[inline]
pub(super) fn decode_error_kind(errno: RawOsError) -> ErrorKind {
    let functions = OsFunctions::DEFAULT;
    (functions.decode_error_kind)(errno)
}

/// Reports whether a raw OS error code represents an interruption.
#[inline]
pub(super) fn is_interrupted(errno: RawOsError) -> bool {
    let functions = OsFunctions::DEFAULT;
    (functions.is_interrupted)(errno)
}
