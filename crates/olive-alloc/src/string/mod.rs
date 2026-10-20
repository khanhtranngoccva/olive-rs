//! An owned, growable UTF-8 string that never panics on allocation failure.
//!
//! This is Olive's port of the standard library's [`String`](crate::stock_alloc::string::String).
//! Where std's `String` grows by reserving capacity and *panics* when the global
//! allocator cannot satisfy a request or when the request cannot be satisfied otherwise,
//! Olive's `String` returns a [`Result`] instead, so callers can react to these
//! errors gracefully.
//!
//! Like the rest of Olive, the type itself assumes no allocator: the default
//! methods route through the process-wide global allocator ([`Global`]), while
//! the `_in(...)` variants accept any [`Allocator`], giving callers a seam for
//! custom allocators without changing the trait surface.
//!
//! # Fallibility
//!
//! Operations that only read or mutate already-backed bytes are infallible;
//! operations that may grow the buffer return a [`Result`]:
//!
//! - [`try_reserve`](String::try_reserve) / [`try_reserve_exact`](String::try_reserve_exact)
//!   — the primary entry points, surfacing [`TryReserveError`] directly.
//! - Growth-driven mutations (`try_push_str`, `try_push_char`, `try_insert_str`, …)
//!   propagate a reservation failure wrapped in their own error type.
//! - Collection-building entry points implement the fallible traits from
//!   `olive_core::try_traits`: [`TryClone`], [`TryExtend`],
//!   [`TryFromIterator`], and (via blanket impls) [`TryCollect`](olive_core::try_traits::TryCollect) or
//!   [`TryCollectInto`](olive_core::try_traits::TryCollectInto).
//!
//! # Representation
//! - Like the original [`String`](crate::stock_alloc::string::String), it hosts a byte
//!   vector, with constraints of any other byte vector, including having a pointer, length,
//!   and capacity, and is stored on the heap.
//!
//! # UTF-8 and Deref
//! All content is guaranteed to be valid UTF-8; the public API exposes it as `&str`
//! via [`Deref`], so that [`String`] inherits all its methods.
//!
//! [`Deref`]: core::ops::Deref

mod add;
mod allocation;
mod cmp;
mod construction;
mod convert;
mod encode;
mod into_chars;
mod mutation;
mod query;
mod reconstitution;
mod traits;

pub use add::Concat;
use core::fmt::{self, Display, Write};
pub use into_chars::IntoChars;

use crate::alloc::{Allocator, Global};
use crate::vec::Vec;

pub use allocation::{
    TryStringExtendFromWithinError, TryStringInsertError, TryStringReplaceRangeError,
    TryStringSplitOffError,
};
pub use encode::{FromUtf8Error, TryFromUtf16Error, TryFromUtf16ErrorKind};
pub use mutation::{TryStringRemoveError, TryStringTruncateError};

/// An owned, growable UTF-8 string that falls back to an error rather than
/// panicking when the allocator cannot satisfy a request.
pub struct String<A: Allocator = Global> {
    buf: Vec<u8, A>,
}

/// A fallible analogue of [`ToString`](stock_alloc::string::ToString),
/// delegating to [`Display`] but returning a [`Result`] instead of
/// panicking or aborting on allocation failure.
///
/// # Error type
///
/// Both methods return [`fmt::Error`], *not* [`TryReserveError`]. That is
/// deliberate and honest: rendering is driven by [`core::fmt::write`], which
/// funnels every possible failure, including the [`String`] allocation failures,
/// into the same opaque unit [`fmt::Error`]. There is no way at this layer
/// to distinguish those causes, so we do not pretend there is by mapping them onto a
/// fabricated [`TryReserveError`].
///
/// If you need to know whether your buffer specifically ran out of
/// memory, build the string yourself with [`try_push_str`](String::try_push_str)
/// / [`try_reserve`](String::try_reserve), which surface the precise [`TryReserveError`].
pub trait TryToString: Display {
    /// Renders the receiver via [`Display`] into a new [`String`] backed by the
    /// global allocator ([`Global`]).
    ///
    /// # Errors
    ///
    /// Returns [`fmt::Error`] if the render does not complete (see the trait
    /// docs for why the cause is not recoverable here). The receiver is left
    /// untouched; on failure the partially-filled buffer is dropped.
    fn try_to_string(&self) -> Result<String<Global>, fmt::Error> {
        self.try_to_string_in(Global)
    }

    /// Renders the receiver via [`Display`] into a new [`String`] backed by
    /// `alloc`.
    ///
    /// This is the explicit-allocator counterpart of [`try_to_string`](Self::try_to_string):
    /// same rendering, caller-chosen destination allocator. Rendering is driven
    /// by [`fmt::Write`] on the freshly-built buffer, so it works for any
    /// [`Display`] value, not just strings.
    ///
    /// # Errors
    ///
    /// Returns [`fmt::Error`] if the render does not complete (see the trait
    /// docs for why the cause is not recoverable here). On failure the
    /// partially-filled buffer is dropped and the receiver is left untouched.
    fn try_to_string_in<A2: Allocator>(&self, alloc: A2) -> Result<String<A2>, fmt::Error> {
        let mut out = String::<A2>::new_in(alloc);
        // `write_fmt` calls back into `fmt::Write::write_str` for each emitted
        // fragment. Any failure — our buffer OOMing, the `Display` impl's own
        // allocations failing, or a panic in `fmt` — arrives as the same opaque
        // unit `fmt::Error`; we return it as-is rather than inventing a more
        // specific cause we cannot actually observe.
        out.write_fmt(core::format_args!("{self}"))?;
        Ok(out)
    }
}

// Blanket impl: every `Display` type can render itself into an owned Olive
// `String`. This mirrors std's `impl<T: Display> ToString for T`, made fallible.
impl<T: Display + ?Sized> TryToString for T {}

// ──────────────────────────────────────────────────────────────────────────────
// Conversions
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use core::fmt::{self, Display};
    use core::ops::Deref;

    use super::*;
    use crate::alloc::CountingAllocator;
    use olive_test_helpers::FailAlloc;

    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    // ── TryToString ───────────────────────────────────────────────────────────

    #[test]
    fn try_to_string_matches_display() {
        let s = mk("héllo wörld ✓");
        assert_eq!(s.try_to_string().unwrap().deref(), "héllo wörld ✓");
    }

    #[test]
    fn try_to_string_empty() {
        let s = String::new();
        assert_eq!(s.try_to_string().unwrap().deref(), "");
    }

    #[test]
    fn try_to_string_in_custom_allocator() {
        let alloc = CountingAllocator::new();
        let s = mk("payload");
        // Pass a reference: `CountingAllocator` is not `Copy`, but its counters
        // live behind interior mutability, so the string's stored `&alloc`
        // observes the same instance we hold here.
        let out: String<&CountingAllocator> = s.try_to_string_in(&alloc).unwrap();
        assert_eq!(out.deref(), "payload");
        // The output buffer was allocated through our counting allocator.
        assert!(alloc.allocations() >= 1);
    }

    #[test]
    fn try_to_string_oom_returns_fmt_error() {
        let s = mk("data");
        // A failing allocator must surface an error, not panic. Because rendering
        // goes through `core::fmt::write`, the failure arrives as the opaque unit
        // `fmt::Error` — there is no cause to inspect here, only that it failed.
        let _err: fmt::Error = s.try_to_string_in(FailAlloc).unwrap_err();
    }

    /// The blanket `impl<T: Display> TryToString for T` means any `Display` type
    /// — not just `String` — can render itself into an owned Olive string.
    #[test]
    fn try_to_string_blanket_for_non_string_display() {
        // Primitives implement `Display` via std; our trait follows.
        let n: i32 = -42;
        assert_eq!(n.try_to_string().unwrap().deref(), "-42");

        // A custom `Display` type renders through its own formatter.
        struct Pair(i32, i32);
        impl Display for Pair {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "({},{})", self.0, self.1)
            }
        }
        let p = Pair(7, 8);
        assert_eq!(p.try_to_string().unwrap().deref(), "(7,8)");
    }

    #[test]
    fn try_to_string_does_not_mutate_receiver() {
        let s = mk("source");
        let _ = s.try_to_string().unwrap();
        assert_eq!(s.deref(), "source");
    }
}
