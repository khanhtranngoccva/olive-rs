//! `Concat`: a fallible string-concatenation builder using the
//! "results-as-templates" pattern.
//!
//! ## Design
//!
//! [`Concat`] is a newtype wrapping `Result<String<A>, TryReserveError>`. It
//! exists to solve two problems simultaneously:
//!
//! 1. **Orphan rule**: Rust forbids implementing foreign traits (`core::ops::Add`,
//!    `AddAssign`) for `Result<T, E>` even when `T` is local. By introducing our
//!    own `Concat` type, we get a local receiver and can implement the standard
//!    operators directly.
//!
//! 2. **Fallibility without panics**: naive `String + &str` would panic on OOM.
//!    `Concat` carries the error internally; the user calls [`Concat::finish`]
//!    to extract the final `Result<String, TryReserveError>` when satisfied.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use olive_alloc::string::{String, Concat};
//!
//! let s = String::try_from_str("hello").unwrap();
//! // Bare String implements Add → produces Concat
//! let c = s + " world";          // Concat (internally Ok)
//! let c = c + "!";               // Concat + &str
//!
//! // Anything with AsRef<str> works as RHS:
//! let other = String::try_from_str("?").unwrap();
//! let c = c + &other;            // &String via AsRef<str>
//!
//! // When done, extract the result:
//! let result: Result<String, TryReserveError> = c.finish();
//! ```
//!
//! Mixing bare `String` and `Concat` in a chain works naturally because
//! `String` implements `Add<Rhs: AsRef<str>>` producing a `Concat`, and
//! `Concat` also implements `Add<Rhs: AsRef<str>>` for continued chaining.

use core::fmt;
use core::ops::{Add, AddAssign};

use olive_core::alloc_errors::TryReserveError;

use crate::alloc::{Allocator, Global};
use crate::string::String;

// ── Concat newtype ───────────────────────────────────────────────────────────

/// A fallible string-concatenation accumulator.
///
/// Wraps `Result<String<A>, TryReserveError>` so that each `+` step either
/// succeeds (extending the inner string) or records the first allocation
/// failure. Once an error is recorded, further `+` operations are no-ops
/// (the error is sticky), avoiding use-after-failure UB.
///
/// Call [`finish`](Self::finish) to consume the `Concat` and obtain the
/// underlying `Result<String<A>, TryReserveError>`.
#[must_use = "Concat is a builder; call finish() to extract the Result"]
pub struct Concat<A: Allocator = Global>(Result<String<A>, TryReserveError>);

impl<A: Allocator> Concat<A> {
    /// Consume this `Concat` and return the accumulated string (or the error
    /// that caused accumulation to halt).
    pub fn finish(self) -> Result<String<A>, TryReserveError> {
        self.0
    }

    /// Whether concatenation has succeeded so far.
    pub fn is_ok(&self) -> bool {
        self.0.is_ok()
    }

    /// Whether an allocation failure has been recorded.
    pub fn is_err(&self) -> bool {
        self.0.is_err()
    }

    /// Borrow the inner string if still in the `Ok` state.
    pub fn as_str(&self) -> Option<&str> {
        self.0.as_ref().ok().map(|s| s.as_str())
    }
}

impl<A: Allocator> fmt::Debug for Concat<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Ok(s) => f.debug_tuple("Concat").field(&s.as_str()).finish(),
            Err(e) => f.debug_tuple("Concat").field(&e).finish(),
        }
    }
}

impl<A: Allocator> From<Result<String<A>, TryReserveError>> for Concat<A> {
    fn from(result: Result<String<A>, TryReserveError>) -> Self {
        Self(result)
    }
}

impl<A: Allocator> From<String<A>> for Concat<A> {
    fn from(s: String<A>) -> Self {
        Self(Ok(s))
    }
}

// ── Add: String<A> LHS → Concat output ───────────────────────────────────────
//
// The first `+` on a bare String produces a Concat, transitioning into the
// results-as-templates world. Any type implementing `AsRef<str>` works as
// the RHS: `&str`, `String<A>`, `&String<A>`, future interned symbols, etc.

impl<A: Allocator, Rhs: AsRef<str>> Add<Rhs> for String<A> {
    type Output = Concat<A>;

    fn add(mut self, rhs: Rhs) -> Self::Output {
        match self.try_push_str(rhs.as_ref()) {
            Ok(()) => Concat(Ok(self)),
            Err(e) => Concat(Err(e)),
        }
    }
}

// ── Add: Concat LHS (continuing the chain) ───────────────────────────────────
//
// Once in Concat-world, every subsequent `+` stays in Concat-world. If the
// inner Result is already Err, the operation is a no-op (sticky error).

impl<A: Allocator, Rhs: AsRef<str>> Add<Rhs> for Concat<A> {
    type Output = Concat<A>;

    fn add(self, rhs: Rhs) -> Self::Output {
        match self.0 {
            Ok(mut s) => match s.try_push_str(rhs.as_ref()) {
                Ok(()) => Concat(Ok(s)),
                Err(e) => Concat(Err(e)),
            },
            Err(e) => Concat(Err(e)),
        }
    }
}

impl<A: Allocator> Add<Concat<A>> for Concat<A> {
    type Output = Concat<A>;

    fn add(self, rhs: Concat<A>) -> Self::Output {
        match self.0 {
            Ok(mut s) => match rhs.0 {
                Ok(rhs_s) => match s.try_push_str(rhs_s.as_str()) {
                    Ok(()) => Concat(Ok(s)),
                    Err(e) => Concat(Err(e)),
                },
                Err(e) => Concat(Err(e)),
            },
            Err(e) => Concat(Err(e)),
        }
    }
}

// ── AddAssign: Concat (in-place mutation) ────────────────────────────────────

impl<A: Allocator, Rhs: AsRef<str>> AddAssign<Rhs> for Concat<A> {
    fn add_assign(&mut self, rhs: Rhs) {
        if let Ok(ref mut s) = self.0 {
            if let Err(e) = s.try_push_str(rhs.as_ref()) {
                self.0 = Err(e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::format;

    use core::alloc::Layout;
    use core::ptr::NonNull;

    use super::*;
    use crate::alloc::AllocError;

    /// An allocator whose every allocation fails. Used to exercise OOM paths.
    #[derive(Default)]
    struct FailAlloc;

    unsafe impl Allocator for FailAlloc {
        fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            Err(AllocError)
        }
        unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
    }

    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    // ── Basic: String + &str → Concat ─────────────────────────────────────────

    #[test]
    fn string_plus_ref_str_basic() {
        let c = mk("hello") + " world";
        assert!(c.is_ok());
        assert_eq!(c.as_str(), Some("hello world"));
        assert_eq!(c.finish().unwrap().as_str(), "hello world");
    }

    #[test]
    fn string_plus_ref_str_empty_rhs() {
        let c = mk("abc") + "";
        assert_eq!(c.finish().unwrap().as_str(), "abc");
    }

    #[test]
    fn string_plus_ref_str_multibyte() {
        let c = mk("héllo") + " wörld ✓";
        assert_eq!(c.finish().unwrap().as_str(), "héllo wörld ✓");
    }

    // ── Basic: String + String (via AsRef<str>) → Concat ──────────────────────

    #[test]
    fn string_plus_string_basic() {
        let c = mk("foo") + mk("-bar");
        assert_eq!(c.finish().unwrap().as_str(), "foo-bar");
    }

    #[test]
    fn string_plus_string_empty_both() {
        let c = mk("") + mk("");
        assert_eq!(c.finish().unwrap().as_str(), "");
    }

    // ── Basic: String + &String (via AsRef<str>) → Concat ─────────────────────

    #[test]
    fn string_plus_ref_string_basic() {
        let rhs = mk("_suffix");
        let c = mk("prefix") + &rhs;
        assert_eq!(c.finish().unwrap().as_str(), "prefix_suffix");
    }

    // ── Chaining: Concat + various RHS ────────────────────────────────────────

    #[test]
    fn concat_chain_mixed_types() {
        let c = mk("one") + "-" + mk("two") + "-" + "three";
        assert_eq!(c.finish().unwrap().as_str(), "one-two-three");
    }

    #[test]
    fn concat_concat_merge() {
        let a = mk("hello ") + "world";
        let b = mk("!") + "🎉";
        let c = a + b;
        assert_eq!(c.finish().unwrap().as_str(), "hello world!🎉");
    }

    #[test]
    fn concat_long_chain() {
        let c = mk("a") + "b" + "c" + "d" + "e" + "f" + "g" + "h";
        assert_eq!(c.finish().unwrap().as_str(), "abcdefgh");
    }

    // ── Owned String via IntoConcatPart ───────────────────────────────────────

    #[test]
    fn concat_owned_string_via_asref() {
        let owned = mk("owned");
        let c = mk("val=") + owned;
        assert_eq!(c.finish().unwrap().as_str(), "val=owned");
    }

    // ── Error propagation ─────────────────────────────────────────────────────

    #[test]
    fn concat_sticky_error_no_panic() {
        let s: String<FailAlloc> = String::new_in(FailAlloc);
        let c = s + "data";
        assert!(c.is_err());
        // Further ops are no-ops, don't panic
        let c2 = c + "more" + "!";
        assert!(c2.is_err());
    }

    #[test]
    fn concat_fail_alloc_first_step() {
        let s: String<FailAlloc> = String::new_in(FailAlloc);
        let c = s + "x";
        assert!(c.is_err());
        assert!(c.finish().is_err());
    }

    #[test]
    fn concat_from_result_err() {
        let err = TryReserveError::new_capacity_overflow();
        let c: Concat = Concat::from(Err::<String, _>(err));
        assert!(c.is_err());
        // Sticky: adding more doesn't change the error
        let c2 = c + "extra";
        assert!(c2.is_err());
    }

    #[test]
    fn concat_from_result_ok() {
        let c: Concat = Concat::from(Ok(mk("start")));
        assert!(c.is_ok());
        let c2 = c + " end";
        assert_eq!(c2.finish().unwrap().as_str(), "start end");
    }

    // ── AddAssign ─────────────────────────────────────────────────────────────

    #[test]
    fn add_assign_ref_str_success() {
        let mut c = mk("hello") + " ";
        c += "world";
        assert_eq!(c.finish().unwrap().as_str(), "hello world");
    }

    #[test]
    fn add_assign_string_success() {
        let mut c = mk("foo") + "-";
        c += mk("bar");
        assert_eq!(c.finish().unwrap().as_str(), "foo-bar");
    }

    #[test]
    fn add_assign_ref_str_short() {
        let mut c = Concat::from(mk("ab"));
        c += "c";
        assert_eq!(c.finish().unwrap().as_str(), "abc");
    }

    #[test]
    fn add_assign_already_err_no_change() {
        let mut c: Concat = Concat::from(Err(TryReserveError::new_capacity_overflow()));
        c += "attempt";
        assert!(c.is_err());
    }

    #[test]
    fn add_assign_fail_alloc_sets_err() {
        let mut c: Concat<FailAlloc> = Concat::from(String::new_in(FailAlloc));
        c += "data";
        assert!(c.is_err());
    }

    // ── Mixed bare → Concat transition ────────────────────────────────────────

    #[test]
    fn mixed_bare_string_then_concat_ops() {
        let c1 = mk("start") + "!";
        let c2 = c1 + " middle" + "end";
        assert_eq!(c2.finish().unwrap().as_str(), "start! middleend");
    }

    // ── Debug formatting ──────────────────────────────────────────────────────

    #[test]
    fn debug_ok_shows_content() {
        let c = mk("hi") + "!";
        let dbg = format!("{:?}", c);
        assert!(dbg.contains("hi!"), "got: {}", dbg);
    }

    #[test]
    fn debug_err_shows_error() {
        let c: Concat = Concat::from(Err(TryReserveError::new_capacity_overflow()));
        let dbg = format!("{:?}", c);
        assert!(
            dbg.contains("CapacityOverflow") || dbg.contains("capacity"),
            "got: {}",
            dbg
        );
    }
}
