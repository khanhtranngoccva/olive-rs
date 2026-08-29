//! # `olive_core`
//!
//! The foundation of the **Olive** stack: a fully-fallible re-port of Rust's
//! `core`. Every operation that can fail — above all, any heap allocation — is
//! expressed as returning a [`Result`] rather than panicking, so out-of-memory
//! conditions become recoverable instead of fatal.
//!
//! This crate has **no dependencies** and never touches `std`. It mirrors the
//! entire stable surface of `core` (glob-re-exported below) and layers on the
//! pieces everything else in the stack builds on:
//!
//! * [`allocator`] — the ported allocator API: [`Layout`], [`AllocError`], the
//!   [`Allocator`] trait, the default [`Global`] allocator, and the free-standing
//!   raw-pointer functions. The canonical seam every Olive collection allocates
//!   through.
//! * [`alloc_errors`] — [`TryReserveError`](alloc_errors::TryReserveError), the
//!   collection-level capacity-reservation error (and a re-export of
//!   [`AllocError`](allocator::AllocError)).
//! * [`try_traits`] — the foundational fallible traits, each in its own module:
//!   [`TryClone`](try_traits::try_clone), [`TryToOwned`](try_traits::try_to_owned),
//!   [`TryFromIterator`](try_traits::try_from_iterator), [`TryCollect`](try_traits::try_collect),
//!   [`TryExtend`](try_traits::try_extend) / [`TryExtendFromSlice`](try_traits::try_extend).
//! * [`recovery`] — [`Resume`] / [`Stall`] for resuming a failed fallible
//!   iteration without losing data.
//!
//! # Mirroring `core`
//!
//! Everything in the standard `core` crate is glob-re-exported at the crate root
//! via [`pub use core::*;`](#re-exports). That means `olive_core::option`,
//! `olive_core::slice`, `olive_core::fmt`, … all resolve exactly as their
//! `core::` counterparts do. Olive-specific additions are layered on top without
//! hiding any of them.
//!
//! # Naming convention
//!
//! A method that can fail is prefixed `try_` (e.g. `try_reserve`,
//! `try_extend`). Infallible operations keep their plain names. Trait methods
//! inherit the name they override; free functions and inherent methods that are
//! fallible carry the `try_` prefix.
//!
//! # ``no_std`` compatibility
//!
//! This crate is `#![no_std]`. It does not require an allocator to be present;
//! it merely defines the error types and traits that fallible allocation uses.
//! Anything needing the standard runtime (notably `catch_unwind` /
//! `resume_unwind`) lives in `olive_std`, which sits above this crate.

#![no_std]
// Clippy configuration: pedantic as a baseline, with arithmetic side effects
// denied (library code must use checked/wrapping/saturating ops explicitly).
// Scoped to non-test builds so test helpers aren't held to the same standard.
#![cfg_attr(
    not(test),
    warn(clippy::pedantic),
    deny(clippy::arithmetic_side_effects)
)]

/// Allocation errors: [`AllocError`] and [`TryReserveError`].
pub mod alloc_errors;
/// The ported allocator API: [`Layout`], [`AllocError`], [`Allocator`],
/// [`Global`], and the free-standing raw-pointer functions.
///
/// [`Layout`]:
pub mod alloc;
/// Iterator recovery primitives: [`Resume`] and the [`Stall`] trait.
pub mod recovery;
/// Foundational fallible traits.
pub mod try_traits;

#[doc(hidden)]
pub mod prelude;

// ── Mirror the entire stable `core` surface ───────────────────────────────────
// Glob-re-export every item in `core` so `olive_core` is a drop-in superset of
// `core`. Olive's own modules (`allocator`, `alloc_errors`, `recovery`,
// `try_traits`) are declared above and take precedence over any name collision
// because explicit items shadow glob imports. Notably, our own `Layout` /
// `AllocError` / `Allocator` here shadow the identically-named items that the
// `core::alloc` glob would otherwise bring into scope — intentional, so the
// canonical Olive allocator API wins.
pub use core::*;

pub use try_traits::{
    TryClone, TryCloneError, TryCollect, TryCollectInto, TryExtend, TryExtendFromSlice,
    TryFromIterator, TryToOwned, TryToOwnedError,
};
