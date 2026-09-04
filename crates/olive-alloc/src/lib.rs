//! A fully-fallible re-port of Rust's `alloc` crate. Every heap-owning type —
//! `Vec`, `String`, `BTreeMap`/`BTreeSet`, `Rc`/`Arc`, `Cow`, `Box` — is rewritten
//! so that any operation which can fail (above all, allocation) returns a
//! [`Result`] instead of panicking.
//!
//! Module organization mirrors the original `alloc` crate:
//! - [`borrow`] — [`TryToOwned`], the fallible analogue of [`ToOwned`](stock_alloc::borrow::ToOwned).
//! - [`boxed`] — fallible `Box`.
//! - [`vec`] — fallible `Vec`.
//! - [`string`] — fallible `String` / `str` extensions, plus `try_format!`.
//! - [`collections`] — `BTreeMap`, `BTreeSet`, `LinkedList`, `BinaryHeap`.
//! - [`rc`] / [`arc`] — reference-counted pointers with fallible construction.

#![no_std]
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]
// Require every `unsafe fn` / `unsafe impl` to carry a `# Safety` doc section.
#![deny(clippy::missing_safety_doc)]

// Need extern stub for documentation.
extern crate alloc as stock_alloc;

// Modules are added incrementally as the port progresses. Each module mirrors
// the corresponding `alloc` submodule but with fallible operations throughout.
pub mod alloc;
pub mod borrow;
pub mod boxed;
mod raw_vec;
pub mod string;
pub mod vec;

#[cfg(test)]
mod test_helpers;

/// The fallible `ToOwned` analogue and its error type.
pub use borrow::{TryToOwned, TryToOwnedError};
/// The fallible `ToString` analogue, delegating to [`Display`](core::fmt::Display).
pub use string::TryToString;
