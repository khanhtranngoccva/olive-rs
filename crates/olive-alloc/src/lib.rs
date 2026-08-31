//! # olive-alloc
//!
//! A fully-fallible re-port of Rust's `alloc` crate. Every heap-owning type —
//! `Vec`, `String`, `BTreeMap`/`BTreeSet`, `Rc`/`Arc`, `Cow`, `Box` — is rewritten
//! so that any operation which can fail (above all, allocation) returns a
//! [`Result`] instead of panicking.
//!
//! This crate depends only on [`olive_core`]; it deliberately does **not** depend
//! on the standard library's `alloc` crate, because we control our own code here.
//!
//! Module organization mirrors the original `alloc` crate:
//! - [`boxed`] — fallible `Box`.
//! - [`vec`] — fallible `Vec`.
//! - [`string`] — fallible `String` / `str` extensions, plus `try_format!`.
//! - [`borrow`] — borrow semantics (`Cow`, `ToOwned` analogues).
//! - [`collections`] — `BTreeMap`, `BTreeSet`, `LinkedList`, `BinaryHeap`.
//! - [`rc`] / [`arc`] — reference-counted pointers with fallible construction.

#![no_std]
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]
// FIXME: add deny lint for unsafe functions and invocations without SAFETY header

// Modules are added incrementally as the port progresses. Each module mirrors
// the corresponding `alloc` submodule but with fallible operations throughout.

pub mod alloc;
pub mod boxed;
pub mod vec;
mod raw_vec;
