//! Foundational fallible traits.
//!
//! These are the drop-in, `Result`-returning analogues of the standard library's
//! infallible-by-assumption traits (`Clone`, `FromIterator`, `Extend`). They let
//! callers write OOM-resilient code without risking crashing the program.
//!
//! Each trait lives in its own module; this module is a convenience umbrella
//! that re-exports all of them under one path.
//!
//! # Naming convention
//!
//! - Trait methods inherit the name they override, prefixed `try_`
//!   (`try_clone`, `try_from_iter`, `try_extend`, …).
//! - Free functions and inherent methods that are fallible carry a `try_` prefix
//!   (e.g. `try_collect`, `try_to_string`).
//!
//! # Discipline
//!
//! The infallible variants (`Clone::clone`, `Extend::extend`, …) are **not**
//! implemented by Olive types on top of these traits; doing so would silently
//! reintroduce panicking allocation paths.
//!
//! # Atomicity
//! `TryExtend` and `TryExtendFromSlice` are declared *resumable* and *non-atomic*:
//! their errors carry the unconsumed remainder so a failed operation can be retried
//! without losing data.
//! `TryClone` and friends are atomic single-shot operations.

pub mod try_clone;
pub mod try_collect;
pub mod try_default;
pub mod try_extend;
pub mod try_from_iterator;

pub use try_clone::{TryClone, TryCloneError, TryCloneToUninit};
pub use try_collect::{TryCollect, TryCollectInto};
pub use try_default::{TryDefault, TryDefaultError};
pub use try_extend::{TryExtend, TryExtendFromSlice};
pub use try_from_iterator::TryFromIterator;
