//! Thread-safe synchronization primitives, mirroring the standard library's
//! [`sync`](stock_alloc::sync) module.
//!
//! This module currently hosts the fallible [`Arc`] / [`Weak`] pair; other
//! synchronization types will land here incrementally as the port progresses.

mod arc;

pub use self::arc::{Arc, Weak};
