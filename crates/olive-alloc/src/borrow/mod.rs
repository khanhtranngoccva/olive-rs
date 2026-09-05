//! Borrow semantics and owned-value construction.
//!
//! This module hosts [`TryToOwned`], the fallible analogue of
//! [`ToOwned`](stock_alloc::borrow::ToOwned). Like std, the trait is defined here (in the
//! `alloc` layer) and applied to essentially every type through a single blanket
//! impl over [`TryClone`](olive_core::try_traits::TryClone).

pub mod cow;
pub mod try_to_owned;
pub use cow::Cow;
pub use try_to_owned::{TryToOwned, TryToOwnedError};
