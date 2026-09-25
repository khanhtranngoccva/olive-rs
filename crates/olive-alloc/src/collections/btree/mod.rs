//! A fallible port of `alloc::collections::BTreeMap`.

mod borrow;
mod construction;
mod entry;
mod fix;
mod insertion;
#[cfg(test)]
mod invariant;
mod iter;
mod map;
mod mem;
mod mutation;
mod navigate;
mod node;
mod query;
mod remove;
mod scratch;
mod search;
mod set_val;
mod traits;

pub use iter::{IntoIter, Iter, IterMut};
pub use map::BTreeMap;
