//! A fallible port of `alloc::collections::BTreeMap`.

mod borrow;
mod construction;
mod entry;
mod fix;
mod insertion;
mod iter;
mod map;
mod mem;
mod mutation;
mod node;
mod query;
mod remove;
mod scratch;
mod search;
mod set_val;
mod traits;
mod navigate;

pub use map::BTreeMap;
pub use iter::{IntoIter, Iter, IterMut};
