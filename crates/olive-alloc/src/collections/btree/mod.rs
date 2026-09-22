//! A fallible port of `alloc::collections::BTreeMap`.

mod borrow;
mod entry;
mod map;
pub(super) mod node;
mod scratch;
mod mem;

pub use map::BTreeMap;
