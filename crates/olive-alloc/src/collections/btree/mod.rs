//! A fallible port of `alloc::collections::BTreeMap`.

mod borrow;
mod entry;
mod map;
mod mem;
pub(super) mod node;
mod scratch;
mod search;
mod set_val;

pub use map::BTreeMap;
