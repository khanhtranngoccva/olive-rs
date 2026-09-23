//! A fallible port of `alloc::collections::BTreeMap`.

mod borrow;
mod entry;
mod fix;
mod map;
mod mem;
mod node;
mod remove;
mod scratch;
mod search;
mod set_val;
mod navigate;

pub use map::BTreeMap;
