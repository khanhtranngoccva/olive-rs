//! The Olive prelude.
//!
//! Re-exports the foundational fallible traits and allocation errors so that a
//! single `use olive_core::prelude::*;` brings in everything most fallible code
//! needs.

pub use crate::recovery::{ResumableSource, Stall};
pub use crate::try_traits::try_clone::{TryClone, TryCloneToUninit};
pub use crate::try_traits::try_collect::{TryCollect, TryCollectInto};
pub use crate::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
pub use crate::try_traits::try_from_iterator::TryFromIterator;
pub use core::prelude::*;
