//! The Olive prelude.
//!
//! Re-exports the foundational fallible traits and allocation errors so that a
//! single `use olive_core::prelude::*;` brings in everything most fallible code
//! needs.
//!
//! The panic machinery (`Location`, `UnwindSafe`, `AssertUnwindSafe`,
//! `PanicInfo`) is not re-listed here: it comes from `core` via the crate-root
//! glob, and the `panic!` macro is always in scope through the language
//! prelude. Callers who want those under an explicit import can use
//! `use core::panic::*;`.

pub use crate::recovery::{ResumableSource, Resume, Stall};
pub use crate::try_traits::try_clone::{TryClone, TryCloneError};
pub use crate::try_traits::try_collect::{TryCollect, TryCollectInto};
pub use crate::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
pub use crate::try_traits::try_from_iterator::TryFromIterator;
pub use crate::try_traits::try_to_owned::{TryToOwned, TryToOwnedError};
pub use core::prelude::*;
