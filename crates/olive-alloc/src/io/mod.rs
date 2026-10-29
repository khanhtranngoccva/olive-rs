//! The `io` module port for `olive-alloc`.
//!
//! This includes, but is not limited to, I/O traits like `Read`
//! and the [`io::Error`](Error) core type.

pub mod error;

pub use self::error::IoErrorAllocExt;
pub use olive_core::io::error::{Error, ErrorKind};
