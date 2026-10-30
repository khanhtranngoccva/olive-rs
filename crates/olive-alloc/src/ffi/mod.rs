//! C-compatible, FFI-safe string types.
//!
//! Mirrors the `alloc::ffi` module of the standard library with fallible
//! allocation throughout: every constructor that touches the heap returns a
//! [`Result`](core::result::Result) instead of panicking on OOM.

mod c_str;

pub use c_str::{
    CString, CStringIntoStringError, CStringNulError, TryCStringFromVecWithNulError,
    TryCStringNewError,
};
