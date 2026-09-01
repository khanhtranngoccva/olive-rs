//! Procedural macros for the Olive fallible standard library.
//!
//! These are host-side `proc-macro` helpers that generate boilerplate
//! [`TryClone`](::olive_core::try_traits::try_clone::TryClone) implementations
//! so that downstream crates (and Olive itself) do not have to hand-write the
//! repetitive per-field / per-tuple glue. The generated code is emitted into the
//! *calling* crate, where it resolves against that crate's own visibility — so
//! the trait path used in the expansion is the absolute `::olive_core::…` path,
//! which every Olive member depends on directly.

extern crate proc_macro;

mod try_clone;

use proc_macro::TokenStream;

/// Derives [`TryClone`](::olive_core::try_traits::try_clone::TryClone) for a
/// struct or enum.
///
/// Every field in every variant must itself implement `TryClone`. The generated
/// implementation clones each field fallibly and propagates the first error
/// encountered, dropping any already-cloned prefix on failure.
#[proc_macro_derive(TryClone)]
pub fn derive_try_clone(input: TokenStream) -> TokenStream {
    try_clone::derive_try_clone(input)
}

/// Generates `TryClone` implementations for tuples of arities 1 through `max`
/// (inclusive), clamped to the range 1..=16. Arity 0 (the unit type) is covered
/// separately by the primitive impls in `olive-core`.
///
/// # Example
///
/// ```ignore
/// olive_macros::try_clone_tuples!(12);
/// ```
#[proc_macro]
pub fn try_clone_tuples(input: TokenStream) -> TokenStream {
    try_clone::try_clone_tuples(input)
}
