//! Procedural macros for the Olive fallible standard library.
//!
//! The user is advised not to use this crate directly.

extern crate proc_macro;

mod try_clone;
mod try_default;

use proc_macro::TokenStream;

/// Derives `TryClone` for a struct or enum.
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

/// Derives `TryDefault` for a struct or enum.
///
/// Every field in the struct must itself implement `TryDefault`; the generated
/// implementation constructs each field fallibly and propagates the first error
/// encountered, dropping any already-constructed prefix on failure.
///
/// For enums, exactly one variant must be marked with `#[try_default]` (a unit
/// variant); the derived `try_default()` builds that variant. The modifier is
/// named `try_default` rather than `default` so it cannot collide with
/// `#[derive(Default)]`'s own `#[default]` variant marker when both derives are
/// present on the same type.
#[proc_macro_derive(TryDefault, attributes(try_default))]
pub fn derive_try_default(input: TokenStream) -> TokenStream {
    try_default::derive_try_default(input)
}

/// Generates `TryDefault` implementations for tuples of arities 1 through `max`
/// (inclusive), clamped to the range 1..=16. Arity 0 (the unit type) is covered
/// separately by the primitive impls in `olive-core`.
///
/// # Example
///
/// ```ignore
/// olive_macros::try_default_tuples!(16);
/// ```
#[proc_macro]
pub fn try_default_tuples(input: TokenStream) -> TokenStream {
    try_default::try_default_tuples(input)
}
