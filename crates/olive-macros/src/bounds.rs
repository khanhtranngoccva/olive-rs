//! Shared where-clause construction for Olive derive macros.

use quote::ToTokens;
use std::collections::HashSet;
use syn::punctuated::Punctuated;
use syn::{Data, DeriveInput, Fields};

/// Collect the types of every field of a struct or variant.
pub(crate) fn field_types(fields: &Fields) -> Vec<&syn::Type> {
    match fields {
        Fields::Named(f) => f.named.iter().map(|x| &x.ty).collect(),
        Fields::Unnamed(f) => f.unnamed.iter().map(|x| &x.ty).collect(),
        Fields::Unit => Vec::new(),
    }
}

/// Build a single `Trait(...)` [`syn::TypeParamBound`] from an absolute path.
pub(crate) fn trait_bound(trait_path: &syn::Path) -> syn::TypeParamBound {
    syn::TypeParamBound::Trait(syn::TraitBound {
        paren_token: None,
        modifier: syn::TraitBoundModifier::None,
        lifetimes: None,
        path: trait_path.clone(),
    })
}

/// Build the where-clause predicates for a generated impl.
///
/// Emits:
/// 1. The user's own where-clause predicates, copied verbatim.
/// 2. `FieldType: Trait` for every distinct field type as written — including
///    bare type parameters (`T`) and GAT projections (`I::Item`). This is the
///    "perfect derive" strategy: bound exactly what is needed, and let the
///    compiler do the remainder of the checking for the referred types.
///
/// Identical predicates are deduplicated by their token-stream text.
pub(crate) fn build_where_clauses(
    input: &DeriveInput,
    trait_path: &syn::Path,
) -> Punctuated<syn::WherePredicate, syn::token::Comma> {
    build_where_clauses_filtered(input, trait_path, |_| true)
}

/// Same as [`build_where_clauses`] but only field types for which `keep`
/// returns `true` get a bound. 
/// 
/// Used by `derive(TryDefault)` on enums so that
/// only the `#[try_default]`-marked variant's fields are constrained — other
/// variants may hold types that do not implement the trait.
pub(crate) fn build_where_clauses_filtered<F>(
    input: &DeriveInput,
    trait_path: &syn::Path,
    mut keep: F,
) -> Punctuated<syn::WherePredicate, syn::token::Comma>
where
    F: FnMut(&syn::Type) -> bool,
{
    let mut predicates: Punctuated<syn::WherePredicate, syn::token::Comma> = Punctuated::new();

    // Preserve the user's existing where-clause predicates verbatim.
    if let Some(user_clause) = &input.generics.where_clause {
        for pred in &user_clause.predicates {
            predicates.push(pred.clone());
        }
    }

    // Field type bounds: every distinct field type (as written, gated by
    // `keep`) gets `FieldType: Trait`.
    let mut collect = |ty: &syn::Type| {
        if !keep(ty) {
            return;
        }
        let mut bounds: Punctuated<syn::TypeParamBound, syn::token::Plus> = Punctuated::new();
        bounds.push(trait_bound(trait_path));
        predicates.push(syn::WherePredicate::Type(syn::PredicateType {
            lifetimes: None,
            bounded_ty: ty.clone(),
            colon_token: Default::default(),
            bounds,
        }));
    };

    match &input.data {
        Data::Struct(d) => {
            for ty in field_types(&d.fields) {
                collect(ty);
            }
        }
        Data::Enum(d) => {
            for v in &d.variants {
                for ty in field_types(&v.fields) {
                    collect(ty);
                }
            }
        }
        // No-op - we simply wait until the outer implementation emits a compiler error.
        Data::Union(_) => {}
    }

    // Deduplicate identical predicates (compared via their token stream text).
    let mut seen: HashSet<String> = HashSet::new();
    let mut deduped: Punctuated<syn::WherePredicate, syn::token::Comma> = Punctuated::new();
    for pred in predicates {
        let key = pred.to_token_stream().to_string();
        if !seen.insert(key) {
            continue;
        }
        deduped.push(pred);
    }

    deduped
}

/// Convenience: parse a static string into a `syn::Path` (panics at macro
/// expansion time if the constant is malformed, which is a programmer error).
pub(crate) fn parse_trait_path(path_str: &str) -> syn::Path {
    syn::parse_str::<syn::Path>(path_str).expect("static trait path parses")
}

/// Convenience: parse a static string into a `syn::Type`.
pub(crate) fn parse_error_type(path_str: &str) -> syn::Type {
    syn::parse_str::<syn::Type>(path_str).expect("static error path parses")
}
