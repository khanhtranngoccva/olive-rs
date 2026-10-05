//! `TryDefault` macro generators.
//!
//! The macro implementation emits code that uses the absolute
//! `::olive_core::try_traits::try_default::{TryDefault, TryDefaultError}` paths so the
//! expansion is correct regardless of where it is invoked - from within or outside
//! the Olive workspace.

use core::num::NonZero;
use proc_macro::TokenStream;
use quote::quote;
use std::collections::HashSet;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

use crate::bounds::{
    build_where_clauses, build_where_clauses_filtered, field_types, parse_error_type,
    parse_trait_path,
};

/// Absolute path to the trait as seen from generated code in any dependent crate.
const TRAIT_PATH: &str = "::olive_core::try_traits::try_default::TryDefault";
/// Absolute path to the error type as seen from generated code.
const ERROR_PATH: &str = "::olive_core::try_traits::try_default::TryDefaultError";

/// Attribute marking the enum variant a `#[derive(TryDefault)]` should build.
///
/// Named `try_default` rather than `default` because `#[default]` already has a
/// meaning on enums (it drives `#[derive(Default)]`) and the two modifiers must
/// not collide when both derives are present on the same type.
const TRY_DEFAULT_ATTR: &str = "try_default";

fn try_default_attr_error(ident: &syn::Ident) -> TokenStream {
    syn::Error::new_spanned(
        ident,
        "#[derive(TryDefault)] on an enum requires exactly one variant marked \
         with #[try_default]",
    )
    .to_compile_error()
    .into()
}

pub(crate) fn derive_try_default(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let trait_path = parse_trait_path(TRAIT_PATH);
    let error_path = parse_error_type(ERROR_PATH);

    // Locate the #[try_default]-marked enum variant once. Exactly one variant
    // may carry the attribute; duplicates are rejected here so the rest of the
    // expansion can rely on `marked` being unique.
    let mut marked_enum_variant: Option<&syn::Variant> = None;
    if let Data::Enum(data) = &input.data {
        for variant in &data.variants {
            if variant
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident(TRY_DEFAULT_ATTR))
                && marked_enum_variant.replace(variant).is_some()
            {
                return try_default_attr_error(name);
            }
        }
    }

    // Only the marked variant's fields need the trait bound; other variants may
    // hold types that do not implement it.
    let marked_enum_variant_tys: HashSet<&syn::Type> = marked_enum_variant
        .map(|v| field_types(&v.fields).into_iter().collect())
        .unwrap_or_default();

    let predicates = if marked_enum_variant.is_some() {
        let marked_set = marked_enum_variant_tys.clone();
        build_where_clauses_filtered(&input, &trait_path, |ty| marked_set.contains(ty))
    } else {
        build_where_clauses(&input, &trait_path)
    };
    let where_clause = syn::WhereClause {
        where_token: Default::default(),
        predicates,
    };

    let (impl_generics, ty_generics, _) = input.generics.split_for_impl();

    // Every field must be constructible via `TryDefault`; we express that bound
    // by simply calling `<FieldType as TryDefault>::try_default()` on each field
    // and letting the compiler's resolution enforce it at the call site.
    let default_body = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let field_defaults = fields.named.iter().map(|field| {
                    let ident = &field.ident;
                    let ty = &field.ty;
                    quote! {
                        #ident: <#ty as ::olive_core::try_traits::try_default::TryDefault>::try_default()?,
                    }
                });
                quote! {
                    Ok(Self { #(#field_defaults)* })
                }
            }
            Fields::Unnamed(fields) => {
                let field_defaults = fields.unnamed.iter().map(|field| {
                    let ty = &field.ty;
                    quote! {
                        <#ty as ::olive_core::try_traits::try_default::TryDefault>::try_default()?,
                    }
                });
                quote! {
                    Ok(Self (#(#field_defaults)*))
                }
            }
            Fields::Unit => {
                quote! {
                    Ok(Self)
                }
            }
        },
        Data::Enum(_) => {
            let Some(variant) = marked_enum_variant else {
                return try_default_attr_error(name);
            };

            // Fill the marked variant's fields via `TryDefault`.
            match &variant.fields {
                Fields::Named(fields) => {
                    let fd = fields.named.iter().map(|f| {
                        let ident = &f.ident;
                        let ty = &f.ty;
                        quote! { #ident: <#ty as #trait_path>::try_default()?, }
                    });
                    let vi = &variant.ident;
                    quote! { Ok(Self::#vi { #(#fd)* }) }
                }
                Fields::Unnamed(fields) => {
                    let fd = fields.unnamed.iter().map(|f| {
                        let ty = &f.ty;
                        quote! { <#ty as #trait_path>::try_default()?, }
                    });
                    let vi = &variant.ident;
                    quote! { Ok(Self::#vi (#(#fd)*)) }
                }
                Fields::Unit => {
                    let vi = &variant.ident;
                    quote! { Ok(Self::#vi) }
                }
            }
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(
                name,
                "#[derive(TryDefault)] is not supported on unions",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics #trait_path for #name #ty_generics #where_clause {
            #[inline]
            fn try_default() -> ::core::result::Result<Self, #error_path> {
                #default_body
            }
        }
    };

    TokenStream::from(expanded)
}

pub(crate) fn try_default_tuples(input: TokenStream) -> TokenStream {
    let lit = match syn::parse::<syn::LitInt>(input) {
        Ok(lit) => lit,
        Err(e) => return e.to_compile_error().into(),
    };
    let max: NonZero<usize> = match lit.base10_parse() {
        Ok(n) => n,
        Err(e) => {
            return syn::Error::new_spanned(&lit, format_args!("invalid tuple arity: {e}"))
                .to_compile_error()
                .into();
        }
    };

    let trait_path = parse_trait_path(TRAIT_PATH);
    let error_path = parse_error_type(ERROR_PATH);

    let mut output = Vec::new();

    for arity in 1..=max.get() {
        let type_params: Vec<_> = (0..arity).map(|i| quote::format_ident!("T{i}")).collect();
        let bounds: Vec<_> = type_params
            .iter()
            .map(|t| quote!(#t: #trait_path))
            .collect();

        let types_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            for (i, tp) in type_params.iter().enumerate() {
                if i > 0 {
                    ts.extend(quote!(,));
                }
                ts.extend(quote!(#tp));
            }
            ts
        };

        let tuple_ty = if arity == 1 {
            quote!((#types_joined ,))
        } else {
            quote!((#types_joined))
        };

        let fields_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            for (i, tp) in type_params.iter().enumerate() {
                if i > 0 {
                    ts.extend(quote!(,));
                }
                ts.extend(quote!(<#tp as #trait_path>::try_default()?));
            }
            ts
        };

        let body = if arity == 1 {
            quote!(((#fields_joined ),))
        } else {
            quote!((#fields_joined))
        };

        output.push(quote! {
            impl<#(#bounds),*> #trait_path for #tuple_ty {
                #[inline]
                fn try_default() -> ::core::result::Result<Self, #error_path> {
                    Ok(#body)
                }
            }
        });
    }

    TokenStream::from(quote!(#(#output)*))
}
