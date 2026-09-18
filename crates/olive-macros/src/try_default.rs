//! `TryDefault` macro generators.
//!
//! Both entry points emit code that resolves against the absolute
//! `::olive_core::try_traits::try_default::{TryDefault, TryDefaultError}` path so
//! the expansion is correct regardless of where it is invoked from within the
//! Olive workspace. Every member depends on `olive-core`; additionally
//! `olive-core` declares `extern crate self as olive_core;` at its crate root so
//! the same absolute path also resolves for the impls generated *inside*
//! `olive-core` itself (where the crate would otherwise only be reachable as
//! `crate`).

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

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
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Every field must be constructible via `TryDefault`; we express that bound
    // by simply calling `<FieldType as TryDefault>::try_default()` on each field
    // and letting the compiler's resolution enforce it at the call site. We do
    // NOT add an explicit supertrait — `TryDefault` stands alone, mirroring how
    // `derive(TryClone)` composes over `TryClone` without a `Clone` supertrait.
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
        Data::Enum(data) => {
            // Exactly one variant may carry `#[try_default]`.
            let mut marked: Option<&syn::Variant> = None;
            for variant in &data.variants {
                if variant
                    .attrs
                    .iter()
                    .any(|attr| attr.path().is_ident(TRY_DEFAULT_ATTR))
                    && marked.replace(variant).is_some()
                {
                    return try_default_attr_error(name);
                }
            }
            let Some(variant) = marked else {
                return try_default_attr_error(name);
            };

            // The marked variant must be unit-like: its default value is
            // unambiguous only when there are no fields to fill in.
            match &variant.fields {
                Fields::Unit => {}
                _ => {
                    return syn::Error::new_spanned(
                        &variant.ident,
                        format!(
                            "the #[{attr}] variant of #[derive(TryDefault)] must be a unit variant",
                            attr = TRY_DEFAULT_ATTR,
                        ),
                    )
                    .to_compile_error()
                    .into();
                }
            }

            let variant_ident = &variant.ident;
            quote! {
                Ok(Self::#variant_ident)
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

    let trait_path = syn::parse_str::<syn::Path>(TRAIT_PATH).expect("static path parses");
    let error_path = syn::parse_str::<syn::Type>(ERROR_PATH).expect("static path parses");

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
    let max: usize = syn::parse::<syn::LitInt>(input)
        .ok()
        .and_then(|lit| lit.base10_parse().ok())
        .unwrap_or(12);
    let max = max.clamp(1, 16);

    let trait_path = syn::parse_str::<syn::Path>(TRAIT_PATH).expect("static path parses");
    let error_path = syn::parse_str::<syn::Type>(ERROR_PATH).expect("static path parses");

    let mut output = Vec::new();

    for arity in 1..=max {
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
                ts.extend(quote!(#tp::try_default()?));
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
