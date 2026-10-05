//! `TryClone` macro generators.
//!
//! The macro implementation emits code that uses the absolute
//! `::olive_core::try_traits::try_clone::{TryClone, TryCloneError}` paths so the
//! expansion is correct regardless of where it is invoked - from within or outside
//! the Olive workspace.

use std::num::NonZero;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

use crate::bounds::{build_where_clauses, parse_error_type, parse_trait_path};

/// Absolute path to the trait as seen from generated code in any dependent crate.
const TRAIT_PATH: &str = "::olive_core::try_traits::try_clone::TryClone";
/// Absolute path to the error type as seen from generated code.
const ERROR_PATH: &str = "::olive_core::try_traits::try_clone::TryCloneError";

pub(crate) fn derive_try_clone(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let trait_path = parse_trait_path(TRAIT_PATH);
    let error_path = parse_error_type(ERROR_PATH);

    let predicates = build_where_clauses(&input, &trait_path);
    let where_clause = syn::WhereClause {
        where_token: Default::default(),
        predicates,
    };

    // The where clause is populated above.
    let (impl_generics, ty_generics, _) = input.generics.split_for_impl();

    // Every field must be cloneable via `TryClone`; we express that bound by
    // simply calling `.try_clone()` on each field and letting the compiler's
    // resolution enforce it at the call site.
    let clone_body = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let field_clones = fields.named.iter().map(|field| {
                    let ident = &field.ident;
                    let ty = &field.ty;
                    quote! {
                        #ident: <#ty as #trait_path>::try_clone(&self.#ident)?,
                    }
                });
                quote! {
                    Ok(Self { #(#field_clones)* })
                }
            }
            Fields::Unnamed(fields) => {
                let field_clones = fields.unnamed.iter().enumerate().map(|(i, field)| {
                    let idx = syn::Index::from(i);
                    let ty = &field.ty;
                    quote! {
                        <#ty as #trait_path>::try_clone(&self.#idx)?,
                    }
                });
                quote! {
                    Ok(Self (#(#field_clones)*))
                }
            }
            Fields::Unit => {
                quote! {
                    Ok(Self)
                }
            }
        },
        Data::Enum(data) => {
            let arms = data.variants.iter().map(|variant| {
                let variant_ident = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) => {
                        let field_idents: Vec<_> = fields
                            .named
                            .iter()
                            .map(|f| f.ident.as_ref().unwrap())
                            .collect();
                        let field_patterns: Vec<_> =
                            field_idents.iter().map(|id| quote!(#id)).collect();
                        let field_tys: Vec<_> = fields.named.iter().map(|f| f.ty.clone()).collect();
                        let field_clones: Vec<_> = field_idents
                            .iter()
                            .zip(field_tys.iter())
                            .map(|(id, ty)| {
                                quote! {
                                    #id: <#ty as #trait_path>::try_clone(#id)?,
                                }
                            })
                            .collect();
                        quote! {
                            Self::#variant_ident { #(#field_patterns),* } => {
                                Ok(Self::#variant_ident { #(#field_clones)* })
                            },
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let field_names: Vec<_> = (0..fields.unnamed.len())
                            .map(|i| quote::format_ident!("f{i}"))
                            .collect();
                        let field_tys: Vec<_> =
                            fields.unnamed.iter().map(|f| f.ty.clone()).collect();
                        let field_clones: Vec<_> = field_names
                            .iter()
                            .zip(field_tys.iter())
                            .map(|(fn_, ty)| {
                                quote! {
                                    <#ty as #trait_path>::try_clone(#fn_)?,
                                }
                            })
                            .collect();
                        quote! {
                            Self::#variant_ident (#(#field_names),*) => {
                                Ok(Self::#variant_ident (#(#field_clones)*))
                            },
                        }
                    }
                    Fields::Unit => {
                        quote! {
                            Self::#variant_ident => Ok(Self::#variant_ident),
                        }
                    }
                }
            });
            quote! {
                match self {
                    #(#arms)*
                }
            }
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(name, "#[derive(TryClone)] is not supported on unions")
                .to_compile_error()
                .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics #trait_path for #name #ty_generics #where_clause {
            #[inline]
            fn try_clone(&self) -> ::core::result::Result<Self, #error_path> {
                #clone_body
            }
        }
    };

    TokenStream::from(expanded)
}

pub(crate) fn try_clone_tuples(input: TokenStream) -> TokenStream {
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

        let tuple_pat = if arity == 1 {
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
                let idx = syn::Index::from(i);
                ts.extend(quote!(<#tp as #trait_path>::try_clone(&self.#idx)?));
            }
            ts
        };

        let body = if arity == 1 {
            quote!(((#fields_joined ),))
        } else {
            quote!((#fields_joined))
        };

        output.push(quote! {
            impl<#(#bounds),*> #trait_path for #tuple_pat {
                #[inline]
                fn try_clone(&self) -> ::core::result::Result<Self, #error_path> {
                    Ok(#body)
                }
            }
        });
    }

    TokenStream::from(quote!(#(#output)*))
}
