//! `TryClone` macro generators.
//!
//! Both entry points emit code that resolves against the absolute
//! `::olive_core::try_traits::try_clone::{TryClone, TryCloneError}` path so the
//! expansion is correct regardless of where it is invoked from within the Olive
//! workspace. Every member depends on `olive-core`; additionally `olive-core`
//! declares `extern crate self as olive_core;` at its crate root so the same
//! absolute path also resolves for the impls generated *inside* `olive-core`
//! itself (where the crate would otherwise only be reachable as `crate`).

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Absolute path to the trait as seen from generated code in any dependent crate.
const TRAIT_PATH: &str = "::olive_core::try_traits::try_clone::TryClone";
/// Absolute path to the error type as seen from generated code.
const ERROR_PATH: &str = "::olive_core::try_traits::try_clone::TryCloneError";

/// Add a `TryClone` bound to every type parameter, mirroring how
/// `#[derive(Clone)]` adds a `Clone` bound to each type parameter of the impl.
fn add_try_clone_bounds(mut generics: syn::Generics, trait_path: &syn::Path) -> syn::Generics {
    for param in generics.type_params_mut() {
        let bound = syn::TypeParamBound::Trait(syn::TraitBound {
            paren_token: None,
            modifier: syn::TraitBoundModifier::None,
            lifetimes: None,
            path: trait_path.clone(),
        });
        param.bounds.push(bound);
    }
    generics
}

pub(crate) fn derive_try_clone(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let trait_path = syn::parse_str::<syn::Path>(TRAIT_PATH).expect("static path parses");
    let error_path = syn::parse_str::<syn::Type>(ERROR_PATH).expect("static path parses");

    // Mirror `derive(Clone)`: constrain every type parameter with `TryClone`.
    let bounded_generics = add_try_clone_bounds(input.generics.clone(), &trait_path);
    let (impl_generics, ty_generics, where_clause) = bounded_generics.split_for_impl();

    // Every field must be cloneable via `TryClone`; we express that bound by
    // simply calling `.try_clone()` on each field and letting the compiler's
    // resolution enforce it at the call site. We do NOT add an explicit
    // `T: TryClone` supertrait because Olive's `TryClone` has no `Clone`
    // supertrait — adding one would contradict the project's "retire Clone"
    // directive.
    let clone_body = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let field_clones = fields.named.iter().map(|field| {
                    let ident = &field.ident;
                    quote! {
                        #ident: self.#ident.try_clone()?,
                    }
                });
                quote! {
                    Ok(Self { #(#field_clones)* })
                }
            }
            Fields::Unnamed(fields) => {
                let indices = 0..fields.unnamed.len();
                let field_clones = indices.map(|i| {
                    let idx = syn::Index::from(i);
                    quote! {
                        self.#idx.try_clone()?,
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
                    Fields::Named(fields) if !fields.named.is_empty() => {
                        let field_idents: Vec<_> = fields
                            .named
                            .iter()
                            .map(|f| f.ident.as_ref().unwrap())
                            .collect();
                        let field_patterns: Vec<_> =
                            field_idents.iter().map(|id| quote!(#id)).collect();
                        let field_clones = field_idents.iter().map(|id| {
                            quote! {
                                #id: #id.try_clone()?,
                            }
                        });
                        quote! {
                            Self::#variant_ident { #(#field_patterns),* } => {
                                Ok(Self::#variant_ident { #(#field_clones)* })
                            },
                        }
                    }
                    Fields::Named(_) | Fields::Unit => {
                        quote! {
                            Self::#variant_ident => Ok(Self::#variant_ident),
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let field_names: Vec<_> = (0..fields.unnamed.len())
                            .map(|i| quote::format_ident!("f{i}"))
                            .collect();
                        let field_clones = field_names.iter().map(|fn_| {
                            quote! {
                                #fn_.try_clone()?,
                            }
                        });
                        quote! {
                            Self::#variant_ident (#(#field_names),*) => {
                                Ok(Self::#variant_ident (#(#field_clones)*))
                            },
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

        let tuple_pat = if arity == 1 {
            quote!((#types_joined ,))
        } else {
            quote!((#types_joined))
        };

        let fields_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            for i in 0..arity {
                if i > 0 {
                    ts.extend(quote!(,));
                }
                let idx = syn::Index::from(i);
                ts.extend(quote!(self.#idx.try_clone()?));
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
