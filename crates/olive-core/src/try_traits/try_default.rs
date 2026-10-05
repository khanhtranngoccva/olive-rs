//! [`TryDefault`]: a fallible analogue of [`core::default::Default`].

use crate::alloc::AllocError;
use crate::alloc_errors::TryReserveError;
use core::fmt;

/// Error returned when a fallible default construction fails.
///
/// Mirrors the shape of [`TryCloneError`](super::try_clone::TryCloneError):
/// a fixed set of variants covering every realistic failure mode for
/// constructing a default value, so that generic code and derive macros can
/// reason about the error uniformly without an associated type.
#[derive(Clone, PartialEq, Eq)]
pub enum TryDefaultError {
    /// A capacity reservation on a collection failed (overflow or OOM) during
    /// default construction.
    Reserve(TryReserveError),
    /// A single heap allocation failed (no reserve phase — e.g. an allocator
    /// that eagerly pools blocks).
    Alloc(AllocError),
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Debug for TryDefaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f.debug_tuple("TryDefaultError::Reserve").field(e).finish(),
            Self::Alloc(e) => f.debug_tuple("TryDefaultError::Alloc").field(e).finish(),
            Self::Other(msg) => f.debug_tuple("TryDefaultError::Other").field(msg).finish(),
        }
    }
}

impl fmt::Display for TryDefaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "default construction failed: {e}"),
            Self::Alloc(_) => write!(f, "default construction failed: memory allocation failed"),
            Self::Other(msg) => write!(f, "default construction failed: {msg}"),
        }
    }
}

impl core::error::Error for TryDefaultError {}

impl From<TryReserveError> for TryDefaultError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<AllocError> for TryDefaultError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

/// A fallible analogue of [`core::default::Default`].
///
/// Unlike [`Default`], whose implementations are required to be infallible (and
/// whose `Vec`-style constructors historically panic on allocation failure),
/// [`TryDefault`] returns a [`Result`] so that construction of an empty or
/// default value can fail gracefully when it must reserve capacity.
///
/// Most types implement this infallibly — an empty value needs no allocation at
/// all — but the error channel exists for the cases where "empty" still means
/// "allocate something" (e.g. a map that eagerly sizes its buckets, or an
/// allocator that pools blocks up front).
///
/// # Scope and limitations
///
/// This trait is intended for **plain-old-data (POD) and lightweight container**
/// types whose default construction is either trivially cheap or involves a
/// small, bounded allocation. It is *not* intended for:
///
/// - **Non-data / resource-holding types.** Their "default" would require opening a
///   file descriptor, binding a socket, or else — operations that are genuinely fallible
///   in ways unrelated to allocation and that belong in their own constructors.
///   You can add default creation behavior using idiomatic Rust approaches, including:
///     - Making the primary constructor accept a data-only configuration struct that
///       implements [`TryDefault`].
///     - You can also use the builder pattern instead. This is the approach that
///       `tokio` employs.
/// - **Types with no defaults.** Types having no meaningful default (e.g. [`core::num::NonZero`])
///   should omit this impl entirely rather than always fail.
///
/// These limitations allow the error mode of [`TryDefault`] to be the fixed
/// [`TryDefaultError`] return type (rather than an associated type). This
/// enables uniform generic composition and future `#[derive(TryDefault)]`
/// support. In practice, default-construction failures are almost exclusively
/// allocation-related, which the three variants cover.
pub trait TryDefault: Sized {
    /// Construct the default value, failing instead of panicking if the
    /// construction requires a failed allocation.
    ///
    /// # Errors
    ///
    /// Returns [`TryDefaultError`] if a capacity reservation or allocation
    /// fails during construction.
    fn try_default() -> Result<Self, TryDefaultError>;
}

// Infallible defaults for primitive and marker types: building them performs no
// allocation whatsoever, so they never fail.
macro_rules! impl_try_default_infallible {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryDefault for $t {
                #[inline]
                fn try_default() -> Result<Self, TryDefaultError> {
                    Ok(Self::default())
                }
            }
        )*
    };
}

impl_try_default_infallible!(u8, u16, u32, u64, u128, usize);
impl_try_default_infallible!(i8, i16, i32, i64, i128, isize);
impl_try_default_infallible!(bool, char, (), f32, f64);

// Tuples: default-construct each field left-to-right via [`TryDefault`],
// short-circuiting on the first failure. An early error drops exactly the
// constructed prefix — never a fully-built value. Arity 0 is covered by the
// unit impl above; arities 1..=16 are generated by the `olive-macros` proc
// macro (a host-side dependency that emits tokens at compile time and adds no
// runtime dependency).
olive_macros::try_default_tuples!(16);

impl<T> TryDefault for Option<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_macros::TryDefault;

    #[test]
    fn primitives_are_infallible() {
        assert_eq!(u32::try_default().unwrap(), 0);
        assert_eq!(isize::try_default().unwrap(), 0);
        assert_eq!(char::try_default().unwrap(), '\0');
        assert!(matches!(f64::try_default().unwrap(), 0.0));
        assert_eq!(Option::<i32>::try_default().unwrap(), None);
    }

    #[test]
    fn error_variants_are_distinct() {
        assert_ne!(
            TryDefaultError::Alloc(AllocError),
            TryDefaultError::Other("x")
        );
        assert_eq!(TryDefaultError::Other("a"), TryDefaultError::Other("a"));
    }

    #[test]
    fn from_alloc_error_works() {
        let err: TryDefaultError = AllocError.into();
        assert!(matches!(err, TryDefaultError::Alloc(_)));
    }

    #[test]
    fn from_reserve_error_works() {
        let reserve = TryReserveError::new_capacity_overflow();
        let err: TryDefaultError = reserve.into();
        assert!(matches!(err, TryDefaultError::Reserve(r) if r.is_capacity_overflow()));
    }

    #[test]
    fn tuples_are_infallible() {
        // Spot-check a few arities across the generated range.
        assert_eq!(<(u8,) as TryDefault>::try_default().unwrap(), (0,));
        assert_eq!(<(u8, i32) as TryDefault>::try_default().unwrap(), (0, 0));
        assert_eq!(
            <(bool, char, f64) as TryDefault>::try_default().unwrap(),
            (false, '\0', 0.0)
        );
        assert_eq!(
            <(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8) as TryDefault>::try_default()
                .unwrap(),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        );
    }

    mod perfect_derive {
        use super::*;

        // These test cases govern the basic behaviors.

        #[derive(TryDefault, Debug, PartialEq)]
        struct NamedFields {
            a: u32,
            b: bool,
        }

        #[derive(TryDefault, Debug, PartialEq)]
        struct Positional(char, i64);

        #[derive(TryDefault, Debug, PartialEq)]
        struct Unit;

        #[derive(TryDefault, Debug, PartialEq)]
        enum EnumWithDefault {
            #[try_default]
            Empty,
            #[expect(unused, reason = "test cases do not use it yet")]
            Full(u32),
        }

        #[test]
        fn derive_struct_named_fields() {
            let v = NamedFields::try_default().unwrap();
            assert_eq!(v.a, 0);
            assert!(!v.b);
        }

        #[test]
        fn derive_struct_positional_and_unit() {
            let p = Positional::try_default().unwrap();
            assert_eq!((p.0, p.1), ('\0', 0));
            assert_eq!(Unit::try_default().unwrap(), Unit);
        }

        #[test]
        fn derive_enum_picks_marked_variant() {
            match EnumWithDefault::try_default().unwrap() {
                EnumWithDefault::Empty => {}
                EnumWithDefault::Full(_) => panic!("expected the #[try_default] variant"),
            }
        }

        /// A fake transformer exposing `Item` through an associated type for
        /// static typing. It is intentionally non-`TryDefault`.
        struct FakeTransformer<Item>(core::marker::PhantomData<Item>);
        impl<Item> Iterator for FakeTransformer<Item> {
            type Item = Item;
            fn next(&mut self) -> Option<Self::Item> {
                None
            }
        }

        // Behavior 1: the bound applies to the shallow field type — here the
        // projection `<I as Iterator>::Item`, so the derive works even though
        // `I` itself is not `TryDefault`.
        #[derive(TryDefault)]
        struct TransformedItem<I: Iterator> {
            item: I::Item,
        }

        #[test]
        fn derive_bounds_shallow_field_type() {
            let t = TransformedItem::<FakeTransformer<u32>>::try_default().unwrap();
            assert_eq!(t.item, 0);
        }

        // Behavior 2: the bound applies only to the marked variant's fields.
        // `NotDefault` is not `TryDefault`, yet the enum derives fine because
        // the unmarked variant holds it.

        /// A type that intentionally does NOT implement `TryDefault`.
        #[derive(Debug)]
        struct NotDefault(#[expect(dead_code, reason = "only used as a payload type")] u8);

        #[derive(TryDefault)]
        enum MarkedOnly {
            #[allow(dead_code)]
            Heavy(NotDefault),
            #[try_default]
            Zero,
        }

        #[test]
        fn derive_bounds_only_marked_variant() {
            match MarkedOnly::try_default().unwrap() {
                MarkedOnly::Zero => {}
                _ => panic!("expected Zero"),
            }
        }

        #[derive(Debug)]
        struct MyStr;
        impl TryDefault for MyStr {
            fn try_default() -> Result<Self, TryDefaultError> {
                Ok(MyStr)
            }
        }

        #[derive(TryDefault, Debug)]
        enum MarkedNamedFields {
            #[expect(dead_code, reason = "test does not invoke this variant")]
            Other(NotDefault),
            #[try_default]
            Named {
                #[expect(dead_code, reason = "test only checks which variant is produced")]
                name: MyStr,
            },
        }

        #[derive(TryDefault, Debug)]
        enum MarkedTupleFields {
            #[expect(dead_code, reason = "test does not invoke this variant")]
            Other(NotDefault),
            #[try_default]
            Pair(i32, bool),
        }

        #[test]
        fn derive_enum_marked_variant_fills_named_fields() {
            match MarkedNamedFields::try_default().unwrap() {
                MarkedNamedFields::Named { .. } => {}
                _ => panic!("expected Named"),
            }
        }

        #[test]
        fn derive_enum_marked_variant_fills_tuple_fields() {
            match MarkedTupleFields::try_default().unwrap() {
                MarkedTupleFields::Pair(a, b) => assert_eq!((a, b), (0, false)),
                _ => panic!("expected Pair"),
            }
        }

        // Behavior 3: bare type parameters and GATs do not contribute to the
        // bounds unless they are themselves the shallow field type. Here the
        // field type is `FakeDefaultBox<I::Item>` (unconditionally
        // `TryDefault`), so neither `I` nor `I::Item` needs `TryDefault`.

        /// A container with an *unconditional* `TryDefault` impl: constructing
        /// it never requires its parameter to be `TryDefault`.
        struct FakeDefaultBox<T>(core::marker::PhantomData<T>);

        impl<T> TryDefault for FakeDefaultBox<T> {
            fn try_default() -> Result<Self, TryDefaultError> {
                Ok(FakeDefaultBox(core::marker::PhantomData))
            }
        }

        #[derive(TryDefault)]
        struct NestedBox<I: Iterator> {
            #[expect(dead_code, reason = "test only checks that construction succeeds")]
            boxed_item: FakeDefaultBox<I::Item>,
        }

        #[test]
        fn derive_ignores_non_shallow_parameters() {
            let n = NestedBox::<FakeTransformer<NotDefault>>::try_default().unwrap();
            let NestedBox { boxed_item: _ } = n;
        }
    }

    #[test]
    fn derive_propagates_field_failure() {
        // A field type whose default always fails must abort construction and
        // drop any already-constructed prefix (the preceding `u32` is POD, so
        // there is nothing observable to leak here — the contract is the error).
        struct Failing;
        impl TryDefault for Failing {
            fn try_default() -> Result<Self, TryDefaultError> {
                Err(TryDefaultError::Other("no canonical value"))
            }
        }
        struct Mixed {
            #[expect(unused, reason = "test code does not use this field")]
            ok: u32,
            #[expect(unused, reason = "test code does not use this field")]
            bad: Failing,
        }
        impl TryDefault for Mixed {
            fn try_default() -> Result<Self, TryDefaultError> {
                Ok(Mixed {
                    ok: u32::try_default()?,
                    bad: Failing::try_default()?,
                })
            }
        }
        assert!(Mixed::try_default().is_err());
    }
}
