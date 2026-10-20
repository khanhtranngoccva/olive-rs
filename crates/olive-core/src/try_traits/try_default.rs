//! [`TryDefault`]: a fallible analogue of [`core::default::Default`].

use crate::alloc::AllocError;
use crate::alloc_errors::TryReserveError;
use core::fmt;
use core::ptr;

use super::partial_init::PartialInitGuard;

/// Error returned when a fallible default construction fails.
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
/// some of them pretend to be and historically panic/abort on allocation failure),
/// [`TryDefault`] returns a [`Result`] so that construction of an empty or
/// default value can fail gracefully when it must reserve capacity.
///
/// # `Default` vs `TryDefault`
///
/// - [`Default`] is implemented exactly when building the value is actually infallible.
///   In that case, [`Default`] is implemented and [`TryDefault`] must delegate to
///   [`Default::default`].
/// - Where building the value can fail (for example, due to allocation), only [`TryDefault`]
///   is implemented and std's aborting `Default` is not mirrored.
///
/// Many types implement this infallibly — an empty value needs no allocation at
/// all — but the error channel exists for the cases where "empty" still means
/// "allocate something" (e.g. a slot map that eagerly reserves its first slot,
/// or an allocator that pools blocks up front).
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
/// enables uniform generic composition and `#[derive(TryDefault)]`
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

// Arrays: default-construct each element left-to-right via [`TryDefault`],
// short-circuiting on the first failure. An early error drops exactly the
// constructed prefix.
impl<T: TryDefault, const N: usize> TryDefault for [T; N] {
    fn try_default() -> Result<Self, TryDefaultError> {
        let mut uninit = core::mem::MaybeUninit::<[T; N]>::uninit();
        // SAFETY: `uninit` is aligned for `[T; N]`, hence for `T`, and the
        // backing block holds room for exactly `N` elements.
        let dst = uninit.as_mut_ptr().cast::<T>();
        // Guard rolls back any partially-written prefix on early exit.
        let mut guard = PartialInitGuard::new(dst);
        for i in 0..N {
            let value = T::try_default()?;
            // SAFETY: `i < N`; the slot at `dst + i` is inside the array and
            // has not yet been initialized.
            unsafe {
                ptr::write(dst.add(i), value);
            }
            guard.advance();
        }
        // Every element is now initialized; forget the guard so it does not
        // drop the freshly-written data, then hand the value over.
        core::mem::forget(guard);
        // SAFETY: all `N` elements have been initialized by the loop above.
        Ok(unsafe { uninit.assume_init() })
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

    mod arrays {
        use super::*;
        use core::cell::Cell;
        use std::panic::{AssertUnwindSafe, catch_unwind};
        use std::thread_local;

        // `try_default` has no local data to store test states,
        // so all state lives in thread-local variables; each test
        // runs on its own thread and arms it explicitly.
        thread_local! {
            /// Remaining successful default constructions before the element
            /// types below start failing or panicking.
            static DEFAULT_BUDGET: Cell<u32> = const { Cell::new(0) };
            /// Successful `FallibleDefault` constructions so far.
            static FALLIBLE_CONSTRUCTED: Cell<u32> = const { Cell::new(0) };
            /// `FallibleDefault` drops observed so far.
            static FALLIBLE_DROPPED: Cell<u32> = const { Cell::new(0) };
            /// Successful `PanickyDefault` constructions so far.
            static PANICKY_CONSTRUCTED: Cell<u32> = const { Cell::new(0) };
            /// `PanickyDefault` drops observed so far.
            static PANICKY_DROPPED: Cell<u32> = const { Cell::new(0) };
        }

        /// Arms a budget of `remaining` successful constructions; once it is
        /// exhausted, `CountingDefault` fails and `PanickyDefault` panics.
        fn arm_budget(remaining: u32) {
            DEFAULT_BUDGET.with(|budget| budget.set(remaining));
        }

        /// Consumes one unit of the shared budget, returning `false` when it
        /// is exhausted.
        fn consume_budget() -> bool {
            DEFAULT_BUDGET.with(|budget| {
                let remaining = budget.get();
                if remaining == 0 {
                    false
                } else {
                    budget.set(remaining - 1);
                    true
                }
            })
        }

        /// An element type whose default fails (via the `Result` channel) once
        /// the shared budget is exhausted, counting constructions and drops so
        /// the test can observe exactly which prefix was rolled back.
        struct FallibleDefault;

        impl TryDefault for FallibleDefault {
            fn try_default() -> Result<Self, TryDefaultError> {
                if !consume_budget() {
                    return Err(TryDefaultError::Other("counting default budget exhausted"));
                }
                FALLIBLE_CONSTRUCTED.with(|count| count.set(count.get() + 1));
                Ok(Self)
            }
        }

        impl Drop for FallibleDefault {
            fn drop(&mut self) {
                FALLIBLE_DROPPED.with(|count| count.set(count.get() + 1));
            }
        }

        /// An element type whose default panics once the shared budget is
        /// exhausted, with the same construction/drop observability.
        struct PanickyDefault;

        impl TryDefault for PanickyDefault {
            fn try_default() -> Result<Self, TryDefaultError> {
                if !consume_budget() {
                    panic!("armed PanickyDefault::try_default panicked");
                }
                PANICKY_CONSTRUCTED.with(|count| count.set(count.get() + 1));
                Ok(Self)
            }
        }

        impl Drop for PanickyDefault {
            fn drop(&mut self) {
                PANICKY_DROPPED.with(|count| count.set(count.get() + 1));
            }
        }

        #[test]
        fn try_default_array_of_primitives() {
            assert_eq!(<[u8; 3]>::try_default().unwrap(), [0, 0, 0]);
            let empty: [i32; 0] = <[i32; 0]>::try_default().unwrap();
            assert_eq!(empty, []);
            // Nested arrays compose: the inner `[u8; 2]` is itself
            // `TryDefault`, so the outer impl can default-construct each slot.
            assert_eq!(<[[u8; 2]; 3]>::try_default().unwrap(), [[0, 0]; 3]);
        }

        #[test]
        fn try_default_array_of_zerosized() {
            assert_eq!(<[(); 4]>::try_default().unwrap(), [(), (), (), ()]);
        }

        #[test]
        fn try_default_array_counts_and_drops_all() {
            FALLIBLE_CONSTRUCTED.with(|count| count.set(0));
            FALLIBLE_DROPPED.with(|count| count.set(0));
            arm_budget(u32::MAX);
            let arr = <[FallibleDefault; 4]>::try_default().unwrap();
            assert_eq!(FALLIBLE_CONSTRUCTED.with(Cell::get), 4);
            assert_eq!(FALLIBLE_DROPPED.with(Cell::get), 0);
            drop(arr);
            assert_eq!(FALLIBLE_DROPPED.with(Cell::get), 4);
        }

        #[test]
        fn try_default_array_drops_prefix_on_failure() {
            FALLIBLE_CONSTRUCTED.with(|count| count.set(0));
            FALLIBLE_DROPPED.with(|count| count.set(0));
            // A budget of 2: the first two element defaults succeed, the third
            // fails.
            arm_budget(2);
            let res = <[FallibleDefault; 8]>::try_default();
            assert!(res.is_err());
            // Exactly the constructed prefix was dropped, each exactly once —
            // no element leaked and nothing was dropped twice.
            assert_eq!(FALLIBLE_CONSTRUCTED.with(Cell::get), 2);
            assert_eq!(FALLIBLE_DROPPED.with(Cell::get), 2);
        }

        #[test]
        fn try_default_array_drops_prefix_on_empty_budget() {
            FALLIBLE_CONSTRUCTED.with(|count| count.set(0));
            FALLIBLE_DROPPED.with(|count| count.set(0));
            // No budget at all: the very first element fails, so nothing was
            // ever constructed and the guard must drop nothing.
            arm_budget(0);
            let res = <[FallibleDefault; 8]>::try_default();
            assert!(res.is_err());
            assert_eq!(FALLIBLE_CONSTRUCTED.with(Cell::get), 0);
            assert_eq!(FALLIBLE_DROPPED.with(Cell::get), 0);
        }

        #[test]
        fn try_default_array_panics_drops_prefix_once() {
            PANICKY_CONSTRUCTED.with(|count| count.set(0));
            PANICKY_DROPPED.with(|count| count.set(0));
            // A budget of 2: the first two element defaults succeed, the third
            // panics.
            arm_budget(2);
            let res = catch_unwind(AssertUnwindSafe(|| {
                let _ = <[PanickyDefault; 8]>::try_default();
            }));
            assert!(res.is_err(), "the exhausted element default must panic");
            // Unwinding rolled back exactly the constructed prefix, each
            // element dropped exactly once.
            assert_eq!(PANICKY_CONSTRUCTED.with(Cell::get), 2);
            assert_eq!(PANICKY_DROPPED.with(Cell::get), 2);
        }
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
