//! [`Cow`]: a fallible clone-on-write smart pointer.
//!
//! This is Olive's port of the standard library's
//! [`Cow`](stock_alloc::borrow::Cow). It is an enum that either borrows data or
//! owns it, letting callers express "maybe I'll need to own this" without paying
//! for the allocation up front.
//!
//! # Fallibility
//!
//! Methods that can allocate return a [`Result`] rather than panicking on allocation
//! failure, mirroring the rest of the crate. Every other method is infallible:
//! reading, comparing, converting, and dropping a `Cow` cannot fail.

use core::borrow::Borrow;
use core::fmt;
use core::ops::Deref;

use olive_core::prelude::*;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

use super::try_to_owned::{TryToOwned, TryToOwnedError};

/// An enum that represents either of two common ownership patterns for a piece
/// of data: borrowing or owning.
///
/// If you have an API which should accept the common case of a
/// borrowed reference, but allow manipulating and keeping the data if
/// you want, you can write your API against [`Cow`].
///
/// The owned arm is whatever [`TryToOwned::Owned`] names for `B`. For example,
/// [`str`]'s owned variant is a [`String`](crate::string::String).
pub enum Cow<'b, B>
where
    B: ?Sized + TryToOwned,
{
    /// A value representing a borrowed resource.
    Borrowed(&'b B),
    /// A value representing an owned resource.
    Owned(B::Owned),
}

impl<'b, B: ?Sized + TryToOwned> Cow<'b, B> {
    /// Returns true if the `Cow` is in the borrowed state.
    #[inline]
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Returns the contained `Borrowed` value, if any.
    ///
    /// If the `Cow` is in the borrowed state, the inner reference is returned;
    /// otherwise (the `Owned` state) `None` is returned.
    #[inline]
    pub fn borrowed(&self) -> Option<&'b B> {
        match self {
            Self::Borrowed(b) => Some(*b),
            Self::Owned(_) => None,
        }
    }

    /// Converts this `Cow` into an owned `B::Owned`, consuming it.
    ///
    /// If the `Cow` is already `Owned`, the value is returned as-is with no copy.
    /// If it is `Borrowed`, the data is cloned into an owned value.
    ///
    /// # Errors
    ///
    /// Returns a [`TryToOwnedError`] if cloning the borrowed data into an owned
    /// value fails (typically out-of-memory). On error the original borrow is
    /// consumed and discarded.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::borrow::cow::Cow;
    /// use olive_alloc::string::String;
    ///
    /// let borrowed: Cow<str> = Cow::Borrowed("hi");
    /// let owned: String = borrowed.try_into_owned().unwrap();
    /// assert_eq!(&*owned, "hi");
    /// ```
    pub fn try_into_owned(self) -> Result<<B as TryToOwned>::Owned, TryToOwnedError> {
        match self {
            Self::Borrowed(b) => b.try_to_owned(),
            Self::Owned(o) => Ok(o),
        }
    }

    /// Mutably borrows the **owned** contents of the `Cow`, ensuring they are owned first.
    ///
    /// If the `Cow` is already `Owned`, a mutable view of the payload is returned
    /// as-is. If it is `Borrowed`, the data is fallibly cloned into an owned value
    /// and the `Cow` is switched to the `Owned` variant before returning.
    ///
    /// This is the clone-on-write seam: it is one of the two operations that may
    /// allocate, and therefore one of the two that can fail.
    ///
    /// Unlike std's `to_mut` — which returns a `&mut B` (the *borrowed* element
    /// type) — this returns a `&mut B::Owned` (the *owned* type). That is strictly
    /// more versatile: for `Cow<str>` you get a `&mut String` with the full
    /// fallible API (`try_push_str`, `try_insert_str`, …), not just a read-only
    /// `&mut str`. When you only need element-level mutation, dereference down to
    /// `&mut B` via the owned type's `DerefMut`/`AsMut`.
    ///
    /// # Errors
    ///
    /// Returns a [`TryToOwnedError`] if cloning the borrowed data into an owned
    /// value fails (typically out-of-memory). On error the `Cow` is left
    /// untouched, still holding its original borrow.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::borrow::cow::Cow;
    /// use olive_alloc::boxed::Box;
    /// use olive_alloc::string::String;
    /// use olive_alloc::vec::Vec;
    ///
    /// // A borrowed string `Cow` flips into an owned one, exposing the full
    /// // `String` API through the returned `&mut String`.
    /// let mut cow: Cow<str> = Cow::Borrowed("hello");
    /// let s: &mut String = cow.try_to_mut().unwrap();
    /// s.try_push_str("!").unwrap();
    /// assert_eq!(&*cow, "hello!");
    ///
    /// // For slices, the owned form is a `Vec<i32>` (see `TryToOwned for [T]`),
    /// // so the returned reference is a `&mut Vec<i32>`.
    /// let mut vec_cow: Cow<[i32]> = Cow::Borrowed(&[1, 2, 3]);
    /// let vec_owned: &mut Vec<i32> = vec_cow.try_to_mut().unwrap();
    /// (*vec_owned)[0] += 10;
    /// assert_eq!(*vec_cow, [11, 2, 3]);
    /// ```
    pub fn try_to_mut(&mut self) -> Result<&mut <B as TryToOwned>::Owned, TryToOwnedError> {
        // Ensure we own the data, cloning from the borrow if necessary.
        match self {
            Self::Owned(_) => {}
            Self::Borrowed(b) => {
                // Fallibly lift the borrowed data into its owned form. On error
                // the `Cow` is left untouched, still holding the original borrow.
                let owned = b.try_to_owned()?;
                *self = Self::Owned(owned);
            }
        }
        // Now guaranteed to be `Owned`; hand back a mutable view of the payload.
        match self {
            Self::Owned(o) => Ok(o),
            Self::Borrowed(_) => unreachable!("switched to Owned above"),
        }
    }
}

// Two `Cow`s are comparable whenever their *borrowed* element types are mutually
// comparable. This single generic impl is the whole story for `Cow == Cow`:
// setting `A = B` recovers same-type comparison, and any `A` whose element type
// compares against `B` gives cross-type comparison (e.g. `Cow<str>` vs a
// `Cow` whose owned form borrows as something `str`-comparable). Both sides are
// dereffed down to their `&A` / `&B` targets and handed to the leaf comparison,
// so no container-specific equality logic lives here — it all rides on the
// existing `PartialEq` impls for the element types.
//
// Note this deliberately mirrors std's `PartialEq<Cow<'b, B>> for Cow<'a, A>`.
// That one impl is what "blocks" any generic `PartialEq` on `Cow`.
#[allow(clippy::needless_lifetimes, reason = "explicit descriptive lifetime")]
impl<'a, 'b, A: ?Sized, B: ?Sized> PartialEq<Cow<'b, B>> for Cow<'a, A>
where
    A: PartialEq<B> + TryToOwned,
    B: TryToOwned,
{
    #[inline]
    fn eq(&self, other: &Cow<'b, B>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

// This implementation is impossible:
// (1/mandatory impl) Cow<A> PartialEq Cow<B> if A PartialEq B
// (2) A PartialEq Cow<B> if A PartialEq B
// (3 - corollary of 2 - generic application A -> Cow<A>) Cow<A> PartialEq Cow<B> if Cow<A> PartialEq B

// impl<'a, 'b, A: ?Sized, B: ?Sized> PartialEq<Cow<'b, B>> for A
// where
//     A: PartialEq<B> + TryToOwned,
//     B: TryToOwned,
// {
//     #[inline]
//     fn eq(&self, other: &Cow<'b, B>) -> bool {
//         PartialEq::eq(&**self, &**other)
//     }
// }

// This implementation is impossible:
// (1/mandatory impl) Cow<A> PartialEq Cow<B> if A PartialEq B
// (2) Cow<A> PartialEq B if A PartialEq B
// (3 - corollary of 2 - generic application B -> Cow<B>) Cow<A> PartialEq Cow<B> if Cow<A> PartialEq Cow<B>

// impl<'a, 'b, A: ?Sized, B: ?Sized> PartialEq<B> for Cow<'a, A>
// where
//     A: PartialEq<B> + TryToOwned,
//     B: TryToOwned,
// {
//     #[inline]
//     fn eq(&self, other: &Cow<'b, B>) -> bool {
//         PartialEq::eq(&**self, &**other)
//     }
// }

// `Eq` is the marker half of that comparison: it holds exactly when the element
// type is an equivalence relation over itself.
impl<B: ?Sized> Eq for Cow<'_, B> where B: Eq + TryToOwned {}

// Cow is Send only when both variants are Send.
unsafe impl<B> Send for Cow<'_, B>
where
    B: Send + TryToOwned + ?Sized,
    B::Owned: Send,
{
}

// Cow is Sync only when both variants are Sync.
unsafe impl<B> Sync for Cow<'_, B>
where
    B: ?Sized + TryToOwned + Sync,
    B::Owned: Sync,
{
}

impl<B> fmt::Debug for Cow<'_, B>
where
    B: ?Sized + TryToOwned + fmt::Debug,
    B::Owned: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Borrowed(b) => fmt::Debug::fmt(b, f),
            Self::Owned(o) => fmt::Debug::fmt(o, f),
        }
    }
}

impl<B> fmt::Display for Cow<'_, B>
where
    B: ?Sized + TryToOwned + fmt::Display,
    B::Owned: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Borrowed(b) => fmt::Display::fmt(b, f),
            Self::Owned(o) => fmt::Display::fmt(o, f),
        }
    }
}

impl<B> Deref for Cow<'_, B>
where
    B: ?Sized + TryToOwned,
{
    type Target = B;

    #[inline]
    fn deref(&self) -> &B {
        self.borrow()
    }
}

impl<B> Borrow<B> for Cow<'_, B>
where
    B: ?Sized + TryToOwned,
{
    #[inline]
    fn borrow(&self) -> &B {
        match *self {
            Cow::Borrowed(b) => b,
            Cow::Owned(ref o) => o.borrow(),
        }
    }
}

impl<B> AsRef<B> for Cow<'_, B>
where
    B: ?Sized + TryToOwned,
{
    #[inline]
    fn as_ref(&self) -> &B {
        self.borrow()
    }
}

impl<B: ?Sized + TryToOwned> TryDefault for Cow<'_, B>
where
    <B as TryToOwned>::Owned: TryDefault,
{
    /// Constructs an empty `Cow` in its **owned** state, holding a freshly-defaulted
    /// `B::Owned`.
    fn try_default() -> Result<Self, TryDefaultError> {
        <B as TryToOwned>::Owned::try_default().map(Self::Owned)
    }
}

impl<B: ?Sized + TryToOwned> TryClone for Cow<'_, B>
where
    <B as TryToOwned>::Owned: TryClone,
{
    /// Clones the `Cow`. A borrowed `Cow` is cheaply re-borrowed (no allocation);
    /// an owned `Cow` fallibly clones its payload.
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            Cow::Borrowed(b) => Ok(Cow::Borrowed(b)),
            Cow::Owned(o) => Ok(Cow::Owned(o.try_clone()?)),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::string::String;
    use crate::vec::Vec;

    /// Build a `String` from a literal, unwrapping the fallible conversion.
    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    #[test]
    fn is_borrowed_reports_state() {
        let cow: Cow<str> = Cow::Borrowed("x");
        assert!(cow.is_borrowed());
        let cow: Cow<str> = Cow::Owned(mk("y"));
        assert!(!cow.is_borrowed());
    }

    #[test]
    fn borrowed_returns_inner_or_none() {
        let s = "abc";
        let cow: Cow<str> = Cow::Borrowed(s);
        assert_eq!(cow.borrowed(), Some(s));
        let cow: Cow<str> = Cow::Owned(mk("def"));
        assert_eq!(cow.borrowed(), None);
    }

    #[test]
    fn borrowed_as_ref_and_deref() {
        let s = "hello";
        let cow: Cow<str> = Cow::Borrowed(s);
        // Compare the dereffed `str` against a `String` (the crate provides
        // `PartialEq<String> for str` but no reflexive `str == &str`).
        assert_eq!(&*cow, &mk("hello"));
    }

    #[test]
    fn owned_as_ref_and_deref() {
        let cow: Cow<str> = Cow::Owned(mk("world"));
        assert_eq!(&*cow, &mk("world"));
    }

    #[test]
    fn try_into_owned_borrowed_clones() {
        let cow: Cow<str> = Cow::Borrowed("become owned");
        let owned: String = cow.try_into_owned().unwrap();
        assert_eq!(&*owned, "become owned");
    }

    #[test]
    fn try_into_owned_owned_is_passthrough() {
        let cow: Cow<str> = Cow::Owned(mk("already"));
        let owned: String = cow.try_into_owned().unwrap();
        assert_eq!(&*owned, "already");
    }

    // `try_to_mut` returns a `&mut B::Owned`; on an already-owned `Cow` it is a
    // no-op that hands straight back a mutable view of the payload.
    #[test]
    fn try_to_mut_on_owned_is_noop() {
        let mut cow: Cow<str> = Cow::Owned(mk("keep"));
        let s: &mut String = cow.try_to_mut().unwrap();
        assert_eq!(&**s, "keep");
        // Still owned, contents intact.
        assert!(!cow.is_borrowed());
        assert_eq!(&*cow, "keep");
    }

    // On a borrowed `Cow`, `try_to_mut` clones into an owned value and then
    // exposes it mutably — the clone-on-write moment. We prove the flip happened
    // by observing the returned `&mut String` equals the source and that the
    // `Cow` switched to its `Owned` arm.
    #[test]
    fn try_to_mut_on_borrowed_flips_to_owned() {
        let mut cow: Cow<str> = Cow::Borrowed("clone me");
        let s: &mut String = cow.try_to_mut().unwrap();
        // The returned reference points at freshly-allocated owned storage whose
        // contents match the original borrow.
        assert_eq!(&**s, "clone me");
        // The Cow flipped to Owned and still holds the cloned contents.
        assert!(!cow.is_borrowed());
        assert_eq!(&*cow, "clone me");
    }

    #[test]
    fn borrowed_variant_holds_reference() {
        let s = "from ref";
        let cow: Cow<str> = Cow::Borrowed(s);
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(&*cow, &mk("from ref"));
    }

    #[test]
    fn owned_variant_holds_string() {
        let cow: Cow<str> = Cow::Owned(mk("from owned"));
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(&*cow, &mk("from owned"));
    }

    // Equality between a `Cow<str>` and a `String` works in both directions: the
    // derived `PartialEq for Cow` compares arm-by-arm, and our `Deref` lets the
    // dereffed `str` compare against a `String` via the crate's cross-type impl.
    #[test]
    fn partial_eq_cross_types() {
        let cow: Cow<str> = Cow::Borrowed("equal");
        let owned: String = mk("equal");
        assert_eq!(&*cow, &owned);
        assert_eq!(&owned, &*cow);
        let different: String = mk("different");
        assert_ne!(&*cow, &different);
    }

    // Same-type `Cow == Cow`: the generic cross-Cow impl with `C = B`. Both arms
    // deref to their element type, so equality is content-based regardless of
    // which variant (borrowed vs owned) each side sits in.
    #[test]
    fn cow_eq_same_type_both_directions() {
        let a: Cow<str> = Cow::Borrowed("same");
        let b: Cow<str> = Cow::Owned(mk("same"));
        assert!(a == b);
        assert!(b == a);
        assert!(!(a != b));

        let c: Cow<str> = Cow::Borrowed("diff");
        assert!(a != c);
        assert!(c != a);
    }

    // Same-type slice `Cow == Cow`, mixing borrowed and owned (`Vec`) arms.
    #[test]
    fn cow_eq_slice_same_type() {
        let data: Vec<i32> = Vec::try_from(&[1, 2, 3][..]).unwrap();
        let borrowed: Cow<[i32]> = Cow::Borrowed(data.as_slice());
        let owned: Cow<[i32]> = Cow::Owned(Vec::try_from(&[1, 2, 3][..]).unwrap());
        assert!(borrowed == owned);
        assert!(owned == borrowed);
    }

    // Same-type `Cow == Cow` where both sides sit in the *owned* arm, proving the
    // deref goes through `Deref` on the owned side too (not just the borrowed
    // reference). For `Cow<str>` the owned form is `String`, whose `Deref` yields
    // the underlying `&str`.
    #[test]
    fn cow_eq_same_type_both_owned_arms() {
        let a: Cow<str> = Cow::Owned(mk("dup"));
        let b: Cow<str> = Cow::Owned(mk("dup"));
        assert!(a == b);
        assert!(b == a);
        assert!(!(a != b));
    }

    #[test]
    fn debug_and_display_forward() {
        let cow: Cow<str> = Cow::Borrowed("shown");
        // `Debug`/`Display` forward to the inner `str`; verify via equality of
        // the dereffed contents rather than formatting (no macros in scope).
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(&*cow, &mk("shown"));
    }

    #[test]
    fn slice_cow_borrowed_try_into_owned() {
        let mut data: Vec<i32> = Vec::new();
        for v in [1, 2, 3] {
            data.try_push(v).unwrap();
        }
        let cow: Cow<[i32]> = Cow::Borrowed(data.as_slice());
        assert_eq!(cow.as_ref(), &[1, 2, 3]);
        let owned: Vec<i32> = cow.try_into_owned().unwrap();
        assert_eq!(&*owned, [1, 2, 3]);
    }

    #[test]
    fn slice_cow_try_to_mut_clones_then_mutates() {
        let mut data: Vec<i32> = Vec::new();
        for v in [4, 5] {
            data.try_push(v).unwrap();
        }
        let mut cow: Cow<[i32]> = Cow::Borrowed(data.as_slice());
        // For slices the owned form is a `Vec<i32>` (see `TryToOwned for [T]`).
        let vec_owned: &mut Vec<i32> = cow.try_to_mut().unwrap();
        (*vec_owned)[0] += 100;
        assert!(!cow.is_borrowed());
        assert_eq!(*cow, [104, 5]);
    }

    #[test]
    fn send_sync_bounds_hold() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Cow<'_, str>>();
        assert_sync::<Cow<'_, str>>();
        assert_send::<Cow<'_, [i32]>>();
        assert_sync::<Cow<'_, [i32]>>();
    }

    // A custom borrowed type whose owned conversion always fails lets us verify
    // that `try_to_mut` surfaces the allocation error and leaves the `Cow`
    // untouched (still `Borrowed`). `Owned` is the type itself, which trivially
    // satisfies the required `Borrow<Self>` bound via std's reflexive impl — we
    // never actually construct one, since conversion fails.
    #[test]
    fn try_to_mut_fails_when_conversion_fails() {
        #[derive(Debug)]
        struct FailingBytes;
        impl TryToOwned for FailingBytes {
            type Owned = FailingBytes;
            fn try_to_owned(&self) -> Result<FailingBytes, TryToOwnedError> {
                Err(TryToOwnedError::Alloc(crate::alloc::AllocError))
            }
        }

        let mut cow: Cow<FailingBytes> = Cow::Borrowed(&FailingBytes);
        let err = cow.try_to_mut().unwrap_err();
        assert!(matches!(err, TryToOwnedError::Alloc(_)));
        // The Cow must remain a Borrowed pointing at the same value.
        assert!(matches!(cow, Cow::Borrowed(_)));
    }

    // Same guarantee for `try_into_owned`: a failed conversion yields an error
    // rather than panicking.
    #[test]
    fn try_into_owned_fails_when_conversion_fails() {
        #[derive(Debug)]
        struct FailingSlice;
        impl TryToOwned for FailingSlice {
            type Owned = FailingSlice;
            fn try_to_owned(&self) -> Result<FailingSlice, TryToOwnedError> {
                Err(TryToOwnedError::Reserve(
                    olive_core::alloc_errors::TryReserveError::new_capacity_overflow(),
                ))
            }
        }

        let cow: Cow<FailingSlice> = Cow::Borrowed(&FailingSlice);
        let err = cow.try_into_owned().unwrap_err();
        assert!(matches!(err, TryToOwnedError::Reserve(_)));
    }
}
