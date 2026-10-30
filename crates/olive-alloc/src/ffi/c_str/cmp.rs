//! Comparison impls for [`CStr`].
use olive_core::alloc::Allocator;

use super::CString;
use crate::borrow::Cow;
use crate::boxed::Box;
use core::{cmp::Ordering, ffi::CStr};

// ---------------------------------------------------------------------------
// Macro partial comparisons
// ---------------------------------------------------------------------------

/// Emits four cross-type impls: `PartialEq<RHS> for LHS`, `PartialEq<LHS> for
/// RHS`, `PartialOrd<RHS> for LHS`, and `PartialOrd<LHS> for RHS`. Each body
/// pins both operands to `$prim` (always `CStr`) via an explicit type
/// annotation before delegating, so the result depends only on the C string
/// contents.
///
/// Usage: `impl_cmp!((generic_params), LhsTy, RhsTy, PrimTy);`
/// where `(generic_params)` is the exact token list for the `impl<…>` header.
macro_rules! impl_cmp {
    (($($generics:tt)*), $lhs:ty, $rhs:ty, $prim:ty) => {
        #[allow(unused_lifetimes)]
        impl<$($generics)*> core::cmp::PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                let lhs: &$prim = self;
                let rhs: &$prim = other;
                core::cmp::PartialEq::eq(lhs, rhs)
            }
        }

        #[allow(unused_lifetimes)]
        impl<$($generics)*> core::cmp::PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                let lhs: &$prim = self;
                let rhs: &$prim = other;
                core::cmp::PartialEq::eq(lhs, rhs)
            }
        }

        #[allow(unused_lifetimes)]
        impl<$($generics)*> core::cmp::PartialOrd<$rhs> for $lhs {
            #[inline]
            fn partial_cmp(&self, other: &$rhs) -> Option<core::cmp::Ordering> {
                let lhs: &$prim = self;
                let rhs: &$prim = other;
                core::cmp::PartialOrd::partial_cmp(lhs, rhs)
            }
        }

        #[allow(unused_lifetimes)]
        impl<$($generics)*> core::cmp::PartialOrd<$lhs> for $rhs {
            #[inline]
            fn partial_cmp(&self, other: &$lhs) -> Option<core::cmp::Ordering> {
                let lhs: &$prim = self;
                let rhs: &$prim = other;
                core::cmp::PartialOrd::partial_cmp(lhs, rhs)
            }
        }
    };
}

// CString<A> vs borrowed CStr targets.
// CString<A> <-> CStr
impl_cmp!((A: Allocator), CString<A>, CStr, CStr);
// CString<A> <-> &CStr
impl_cmp!(('ref_b, A: Allocator), CString<A>, &'ref_b CStr, CStr);

// CString<A> vs boxed CStr targets.
// CString<A> <-> Box<CStr, B>
impl_cmp!((A: Allocator, B: Allocator), CString<A>, Box<CStr, B>, CStr);
// CString<A> <-> &Box<CStr, B>
impl_cmp!(('ref_b, A: Allocator, B: Allocator), CString<A>, &'ref_b Box<CStr, B>, CStr);

// Box<CStr, A> vs borrowed CStr targets.
// Box<CStr, A> <-> CStr
impl_cmp!((A: Allocator), Box<CStr, A>, CStr, CStr);
// &CStr <-> Box<CStr, B>
impl_cmp!(('ref_a, B: Allocator), &'ref_a CStr, Box<CStr, B>, CStr);

// Cow<'a, CStr> vs borrowed CStr targets.
// Cow<CStr> <-> CStr
impl_cmp!(('a), Cow<'a, CStr>, CStr, CStr);
// Cow<CStr> <-> &CStr
impl_cmp!(('a, 'ref_b), Cow<'a, CStr>, &'ref_b CStr, CStr);

// Cow<'a, CStr> vs boxed CStr targets.
// Cow<'a, CStr> <-> Box<CStr, B>
impl_cmp!(('a, B: Allocator), Cow<'a, CStr>, Box<CStr, B>, CStr);
// Cow<'a, CStr> <-> &Box<CStr, B>
impl_cmp!(('a, 'ref_b, B: Allocator), Cow<'a, CStr>, &'ref_b Box<CStr, B>, CStr);

// Cow<'a, CStr> vs CString targets.
// CString<A> <-> Cow<'a, CStr>
impl_cmp!(('a, A: Allocator), CString<A>, Cow<'a, CStr>, CStr);

// ---------------------------------------------------------------------------
// Same-type comparisons
// ---------------------------------------------------------------------------

impl<A: Allocator, B: Allocator> PartialEq<CString<B>> for CString<A> {
    /// Compares two strings by their full byte sequences, terminating NUL
    /// included, regardless of their allocators.
    fn eq(&self, other: &CString<B>) -> bool {
        PartialEq::eq(&*self.inner, &*other.inner)
    }
}

impl<A: Allocator> Eq for CString<A> {}

impl<A: Allocator, B: Allocator> PartialOrd<CString<B>> for CString<A> {
    /// Compares two strings lexicographically by their bytes, regardless of their
    /// allocators.
    fn partial_cmp(&self, other: &CString<B>) -> Option<Ordering> {
        Some(Ord::cmp(&*self.inner, &*other.inner))
    }
}

impl<A: Allocator> Ord for CString<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&*self.inner, &*other.inner)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::alloc::Global;
    use crate::borrow::TryToOwned;
    use crate::ffi::CString;
    use crate::vec::Vec;
    use core::cmp::Ordering;
    use olive_test_helpers::CountingAllocator;

    fn c_str(bytes: &[u8]) -> &CStr {
        CStr::from_bytes_with_nul(bytes).expect("the test bytes hold exactly one trailing nul")
    }

    fn cow_borrowed(bytes: &[u8]) -> Cow<'_, CStr> {
        Cow::Borrowed(c_str(bytes))
    }

    fn cow_owned(bytes: &[u8]) -> Cow<'static, CStr> {
        let owned: CString = c_str(bytes).try_to_owned().expect("allocation cannot fail");
        Cow::Owned(owned)
    }

    fn cstring(bytes: &[u8]) -> CString {
        cstring_in(bytes, Global)
    }

    fn cstring_in<A: Allocator>(bytes: &[u8], alloc: A) -> CString<A> {
        // SAFETY: the checked input is a byte literal without NUL bytes.
        unsafe {
            CString::try_from_vec_unchecked(
                Vec::try_from_slice_in(bytes, alloc).expect("allocation cannot fail"),
            )
        }
        .expect("should allocate string, the input has no interior NUL byte")
    }

    fn boxed_c_str(bytes: &[u8]) -> Box<CStr> {
        cstring(bytes).into_boxed_c_str()
    }

    fn boxed_c_str_in<A: Allocator>(bytes: &[u8], alloc: A) -> Box<CStr, A> {
        cstring_in(bytes, alloc).into_boxed_c_str()
    }

    #[test]
    fn equal_strings_are_equal() {
        assert_eq!(cstring(b"a"), cstring(b"a"));
        assert_ne!(cstring(b"a"), cstring(b"b"));
        assert_ne!(cstring(b"a"), cstring(b"ab"));
    }

    #[test]
    fn strings_order_lexicographically() {
        assert!(cstring(b"a") < cstring(b"ab"));
        assert!(cstring(b"ab") > cstring(b"a"));
        assert!(cstring(b"ab") < cstring(b"b"));
        assert!(cstring(b"b") > cstring(b"ab"));
        assert!(cstring(b"a") < cstring(b"a\x01"));
    }

    #[test]
    fn strings_compare_regardless_of_their_allocators() {
        let alloc = CountingAllocator::new();
        let counting_cstring = cstring_in(b"a", &alloc);

        assert_eq!(cstring(b"a"), counting_cstring);
        assert_eq!(counting_cstring, cstring(b"a"));
        assert_ne!(cstring(b"b"), counting_cstring);
        assert!(counting_cstring < cstring(b"ab"));
        assert!(cstring(b"") < counting_cstring);
    }

    #[test]
    fn cow_borrowed_equals_the_c_str() {
        let source = c_str(b"foo\0");
        let cow = cow_borrowed(b"foo\0");
        // Both directions.
        assert!(cow == *source);
        assert!(source == cow);
        // Unequal, both directions.
        let other = c_str(b"bar\0");
        assert!(cow != *other);
        assert!(other != cow);
    }

    #[test]
    fn cow_owned_equals_the_c_str() {
        let source = c_str(b"foo\0");
        let cow = cow_owned(b"foo\0");
        assert!(cow == *source);
        assert!(source == cow);
        let other = c_str(b"bar\0");
        assert!(cow != *other);
        assert!(other != cow);
    }

    #[test]
    fn cow_borrowed_orders_against_the_c_str() {
        let cow = cow_borrowed(b"bar\0");
        let first = c_str(b"bar\0");
        let second = c_str(b"foo\0");
        assert_eq!(cow.partial_cmp(first), Some(Ordering::Equal));
        assert_eq!(first.partial_cmp(&cow), Some(Ordering::Equal));
        assert_eq!(cow.partial_cmp(second), Some(Ordering::Less));
        assert_eq!(second.partial_cmp(&cow), Some(Ordering::Greater));
    }

    #[test]
    fn cow_owned_orders_against_the_c_str() {
        let cow = cow_owned(b"bar\0");
        let first = c_str(b"bar\0");
        let second = c_str(b"foo\0");
        assert_eq!(cow.partial_cmp(first), Some(Ordering::Equal));
        assert_eq!(first.partial_cmp(&cow), Some(Ordering::Equal));
        assert_eq!(cow.partial_cmp(second), Some(Ordering::Less));
        assert_eq!(second.partial_cmp(&cow), Some(Ordering::Greater));
    }

    #[test]
    fn cstring_equals_the_c_str() {
        let source = c_str(b"foo\0");
        let c_string = cstring(b"foo");
        // Both directions.
        assert!(c_string == *source);
        assert!(*source == c_string);
        // Unequal, both directions.
        let other = c_str(b"bar\0");
        assert!(c_string != *other);
        assert!(*other != c_string);
    }

    #[test]
    fn cstring_equals_the_c_str_reference() {
        let source = c_str(b"foo\0");
        let c_string = cstring(b"foo");
        // Both directions.
        assert!(c_string == source);
        assert!(source == c_string);
        // Unequal, both directions.
        let other = c_str(b"bar\0");
        assert!(c_string != other);
        assert!(other != c_string);
    }

    #[test]
    fn cstring_equals_the_cow() {
        let c_string = cstring(b"foo");
        let borrowed = cow_borrowed(b"foo\0");
        let owned = cow_owned(b"foo\0");
        // Both directions, both arms.
        assert!(c_string == borrowed);
        assert!(borrowed == c_string);
        assert!(c_string == owned);
        assert!(owned == c_string);
        // Unequal, both directions.
        let other = cow_owned(b"bar\0");
        assert!(c_string != other);
        assert!(other != c_string);
    }

    #[test]
    fn cstring_orders_against_the_c_str() {
        let c_string = cstring(b"bar");
        let first = c_str(b"bar\0");
        let second = c_str(b"foo\0");
        assert_eq!(c_string.partial_cmp(first), Some(Ordering::Equal));
        assert_eq!(c_string.partial_cmp(&first), Some(Ordering::Equal));
        assert_eq!(first.partial_cmp(&c_string), Some(Ordering::Equal));
        assert_eq!((&first).partial_cmp(&c_string), Some(Ordering::Equal));
        assert_eq!(c_string.partial_cmp(second), Some(Ordering::Less));
        assert_eq!(second.partial_cmp(&c_string), Some(Ordering::Greater));
    }

    #[test]
    fn cstring_orders_against_the_cow() {
        let c_string = cstring(b"bar");
        let equal = cow_borrowed(b"bar\0");
        let greater = cow_owned(b"foo\0");
        assert_eq!(c_string.partial_cmp(&equal), Some(Ordering::Equal));
        assert_eq!(equal.partial_cmp(&c_string), Some(Ordering::Equal));
        assert_eq!(c_string.partial_cmp(&greater), Some(Ordering::Less));
        assert_eq!(greater.partial_cmp(&c_string), Some(Ordering::Greater));
    }

    #[test]
    fn cstring_compares_against_borrowed_targets_regardless_of_its_allocator() {
        let alloc = CountingAllocator::new();
        let c_string = cstring_in(b"a", &alloc);
        let source = c_str(b"a\0");
        let cow = cow_borrowed(b"a\0");

        assert_eq!(c_string, *source);
        assert_eq!(*source, c_string);
        assert_eq!(c_string, cow);
        assert_eq!(cow, c_string);
        let longer = c_str(b"ab\0");
        let cow_longer = cow_borrowed(b"ab\0");
        assert!(c_string < *longer);
        assert!(*longer > c_string);
        assert!(c_string < *cow_longer);
        assert!(*cow_longer > c_string);
    }

    #[test]
    fn cstring_equals_the_boxed_c_str() {
        let boxed = boxed_c_str(b"foo");
        let c_string = cstring(b"foo");
        // Both directions.
        assert!(c_string == boxed);
        assert!(boxed == c_string);
        // Unequal, both directions.
        let other = boxed_c_str(b"bar");
        assert!(c_string != other);
        assert!(other != c_string);
    }

    #[test]
    fn cstring_equals_the_boxed_c_str_reference() {
        let boxed = boxed_c_str(b"foo");
        let boxed_ref: &Box<CStr> = &boxed;
        let c_string = cstring(b"foo");
        // Both directions.
        assert!(c_string == boxed_ref);
        assert!(boxed_ref == c_string);
        // Unequal, both directions.
        let other = boxed_c_str(b"bar");
        let other_ref: &Box<CStr> = &other;
        assert!(c_string != other_ref);
        assert!(other_ref != c_string);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn cstring_orders_against_the_boxed_c_str() {
        let c_string = cstring(b"bar");
        let equal = boxed_c_str(b"bar");
        let greater = boxed_c_str(b"foo");
        assert_eq!(c_string.partial_cmp(&equal), Some(Ordering::Equal));
        assert_eq!(c_string.partial_cmp(&&equal), Some(Ordering::Equal));
        assert_eq!(equal.partial_cmp(&c_string), Some(Ordering::Equal));
        assert_eq!((&&equal).partial_cmp(&c_string), Some(Ordering::Equal));
        assert_eq!(c_string.partial_cmp(&greater), Some(Ordering::Less));
        assert_eq!(c_string.partial_cmp(&&greater), Some(Ordering::Less));
        assert_eq!(greater.partial_cmp(&c_string), Some(Ordering::Greater));
        assert_eq!((&&greater).partial_cmp(&c_string), Some(Ordering::Greater));
        assert!(c_string < greater);
        assert!(greater > c_string);
        assert!(c_string < &greater);
        assert!(&greater > c_string);
    }

    #[test]
    fn boxed_c_str_equals_the_c_str() {
        let source = c_str(b"foo\0");
        let boxed = boxed_c_str(b"foo");
        // Both directions, by value and by reference.
        assert!(boxed == *source);
        assert!(*source == boxed);
        assert!(boxed == source);
        assert!(source == boxed);
        // Unequal, both directions.
        let other = c_str(b"bar\0");
        assert!(boxed != *other);
        assert!(*other != boxed);
        assert!(boxed != other);
        assert!(other != boxed);
    }

    #[test]
    fn boxed_c_str_orders_against_the_c_str() {
        let boxed = boxed_c_str(b"bar");
        let first = c_str(b"bar\0");
        let second = c_str(b"foo\0");
        assert_eq!(boxed.partial_cmp(first), Some(Ordering::Equal));
        assert_eq!(boxed.partial_cmp(&first), Some(Ordering::Equal));
        assert_eq!(first.partial_cmp(&boxed), Some(Ordering::Equal));
        assert_eq!((&first).partial_cmp(&boxed), Some(Ordering::Equal));
        assert_eq!(boxed.partial_cmp(second), Some(Ordering::Less));
        assert_eq!(boxed.partial_cmp(&second), Some(Ordering::Less));
        assert_eq!(second.partial_cmp(&boxed), Some(Ordering::Greater));
        assert_eq!((&second).partial_cmp(&boxed), Some(Ordering::Greater));
        assert!(boxed < second);
        assert!(boxed < *second);
        assert!(second > boxed);
        assert!(*second > boxed);
    }

    #[test]
    fn cow_equals_the_boxed_c_str() {
        let boxed = boxed_c_str(b"foo");
        let boxed_ref: &Box<CStr> = &boxed;
        let borrowed = cow_borrowed(b"foo\0");
        let owned = cow_owned(b"foo\0");
        // Both directions, both arms, by value and by reference.
        assert!(borrowed == boxed);
        assert!(boxed == borrowed);
        assert!(owned == boxed);
        assert!(boxed == owned);
        assert!(borrowed == boxed_ref);
        assert!(boxed_ref == borrowed);
        assert!(owned == boxed_ref);
        assert!(boxed_ref == owned);
        // Unequal, both directions.
        let other = boxed_c_str(b"bar");
        let other_ref: &Box<CStr> = &other;
        assert!(owned != other);
        assert!(other != owned);
        assert!(borrowed != other_ref);
        assert!(other_ref != borrowed);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn cow_orders_against_the_boxed_c_str() {
        let borrowed = cow_borrowed(b"bar\0");
        let owned = cow_owned(b"bar\0");
        let equal = boxed_c_str(b"bar");
        let greater = boxed_c_str(b"foo");
        assert_eq!(borrowed.partial_cmp(&equal), Some(Ordering::Equal));
        assert_eq!(borrowed.partial_cmp(&&equal), Some(Ordering::Equal));
        assert_eq!(equal.partial_cmp(&borrowed), Some(Ordering::Equal));
        assert_eq!((&&equal).partial_cmp(&borrowed), Some(Ordering::Equal));
        assert_eq!(owned.partial_cmp(&equal), Some(Ordering::Equal));
        assert_eq!(owned.partial_cmp(&&equal), Some(Ordering::Equal));
        assert_eq!(equal.partial_cmp(&owned), Some(Ordering::Equal));
        assert_eq!((&&equal).partial_cmp(&owned), Some(Ordering::Equal));
        assert_eq!(owned.partial_cmp(&greater), Some(Ordering::Less));
        assert_eq!(owned.partial_cmp(&&greater), Some(Ordering::Less));
        assert_eq!(greater.partial_cmp(&owned), Some(Ordering::Greater));
        assert_eq!((&&greater).partial_cmp(&owned), Some(Ordering::Greater));
        assert!(borrowed < greater);
        assert!(borrowed < &greater);
        assert!(greater > borrowed);
        assert!(&greater > borrowed);
        assert!(owned < greater);
        assert!(owned < &greater);
        assert!(greater > owned);
        assert!(&greater > owned);
    }

    #[test]
    fn owned_targets_compare_regardless_of_its_allocator() {
        let alloc = CountingAllocator::new();
        let c_string = cstring_in(b"a", &alloc);
        let boxed = boxed_c_str_in(b"a", &alloc);
        let cow = cow_owned(b"a\0");
        let boxed_ref: &Box<CStr, _> = &boxed;

        assert_eq!(c_string, boxed);
        assert_eq!(boxed, c_string);
        assert_eq!(c_string, boxed_ref);
        assert_eq!(boxed_ref, c_string);

        assert_eq!(boxed, cow);
        assert_eq!(cow, boxed);
        assert_eq!(boxed_ref, cow);
        assert_eq!(cow, boxed_ref);

        assert_eq!(c_string, cow);
        assert_eq!(cow, c_string);

        let longer = boxed_c_str(b"ab");
        assert!(c_string < longer);
        assert!(longer > c_string);
        assert!(cow < longer);
        assert!(longer > cow);
        assert!(boxed < longer);
        assert!(longer > boxed);
    }

    #[test]
    fn empty_cows_compare_by_content() {
        let empty = c_str(b"\0");
        let cow = cow_owned(b"\0");
        assert!(cow == *empty);
        assert!(empty == cow);
        assert_eq!(cow.partial_cmp(&empty), Some(Ordering::Equal));
    }
}
