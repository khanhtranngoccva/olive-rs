//! Comparison impls for [`CStr`].
use olive_core::alloc::Allocator;

use super::CString;
use crate::borrow::Cow;
use core::{cmp::Ordering, ffi::CStr};

// ---------------------------------------------------------------------------
// Cross-type comparisons
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

// Cow<'a, CStr> vs borrowed CStr targets.
// Cow<CStr> <-> CStr
impl_cmp!(('a), Cow<'a, CStr>, CStr, CStr);
// Cow<CStr> <-> &CStr
impl_cmp!(('a, 'ref_b), Cow<'a, CStr>, &'ref_b CStr, CStr);

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
        let cow = cow_owned(b"foo\0");
        let first = c_str(b"bar\0");
        let second = c_str(b"bar\0");
        assert_eq!(cow.partial_cmp(first), Some(Ordering::Equal));
        assert_eq!(first.partial_cmp(&cow), Some(Ordering::Equal));
        assert_eq!(cow.partial_cmp(second), Some(Ordering::Less));
        assert_eq!(second.partial_cmp(&cow), Some(Ordering::Greater));
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
