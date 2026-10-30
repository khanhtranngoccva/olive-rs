//! Trait implementations for the [`CString`] type.

use super::CString;
use crate::alloc::{Allocator, AllocatorTryClone};
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::ffi::CStr;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Deref;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

// ---------------------------------------------------------------------------
// Comparison and hashing
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

impl<A: Allocator> Hash for CString<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&self.inner, state);
    }
}

// ---------------------------------------------------------------------------
// Borrow, AsRef and Deref
// ---------------------------------------------------------------------------

impl<A: Allocator> Borrow<CStr> for CString<A> {
    /// Borrows the string as a [`CStr`].
    fn borrow(&self) -> &CStr {
        self.as_c_str()
    }
}

impl<A: Allocator> AsRef<CStr> for CString<A> {
    fn as_ref(&self) -> &CStr {
        self.borrow()
    }
}

impl<A: Allocator> Deref for CString<A> {
    type Target = CStr;

    fn deref(&self) -> &CStr {
        self.borrow()
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl<A: Allocator> fmt::Debug for CString<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.as_c_str(), f)
    }
}

// ---------------------------------------------------------------------------
// TryClone
// ---------------------------------------------------------------------------

impl<A: AllocatorTryClone> TryClone for CString<A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let inner = self.inner.try_clone()?;
        Ok(Self { inner })
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::CString;
    use crate::alloc::{CountingAllocator, Global};
    use crate::vec::Vec;
    use core::borrow::Borrow;
    use core::ffi::CStr;
    use olive_core::alloc::Allocator;
    use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
    use olive_test_helpers::{BudgetedAlloc, CloneBudget, FlakyCloneAlloc};
    use std::hash::{DefaultHasher, Hash, Hasher};

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
    fn equal_strings_hash_equally() {
        fn hash(value: &CString) -> u64 {
            let mut hasher = DefaultHasher::new();
            Hash::hash(value, &mut hasher);
            Hasher::finish(&hasher)
        }

        assert_eq!(hash(&cstring(b"foo")), hash(&cstring(b"foo")));
        assert_ne!(hash(&cstring(b"foo")), hash(&cstring(b"bar")));
    }

    #[test]
    fn cstring_debug_matches_the_c_str_debug() {
        let c_string = cstring(b"foo");
        let c_str: &CStr = c_string.borrow();
        std::assert_eq!(std::format!("{c_string:?}"), std::format!("{c_str:?}"));
    }

    #[test]
    fn cstring_try_clone_copies_the_bytes() {
        let original = cstring_in(b"abc", Global);
        let cloned: CString = original.try_clone().expect("allocation cannot fail");

        assert_eq!(original.as_bytes_with_nul(), b"abc\0");
        assert_eq!(cloned.as_bytes_with_nul(), b"abc\0");
        let original_c_str: &CStr = original.borrow();
        let cloned_c_str: &CStr = cloned.borrow();
        assert_ne!(
            original_c_str.as_ptr(),
            cloned_c_str.as_ptr(),
            "a clone must own its own buffer"
        );
    }

    #[test]
    fn cstring_try_clone_backs_both_strings_with_the_same_allocator() {
        let alloc = CountingAllocator::new();
        // SAFETY: the byte literal has no NUL bytes.
        let original = cstring_in(b"ab", &alloc);
        let baseline_allocations = alloc.allocations();
        let baseline_deallocations = alloc.deallocations();
        let cloned: CString<&CountingAllocator> = original
            .try_clone()
            .expect("cloning a shared handle cannot fail");

        // The clone adds exactly one allocation on the shared allocator, and
        // dropping the two strings gives every block back.
        assert_eq!(alloc.allocations(), baseline_allocations + 1);
        assert_eq!(alloc.deallocations(), baseline_deallocations);
        drop(cloned);
        assert_eq!(alloc.deallocations(), baseline_deallocations + 1);
        drop(original);
        assert_eq!(alloc.deallocations(), baseline_allocations + 1);
    }

    #[test]
    fn cstring_try_clone_fails_when_the_buffer_allocation_fails() {
        let alloc = BudgetedAlloc::new(4);
        // SAFETY: the byte literal has no NUL bytes.
        let original = cstring_in(b"ab", alloc.clone());
        // Cloning the allocator handle succeeds, but the buffer allocation
        // cannot once the budget is gone.
        alloc.drain();
        let error = original.try_clone().unwrap_err();
        assert!(matches!(error, TryCloneError::Alloc(_)));
        // The original is untouched by the failed clone.
        assert_eq!(original.as_bytes_with_nul(), b"ab\0");
    }

    #[test]
    fn cstring_try_clone_propagates_allocator_failure() {
        let alloc = FlakyCloneAlloc::new(CloneBudget::new(0));
        // SAFETY: the byte literal has no NUL bytes.
        let original = unsafe {
            CString::try_from_vec_unchecked(
                Vec::try_from_slice_in(b"ab", alloc).expect("allocation cannot fail"),
            )
        }
        .expect("construction never clones the allocator");

        let error = original.try_clone().unwrap_err();
        assert!(matches!(
            error,
            TryCloneError::Other("clone budget exhausted")
        ));
        // The original is untouched by the failed clone.
        assert_eq!(original.as_bytes_with_nul(), b"ab\0");
    }

    #[test]
    fn cstring_borrows_as_a_cstr() {
        let c_string = cstring_in(b"hi", Global);
        let borrowed: &CStr = c_string.borrow();
        assert_eq!(borrowed.to_bytes_with_nul(), b"hi\0");
        // Methods like `as_ptr` work through the borrow.
        assert_eq!(
            borrowed.as_ptr(),
            c_string.as_bytes_with_nul().as_ptr().cast::<i8>()
        );
    }
}
