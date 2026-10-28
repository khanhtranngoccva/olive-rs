//! Macros for building Olive [`Vec`]s.
//!
//! [`try_vec!`] is the fallible counterpart of the standard library's `vec!`:
//! the same three invocation shapes, with every form that can allocate
//! returning a `Result` instead of panicking or aborting on out-of-memory.

/// The fallible analogue of the standard library's `vec!` macro.
///
/// Supports the same invocation shapes std's `vec!` does, each backed by the
/// global allocator:
///
/// - `try_vec![]` — an empty vector. This never allocates, so the arm is
///   infallible and yields the vector itself rather than a `Result`.
/// - `try_vec![elem; n]` — a vector holding `n` clones of `elem`, via
///   [`Vec::try_from_elem`](crate::vec::Vec::try_from_elem). As in std, the
///   element expression is evaluated exactly once and then cloned, and the
///   element type must implement
///   [`TryClone`](olive_core::try_traits::TryClone) (std demands `Clone` for
///   the same form). Expands to
///   `Result<Vec<T, Global>, TryVecWithCloneError>`.
/// - `try_vec![a, b, c]` — a vector holding exactly the listed elements,
///   collected via
///   [`Vec::try_from_iter_in`](crate::vec::Vec::try_from_iter_in) with the global
///   allocator. A trailing comma is allowed. Expands to
///   `Result<Vec<T, Global>, TryReserveError>`.
///
/// # Examples
///
/// ```
/// use olive_alloc::try_vec;
/// use olive_alloc::vec::Vec;
///
/// let empty: Vec<i32> = try_vec![];
/// let repeated = try_vec![7; 3].expect("allocation succeeds");
/// let listed = try_vec![1, 2, 3].expect("allocation succeeds");
///
/// assert!(empty.is_empty());
/// assert_eq!(repeated.as_slice(), &[7, 7, 7]);
/// assert_eq!(listed.as_slice(), &[1, 2, 3]);
/// ```
#[macro_export]
macro_rules! try_vec {
    () => {
        $crate::vec::Vec::new()
    };

    ($elem:expr; $n:expr) => {
        $crate::vec::Vec::try_from_elem(&$elem, $n)
    };

    ($($x:expr),+ $(,)?) => {
        $crate::vec::Vec::try_from_iter_in([$($x),+], $crate::alloc::Global)
    };
}

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::try_vec;
    use crate::vec::TryVecWithCloneError;
    use crate::vec::Vec;
    use olive_core::alloc_errors::TryReserveError;
    use olive_test_helpers::MaybeUnclonable;

    #[test]
    fn empty_is_infallible_and_unallocated() {
        // Compile-time check: the empty form is the vector itself, not a
        // `Result`, since nothing is allocated.
        let v: Vec<i32> = try_vec![];
        assert!(v.is_empty());
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn listed_elements_collect_in_order() {
        let v = try_vec![1, 2, 3].expect("alloc ok");
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn listed_elements_accept_trailing_comma() {
        let v = try_vec![1, 2, 3,].expect("alloc ok");
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn listed_single_element() {
        let v = try_vec![9].expect("alloc ok");
        assert_eq!(v.as_slice(), &[9]);
    }

    #[test]
    fn listed_error_type_is_try_reserve_error() {
        // Compile-time check of the documented error type.
        let r: Result<Vec<i32>, TryReserveError> = try_vec![1, 2];
        assert_eq!(r.expect("alloc ok").as_slice(), &[1, 2]);
    }

    #[test]
    fn listed_element_type_is_inferred() {
        let v: Vec<i32> = try_vec![1, 2, 3].expect("alloc ok");
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn repeated_elem_clones_n_times() {
        let v = try_vec![7; 4].expect("alloc ok");
        assert_eq!(v.as_slice(), &[7, 7, 7, 7]);
    }

    #[test]
    fn repeated_elem_zero_count_is_empty() {
        let v = try_vec![7; 0].expect("alloc ok");
        assert!(v.is_empty());
    }

    #[test]
    fn repeated_elem_accepts_runtime_count() {
        let n = 3 + 1;
        let v = try_vec![5; n].expect("alloc ok");
        assert_eq!(v.as_slice(), &[5, 5, 5, 5]);
    }

    #[test]
    fn repeated_elem_evaluates_element_once() {
        // Matches std: the element expression is evaluated exactly once and
        // cloned for the remaining slots.
        let mut calls = 0;
        let mut make = || {
            calls += 1;
            42
        };
        let v = try_vec![make(); 3].expect("alloc ok");
        assert_eq!(calls, 1);
        assert_eq!(v.as_slice(), &[42, 42, 42]);
    }

    #[test]
    fn repeated_elem_keeps_original_intact() {
        let mut s = crate::string::String::try_from_str("ab").expect("alloc ok");
        let v = try_vec![s; 2].expect("alloc ok");
        s.try_push_str("c").expect("push ok");
        assert_eq!(v.len(), 2);
        assert_eq!(&*v[0], "ab");
        assert_eq!(&*v[1], "ab");
        // The original was only borrowed for cloning, so it is still usable.
        assert_eq!(&*s, "abc");
    }

    #[test]
    fn repeated_error_type_is_try_vec_with_clone_error() {
        // Compile-time check of the documented error type.
        let r: Result<Vec<i32>, TryVecWithCloneError> = try_vec![1; 2];
        assert_eq!(r.expect("alloc ok").as_slice(), &[1, 1]);
    }

    #[test]
    fn repeated_unclonable_element_reports_clone_error() {
        let item = MaybeUnclonable::unclonable(42i32);
        let e = try_vec![item; 3].expect_err("clone should fail");
        assert!(matches!(e, TryVecWithCloneError::Clone(_)));
    }

    #[test]
    fn repeated_zero_count_never_clones() {
        // Mirrors `try_from_elem`: nothing is produced, so nothing is cloned,
        // and an element that cannot clone is never even asked to.
        let item = MaybeUnclonable::unclonable(42i32);
        let v = try_vec![item; 0].expect("alloc ok");
        assert!(v.is_empty());
    }
}
