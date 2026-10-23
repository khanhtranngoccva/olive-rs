use olive_core::alloc::Allocator;
use olive_core::alloc::AllocatorTryDefault;
use olive_core::alloc_errors::TryReserveError;
use olive_core::try_traits::TryClone;
use olive_core::try_traits::TryDefault;
use olive_core::try_traits::TryDefaultError;

use super::TryVecWithCloneError;
use super::TryVecWithClosureError;
use super::Vec;
use crate::alloc::Global;
use crate::raw_vec::RawVec;

// ---------------------------------------------------------------------------
// Constructors — global allocator
// ---------------------------------------------------------------------------

impl<T> Vec<T, Global> {
    /// Constructs a new, empty `Vec<T>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            raw: RawVec::new(),
            len: 0,
        }
    }

    /// Constructs a new, empty `Vec<T>` with exactly the specified capacity.
    ///
    /// The vector can hold `capacity` elements without reallocating; unlike
    /// the fallible growth methods, this constructor does not allocate more
    /// than requested. If `capacity` is zero, the vector will not allocate.
    ///
    /// ## Determinism
    ///
    /// The capacity of the returned vector is deterministic. For
    /// non-zero-sized `T`, the capacity is exactly `capacity`, so callers can
    /// rely on `self.capacity() == capacity` when asserting on allocations in
    /// tests. For zero-sized `T`, the capacity is always reported as
    /// `usize::MAX` regardless of the request (and no memory is allocated).
    ///
    /// The method does not ask for more allocation memory than needed: if
    /// the allocator returns a buffer larger than the request, the reported
    /// capacity is still clamped to the requested `capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// initial allocation fails.
    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity_in(capacity, Global)
    }

    /// Creates a `Vec<T>` containing `value` cloned `count` times.
    ///
    /// Equivalent to `vec![value; count]` but fully fallible.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    pub fn try_from_elem(value: &T, count: usize) -> Result<Self, TryVecWithCloneError>
    where
        T: TryClone,
    {
        Self::try_from_elem_in(value, count, Global)
    }

    /// Creates a `Vec<T>` by calling `f` `n` times, collecting the results.
    ///
    /// The closure may return an error at any point; construction stops
    /// immediately and the partially-built buffer is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithClosureError<E>`] if either the capacity
    /// reservation fails or the closure returns `Err(e)`.
    pub fn try_from_fn<E, F>(n: usize, mut f: F) -> Result<Self, TryVecWithClosureError<E>>
    where
        F: FnMut() -> Result<T, E>,
    {
        let mut vec = Self::new();
        if n > 0 {
            vec.try_reserve_exact(n)
                .map_err(TryVecWithClosureError::Reserve)?;
        }
        for _ in 0..n {
            match f() {
                Ok(item) => {
                    // SAFETY: Capacity was reserved above, so this cannot fail.
                    unsafe { vec.force_push(item) };
                }
                Err(e) => return Err(TryVecWithClosureError::Closure(e)),
            }
        }
        Ok(vec)
    }
}

// ---------------------------------------------------------------------------
// Constructors — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Creates an empty `Vec` using the given allocator.
    ///
    /// No allocation happens here, so this cannot fail.
    #[must_use]
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            raw: RawVec::new_in(alloc),
            len: 0,
        }
    }

    /// Creates an empty `Vec` with room to hold exactly `capacity` elements.
    ///
    /// The vector can hold `capacity` elements without reallocating; unlike
    /// the fallible growth methods, this constructor does not allocate more
    /// than requested. If `capacity` is zero, the vector will not allocate.
    ///
    /// ## Determinism
    ///
    /// The capacity of the returned vector is deterministic. For
    /// non-zero-sized `T`, the capacity is exactly `capacity`, so callers can
    /// rely on `self.capacity() == capacity` when asserting on allocations in
    /// tests. For zero-sized `T`, the capacity is always reported as
    /// `usize::MAX` regardless of the request (and no memory is allocated).
    ///
    /// The method does not ask for more allocation memory than needed: if
    /// the allocator returns a buffer larger than the request, the reported
    /// capacity is still clamped to the requested `capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// initial allocation fails.
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, TryReserveError> {
        let raw = RawVec::try_with_capacity_in(capacity, alloc)?;
        Ok(Self { raw, len: 0 })
    }

    /// Creates a `Vec<T>` containing `value` cloned `count` times.
    ///
    /// Equivalent to `vec![value; count]` but fully fallible.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    pub fn try_from_elem_in(value: &T, count: usize, alloc: A) -> Result<Self, TryVecWithCloneError>
    where
        T: TryClone,
    {
        let mut vec =
            Self::try_with_capacity_in(count, alloc).map_err(TryVecWithCloneError::Reserve)?;

        for _ in 0..count {
            match value.try_clone() {
                Ok(cloned) => {
                    // SAFETY: Capacity was reserved above, so this cannot fail.
                    unsafe { vec.force_push(cloned) };
                }
                Err(e) => return Err(TryVecWithCloneError::Clone(e)),
            }
        }
        Ok(vec)
    }
}

// ---------------------------------------------------------------------------
// Default construction — global allocator
// ---------------------------------------------------------------------------

impl<T> Default for Vec<T, Global> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Default construction — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: AllocatorTryDefault> TryDefault for Vec<T, A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new_in(A::try_default()?))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};
    use olive_test_helpers::{FailAlloc, FailDefaultAlloc, MaybeUnclonable};

    #[test]
    fn new_is_empty() {
        let v: Vec<i32> = Vec::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn new_in_custom_allocator() {
        let v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        assert!(v.is_empty());
        // No allocation happened yet, so this is still usable.
        assert_eq!(v.as_slice(), []);
    }

    #[test]
    fn try_with_capacity_allocates_room() {
        let v = Vec::<i32>::try_with_capacity(10).expect("alloc ok");
        assert!(v.is_empty());
        // We guarantee determinism for the RawVec, lest we cannot deterministically
        // test its behaviors.
        assert!(v.capacity() == 10);
    }

    #[test]
    fn try_with_capacity_in_allocates_room() {
        let v = Vec::<i32, Global>::try_with_capacity_in(10, Global).expect("alloc ok");
        assert!(v.is_empty());
        // Deterministic RawVec: the capacity is exactly what was asked for.
        assert!(v.capacity() == 10);
    }

    #[test]
    fn with_capacity_in_fail_alloc_reports_alloc() {
        let err: TryReserveError =
            Vec::<i32, FailAlloc>::try_with_capacity_in(8, FailAlloc).expect_err("should fail");
        assert!(err.is_alloc());
    }

    #[test]
    fn try_with_capacity_zero_is_fine() {
        let v = Vec::<i32>::try_with_capacity(0).expect("zero cap ok");
        assert!(v.is_empty());
    }

    #[test]
    fn from_elem_clones_count_times() {
        let v = Vec::try_from_elem(&7i32, 4).expect("elem ok");
        assert_eq!(v.as_slice(), &[7, 7, 7, 7]);
    }

    #[test]
    fn from_elem_zero_count_is_empty() {
        let v = Vec::try_from_elem(&7i32, 0).expect("empty ok");
        assert!(v.is_empty());
    }

    #[test]
    fn from_elem_in_fills_with_clones() {
        let v = Vec::try_from_elem_in(&7i32, 4, Global).expect("elem ok");
        assert_eq!(v.as_slice(), &[7, 7, 7, 7]);
    }

    #[test]
    fn from_elem_in_clonable_fills() {
        let item = MaybeUnclonable::clonable(7i32);
        let v = Vec::try_from_elem_in(&item, 3, Global).expect("clone ok");
        assert_eq!(v.len(), 3);
        assert!(v.as_slice().iter().all(|m| m.inner == 7));
    }

    #[test]
    fn from_elem_in_fail_alloc_reports_reserve() {
        let e = Vec::try_from_elem_in(&1i32, 4, FailAlloc).expect_err("should fail");
        assert!(matches!(e, TryVecWithCloneError::Reserve(r) if r.is_alloc()));
    }

    #[test]
    fn from_elem_in_unclonable_reports_clone_error() {
        let item = MaybeUnclonable::unclonable(42i32);
        let e = Vec::try_from_elem_in(&item, 3, Global).expect_err("clone should fail");
        assert!(matches!(e, TryVecWithCloneError::Clone(_)));
    }

    #[test]
    fn default_matches_new() {
        let v: Vec<i32> = Default::default();
        assert!(v.is_empty());
    }

    #[test]
    fn vec_try_default_succeeds() {
        let v = <Vec<u32> as TryDefault>::try_default().expect("default ok");
        assert!(v.is_empty());
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn vec_try_default_allocator_failure_propagates() {
        let e = <Vec<u32, FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }
}
