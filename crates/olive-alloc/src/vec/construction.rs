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
