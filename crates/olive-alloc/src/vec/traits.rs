//! Trait implementations for [`Vec`](super::Vec).
use super::RollbackGuard;
use super::TryVecWithCloneError;
use super::Vec;
use crate::alloc::Global;
use crate::borrow::Cow;
use core::borrow::Borrow;
use core::borrow::BorrowMut;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Deref;
use core::ops::DerefMut;
use core::ops::Index;
use core::ops::IndexMut;
use olive_core::alloc::Allocator;
use olive_core::alloc::AllocatorTryClone;
use olive_core::alloc_errors::TryReserveError;
use olive_core::recovery::ResumableSource;
use olive_core::recovery::Resume;
use olive_core::try_traits::TryCloneError;
use olive_core::try_traits::TryExtend;
use olive_core::try_traits::TryExtendFromSlice;
use olive_core::try_traits::TryFromIterator;
use olive_core::try_traits::try_clone::TryClone;

// ---------------------------------------------------------------------------
// Dereferencing, borrowing and indexing
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Borrow<[T]> for Vec<T, A> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> BorrowMut<[T]> for Vec<T, A> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, A: Allocator> Deref for Vec<T, A> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> DerefMut for Vec<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, A: Allocator> Index<usize> for Vec<T, A> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        // Bounds-checked by the slice machinery.
        &self.as_slice()[index]
    }
}

impl<T, A: Allocator> IndexMut<usize> for Vec<T, A> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.as_mut_slice()[index]
    }
}

impl<T, A: Allocator> Index<core::ops::RangeFull> for Vec<T, A> {
    type Output = [T];

    #[inline]
    fn index(&self, _: core::ops::RangeFull) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> IndexMut<core::ops::RangeFull> for Vec<T, A> {
    #[inline]
    fn index_mut(&mut self, _: core::ops::RangeFull) -> &mut [T] {
        self.as_mut_slice()
    }
}

// ---------------------------------------------------------------------------
// PartialEq / Eq
// ---------------------------------------------------------------------------

impl<T, U, A1: Allocator, A2: Allocator> PartialEq<Vec<U, A2>> for Vec<T, A1>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &Vec<U, A2>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq, A: Allocator> Eq for Vec<T, A> {}

/// Emits a cross-type `PartialEq<$rhs> for $lhs` impl between a `Vec` and a
/// slice, an array (or array reference), or a slice `Cow`, in either direction.
/// Mirrors std's `__impl_slice_eq1`.
macro_rules! __impl_slice_eq1 {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty $(where $ty:ty: $bound:ident)?,) => {
        impl<T, U, $($vars)*> core::cmp::PartialEq<$rhs> for $lhs
        where
            T: PartialEq<U>,
            $($ty: $bound)?
        {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                // Derefs to slice type.
                let lhs: &[_] = &self[..];
                let rhs: &[_] = &other[..];
                lhs == rhs
            }
        }
    };
}

// Cross-type equality against slices, mutable slices, arrays, and slice `Cow`s,
// in both directions. Each reduces to an element-wise compare of the two
// logical element sequences (see `__impl_slice_eq1`).

// TODO: allow comparing types liberally in this manner is
// a potential source of kernel bugs.

// Vec -> slice
__impl_slice_eq1! { [A: Allocator] Vec<T, A>, [U], }
__impl_slice_eq1! { [A: Allocator] Vec<T, A>, &[U], }
__impl_slice_eq1! { [A: Allocator] Vec<T, A>, &mut [U], }

// slice -> Vec
__impl_slice_eq1! { [A: Allocator] [T], Vec<U, A>, }
__impl_slice_eq1! { [A: Allocator] &[T], Vec<U, A>, }
__impl_slice_eq1! { [A: Allocator] &mut [T], Vec<U, A>, }

// Vec -> array
__impl_slice_eq1! { [A: Allocator, const N: usize] Vec<T, A>, [U; N], }
__impl_slice_eq1! { [A: Allocator, const N: usize] Vec<T, A>, &[U; N], }
__impl_slice_eq1! { [A: Allocator, const N: usize] Vec<T, A>, &mut [U; N], }

// array -> Vec
__impl_slice_eq1! { [A: Allocator, const N: usize] [T; N], Vec<U, A>, }
__impl_slice_eq1! { [A: Allocator, const N: usize] &[T; N], Vec<U, A>, }
__impl_slice_eq1! { [A: Allocator, const N: usize] &mut [T; N], Vec<U, A>, }

// Vec/slice/array -> Cow
__impl_slice_eq1! { [A: Allocator] Vec<T, A>, Cow<'_, [U]>  where U: TryClone, }
__impl_slice_eq1! { [] [T], Cow<'_, [U]> where U: TryClone, }
__impl_slice_eq1! { [] &[T], Cow<'_, [U]> where U: TryClone, }
__impl_slice_eq1! { [] &mut [T], Cow<'_, [U]> where U: TryClone, }
__impl_slice_eq1! { [const N: usize] [T; N], Cow<'_, [U]> where U: TryClone, }
__impl_slice_eq1! { [const N: usize] &[T; N], Cow<'_, [U]> where U: TryClone, }
__impl_slice_eq1! { [const N: usize] &mut [T; N], Cow<'_, [U]> where U: TryClone, }

// Cow -> Vec/slice/array
__impl_slice_eq1! { [A: Allocator] Cow<'_, [T]>, Vec<U, A> where T: TryClone, }
__impl_slice_eq1! { [] Cow<'_, [T]>, [U] where T: TryClone, }
__impl_slice_eq1! { [] Cow<'_, [T]>, &[U] where T: TryClone, }
__impl_slice_eq1! { [] Cow<'_, [T]>, &mut [U] where T: TryClone, }
__impl_slice_eq1! { [const N: usize] Cow<'_, [T]>, [U; N] where T: TryClone, }
__impl_slice_eq1! { [const N: usize] Cow<'_, [T]>, &[U; N] where T: TryClone, }
__impl_slice_eq1! { [const N: usize] Cow<'_, [T]>, &mut [U; N] where T: TryClone, }

// ---------------------------------------------------------------------------
// PartialOrd / Ord
// ---------------------------------------------------------------------------

impl<T, A1: Allocator, A2: Allocator> PartialOrd<Vec<T, A2>> for Vec<T, A1>
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Vec<T, A2>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Ord, A: Allocator> Ord for Vec<T, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------

impl<T: Hash, A: Allocator> Hash for Vec<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the length first so that a vector which is a strict prefix of
        // another cannot collide with it. (We can't use the unstable
        // `write_length_prefix`; hashing the raw `usize` achieves the same
        // disambiguation.)
        self.len().hash(state);
        self.iter().for_each(|elem| elem.hash(state));
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl<T, A: Allocator> fmt::Debug for Vec<T, A>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// ---------------------------------------------------------------------------
// Send + Sync
// ---------------------------------------------------------------------------

// SAFETY: `Vec` never hands out references that outlive the buffer, and moving
// a `Vec` moves its whole allocation. Sound iff `T` itself is `Send`/`Sync`.
unsafe impl<T: Send, A: Allocator + Send> Send for Vec<T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for Vec<T, A> {}

// ---------------------------------------------------------------------------
// TryExtend
// ---------------------------------------------------------------------------

impl<T, A: Allocator> TryExtend<T> for Vec<T, A> {
    type Error = TryReserveError;

    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = T>,
    {
        let (head, mut inner, hint) = source.decompose_with_size_hint();
        // Ignore over-reserve.
        let _ = self.try_reserve_adaptive(hint.estimated_total());
        // Push the head first.
        if let Some(head) = head {
            if let Err((head, err)) = self.try_push_give_back(head) {
                return Err((Resume::new(head, inner), err));
            }
        }

        // Push the remainder. While we have spare capacity this is a cheap
        // force_push; once capacity is exhausted (under-hinted or OOM'd) grow
        // one slot at a time, stranding the current element on failure.
        while let Some(next) = inner.next() {
            if self.len == self.capacity() {
                // SAFETY: `len == capacity` holds here.
                if let Err(e) = unsafe { self.raw.try_grow_one() } {
                    return Err((Resume::new(next, inner), e));
                }
            }
            // SAFETY: a spare slot was just confirmed.
            unsafe { self.force_push(next) };
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TryExtendFromSlice
// ---------------------------------------------------------------------------

impl<T, A: Allocator> TryExtendFromSlice<T> for Vec<T, A>
where
    T: TryClone,
{
    type Error = TryVecWithCloneError;

    fn try_extend_from_slice<'s>(&mut self, other: &'s [T]) -> Result<(), (&'s [T], Self::Error)> {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve_adaptive(other.len())
            .map_err(|e| (other, TryVecWithCloneError::Reserve(e)))?;
        let mut i = 0usize;
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    // SAFETY: capacity was reserved above for all of `other`.
                    unsafe { self.force_push(cloned) };
                    #[allow(clippy::arithmetic_side_effects, reason = "i <= other.len()")]
                    {
                        i += 1;
                    }
                }
                Err(e) => return Err((&other[i..], TryVecWithCloneError::Clone(e))),
            }
        }
        Ok(())
    }
}

impl<T, A: Allocator> Vec<T, A> {
    /// Extends the vector with each element of `other`, cloned via
    /// [`TryClone`].
    ///
    /// If a clone fails mid-way, the vector is truncated back to its length at
    /// the start of the call so no partially-appended elements remain.
    ///
    /// # Errors
    ///
    /// Returns [`TryVecWithCloneError`] on a reservation or clone failure.
    pub fn try_extend_from_slice_with_rollback(
        &mut self,
        other: &[T],
    ) -> Result<(), TryVecWithCloneError>
    where
        T: TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve_adaptive(other.len())
            .map_err(TryVecWithCloneError::Reserve)?;
        let len_before = self.len;
        let guard = RollbackGuard(&raw mut *self, len_before);
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    // SAFETY: capacity was reserved above for all of `other`.
                    unsafe { self.force_push(cloned) };
                }
                Err(e) => {
                    return Err(TryVecWithCloneError::Clone(e));
                }
            }
        }
        core::mem::forget(guard);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TryClone
// ---------------------------------------------------------------------------

impl<T: TryClone, A: AllocatorTryClone> TryClone for Vec<T, A> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = Self::try_with_capacity_in(self.len, self.raw.allocator().try_clone()?)?;
        for elem in self.iter() {
            match elem.try_clone() {
                // SAFETY: capacity was reserved above for every element.
                Ok(cloned) => unsafe { out.force_push(cloned) },
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TryFromIterator: any iterator, Global allocator
// ---------------------------------------------------------------------------

impl<T> TryFromIterator<T> for Vec<T, Global> {
    type Error = TryReserveError;

    fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Self::Error> {
        Self::try_from_iter_in(iter, Global)
    }
}

impl<T, A: Allocator> Vec<T, A> {
    /// Fallibly collects an iterator into a [`Vec<T>`] on the given allocator,
    /// using the size hint to pre-allocate when possible.
    ///
    /// This is the allocator-aware backend for
    /// [`TryFromIterator`](olive_core::try_traits::try_from_iterator::TryFromIterator).
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving capacity for an incoming element
    /// fails.
    pub fn try_from_iter_in<I: IntoIterator<Item = T>>(
        iter: I,
        alloc: A,
    ) -> Result<Self, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let mut vec = Self::new_in(alloc);
        let cap = upper.unwrap_or(lower);
        let _ = vec.try_reserve_adaptive(cap);
        for item in iter {
            // The iterator may yield more elements than its hint promised.
            if vec.len == vec.capacity() {
                // SAFETY: `len == capacity` holds here.
                unsafe { vec.raw.try_grow_one()? };
            }
            // SAFETY: we just confirmed there is a spare slot.
            unsafe { vec.force_push(item) };
        }
        Ok(vec)
    }
}
