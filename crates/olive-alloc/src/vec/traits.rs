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
use core::slice::SliceIndex;
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

impl<T, I, A> Index<I> for Vec<T, A>
where
    I: SliceIndex<[T]>,
    A: Allocator,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        Index::index(&**self, index)
    }
}

impl<T, I, A> IndexMut<I> for Vec<T, A>
where
    I: SliceIndex<[T]>,
    A: Allocator,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        IndexMut::index_mut(&mut **self, index)
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

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::borrow::Cow;
    use olive_core::try_traits::try_clone::TryCloneError;
    use olive_core::try_traits::try_collect::TryCollect;
    use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
    use olive_core::try_traits::try_from_iterator::TryFromIterator;
    use olive_test_helpers::{
        BudgetedAlloc, CloneBudget, CountingAllocator, FailAlloc, FlakyClone,
        FlakyCloneTrackedItem, Ledger, PanicArmer,
    };
    use std::format;
    use std::sync::Arc;

    /// Builds an olive `Vec` from literals without relying on a `vec!` macro.
    fn mk_vec(items: &[i32]) -> Vec<i32, Global> {
        let mut v = Vec::new();
        for x in items {
            v.try_push(*x).unwrap();
        }
        v
    }

    /// An iterator whose `size_hint` advertises a far larger upper bound than the
    /// number of elements it actually yields — the canonical "over-reporting" case.
    struct OverhintedIter {
        remaining: usize,
        advertised_upper: usize,
    }

    impl Iterator for OverhintedIter {
        type Item = i32;

        fn next(&mut self) -> Option<i32> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            let value = (self.advertised_upper - self.remaining) as i32;
            Some(value)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (
                self.remaining.min(self.advertised_upper),
                Some(self.advertised_upper),
            )
        }
    }

    #[test]
    fn deref_as_slice() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(2).unwrap();
        let s: &[i32] = &v;
        assert_eq!(s, &[1, 2]);
    }

    #[test]
    fn deref_mut_as_slice() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(2).unwrap();
        let s: &mut [i32] = &mut v;
        s[0] = 42;
        assert_eq!(v.as_slice(), &[42, 2]);
    }

    #[test]
    fn index_get_and_set() {
        let mut v = Vec::new();
        for i in 0..3 {
            v.try_push(i).unwrap();
        }
        assert_eq!(v[1], 1);
        v[1] = 99;
        assert_eq!(v.as_slice(), &[0, 99, 2]);
    }

    #[test]
    fn range_full_index() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(2).unwrap();
        let all: &[i32] = &v[..];
        assert_eq!(all, &[1, 2]);
    }

    #[test]
    fn range_partial_index() {
        let mut v = Vec::new();
        for i in 0..4 {
            v.try_push(i).unwrap();
        }
        // Any `SliceIndex` works directly on the Vec: `usize`, `RangeFull`
        // and the partial ranges alike.
        let mid: &[i32] = &v[1..3];
        assert_eq!(mid, &[1, 2]);
        let tail: &[i32] = &v[2..];
        assert_eq!(tail, &[2, 3]);
        let head: &mut [i32] = &mut v[..2];
        head[0] = 9;
        assert_eq!(v.as_slice(), &[9, 1, 2, 3]);
        let first_two: &[i32] = &v[..=1];
        assert_eq!(first_two, &[9, 1]);
    }

    #[test]
    fn vec_partial_eq_same_len_equal() {
        let a = mk_vec(&[1, 2, 3]);
        let b = mk_vec(&[1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn vec_partial_eq_differs_at_element() {
        let a = mk_vec(&[1, 2, 3]);
        let b = mk_vec(&[1, 2, 4]);
        assert_ne!(a, b);
    }

    #[test]
    fn vec_partial_eq_different_lengths() {
        let a = mk_vec(&[1, 2, 3]);
        let b = mk_vec(&[1, 2]);
        assert_ne!(a, b);
    }

    #[test]
    fn vec_partial_eq_cross_type_vs_slice() {
        let v = mk_vec(&[1, 2, 3]);
        let s: &[i32] = &[1, 2, 3];
        let s_mut: &mut [i32] = &mut [1, 2, 3];
        assert_eq!(v, s);
        assert_eq!(v, s_mut);
        assert_eq!(v, *s);

        let ne = mk_vec(&[1, 2, 4]);
        assert_ne!(ne, s);
        assert_ne!(ne, s_mut);
        assert_ne!(ne, *s);
    }

    #[test]
    fn vec_partial_eq_reverse_vs_slice() {
        let v = mk_vec(&[1, 2, 3]);
        let s: &[i32] = &[1, 2, 3];
        let s_mut: &mut [i32] = &mut [1, 2, 3];

        assert_eq!(s, v);
        assert_eq!(s_mut, v);
        assert_eq!(*s, v);

        let ne: &[i32] = &[1, 2, 4];
        let ne_mut: &mut [i32] = &mut [1, 2, 4];

        assert_ne!(ne, v);
        assert_ne!(ne_mut, v);
        assert_ne!(*ne, v);
    }

    #[test]
    fn vec_partial_eq_cross_type_vs_array() {
        let v = mk_vec(&[1, 2, 3]);
        let mut a: [i32; 3] = [1, 2, 3];
        assert_eq!(v, a);
        assert_eq!(v, &a);
        assert_eq!(v, &mut a);

        let ne = mk_vec(&[1, 2, 4]);
        assert_ne!(ne, a);
        assert_ne!(ne, &a);
        assert_ne!(ne, &mut a);
    }

    #[test]
    fn vec_partial_eq_reverse_vs_array() {
        let v = mk_vec(&[1, 2, 3]);
        let mut a: [i32; 3] = [1, 2, 3];
        assert_eq!(a, v);
        assert_eq!(&a, v);
        assert_eq!(&mut a, v);

        let mut ne: [i32; 3] = [1, 2, 4];
        assert_ne!(ne, v);
        assert_ne!(&ne, v);
        assert_ne!(&mut ne, v);

        // Length and element mismatches compare unequal.
        assert_ne!(a, mk_vec(&[1, 2, 4]));
        assert_ne!(a, mk_vec(&[1, 2]));
    }

    #[test]
    fn vec_partial_eq_cross_allocator() {
        use olive_test_helpers::allocators::ByteCapAlloc;

        let a = mk_vec(&[1, 2, 3]);
        let mut b: Vec<i32, ByteCapAlloc> =
            Vec::try_with_capacity_in(3, ByteCapAlloc::new(64)).expect("should create vector");
        for x in [1, 2, 3] {
            b.try_push(x).unwrap();
        }
        assert_eq!(a, b);
        assert_eq!(b, a);

        let mut c: Vec<i32, ByteCapAlloc> =
            Vec::try_with_capacity_in(3, ByteCapAlloc::new(64)).expect("should create vector");
        for x in [1, 2, 4] {
            c.try_push(x).unwrap();
        }
        assert_ne!(a, c);
    }

    #[test]
    fn cow_slice_eq_vec_and_slices() {
        let src: &[i32] = &[1, 2, 3];

        let owned: Cow<'_, [i32]> = Cow::Owned(mk_vec(&[1, 2, 3]));
        let borrowed: Cow<'_, [i32]> = Cow::Borrowed(src);

        let v = mk_vec(&[1, 2, 3]);
        assert_eq!(borrowed, v);
        assert_eq!(v, borrowed);
        assert_eq!(owned, v);
        assert_eq!(v, owned);

        let other: &[i32] = &[1, 2, 3];
        assert_eq!(borrowed, other);
        assert_eq!(other, borrowed);
        assert_eq!(owned, other);
        assert_eq!(other, owned);
        let other_mut: &mut [i32] = &mut [1, 2, 3];
        assert_eq!(borrowed, other_mut);
        assert_eq!(other_mut, borrowed);
        assert_eq!(owned, other_mut);
        assert_eq!(other_mut, owned);
        let other_arr: [i32; 3] = [1, 2, 3];
        assert_eq!(borrowed, other_arr);
        assert_eq!(other_arr, borrowed);
        assert_eq!(owned, other_arr);
        assert_eq!(other_arr, owned);

        // Length and element mismatches compare unequal.
        assert_ne!(borrowed, mk_vec(&[1, 2]));
        assert_ne!(borrowed, mk_vec(&[1, 2, 4]));
        assert_ne!(owned, mk_vec(&[1, 2]));
        assert_ne!(owned, mk_vec(&[1, 2, 4]));
        assert_ne!(mk_vec(&[1, 2]), borrowed);
        assert_ne!(mk_vec(&[1, 2, 4]), borrowed);
        assert_ne!(mk_vec(&[1, 2]), owned);
        assert_ne!(mk_vec(&[1, 2, 4]), owned);
    }

    #[test]
    fn vec_partial_ord_cross_allocator() {
        use core::cmp::{Ordering, PartialOrd};
        use olive_test_helpers::allocators::ByteCapAlloc;

        let a = mk_vec(&[1, 2]);
        let mut b: Vec<i32, ByteCapAlloc> =
            Vec::try_with_capacity_in(3, ByteCapAlloc::new(64)).expect("should create vector");
        for x in [1, 2, 3] {
            b.try_push(x).unwrap();
        }
        assert_eq!(PartialOrd::partial_cmp(&a, &b), Some(Ordering::Less));
        assert_eq!(PartialOrd::partial_cmp(&b, &a), Some(Ordering::Greater));

        let mut equal: Vec<i32, ByteCapAlloc> =
            Vec::try_with_capacity_in(2, ByteCapAlloc::new(64)).expect("should create vector");
        for x in [1, 2] {
            equal.try_push(x).unwrap();
        }
        assert_eq!(PartialOrd::partial_cmp(&a, &equal), Some(Ordering::Equal));
    }

    #[test]
    fn vec_partial_ord_less_than_by_prefix_then_length() {
        use core::cmp::{Ordering, PartialOrd};
        let a = mk_vec(&[1, 2]);
        let b = mk_vec(&[1, 2, 3]);
        // Explicit `PartialOrd` dispatch — operators would route through `Ord`.
        assert_eq!(PartialOrd::partial_cmp(&a, &b), Some(Ordering::Less));
        assert_eq!(PartialOrd::partial_cmp(&b, &a), Some(Ordering::Greater));
    }

    #[test]
    fn vec_partial_ord_lexicographic() {
        use core::cmp::{Ordering, PartialOrd};
        let a = mk_vec(&[1, 2, 3]);
        let b = mk_vec(&[1, 3, 2]);
        assert_eq!(PartialOrd::partial_cmp(&a, &b), Some(Ordering::Less));
        assert_eq!(PartialOrd::partial_cmp(&b, &a), Some(Ordering::Greater));
    }

    #[test]
    fn vec_ord_less_than_by_prefix_then_length() {
        use core::cmp::{Ord, Ordering};
        let a = mk_vec(&[1, 2]);
        let b = mk_vec(&[1, 2, 3]);
        assert_eq!(Ord::cmp(&a, &b), Ordering::Less);
        assert_eq!(Ord::cmp(&b, &a), Ordering::Greater);
    }

    #[test]
    fn vec_ord_lexicographic() {
        use core::cmp::{Ord, Ordering};
        let a = mk_vec(&[1, 2, 3]);
        let b = mk_vec(&[1, 3, 2]);
        assert_eq!(Ord::cmp(&a, &b), Ordering::Less);
        assert_eq!(Ord::cmp(&b, &a), Ordering::Greater);
    }

    #[test]
    fn vec_hash_consistent_for_equal_vectors() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let a = mk_vec(&[1, 2, 3]);
        let b = mk_vec(&[1, 2, 3]);

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn vec_hash_differs_for_different_vectors() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let a = mk_vec(&[1, 2, 3]);
        let b = mk_vec(&[1, 2, 4]);

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish());
    }

    #[test]
    fn debug_format_lists_elements() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(2).unwrap();
        let s = format!("{v:?}");
        assert_eq!(s, "[1, 2]");
    }

    #[test]
    fn try_extend_trait_success() {
        let mut v: Vec<i32> = Vec::new();
        v.try_extend(0..3).expect("extend ok");
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn try_extend_trait_oom_carries_resume() {
        let mut v: Vec<i32, FailAlloc> = Vec::new_in(FailAlloc);
        let (resume, err) = v.try_extend(0..3).expect_err("oom");
        assert!(err.is_alloc());
        // Nothing was committed (because the vec has no capacity yet).
        assert!(v.is_empty());
        // The resume carries the unconsumed remainder.
        let remaining: std::vec::Vec<i32> = resume.into_remainder().collect();
        assert!(!remaining.is_empty());
    }

    #[test]
    fn try_extend_retry_recovers() {
        // Simulate: first attempt fails partway, retry with the resume completes.
        // We can't force a real mid-way OOM easily, so verify the resume round-trips
        // through a second call on a healthy allocator.
        let mut v: Vec<i32> = Vec::new();
        let start = Resume::compose(Some(0), 1..4);
        v.try_extend(start).expect("retry ok");
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn try_extend_overhint_falls_back_to_incremental_growth() {
        // An iterator that advertises a huge upper bound but yields only a few
        // elements. The upfront batch reserve is too large for the byte-capped
        // allocator and fails silently; extend then proceeds via per-element
        // reserves that each stay within the cap.
        use olive_test_helpers::allocators::ByteCapAlloc;

        let alloc = ByteCapAlloc::new(64);
        let mut v: Vec<i32, _> = Vec::new_in(alloc.clone());

        v.try_extend(OverhintedIter {
            remaining: 4,
            advertised_upper: 10000,
        })
        .map_err(|(_, e)| e)
        .expect("incremental growth should succeed");
        assert_eq!(v.as_slice(), &[9997, 9998, 9999, 10000]);
    }

    #[test]
    fn try_extend_underhint_grows_mid_iteration() {
        // An iterator whose size hint severely underestimates the true count
        // forces mid-extension growth once the initially reserved capacity is
        // exhausted.
        #[derive(Debug)]
        struct Underhinted(core::ops::Range<i32>);
        impl Iterator for Underhinted {
            type Item = i32;
            fn next(&mut self) -> Option<i32> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(1))
            }
        }

        let mut v: Vec<i32> = Vec::new();
        v.try_push(99).unwrap();
        v.try_extend(Underhinted(0..8)).expect("extend ok");
        assert_eq!(v.as_slice(), &[99, 0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn try_extend_from_slice_trait_success() {
        let mut v: Vec<i32> = Vec::new();
        v.try_extend_from_slice(&[7, 8]).expect("ok");
        assert_eq!(v.as_slice(), &[7, 8]);
    }

    #[test]
    fn try_extend_from_slice_uses_adaptive_reserve_under_byte_cap() {
        use olive_test_helpers::allocators::ByteCapAlloc;
        let mut v: Vec<i32, ByteCapAlloc> =
            Vec::try_with_capacity_in(15, ByteCapAlloc::new(64)).expect("should create vector");
        v.try_extend(0..15).expect("capacity is sufficient");
        v.try_extend_from_slice(&[15])
            .expect("adaptive reserve should absorb the over-provisioned request");
        assert_eq!(v.len(), 16);
        assert_eq!(v.last(), Some(&15));
    }

    #[test]
    fn try_extend_from_slice_trait_returns_remainder_on_clone_fail() {
        let mut v: Vec<FlakyClone> = Vec::new();
        let mut src: Vec<FlakyClone> = Vec::new();
        src.try_push(FlakyClone::new(2)).unwrap();
        src.try_push(FlakyClone::at(1, 2)).unwrap();
        src.try_push(FlakyClone::at(2, 2)).unwrap(); // fails to clone
        let (rest, e) = v
            .try_extend_from_slice(src.as_slice())
            .expect_err("clone fail");
        assert!(matches!(
            e,
            TryVecWithCloneError::Clone(TryCloneError::Other(_))
        ));
        // Remainder begins at the failing element.
        assert_eq!(rest.len(), 1);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn extend_from_slice_with_rollback_success() {
        let mut v = Vec::new();
        v.try_push(1).unwrap();
        v.try_extend_from_slice_with_rollback(&[2, 3])
            .expect("extend ok");
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn extend_from_slice_rolls_back_on_clone_failure() {
        let ledger = Arc::new(Ledger::new());
        let budget = CloneBudget::new(2);

        // One live seed in the destination (id 0).
        let mut v: Vec<FlakyCloneTrackedItem> = Vec::new();
        v.try_push(FlakyCloneTrackedItem::construct(&ledger, budget.share()))
            .unwrap();

        // Source slice of three payloads (ids 1, 2, 3). Extending clones them in
        // order; the budget allows exactly two successful clones (of ids 1 and 2),
        // then the third clone (of id 3) fails → rollback discards the two appends.
        let mut fv: Vec<FlakyCloneTrackedItem> = Vec::new();
        for _ in 0..3 {
            fv.try_push(FlakyCloneTrackedItem::construct(&ledger, budget.share()))
                .unwrap();
        }

        let e = v
            .try_extend_from_slice_with_rollback(fv.as_slice())
            .expect_err("clone fail");
        assert!(matches!(e, TryVecWithCloneError::Clone(_)));

        // Destination rolled back to just its seed.
        assert_eq!(v.len(), 1);
        // The two transient copies of source ids 1 and 2 were created as NEW ids
        // (4 and 5) and dropped during rollback. Still alive at this point: id 0
        // (seed, in v) and ids 1, 2, 3 (the sources, in fv). No double-free, and
        // exactly the two transients have been dropped.
        assert_eq!(ledger.live_ids(), [0, 1, 2, 3]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.total_allocated(), 6);
        assert_eq!(
            ledger.drop_count(4),
            1,
            "transient copy of source[0] dropped once"
        );
        assert_eq!(
            ledger.drop_count(5),
            1,
            "transient copy of source[1] dropped once"
        );

        // Tear down both vecs: seed (0) + sources (1,2,3) all drop → 5 more, total 7.
        drop(fv);
        drop(v);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..6));
    }

    #[test]
    fn extend_from_slice_with_rollback_is_panic_safe() {
        let ledger = Arc::new(Ledger::new());
        let armer = Arc::new(PanicArmer::new());

        // One ticket: the first source clone succeeds, the second fails, so
        // the rollback has exactly one transient to drop — and that drop is
        // the one that panics.
        let budget = CloneBudget::new(1);

        /// A tracked payload whose clone shares the armer and whose drop
        /// panics exactly once while armed.
        struct RollbackPanicPayload {
            id: u32,
            ledger: Arc<Ledger>,
            armer: Arc<PanicArmer>,
            budget: CloneBudget,
        }

        impl RollbackPanicPayload {
            fn construct(
                ledger: &Arc<Ledger>,
                armer: &Arc<PanicArmer>,
                budget: &CloneBudget,
            ) -> Self {
                let id = ledger.allocate();
                ledger.register(id);
                Self {
                    id,
                    ledger: ledger.clone(),
                    armer: armer.clone(),
                    budget: budget.share(),
                }
            }
        }

        impl TryClone for RollbackPanicPayload {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                if !self.budget.try_consume() {
                    return Err(TryCloneError::Other("rollback budget exhausted"));
                }
                let id = self.ledger.allocate();
                self.ledger.register(id);
                Ok(Self {
                    id,
                    ledger: self.ledger.clone(),
                    armer: self.armer.clone(),
                    budget: self.budget.share(),
                })
            }
        }

        impl Drop for RollbackPanicPayload {
            fn drop(&mut self) {
                self.ledger.unregister(self.id);
                if self.armer.fetch_and_disarm() {
                    panic!("forced panic in rollback drop");
                }
            }
        }

        let mut v: Vec<RollbackPanicPayload> = Vec::new();
        v.try_push(RollbackPanicPayload::construct(&ledger, &armer, &budget))
            .unwrap();
        let mut source: Vec<RollbackPanicPayload> = Vec::new();
        for _ in 0..3 {
            source
                .try_push(RollbackPanicPayload::construct(&ledger, &armer, &budget))
                .unwrap();
        }

        armer.arm();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = v.try_extend_from_slice_with_rollback(source.as_slice());
        }));

        assert!(
            result.is_err(),
            "the rollback must surface the panicking drop"
        );
        // The destination rolled back to just its original.
        assert_eq!(v.len(), 1);
        drop(source);
        drop(v);
        // Seed and sources dropped exactly once; the transient's own drop ran
        // before panicking; nothing was dropped twice.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    #[test]
    fn try_clone_vec_basic() {
        let mut v: Vec<i32> = Vec::new();
        for i in 0..5 {
            v.try_push(i).unwrap();
        }
        let c = v.try_clone().expect("clone ok");
        assert_eq!(c.as_slice(), &[0, 1, 2, 3, 4]);
        // Independent buffer.
        assert_ne!(v.as_ptr(), c.as_ptr());
    }

    #[test]
    fn try_clone_empty_vec() {
        let v: Vec<i32> = Vec::new();
        let c = v.try_clone().expect("clone ok");
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_element_failure_leaves_source_intact() {
        let ledger = Arc::new(Ledger::new());
        // Two tickets: the clones of the first two elements succeed, the
        // third fails mid-way through building the copy.
        let budget = CloneBudget::new(2);
        let mut v: Vec<FlakyCloneTrackedItem> = Vec::new();
        for _ in 0..3 {
            v.try_push(FlakyCloneTrackedItem::construct(&ledger, budget.share()))
                .unwrap();
        }
        let e = v.try_clone().unwrap_err();
        assert!(matches!(e, TryCloneError::Other("budget exhausted")));
        // The source is untouched; the two transient clones were dropped once.
        assert_eq!(v.len(), 3);
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        assert_eq!(ledger.drop_count(5), 0);
        drop(v);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    #[test]
    fn try_clone_reports_allocation_failure() {
        let alloc = BudgetedAlloc::new(4);
        let mut v: Vec<i32, BudgetedAlloc> = Vec::new_in(alloc.clone());
        for i in 0..3 {
            v.try_push(i).unwrap();
        }
        // Drain the shared budget: the clone's own reservation must fail.
        alloc.drain();
        let e = v.try_clone().unwrap_err();
        assert!(matches!(e, TryCloneError::Reserve(r) if r.is_alloc()));
        // The source is untouched.
        assert_eq!(v.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn try_collect_uses_size_hint() {
        let v: Vec<i32> = (0..5).try_collect().expect("collect ok");
        assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn try_collect_grows_past_hint() {
        // An iterator that lies about its size hint (claims small, yields more).
        struct LyingIter {
            n: usize,
            yielded: usize,
        }
        impl Iterator for LyingIter {
            type Item = i32;
            fn next(&mut self) -> Option<i32> {
                if self.yielded < self.n {
                    let v = self.yielded as i32;
                    self.yielded += 1;
                    Some(v)
                } else {
                    None
                }
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(0)) // Deliberately under-reports.
            }
        }
        let v: Vec<i32> = LyingIter { n: 10, yielded: 0 }
            .try_collect()
            .expect("grow ok");
        assert_eq!(
            v.as_slice(),
            std::vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].as_slice()
        );
    }

    #[test]
    fn try_from_iterator_collects() {
        let v: Vec<i32> = TryFromIterator::try_from_iter(0..4).expect("iter ok");
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn try_from_iter_in_fail_alloc_reports_reserve() {
        let e = Vec::<i32, FailAlloc>::try_from_iter_in(0..3, FailAlloc).expect_err("should fail");
        assert!(e.is_alloc());
    }

    #[test]
    fn try_from_iter_in_builds_vec_on_target_allocator() {
        let alloc = CountingAllocator::new();
        let v: Vec<i32, _> = Vec::try_from_iter_in(1..=4, &alloc).expect("create OK");
        assert_eq!(alloc.allocations(), 1);
        assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
        drop(v);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn try_from_iter_in_fails_when_allocation_fails() {
        use olive_core::alloc_errors::TryReserveErrorKind;
        // The `Global`-scoped trait impl can't target a failing allocator, so drive
        // the same code path through the `_in` seam with one that does.
        let res: Result<Vec<i32, FailAlloc>, _> = Vec::try_from_iter_in(1..=4, FailAlloc);
        let r = res.expect_err("allocation should fail");
        assert!(matches!(r.kind(), TryReserveErrorKind::AllocError { .. }))
    }

    #[test]
    fn try_from_iter_in_absorbs_overhint_and_grows_incrementally() {
        use olive_test_helpers::allocators::ByteCapAlloc;
        // The iterator advertises an upper bound of 1_000 but yields only 4
        // elements (16 bytes).
        let iter = OverhintedIter {
            remaining: 4,
            advertised_upper: 1_000,
        };
        let v: Vec<i32, ByteCapAlloc> = Vec::try_from_iter_in(iter, ByteCapAlloc::new(64))
            .expect("over-hint should be absorbed via incremental growth");
        assert_eq!(v.as_slice(), &[997, 998, 999, 1_000]);
    }

    #[test]
    fn try_from_iter_in_still_fails_on_genuine_oom_under_byte_cap() {
        use olive_test_helpers::allocators::ByteCapAlloc;
        let iter = std::iter::repeat(1i32).take(20);
        let res: Result<Vec<i32, ByteCapAlloc>, _> =
            Vec::try_from_iter_in(iter, ByteCapAlloc::new(64));
        let e = res.expect_err("genuine OOM should surface");
        assert!(e.is_alloc())
    }
}
