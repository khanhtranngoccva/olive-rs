//! Trait implementations for [`Vec`](super::Vec).
use super::Vec;
use crate::borrow::Cow;
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};
use olive_core::alloc::Allocator;
use olive_core::try_traits::try_clone::TryClone;

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
