//! Trait implementations for [`Vec`](super::Vec).
use super::Vec;
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};
use olive_core::alloc::Allocator;

// ---------------------------------------------------------------------------
// PartialEq / Eq
// ---------------------------------------------------------------------------

impl<T: PartialEq, A: Allocator> PartialEq for Vec<T, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq, A: Allocator> Eq for Vec<T, A> {}

/// Emits a cross-type `PartialEq<RHS> for LHS` impl between a `Vec` and a
/// slice, array, or vector type. Mirrors std's `__impl_slice_eq1`.
///
/// The RHS is indexed with `..` (uniformly valid for every supported shape —
/// `Vec`, `[U]`, `[U; N]`, `&[U]`, `&mut [U]`), so the comparison reduces to a
/// plain slice-vs-slice element-wise compare. A length check runs first so
/// unequal-length operands short-circuit without touching any elements.
///
/// # Usage
///
/// ```ignore
/// __impl_vec_eq1! { [] Vec<T, A>, Vec<U, B>, }
/// __impl_vec_eq1! { [const N: usize] Vec<T, A>, [U; N], }
/// ```
macro_rules! __impl_vec_eq1 {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty,) => {
        impl<T, U, A: Allocator, $($vars)*> core::cmp::PartialEq<$rhs> for $lhs
        where
            T: PartialEq<U>,
        {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                if self.len() != other.len() {
                    return false;
                }
                let lhs: &[T] = self;
                let rhs: &[U] = &other[..];
                lhs == rhs
            }
        }
    };
}

// Cross-type equality against slices, arrays, and vectors. Each reduces to an
// element-wise compare of the two logical sequences (see `__impl_vec_eq1`).
__impl_vec_eq1! { [] Vec<T, A>, &[U], }
__impl_vec_eq1! { [] Vec<T, A>, &mut [U], }
__impl_vec_eq1! { [const N: usize] Vec<T, A>, [U; N], }
__impl_vec_eq1! { [const N: usize] Vec<T, A>, &[U; N], }
__impl_vec_eq1! { [const N: usize] Vec<T, A>, &mut [U; N], }

// ---------------------------------------------------------------------------
// PartialOrd / Ord
// ---------------------------------------------------------------------------

impl<T: PartialOrd, A: Allocator> PartialOrd for Vec<T, A> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
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
