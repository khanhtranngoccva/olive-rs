//! Internal macros for [`VecDeque`](super::VecDeque).

/// Emits a cross-type `PartialEq<RHS> for LHS` impl between a `VecDeque` and a
/// slice, array, or vector type. Mirrors std's `__impl_slice_eq1`.
///
/// The deque's circular buffer exposes two slices via [`as_slices`](super::VecDeque::as_slices);
/// the RHS is indexed with `..` (uniformly valid for every supported shape —
/// `Vec`, `[U]`, `[U; N]`, `&[U]`, `&mut [U]`) and split at the same point, so
/// each deque half is compared against the corresponding contiguous chunk of
/// the RHS. A length check runs first so unequal-length operands short-circuit
/// without touching any elements.
///
/// # Usage
///
/// ```ignore
/// __impl_slice_eq1! { [] VecDeque<T, A>, Vec<U, A>, }
/// __impl_slice_eq1! { [const N: usize] VecDeque<T, A>, [U; N], }
/// ```
macro_rules! __impl_slice_eq1 {
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
                let (sa, sb) = self.as_slices();
                let (oa, ob) = other[..].split_at(sa.len());
                sa == oa && sb == ob
            }
        }
    };
}

pub(super) use __impl_slice_eq1;
