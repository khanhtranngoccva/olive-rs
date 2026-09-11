use core::cmp::Ordering;

/// Represents an index that can be safely used to index the VecDeque buffer.
/// It exists as a separate type to avoid passing logical (unwrapped) indices to various
/// VecDeque functions by accident.
///
/// The invariant of this index is that it is always < VecDeque capacity, unless the VecDeque
/// is empty (in that case the index can be 0 when the capacity is 0).
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
#[repr(transparent)]
pub(super) struct WrappedIndex(usize);

impl WrappedIndex {
    /// The newly constructed index has to be in-bounds for the VecDeque
    /// that uses the index.
    #[inline(always)]
    pub(super) fn from_arbitrary_number(index: usize) -> Self {
        Self(index)
    }

    /// Safety invariant: the newly constructed index must still be in-bounds for the VecDeque.
    /// Used by mutation methods (push/pop/remove), which have not landed yet.
    #[expect(
        unused,
        reason = "used by push/pop/remove, which have not landed yet"
    )]
    #[inline(always)]
    pub(super) unsafe fn add(self, offset: usize) -> Self {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "caller assert that the result is in-bounds"
        )]
        Self(self.0 + offset)
    }

    /// Safety invariant: the newly constructed index must still be in-bounds for the VecDeque.
    /// Used by mutation methods (push/pop/remove), which have not landed yet.
    #[expect(
        unused,
        reason = "used by push/pop/remove, which have not landed yet"
    )]
    #[inline(always)]
    pub(super) unsafe fn sub(self, offset: usize) -> Self {
        debug_assert!(self.0 >= offset);
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "caller asserts that the result is in-bounds"
        )]
        {
            Self(self.0 - offset)
        }
    }

    #[inline(always)]
    pub(super) const fn zero() -> Self {
        Self(0)
    }

    /// Used by mutation methods (push/pop/remove), which have not landed yet.
    #[expect(
        unused,
        reason = "used by push/pop/remove, which have not landed yet"
    )]
    #[inline(always)]
    pub(super) fn abs_diff(self, other: Self) -> usize {
        self.0.abs_diff(other.0)
    }

    #[inline(always)]
    pub(super) fn as_index(self) -> usize {
        self.0
    }

    /// Used by mutation methods (push/pop/remove), which have not landed yet.
    #[expect(
        unused,
        reason = "used by push/pop/remove, which have not landed yet"
    )]
    #[inline(always)]
    pub(super) fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl core::fmt::Display for WrappedIndex {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl PartialEq<usize> for WrappedIndex {
    #[inline(always)]
    fn eq(&self, other: &usize) -> bool {
        self.0.eq(other)
    }
}

impl PartialOrd<usize> for WrappedIndex {
    #[inline(always)]
    fn partial_cmp(&self, other: &usize) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}
