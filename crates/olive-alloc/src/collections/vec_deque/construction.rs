use super::VecDeque;
use super::wrapped_index::WrappedIndex;
use crate::alloc::Global;
use crate::raw_vec::RawVec;
use olive_core::alloc::Allocator;

// ---------------------------------------------------------------------------
// Constructors — global allocator
// ---------------------------------------------------------------------------

impl<T> VecDeque<T, Global> {
    /// Constructs a new, empty `VecDeque<T>`.
    ///
    /// The deque will not allocate until elements are pushed onto it.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self::new_in(Global)
    }
}

impl<T, A: Allocator> VecDeque<T, A> {
    /// Like [`Self::new`], but parameterized over the choice of allocator for
    /// the returned `VecDeque`.
    #[must_use]
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            buf: RawVec::new_in(alloc),
            head: WrappedIndex::zero(),
            len: 0,
        }
    }
}
