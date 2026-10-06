use core::cmp::Ordering;
use core::fmt::{self, Debug};
use core::iter::{DoubleEndedIterator, FusedIterator};
use olive_core::iter::DoubleEndedPeekable;
use olive_core::try_traits::{TryClone, TryCloneError};

/// Core of an iterator that merges the output of two strictly ascending iterators,
/// for instance a union or a symmetric difference.
pub(crate) struct MergeIterInner<I: DoubleEndedIterator + FusedIterator> {
    a: DoubleEndedPeekable<I>,
    b: DoubleEndedPeekable<I>,
}

impl<I: DoubleEndedIterator + FusedIterator> Clone for MergeIterInner<I>
where
    I: Clone,
    I::Item: Clone,
{
    fn clone(&self) -> Self {
        Self {
            a: self.a.clone(),
            b: self.b.clone(),
        }
    }
}

impl<I: DoubleEndedIterator + FusedIterator> TryClone for MergeIterInner<I>
where
    I: TryClone,
    I::Item: TryClone,
{
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Self {
            a: self.a.try_clone()?,
            b: self.b.try_clone()?,
        })
    }
}

impl<I: DoubleEndedIterator + FusedIterator> Debug for MergeIterInner<I>
where
    I: Debug,
    I::Item: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MergeIterInner")
            .field(&self.a)
            .field(&self.b)
            .finish()
    }
}

impl<I: DoubleEndedIterator + FusedIterator> MergeIterInner<I> {
    /// Creates a new core for an iterator merging a pair of sources.
    pub(crate) fn new(a: I, b: I) -> Self {
        MergeIterInner {
            a: DoubleEndedPeekable::new(a),
            b: DoubleEndedPeekable::new(b),
        }
    }

    /// Returns the next pair of items stemming from the pair of sources
    /// being merged.
    ///
    /// If both returned options contain a value, that value
    /// is equal and occurs in both sources.
    ///
    /// If one of the returned options contains a value, that value
    /// doesn't occur in the other source (or the sources are not
    /// strictly ascending).  
    ///
    /// If neither returned option contains a value, iteration has finished
    /// and subsequent calls will return the same empty pair.
    pub(crate) fn nexts<Cmp: Fn(&I::Item, &I::Item) -> Ordering>(
        &mut self,
        cmp: Cmp,
    ) -> (Option<I::Item>, Option<I::Item>) {
        let mut a_next = self.a.peek_front();
        let mut b_next = self.b.peek_front();
        if let (Some(a1), Some(b1)) = (&a_next, &b_next) {
            match cmp(a1, b1) {
                // The smaller head is yielded now; the larger one stays
                // cached for the next call.
                Ordering::Less => b_next = None,
                Ordering::Greater => a_next = None,
                Ordering::Equal => (),
            }
        }
        (
            a_next.is_some().then(|| self.a.next().unwrap()),
            b_next.is_some().then(|| self.b.next().unwrap()),
        )
    }

    /// Returns the next pair of items stemming from the pair of sources
    /// being merged, advancing them from the back.
    ///
    /// Behaves like [`nexts`](Self::nexts) but walks both sources in
    /// descending order, so it pairs up the largest remaining elements.
    pub(crate) fn nexts_back<Cmp: Fn(&I::Item, &I::Item) -> Ordering>(
        &mut self,
        cmp: Cmp,
    ) -> (Option<I::Item>, Option<I::Item>) {
        let mut a_next_back = self.a.peek_back();
        let mut b_next_back = self.b.peek_back();
        if let (Some(a1), Some(b1)) = (&a_next_back, &b_next_back) {
            match cmp(a1, b1) {
                // The larger tail is yielded now; the smaller one stays
                // cached for the next call.
                Ordering::Less => a_next_back = None,
                Ordering::Greater => b_next_back = None,
                Ordering::Equal => (),
            }
        }
        (
            a_next_back.is_some().then(|| self.a.next_back().unwrap()),
            b_next_back.is_some().then(|| self.b.next_back().unwrap()),
        )
    }

    /// Returns a pair of upper bounds for the `size_hint` of the final iterator.
    pub(crate) fn lens(&self) -> (usize, usize)
    where
        I: ExactSizeIterator,
    {
        (self.a.len(), self.b.len())
    }
}
