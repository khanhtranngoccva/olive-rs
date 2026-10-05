use core::fmt;
use core::fmt::Debug;

use super::BTreeSet;
use super::iter::Iter;
use crate::alloc::Global;
use olive_core::alloc::Allocator;

/// A lazy iterator producing elements in the intersection of `BTreeSet`s.
///
/// This `struct` is created by the [`intersection`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`intersection`]: BTreeSet::intersection
#[must_use = "this returns the intersection as an iterator, \
              without modifying either input set"]
pub struct Intersection<'a, T: 'a, A: Allocator + Clone = Global> {
    inner: IntersectionInner<'a, T, A>,
}

enum IntersectionInner<'a, T: 'a, A: Allocator + Clone> {
    /// Iterate similarly sized sets jointly, spotting matches along the way
    Stitch {
        a: Iter<'a, T>,
        b: Iter<'a, T>,
    },
    /// Iterate a small set, look up in the large set
    Search {
        small_iter: Iter<'a, T>,
        large_set: &'a BTreeSet<T, A>,
    },
    Answer(Option<&'a T>), // return a specific element or emptiness
}

impl<T: Debug, A: Allocator + Clone> Debug for IntersectionInner<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntersectionInner::Stitch { a, b } => f
                .debug_struct("Stitch")
                .field("a", a)
                .field("b", b)
                .finish(),
            IntersectionInner::Search {
                small_iter,
                large_set,
            } => f
                .debug_struct("Search")
                .field("small_iter", small_iter)
                .field("large_set", large_set)
                .finish(),
            IntersectionInner::Answer(x) => f.debug_tuple("Answer").field(x).finish(),
        }
    }
}
