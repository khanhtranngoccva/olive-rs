//! A poison-pill payload that fails its own clone after a set number of tries.

use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

/// A *poison pill*: a value whose `try_clone` succeeds while its internal
/// counter is below `threshold`, incrementing the counter each successful
/// clone, then fails forever once the counter reaches `threshold`.
///
/// Retained as the older of the two mechanisms; prefer
/// [`MaybeUnclonable`](super::MaybeUnclonable) unless the test genuinely needs
/// the failure to land after a fixed number of clones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlakyClone {
    /// Successful clones performed so far.
    pub count: u32,
    /// Number of successful clones permitted before `try_clone` starts failing.
    pub threshold: u32,
}

impl FlakyClone {
    /// Shorthand constructor for the common case of starting at 0: the element
    /// clones successfully `threshold` times, then fails forever.
    pub const fn new(threshold: u32) -> Self {
        Self {
            count: 0,
            threshold,
        }
    }

    /// Places the element at an arbitrary point in its clone sequence: it has
    /// already performed `count` clones and will fail once `count` reaches
    /// `threshold`.
    ///
    /// This is the explicit form of the struct literal that call sites used to
    /// write by hand.
    pub const fn at(count: u32, threshold: u32) -> Self {
        Self { count, threshold }
    }

    /// An element that has already exhausted its budget: its very first clone
    /// fails.
    pub const fn exhausted() -> Self {
        Self {
            count: 0,
            threshold: 0,
        }
    }
}

impl TryClone for FlakyClone {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if self.count >= self.threshold {
            Err(TryCloneError::Other("flaky"))
        } else {
            Ok(Self {
                count: self.count + 1,
                threshold: self.threshold,
            })
        }
    }
}
