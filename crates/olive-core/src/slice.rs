//! Re-exports of [`core::slice`] API items plus Olive's fallible range-resolution
//! helpers layered on top.
//!
//! This module shadows the glob-imported [`core::slice`] from the crate root so that
//! both original items and Olive-specific items — notably [`try_range`] — are reachable at
//! `olive_core::slice::…`.
pub use core::slice::*;

use core::fmt;
use core::ops::{Bound, Range, RangeBounds, RangeTo};

/// Error returned by [`try_range`] when a [`RangeBounds`] cannot be resolved into a
/// concrete [`Range`] against the given length.
///
/// The four variants correspond one-to-one with the failure modes of std's
/// panicking [`core::slice::range`]: an excluded start that overflows,
/// an inclusive/excluded end that overflows, a start that lands past the end,
/// and an end that exceeds the slice length.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrySliceRangeError {
    /// Resolving an excluded start range bound required adding one, which overflowed.
    StartOverflow,
    /// Resolving an included end range bound required adding one, which overflowed.
    EndOverflow,
    /// The resolved start index is greater than the resolved end index (a
    /// reversed range). All 3 fields are deliberately populated for
    /// diagnostics.
    StartExceedsEnd {
        /// The computed start of the range.
        start: usize,
        /// The computed end of the range (exclusive).
        end: usize,
        /// The length the range was validated against.
        len: usize,
    },
    /// The resolved end index exceeds the supplied length (or slice end bound).
    /// All 3 fields are deliberately populated for diagnostics.
    EndExceedsBound {
        /// The computed start of the range.
        start: usize,
        /// The computed end of the range (exclusive).
        end: usize,
        /// The length the range was validated against.
        len: usize,
    },
}

impl fmt::Debug for TrySliceRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartOverflow => f.debug_tuple("TrySliceRangeError::StartOverflow").finish(),
            Self::EndOverflow => f.debug_tuple("TrySliceRangeError::EndOverflow").finish(),
            Self::StartExceedsEnd { start, end, len } => f
                .debug_struct("TrySliceRangeError::StartExceedsEnd")
                .field("start", start)
                .field("end", end)
                .field("len", len)
                .finish(),
            Self::EndExceedsBound { start, end, len } => f
                .debug_struct("TrySliceRangeError::EndExceedsBound")
                .field("start", start)
                .field("end", end)
                .field("len", len)
                .finish(),
        }
    }
}

impl fmt::Display for TrySliceRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartOverflow => write!(f, "start bound overflowed while resolving range"),
            Self::EndOverflow => write!(f, "end bound overflowed while resolving range"),
            Self::StartExceedsEnd { start, end, .. } => {
                write!(f, "range start {start} exceeds end {end}")
            }
            Self::EndExceedsBound { end, len, .. } => {
                write!(f, "range end {end} exceeds length {len}")
            }
        }
    }
}

impl core::error::Error for TrySliceRangeError {}

/// Fallibly resolves a [`RangeBounds<usize>`] into a concrete [`Range<usize>`]
/// validated against a length, returning a [`Result`] instead of panicking.
///
/// This mirrors std's unstable [`core::slice::range`].
///
/// # Errors
///
/// Returns [`TrySliceRangeError`] if an excluded/inclusive edge overflows when
/// converted, the resolved start exceeds the resolved end, or the resolved end
/// exceeds the supplied length.
pub fn try_range<R>(range: R, bounds: RangeTo<usize>) -> Result<Range<usize>, TrySliceRangeError>
where
    R: RangeBounds<usize>,
{
    let len = bounds.end;

    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(start) => start
            .checked_add(1)
            .ok_or(TrySliceRangeError::StartOverflow)?,
        Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        Bound::Included(end) => end.checked_add(1).ok_or(TrySliceRangeError::EndOverflow)?,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };

    if start > end {
        return Err(TrySliceRangeError::StartExceedsEnd { start, end, len });
    }
    if end > len {
        return Err(TrySliceRangeError::EndExceedsBound { start, end, len });
    }

    Ok(Range { start, end })
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::format;

    #[test]
    fn try_range_resolves_common_forms() {
        assert_eq!(try_range(1..2, ..3usize).unwrap(), 1..2);
        assert_eq!(try_range(..2, ..3).unwrap(), 0..2);
        assert_eq!(try_range(1.., ..3).unwrap(), 1..3);
        assert_eq!(try_range(.., ..3).unwrap(), 0..3);
        // Inclusive edges add one.
        assert_eq!(try_range(1..=2, ..5).unwrap(), 1..3);
        assert_eq!(try_range(..=1, ..5).unwrap(), 0..2);
    }

    #[test]
    fn try_range_zero_length_is_valid() {
        assert_eq!(try_range(0..0, ..0usize).unwrap(), 0..0);
        assert_eq!(try_range(.., ..0).unwrap(), 0..0);
    }

    #[test]
    fn try_range_excluded_start_overflow_is_reversed() {
        // An *excluded* start bound is not incremented (matching std), so
        // `usize::MAX..` resolves to start == usize::MAX and then trips the
        // ordering check rather than the overflow check.
        let e = try_range(usize::MAX.., ..10usize).unwrap_err();
        assert_eq!(
            e,
            TrySliceRangeError::StartExceedsEnd {
                start: usize::MAX,
                end: 10,
                len: 10,
            }
        );
    }

    #[test]
    fn try_range_end_overflow() {
        let e = try_range(0..=usize::MAX, ..10usize).unwrap_err();
        assert_eq!(e, TrySliceRangeError::EndOverflow);
    }

    #[test]
    fn try_range_start_exceeds_end() {
        #[allow(clippy::reversed_empty_ranges, reason = "testing failure path")]
        let e = try_range(2..1, ..5usize).unwrap_err();
        assert_eq!(
            e,
            TrySliceRangeError::StartExceedsEnd {
                start: 2,
                end: 1,
                len: 5
            }
        );
    }

    #[test]
    fn try_range_end_exceeds_bound() {
        let e = try_range(1..4, ..3usize).unwrap_err();
        assert_eq!(
            e,
            TrySliceRangeError::EndExceedsBound {
                start: 1,
                end: 4,
                len: 3
            }
        );
    }

    #[test]
    fn try_range_error_display() {
        let e = TrySliceRangeError::StartOverflow;
        assert_eq!(
            format!("{e}"),
            "start bound overflowed while resolving range"
        );
        let e = TrySliceRangeError::EndExceedsBound {
            start: 1,
            end: 4,
            len: 3,
        };
        assert_eq!(format!("{e}"), "range end 4 exceeds length 3");
    }
}
