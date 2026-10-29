//! The custom payload of an [`Error`](super::Error).
//!
//! Unlike upstream, whose unpacked representation stores the custom data
//! as `Box<Custom>` (one allocation for the boxed implementation, one for
//! the boxed payload), the `Custom` is embedded directly in the error
//! representation here, so that instantiating a custom error costs exactly
//! one allocation. This reduces chances of allocation failure, but leads to
//! a larger error payload size.

use super::ErrorKind;
use core::fmt;
use core::mem;
use core::ptr::NonNull;

/// A user-created allocated error, embedded in the error representation
/// and pointing at its boxed payload.
///
// As with `SimpleMessage`, the
// `#[repr(align(4))]` here is only needed by a future bit-packed
// representation; in practice the alignment is almost certainly
// already this high or higher.
#[doc(hidden)]
#[repr(align(4))]
pub struct Custom {
    /// The [`ErrorKind`] to return from [`Error::kind`](super::Error::kind).
    pub(super) kind: ErrorKind,

    /// The payload itself; effectively a `Box` over `dyn Error` allocated
    /// with `olive-alloc`'s global allocator.
    // INVARIANT: `error` must be a valid pointer and it must be safe to
    // call `error_drop` with it once. The pointee must have been
    // allocated and owned as a `Box` of `olive-alloc`'s global
    // allocator, so that it can be reclaimed with
    // `olive_alloc::boxed::Box::from_raw`.
    error: NonNull<dyn core::error::Error + Send + Sync>,

    /// Releases the payload, freeing the allocation that backs it.
    // INVARIANT: must be safe to call `error_drop` exactly once with
    // `error`, and must free the boxed allocation.
    error_drop: unsafe fn(NonNull<dyn core::error::Error + Send + Sync>),
}

// SAFETY: All members of `Custom` are `Send`
unsafe impl Send for Custom {}

// SAFETY: All members of `Custom` are `Sync`
unsafe impl Sync for Custom {}

/// Formats the payload through the `Debug` impl every `dyn Error` carries
/// as a supertrait. The wrapper avoids a trait-upcasting coercion: passing
/// `&dyn Error` where `&dyn Debug` is expected would require it, but
/// instantiating `Debug::fmt` doesn't.
struct PayloadDebug<'a>(&'a (dyn core::error::Error + Send + Sync + 'static));

impl fmt::Debug for PayloadDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.0, f)
    }
}

#[allow(
    clippy::missing_fields_in_debug,
    reason = "only the `kind` and `error` field matters for an error"
)]
impl fmt::Debug for Custom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Custom")
            .field("kind", &self.kind)
            .field("error", &PayloadDebug(self.error_ref()))
            .finish()
    }
}

impl Drop for Custom {
    fn drop(&mut self) {
        // SAFETY: By `Custom` invariants, this is a drop call on a valid
        // pointer.
        unsafe {
            (self.error_drop)(self.error);
        }
    }
}

impl Custom {
    /// Creates a custom payload from its raw parts.
    ///
    /// # Safety
    ///
    /// * `error` must be valid for up to a static lifetime, and own its
    ///   pointee.
    /// * `error_drop` must be safe to call for the pointer `error`
    ///   exactly once.
    /// * `error` must point at a payload that was allocated and owned as
    ///   a `Box` of `olive-alloc`'s global allocator, and `error_drop`
    ///   must release exactly that box. [`Custom`] is a stand-in for
    ///   `olive_alloc::boxed::Box`, which this core cannot depend on; the box
    ///   provenance is what allows a layer above to reconstruct the box
    ///   safely (olive-std's `into_inner`).
    ///   Payloads from other containers, such as `Arc`, are not valid
    ///   here, since they cannot be turned back into a `Box`.
    pub unsafe fn from_raw(
        kind: ErrorKind,
        error: NonNull<dyn core::error::Error + Send + Sync>,
        error_drop: unsafe fn(NonNull<dyn core::error::Error + Send + Sync>),
    ) -> Custom {
        // INVARIANT: Function preconditions match type invariants.
        Custom {
            kind,
            error,
            error_drop,
        }
    }

    /// Consumes the [`Custom`] payload and return the raw error pointer.
    ///
    /// The returned pointer keeps the box provenance guaranteed by
    /// [`Custom::from_raw`], so it can be reclaimed with
    /// `olive_alloc::boxed::Box::from_raw`.
    #[must_use]
    pub fn into_raw(self) -> NonNull<dyn core::error::Error + Send + Sync> {
        let ptr = self.error;
        // Avoid `Custom::drop` which would free the error pointer.
        mem::forget(self);
        ptr
    }

    /// Returns a reference to the payload.
    pub(super) fn error_ref(&self) -> &(dyn core::error::Error + Send + Sync + 'static) {
        // SAFETY: `from_raw` ensures `error` is a valid pointer up to a
        // static lifetime and is owned by `self`.
        unsafe { self.error.as_ref() }
    }

    /// Returns a mutable reference to the payload.
    pub(super) fn error_mut(&mut self) -> &mut (dyn core::error::Error + Send + Sync + 'static) {
        // SAFETY: `from_raw` ensures `error` is a valid pointer up to a
        // static lifetime and is owned by `self`.
        unsafe { self.error.as_mut() }
    }
}
