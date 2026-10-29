//! The allocation-backed constructors and conversions of the I/O error.
//!
//! This is the port of nightly `alloc`'s `io::error`: the part of the error
//! API that needs a heap and therefore cannot live in
//! [`olive_core::io::error`]. Divergences from upstream, on top of the
//! crate-wide fallibility convention:
//!
//! * Upstream adds its allocation-backed methods to `Error` as an inherent
//!   impl inside `alloc`, enabled by the internal
//!   `rustc_allow_incoherent_impl` attribute. Such usage is forbidden,
//!   so the methods are offered as the [`IoErrorAllocExt`] extension
//!   trait instead.
//! * Upstream's `Error::new` and `Error::other` become fallible
//!   [`IoErrorAllocExt::try_new`] and [`IoErrorAllocExt::try_other`], because
//!   boxing the payload can fail; [`IoErrorAllocExt::new_boxed`] covers the
//!   already-boxed case without allocating.
//! * The `Custom`/`CustomOwner` two-box chain does not exist here:
//!   `olive-core`'s [`Custom`] embeds the payload pointer directly.
//! * `From<TryReserveError>` already lives in `olive-core`; the orphan
//!   rule forbids a second impl here, and the same rule turns the lossless
//!   conversion of the `try_traits` errors into named constructors rather
//!   than `From` impls.

use crate::alloc::AllocError;
use crate::boxed::Box;
use core::ptr::NonNull;
use olive_core::io::error::custom::Custom;
use olive_core::io::error::{Error, ErrorKind};
use olive_core::try_traits::{TryCloneError, TryDefaultError};

/// The allocation-backed extension methods of the I/O error.
///
/// Import this trait to reach the [`Error`] methods that need a heap.
/// Upstream declares them as inherent methods of `Error` inside `alloc`,
/// which the orphan rules forbid here; see the module documentation.
pub trait IoErrorAllocExt: Sized {
    /// Creates a new I/O error from a known kind of error as well as an
    /// arbitrary error payload.
    ///
    /// This is used to generically create I/O errors which do not originate
    /// from the OS itself. The `error` argument is an arbitrary payload
    /// which will be contained in this [`Error`].
    ///
    /// If no extra payload is required, use the [`From`] conversion
    /// from [`ErrorKind`], which does not allocate.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the payload cannot be boxed.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::fmt;
    ///
    /// use olive_alloc::io::error::IoErrorAllocExt;
    /// use olive_alloc::io::{Error, ErrorKind};
    ///
    /// #[derive(Debug)]
    /// struct ReadFailure;
    ///
    /// impl fmt::Display for ReadFailure {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         f.write_str("read failure")
    ///     }
    /// }
    ///
    /// impl core::error::Error for ReadFailure {}
    ///
    /// let error = Error::try_new(ErrorKind::Interrupted, ReadFailure).unwrap();
    /// assert_eq!(error.kind(), ErrorKind::Interrupted);
    /// ```
    fn try_new<E>(kind: ErrorKind, error: E) -> Result<Error, AllocError>
    where
        E: core::error::Error + Send + Sync + 'static;

    /// Creates a new I/O error with [`ErrorKind::Other`] as its kind.
    ///
    /// This is a shortcut for [`IoErrorAllocExt::try_new`] with
    /// [`ErrorKind::Other`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the payload cannot be boxed.
    fn try_other<E>(error: E) -> Result<Error, AllocError>
    where
        E: core::error::Error + Send + Sync + 'static;

    /// Creates a new I/O error whose payload is an already-boxed error.
    ///
    /// Unlike [`IoErrorAllocExt::try_new`], this cannot fail: the payload is
    /// not boxed again, only linked into the error.
    #[must_use]
    fn new_boxed(kind: ErrorKind, error: Box<dyn core::error::Error + Send + Sync>) -> Error;

    /// Consumes the `Error`, returning its inner error (if any).
    ///
    /// If this [`Error`] was constructed via [`IoErrorAllocExt::try_new`],
    /// [`IoErrorAllocExt::try_other`], or [`IoErrorAllocExt::new_boxed`], this
    /// returns `Some`; otherwise it returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::io::error::IoErrorAllocExt;
    /// use olive_alloc::io::{Error, ErrorKind};
    ///
    /// assert!(Error::from(ErrorKind::NotFound).into_inner().is_none());
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    fn into_inner(self) -> Option<Box<dyn core::error::Error + Send + Sync>>;

    /// Attempts to downcast the boxed error payload to a concrete type.
    ///
    /// Returns `Ok` with the payload when this [`Error`] carries a custom
    /// payload of exactly type `E`, and `Err(self)` otherwise, leaving the
    /// error untouched for further inspection.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` when the error does not carry a custom payload
    /// of exactly type `E`.
    ///
    /// # Panics
    ///
    /// Panics if the payload fails to downcast after its type was verified
    /// through [`Error::get_ref`], which is the library's logic bug rather
    /// than a recoverable condition.
    fn downcast<E>(self) -> Result<E, Self>
    where
        E: core::error::Error + Send + Sync + 'static;

    /// Converts a [`TryCloneError`] into an [`Error`], keeping it as the
    /// payload.
    ///
    /// This method is the lossless counterpart of the `From<TryCloneError>` impl
    /// in `olive-core`, which must discard the source error because the core
    /// cannot allocate.
    ///
    /// Here the error is boxed as the custom payload, so
    /// [`Error::get_ref`], `source()`, and downcasting still reach it. The
    /// kind mapping is unchanged: allocation failures
    /// ([`TryCloneError::Reserve`] and [`TryCloneError::Alloc`]) map to
    /// [`ErrorKind::OutOfMemory`], logic failures ([`TryCloneError::Other`])
    /// to [`ErrorKind::Uncategorized`].
    ///
    /// The orphan rule forbids expressing this as a `From`/`TryFrom` impl,
    /// since both [`Error`] and [`TryCloneError`] are foreign to this crate.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the payload cannot be boxed.
    fn try_from_try_clone_error(err: TryCloneError) -> Result<Error, AllocError>;

    /// Converts a [`TryDefaultError`] into an [`Error`], keeping it as the
    /// payload.
    ///
    /// This behaves exactly like [`IoErrorAllocExt::try_from_try_clone_error`],
    /// for the same reasons.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the payload cannot be boxed.
    fn try_from_try_default_error(err: TryDefaultError) -> Result<Error, AllocError>;
}

impl IoErrorAllocExt for Error {
    fn try_new<E>(kind: ErrorKind, error: E) -> Result<Error, AllocError>
    where
        E: core::error::Error + Send + Sync + 'static,
    {
        Ok(Self::new_boxed(kind, try_box_dyn_error(error)?))
    }

    fn try_other<E>(error: E) -> Result<Error, AllocError>
    where
        E: core::error::Error + Send + Sync + 'static,
    {
        Self::try_new(ErrorKind::Other, error)
    }

    fn new_boxed(kind: ErrorKind, error: Box<dyn core::error::Error + Send + Sync>) -> Error {
        let payload = Box::into_non_null(error);
        // SAFETY: `payload` was just extracted from a `Box` of this crate's
        // global allocator, and `drop_box_raw` releases exactly that box.
        let custom = unsafe { Custom::from_raw(kind, payload, drop_box_raw) };
        Error::from_custom(custom)
    }

    fn into_inner(self) -> Option<Box<dyn core::error::Error + Send + Sync>> {
        let custom = self.into_custom().ok()?;
        let payload = custom.into_raw();
        // SAFETY: By `Custom`'s construction contract, the payload carries
        // the provenance of a `Box` of olive-alloc's global allocator, so
        // it can be reclaimed as such.
        Some(unsafe { Box::from_non_null(payload) })
    }

    fn downcast<E>(self) -> Result<E, Self>
    where
        E: core::error::Error + Send + Sync + 'static,
    {
        if self.get_ref().is_none_or(|payload| !payload.is::<E>()) {
            return Err(self);
        }
        let payload = self
            .into_inner()
            .expect("the error was verified to carry a custom payload");
        match payload.downcast::<E>() {
            // `Box` cannot move out through `Deref`, so the value is
            // extracted with `into_inner`, which also deallocates the
            // payload block.
            Ok(boxed) => Ok(Box::into_inner(boxed)),
            Err(_) => unreachable!("the payload type was verified by `get_ref`"),
        }
    }

    fn try_from_try_clone_error(err: TryCloneError) -> Result<Error, AllocError> {
        let kind = match err {
            TryCloneError::Reserve(_) | TryCloneError::Alloc(_) => ErrorKind::OutOfMemory,
            TryCloneError::Other(_) => ErrorKind::Uncategorized,
        };
        Ok(Self::new_boxed(kind, try_box_dyn_error(err)?))
    }

    fn try_from_try_default_error(err: TryDefaultError) -> Result<Error, AllocError> {
        let kind = match err {
            TryDefaultError::Reserve(_) | TryDefaultError::Alloc(_) => ErrorKind::OutOfMemory,
            TryDefaultError::Other(_) => ErrorKind::Uncategorized,
        };
        Ok(Self::new_boxed(kind, try_box_dyn_error(err)?))
    }
}

/// Releases a boxed error payload.
///
/// This is the drop function installed in every [`Custom`] built by this
/// module.
///
/// # Safety
///
/// `ptr` must be a live payload that was boxed as a
/// `Box<dyn core::error::Error + Send + Sync>` of olive-alloc's global
/// allocator, and must not have been released yet.
unsafe fn drop_box_raw(ptr: NonNull<dyn core::error::Error + Send + Sync>) {
    // SAFETY: The caller guarantees `ptr` is a live box allocation.
    drop(unsafe { Box::from_non_null(ptr) });
}

/// Boxes an error value and views it as the unsized payload type.
fn try_box_dyn_error<E>(error: E) -> Result<Box<dyn core::error::Error + Send + Sync>, AllocError>
where
    E: core::error::Error + Send + Sync + 'static,
{
    let boxed: Box<E> = Box::try_new(error)?;
    let raw: *mut E = Box::into_raw(boxed);
    // Unsize coercion attaches the vtable metadata; the allocation behind
    // the pointer is unchanged.
    let fat: *mut (dyn core::error::Error + Send + Sync) = raw;
    // SAFETY: `fat` is the same live, non-null allocation that `boxed`
    // owned, now viewed through the trait-object metadata.
    Ok(unsafe { Box::from_non_null(NonNull::new_unchecked(fat)) })
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::{
        alloc::Layout,
        fmt::{self, Display},
    };
    use std::string::ToString;

    use olive_core::{alloc_errors::TryReserveError, const_error};

    use super::*;

    #[derive(Debug)]
    struct TestError {
        message: &'static str,
    }

    impl TestError {
        fn new(message: &'static str) -> Self {
            Self { message }
        }
    }

    impl Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.message)
        }
    }

    impl core::error::Error for TestError {}

    #[test]
    fn try_new_and_into_inner_roundtrip() {
        let error =
            Error::try_new(ErrorKind::Interrupted, TestError::new("custom failure")).unwrap();
        assert_eq!(error.kind(), ErrorKind::Interrupted);
        assert!(error.raw_os_error().is_none());

        let reference = error.get_ref().expect("custom payload expected");
        assert_eq!(reference.to_string(), "custom failure");

        let inner = error.into_inner().expect("custom payload expected");
        assert_eq!(inner.to_string(), "custom failure");
    }

    #[test]
    fn try_other_uses_other_kind() {
        let error = Error::try_other(TestError::new("oh no!")).unwrap();
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.get_ref().unwrap().to_string(), "oh no!");
        assert_eq!(error.into_inner().unwrap().to_string(), "oh no!");
    }

    #[test]
    fn into_inner_is_none_for_simple_error() {
        assert!(Error::from(ErrorKind::NotFound).into_inner().is_none());
        assert!(
            const_error!(ErrorKind::NotFound, "const error")
                .into_inner()
                .is_none()
        );
    }

    #[test]
    fn new_boxed_takes_an_existing_payload() {
        let boxed = try_box_dyn_error(TestError::new("boxed")).unwrap();
        let error = Error::new_boxed(ErrorKind::Unsupported, boxed);
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.into_inner().expect("payload expected").to_string(),
            "boxed"
        );
    }

    #[test]
    fn custom_roundtrip_through_into_custom() {
        let error = Error::try_other(TestError::new("payload")).unwrap();

        let custom = error.into_custom().expect("custom payload expected");
        // Rebuilding must not release the payload here; only the final
        // `into_inner` may.
        let error = Error::from_custom(custom);
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(
            error.into_inner().expect("payload expected").to_string(),
            "payload"
        );
    }

    #[test]
    fn downcast_roundtrip() {
        let error = Error::try_other(TestError::new("downcast me")).unwrap();
        let recovered = error.downcast::<TestError>().expect("matching type");
        assert_eq!(recovered.message, "downcast me");

        let error = Error::try_other(TestError::new("keep me")).unwrap();
        let error = error
            .downcast::<TryCloneError>()
            .expect_err("payload has another type");
        assert!(error.into_inner().is_some());

        let simple = Error::from(ErrorKind::NotFound);
        assert!(simple.downcast::<TestError>().is_err());

        let const_err = const_error!(ErrorKind::NotFound, "const error");
        assert!(const_err.downcast::<TestError>().is_err());
    }

    #[test]
    fn try_clone_error_is_kept_as_payload() {
        let error = Error::try_from_try_clone_error(TryCloneError::Other("no clone")).unwrap();
        assert_eq!(error.kind(), ErrorKind::Uncategorized);
        let payload = error.get_ref().expect("custom payload expected");
        assert!(matches!(
            payload.downcast_ref::<TryCloneError>(),
            Some(TryCloneError::Other("no clone"))
        ));

        let error = Error::try_from_try_clone_error(TryCloneError::Alloc(AllocError)).unwrap();
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);
        let error = Error::try_from_try_clone_error(TryCloneError::Reserve(
            TryReserveError::new_capacity_overflow(),
        ))
        .unwrap();
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);
        let error = Error::try_from_try_clone_error(TryCloneError::Reserve(
            TryReserveError::new_alloc(Layout::new::<u64>()),
        ))
        .unwrap();
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);
    }

    #[test]
    fn try_default_error_is_kept_as_payload() {
        let error =
            Error::try_from_try_default_error(TryDefaultError::Other("no default")).unwrap();
        assert_eq!(error.kind(), ErrorKind::Uncategorized);
        let payload = error.get_ref().expect("custom payload expected");
        assert!(matches!(
            payload.downcast_ref::<TryDefaultError>(),
            Some(TryDefaultError::Other("no default"))
        ));

        let error = Error::try_from_try_default_error(TryDefaultError::Alloc(AllocError)).unwrap();
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);
        let error = Error::try_from_try_default_error(TryDefaultError::Reserve(
            TryReserveError::new_capacity_overflow(),
        ))
        .unwrap();
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);
        let error = Error::try_from_try_default_error(TryDefaultError::Reserve(
            TryReserveError::new_alloc(Layout::new::<u64>()),
        ))
        .unwrap();
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);
    }
}
