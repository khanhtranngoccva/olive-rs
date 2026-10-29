//! The error type for I/O operations, ported from nightly `core`'s
//! `core::io::Error` (the restructure that moved `std::io::Error` down into `core`).
//!
//! Porting notes, relative to upstream:
//!
//! * The payload representation is the safe, unpacked enum in
//!   `representation`; the bit-packed optimization can land later behind a
//!   `cfg_attr(path = ...)` switch without touching this file.
//! * Nothing here allocates, panics, or aborts.
//! * There is no `Error::new`, `Error::other`, or `last_os_error`.
//!   Custom payloads enter through [`Error::from_custom`], which
//!   embeds a [`Custom`] holding a pointer to an already-boxed
//!   payload and the drop function that releases it. Building one
//!   costs a single allocation: the boxed payload itself.
//!   `olive-alloc` and `olive-std` can have implementations that
//!   invoke this.
//! * The OS-function plumbing lives in `os_functions` (with and without
//!   atomic support, similar to upstream)

// Selects the OS-function plumbing at compile time. The precise
// `unstable_target_has_atomic_load_store = "ptr"` cfg is emitted by the
// build script. The stable `cfg(target_has_atomic = "ptr")` is the
// conservative fallback (a strict subset: RMW implies load/store).
#[cfg_attr(
    any(
        unstable_target_has_atomic_load_store = "ptr",
        target_has_atomic = "ptr",
    ),
    path = "os_functions_atomic.rs"
)]
#[cfg_attr(
    not(any(
        unstable_target_has_atomic_load_store = "ptr",
        target_has_atomic = "ptr",
    )),
    path = "os_functions.rs"
)]
mod os_functions;

/// The custom payload of an [`Error`].
#[doc(hidden)]
pub mod custom;

mod representation;

use olive_macros::TryClone;

use self::custom::Custom;
use self::os_functions::{decode_error_kind, format_os_error, is_interrupted, set_functions};
use self::representation::Repr;
use crate::alloc_errors::TryReserveError;
use crate::try_traits::{TryCloneError, TryDefaultError};
use core::fmt;
use core::result;

/// A specialized [`Result`] type for I/O operations.
///
/// This type alias is generally used to avoid writing out [`io::Error`](Error)
/// directly and is otherwise a direct mapping to [`Result`].
///
/// To avoid shadowing of the core [`Result`], the `io::Result` name should be used.
///
/// [`Result`]: crate::result::Result
pub type Result<T> = result::Result<T, Error>;

/// The error type for I/O operations.
///
/// This is Olive's fully-fallible port of `std::io::Error`.
///
/// Errors mostly originate from the underlying OS, but custom instances of
/// `Error` can be created with crafted error messages and/or a particular value of
/// [`ErrorKind`].
///
/// Like upstream, an error is one of four shapes behind a single representation:
/// a raw OS error code, a plain [`ErrorKind`], a `&'static` [`SimpleMessage`]
/// (constructible in `const` context through [`const_error!`]), or a custom
/// payload ([`Custom`]) that is usually a `Box` from `olive-alloc`.
pub struct Error {
    repr: Repr,
}

impl fmt::Debug for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.repr, fmt)
    }
}

/// Creates a constant [`Error`] from an [`ErrorKind`] and a string literal.
///
/// This method does not allocate and can be used in constant contexts.
///
/// ```
/// const FAIL: olive_core::io::error::Error = olive_core::const_error!(
///     olive_core::io::error::ErrorKind::Unsupported,
///     "tried something that never works",
/// );
/// ```
#[macro_export]
macro_rules! const_error {
    ($kind:expr, $message:expr $(,)?) => {
        $crate::io::error::Error::from_static_message(
            const {
                &$crate::io::error::SimpleMessage {
                    kind: $kind,
                    message: $message,
                }
            },
        )
    };
}

/// Common errors constants for use in std
#[doc(hidden)]
impl Error {
    #[doc(hidden)]
    pub const INVALID_UTF8: Self =
        const_error!(ErrorKind::InvalidData, "stream did not contain valid UTF-8");

    #[doc(hidden)]
    pub const READ_EXACT_EOF: Self =
        const_error!(ErrorKind::UnexpectedEof, "failed to fill whole buffer");

    #[doc(hidden)]
    pub const UNKNOWN_THREAD_COUNT: Self = const_error!(
        ErrorKind::NotFound,
        "the number of hardware threads is not known for the target platform",
    );

    #[doc(hidden)]
    pub const UNSUPPORTED_PLATFORM: Self = const_error!(
        ErrorKind::Unsupported,
        "operation not supported on this platform"
    );

    #[doc(hidden)]
    pub const WRITE_ALL_EOF: Self =
        const_error!(ErrorKind::WriteZero, "failed to write whole buffer");

    #[doc(hidden)]
    pub const ZERO_TIMEOUT: Self =
        const_error!(ErrorKind::InvalidInput, "cannot set a 0 duration timeout");

    #[doc(hidden)]
    pub const NO_ADDRESSES: Self = const_error!(
        ErrorKind::InvalidInput,
        "could not resolve to any addresses"
    );
}

/// The four shapes an I/O error can take, parametrized over the type of
/// the custom payload.
enum ErrorData<Custom> {
    /// An error code straight from the operating system.
    Os(RawOsError),

    /// An [`ErrorKind`] with no attached detail.
    Simple(ErrorKind),

    /// A static kind/message pair, usable in `const` context.
    SimpleMessage(&'static SimpleMessage),

    /// A custom payload whose drop behavior is supplied externally. It is
    /// practically an error kind and a olive-alloc's `Box` over the global
    /// allocator.
    Custom(Custom),
}

/// A kind/message pair with a `'static` lifetime, usable as an error
/// payload in `const` context (see [`const_error!`]).
#[derive(Debug)]
#[repr(align(4))]
pub struct SimpleMessage {
    /// The kind this message carries.
    pub kind: ErrorKind,

    /// The human-readable message.
    pub message: &'static str,
}

impl From<ErrorKind> for Error {
    /// Converts an [`ErrorKind`] into an [`Error`] with no attached detail.
    #[inline]
    fn from(kind: ErrorKind) -> Error {
        Error {
            repr: Repr::new_simple(kind),
        }
    }
}

impl Error {
    /// Creates an [`Error`] whose payload is a custom boxed error type.
    ///
    /// This does not perform any allocation.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub fn from_custom(custom: Custom) -> Error {
        Error {
            repr: Repr::new_custom(custom),
        }
    }

    /// Takes the custom payload out of an [`Error`], if there is one.
    ///
    /// By the [`Custom`] construction contract (see
    /// [`Custom::from_raw`]), the payload of the returned [`Custom`] has
    /// olive-alloc `Box` provenance.
    #[doc(hidden)]
    #[inline]
    pub fn into_custom(self) -> result::Result<Custom, Self> {
        if matches!(self.repr.data(), ErrorData::Custom(..)) {
            let ErrorData::Custom(c) = self.repr.into_data() else {
                // SAFETY: Checked above using `matches!`.
                unsafe { core::hint::unreachable_unchecked() }
            };
            Ok(c)
        } else {
            Err(self)
        }
    }

    /// Creates a constant [`Error`] from a static kind/message pair.
    ///
    /// This does not allocate. You should not use this directly, and
    /// instead use [`const_error!`].
    #[inline]
    #[must_use]
    #[doc(hidden)]
    pub const fn from_static_message(msg: &'static SimpleMessage) -> Error {
        Self {
            repr: Repr::new_simple_message(msg),
        }
    }

    /// Creates an [`Error`] from a raw OS error code, installing the given
    /// [`OsFunctions`] table so that later calls to [`Error::kind`],
    /// [`Error::is_interrupted`], and the `Display`/`Debug`
    /// implementations can decode and format the code.
    ///
    /// # Safety
    ///
    /// `functions` must point to data that is entirely constant; it must
    /// not be created during runtime.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub unsafe fn from_raw_os_error_with_functions(
        code: RawOsError,
        functions: &'static OsFunctions,
    ) -> Error {
        // SAFETY: Caller ensures `functions` is a constant not created at runtime.
        unsafe {
            set_functions(functions);
        }
        Error {
            repr: Repr::new_os(code),
        }
    }

    /// Returns the raw OS error code the error was created from, if any.
    ///
    /// If this [`Error`] was not constructed from a raw OS error code
    /// (via `last_os_error`) or (via `from_raw_os_error`), this returns [`None`].
    #[must_use]
    pub fn raw_os_error(&self) -> Option<RawOsError> {
        match self.repr.data() {
            ErrorData::Os(errno) => Some(errno),
            ErrorData::Custom(..) | ErrorData::Simple(..) | ErrorData::SimpleMessage(..) => None,
        }
    }

    /// Returns a reference to the custom payload, if there is one.
    ///
    /// If this [`Error`] was constructed via [`new`][new] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    #[must_use]
    pub fn get_ref(&self) -> Option<&(dyn core::error::Error + Send + Sync + 'static)> {
        match self.repr.data() {
            ErrorData::Custom(c) => Some(c.error_ref()),
            ErrorData::Os(..) | ErrorData::Simple(..) | ErrorData::SimpleMessage(..) => None,
        }
    }

    /// Returns a mutable reference to the custom payload, if there is one.
    ///
    /// If this [`Error`] was constructed via [`new`][new] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    pub fn get_mut(&mut self) -> Option<&mut (dyn core::error::Error + Send + Sync + 'static)> {
        match self.repr.data_mut() {
            ErrorData::Custom(c) => Some(c.error_mut()),
            ErrorData::Os(..) | ErrorData::Simple(..) | ErrorData::SimpleMessage(..) => None,
        }
    }

    /// Returns the corresponding [`ErrorKind`] for this error.
    ///
    /// This may be a value set by Rust code constructing custom `io::Error`s,
    /// or if this `io::Error` was sourced from the operating system,
    /// it will be a value inferred from the system's error encoding.
    /// See [`last_os_error`] for more details.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self.repr.data() {
            // This can race with a call to `set_functions`, in which case
            // this may return the old or new kind for a given error code.
            ErrorData::Os(errno) => decode_error_kind(errno),
            ErrorData::Simple(kind) => kind,
            ErrorData::SimpleMessage(msg) => msg.kind,
            ErrorData::Custom(c) => c.kind,
        }
    }

    /// Returns `true` if the error indicates the operation was interrupted
    /// and should be retried.
    #[must_use]
    pub fn is_interrupted(&self) -> bool {
        match self.repr.data() {
            ErrorData::Os(errno) => is_interrupted(errno),
            ErrorData::Simple(kind) => kind == ErrorKind::Interrupted,
            ErrorData::SimpleMessage(msg) => msg.kind == ErrorKind::Interrupted,
            ErrorData::Custom(c) => c.kind == ErrorKind::Interrupted,
        }
    }

    // An error is Sync and Send if its custom payload is.
    fn _assert_error_is_sync_send() {
        fn is_sync_send<T: Sync + Send>() {}
        is_sync_send::<Error>();
    }
}

/// Renders the description of an OS error for `Debug`/`Display` output.
///
/// Upstream builds this lazily with `fmt::from_fn`, which is unavailable for
/// this MSRV; a small private struct achieves the same within the pinned MSRV.
struct OsErrorDescription(RawOsError);

impl fmt::Display for OsErrorDescription {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_os_error(self.0, fmt)
    }
}

impl fmt::Debug for OsErrorDescription {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegates to the Display implementation.
        write!(fmt, "\"{self}\"")
    }
}

impl fmt::Debug for Repr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.data() {
            ErrorData::Os(code) => fmt
                .debug_struct("Os")
                .field("code", &code)
                .field("kind", &decode_error_kind(code))
                .field("message", &OsErrorDescription(code))
                .finish(),
            ErrorData::Custom(c) => fmt::Debug::fmt(c, fmt),
            ErrorData::Simple(kind) => fmt.debug_tuple("Kind").field(&kind).finish(),
            ErrorData::SimpleMessage(msg) => fmt
                .debug_struct("Error")
                .field("kind", &msg.kind)
                .field("message", &msg.message)
                .finish(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.repr.data() {
            ErrorData::Os(code) => {
                let detail = OsErrorDescription(code);
                write!(fmt, "{detail} (os error {code})")
            }
            ErrorData::Custom(c) => fmt::Display::fmt(c.error_ref(), fmt),
            ErrorData::Simple(kind) => kind.fmt(fmt),
            ErrorData::SimpleMessage(msg) => msg.message.fmt(fmt),
        }
    }
}

impl core::error::Error for Error {
    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn core::error::Error> {
        match self.repr.data() {
            ErrorData::Os(..) | ErrorData::Simple(..) | ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => c.error_ref().cause(),
        }
    }

    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self.repr.data() {
            ErrorData::Os(..) | ErrorData::Simple(..) | ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => c.error_ref().source(),
        }
    }
}

/// The vtable of OS functions installed by the platform layer.
///
/// Upstream defines this in `core` with `std` as the installer; Olive does
/// the same, with olive-std as the eventual installer.
#[doc(hidden)]
#[derive(Debug)]
pub struct OsFunctions {
    /// Formats the human-readable description of a raw OS error code.
    pub format_os_error: fn(RawOsError, &mut fmt::Formatter<'_>) -> fmt::Result,

    /// Decodes a raw OS error code into an [`ErrorKind`].
    pub decode_error_kind: fn(RawOsError) -> ErrorKind,

    /// Reports whether a raw OS error code represents an interruption.
    pub is_interrupted: fn(RawOsError) -> bool,
}

impl OsFunctions {
    /// The default table, used until the platform layer installs its own.
    ///
    /// FIXME: on platforms without atomics, one should not resort to the error defaulting?
    const DEFAULT: &'static OsFunctions = &OsFunctions {
        format_os_error: |_, _| Ok(()),
        decode_error_kind: |_| ErrorKind::Uncategorized,
        is_interrupted: |_| false,
    };
}

cfg_if::cfg_if! {
    if #[cfg(target_os = "uefi")] {
        type RawOsErrorPrimitive = usize;
    } else if #[cfg(target_pointer_width = "16")] {
        type RawOsErrorPrimitive = i16;
    } else {
        type RawOsErrorPrimitive = i32;
    }
}

/// The type of raw OS error codes.
///
/// This is an [`i32`] on all currently supported platforms, but platforms
/// added in the future (such as UEFI) may use a different primitive type
/// like [`usize`] or [`i16`]. Use `as` or [`Into`] conversions where
/// applicable to ensure maximum portability. Upstream expresses the same
/// ladder with the unstable `cfg_select!` macro.
pub type RawOsError = RawOsErrorPrimitive;

/// A list specifying general categories of I/O errors.
///
/// This list is intended to grow over time, and it is not recommended to
/// exhaustively match against it.
///
/// It is used with the [`io::Error`][error] type.
///
/// The full variant set mirrors nightly upstream, including the kinds upstream
/// still keeps behind unstable features; in Olive they are simply available,
/// since this is Olive's own type.
///
/// [error]: Error
///
/// # Handling errors and matching on `ErrorKind`
///
/// In application code, use `match` for the `ErrorKind` values you are
/// expecting; use `_` to match "all other errors".
///
/// In comprehensive and thorough tests that want to verify that a test doesn't
/// return any known incorrect error kind, you may want to cut-and-paste the
/// current full list of errors from here into your test code, and then match
/// `_` as the correct case. This seems counterintuitive, but it will make your
/// tests more robust. In particular, if you want to verify that your code does
/// produce an unrecognized error kind, the robust solution is to check for all
/// the recognized error kinds and fail in those cases.
#[derive(TryClone, Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An entity was not found, often a file.
    NotFound,
    /// The operation lacked the necessary privileges to complete.
    PermissionDenied,
    /// The connection was refused by the remote server.
    ConnectionRefused,
    /// The connection was reset by the remote server.
    ConnectionReset,
    /// The remote host is not reachable.
    HostUnreachable,
    /// The network containing the remote host is not reachable.
    NetworkUnreachable,
    /// The connection was aborted (terminated) by the remote server.
    ConnectionAborted,
    /// The network operation failed because it was not connected yet.
    NotConnected,
    /// A socket address could not be bound because the address is already in
    /// use elsewhere.
    AddrInUse,
    /// A nonexistent interface was requested or the requested address was not
    /// local.
    AddrNotAvailable,
    /// The system's networking is down.
    NetworkDown,
    /// The operation failed because a pipe was closed.
    BrokenPipe,
    /// An entity already exists, often a file.
    AlreadyExists,
    /// The operation needs to block to complete, but the blocking operation was
    /// requested to not occur.
    WouldBlock,
    /// A filesystem object is, unexpectedly, not a directory.
    ///
    /// For example, a filesystem path was specified where one of the intermediate directory
    /// components was, in fact, a plain file.
    NotADirectory,
    /// The filesystem object is, unexpectedly, a directory.
    ///
    /// A directory was specified when a non-directory was expected.
    IsADirectory,
    /// A non-empty directory was specified where an empty directory was expected.
    DirectoryNotEmpty,
    /// The filesystem or storage medium is read-only, but a write operation was attempted.
    ReadOnlyFilesystem,
    /// Loop in the filesystem or IO subsystem; often, too many levels of symbolic links.
    ///
    /// There was a loop (or excessively long chain) resolving a filesystem object
    /// or file IO object.
    ///
    /// On Unix this is usually the result of a symbolic link loop; or, of exceeding the
    /// system-specific limit on the depth of symlink traversal.
    FilesystemLoop,
    /// Stale network file handle.
    ///
    /// With some network filesystems, notably NFS, an open file (or directory) can be invalidated
    /// by problems with the network or server.
    StaleNetworkFileHandle,
    /// A parameter was incorrect.
    InvalidInput,
    /// Data not valid for the operation were encountered.
    ///
    /// Unlike [`InvalidInput`], this typically means that the operation
    /// parameters were valid, however the error was caused by malformed
    /// input data (which is derived from these parameters).
    ///
    /// For example, a function that reads a file into a string will error with
    /// `InvalidData` if the file's contents are not valid UTF-8.
    ///
    /// [`InvalidInput`]: ErrorKind::InvalidInput
    InvalidData,
    /// The I/O operation's timeout expired, causing it to be canceled.
    TimedOut,
    /// An error returned when an operation could not be completed because a
    /// call to [`write`][write] returned [`Ok(0)`].
    ///
    /// This typically means that an operation could only succeed if it wrote a
    /// particular number of bytes but only a smaller number of bytes could be
    /// written.
    ///
    /// [write]: crate::io::Write::write
    /// [`Ok(0)`]: Ok
    WriteZero,
    /// The underlying storage (typically, a filesystem) is full.
    ///
    /// This does not include out of quota errors.
    StorageFull,
    /// Seek on unseekable file.
    ///
    /// Seeking was attempted on an open file handle which is not suitable for seeking - for
    /// example, on Unix, a named pipe opened with `File::open`.
    NotSeekable,
    /// Filesystem quota or some other kind of quota was exceeded.
    QuotaExceeded,
    /// File larger than allowed or supported.
    ///
    /// This might arise from a hard limit of the underlying filesystem or file access API, or from
    /// an administratively imposed resource limitation.  Simple disk full, and out of quota, have
    /// their own errors.
    FileTooLarge,
    /// Resource is busy.
    ResourceBusy,
    /// Executable file is busy.
    ///
    /// An attempt was made to write to a file which is also in use as a running program.  (Not all
    /// operating systems detect this situation.)
    ExecutableFileBusy,
    /// Deadlock (avoided).
    ///
    /// A file locking operation would result in deadlock.  This situation is typically detected, if
    /// at all, on a best-effort basis.
    Deadlock,
    /// Cross-device or cross-filesystem (hard) link or rename.
    CrossesDevices,
    /// Too many (hard) links to the same filesystem object.
    ///
    /// The filesystem does not support making so many hardlinks to the same file.
    TooManyLinks,
    /// A filename was invalid.
    ///
    /// This error can also occur if a length limit for a name was exceeded.
    InvalidFilename,
    /// Program argument list too long.
    ///
    /// When trying to run an external program, a system or process limit on the size of the
    /// arguments would have been exceeded.
    ArgumentListTooLong,
    /// This operation was interrupted.
    ///
    /// Interrupted operations can typically be retried.
    Interrupted,
    /// This operation is unsupported on this platform.
    ///
    /// This means that the operation can never succeed.
    Unsupported,

    // ErrorKinds which are primarily categorisations for OS error
    // codes should be added above.
    //
    /// An error returned when an operation could not be completed because an
    /// "end of file" was reached prematurely.
    ///
    /// This typically means that an operation could only succeed if it read a
    /// particular number of bytes but only a smaller number of bytes could be
    /// read.
    UnexpectedEof,

    /// An operation could not be completed, because it failed
    /// to allocate enough memory.
    OutOfMemory,

    /// The operation was partially successful and needs to be checked
    /// later on due to not blocking.
    InProgress,

    /// The process or the whole system has reached its limit on the number of
    /// open files or sockets.
    TooManyOpenFiles,

    /// A low-level input/output error.
    ///
    /// This usually indicates a hardware or device-level failure, such as a bad
    /// disk sector or a removed device, but the operating system may also report
    /// it for other low-level I/O conditions.
    InputOutputError,

    // "Unusual" error kinds which do not correspond simply to (sets
    // of) OS error codes, should be added just above this comment.
    // `Other` and `Uncategorized` should remain at the end:
    //
    /// A custom error that does not fall under any other I/O error kind.
    ///
    /// This can be used to construct your own [`Error`][error]s that do not match any
    /// [`ErrorKind`].
    ///
    /// This [`ErrorKind`] is not used by the standard library.
    ///
    /// Errors from the standard library that do not fall under any of the I/O
    /// error kinds cannot be `match`ed on, and will only match a wildcard (`_`) pattern.
    /// New [`ErrorKind`]s might be added in the future for some of those.
    ///
    /// [error]: Error
    Other,

    /// Any I/O error from the standard library (olive-std) that's not part of this
    /// list.
    ///
    /// Errors that are `Uncategorized` now may move to a different or a new
    /// [`ErrorKind`] variant in the future. It is not recommended to match
    /// an error against `Uncategorized`; use a wildcard match (`_`) instead.
    #[doc(hidden)]
    Uncategorized,
}

impl ErrorKind {
    /// Returns the human-readable description of this kind.
    #[allow(clippy::trivially_copy_pass_by_ref, reason = "std uses by-ref passing")]
    const fn as_str(&self) -> &'static str {
        #[allow(clippy::enum_glob_use)]
        use ErrorKind::*;

        match *self {
            // tidy-alphabetical-start
            AddrInUse => "address in use",
            AddrNotAvailable => "address not available",
            AlreadyExists => "entity already exists",
            ArgumentListTooLong => "argument list too long",
            BrokenPipe => "broken pipe",
            ConnectionAborted => "connection aborted",
            ConnectionRefused => "connection refused",
            ConnectionReset => "connection reset",
            CrossesDevices => "cross-device link or rename",
            Deadlock => "deadlock",
            DirectoryNotEmpty => "directory not empty",
            ExecutableFileBusy => "executable file busy",
            FileTooLarge => "file too large",
            FilesystemLoop => "filesystem loop or indirection limit (e.g. symlink loop)",
            HostUnreachable => "host unreachable",
            InProgress => "in progress",
            InputOutputError => "input/output error",
            Interrupted => "operation interrupted",
            InvalidData => "invalid data",
            InvalidFilename => "invalid filename",
            InvalidInput => "invalid input parameter",
            IsADirectory => "is a directory",
            NetworkDown => "network down",
            NetworkUnreachable => "network unreachable",
            NotADirectory => "not a directory",
            NotConnected => "not connected",
            NotFound => "entity not found",
            NotSeekable => "seek on unseekable file",
            Other => "other error",
            OutOfMemory => "out of memory",
            PermissionDenied => "permission denied",
            QuotaExceeded => "quota exceeded",
            ReadOnlyFilesystem => "read-only filesystem or storage medium",
            ResourceBusy => "resource busy",
            StaleNetworkFileHandle => "stale network file handle",
            StorageFull => "no storage space",
            TimedOut => "timed out",
            TooManyLinks => "too many links",
            TooManyOpenFiles => "too many open files",
            Uncategorized => "uncategorized error",
            UnexpectedEof => "unexpected end of file",
            Unsupported => "unsupported",
            WouldBlock => "operation would block",
            WriteZero => "write zero",
            // tidy-alphabetical-end
        }
    }
}

impl fmt::Display for ErrorKind {
    /// Shows a human-readable description of the [`ErrorKind`].
    ///
    /// This is similar to the `Display` impl for [`Error`], but does not
    /// require first converting to an `Error`.
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(self.as_str())
    }
}

impl From<TryReserveError> for Error {
    /// Converts a [`TryReserveError`] into an [`Error`] with
    /// [`ErrorKind::OutOfMemory`].
    ///
    /// The [`TryReserveError`] is not kept as the error's `source()`.
    fn from(_: TryReserveError) -> Error {
        ErrorKind::OutOfMemory.into()
    }
}

impl From<TryCloneError> for Error {
    /// Converts a [`TryCloneError`] into an [`Error`].
    ///
    /// Allocation failures ([`TryCloneError::Reserve`] and
    /// [`TryCloneError::Alloc`]) map to [`ErrorKind::OutOfMemory`], and
    /// logic failures ([`TryCloneError::Other`]) map to
    /// [`ErrorKind::Uncategorized`].
    ///
    /// The [`TryCloneError`] is not kept as the error's `source()`, since
    /// that would require an allocation, which this core cannot perform.
    fn from(err: TryCloneError) -> Error {
        let kind = match err {
            TryCloneError::Reserve(_) | TryCloneError::Alloc(_) => ErrorKind::OutOfMemory,
            TryCloneError::Other(_) => ErrorKind::Uncategorized,
        };
        kind.into()
    }
}

impl From<TryDefaultError> for Error {
    /// Converts a [`TryDefaultError`] into an [`Error`].
    ///
    /// Allocation failures ([`TryDefaultError::Reserve`] and
    /// [`TryDefaultError::Alloc`]) map to [`ErrorKind::OutOfMemory`], and
    /// logic failures ([`TryDefaultError::Other`]) map to
    /// [`ErrorKind::Uncategorized`].
    ///
    /// The [`TryCloneError`] is not kept as the error's `source()`, since
    /// that would require an allocation, which this core cannot perform.
    fn from(err: TryDefaultError) -> Error {
        let kind = match err {
            TryDefaultError::Reserve(_) | TryDefaultError::Alloc(_) => ErrorKind::OutOfMemory,
            TryDefaultError::Other(_) => ErrorKind::Uncategorized,
        };
        kind.into()
    }
}

/// Static I/O function vtable for testing.
/// To avoid incorrect behavior while testing the core library,
/// do not set the I/O function to anything other than this table.
#[cfg(test)]
pub(crate) static TEST_FUNCTIONS: OsFunctions = OsFunctions {
    format_os_error: |_, _| Ok(()),
    decode_error_kind: |code| {
        if code == 2 {
            ErrorKind::NotFound
        } else if code == 4 {
            ErrorKind::Interrupted
        } else {
            ErrorKind::Uncategorized
        }
    },
    is_interrupted: |code| code == 4,
};

#[cfg(test)]
mod tests {
    extern crate std;

    use core::alloc::Layout;

    use super::*;
    use crate::alloc::AllocError;

    #[test]
    fn kind_roundtrip_and_display() {
        let error = Error::from(ErrorKind::NotFound);
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.raw_os_error().is_none());
        assert!(!error.is_interrupted());
        assert!(std::format!("{error}") == "entity not found");
        assert!(std::format!("{:?}", error).contains("Kind"));
    }

    #[test]
    fn const_message_error() {
        const FAIL: Error =
            crate::const_error!(ErrorKind::Unsupported, "tried something that never works");

        assert_eq!(FAIL.kind(), ErrorKind::Unsupported);
        assert!(std::format!("{FAIL}") == "tried something that never works");
    }

    #[test]
    fn raw_os_error_with_functions() {
        // SAFETY: `TEST_FUNCTIONS` is entirely constant.
        let error = unsafe { Error::from_raw_os_error_with_functions(2, &TEST_FUNCTIONS) };
        assert_eq!(error.raw_os_error(), Some(2));
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(!error.is_interrupted());

        // SAFETY: `TEST_FUNCTIONS` is entirely constant.
        let interrupted = unsafe { Error::from_raw_os_error_with_functions(4, &TEST_FUNCTIONS) };
        assert!(interrupted.is_interrupted());
    }

    #[test]
    fn into_custom_returns_self_back() {
        let error = Error::from(ErrorKind::NotFound);
        assert!(error.into_custom().is_err());
    }

    #[test]
    fn try_reserve_error_maps_to_oom() {
        let error = Error::from(TryReserveError::new_capacity_overflow());
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);
    }

    #[test]
    fn try_clone_error_maps_to_kinds() {
        let error = Error::from(TryCloneError::Alloc(AllocError));
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);

        let error = Error::from(TryCloneError::Reserve(
            TryReserveError::new_capacity_overflow(),
        ));
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);

        let error = Error::from(TryCloneError::Reserve(TryReserveError::new_alloc(
            Layout::new::<u64>(),
        )));
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);

        let error = Error::from(TryCloneError::Other("cannot clone an open file"));
        assert_eq!(error.kind(), ErrorKind::Uncategorized);
    }

    #[test]
    fn try_default_error_maps_to_kinds() {
        let error = Error::from(TryDefaultError::Alloc(AllocError));
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);

        let error = Error::from(TryDefaultError::Reserve(
            TryReserveError::new_capacity_overflow(),
        ));
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);

        let error = Error::from(TryDefaultError::Reserve(TryReserveError::new_alloc(
            Layout::new::<u64>(),
        )));
        assert_eq!(error.kind(), ErrorKind::OutOfMemory);

        let error = Error::from(TryDefaultError::Other("cannot default a raw fd"));
        assert_eq!(error.kind(), ErrorKind::Uncategorized);
    }
}
