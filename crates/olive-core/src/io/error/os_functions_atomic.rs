//! OS-function plumbing with atomic support.
//!
//! Upstream `core` stores the platform's [`OsFunctions`] vtable in a global
//! atomic pointer that `std` installs during startup, so that `core` can
//! render OS error strings and decode error kinds without any link-time
//! dependency on the platform layer.
//!
//! Dispatch costs one relaxed atomic load. Olive mirrors that design verbatim.

use super::{ErrorKind, OsFunctions, RawOsError};
use core::fmt;
use core::ptr::NonNull;
use core::sync::atomic;

/// These default functions are not reachable, but have them just to be safe.
static OS_FUNCTIONS: atomic::AtomicPtr<OsFunctions> =
    atomic::AtomicPtr::new(core::ptr::from_ref(OsFunctions::DEFAULT).cast_mut());

fn get_os_functions() -> &'static OsFunctions {
    unsafe {
        // SAFETY:
        //  * `OS_FUNCTIONS` is initially a pointer to `OsFunctions::DEFAULT`, which is valid for a static lifetime.
        //  * `OS_FUNCTIONS` can only be changed by `set_functions`, which only accepts `&'static OsFunctions`.
        //  * Therefore, `OS_FUNCTIONS` must always contain a valid non-null pointer with a static lifetime.
        //  * `Relaxed` ordering is sufficient as the only way to write to `OS_FUNCTIONS` is through
        //    `set_functions`, which has as a safety precondition that any value passed in must
        //    be constant and not created during runtime.
        &*OS_FUNCTIONS.load(atomic::Ordering::Relaxed)
    }
}

/// Sets the platform's I/O error machinery.
///
/// This function stores the provided table in the global atomic pointer.
///
/// # Safety
///
/// The provided reference must point to data that is entirely constant; it must
/// not be created during runtime.
pub(super) unsafe fn set_functions(f: &'static OsFunctions) {
    #[cold]
    fn set_functions_inner(f: &'static OsFunctions) {
        OS_FUNCTIONS.store(NonNull::from(f).as_ptr(), atomic::Ordering::Relaxed);
    }

    if OS_FUNCTIONS.load(atomic::Ordering::Relaxed) != NonNull::from(f).as_ptr() {
        set_functions_inner(f);
    }
}

/// Formats the human-readable description of a raw OS error code.
#[inline]
pub(super) fn format_os_error(errno: RawOsError, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    let functions = get_os_functions();
    (functions.format_os_error)(errno, fmt)
}

/// Decodes a raw OS error code into an [`ErrorKind`].
#[inline]
pub(super) fn decode_error_kind(errno: RawOsError) -> ErrorKind {
    let functions = get_os_functions();
    (functions.decode_error_kind)(errno)
}

/// Reports whether a raw OS error code represents an interruption.
#[inline]
pub(super) fn is_interrupted(errno: RawOsError) -> bool {
    let functions = get_os_functions();
    (functions.is_interrupted)(errno)
}
