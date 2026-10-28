//! Macros for building Olive [`String`]s.
//!
//! [`try_format!`] is the fallible counterpart of the standard library's
//! `format!`: identical syntax, but the output is written into Olive's
//! allocation-fallible [`String`], so an out-of-memory condition surfaces as
//! [`fmt::Error`](core::fmt::Error) instead of a panic or abort.

/// The fallible analogue of the standard library's `format!` macro.
///
/// Renders its arguments exactly as `format!` would (`core::format_args!`
/// syntax, including implicit capture of in-scope variables), but writes the
/// output into an Olive [`String`] allocated by the global allocator and
/// returns:
///
/// - `Ok(String<Global>)` holding the formatted output, or
/// - `Err(fmt::Error)` when the render did not complete.
///
/// As with [`TryToString`], every possible failure (the destination buffer
/// running out of memory, a formatter's own allocation failing, or other errors)
/// is funneled by `core::fmt::write` into the same opaque unit
/// error, so the cause cannot be distinguished here.
///
/// # Examples
///
/// ```
/// use olive_alloc::try_format;
///
/// let who = "world";
/// let s = try_format!("hello, {who}!").expect("allocation succeeds");
/// assert_eq!(s, "hello, world!");
/// ```
///
/// [`String`]: crate::string::String
/// [`TryToString`]: crate::string::TryToString
#[macro_export]
macro_rules! try_format {
    ($($arg:tt)*) => {{
        let mut out = $crate::string::String::new();
        core::fmt::Write::write_fmt(&mut out, core::format_args!($($arg)*))
            .map(|()| out)
    }};
}

/// The allocator-aware counterpart of [`try_format!`].
///
/// Behaves exactly like [`try_format!`], but the destination string is built
/// on the allocator given as the *first* argument, mirroring
/// `TryToString::try_to_string_in`. The return type is
/// `Result<String<A>, fmt::Error>` for the chosen allocator `A`.
///
/// # Examples
///
/// ```
/// use olive_alloc::{alloc::Global, try_format_in};
///
/// let s = try_format_in!(Global, "{} + {} = {}", 1, 2, 1 + 2)
///     .expect("allocation succeeds");
/// assert_eq!(s, "1 + 2 = 3");
/// ```
#[macro_export]
macro_rules! try_format_in {
    ($alloc:expr, $($arg:tt)*) => {{
        let mut out = $crate::string::String::new_in($alloc);
        core::fmt::Write::write_fmt(&mut out, core::format_args!($($arg)*))
            .map(|()| out)
    }};
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::fmt;

    use crate::alloc::CountingAllocator;
    use crate::alloc::Global;
    use crate::string::String;
    use crate::{try_format, try_format_in};
    use olive_test_helpers::FailAlloc;

    #[test]
    fn positional_arguments_render() {
        let s = try_format!("{}-{}", 1, "two").expect("format ok");
        assert_eq!(s, "1-two");
    }

    #[test]
    fn implicit_capture_renders() {
        let who = "world";
        let s = try_format!("hello, {who}!").expect("format ok");
        assert_eq!(s, "hello, world!");
    }

    #[test]
    fn empty_format_yields_empty_string() {
        let s = try_format!("").expect("format ok");
        assert!(s.is_empty());
    }

    #[test]
    fn result_holds_an_olive_string() {
        // Compile-time check of the success type: the global-allocator string.
        let s: String<Global> = try_format!("x").expect("format ok");
        assert_eq!(s, "x");
    }

    #[test]
    fn in_variant_uses_requested_allocator() {
        let s: String<Global> =
            try_format_in!(Global, "{} + {} = {}", 1, 2, 1 + 2).expect("format ok");
        assert_eq!(s, "1 + 2 = 3");
    }

    #[test]
    fn in_variant_allocates() {
        let alloc = CountingAllocator::default();
        let s = try_format_in!(&alloc, "hello").expect("format ok");
        assert_eq!(s, "hello");
        // Exactly one reservation backs the five rendered bytes.
        assert!(alloc.allocations() >= 1);
    }

    #[test]
    fn allocation_failure_is_reported_as_fmt_error() {
        // `fmt::Error` is a unit struct; the type annotation is the assertion.
        let e: fmt::Error = try_format_in!(FailAlloc, "hello").expect_err("should fail");
        let _ = e;
    }
}
