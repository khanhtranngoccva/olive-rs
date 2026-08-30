/// Re-exports of [`core::mem`] API items.
pub use core::mem::*;

// Source: https://docs.rs/crate/oct/0.39.0, needs MIT citation.
// This allows conversion between types that can't be verified by the compiler to
// have exactly equal size.
/// Unsafely transmutes an object to another type.
///
/// The source value is reinterpreted as-is as an object of the destination type.
/// Padding bytes are not preserved, but the initialisation states of
/// [`MaybeUninit`] fields are.
///
/// [`MaybeUninit`]: core::mem::MaybeUninit
///
/// This function is massively unsafe but also allows for much more (sound) usage
/// than [`transmute`] and [`core::mem::transmute`].
///
/// # Safety
///
/// The following guarantees must be upheld when transmuting objects:
///
/// * `U` and `T` must be of the exact same size.
/// * Any uninitialised byte in `value` must also be permitted as uninitialised by
///   `U`.
/// * Any initialised byte in `value` must have value that is also permitted by
///   `U`.
/// * `value` -- if a pointer -- was initially transmuted from an integer.
///
/// A violation of any of these requirements results in undefined behaviour. See
/// also the documentation for [`core::mem::transmute`].
#[inline(always)]
#[must_use]
#[cfg_attr(miri, track_caller)]
pub const unsafe fn transmute_unchecked<T, U>(value: T) -> U {
    #[repr(C)]
    union Transmute<Src, Dst> {
        src: ManuallyDrop<Src>,
        dst: ManuallyDrop<Dst>,
    }

    // Wrap the object in the union.
    let transmute = Transmute {
        src: ManuallyDrop::new(value),
    };

    // Reread the object as the destination type.
    // SAFETY: Caller guarantees correct representation.
    unsafe { ManuallyDrop::into_inner(transmute.dst) }
}
