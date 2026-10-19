//! [`TryClone`]: a fallible analogue of [`core::clone::Clone`].

use crate::alloc::AllocError;
use crate::alloc_errors::TryReserveError;
use core::fmt;
use core::ptr;

use super::partial_init::PartialInitGuard;

/// Error returned when a fallible clone operation fails.
#[derive(Clone, PartialEq, Eq)]
pub enum TryCloneError {
    /// A capacity reservation on a collection failed (overflow or OOM).
    Reserve(TryReserveError),
    /// A single heap allocation failed (no reserve phase — e.g. a leaf
    /// allocation such as a `Box`, `Arc`, or `Rc` node).
    Alloc(AllocError),
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Debug for TryCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f.debug_tuple("TryCloneError::Reserve").field(e).finish(),
            Self::Alloc(e) => f.debug_tuple("TryCloneError::Alloc").field(e).finish(),
            Self::Other(msg) => f.debug_tuple("TryCloneError::Other").field(msg).finish(),
        }
    }
}

impl fmt::Display for TryCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "clone failed: {e}"),
            Self::Alloc(_) => write!(f, "clone failed: memory allocation failed"),
            Self::Other(msg) => write!(f, "clone failed: {msg}"),
        }
    }
}

impl core::error::Error for TryCloneError {}

impl From<TryReserveError> for TryCloneError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<AllocError> for TryCloneError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

/// A fallible analogue of [`core::clone::Clone`].
///
/// Unlike [`Clone`], which panics on allocation failure, [`TryClone`] returns a
/// [`Result`] so callers can handle out-of-memory gracefully.
///
/// # Scope
///
/// This trait  works best with lightweight containers (Rc/Arc, etc.),
/// "plain old data" (POD) objects, and other objects that do not invoke
/// any special logic during cloning.
///
/// Complex objects like file handles are generally not suitable for implementing
/// [`TryClone`], and should instead expose custom methods with better error reporting.
///
/// These limitations allow the error mode of [`TryClone`] to be the fixed
/// [`TryCloneError`] return type (rather than an associated type). This
/// enables uniform generic composition and `#[derive(TryDefault)]`
/// support. In practice, clone failures are almost exclusively
/// allocation-related, which the three variants cover.
///
/// # Panic policy
///
/// Transient or domain-level failures (allocation exhaustion, capacity overflow,
/// invalid access, etc.) **must** surface as `Err(TryCloneError)`, never as a panic.
///
/// A panic from `try_clone` is only permissible when it signals an unrecoverable
/// programming error — e.g. a broken internal invariant that should be impossible
/// under correct usage. In that case the bug lies upstream and unwinding to the
/// nearest handler is appropriate.
///
/// Inner values should also be cloned via [`TryClone`] rather than [`Clone`],
/// so that their transient errors propagate through the `Result` channel.
///
/// # Laziness
///
/// If cloning requires allocating memory (e.g. growing a buffer), reserve the
/// backing storage **before** performing any logical work such as recursively
/// cloning inner fields if possible. This way an allocation failure
/// short-circuits early and avoids wasted computation or intermediate values.
pub trait TryClone: Sized {
    /// Attempt to clone `self`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if a capacity reservation or allocation fails.
    /// See the [trait-level panic policy](#panic-policy) for when panics are
    /// permissible versus when errors must be returned.
    fn try_clone(&self) -> Result<Self, TryCloneError>;

    /// Fallibly overwrite `self` with a copy of `source`, mirroring
    /// [`core::clone::Clone::clone_from`] but returning a [`Result`] instead of
    /// panicking on allocation failure.
    ///
    /// The default implementation clones `source` into a fresh value and swaps it
    /// in via [`core::mem::replace`], so it never leaks or double-frees even if
    /// the clone fails midway — on error, `self` is left unchanged. Methods can
    /// override this implementation and write custom logic to improve its performance.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if cloning `source` fails; on that path `self` is
    /// guaranteed to be unchanged. See the [trait-level panic policy](#panic-policy).
    fn try_clone_from(&mut self, source: &Self) -> Result<(), TryCloneError> {
        let new_value = source.try_clone()?;
        // Swap in the fresh value, dropping the old one. On the error path above
        // we never reach here, so `self` is left untouched.
        drop(core::mem::replace(self, new_value));
        Ok(())
    }
}

// Infallible `Copy` primitives: cloning is a bit-for-bit copy with no allocation.
macro_rules! impl_try_clone_copy {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryClone for $t {
                #[inline]
                fn try_clone(&self) -> Result<Self, TryCloneError> {
                    Ok(*self)
                }
            }
        )*
    };
}

impl_try_clone_copy!(u8, u16, u32, u64, u128, usize);
impl_try_clone_copy!(i8, i16, i32, i64, i128, isize);
impl_try_clone_copy!(f32, f64);
impl_try_clone_copy!(bool, char, ());

// Tuples: clone each field left-to-right via [`TryClone`], short-circuiting on
// the first failure. An early error drops exactly the cloned prefix — never the
// whole source. Arity 0 is covered by the unit impl above; arities 1..=16 are
// generated by the `olive-macros` proc macro (a host-side dependency that emits
// tokens at compile time and adds no runtime dependency).
olive_macros::try_clone_tuples!(16);

// Immutable references are always cloneable.
impl<T: ?Sized> TryClone for &T {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(self)
    }
}

impl<T: TryClone> TryClone for Option<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            Some(v) => Ok(Some(v.try_clone()?)),
            None => Ok(None),
        }
    }
}

impl<T: TryClone, E: TryClone> TryClone for Result<T, E> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            Ok(v) => Ok(Ok(v.try_clone()?)),
            Err(e) => Ok(Err(e.try_clone()?)),
        }
    }
}

impl<T: TryClone, const N: usize> TryClone for [T; N] {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut uninit = core::mem::MaybeUninit::<[T; N]>::uninit();
        // SAFETY: `uninit` is aligned for `[T; N]`, hence for `T`, and the
        // backing block holds room for exactly `N` elements.
        let dst = uninit.as_mut_ptr().cast::<T>();
        // Guard rolls back any partially-written prefix on early exit.
        let mut guard = PartialInitGuard::new(dst);
        for (i, elem) in self.iter().enumerate() {
            let cloned = elem.try_clone()?;
            // SAFETY: `i < N`; the slot at `dst + i` is inside the array and
            // has not yet been initialized.
            unsafe {
                ptr::write(dst.add(i), cloned);
            }
            guard.advance();
        }
        // Every element is now initialized; forget the guard so it does not
        // drop the freshly-written data, then hand the value over.
        core::mem::forget(guard);
        // SAFETY: all `N` elements have been initialized by the loop above.
        Ok(unsafe { uninit.assume_init() })
    }
}

// ---------------------------------------------------------------------------
// TryCloneToUninit — cloning a (possibly unsized) value into uninitialized memory
// ---------------------------------------------------------------------------

/// A fallible generalization of [`core::clone::Clone`] to dynamically-sized
/// types stored in arbitrary containers.
///
/// Unlike std's [`CloneToUninit`](core::clone::CloneToUninit),
/// which panics/aborts when a clone fails, [`TryCloneToUninit`] routes every fallible
/// step through [`TryClone`]: a clone that runs out of memory returns
/// [`Err`] instead of unwinding, leaving `dest` in a state the caller can
/// safely discard.
///
/// This is the backer behind `Box::try_clone_from_ref[_in]` and similar functions: the
/// caller allocates a fresh block, then invokes [`Self::try_clone_to_uninit`]
/// to populate it.
///
/// # Safety
/// Implementations must ensure that when `try_clone_to_uninit(dest)` performs,
/// it always leaves *dest fully initialized as a valid value of type Self
/// or fully uninitialized (i.e. completely cleaned up if possible).
pub unsafe trait TryCloneToUninit {
    /// Perform a fallible clone-assignment from `self` to `dest`.
    ///
    /// This is analogous to `ptr::write(dest.cast(), self.clone())`, except that
    /// `Self` may be a dynamically-sized type (`!Sized`) and the operation may
    /// fail.
    ///
    /// Before this function is called, `dest` may point to uninitialized memory.
    /// After it returns `Ok(())`, `dest` points to initialized memory; it will
    /// be sound to create a `&Self` reference from the pointer with the
    /// [pointer metadata](core::ptr::metadata) from `self`.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// * `dest` must be [valid](core::ptr#safety) for writes for
    ///   `size_of_val(self)` bytes.
    /// * `dest` must be properly aligned to `align_of_val(self)`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if a capacity reservation or allocation fails
    /// while cloning.
    ///
    /// On error, `dest` must be treated as uninitialized memory: it must not
    /// be read or dropped, because even if it was previously valid,
    /// it may have been partially overwritten.
    ///
    /// The caller is responsible for deallocating the block pointed to by
    /// `dest` if applicable.
    unsafe fn try_clone_to_uninit(&self, dest: *mut u8) -> Result<(), TryCloneError>;
}

unsafe impl TryCloneToUninit for str {
    #[inline]
    unsafe fn try_clone_to_uninit(&self, dest: *mut u8) -> Result<(), TryCloneError> {
        // `str` is just a `[u8]` with a UTF-8 invariant; a plain byte copy
        // suffices.
        // SAFETY: caller guarantees `dest` is valid for `self.len()` bytes and
        // aligned to 1; `self.as_ptr()` points at `self.len()` readable bytes.
        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), dest, self.len());
        }
        Ok(())
    }
}

unsafe impl TryCloneToUninit for core::ffi::CStr {
    #[inline]
    unsafe fn try_clone_to_uninit(&self, dest: *mut u8) -> Result<(), TryCloneError> {
        // A `CStr` is a `[c_char]` terminated by a NUL byte; its metadata (the
        // length) includes that terminator.
        // SAFETY: caller guarantees `dest` is valid for
        // `to_bytes_with_nul().len()` bytes; the source exposes exactly that
        // many readable bytes.
        unsafe {
            let bytes = self.to_bytes_with_nul();
            ptr::copy_nonoverlapping(bytes.as_ptr(), dest, bytes.len());
        }
        Ok(())
    }
}

unsafe impl<T: TryClone> TryCloneToUninit for T {
    #[inline]
    unsafe fn try_clone_to_uninit(&self, dest: *mut u8) -> Result<(), TryCloneError> {
        let cloned = self.try_clone()?;
        // SAFETY: caller guarantees `dest` is valid for `size_of::<T>()` bytes
        // and aligned to `align_of::<T>()`; writing `cloned` initializes it fully.
        unsafe {
            ptr::write(dest.cast::<T>(), cloned);
        }
        Ok(())
    }
}

unsafe impl<T: TryClone> TryCloneToUninit for [T] {
    #[inline]
    unsafe fn try_clone_to_uninit(&self, dest: *mut u8) -> Result<(), TryCloneError> {
        let len = self.len();
        let dest_elems = dest.cast::<T>();
        // Guard rolls back any partially-written prefix on early exit.
        let mut guard = PartialInitGuard::new(dest_elems);
        #[allow(clippy::needless_range_loop)]
        for i in 0..len {
            let elem = self[i].try_clone()?;
            // SAFETY: index `i` is within the caller-guaranteed-valid block and
            // has not yet been initialized.
            unsafe {
                ptr::write(dest_elems.add(i), elem);
            }
            guard.advance();
        }
        // Every element is now initialized; forget the guard so it does not
        // drop the freshly-written data. Ownership belongs to the destination.
        core::mem::forget(guard);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::testing::{CloneBudget, Ledger, MaybeUnclonable, PanickyClone, TrackedItem};
    use core::mem::MaybeUninit;
    use olive_macros::TryClone;
    use std::format;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;

    #[test]
    fn try_clone_primitives() {
        assert_eq!((42u8).try_clone().unwrap(), 42);
        assert_eq!((-5i32).try_clone().unwrap(), -5);
        assert!(true.try_clone().unwrap());
        assert_eq!('x'.try_clone().unwrap(), 'x');
    }

    #[test]
    fn try_clone_option_and_result() {
        let o: Option<i32> = Some(7);
        assert_eq!(o.try_clone().unwrap(), Some(7));
        let r: Result<i32, bool> = Ok(9);
        assert_eq!(r.try_clone().unwrap(), Ok(9));
    }

    #[test]
    fn try_clone_option_and_result_failure_paths() {
        let none: Option<CloneBudget> = None;
        assert!(none.try_clone().is_ok());

        let some: Option<CloneBudget> = Some(CloneBudget::new(0));
        assert!(some.try_clone().is_err());

        let ok: Result<CloneBudget, u8> = Ok(CloneBudget::new(0));
        assert!(ok.try_clone().is_err());

        let err: Result<u8, CloneBudget> = Err(CloneBudget::new(0));
        assert!(err.try_clone().is_err());
    }

    #[test]
    fn try_clone_tuples() {
        // Spot-check a few arities across the generated range.
        assert_eq!((1i32,).try_clone().unwrap(), (1,));
        assert_eq!((1i32, "two").try_clone().unwrap(), (1, "two"));
        assert_eq!((1u8, 2u8, 3u8).try_clone().unwrap(), (1, 2, 3));
        assert_eq!(
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).try_clone().unwrap(),
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        );
    }

    #[test]
    fn try_clone_tuple_drops_prefix_on_failure() {
        let ledger = Arc::new(Ledger::new());
        // A budget of 2: the first two field clones succeed, the third fails.
        let budget = Arc::new(CloneBudget::new(2));
        let src = (
            TrackedItem::construct(&ledger, budget.share()),
            TrackedItem::construct(&ledger, budget.share()),
            TrackedItem::construct(&ledger, budget.share()),
        );
        let res = src.try_clone();
        assert!(res.is_err());
        // 2 items should succeed and drop.
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        assert_eq!(ledger.drop_count(5), 0);
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.total_allocated(), 5);
        drop(src);
        assert!(ledger.all_dropped_once(0..5));
        assert!(ledger.leaked_ids().is_empty());
    }

    #[test]
    fn try_clone_tuple_panics_drops_prefix_once() {
        let ledger = Arc::new(Ledger::new());
        let src = (
            TrackedItem::construct(&ledger, PanickyClone::new()),
            TrackedItem::construct(&ledger, PanickyClone::new()),
            TrackedItem::construct(&ledger, PanickyClone::new()),
        );
        // Only the third field's clone panics; the first two succeed.
        src.2.inner.arm();
        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = src.try_clone();
        }));
        assert!(res.is_err(), "the armed field clone must panic");
        // 2 items should succeed and drop.
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        assert_eq!(ledger.drop_count(5), 0);
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.total_allocated(), 5);
        drop(src);
        assert!(ledger.all_dropped_once(0..5));
        assert!(ledger.leaked_ids().is_empty());
    }

    #[test]
    fn try_clone_array() {
        let a = [1u8, 2, 3];
        assert_eq!(a.try_clone().unwrap(), [1, 2, 3]);
        let empty: [i32; 0] = [];
        assert_eq!(empty.try_clone().unwrap(), []);
        let zsts: [(); 4] = [(); 4];
        assert_eq!(zsts.try_clone().unwrap(), [(), (), (), ()]);
        // A non-Copy element type exercises the real clone path (olive-core
        // is no_std, so owned heap types are out of scope here).
        struct NC(u8);
        impl TryClone for NC {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                Ok(NC(self.0))
            }
        }
        let elems = [NC(1), NC(2)];
        let cloned = elems.try_clone().unwrap();
        assert_eq!(cloned[0].0, 1);
        assert_eq!(cloned[1].0, 2);
    }

    #[test]
    fn try_clone_array_drops_prefix_on_failure() {
        let ledger = Arc::new(Ledger::new());
        // A budget of 2: the first two element clones succeed, the third
        // fails.
        let budget = Arc::new(CloneBudget::new(2));
        let src = [
            TrackedItem::construct(&ledger, budget.share()),
            TrackedItem::construct(&ledger, budget.share()),
            TrackedItem::construct(&ledger, budget.share()),
        ];
        // 2 items should succeed and drop.
        let res = src.try_clone();
        assert!(res.is_err());
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        assert_eq!(ledger.drop_count(5), 0);
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.total_allocated(), 5);
        drop(src);
        assert!(ledger.all_dropped_once(0..5));
        assert!(ledger.leaked_ids().is_empty());
    }

    #[test]
    fn try_clone_array_panics_drops_prefix_once() {
        let ledger = Arc::new(Ledger::new());
        let src = [
            TrackedItem::construct(&ledger, PanickyClone::new()),
            TrackedItem::construct(&ledger, PanickyClone::new()),
            TrackedItem::construct(&ledger, PanickyClone::new()),
        ];
        // Only the third element's clone panics; the first two succeed.
        src[2].inner.arm();
        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = src.try_clone();
        }));
        assert!(res.is_err(), "the armed element clone must panic");
        // 2 items should succeed and drop.
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        assert_eq!(ledger.drop_count(5), 0);
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.total_allocated(), 5);
        drop(src);
        assert!(ledger.all_dropped_once(0..5));
        assert!(ledger.leaked_ids().is_empty());
    }

    #[test]
    fn try_clone_error_display() {
        let e = TryCloneError::Other("demo");
        assert_eq!(format!("{e}"), "clone failed: demo");
        let r = TryCloneError::Reserve(TryReserveError::new_capacity_overflow());
        assert!(format!("{r}").starts_with("clone failed"));
    }

    #[test]
    fn try_clone_from_overwrites_in_place() {
        let mut a: i32 = 1;
        let b: i32 = 42;
        a.try_clone_from(&b).unwrap();
        assert_eq!(a, 42);
    }

    #[test]
    fn try_clone_from_leaves_self_unchanged_on_error() {
        // A type whose clone always fails must leave `self` untouched.
        let mut a = MaybeUnclonable::clonable(1);
        let b = MaybeUnclonable::unclonable(2);
        let res = a.try_clone_from(&b);
        assert!(res.is_err());
        // `a` still exists and is unchanged (identity preserved by replace).
        assert_eq!(a.inner, 1);
    }

    // --- TryCloneToUninit -------------------------------------------------

    #[test]
    fn try_clone_to_uninit_sized() {
        let src: i32 = 42;
        let mut dest = MaybeUninit::<i32>::uninit();
        // SAFETY: `dest` is a valid, aligned slot for one `i32`.
        unsafe { src.try_clone_to_uninit(dest.as_mut_ptr().cast::<u8>()) }.unwrap();
        assert_eq!(unsafe { dest.assume_init() }, 42);
    }

    #[test]
    fn try_clone_to_uninit_str() {
        let src: &str = "hello";
        let mut buf = [0u8; 5];
        // SAFETY: `buf` holds 5 writable bytes, matching `src.len()`.
        unsafe { src.try_clone_to_uninit(buf.as_mut_ptr()) }.unwrap();
        assert_eq!(core::str::from_utf8(&buf).unwrap(), "hello");
    }

    #[test]
    fn try_clone_to_uninit_slice() {
        let src: &[i32] = &[1, 2, 3];
        let mut buf = [0i32; 3];
        // SAFETY: `buf` holds 3 writable, aligned `i32` slots.
        unsafe { src.try_clone_to_uninit(buf.as_mut_ptr().cast::<u8>()) }.unwrap();
        assert_eq!(buf, [1, 2, 3]);
    }

    #[test]
    fn try_clone_to_uninit_slice_empty() {
        let src: &[i32] = &[];
        let mut buf = [0i32; 0];
        // SAFETY: zero-length write is trivially valid.
        unsafe { src.try_clone_to_uninit(buf.as_mut_ptr().cast::<u8>()) }.unwrap();
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn try_clone_to_uninit_cstr_preserves_nul() {
        let src = c"hi";
        // Buffer must hold the payload plus the terminating NUL.
        let mut buf = [0u8; 3];
        // SAFETY: `buf` holds 3 writable bytes, matching the CStr's length.
        unsafe { src.try_clone_to_uninit(buf.as_mut_ptr()) }.unwrap();
        assert_eq!(&buf[..], b"hi\0");
        // Round-trip through a reconstructed fat pointer to confirm validity.
        let rebuilt = unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(&buf) };
        assert_eq!(rebuilt, src);
    }

    #[test]
    fn try_clone_to_uninit_drops_prefix_on_failure() {
        let ledger = Arc::new(Ledger::new());
        // Only the third element's clone fails; the first two succeed.
        let src = [
            TrackedItem::construct(&ledger, MaybeUnclonable::clonable(5)),
            TrackedItem::construct(&ledger, MaybeUnclonable::clonable(1)),
            TrackedItem::construct(&ledger, MaybeUnclonable::unclonable(9)),
        ];

        let mut buf = [const { MaybeUninit::<TrackedItem<PanickyClone>>::uninit() }; 3];
        unsafe {
            let _e = src[..]
                .try_clone_to_uninit(buf.as_mut_ptr().cast::<u8>())
                .expect_err("the armed element clone must fail");
        }

        // 2 items should succeed and drop.
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        assert_eq!(ledger.drop_count(5), 0);
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.total_allocated(), 5);
        drop(src);
        assert!(ledger.all_dropped_once(0..5));
        assert!(ledger.leaked_ids().is_empty());
    }

    #[test]
    fn try_clone_to_uninit_slice_panics_drops_prefix_once() {
        let ledger = Arc::new(Ledger::new());
        let src = [
            TrackedItem::construct(&ledger, PanickyClone::new()),
            TrackedItem::construct(&ledger, PanickyClone::new()),
            TrackedItem::construct(&ledger, PanickyClone::new()),
        ];
        // Only the third element's clone panics; the first two succeed.
        src[2].inner.arm();

        let mut buf = [const { MaybeUninit::<TrackedItem<PanickyClone>>::uninit() }; 3];
        let res = catch_unwind(AssertUnwindSafe(|| unsafe {
            let _ = src[..].try_clone_to_uninit(buf.as_mut_ptr().cast::<u8>());
        }));
        assert!(res.is_err(), "the armed element clone must panic");

        // 2 items should succeed and drop.
        assert_eq!(ledger.drop_count(3), 1);
        assert_eq!(ledger.drop_count(4), 1);
        assert_eq!(ledger.drop_count(5), 0);
        assert_eq!(ledger.live_ids(), [0, 1, 2]);
        assert!(ledger.double_dropped().is_empty());
        assert_eq!(ledger.total_allocated(), 5);
        drop(src);
        assert!(ledger.all_dropped_once(0..5));
        assert!(ledger.leaked_ids().is_empty());
    }

    #[test]
    fn try_clone_to_uninit_failing_element_leaves_dest_untouched_and_no_leaks() {
        let ledger = Arc::new(Ledger::new());
        type Test = TrackedItem<MaybeUnclonable<i32>>;
        let src: Test = TrackedItem::construct(&ledger, MaybeUnclonable::unclonable(7));

        let mut buf = [0u8; size_of::<Test>()];
        // SAFETY: `buf` is a valid slot of `size_of::<Test>()` bytes for one
        // item; the failed clone writes nothing to it.
        let res = unsafe { src.try_clone_to_uninit(buf.as_mut_ptr()) };
        assert!(res.is_err());
        assert_eq!(ledger.live_ids(), [0]);
        assert_eq!(ledger.drop_count(1), 0);
        // Nothing was written: every byte must still be zero.
        assert_eq!(buf, [0; size_of::<Test>()]);
    }

    // --- Derive: perfect-derive field-type bounds ---------------------------
    mod perfect_derive {
        use super::*;

        /// A fake iterator that exposes the Item for static typing.
        /// It is intentionally non-`TryClone`.
        struct FakeIter<Item>(core::marker::PhantomData<Item>);

        impl<Item> Iterator for FakeIter<Item> {
            type Item = Item;
            fn next(&mut self) -> Option<Self::Item> {
                None
            }
        }

        /// A fake box that mocks the test case where the iterator is not `TryClone` but its item is.
        #[derive(TryClone)]
        struct FakeItemBox<I: Iterator> {
            item: I::Item,
        }

        /// A type that intentionally does NOT implement `TryClone`.
        #[derive(Debug)]
        struct NotCloned(#[expect(dead_code, reason = "only used as a type parameter")] u8);

        /// A container with an *unconditional* `TryClone` impl: cloning it never
        /// requires its parameter to be `TryClone`.
        struct FakeCloneBox<T>(core::marker::PhantomData<T>);

        impl<T> TryClone for FakeCloneBox<T> {
            fn try_clone(&self) -> Result<Self, TryCloneError> {
                Ok(FakeCloneBox(core::marker::PhantomData))
            }
        }

        /// A container that invokes the unconditional `TryClone` behavior even if `I` and `I::Item` is not `TryClone`.
        #[derive(TryClone)]
        struct NestedFakeItemBox<I: Iterator> {
            item: FakeCloneBox<I::Item>,
        }

        // The macro is a perfect derive macro - the FakeIter does not have TryClone, but Item has TryClone,
        // so it works.
        #[test]
        fn derive_binds_associated_type_projection() {
            let b = FakeItemBox::<FakeIter<u32>> { item: 7 };
            assert_eq!(b.try_clone().unwrap().item, 7);
        }

        // Both bounded types (FakeIter and NotCloned) do not implement TryClone, but Boxed has TryClone.
        #[test]
        fn derive_nested_projection_in_unconditional_container() {
            let b = NestedFakeItemBox::<FakeIter<NotCloned>> {
                item: FakeCloneBox(core::marker::PhantomData),
            };
            let NestedFakeItemBox { item: _ } = b.try_clone().unwrap();
        }
    }
}
