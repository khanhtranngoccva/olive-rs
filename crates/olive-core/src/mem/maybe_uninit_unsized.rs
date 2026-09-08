//! [`MaybeUninitUnsized`] — a [`core::mem::MaybeUninit`] shim for unsized types.

use core::fmt;
use core::mem::ManuallyDrop;

/// A [`core::mem::MaybeUninit`] that works with unsized types (`?Sized`).
///
/// [`core::mem::MaybeUninit<T>`] is declared as `pub union MaybeUninit<T>` with an
/// implicit `Sized` bound on `T`, so it cannot represent an uninitialized value of
/// a dynamically sized type (a slice, `str`, `dyn Trait`, …). This type fills that
/// gap.
///
/// # Layout
///
/// Like std's `MaybeUninit`, this is a `#[repr(transparent)]` wrapper whose sole
/// payload is a [`core::mem::ManuallyDrop<T>`]. Because a transparent single-field
/// wrapper inherits its field's size, alignment, *and* fat pointer metadata if any,
/// a fat pointer of type `*const T` casts losslessly to `*const Self` (and vice versa):
/// both are fat pointers of identical width carrying the same metadata word.
///
/// # Rationale
///
/// This wrapper is mainly used to semantically denote a potentially unsized and
/// uninitialized block of memory.
///
/// A common use case of this wrapper is to reinterpret a fat pointer over an uninitialized
/// or partially unitialized `T` as a *shared reference* to this wrapper, which then computes
/// the pointee's layout through a real reference rather than a raw fat pointer.
/// No bytes of the pointee are ever read.
#[repr(transparent)]
pub struct MaybeUninitUnsized<T: ?Sized> {
    /// Payload carrying `T`'s full layout identity (size, alignment, metadata)
    /// while suppressing `T`'s destructor, exactly like `MaybeUninit<T>` does.
    inner: ManuallyDrop<T>,
}

impl<T: ?Sized> MaybeUninitUnsized<T> {
    /// Reinterprets a shared reference as an arbitrary (possibly
    /// uninitialized) memory region carrying `T`'s layout and metadata as a
    /// shared reference to this wrapper.
    #[inline]
    pub const fn from_ref(ptr: &T) -> &Self {
        // SAFETY: `Self` is `#[repr(transparent)]` over `ManuallyDrop<T>`, which
        // has `T`'s exact layout including metadata; the cast preserves
        // provenance and touches no byte.
        unsafe { &*(core::ptr::from_ref(ptr) as *const Self) }
    }

    /// Reinterprets a mutable reference as an arbitrary (possibly
    /// uninitialized) memory region carrying `T`'s layout and metadata as a
    /// mutable reference to this wrapper.
    #[inline]
    pub const fn from_mut(ptr: &mut T) -> &mut Self {
        // SAFETY: see [`from_ref`](Self::from_ref); the cast preserves
        // provenance and touches no byte.
        unsafe { &mut *(core::ptr::from_mut(ptr) as *mut Self) }
    }

    /// Reinterprets a raw pointer as an arbitrary (possibly uninitialized)
    /// memory region carrying `T`'s layout and metadata, yielding a shared
    /// reference to this wrapper.
    ///
    /// This is the named, documented form of `unsafe { &*ptr }`: it makes the
    /// preconditions explicit rather than burying them in an anonymous cast at
    /// the call site.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr` is valid for reads of a `T` for the
    /// duration `'a` (properly aligned, non-null, with correct metadata), and
    /// that the memory is not mutated during `'a` (except inside `UnsafeCell`).
    pub const unsafe fn from_ptr<'a>(ptr: *const T) -> &'a Self {
        // SAFETY: caller guarantees `ptr` satisfies the reference contract for
        // `'a`; the cast is a lossless fat-pointer re-tag touching no byte.
        unsafe { &*(ptr as *const Self) }
    }

    /// Reinterprets a raw pointer as an arbitrary (possibly uninitialized)
    /// memory region carrying `T`'s layout and metadata, yielding a mutable
    /// reference to this wrapper.
    ///
    /// This is the named, documented form of `unsafe { &mut *ptr }`: it makes
    /// the preconditions explicit rather than burying them in an anonymous cast
    /// at the call site.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr` is valid for reads and writes of a
    /// `T` for the duration `'a` (properly aligned, non-null, with correct
    /// metadata), and that no other reference or pointer alias to the same
    /// memory not derived from this reference is used during `'a`.
    #[inline]
    pub const unsafe fn from_mut_ptr<'a>(ptr: *mut T) -> &'a mut Self {
        // SAFETY: caller guarantees `ptr` satisfies the unique-borrow contract
        // for `'a`; the cast is a lossless fat-pointer re-tag touching no byte.
        unsafe { &mut *(ptr as *mut Self) }
    }

    /// Reinterprets a raw pointer as an arbitrary (possibly uninitialized)
    /// memory region carrying `T`'s layout and metadata, yielding a raw pointer
    /// to this wrapper.
    ///
    /// Unlike [`from_ptr`](Self::from_ptr), this does not establish a borrow.
    #[inline]
    pub const fn ptr_from_ptr(ptr: *const T) -> *const Self {
        ptr as *const Self
    }

    /// Reinterprets a raw pointer as an arbitrary (possibly uninitialized)
    /// memory region carrying `T`'s layout and metadata, yielding a raw pointer
    /// to this wrapper.
    ///
    /// Unlike [`from_mut_ptr`](Self::from_mut_ptr), this does not establish a
    /// borrow.
    #[inline]
    pub const fn ptr_from_mut_ptr(ptr: *mut T) -> *mut Self {
        ptr as *mut Self
    }

    /// Returns a mutable pointer to the contained value.
    ///
    /// It is undefined behavior to turn this into a reference unless the value is fully initialized.
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        (&raw mut self.inner) as *mut T
    }

    /// Returns a pointer to the contained value.
    ///
    /// Use [`assume_init_ref`](Self::assume_init_ref) if you need a typed
    /// `&T` instead.
    ///
    /// # Safety
    ///
    /// It is undefined behavior to turn this into a reference unless the value is fully initialized.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        (&raw const self.inner) as *const T
    }

    /// Drops whatever value was contained in `self`, if any.
    ///
    /// # Safety
    ///
    /// It is your responsibility to ensure that this object actually contains
    /// an initialized `T`.
    #[inline]
    pub unsafe fn assume_init_drop(&mut self) {
        // SAFETY: caller guarantees the value is initialized; we read it out
        // and let normal drop glue run on the temporary.
        unsafe {
            let ptr = self.as_mut_ptr();
            core::ptr::drop_in_place(ptr);
        }
    }

    /// Interprets `self` as a reference.
    ///
    /// # Safety
    ///
    /// It is your responsibility to ensure that self is actually initialized.
    #[inline]
    pub unsafe fn assume_init_ref(&self) -> &T {
        // SAFETY: caller guarantees initialization; `as_ptr` carries `T`'s
        // metadata correctly (see its docs).
        unsafe { &*(self.as_ptr()) }
    }

    /// Interprets self as a mutable reference.
    ///
    /// # Safety
    ///
    /// It is your responsibility to ensure that self is actually initialized.
    #[inline]
    pub unsafe fn assume_init_mut(&mut self) -> &mut T {
        // SAFETY: caller guarantees initialization; see `assume_init_ref`.
        unsafe { &mut *(self.as_mut_ptr()) }
    }
}

impl<T: ?Sized> fmt::Debug for MaybeUninitUnsized<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Mirror std: print only the type name, never touch contents, so this
        // is safe even when the wrapper is uninitialized.
        f.debug_struct("MaybeUninitUnsized").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::{Layout, LayoutExt};

    /// The canonical consumer idiom: route an unsized fat pointer through
    /// [`LayoutExt::for_value_pointer`] and recover the pointee's layout. This
    /// exercises the stable (non-bootstrap) path that casts `*const T` to
    /// `*const MaybeUninitUnsized<T>` and reads the layout off a reference —
    /// proving the wrapper carries `T`'s metadata correctly.
    #[test]
    fn for_value_pointer_recovers_slice_layout() {
        let data: [u8; 4] = [1, 2, 3, 4];
        let slice: &[u8] = &data[..];
        let ptr: *const [u8] = slice as *const [u8];
        // SAFETY: `ptr` points at four initialized u8s with correct length
        // metadata; no bytes need be read to compute the layout.
        let layout = unsafe { Layout::for_value_pointer(ptr) };
        assert_eq!(layout.size(), 4);
        assert_eq!(layout.align(), 1);
    }

    /// Same idiom for a `dyn Trait` payload, confirming the vtable-bearing
    /// metadata survives the round-trip through the wrapper.
    #[test]
    fn for_value_pointer_recovers_dyn_trait_layout() {
        trait Marker {}
        struct Thing;
        impl Marker for Thing {}
        let thing = Thing;
        let dyn_ptr: *const dyn Marker = &thing as *const dyn Marker;
        // SAFETY: `dyn_ptr` points at a valid `Thing` with a correct vtable.
        let layout = unsafe { Layout::for_value_pointer(dyn_ptr) };
        // `Thing` is a ZST, so its layout is size 0 / align 1.
        assert_eq!(layout.size(), 0);
        assert_eq!(layout.align(), 1);
    }

    /// `from_ref` + `as_ptr` must round-trip the data address and metadata of an
    /// unsized value: reinterpreting a `&[u8]` as a wrapper and back must yield
    /// a pointer indistinguishable from the original.
    #[test]
    fn from_ref_round_trips_metadata() {
        let data: [i32; 3] = [10, 20, 30];
        let slice: &[i32] = &data[..];
        let orig: *const [i32] = slice as *const [i32];
        // `slice` is a valid, aligned reference with correct metadata.
        let wrapped = MaybeUninitUnsized::from_ref(slice);
        // Reading the pointer back does not dereference the pointee.
        let recovered = wrapped.as_ptr();
        assert_eq!(recovered.addr(), orig.addr());
        // Metadata (length) must match too. Explicit `&` keeps the reference
        // creation visible to the linter rather than relying on an implicit
        // autoref through a raw-pointer deref.
        assert_eq!(unsafe { (&*recovered).len() }, 3);
    }

    /// `from_mut` + `assume_init_mut` must recover the same mutable reference
    /// (same address and metadata) that was passed in.
    #[test]
    fn from_mut_round_trips_metadata() {
        let mut data: [i32; 3] = [10, 20, 30];
        let orig_addr = (&mut data[..] as *mut [i32]).addr();
        // SAFETY: `&mut data[..]` is a valid, aligned reference with correct metadata.
        let wrapped = MaybeUninitUnsized::from_mut(&mut data[..]);
        // SAFETY: the backing array is fully initialized.
        let recovered = unsafe { wrapped.assume_init_mut() };
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered.as_mut_ptr().addr(), orig_addr);
    }

    /// `from_ptr` / `from_mut_ptr` now yield references directly (the named,
    /// documented form of `&*ptr`). The returned wrapper reference must alias
    /// the exact memory the input pointer addressed, carrying the same data
    /// word and metadata, so routing it through `as_ptr` recovers the original
    /// pointer bit-for-bit.
    #[test]
    fn from_ptr_yields_aliasing_reference() {
        let mut data: [u8; 4] = [1, 2, 3, 4];
        let slice: &[u8] = &data[..];
        let ptr: *const [u8] = slice as *const [u8];

        // SAFETY: `ptr` is valid for reads of `[u8]` for this scope, aligned,
        // with correct length metadata; no conflicting alias occurs here.
        let wrapped: &MaybeUninitUnsized<[u8]> = unsafe { MaybeUninitUnsized::from_ptr(ptr) };
        // The wrapper ref must point at the same memory as the input pointer.
        let recovered = wrapped.as_ptr();
        assert_eq!(recovered.addr(), ptr.addr());
        // Metadata (length) must survive the round-trip.
        assert_eq!(unsafe { (&*recovered).len() }, 4);

        let mptr: *mut [u8] = &mut data[..] as *mut [u8];
        // SAFETY: `mptr` is valid for reads/writes of `[u8]` for this scope,
        // uniquely aliased; no other reference to `data` is live here.
        let mwrapped: &mut MaybeUninitUnsized<[u8]> =
            unsafe { MaybeUninitUnsized::from_mut_ptr(mptr) };
        let mrecovered = mwrapped.as_mut_ptr();
        assert_eq!(mrecovered.addr(), mptr.addr());
        assert_eq!(unsafe { (&*mrecovered).len() }, 4);
    }

    /// `ptr_from_ptr` / `ptr_from_mut_ptr` are the borrow-free raw-pointer
    /// forms: pure fat-pointer re-tags whose result carries exactly the data
    /// word and metadata of the input, and casts back losslessly.
    #[test]
    fn ptr_from_ptr_retags_losslessly() {
        let mut data: [u8; 4] = [1, 2, 3, 4];
        let slice: &[u8] = &data[..];
        let ptr: *const [u8] = slice as *const [u8];

        // SAFETY: `ptr` is a valid, aligned `*const [u8]`.
        let wrapped: *const MaybeUninitUnsized<[u8]> = MaybeUninitUnsized::ptr_from_ptr(ptr);
        // Round-trip back to `*const [u8]` and compare both words.
        let back: *const [u8] = wrapped as *const [u8];
        assert_eq!(back.addr(), ptr.addr());
        // SAFETY: `back` aliases `data`, which is alive and initialized.
        assert_eq!(unsafe { (&*back).len() }, 4);

        let mptr: *mut [u8] = &mut data[..] as *mut [u8];
        // SAFETY: `mptr` is a valid, aligned `*mut [u8]`.
        let mwrapped: *mut MaybeUninitUnsized<[u8]> = MaybeUninitUnsized::ptr_from_mut_ptr(mptr);
        let mback: *mut [u8] = mwrapped as *mut [u8];
        assert_eq!(mback.addr(), mptr.addr());
        // SAFETY: `mback` aliases `data`, which is alive and initialized.
        assert_eq!(unsafe { (&*mback).len() }, 4);
    }
}
