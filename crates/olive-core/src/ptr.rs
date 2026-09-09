//! Pointer extension APIs layered on top of [`core::ptr`].
//!
//! This module glob-re-exports the entire stable surface of [`core::ptr`] and
//! adds Olive-specific extensions, most notably the [`PointerExt`] trait for
//! relocating a fat pointer's data address while preserving its metadata.

pub use core::ptr::*;

/// Extension methods on raw pointers that are not available on the standard
/// library's inherent pointer API.
///
/// Every method is provided for both `*const T` and `*mut T`; the two impls share
/// identical semantics (a const pointer simply cannot be mutated in place, but
/// these methods only ever *produce* new pointers).
pub trait PointerExt<T: ?Sized>: Sized {
    /// The relocated pointer type: same mutability as `Self`, pointing at `U`.
    type CastedWithMetadata<U: ?Sized>;

    /// Splices the data address of `self` into a copy of `old`, keeping `old`'s
    /// metadata word intact (a slice length, a `str` byte length, or a trait
    /// object's vtable). The result points at wherever `self` points, but with
    /// the shape `old` describes.
    ///
    /// This is the primitive that lets Olive relocate a freshly cloned value
    /// onto its destination allocation while recovering the source's metadata —
    /// see [`Box::try_clone_from_ref_in`](../alloc/boxed/index.html) for the
    /// canonical consumer.
    ///
    /// # Safety
    ///
    /// Both arguments must be valid pointers for their respective types:
    ///
    /// - `self` must be non-null, properly aligned for `T`, and either dangling
    ///   or pointing into a live allocation. It supplies only the data word.
    /// - `old` must be non-null, properly aligned for `U`, and either dangling
    ///   or pointing into a live allocation. It supplies only the metadata word.
    ///
    /// This function performs **no reads or writes**; it merely rearranges the
    /// two halves of a fat pointer. Soundness of dereferencing the *result* is
    /// the caller's responsibility and requires, in addition to the above:
    ///
    /// - the region `size_of_val(old)` bytes long starting at `self`'s address
    ///   is initialized and valid for `U` (the caller has already copied the
    ///   value there), and
    /// - the metadata carried by `old` is consistent with the value now stored
    ///   at `self` (same length / same dynamic type).
    ///
    /// # Miri
    ///
    /// The implementation branches on `cfg(unstable_features)`, and the two
    /// branches behave very differently under Miri's strict provenance model:
    ///
    /// - **Unstable path** (`unstable_features` active — nightly, or a stable
    ///   build with `RUSTC_BOOTSTRAP` set): uses `with_metadata_of`, which keeps
    ///   *this* pointer's data word and provenance and grafts on `old`'s
    ///   metadata. Because provenance comes from the destination allocation,
    ///   the result is sound under Miri. This is the path we want under Miri.
    /// - **Stable path** (`unstable_features` inactive): reconstructs the fat
    ///   pointer byte-by-byte via XOR-mask identification of the data-word
    ///   region. The resulting pointer carries no provenance tag recognizable
    ///   by Miri, so dereferencing it will trigger a provenance violation. We
    ///   therefore forbid running this path under Miri. However, this approach is
    ///   tested to work for Rust's default backend
    ///
    /// To prevent silently exercising the unsound branch, the crate's build script
    /// refuses to compile under a genuine `cargo miri` invocation
    /// unless `unstable_features` is active, forcing every real Miri build onto
    /// the sound branch above.
    #[must_use]
    unsafe fn cast_with_metadata<U: ?Sized>(self, old: *const U) -> Self::CastedWithMetadata<U>;
}

/// The inner implementation of calculating the address field of an arbitrary thin or 
/// fat pointer.
/// 
/// This implementation is ideally called exactly once per type per compilation cycle.
const fn address_word_offset_inner<S: ?Sized>() -> usize
where
    *const S: Sized,
{
    // Strategy: a zero-initialized fat pointer has data=0 and metadata=0.
    // Applying `wrapping_byte_sub(1)` wraps the data word from 0 to !0 (all
    // ones) while metadata remains 0. The first byte position where the two
    // values differ marks the start of the data word.
    use core::mem::{MaybeUninit, size_of_val};
    let template: MaybeUninit<*const S> = MaybeUninit::zeroed();
    let zeroes = unsafe { template.assume_init() };
    // wrapping_byte_sub(1): data word 0→!0, metadata unaffected.
    let ones = zeroes.wrapping_byte_sub(1);
    let size = size_of_val(&zeroes);
    let z = (&raw const zeroes).cast::<u8>();
    let o = (&raw const ones).cast::<u8>();

    let mut i = 0;
    while i < size && unsafe { *z.add(i) } == unsafe { *o.add(i) } {
        #[allow(clippy::arithmetic_side_effects, reason = "asserted i < size")]
        {
            i += 1;
        }
    }
    assert!(i < size, "could not locate data word");
    i
}

/// Determine at compile time the byte offset of the data (address) component
/// within a fat pointer to `S`. The data word is always exactly `size_of::<usize>()`
/// bytes wide; this function locates where it begins.
#[must_use]
pub const fn address_word_offset<S: ?Sized>() -> usize
where
    *const S: Sized,
{
    // Invoke the inner function here. This allows the offset
    // to be computed and stored once for the whole type.
    const { address_word_offset_inner::<S>() }
}

#[cfg(not(unstable_features))]
#[inline(always)]
fn stable_graft<D: ?Sized, U: ?Sized>(dest: *const D, src: *const U) -> *const U {
    use core::mem::{MaybeUninit, size_of};

    // Fast path: thin pointer. No metadata exists; just emit `dest`'s address.
    if size_of::<*const U>() == size_of::<usize>() {
        let mut out: MaybeUninit<*const U> = MaybeUninit::uninit();
        unsafe {
            *out.as_mut_ptr().cast::<usize>() = dest.addr();
        }
        return unsafe { out.assume_init() };
    }

    // Fat pointer path: bulk-copy `src`, overwrite the data word at the
    // compile-time-determined offset with `dest`'s address.
    let data_offset = const { address_word_offset::<U>() };

    // Bulk-copy the entire source pointer (all metadata bytes preserved
    // regardless of width or position), then overwrite the data word.
    let mut out: MaybeUninit<*const U> = MaybeUninit::uninit();
    unsafe {
        out.as_mut_ptr().write(src);
        // Since this write tampers with the pointer's internal contents directly,
        // the pointer no longer holds any compiler provenance data and it 
        // correctly interferes with the compiler's provenance optimization.
        // Calling with_addr() does not achieve the same effect
        #[allow(
            clippy::cast_ptr_alignment,
            reason = "out is a valid location to store the pointer, data_offset is the valid offset 
            pointing to the data field and pre-determined by the compiler"
        )]
        {
            *out.as_mut_ptr().byte_add(data_offset).cast::<usize>() = dest.addr();
        }
    }
    unsafe { out.assume_init() }
}

impl<T: ?Sized> PointerExt<T> for *const T {
    type CastedWithMetadata<U: ?Sized> = *const U;

    #[inline]
    unsafe fn cast_with_metadata<U: ?Sized>(self, old: *const U) -> Self::CastedWithMetadata<U> {
        #[cfg(unstable_features)]
        {
            self.with_metadata_of(old)
        }
        #[cfg(not(unstable_features))]
        {
            stable_graft(self, old)
        }
    }
}

impl<T: ?Sized> PointerExt<T> for *mut T {
    type CastedWithMetadata<U: ?Sized> = *mut U;

    #[inline]
    unsafe fn cast_with_metadata<U: ?Sized>(self, old: *const U) -> Self::CastedWithMetadata<U> {
        #[cfg(unstable_features)]
        {
            self.with_metadata_of(old)
        }
        #[cfg(not(unstable_features))]
        {
            let result: *const U = stable_graft(self.cast_const(), old);
            result.cast_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn slice_relocation_preserves_length() {
        let src: &[i32] = &[10, 20, 30];
        let dest: &mut [i32] = &mut [0, 0, 0];
        // Copy the elements into `dest` so the relocated pointer is sound.
        dest.copy_from_slice(src);
        // Build a fat `*mut [i32]` at `dest`'s base (fresh provenance), then
        // splice in `src`'s length metadata.
        let dest_fat: *mut [i32] = slice_from_raw_parts_mut(dest.as_mut_ptr(), 3);
        let fat = unsafe { dest_fat.cast_with_metadata(src as *const [i32]) };
        // SAFETY: `fat` points at `dest`'s three initialized i32s with len 3.
        let rebuilt = unsafe { &*fat };
        assert_eq!(rebuilt, [10, 20, 30]);
    }

    #[test]
    fn str_relocation_preserves_byte_len() {
        let src: &str = "hello";
        let mut buf = [0u8; 5];
        buf.copy_from_slice(src.as_bytes());
        // Build a fat `*mut str` at `buf`'s base. `*mut [u8]` and `*mut str` share
        // the same layout (data pointer + usize length), so a transmute is sound.
        let bytes: *mut [u8] = slice_from_raw_parts_mut(buf.as_mut_ptr(), 5);
        // SAFETY: `*mut [u8]` and `*mut str` have identical layout (data ptr + usize).
        let dest_fat: *mut str = unsafe { core::mem::transmute(bytes) };
        let fat = unsafe { dest_fat.cast_with_metadata(src as *const str) };
        // SAFETY: `buf` now holds valid UTF-8 of the same length as `src`.
        let rebuilt = unsafe { &*fat };
        assert_eq!(rebuilt, "hello");
    }

    #[test]
    fn sized_relocation_is_plain_pointer() {
        let src: &i32 = &7;
        let mut dest = *src;
        let fat = unsafe { (&mut dest as *mut i32).cast_with_metadata(src as *const i32) };
        // SAFETY: `fat` points at the initialized `dest`.
        assert_eq!(unsafe { *fat }, 7);
    }

    /// Verify that writes through a grafted fat pointer are observable in
    /// memory. This guards against LLVM eliding stores when the grafted
    /// pointer's data word originates from a different allocation than its
    /// metadata source (the provenance-elision bug fixed by the byte-level
    /// reconstruction in `stable_graft`).
    #[test]
    fn write_through_grafted_fat_pointer_is_observable() {
        // A repr(C) struct mimicking RcInner<T>: two usize counters followed
        // by an unsized payload.
        #[repr(C)]
        struct Inner<T: ?Sized> {
            strong: usize,
            weak: usize,
            value: T,
        }

        // Destination: a concrete Inner on the stack. Its payload field is a
        // 4-byte array that we'll treat as a `[u8]` slice via grafting.
        let mut dest = Inner {
            strong: 0,
            weak: 0,
            value: [0u8; 4],
        };

        // Source: a separate slice at a different address. Its length metadata
        // (4) is what gets grafted onto the destination pointer.
        let src_data: [u8; 4] = [10, 20, 30, 40];
        let src: &[u8] = &src_data[..];

        // Thin pointer to `dest.value` (a `[u8; 4]`), cast to `*mut u8` so we
        // can graft slice metadata onto it.
        let payload_base: *mut u8 = (&raw mut dest.value).cast::<u8>();

        // Graft `src`'s length (4) onto `payload_base` → fat `*mut [u8]`.
        // The resulting pointer's data word points at `dest.value` (stack
        // allocation A) while its metadata came from `src` (allocation B).
        // This cross-allocation mismatch is exactly what triggers LLVM's
        // provenance-based store elision if the implementation is wrong.
        //
        // SAFETY: `payload_base` points at `dest.value` (valid, aligned, 4
        // bytes available); `src` is a valid 4-element slice.
        let grafted: *mut [u8] = unsafe { payload_base.cast_with_metadata(src as *const [u8]) };

        // Write through the grafted fat pointer. If LLVM elides this store
        // (believing the pointer targets `src`'s allocation rather than
        // `dest`'s), the assertion below will catch it.
        unsafe {
            let s: &mut [u8] = &mut *grafted;
            s.copy_from_slice(&[10, 20, 30, 40]);
        }

        // Read back through the original stack variable (independent
        // provenance path) to confirm the write landed in memory.
        assert_eq!(dest.value, [10, 20, 30, 40], "payload not written");
    }

    #[test]
    fn dyn_trait_relocation_preserves_vtable() {
        trait Greet {
            fn greet(&self) -> &'static str;
        }
        struct Dog;
        impl Greet for Dog {
            fn greet(&self) -> &'static str {
                "woof"
            }
        }
        struct Cat;
        impl Greet for Cat {
            fn greet(&self) -> &'static str {
                "meow"
            }
        }

        // Two concrete values of different types, each coerced to the same
        // unsized `dyn Greet`. Their fat pointers differ only in the data word
        // (address) — the metadata word holds each type's distinct vtable.
        let dog = Dog;
        let cat = Cat;
        let src_dog: &dyn Greet = &dog;
        let src_cat: &dyn Greet = &cat;

        // Relocate `src_dog`'s fat pointer onto `cat`'s address. The result must
        // keep `Dog`'s vtable (the metadata) while pointing at `cat`'s storage.
        // Because the payload here is a ZST, the data word carries no readable
        // bytes, so dispatch depends entirely on the preserved vtable — which is
        // precisely what this asserts.
        let moved = unsafe { (&cat as *const Cat).cast_with_metadata(src_dog as *const dyn Greet) };
        // SAFETY: `moved` is a valid `dyn Greet` fat pointer: `Dog`'s vtable plus
        // an in-bounds data address. Dispatching through it calls `Dog::greet`.
        // The explicit `&` keeps the reference creation visible rather than
        // relying on an implicit autoref through a raw-pointer deref.
        assert_eq!(unsafe { (&*moved).greet() }, "woof");

        // Symmetrically, relocating `Cat`'s metadata onto `dog`'s address must
        // yield `Cat`'s behavior, confirming the vtable travels with the source
        // pointer rather than being inferred from the destination address.
        let moved_back =
            unsafe { (&dog as *const Dog).cast_with_metadata(src_cat as *const dyn Greet) };
        assert_eq!(unsafe { (&*moved_back).greet() }, "meow");
    }
}
