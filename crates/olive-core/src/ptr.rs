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
    /// # Semantics
    ///
    /// A fat pointer is `(data_word, metadata_word)`. This method returns
    /// `(self.data_word, old.metadata_word)` with the mutability of `Self`:
    ///
    /// - For **sized** `U` there is no metadata word; the result is simply
    ///   `self` reinterpreted as `*const U` / `*mut U`.
    /// - For **unsized** `U` the metadata travels from `old`. Crucially it is
    ///   taken from the *source*, not inferred from the destination address — so
    ///   relocating a `dyn Trait`'s vtable onto a different object's storage
    ///   yields a pointer whose dynamic dispatch still resolves to `old`'s type.
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
    /// In practice this means you should clone the payload pointed to by `old`
    /// into the allocation backing `self` *before* calling this method, exactly
    /// as `try_clone_from_ref_in` does.
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
    /// - **Stable path** (`unstable_features` inactive): uses `with_addr`, which
    ///   keeps *`old`*'s provenance and substitutes `self`'s raw address. When
    ///   `self` addresses a *different* allocation than `old`, the resulting
    ///   pointer carries `old`'s provenance over an out-of-range address, which
    ///   Miri rejects. On real hardware this works without problems, but it is technically
    ///   provenance-unsound, so we forbid running it under Miri.
    ///
    /// To prevent silently exercising the unsound branch, `olive-core`
    /// carry a compile-time guard that errors out when compiled
    /// under Miri without `unstable_features`, forcing every Miri build onto the
    /// clean/sound branch above.
    #[must_use]
    unsafe fn cast_with_metadata<U: ?Sized>(self, old: *const U) -> Self::CastedWithMetadata<U>;
}

impl<T: ?Sized> PointerExt<T> for *const T {
    type CastedWithMetadata<U: ?Sized> = *const U;

    #[inline]
    unsafe fn cast_with_metadata<U: ?Sized>(self, old: *const U) -> Self::CastedWithMetadata<U> {
        // Unstable path: `with_metadata_of` keeps *our* (destination) data word and provenance
        // and grafts on `old`'s metadata. Provenance derives from the destination
        // allocation, so this is sound under Miri's strict model. A compile-time
        // guard guarantees Miri builds always take this branch (see the trait
        // docs), so the provenance-unsound fallback below never runs under Miri.
        #[cfg(unstable_features)]
        {
            self.with_metadata_of(old)
        }
        // Stable path: `with_addr` keeps `old`'s provenance but substitutes our
        // raw address. When `self` addresses a different allocation than `old`,
        // the result carries out-of-range provenance — technically unsound, and
        // rejected by Miri. Hence the guard forbids Miri without
        // `unstable_features`. Correct on real hardware for well-formed inputs.
        #[cfg(not(unstable_features))]
        {
            old.with_addr(self.addr())
        }
    }
}

impl<T: ?Sized> PointerExt<T> for *mut T {
    type CastedWithMetadata<U: ?Sized> = *mut U;

    #[inline]
    unsafe fn cast_with_metadata<U: ?Sized>(self, old: *const U) -> Self::CastedWithMetadata<U> {
        // Unstable path: `with_metadata_of` keeps *our* (destination) data word and provenance
        // and grafts on `old`'s metadata. Provenance derives from the destination
        // allocation, so this is sound under Miri's strict model. A compile-time
        // guard guarantees Miri builds always take this branch (see the trait
        // docs), so the provenance-unsound fallback below never runs under Miri.
        #[cfg(unstable_features)]
        {
            self.with_metadata_of(old)
        }
        // Stable path: `with_addr` keeps `old`'s provenance but substitutes our
        // raw address, then `.cast_mut()` recovers mutability. When `self`
        // addresses a different allocation than `old`, the result carries
        // out-of-range provenance — technically unsound, and rejected by Miri.
        // Hence the guard forbids Miri without `unstable_features`. Correct on
        // real hardware for well-formed inputs.
        #[cfg(not(unstable_features))]
        {
            old.with_addr(self.addr()).cast_mut()
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
        assert_eq!(unsafe { (*moved).greet() }, "woof");

        // Symmetrically, relocating `Cat`'s metadata onto `dog`'s address must
        // yield `Cat`'s behavior, confirming the vtable travels with the source
        // pointer rather than being inferred from the destination address.
        let moved_back =
            unsafe { (&dog as *const Dog).cast_with_metadata(src_cat as *const dyn Greet) };
        assert_eq!(unsafe { (*moved_back).greet() }, "meow");
    }
}
