# Olive — Agent Notes & Open Items

Working notes for the Olive re-port. Kept out of `docs/` on purpose: this is a
scratchpad for decisions and follow-ups, not shipped documentation. If a proper
notes repo appears later, migrate this file there to avoid codebase pollution.

## Allocator-attached references
- Rationale: References do not encode where it is accessed from. Sometimes it is mandatory that the reference is accessed from the correct allocator.
An IRQL >= 2 function by definition in Windows cannot access paged pool, so if a normal reference is passed to it the reviewer cannot statically 
verify for invalid IRQL >= 2 accesses. (basically typed provenance)
- Composition: a reference and a PhantomData of the original allocator
- Abilities: Deref, DerefMut, AsRef, AsRefMut, et cetera.
- Caveat: a raw reference without provenance cannot be upgraded to one with provenance without `unsafe`. Data structures with allocators can expose methods 
that construct references with allocator provenance.

## Strings
FIXME: missing or faulty:
- [x] drain
- [x] try_extend_from_within
- [x] into_chars
- [x] try_remove (only has one error mode - not on char boundary)
- [x] try_replace_range (may allocate, should deal with invalid range and char boundary)
- [x] retain
- [x] ~~retain_atomic~~ — RETIRED: redundant with `retain` (which already seals on panic via RetainGuard); the only extra guarantee (byte-for-byte unchanged on failure) wasn't worth an aux allocation + `AllocatorTryClone` bound. Power users can build-then-swap manually.
- [x] try_split_off (only has one error mode - not on char boundary)
- [x] try_truncate (only has one error mode - not on char boundary, should supersede truncate)
- [x] TryToString trait
- [x] AsMut<str>
- [ ] TryExtend and TryFromIterator on String with boxed str items
- [x] PartialEq with str and boxed str and vice versa (+ &str / &Box<str>)
- [x] TryFrom for str (&str) and Vec<u8> (via FromUtf8Error)

For later patches (commits):
- [ ] TryStrExt::try_replace
- [ ] TryStrExt::try_repeat
- [ ] TryStrExt::replacen
- [ ] TryStrExt::try_to_ascii_uppercase
- [ ] TryStrExt::try_to_ascii_lowercase
- [ ] TryStrExt::try_to_uppercase
- [ ] TryStrExt::try_to_lowercase
- [ ] Leave TryStrExt::try_to_casefold_unnormalized as TODO (nightly, low priority)
- [ ] Leave TryStrExt::word_to_titlecase as TODO (nightly, low priority)
- [x] Concat newtype (`string/add.rs`) — results-as-templates builder:
    - [x] `Concat<A>` wraps `Result<String<A>, TryReserveError>`; call `.finish()` to extract
    - [x] `String<A> + Rhs: AsRef<str>` → `Concat<A>` (first step transitions into Concat-world)
    - [x] `Concat<A> + Rhs: AsRef<str>` → `Concat<A>` (chain continues)
    - [x] `Concat<A> + Concat<A>` → `Concat<A>` (merge two builders)
    - [x] `Concat<A> += Rhs: AsRef<str>` (in-place, sticky error on OOM)
    - [x] `From<String<A>>` and `From<Result<String<A>, TryReserveError>>` for ergonomic construction
    - [x] 22 unit tests covering basic ops, chaining, multibyte, OOM, sticky errors, Debug

NOTE: Orphan rule prevents `impl core::ops::Add/AddAssign for Result<String, E>`. Solved by introducing the `Concat` newtype as a local type — it IS the receiver, so standard `Add`/`AddAssign` impls work directly. RHS bound is simply `AsRef<str>`, which covers `&str`, `String<A>`, `&String<A>`, and any future interned string types automatically. No custom marker trait needed since `char` is not part of the SDK's concatenation story. Just write `my_string + "literal"` to get a `Concat`, then chain freely and call `.finish()` when done.

## Iterator (TryIteratorExt)
- try_cloned() -> returns an iterator `TryCloned` that yields `TryCloneError`, needs 

## Vec
FIXME: missing or faulty (checked against stable `std::vec::Vec`; items with a `try_` counterpart already in place are noted so they aren't duplicated):

### Fallible constructors/methods
- [x] ~~try_splice~~ — SKIPPED (deferred indefinitely). No meaningful parity with std's `Vec::splice` is achievable: std's `Splice::drop` writes *back* into the live vector by pushing any unconsumed replacement items, i.e. its `Drop` impl performs an allocation. Our fallible port cannot express a fallible `Drop` (it can't return a `Result`), so the only honest options for an unconsumed tail are silent leak or panic-in-drop — both violate the "allocation failure is recoverable, never fatal" invariant the crate is built on. Any reimplementation that avoids the write-back drop forfeits std-parity anyway. If revisited, the likely shape is a non-iterator API (e.g. `try_replace_range(range, items)` returning the removed elements as a `Vec<T, A>`, mirroring `String::try_replace_range`) rather than a `Splice` iterator.

### Iterator impls (own files, mirroring std's layout)
Per the codebase convention (see `vec/into_iter.rs` + `string/into_chars.rs`: each iterator type lives in its own file, declared `mod <name>;` and re-exported `pub use <name>::<Type>;`), the iterator struct and its `Iterator`/`Drop`/etc. impls belong in a dedicated file — only the thin `try_drain` constructor method lives in `vec/mod.rs`.
- [ ] Verify `retain` / `retain_mut` seal-on-panic parity with String's `RetainGuard` — confirm the guard covers the closure-panic path so the vec is left logically consistent (length restored, dropped elements deallocated exactly once). Add a regression test that panics mid-retention and asserts no double-free / length corruption under Miri.
- [ ] Later patch: `TryExtendFromSlice` coverage audit.