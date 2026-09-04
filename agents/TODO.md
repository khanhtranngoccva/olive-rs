# Olive — Agent Notes & Open Items

Working notes for the Olive re-port. Kept out of `docs/` on purpose: this is a
scratchpad for decisions and follow-ups, not shipped documentation. If a proper
notes repo appears later, migrate this file there to avoid codebase pollution.

## Strings
FIXME: missing or faulty:
- [ ] drain
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
- [ ] Method to_chainable_add for String (returns Ok with dormant TryReserveError err variant)
- [ ] impl Add:
    - [ ] LHS can be String or Result<String, TryReserveError>
    - [ ] RHS can be &str, String, or Result<String, TryReserveError>
    - [ ] Returns Result<String, TryReserveError>
- [ ] impl AddAssign:
    - [ ] LHS must be Result<String, TryReserveError>
    - [ ] RHS can be &str, String, or Result<String, TryReserveError>
    - [ ] Returns Result<String, TryReserveError>

## Vec
FIXME: missing or faulty (checked against stable `std::vec::Vec`; items with a `try_` counterpart already in place are noted so they aren't duplicated):

### Fallible constructors/methods
- [x] try_drain — DONE. `vec/mod.rs` + `vec/drain.rs`. 13 unit tests.
- [x] ~~try_splice~~ — SKIPPED (deferred indefinitely). No meaningful parity with std's `Vec::splice` is achievable: std's `Splice::drop` writes *back* into the live vector by pushing any unconsumed replacement items, i.e. its `Drop` impl performs an allocation. Our fallible port cannot express a fallible `Drop` (it can't return a `Result`), so the only honest options for an unconsumed tail are silent leak or panic-in-drop — both violate the "allocation failure is recoverable, never fatal" invariant the crate is built on. Any reimplementation that avoids the write-back drop forfeits std-parity anyway. If revisited, the likely shape is a non-iterator API (e.g. `try_replace_range(range, items)` returning the removed elements as a `Vec<T, A>`, mirroring `String::try_replace_range`) rather than a `Splice` iterator.
- [x] try_split_off — DONE. Requires `A: Clone`. Uses `set_len` (not `truncate`) + `copy_nonoverlapping` matching std's bitwise-transfer semantics. 7 unit tests.
- [x] try_extend_from_within — DONE. Requires `T: TryClone`. Reserves first, then clones elements one-by-one into the reserved tail. 7 unit tests.

### Iterator impls (own files, mirroring std's layout)
Per the codebase convention (see `vec/into_iter.rs` + `string/into_chars.rs`: each iterator type lives in its own file, declared `mod <name>;` and re-exported `pub use <name>::<Type>;`), the iterator struct and its `Iterator`/`Drop`/etc. impls belong in a dedicated file — only the thin `try_drain` constructor method lives in `vec/mod.rs`.
- [x] `vec/drain.rs` — DONE. `Drain<'a, T, A>` with full iterator impls + corrected Drop (destroys unconsumed remainder, shifts suffix via `ptr::copy`, adjusts len by `original_count`).
- [x] ~~append~~ — already present as `try_append(other: &mut Self) -> Result<(), TryReserveError>` (vec/mod.rs:1601). No action.
- [x] ~~insert~~ — already present as `try_insert` / `try_insert_give_back` / `try_insert_mut(_give_back)` (vec/mod.rs:557–593). No action.
- [x] ~~remove / swap_remove~~ — already present as `try_remove` / `try_swap_remove` (vec/mod.rs:1106, 1139). No action.
- [x] ~~resize / reserve* / shrink_* / into_boxed_slice / into_array~~ — all already have `try_` variants (see vec/mod.rs). No action.
- [ ] Verify `retain` / `retain_mut` seal-on-panic parity with String's `RetainGuard` — confirm the guard covers the closure-panic path so the vec is left logically consistent (length restored, dropped elements deallocated exactly once). Add a regression test that panics mid-retention and asserts no double-free / length corruption under Miri.
- [ ] Later patch: `TryExtendFromSlice` coverage audit — `impl TryExtendFromSlice<'s, T> for Vec<T, A>` exists (vec/mod.rs:1965); confirm it composes correctly with `try_from_iter_in` and that the `'s` lifetime doesn't leak into the error type. Low priority.