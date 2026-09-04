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
- [x] TryToString trait (subtrait of Display; `try_to_string()` → Global, `try_to_string_in(alloc)` → generic allocator). Blanket `impl<T: Display> TryToString for T` — works for ANY Display type via `fmt::Write`. Added `impl Write for String<A>` (write_str delegates to push_str_inner, OOM→fmt::Error). **Returns `fmt::Error`, NOT `TryReserveError`** — deliberate: rendering funnels through `core::fmt::write`, which collapses every failure cause (our buffer OOM, the Display impl's own internal allocations failing, or a panic in `Display::fmt`) into the opaque unit `fmt::Error`. No way to distinguish them at this layer, so we return it as-is rather than fabricate a sentinel `TryReserveError`. Doc points callers who need precise reserve diagnostics to `try_push_str`/`try_reserve`. Defined in `string/mod.rs`, exported at crate root. 6 tests incl. blanket-for-non-string + OOM-returns-fmt-error.
- [x] AsMut<str>
- [x] TryExtend and TryFromIterator on String with boxed str items
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