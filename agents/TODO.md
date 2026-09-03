# Olive — Agent Notes & Open Items

Working notes for the Olive re-port. Kept out of `docs/` on purpose: this is a
scratchpad for decisions and follow-ups, not shipped documentation. If a proper
notes repo appears later, migrate this file there to avoid codebase pollution.

## Strings
FIXME: missing or faulty:
- [ ] drain
- [ ] try_extend_from_within
- [ ] into_chars
- [ ] try_remove (only has one error mode - not on char boundary)
- [ ] try_replace_range (may allocate, should deal with invalid range and char boundary)
- [ ] retain
- [ ] try_split_off (only has one error mode - not on char boundary)
- [ ] try_truncate (only has one error mode - not on char boundary, should supersede truncate)
- [ ] TryToString trait, subtrait of Display, fallibly dumps to string
- [ ] AsMut<str>
- [ ] TryExtend and TryFromIterator for boxed str
- [ ] PartialEq with str and boxed str and vice versa
- [ ] TryFrom for str and Vec<u8>

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