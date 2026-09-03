# Olive — Agent Notes & Open Items

Working notes for the Olive re-port. Kept out of `docs/` on purpose: this is a
scratchpad for decisions and follow-ups, not shipped documentation. If a proper
notes repo appears later, migrate this file there to avoid codebase pollution.

## Strings
FIXME: missing or faulty:
- from_utf8, FromUtf8Error, from_utf8_unchecked (this method indeed does not allocate)
- try_from_utf8_lossy, try_from_utf8_lossy_in, try_from_utf8_lossy_owned, try_from_utf8_lossy_owned_in (may allocate)
- try_from_utf16, try_from_utf16_lossy, try_from_utf16le, try_from_utf16le_lossy, try_from_utf16be, try_from_utf16be_lossy (global allocator variant)
- try_from_utf16_in, try_from_utf16_lossy_in, try_from_utf16le_in, try_from_utf16le_lossy_in, try_from_utf16be_in, try_from_utf16be_lossy_in (generic allocator variant)
- from_raw_parts, from_raw_parts_in
- as_mut_str, as_mut_vec, as_str (retire as_ptr method)
- clear, drain
- try_into_boxed_str should supersede into_boxed_str and should call into_boxed_slice
- try_extend_from_within
- try_insert (char variant, should deal with boundary)
- into_raw_parts, into_raw_parts_with_allocator
- into_bytes
- into_chars
- try_push (should deal with char) and try_push_str (should deal with str)
- try_remove (only has one error mode - not on char boundary)
- try_replace_range (may allocate, should deal with invalid range and char boundary)
- retain
- try_split_off (only has one error mode - not on char boundary)
- try_truncate (only has one error mode - not on char boundary)
- TryToString trait, subtrait of Display, fallibly dumps to string
- AsRef<str> and AsMut<str>
- BorrowMut<str>
- Add TryDefault
- DerefMut to `str`
- TryExtend and TryFromIterator for String and boxed str
- PartialEq with str and boxed str and vice versa
- TryFrom for str and Vec<u8>

Retirements:
- *Retire* AddAssign (faulty)
- *Retire* Clone and replace with TryClone
- *Retire* Drop (unnecessary)
- *Retire* TryExtendFromSlice

For later patches (commits):
- StrExt::try_replace
- StrExt::try_repeat
- StrExt::replacen
- StrExt::try_to_ascii_uppercase
- StrExt::try_to_ascii_lowercase
- StrExt::try_to_uppercase
- StrExt::try_to_lowercase
- Leave StrExt::try_to_casefold_unnormalized as TODO (nightly, low priority)
- Leave StrExt::word_to_titlecase as TODO (nightly, low priority)
- Method to_chainable_add (return Ok with dormant TryReserveError err variant)
- impl Add:
    - LHS can be String or Result<String, TryReserveError>
    - RHS can be &str, String, or Result<String, TryReserveError>
    - Returns Result<String, TryReserveError>
- impl AddAssign:
    - LHS must be Result<String, TryReserveError>
    - RHS can be &str, String, or Result<String, TryReserveError>
    - Returns Result<String, TryReserveError>