# Testing discipline

Companion to [`README.md`](./README.md); see also [`BUGBOT.md`](./BUGBOT.md) for
the bug classes that motivate much of this.

## Testing aspects

For any container or iterator you touch, cover these aspects:

- **Invariants** — `len <= capacity` survives a mid-operation
  allocation failure; `Rc`/`Arc` strong/weak counts are correct; a
  stalled/resumed iteration loses no elements and drops none twice.
- **Panic safety**: trigger the panic mid-operation, and assert the 
  object is left logically consistent. `PanicArmer`-based abstractions 
  in `panicky_ops` of the `test-helpers` crate can help.
- **Failure paths**: trigger non-fatal failures mid-operation, and assert 
  the object is consistent and the returned error is correct. OOM errors
  should be thoroughly covered.
- **Leak / double-free detection**: prevent leaks and double-frees on successes, 
  failures and panics. Helpful abstractions include the `Ledger` helper, 
  where `all_dropped_once` is the common assertion. It distinguishes leaks
  (an id still live) and double-frees (drop count > 1).
- **Overflow edges**: test empty collections, zero-length slices, ZSTs,
  `usize::MAX`-adjacent capacities, exact boundary indices.
- **Documented bug classes**: check [`BUGBOT.md`](./BUGMOT.md) for common and well-known 
  bug classes.

**Colocate tests with the code they cover**, beside the functions under test; a
split test file stays adjacent to its implementation (`vec/mod.rs` ↔
`vec/tests.rs`). See [`INCREMENTAL.md`](./INCREMENTAL.md).

**Tests needing methods that have not landed yet**: mark them with TODO
directives, and do not remove those TODOs until the methods land.

## Regression discipline

Whenever a bug or bug class is found and fixed, add a regression test reproducing
the scenario, named so its purpose is obvious without reading the body.
`Ledger`-based assertions are the idiomatic way to express "this used to
double-free". When a fix closes a coverage gap noted during incremental landing,
update or remove that note.

## Verification tooling

Via `cargo xtask`:

- `cargo xtask test` — the floor; must pass before anything else.
- `cargo xtask miri-test` — required for any change touching unsafe code, raw
  pointers, `ManuallyDrop`, `ptr::read`/`copy_*`, or allocator lifetimes.
- `cargo xtask leak-test` — when a change could plausibly leak (new `Drop` impl,
  new fallible early-return path, new ownership transfer).

A change is not done until the relevant subset is green. "Compiles and the
happy-path tests pass" is not done for unsafe or fallible code.
