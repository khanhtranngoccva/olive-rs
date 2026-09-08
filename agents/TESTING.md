# Testing discipline

Companion to [`README.md`](./README.md). See also [`INCREMENTAL.md`](./INCREMENTAL.md)
(for when and how tests land alongside a type) and [`BUGBOT.md`](./BUGBOT.md) (for
the bug classes that motivate much of this).

## Exhaustive, and aimed at invariants

Tests must be **exhaustive for every documented behavior and invariant**, and they
must test *against the invariants of the type itself* — not merely against the
happy-path outputs of a few calls. In a fallible re-port of the standard library,
the interesting failures are almost never "did it return the right value"; they
are "after this failed operation, is the object still internally consistent?".
That is what the tests exist to pin down.

For any container or iterator you touch, the tests should cover:

- **Every documented behavior.** If the doc comment promises a particular result,
  error variant, or side effect, there is a test that exercises exactly that
  promise. An undocumented-but-implemented behavior is a smell; either document
  it and test it, or remove it.
- **The type's invariants under failure.** This is the heart of the suite. For
  `Vec` that means `len <= capacity` survives a mid-operation allocation failure;
  for `Rc`/`Arc` that means the strong/weak counts always reconcile; for any
  fallible iterator that means a stalled/resumed iteration loses no elements and
  drops none twice. Assert the invariant directly, not just the returned value.
- **OOM and clone-failure paths.** Use the shared harness in
  [`test_helpers.rs`](../crates/olive-alloc/src/test_helpers.rs): `FailAlloc` to
  force every allocation to fail, `FlakyClone` / `BudgetedFlaky` to place a
  `try_clone` failure at a deterministic point inside a long operation, and
  `LocalCountingAlloc` to watch the allocator instance itself. Failure-in-the-middle
  of an operation is where leaks and double-frees live.
- **Panic safety.** Where a guard seals the object against a closure panic
  (e.g. `RetainGuard`), arm a `PanicArmer`, trigger the panic mid-operation, and
  assert the object is left logically consistent afterward.
- **Leak / double-free detection.** Use the `Ledger` helper rather than ad-hoc
  counters. It registers each payload id as live, records its drop count, and can
  distinguish all three failure classes a container bug introduces after an abort:
  leaks (an id still live), double-frees (drop count > 1), and wrong totals.
  `all_dropped_once` is the common assertion.
- **Edge cases that overflow:** empty collections, zero-length slices, ZSTs,
  `usize::MAX`-adjacent capacities, and the exact boundary indices. Several real
  bugs in this project were pointer/range overflows that only surface at these
  edges (and easily on 16-bit targets — see `BUGBOT.md`).

**Colocate unit tests with the code they cover.** Unit tests for a method block live in
the same file as those functions, right beside them; do not scatter a block's tests
into an unrelated pile. File partitioning exists to make this possible — a cohesive
block gets its own file so its implementation and its tests stay together and are
reviewable as one unit. Where a module's tests are split out for size, keep the
split file adjacent to the implementation it verifies (`vec/mod.rs` ↔
`vec/tests.rs`). See [`INCREMENTAL.md`](./INCREMENTAL.md) for how this shapes the
order in which a type lands.

## Regression discipline

Whenever a bug or a bug *class* is found and fixed, add a regression test that
reproduces the scenario, named so its purpose is obvious without reading the body.
The `Ledger`-based assertions above are the idiomatic way to express "this used to
double-free." When a fix closes a coverage gap that was noted during incremental
landing (see [`INCREMENTAL.md`](./INCREMENTAL.md)), update or remove that note so
the tracker doesn't accumulate stale holes.

## Verification tooling

Run via `xtask`, aliased as `cargo xtask …`:

- `cargo xtask test` — the ordinary unit + integration suite on the pinned stable
  toolchain (MSRV 1.85). This is the floor; it must pass before anything else.
- `cargo xtask miri-test` — runs the suite under Miri for undefined-behavior
  detection. **Required for any change that touches unsafe code, raw pointers,
  `ManuallyDrop`, `ptr::read`/`copy_*`, or allocator lifetimes.** Most of the bugs
  catalogued in `BUGBOT.md` are UB that Miri catches deterministically.
- `cargo xtask leak-test` — nightly + `-Zsanitizer=leak`. Run this when a change
  could plausibly leak memory (new `Drop` impl, new fallible early-return path,
  new ownership transfer).

Do not declare a change done until the relevant subset of the above is green.
"Compiles and the happy-path tests pass" is not done for unsafe or fallible code.
