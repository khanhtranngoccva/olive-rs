# Project orientation

Companion to [`README.md`](./README.md). This file is background context — what
the project *is* and how it is built and verified — so an agent (or a new human)
can orient without hunting through manifests.

## What Olive is

Olive is a **fully-fallible re-port of Rust's `std`/`alloc`/`core`.** Every
operation that can fail — above all, any heap allocation — returns a `Result`
instead of panicking, so out-of-memory becomes recoverable rather than fatal.
Allocation failure is *never* fatal; that is the invariant the whole crate is
built on, and it constrains design choices (it is why, e.g., a fallible `Splice`
was skipped outright — see [`TODO.md`](./TODO.md)).

Three crates, mirroring the standard library's own layering:

- **`olive_core`** — `no_std`, dependency-free. Foundational traits, the allocator
  API (`Layout`, `AllocError`, `Allocator`, `Global`), error types
  (`TryReserveError`), and the fallible trait family (`TryClone`,
  `TryFromIterator`, `TryCollect`, `TryExtend`, `TryDefault`). Glob-re-exports the
  entire stable `core` surface.
- **`olive_alloc`** — `no_std` + alloc. A full mirror of the `alloc` crate
  (`Vec`, `String`, `BTreeMap`/`Set`, `Rc`/`Arc`, `Cow`, …) rewritten to allocate
  fallibly. Depends on `olive_core`.
- **`olive_std`** — `std`. The `std`-only surface (hash collections, FFI types,
  sync primitives). Depends on both.

Supporting crates: `olive-macros` (host-side proc macros generating tuple
`TryClone`/`TryDefault` impls) and `olive-build` (build-script helper deciding when
`#![feature(...)]` is permitted). `xtask` is the dev task runner.

## Naming and design conventions

Key conventions (from `olive-core`'s crate docs):

- A fallible counterpart to an infallible-looking std method is prefixed `try_`
  (`try_reserve`, `try_extend`, `try_insert`, …).
- Trait methods inherit the name they override.
- Panics are reserved for logic bugs and should carry descriptive messages; the
  framework tries not to hide failure modes behind panics (with the accepted
  exception of syntactic operations like indexing).
- All collection allocation goes through the `Allocator` trait (default `Global`),
  never free functions — a single swappable seam for custom allocators and OOM
  simulation in tests.

File layout: each iterator type lives in its own file (`vec/into_iter.rs`,
`string/into_chars.rs`, `vec/drain.rs`), declared `mod <name>;` and re-exported
`pub use <name>::<Type>;`; only the thin constructor method lives in the parent
module. Large modules keep their tests in a sibling `tests.rs`. See
[`INCREMENTAL.md`](./INCREMENTAL.md) for how this layout guides partitioning a new
type.

## Lint posture

`clippy::arithmetic_side_effects` is **denied** on non-test builds (per crate) —
arithmetic must use checked/wrapping/saturating ops explicitly, and each justified
suppression carries a `reason = "asserted …"` naming the invariant that makes it
safe. `clippy::pedantic` warns (at least in `olive-core`; check the target crate's
`lib.rs`). `missing_safety_doc` is denied wherever unsafe appears, and
`missing_docs` / `unused_qualifications` warn workspace-wide.

## Build & verify cheat-sheet

| Command | Purpose |
| --- | --- |
| `cargo xtask test` | Unit + integration suite, stable MSRV 1.85. The floor. |
| `cargo xtask miri-test` | Same suite under Miri. Required for unsafe/pointer/allocator changes. |
| `cargo xtask leak-test` | Nightly + leak sanitizer. For changes that might leak. |
| `cargo xtask control-doc` | Generates full HTML docs for std/alloc/core as a completeness reference. |
| `cargo clippy --workspace` | Lint gate (pedantic + denied arithmetic). |
| `cargo fmt --check` | Formatting gate. |

Toolchain is pinned to Rust 1.85 (`rust-toolchain.toml`); nightly-dependent jobs
(Miri, leak sanitizer) install their own nightly.
