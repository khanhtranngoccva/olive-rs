# Project orientation

Companion to [`README.md`](./README.md).

## What Olive is

Olive is a **fully-fallible re-port of Rust's `std`/`alloc`/`core`**: every
operation that can fail — above all, any heap allocation — returns a `Result`
instead of panicking. Allocation failure and many other errors are *never* fatal; that invariant
constrains every design decision.

- **`olive_core`** — `no_std`, dependency-free. The allocator API (`Layout`,
  `AllocError`, `Allocator`, `Global`), error types (`TryReserveError`), and the
  fallible trait family (`TryClone`, `TryFromIterator`, `TryCollect`, `TryExtend`,
  `TryDefault`). Glob-re-exports the stable `core` surface.
- **`olive_alloc`** — `no_std` + alloc. Mirror of the `alloc` crate (`Vec`,
  `String`, `BTreeMap`/`Set`, `Rc`/`Arc`, `Cow`, ...), rewritten to allocate
  fallibly. Depends on `olive_core`.
- **`olive_std`** — `std`. The std-only surface (hash collections, FFI types,
  sync primitives, etc.). Depends on both.
- Supporting: `olive-macros` (host-side proc macros for tuple
  `TryClone`/`TryDefault` impls), `olive-build` (build-script helper gating
  `#![feature(...)]`), `xtask` (task runner).

## Conventions

- A fallible counterpart to an infallible-looking std method is prefixed `try_`
  (`try_reserve`, `try_extend`, `try_insert`, …). Trait methods inherit the name
  they override.
- Panics should be reserved for logic bugs, with descriptive messages. Note that
  certain logic bugs from downstream code may also cause panics in this framework.
- All collection allocation goes through the `Allocator` trait (default
  `Global`), never free functions — one swappable seam for custom allocators and
  OOM simulation in tests.
- Each auxiliary type lives in its own file, declared `mod <name>;` and
  re-exported; see [`INCREMENTAL.md`](./INCREMENTAL.md).
- Relevant implementation blocks should also live in their own file. 

## Lint posture

- `clippy::arithmetic_side_effects` is **denied** on non-test builds: arithmetic
uses checked/wrapping/saturating explicitly, and each suppression carries
`reason = "asserted ..."` naming the invariant that makes it safe.
- `clippy::pedantic` warns (check the target crate's `lib.rs`)
- `missing_safety_doc` is denied wherever unsafe appears.

## Build & verify

| Command | Purpose |
| --- | --- |
| `cargo xtask test` | Unit + integration suite, stable MSRV 1.85. The floor. |
| `cargo xtask miri-test` | Same suite under Miri. Required for unsafe/pointer/allocator changes. |
| `cargo xtask leak-test` | Nightly + leak sanitizer. For changes that might leak. |
| `cargo xtask control-doc` | Full HTML docs for std/alloc/core as completeness reference. |
| `cargo clippy --workspace` | Lint gate. |
| `cargo fmt --check` | Formatting gate. |

Toolchain is pinned to Rust 1.85 (`rust-toolchain.toml`); nightly-dependent jobs
(Miri, leak sanitizer) install their own nightly.
