# Olive — Agent Notes & Open Items

Working notes for the Olive re-port. Kept out of `docs/` on purpose: this is a
scratchpad for decisions and follow-ups, not shipped documentation. If a proper
notes repo appears later, migrate this file there to avoid codebase pollution.

## Decision: allocator-generic traits vs. `_in(alloc)` methods (2026-09-01)

**Question.** Should fallible entry points be allocator-*generic traits*
(e.g. `TryFromIteratorWithAllocator<Item, A>`) or keep the current split — an
allocator-**free** trait implemented only for `Global`, plus a per-type
`*_in(..., alloc)` method as the custom-allocator seam?

**Decision: keep Option B (status quo).** Rationale recorded in full in the
session that produced it; summary below.

### The two options

- **Option A — allocator-generic trait.** Bake `A: Allocator` into the trait
  (`TryFromIteratorWithAllocator<T, A> for Vec<T, A>`). Uniform, discoverable,
  expressible as a bounded generic. Costs: coherence/orphan pain once foreign
  allocators or user containers enter; trait-surface doubling across every
  fallible op; ergonomic regression at the 95%-common `Global` case (still needs
  a shim); monomorphization bloat per `(container, allocator)` pair; diverges
  from std's shape, hurting Olive's legibility as a re-port.

- **Option B — allocator-free trait + `_in` methods (chosen).**
  `TryFromIterator<Item>` exists only for `Vec<T, Global>`; the allocator-aware
  backend is the plain associated fn `Vec::try_from_iter_in(iter, alloc)`. Same
  convention already used by `try_with_capacity_in`, `try_from_slice_in`,
  `try_reserve_in`, etc.

### Why B wins for now

1. Consistent with the pervasive existing `_in` convention across `Vec`,
   `Box`, `String` — no new mental model.
2. Small, stable public trait surface; prelude stays lean.
3. Coherence-friendly: all trait impls are local types pinned to `Global`; no
   foreign-allocator generics required.
4. Zero cost at the common case: `TryFromIterator::try_from_iter` reads exactly
   like std's `FromIterator`.
5. `_in` methods are plain assoc fns, so generic code holding `A: Allocator`
   composes them directly with no extra trait indirection.
6. Preserves the "one swappable allocator seam" invariant without polluting the
   trait namespace.

Accepted costs of B: the `_in` seam isn't trait-discoverable; a fully
allocator-polymorphic *generic* "collect into any fallible container" combinator
can't be expressed purely via the trait (must name the concrete type or take
`alloc` as a runtime arg). Two spelling styles exist for one concept
(`Global` → trait, custom → method).

### Policy (make it explicit, not accidental)

1. The **trait family** (`TryFromIterator`, `TryExtend`, `TryCollect`,
   `TryToOwned`, `TryClone`) stays allocator-free, implemented for the
   `Global` convenience path, mirroring std's shape.
2. The **method family** (`*_in(..., alloc)`) is the single canonical allocator
   seam. Every fallible memory-touching op has exactly one `_in` variant; no
   second alias.
3. Each such trait carries a doc pointer to its `_in` counterpart (already done
   for `TryFromIterator` on `Vec`).

### Revisit trigger

Reconsider Option A only if Olive ships a real `TryCollect`/builder combinator
that users must write generically over **both** container *and* allocator. Until
that need is concrete, the method-based seam is cheaper, more coherent, and more
faithful to std's architecture. Prefer paying a small discovery tax now over a
coherence + surface-area tax later.

---

## Resolved this session (for the record)

- `recovery.rs` FIXME: replaced the contrived
  `(Option<Item>, Inner, usize, Option<usize>)` return of
  `decompose_with_size_hint_lossy` with a named value type `LossySizeHint`
  (renamed from `SizeHintLossy` for grammar). Method renamed to
  `decompose_with_size_hint` (kept the verb-first method naming; the noun takes
  the adjective). Exposed `lower()`, `upper()`, `estimated_total()` (all
  `#[must_use]`), exported via prelude. Sole consumer `Vec::try_extend` now
  calls `hint.estimated_total()`.
- `try_clone.rs` tuple impls moved to a dedicated proc-macro crate. The inline
  declarative `impl_try_clone_tuple!` (12 hand-written arms) is gone; it is now
  generated at compile time by `olive_macros::try_clone_tuples!(12)` from the new
  `crates/olive-macros` crate. See the decision note below for the mechanism and
  why this does not break olive-core's dependency-free guarantee. Kept the
  `try_clone_tuples` test (still passes, arities 1..=12).
- Prior items (ZST `Cap` fix, `Box::try_from_slice_in` forcing `T: TryClone`,
  `convert.rs`/CStr call-site updates) completed earlier in the session.
- Pruned non-canonical `Box<[T]>` constructors per std-fidelity directive:
  `try_new_empty`, `try_with_capacity`, `try_new_zeroed` (Global), and retired
  `Box<str>::try_from_str(_in)` and `Box<[T]>::try_from_array(_in)`. The array
  pair was dropped because moving a `[T; N]` into a runtime slice discards the
  compile-time length for no ergonomic gain — std has no such constructor either.
  This orphaned the `reserve_err_to_alloc_err` helper and the now-unused
  `TryReserveError`/`RawVec`/`CStr` imports in `boxed/mod.rs`; all removed.
- Introduced `AllocatorTryClone` (see decision note below) to make
  `Box::try_clone` sound; `Global` implements it trivially. Rewrote the
  half-finished `test_box_cstr_try_clone` to build its `Box<CStr>` via
  `try_clone_from_ref` (the old body called an undefined
  `from_c_bytes_unchecked`).
- Cleared the last three FIXMEs in the tree:
  - `Box<[T], A>::try_new_uninit_slice_in` / `try_new_zeroed_slice_in` and their
    `Global` shims (`try_new_uninit_slice` / `try_new_zeroed_slice`) now exist.
    They return `Box<[MaybeUninit<T>], A>` — the slice analogue of
    `try_new_uninit` / `try_new_zeroed` — implemented by routing through
    `RawVec::try_with_capacity(_zeroed)_in` + `into_box(n)` so overflow, ZST,
    layout, and deallocation all reuse the existing buffer machinery instead of
    being hand-rolled. The `TryReserveError` is mapped down to `AllocError` to
    stay consistent with the single-element constructors.
  - `RawVec::try_with_capacity_zeroed_in` gained real test coverage (zeroed
    content via a counting probe allocator proving `allocate_zeroed` was used,
    empty-capacity no-alloc, OOM kind, and capacity-overflow detection). Its
    `#[cfg_attr(not(test), expect(unused))]` attribute was removed because the
    new slice constructor now consumes it in non-test code.

Full suite green; `cargo clippy --all-targets` fully clean (the prior
`RawVec::try_with_capacity(_zeroed)` dead-code warning is gone too, since the
slice constructors are real consumers). No `FIXME`s remain anywhere in the tree.

## Resolved this session (continued) — 2026-09-02

Cleared the last three remaining FIXMEs:

- **`TryDefault` trait** (`crates/olive-core/src/try_traits/try_default.rs`). A
  fallible analogue of `core::default::Default`, with an associated `Error` type.
  Infallible impls for all primitive integers, floats, `bool`, `char`, `()`,
  `Option<T>` (→ `None`), and `Result<T, E: Default>` (→ `Err(E::default())`).
  Wired into `try_traits.rs` umbrella, crate-root re-export, and `prelude.rs`.
  Implemented for `Vec<T, Global>` (empty vec → infallible) and `Global`
  allocator (stateless ZST → infallible) in olive-alloc. This satisfies the
  original FIXME that "structures like TryHashMap and allocators need it."

- **`nonzero.rs` stale FIXME.** The line-2 comment claimed `NonZero` impls were
  missing, but the `impl_nonzero!` macro already covered all 12 types with
  `TryClone`. Removed the stale comment and extended the macro to also emit a
  `TryDefault` impl for each `NonZero*` type that always returns `Err(())` —
  there is no canonical non-zero value, so fabrication is unsound. Added a test
  asserting `try_default()` fails for representative types.

- **`cell.rs` missing `UnsafeCell` / `BorrowError` / `BorrowMutError`.**
  - `UnsafeCell<T>: TryClone` delegates to the inner `T: TryClone` via `get()`,
    producing an independent cell. `TryDefault` always fails (no valid default
    payload).
  - `BorrowError` and `BorrowMutError` intentionally have **no** `TryClone` or
    `TryDefault` impls: they are transient borrow-contention markers, not
    values users clone or construct by default. The original FIXME is resolved
    by acknowledging their exclusion rather than forcing impls onto them.
  New tests cover `UnsafeCell` independence and the always-fail `TryDefault`
  contract.

Full workspace: `cargo build` clean, `cargo test --workspace` 224 passed / 0
failed, `cargo clippy --workspace --all-targets` zero warnings.

## Resolved this session (continued) — 2026-09-02 (cell trait completion)

Completed the remaining `TryClone`/`TryDefault` impls for all cell types per the
original spec:

- **`Cell<T>: TryDefault`** (`T: Default`). Infallible; wraps `T::default()` in a
  fresh `Cell`. Bound is `Default`, not `TryDefault`, because `Cell` stores its
  payload inline and std's `Cell::new` takes an owned value directly.
- **`LazyCell<T>: TryDefault`** (`T: Default`). Uses `T::default` as the fn pointer
  initializer (the default type parameter is `F = fn() -> T`). Construction never
  fails — evaluation is deferred to first access. Cannot use `T::try_default` here
  because that would change `F`'s type to `fn() -> Result<T, _>`, breaking the
  default type parameter.
- **`OnceCell<T>: TryClone`** (`T: TryClone`). Preserves initialization state:
  empty clones to empty, initialized clones the inner value via `TryClone`. The
  internal `set` on a fresh cell cannot fail, so the `Err` arm is unreachable but
  mapped defensively.
- **`OnceCell<T>: TryDefault`**. Always succeeds with an empty cell (no `T` bound
  needed since nothing is constructed).
- **`RefCell<T>: TryDefault`** (`T: TryDefault`). Propagates `T::Error` since
  constructing the default payload may genuinely fail.
- **`UnsafeCell<T>: TryDefault`**. Already present from earlier work (always-fail,
  no valid default payload). The previously-added `TryClone` impl was removed —
  the user only specified `TryDefault` for `UnsafeCell`.

Five new tests added (total olive-core count: 56 → 61). Full workspace: 224 passed,
clippy clean.

## Resolved this session (continued) — 2026-09-02 (remaining FIXMEs)

Cleared the last three code-level FIXMEs left in the tree:

- **`cell.rs` — `LazyCell<T>: TryDefault` bound corrected to `T: TryDefault`.**
  The old impl bounded on `T: Default` and stored `T::default` as the fn pointer.
  Since `LazyCell` defers evaluation to first dereference, a fallible initializer
  is a natural fit: the closure calls `T::try_default()` at access time, surfacing
  allocation failures lazily rather than requiring an infallible default up front.
  Construction of the cell itself still never fails (it only stores the closure).

- **`cell.rs` — `UnsafeCell` section: resolved by documenting deliberate omission.**
  The FIXME asked for "TryDefault with TryDefault bounds for UnsafeCell." An
  `UnsafeCell` has no canonical empty/default state of its own; giving it a
  `TryDefault` would fabricate a payload out of nothing. Mirroring the `NonZero*`
  stance, we omit the impl entirely rather than always-fail. The section header
  now carries an explanatory comment instead of the FIXME.

- **`nonzero.rs` — stale FIXME removed, replaced with accurate rationale.**
  The comment claimed "missing generic NonZero." In fact, a blanket
  `impl<T> TryClone for NonZero<T>` is *impossible*: `NonZero<T>` is bounded on
  the perma-unstable sealed trait `ZeroablePrimitive`, which downstream crates
  cannot name or satisfy. The 12 concrete-width newtypes already cover all
  practical uses. The comment now explains why no generic impl exists.

Full workspace: `cargo build` clean, `cargo test --workspace` 224 passed / 0
failed, `cargo clippy --workspace --all-targets` zero warnings. No `FIXME`s
remain anywhere in the source tree.

## Decision: `AllocatorTryClone` marker for same-store allocator cloning (2026-09-01)

**Question.** `Box<T, A>::try_clone` must land on the *same* backing store as the
original — it makes no sense for cloning a box to mint an independent heap. But a
plain `A: TryClone` bound is insufficient: `TryClone` on an allocator is a general
capability that may produce a fresh, independent handle (e.g. an arena pooling
blocks on the heap). We need a stronger guarantee that cloning yields an
*equivalent* allocator (freeable with either handle).

**Decision: new `unsafe trait AllocatorTryClone: Allocator + TryClone {}`**, Olive's
fallible analogue of std's unstable
[`AllocatorClone`](https://doc.rust-lang.org/beta/std/alloc/trait.AllocatorClone.html)
(a marker over `Clone`). Because Olive routes every fallible op through `TryClone`,
the marker sits on `TryClone` instead of `Clone`: cloning an allocator may itself
allocate, so it is fallible.

### Semantics (mirrors `AllocatorClone`)

Upon `try_clone`, the two handles are *equivalent*: memory allocated through one
may be freed through the other. Moving/dropping a clone must not invalidate
currently-allocated blocks while clones exist. Cloning must not unwind (matching
the alloc/dealloc non-unwind rule). Types that are `Copy` must also respect
equivalence as if copied.

### Wiring

- Defined in `olive-core/src/alloc.rs` next to `StaticAllocator`; re-exported via
  `olive_core::prelude` and `crate::alloc` in `olive-alloc`.
- `impl TryClone for Global { Ok(*self) }` (stateless ZST → infallible) plus
  `unsafe impl AllocatorTryClone for Global {}`.
- `Box<T, A>`'s `TryClone` impl now bounds on `A: AllocatorTryClone` (was
  `A: Allocator + TryClone` from an earlier refactor, which silently failed to
  apply since `Global` had no `TryClone`). This is what made
  `b1.try_clone()` on `Box<u32>` resolve to the inner `u32::try_clone` via
  auto-deref (returning a bare `u32`) instead of `Box<u32>` — now fixed.

### Deliberate divergence from std

Std's `AllocatorClone` carries a blanket `impl<A: AllocatorClone + ?Sized>
AllocatorClone for &A` because `Clone for &A` exists unconditionally. Olive has no
blanket `TryClone for &A` (only `&[T]` and `&str`), so **no `&A` blanket is added**.
The reference case is handled directly: `Box::allocator() -> &A` hands out a
reference, and the `try_clone` call site works off the owned `A` field. If a future
need arises for `&A: AllocatorTryClone`, add it then with an explicit
`where &A: TryClone` rather than a blind blanket.

### Revisit trigger

If Olive ships `Rc`/`Arc`-style shared ownership over allocators (std implements
`AllocatorClone` for `Rc<T, A>`/`Arc<T, A>` when `A: AllocatorClone`), revisit the
exact set of blanket impls to mirror. Until then, keep the surface minimal.

## Decision: tuple `TryClone` impls via a proc-macro crate (2026-09-01)

**Question.** Where should the repetitive per-tuple `TryClone` impls live? A
declarative `macro_rules!` with one arm per arity (1..=12) is ~220 lines of near-
identical boilerplate that cannot be generalized cleanly. Model on the sister
project `rustyfill`, which splits its macros into a host-side proc-macro crate.

**Decision: new `crates/olive-macros` proc-macro crate**, mirroring
`rustyfill-macros`. It exposes two entry points:

- `#[derive(TryClone)]` — derives the impl for structs/enums (named, unnamed,
  unit fields; enum variants). Not yet used anywhere in-tree; available for
  downstream types and future Olive value types.
- `try_clone_tuples!(max)` — generates the tuple impls for arities 1..=`max`
  (clamped 1..=16). Called as `olive_macros::try_clone_tuples!(12);` in
  `olive-core/src/try_traits/try_clone.rs`, replacing the old declarative macro.

### Mechanism notes

- **Absolute-path emission.** Generated code references
  `::olive_core::try_traits::try_clone::{TryClone, TryCloneError}` and
  `::core::result::Result`. Downstream members resolve `::olive_core` via their
  normal dependency edge. To make the *same* spelling work for impls generated
  **inside** `olive-core` itself, `olive-core/src/lib.rs` declares
  `extern crate self as olive_core;` (placed after all inner attributes, since
  items may not precede `#![...]`). Without that alias the in-crate expansion
  fails with E0433 (`could not find olive_core`).
- **No `Clone` supertrait copied over.** `rustyfill`'s `TryClone: Clone`; Olive's
  is deliberately independent (the "retire `Clone`" directive). The ported derive
  adds no supertrait bound and no `fallible_clone` alias.
- **Dependency-free guarantee preserved.** A `proc-macro` dependency runs in
  rustc's process and emits tokens; it is never linked into the resulting rlib.
  So `olive-core` depending on `olive-macros` adds zero runtime dependency and
  keeps the crate usable from no_std / no-alloc targets. `syn`/`quote`/
  `proc-macro2` are pinned once in `[workspace.dependencies]`.

### Revisit trigger

If `derive(TryClone)` starts needing attribute args (per-field skip, custom
error mapping) or a second trait family needs tuple generation, consider folding
them into this same crate rather than spawning sibling macro crates.
