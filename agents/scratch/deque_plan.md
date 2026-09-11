# VecDeque Implementation Plan

Incremental, reviewable steps for landing `VecDeque` in `olive-alloc`. Each step
compiles independently with green tests; no intermediate state leaves the crate
broken.

Steps follow the dependency graph: invariants → getters → construction →
mutation → reconstitution. A method is only implemented once every behavior it
depends on exists. Tests that depend on not-yet-implemented methods are left as
explicit `TODO(deque-step-N)` markers and filled in when the dependency lands.

## Representation (fixed)

Match std's actual `VecDeque` representation: `head` + `len`, where `head` is
a **wrapped index** type rather than a raw `usize`. Storing `len` directly
gives O(1) length access without the `tail`-vs-`head` comparison dance, and
the "exactly full" state is unambiguous (`len == cap`), so no special case is
needed in `Drop`, length derivation, or push/pop.

```rust
struct WrappedIndex(usize)

impl WrappedIndex {
    fn new(val: usize) -> Self { ... }
    const fn zero() -> Self
    fn add(self, rhs: usize) -> Self { ... }
    fn sub(self, rhs: usize) -> Self { ... }
    fn as_index(self) -> usize
    fn is_zero() -> bool
    fn abs_diff() -> bool
}

pub struct VecDeque<T, A: Allocator = Global> {
    buf: RawVec<T, A>,
    head: WrappedIndex, // physical index of the first element; 0 <= val < bound when len > 0
    len: usize,         // number of live elements; invariant: len <= capacity()
}
```

Rules for `WrappedIndex`:
- Operations to the index (including creation) must keep the index less than the capacity
- If the capacity is 0, the index can be 0

Physical slot of logical index `i` is `(head + i) % cap` when `cap > 0`;
in practice code advances `head` via `add_wrapped(1)` / `sub_wrapped(1)`
rather than recomputing modulo.

ZST rules (apply everywhere below):
- For ZSTs, `buf.capacity()` reports `usize::MAX` but the stored cap is 0 and
  the pointer is dangling. All index arithmetic must be gated on
  `size_of::<T>() == 0`; never do `% cap` or `ptr.add(_)` for ZSTs — the
  "slot" is always the same dangling address.
- Growth for ZSTs is impossible by design: any operation that would need to
  grow a ZST buffer returns `Err(CapacityOverflow)` (this is what
  `RawVecInner::grow_*` already does).

Arithmetic discipline: every unchecked index computation carries a per-site
`#[allow(clippy::arithmetic_side_effects, reason = ...)]` whose reason cites
the invariant that makes it safe (mirrors the existing `Drop` impl).

## Step 1 — Module skeleton + struct definition (PARTIAL — needs revision)

The module exists at `crates/olive-alloc/src/collections/vec_deque/mod.rs`
with a raw-`usize` `head` field and `Send`/`Sync` impls. **Revise it** to the
representation above: introduce `WrappedIndex` and change `head` to that
type. The existing `Drop` impl (Step 2) must be updated to use
`head.into_inner()`; its two-region logic is otherwise unchanged.

Verification: `cargo check -p olive-alloc` passes after the refactor.

## Step 2 — Drop invariant (landed; update to `WrappedIndex`)

The existing `Drop` impl is correct in shape; after the Step 1 refactor it
reads `let head = self.head.into_inner();` at the top. Logic:

Live elements occupy up to two contiguous regions starting at `head`:
- Non-wrapped (`head + len <= cap`): single region `[head .. head+len)`.
- Wrapped: `[head .. cap)` then `[0 .. head+len-cap)`.

Algorithm:
1. If `cap == 0 || len == 0`, return early.
2. Compute `end = head + len` (safe: `head < cap`, `len <= cap` ⇒
   `end < 2*cap < usize::MAX`).
3. If `end <= cap`: one `drop_in_place` over `[head..end)`.
4. Else: `drop_in_place([head..cap))` then `drop_in_place([0..end-cap))`.

Panic safety: `self.len` is decremented *before* each `drop_in_place` call
(the wrapped case sets `len = cap - head` after the first region, then `0`
after the second), matching the `Vec::truncate` pattern, so if a destructor
panics during unwind, the tracked range already excludes destroyed elements
and nothing is double-freed.

Note: the currently landed `Drop` does **not** yet decrement `len` before
dropping — add that as part of the Step 1 refactor.

Verification: ledger-based tests asserting every element drops exactly once
for non-wrapped, wrapped, and exactly-full deques.
TODO(deque-step-4): constructing a wrapped/exactly-full deque needs
`push_front`/`push_back`; until then test via a private constructor used only
by tests (or defer these cases wholesale to Step 4).

## Step 3 — Getters

Add to `crates/olive-alloc/src/collections/vec_deque/mod.rs`:

- `len()`, `is_empty()` — read `self.len` directly.
- `capacity()` — `buf.capacity()` (reports `usize::MAX` for ZSTs, matching
  `Vec`).
- `as_ptr()` / `as_mut_ptr()` — raw pointer to the *logical* start:
  `buf.ptr().add(head.into_inner())` when `cap > 0 && !ZST`, else
  `buf.ptr()` (dangling when empty). No modulo needed: `WrappedIndex` keeps
  its value normalized into `0..bound` by construction.
- `allocator()` — forward to `buf.allocator()`.
- `as_slices()` — `(front_slice, back_slice)` split at the wrap point;
  `(&[], &[])` when empty. This is the primitive later steps build on.

Dependencies: none beyond Step 1.

Verification: unit tests for zero-capacity edge case, ZST handling, and
pointer identity against `try_with_capacity` buffers.
TODO(deque-step-5): `as_slices` ordering assertions need a populated deque;
cover those cases in Step 5's tests.

## Step 4 — Construction behaviors

- `new()` / `new_in(alloc)` — empty deque, no allocation
  (`RawVec::new_in`, `head = 0`, `len = 0`).
- `try_with_capacity(cap)` / `try_with_capacity_in(cap, alloc)` — fallible
  pre-allocation via `RawVec::try_with_capacity_in`; `head = 0`, `len = 0`.
  Overflow surfaces as `TryReserveError::CapacityOverflow` from `RawVec`.
- `Default` / `TryDefault` impls (delegate to `new` / fallible note:
  `Default::default` can't fail; `TryDefault` mirrors `try_with_capacity(0)`
  which also can't fail — implement both as infallible delegations).

Dependencies: Step 3 (getters exist so tests can assert on results).

Verification: construction tests, zero-capacity edge case, ZST handling
(`try_with_capacity(N)` succeeds without allocating for ZSTs), overflow
returns `CapacityOverflow`.

## Step 5 — Mutation: push / pop

Core mutation methods. Shared internal helpers first:

- `fn reserve_for_push(&mut self) -> Result<(), TryReserveError>` — grows the
  buffer when `len == cap`. **Must not use `RawVec::try_grow_one`**: its
  precondition is `len == capacity`, and for a fresh ZST buffer the stored cap
  is 0 while `capacity()` reports `usize::MAX`, sending it down the
  overfull/overflow path. Instead call `buf.try_reserve(self.len, 1)`
  (amortized growth, handles first-allocation and ZST uniformly). After a
  successful grow, relocate live elements into the new buffer rotated so
  `head == 0` (matching std's strategy), rebuild `head` as
  `WrappedIndex::new(0, new_cap)`, and refresh `bound`. Relocation uses
  `ptr::copy_nonoverlapping` over the same two-region split as `Drop`.
- `fn maybe_shrink_on_pop(&mut self)` — after a pop, if
  `len < cap / 4 && cap / 4 >= MIN_NON_ZERO_CAP` (where
  `MIN_NON_ZERO_CAP = RawVec::MIN_NON_ZERO_CAP`), attempt
  `buf.try_shrink_to_fit(new_cap)` with `new_cap = max(MIN_NON_ZERO_CAP,
  len.saturating_mul(2))` capped sensibly. On shrink failure: silently keep
  the old buffer (std-equivalent behavior; pops never report OOM). Before
  shrinking, rotate live elements so `head == 0` (shrink replaces the base
  pointer, so indices must be rebuilt relative to it); afterwards rebuild
  `head` with the new bound.

Public API (crate convention: `try_` prefix for fallible ops):

- `try_push_back(&mut self, value: T) -> Result<(), TryReserveError>` — write
  at slot `head.add_wrapped(len).into_inner()` (guarded: ZST writes to
  `buf.ptr()`), advance `len`. Grow via `reserve_for_push` if `len == cap`.
- `try_push_front(&mut self, value: T) -> Result<(), TryReserveError>` —
  retreat `head` via `sub_wrapped(1)` (no-op slot-wise for ZSTs), write
  there, advance `len`. Grow if full.
- `pop_front(&mut self) -> Option<T>` — read at slot `head.into_inner()`,
  advance `head` via `add_wrapped(1)`, decrement `len` *before* `ptr::read`
  (panic-safe, matches `Vec::pop`). Call `maybe_shrink_on_pop`.
- `pop_back(&mut self) -> Option<T>` — decrement `len` first, read at slot
  `head.add_wrapped(len).into_inner()`. Call `maybe_shrink_on_pop`.

Return-type rule: pushes return `Result<(), TryReserveError>`; pops return
plain `Option<T>` because their only potential allocation (opportunistic
shrink) is best-effort and failures are swallowed. Document this choice in
each pop's doc comment.

Dependencies: Steps 2–4 (two-region logic shared with `Drop`; getters for
tests; constructors to build fixtures).

Verification: round-trip tests (push/pop all four corners), OOM simulation
with `FailAlloc` (push fails cleanly, deque unchanged), ledger tests for
drop-on-grow-failure (element handed to a failed push is moved back / not
dropped twice — decide: `try_push_*` takes ownership and drops `value` on
failure, documented accordingly), shrink-threshold tests, ZST push/pop
behavior, Miri run.

## Step 6 — Iteration

- `iter()` → yields `&T` in logical order (front-to-back), wrapping at the
  buffer boundary. Custom iterator holding `&VecDeque`, current physical
  index, remaining count; built on `as_slices()` semantics.
- `iter_mut()` similarly.
- `IntoIterator for &VecDeque`, `&mut VecDeque`, and owned `VecDeque`.

Dependencies: Step 5 (wrapped-state fixtures require `push_front`/`push_back`).

Verification: iterate over non-wrapped, wrapped, and exactly-full deques;
verify order and counts; `count()` vs `len()`.

## Step 7 — From iterators / extend

- `try_from_iter_in(iter, alloc)` — uses size hint to pre-allocate via
  `try_with_capacity_in`, falls back to grow-one-per-element as needed.
- `TryFromIterator` impl for `VecDeque<T, Global>`.
- `try_extend` via the crate's `TryExtend` trait.

Element cloning where required uses fallible `TryClone` (crate-wide
convention); a clone failure mid-extend rolls back via `truncate` (Step 8)
— until `truncate` exists, TODO(deque-step-8) and roll back by popping.

Dependencies: Step 5 (growth machinery), Step 8 (`truncate` for rollback).

Verification: collect tests, under-hinted iterators, over-hinted iterators,
OOM mid-extend (deque holds whatever was pushed before the failure), Miri run.

## Step 8 — Remaining API surface

- `clear()` — drop all, `len = 0`, `head = 0` (infallible).
- `truncate(n)` — drop elements from logical index `n` onward; panic-safe
  `len`-first pattern like `Vec::truncate`. Infallible.
- `Debug`, `Display`? (skip Display), `Eq`/`PartialEq`/`Hash`/`Ord` blanket
  impls mirroring `Vec`.
- Explicitly out of scope for now: `Deref`/`DerefMut` (std does not implement
  them for `VecDeque` because the logical slice may wrap — keep skipped),
  `swap_remove_front/back` (not part of std's `VecDeque` API).

Dependencies: Step 5 (everything here delegates to push/pop/truncate
primitives).

Verification: each method gets targeted tests before moving on.

## Step 9 — Reconstitution

- `into_raw_parts(self) -> (*mut T, usize /*head*/, usize /*len*/)`.
- `unsafe fn from_raw_parts(ptr: *mut T, head: usize, len: usize) -> Self`
  (Global allocator), plus `_in` variants taking an `A`. Both rebuild the
  `WrappedIndex` internally (`bound = buf.capacity()`); the public triple
  stays plain `usize`s so callers don't depend on a private type.
- Safety contract: `head < capacity` (when `len > 0`), `len <= capacity`,
  buffer allocated through the given allocator; document that the caller owns
  dropping the contained elements.

Dependencies: Step 3 (getters, so the triple is derivable consistently),
Step 4 (constructors establish the canonical initial state).

Verification: round-trip `from_raw_parts(into_raw_parts(dq))` preserves
contents/order (needs Step 5 for populated fixtures — TODO(deque-step-5)
until then, test with empty/ZST deques), Miri run.

## Testing strategy (applies to all steps)

- Use `Ledger` + `FlakyTrackedItem` for drop-invariant verification.
- Use `FailAlloc` for OOM paths.
- Use `LocalCountingAlloc` to verify the allocator is dropped exactly once.
- Wrap panic-prone operations in `catch_unwind` + `AssertUnwindSafe` and assert
  ledger consistency afterwards.
- Run under Miri for any step involving raw pointer manipulation.
- Any test whose fixture requires a method from a later step is marked
  `TODO(deque-step-N)` and added when that step lands; each step's test suite
  must be green (modulo those deferred markers) before the next step starts.
