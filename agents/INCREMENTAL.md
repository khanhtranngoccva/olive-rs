# Incremental change discipline

This file is the playbook for *how big* a change may be and *in what order* work
lands. It complements [`README.md`](./README.md) (the index), [`TESTING.md`](./TESTING.md),
[`DOCUMENTATION.md`](./DOCUMENTATION.md), and [`PROJECT.md`](./PROJECT.md).

## Why small changes win here

Changes should be small and reviewable. A request touching more than ~10 items in
a row is very likely to fail from context rot: as a single session grows, the
model's grasp of the surrounding invariants degrades, the diff balloons past what
a person can verify line by line, and the result costs *more* maintenance time
than doing the work in focused passes would have. This is not a soft preference —
it is the single most important rule in this repo, because the cost of a bad large
change is asymmetric: it risks breaking an invariant that only Miri or a careful
reviewer will catch later.

Keep changes incremental so that:

- Each unit of work stays within what one reviewer can fully verify.
- Every intermediate state compiles, tests green, and leaves the crate's
  invariants intact. Nothing half-finished gets committed.
- If something goes wrong, the blast radius is one small step, not a hundred-line
  refactor tangled across several types.

The codebase is **human-first**. A change set that only makes sense to the model
that produced it is a defect, even if it compiles. Favor the shape a maintainer
could reconstruct from the diff alone.

If the user genuinely wants a large experiment — a broad sweep, a speculative
rewrite, a multi-crate migration — do not run it as one giant linear pass.
Partition it into independent subtasks and drive them through subagents, each
scoped to a narrow slice (one type, one module, one error path). Give each
subagent the relevant slice of these files plus the specific invariant(s) it must
preserve, collect its output, and integrate incrementally. The point is to keep
every individual context window small enough to stay coherent while still covering
the whole experiment.

Practical defaults:

- One logical concern per change. If you find yourself editing five unrelated
  methods to "clean up" while implementing one feature, split it.
- Prefer adding over rewriting. Porting a new method next to its siblings is far
  safer than restructuring the neighbors to make room.
- When a task does span several files, sequence it so each file's change is
  independently compilable and testable before moving to the next.

---

## Landing a new type: partition the implementation

Porting a type is the recurring large task in this project, and it is where
incremental discipline pays off most. Do **not** land a type as one blob. Land it
in two phases, in this order.

### Phase 1 — structure and `Drop` first (the starting invariants)

Land the type's **structure** (its fields and their meanings) and its **`Drop`
behavior** before anything else. Together these form the *starting invariants*:
they define what it means for an instance to be valid, and they establish how the
type tears itself down correctly. Everything later builds on top of this, so it
must be right and tested before the surface area grows. For a large type, decide
now where the definition will physically live (module root or its own file — see
[Partitioning the implementation into files](#partitioning-the-implementation-into-files)),
because the structure and `Drop` you land here are exactly the pieces that may
deserve their own file.

In practice this phase delivers:

- The struct definition with its documented field invariants (e.g. `Vec`:
  `len <= capacity`; `Rc`: strong/weak counts reconcile and the block outlives
  every live reference).
- The `Drop` impl, written against those invariants.
- The minimal constructors needed to build a valid instance so `Drop` can be
  exercised at all (typically `new` / `default`, plus whatever allocation setup
  `Drop` relies on).
- Tests proving `Drop` behaves correctly under the invariants — no leak, no
  double-free, correct deallocation layout — using the harness described in
  [`TESTING.md`](./TESTING.md) (`Ledger`, `LocalCountingAlloc`, `FailAlloc`).

**Note gaps explicitly.** At this stage other methods are not implemented yet, so
`Drop` testing necessarily has *insufficient variant coverage* — one cannot reach
every way the object could be mutated because the mutating methods don't exist.
Do not paper over this. Immediately record, right next to the `Drop` tests, 
exactly which variants are untested and *why* (which method hasn't landed yet). 
The note should name the candidate missing method and the specific drop-path it 
would exercise, so that when the method lands, its author
knows to close the gap.

Only once Phase 1 compiles, tests green (including under Miri for the unsafe
teardown), and the coverage-gap note is in place, move on.

### Phase 2 — methods in chunks of 5–10 related items

After the basic invariants are locked, land the type's methods in **chunks of
5–10 related items per block**, grouped by cohesion rather than by convenience.
Natural groupings include:

- A set of **query** methods (`len`, `is_empty`, `as_slice`, `capacity`, …).
- A set of **data-manipulation** methods (`push`, `pop`, `insert`, `remove`,
  `swap_remove`, …).
- A set of **fallible growth/allocation** methods (`try_reserve`,
  `try_reserve_exact`, `try_resize`, `try_extend`, …).
- A set of **conversion/builder** methods (`try_from_*`, `into_*`, iterators).

Each chunk is a self-contained, verifiable unit: implement the group, add its
tests (covering both the happy path *and* the invariants-under-failure cases that
group introduces), get it green, then move to the next chunk. Because Phase 1
already pinned the invariants, each chunk's job is to prove it neither breaks nor
violates them — and to close any `Drop` coverage gap the chunk's methods open.

**Stop at the chunk boundary.** This is the discipline that makes the whole
scheme work, and it is easy to violate under momentum. Once a chunk is implemented,
colocated, tested, and green, **stop there** — do not roll on into the next chunk,
the next type, or a broader cleanup in the same pass. The only exception is when
the user has *explicitly* asked for a long or speculative write; absent that, one
chunk per working session is the ceiling. Continuing past a verified boundary is
exactly where context rot creeps back in: the diff grows beyond what can be
reviewed line by line, and the very invariants the small steps were protecting
become the ones you break. When you stop, leave a one-line note of what the next
chunk should be so the hand-off is clean.

**Colocate the tests with the code they cover.** Each method block ships its own
unit tests in the same file, sitting right beside the functions under test — not
batched off into a separate pile. This is precisely what file partitioning is for:
by giving a cohesive block its own file, the implementation and its tests live
together and stay reviewable as one unit. If a module grows large enough that its
tests are split out, keep that split file adjacent to the implementation it
verifies (`vec/mod.rs` ↔ `vec/tests.rs`) so the pairing is obvious.

Prefer grouping by failure mode too: methods that share an error type or a
shared rollback mechanism belong together, because their tests overlap and their
invariant reasoning is identical. Splitting them across chunks scatters that
reasoning and makes it easier to miss a case.

### Partitioning the implementation into files

File-splitting is not limited to methods — **the entire implementation of a large
type may be spread across multiple files**, including the parts Phase 1 lands. For
a big type, the structure, its documented field invariants, the `Drop` impl, and
the private helpers those rely on can legitimately live in their own file rather
than crowding the module root alongside dozens of public methods. Decide the file
boundaries up front (during Phase 1) so later chunks have somewhere coherent to
land, instead of retrofitting splits after the module has grown unwieldy.

Follow these practices for new items:

- The primary type and its core methods live in the module root (`vec/mod.rs`,
  `string/mod.rs`).
- Each auxiliary type lives in its own file (`vec/into_iter.rs`,
  `string/into_chars.rs`, `vec/drain.rs`), declared `mod <name>;` and re-exported
  `pub use <name>::<Type>;`; only the thin constructor method stays in the parent
  module.
- A large type's items — method blocks, `Drop`, and supporting private
  helpers — may sit in its own file within the module, with the public surface
  re-exported, when keeping it inline would bury the type under unrelated methods.

The point of splitting is to keep **each block's unit tests colocated with the
code they test**: give a cohesive block its own file so its implementation and its
tests live side by side and stay reviewable as one unit. This is not arbitrary
file proliferation — split along a boundary where the resulting file's tests can
be reasoned about and verified on their own. A file whose tests require
understanding three other files has been split in the wrong place.

Sequencing across chunks and files: land and verify one unit (a Phase 1 slice or a
method chunk, and its file placement) before starting the next, so that at every
commit the type is complete enough to compile, its invariants hold, and its tests
are green.
