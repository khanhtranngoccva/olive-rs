# Incremental change discipline

How big a change may be and in what order work lands. Complements
[`README.md`](./README.md).

## Keep changes small

The codebase is human-first: due to LLM inaccuracies, it is required to contribute patches
whose every single line must be verifiable and verified by a human.

- One logical concern per change; prefer adding over rewriting. 
- Every intermediate state must compile with tests green and invariants intact. 
Large changes fail from context rot and produce diffs no one can verify line by line. 
- A large request touching more than ~10 items in a row must be partitioned into
independent subtasks driven through subagents, each scoped to one type, module,
or error path, with the invariants it must preserve stated explicitly. Integrate
their output incrementally.
- When a task spans several files, sequence it so each file's change is
independently compilable and testable before moving to the next.

## Landing a new type: two phases

**Phase 1 — structure and `Drop` first.** Land the struct with documented field
invariants, the `Drop` impl, and the minimal constructors and methods needed to 
exercise `Drop` (typically `new`/`default`). 
- Add tests proving no leak, no double-free, and correct deallocation (see [`TESTING.md`](./TESTING.md)). 
- Note explicitly, right next to those tests, which drop-path variants are unreachable because later methods have not landed yet, so their authors close the gap. 
- For a large type, also decide now where the definition will live (module root or 
own file), since the structure and `Drop` may be large enough to warrant their own file.

**Phase 2 — methods in chunks of 5–10 related items.** Group by category
(queries, data manipulation, fallible growth, conversions).

**Stop at the chunk boundary.** Once a chunk is implemented, tested, and green,
stop — do not roll on into the next chunk, the next type, or a broader cleanup in
the same pass. The only exception is an explicit request for a long or
speculative write. When you stop, leave a one-line note of what the next chunk
should be.

## File layout

- The primary type and its core methods generally live in the module root (`vec/mod.rs`), but may live in a submodule (`btree/map.rs`).
- Each group of one or more related auxiliary types lives in its own file 
  (`vec/into_iter.rs`, `string/into_chars.rs`, `vec/drain.rs`, `btree/iter.rs`),
  declared `mod <name>;` and re-exported `pub use <name>::<Type>;`.
- A large type's items (method blocks, `Drop`, private helpers) may sit in their
  own file, re-exported, when keeping them inline would bloat the code.
- Tests live beside the code they cover; a split test file stays adjacent to its
  implementation (`vec/mod.rs` ↔ `vec/tests.rs`).

Land and verify one unit before starting the next, so that at every commit the
type compiles, its invariants hold, and its tests are green.
