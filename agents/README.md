# Olive — Guidelines for LLM-Assisted Coding

The standing playbook for any LLM (or human) working on Olive. Start here, then
read the file that matches your task. Each file is kept short on purpose so it
loads and reviews quickly.

| File | Read it when… |
| --- | --- |
| [`INCREMENTAL.md`](./INCREMENTAL.md) | Making changes of any size, or landing/porting a type. |
| [`TESTING.md`](./TESTING.md) | Writing or reviewing tests. |
| [`DOCUMENTATION.md`](./DOCUMENTATION.md) | Touching doc comments or inline comments. |
| [`PROJECT.md`](./PROJECT.md) | Needing orientation: crates, conventions, lints, commands. |
| [`BUGBOT.md`](./BUGBOT.md) | Touching an area hit by a historical bug class. |
| [`TODO.md`](./scratch/TODO.md) | Before starting work; add notes for gaps you discover. |

## The non-negotiable core

1. **Work incrementally.** Small, reviewable, always-compiling steps; partition
   large experiments into subagents. See [`INCREMENTAL.md`](./INCREMENTAL.md).
2. **Stop at the chunk boundary.** Once a chunk is implemented, tested, and green,
   stop — do not roll into the next chunk unless the user explicitly asked for a
   long or speculative write. See [`INCREMENTAL.md`](./INCREMENTAL.md).
3. **Test exhaustively against invariants**, including failure paths. See
   [`TESTING.md`](./TESTING.md).
4. **Keep documentation concise and mark LLM-generated additions** with
   `<LLM-generated>` markers, with extra care around safety text. See
   [`DOCUMENTATION.md`](./DOCUMENTATION.md).

When in doubt about intent, defer to the existing code; when in doubt about
correctness, defer to Miri and the leak sanitizer.
