# Olive — Guidelines for LLM-Assisted Coding

This directory is the standing playbook for any LLM (or human) working on the
Olive codebase. Start here, then read the file that matches your task. Keep each
file short on purpose so it stays easy to load and review.

| File | Read it when… |
| --- | --- |
| [`INCREMENTAL.md`](./INCREMENTAL.md) | You are about to make changes of any size, or you are **landing/porting a type**. Covers change sizing, subagent partitioning, and the two-phase order for introducing a new type. |
| [`TESTING.md`](./TESTING.md) | You are writing or reviewing tests. Exhaustive coverage, invariant-focused assertions, the shared test harness, and the Miri / leak-sanitizer gates. |
| [`DOCUMENTATION.md`](./DOCUMENTATION.md) | You are touching doc comments or inline comments. Conciseness, drift, the `<LLM-generated>` marker convention, and the elevated care required around `# Safety` / `SAFETY:` text. |
| [`PROJECT.md`](./PROJECT.md) | You need orientation: what the project is, the crate layout, naming conventions, lint posture, and the build/verify commands. |

Two other files live here too:

- [`BUGBOT.md`](./BUGBOT.md) — the bug *classes* that have historically bitten this
  project. Read it before touching a known-buggy area; its regressions are why
  `TESTING.md` leans so hard on invariants.
- [`TODO.md`](./TODO.md) — open items and deferred decisions. Check it before
  starting work that may already be tracked, and add notes for gaps you discover
  (including the `Drop` coverage holes flagged during incremental landing).

## The non-negotiable core

Across every task, three rules dominate everything else:

1. **Work incrementally.** Small, reviewable, always-compiling steps; partition
   large experiments into subagents. See [`INCREMENTAL.md`](./INCREMENTAL.md).
2. **Test against invariants, exhaustively.** Cover every documented behavior and
   assert the type's invariants under failure, not just happy-path outputs. See
   [`TESTING.md`](./TESTING.md).
3. **Keep documentation concise, consistent, and marked** — with particular care
   around safety text. See [`DOCUMENTATION.md`](./DOCUMENTATION.md).

When in doubt about intent, defer to the existing code; when in doubt about
correctness, defer to Miri and the leak sanitizer.
