# Documentation discipline

Companion to [`README.md`](./README.md). See also [`TESTING.md`](./TESTING.md) —
documentation and tests are two views of the same contract, and both must agree.

Documentation in this repo is load-bearing. Because the crate mirrors the
standard library's API surface, the doc comments are effectively the spec —
implementors read them to know what a method must guarantee, and reviewers check
them against the code. They must therefore be kept **concise** and **carefully
maintained for inconsistencies**: a doc that contradicts the code is worse than no
doc, because it actively misleads the next implementor.

## Rules

- **Concise by default.** Say what the contract is and why, not how the function
  is mechanically implemented. Do not narrate the obvious. If removing a comment
  would leave a reader confused about *behavior or safety*, keep it; if it only
  restates what the signature and body already show plainly, cut it. Self-
  documenting code needs no commentary.
- **No implementation notes where they don't earn their place.** Avoid exposing
  internal mechanics (exact bookkeeping steps, transient state transitions,
  private field choreography) in public docs unless a caller actually depends on
  knowing it. Implementation detail belongs in a short inline comment at the site,
  not in the public doc block.
- **Watch for drift.** When you change a method's behavior, error variants,
  preconditions, or performance characteristics, update its doc in the same
  change. Stale `# Errors` lists, stale `# Safety` preconditions, and stale
  complexity claims are the most common inconsistency class. Grep for other places
  that reference the item (cross-links, sibling methods' docs) and keep them
  agreeing.
- **Match the house style.** Public items carry rustdoc with `# Safety` /
  `# Errors` / `# Panics` sections where applicable; intra-doc links use the
  `[`Item`]` form; module roots (`//!`) explain the module's role and how it
  differs from std. Follow what the neighboring items already do.

## Marking LLM-generated documentation

Any documentation (and any comment) that an LLM adds, modifies, or deletes **must be wrapped
in `<LLM-generated>` markers** so it survives careless commits — i.e. the case
where the user verifies the code but forgets to review the accompanying docs before
committing. The markers let a human later spot, filter, and audit/improve machine-written
prose separately from hand-written prose.

Format:

```rust
/// # Errors
///
///<LLM-generated
/// Returns [`TryReserveError`] if reserving capacity fails, or
/// [`TryCloneError`] if cloning an element fails.
//</LLM-generated>
```

Wrap the smallest sensible region — the paragraph or section the model touched,
not the whole item. Hand-authored text outside the markers stays unmarked. If a
region is edited again later by a human, the human may drop the markers once they
have reviewed and adopted the content.

## SAFETY sections and `SAFETY:` comments — extra care

Be **particularly careful** around `# Safety` doc sections and inline `SAFETY:`
comments. These are the highest-stakes text in the repository: they define the
contract between `unsafe fn`s and their callers, and a vague or wrong safety doc
turns a contained bug into downstream undefined behavior. Specifically:

- Every `unsafe fn` / `unsafe impl` carries a `# Safety` section (enforced by
  `#![deny(clippy::missing_safety_doc)]`). When you add or alter an unsafe item,
  write or update its `# Safety` section to enumerate the *caller's* obligations
  precisely — validity, alignment, initialization, layout-fit, liveness — using
  the same vocabulary the crate already uses ("currently allocated", "fit",
  "properly initialized").
- Every `unsafe { }` block carries a `SAFETY:` comment stating *why* the
  preconditions hold *at that site*. Keep it tied to the concrete guarantee being
  relied on ("capacity was reserved above", "we just checked strong == 1"), not a
  generic "safe".
- Treat safety text as part of the tested surface: if you change the code such
  that a previously-valid justification no longer holds, the `SAFETY:` comment and
  the `# Safety` section are now lying, and that is a bug as serious as the code
  change. Verify the reasoning still stands, ideally under Miri.
- Wrap any safety-related doc/comment you generate in `<LLM-generated>` markers
  too, and flag it explicitly in your summary so the user knows to scrutinize it.
  Never silently weaken a `# Safety` precondition to make a call compile.
