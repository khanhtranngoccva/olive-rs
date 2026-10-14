# Documentation discipline

Companion to [`README.md`](./README.md); see also [`TESTING.md`](./TESTING.md) —
docs and tests are two views of the same contract and must agree.

Doc comments here are the spec: the crate mirrors std's API, so implementors read
them as contracts. A doc that contradicts the code is worse than no doc.

## Rules

- **Mark LLM-generated text — non-negotiable.** Any segment of doc or comment 
  an LLM adds, modifies, or deletes must be wrapped in `<LLM-generated>` markers, 
  around the smallest sensible region. A human may drop the markers after reviewing 
  and adopting the content:

  ```rust
  /// # Errors
  ///
  ///<LLM-generated
  /// Returns [`TryReserveError`] if reserving capacity fails.
  //</LLM-generated>
  ```

- **Concise by default.** Say what the contract is and why, not how the function
  is mechanically implemented. Cut any comment that only restates obvious parts of 
  the code.
- **No implementation notes in public docs.** Internal mechanics belong in a short
  inline comment at the site, not in the public doc block.
- **Update docs with the change.** When behavior, error variants, preconditions,
  or performance characteristics change, update the doc in the same change, and
  check dependent code for drift.

## Safety text — extra care

`# Safety` sections and `SAFETY:` comments are the highest-stakes text here.

- Every `unsafe fn` / `unsafe impl` carries a `# Safety` section enumerating the
  *caller's* obligations (validity, alignment, initialization, layout-fit,
  liveness) in the crate's vocabulary ("currently allocated", "fit", "properly
  initialized"). Enforced by `#![deny(clippy::missing_safety_doc)]`.
- Every `unsafe { }` block carries a `SAFETY:` comment stating why the
  preconditions hold *at that site* — never a generic "safe".
- If a change invalidates safety reasoning, the text is now lying and that is a
  bug as serious as the code change; verify and flag it, ideally under Miri. 
  Wrap any safety text you generate in LLM markers. 
- Never silently weaken a `# Safety` precondition to make a call compile.
