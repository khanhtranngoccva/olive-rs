# Bugbot instructions

This is a non-exhaustive document/playbook to instruct LLM-based bugbots to search for bugs. Most of these bugs/bug classes have already occurred in the codebase at some point, either fixed or remaining latent.

## Overflow bugs
- Pointer overflow when copying zero items - this usually happens when shifting a contiguous slice of items to the left in a collection `C` with base pointer `A`, but the slice is empty and at the end of `C`. Target pointer `A'` would be OOB, and adding an OOB index to the base pointer `A` to reach such target pointer causes overflow. Generally difficult to appear on 32-bit or 64-bit systems, but can *easily* happen on 16-bit embedded targets, at which point a panic is triggered. Safeguard it by preventing that branch from running if there is nothing to copy.
    - These bugs likely live around `ptr::copy_*` calls.
- Computing range bounds often leads to overflow, especially if the bound is around usize::MAX.

## Null/dangling pointer dereference bugs
- Similar to pointer overflow bugs, creating an array or contiguous block with 0 capacity will yield a dangling pointer, and pointer copying routines must not deal with them (this may flag Miri).
    - These bugs likely live around `ptr::copy_*` calls.