# Bugbot instructions

This is a non-exhaustive document/playbook to instruct LLM-based bugbots to search for bugs. Most of these bugs/bug classes have already occurred in the codebase at some point, either fixed or remaining latent.

## General instructions
- When a bug or bug class is encountered and fixed, a *regression test* must be introduced to simulate the scenario.

## Overflow bugs
- Pointer overflow when copying zero items - this usually happens when shifting a contiguous slice of items to the left in a collection `C` with base pointer `A`, but the slice is empty and at the end of `C`. Target pointer `A'` would be OOB, and adding an OOB index to the base pointer `A` to reach such target pointer causes overflow. Generally difficult to appear on 32-bit or 64-bit systems, but can *easily* happen on 16-bit embedded targets, at which point a panic is triggered. Safeguard it by preventing that branch from running if there is nothing to copy.
    - These bugs likely live around `ptr::copy_*`, `ptr::read` and `ptr::add` calls.
- Computing range bounds often leads to overflow, especially if the bound is around usize::MAX.

## Null/dangling pointer bugs
- Branches that deal with ZSTs may misreport the slice length.

## Unintentional leak/no-drop bugs
- The `ManuallyDrop` object suppresses the drop glue. Caution is necessary to make sure these values are eventually hydrated for `Drop` at some point.
    - An actual bug was found and fixed in `IntoIter` where a reference is taken from `self.alloc` to deallocate, but does not call `ManuallyDrop::take()` to hydrate and drop the allocator.

## Double drop bugs
- `ptr::read` usually implies move syntax. Do not move values that are not wrapped with `ManuallyDrop`. 
    - The `Rc`'s `Drop` implementation moves the allocator under the original allocator while constructing a `Weak` reference, and causes a double drop. The fix was done by using references.

## Uninitialized fields
- When initializing an uninitialized structure and (through a raw pointer or through `MaybeUninit`), it is tempting to call `assume_init_*` or dereference the pointer into a reference and write to that reference using as_mut(). This causes a `Drop` to be invoked on an uninitialized value.

## Trait implementation bugs
- A structure or enum or item implements a trait that breaks its invariants. 
    - For example, there was a bug where `IntoChars` implemented `ExactSizeIterator` by mistake.
    - `PartialEq` and `PartialOrd` are especially prone to breaking invariants and implementations *usually* need to follow guidelines. Exceptions include types that act identically like its inner type.
