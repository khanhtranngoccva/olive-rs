//! Proves that olive's `Global` allocator honors a user-installed
//! `#[global_allocator]`.
//!
//! The mechanism: `Global::allocate` calls the free function `alloc`, which
//! delegates to the standard `alloc` crate's exposed `alloc::alloc::alloc`.
//! That dispatches through the program's `#[global_allocator]`. By installing a
//! custom global allocator that records every allocation in an atomic counter,
//! we can observe that olive's allocations flow through it.

// Suppresses Miri running on Windows. This is caused by issue #2678
#[cfg(not(all(windows, miri)))]
mod routing {
    use core::ptr::NonNull;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// A counting global allocator wrapping the system allocator.
    struct CountingAlloc;

    unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
            // SAFETY: forwarding a valid layout to the system allocator.
            unsafe { std::alloc::System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            // SAFETY: `ptr` was allocated by the system allocator with `layout`.
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAlloc = CountingAlloc;

    /// Recovers the base pointer from a fat slice pointer (mirrors the crate's
    /// private `base_ptr`).
    fn base_ptr(slice: NonNull<[u8]>) -> NonNull<u8> {
        // SAFETY: the base address of a `NonNull<[u8]>` produced by `allocate` is
        // non-null.
        unsafe { NonNull::new_unchecked(slice.as_ptr() as *mut u8) }
    }

    #[test]
    fn olive_global_routes_through_custom_global_allocator() {
        use olive_alloc::alloc::{Allocator, Global, Layout};

        let baseline = ALLOC_COUNT.load(Ordering::SeqCst);

        let g = Global;
        let layout = Layout::array::<u64>(16).expect("valid layout");
        let block = g.allocate(layout).expect("allocation succeeds");
        assert_eq!(block.len(), layout.size());

        // Our allocation must have incremented the custom global allocator's count.
        let after = ALLOC_COUNT.load(Ordering::SeqCst);
        assert!(
            after > baseline,
            "expected olive's allocation to route through the custom #[global_allocator]; \
             count did not increase ({baseline} -> {after})"
        );

        // Clean up through the same path.
        unsafe { g.deallocate(base_ptr(block), layout) }
    }
}
