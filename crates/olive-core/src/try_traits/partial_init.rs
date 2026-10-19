//! A shared RAII guard for rolling back partially-initialized buffers.

/// RAII guard that drops the first `written` elements of a destination buffer
/// if the enclosing bulk-construction operation exits early (via a
/// returned error or panic) before completing. [`PartialInitGuard::advance`] 
/// is called after each successful write. On success the caller
/// [`core::mem::forget`]s the guard, handing ownership of the
/// fully-initialized data to the destination buffer instead of letting it be
/// dropped.
pub(crate) struct PartialInitGuard<'a, T> {
    base: *mut T,
    written: usize,
    _lifetime: &'a (),
}

impl<T> PartialInitGuard<'_, T> {
    /// Creates a guard for a destination buffer at `base` with nothing yet
    /// written.
    #[inline]
    pub(crate) fn new(base: *mut T) -> Self {
        Self {
            base,
            written: 0,
            _lifetime: &(),
        }
    }

    /// Record that one more element has been initialized at `base + written`.
    #[inline]
    pub(crate) fn advance(&mut self) {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "it is impossible to write more than u64::MAX elements"
        )]
        {
            self.written += 1;
        }
    }
}

impl<T> Drop for PartialInitGuard<'_, T> {
    fn drop(&mut self) {
        if self.written > 0 {
            // SAFETY: the first `written` slots were each initialized by a prior
            // `ptr::write` before `advance()` was called, and the caller
            // guaranteed the whole region is valid for `written` elements.
            unsafe {
                let prefix = core::ptr::slice_from_raw_parts_mut(self.base, self.written);
                core::ptr::drop_in_place(prefix);
            }
        }
    }
}
