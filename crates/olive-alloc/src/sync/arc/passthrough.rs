//! Passthrough trait implementations for [`Arc`](super::Arc) that delegate
//! directly to the inner payload via deref coercion.

use super::Arc;
use core::fmt::{self, Debug, Display, Formatter};
use olive_core::alloc::Allocator;

impl<T: Debug + ?Sized, A: Allocator> Debug for Arc<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T: Display + ?Sized, A: Allocator> Display for Arc<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::string::String;

    #[test]
    fn arc_debug_delegates_to_inner() {
        let arc = Arc::try_new(std::vec![1, 2, 3]).unwrap();
        let dbg = std::format!("{:?}", arc);
        assert_eq!(dbg, "[1, 2, 3]");
    }

    #[test]
    fn arc_display_delegates_to_inner() {
        let arc = Arc::try_new(String::from("world")).unwrap();
        let disp = std::format!("{}", arc);
        assert_eq!(disp, "world");
    }
}
