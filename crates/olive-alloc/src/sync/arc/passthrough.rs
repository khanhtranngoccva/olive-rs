//! Passthrough trait implementations for [`Arc`](super::Arc) that delegate
//! directly to the inner payload via deref coercion.

use super::Arc;
use core::cmp::{Ordering, PartialOrd};
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
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

impl<T: PartialEq + ?Sized, A: Allocator> PartialEq for Arc<T, A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: Eq + ?Sized, A: Allocator> Eq for Arc<T, A> {}

impl<T: PartialOrd + ?Sized, A: Allocator> PartialOrd for Arc<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: Ord + ?Sized, A: Allocator> Ord for Arc<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: Hash + ?Sized, A: Allocator> Hash for Arc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_core::try_traits::TryClone;
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

    // -- PartialEq / Eq -----------------------------------------------------

    #[test]
    fn arc_partial_eq_same_payload() {
        let a = Arc::try_new(42u32).unwrap();
        let b = Arc::try_new(42u32).unwrap();
        assert_eq!(*a, *b);
    }

    #[test]
    fn arc_partial_eq_different_payload() {
        let a = Arc::try_new(42u32).unwrap();
        let b = Arc::try_new(43u32).unwrap();
        assert_ne!(*a, *b);
    }

    #[test]
    fn arc_partial_eq_non_reflexive() {
        let a = Arc::try_new(f64::NAN).unwrap();
        let b = Arc::try_new(f64::NAN).unwrap();
        assert_ne!(*a, *b);
    }

    #[test]
    fn arc_partial_eq_shared_pointer() {
        let a = Arc::try_new(42u32).unwrap();
        let b = Arc::try_clone(&a).unwrap();
        assert_eq!(*a, *b);
    }

    #[test]
    fn arc_eq_for_eq_types() {
        // Compile-time check: Arc<u32> implements Eq because u32: Eq.
        fn assert_eq_bound<T: Eq>() {}
        assert_eq_bound::<Arc<u32>>();
    }

    // -- PartialOrd / Ord ---------------------------------------------------

    #[test]
    fn arc_partial_ord_less() {
        let a = Arc::try_new(1u32).unwrap();
        let b = Arc::try_new(2u32).unwrap();
        assert!(a.partial_cmp(&b) == Some(Ordering::Less));
    }

    #[test]
    fn arc_partial_ord_greater() {
        let a = Arc::try_new(3u32).unwrap();
        let b = Arc::try_new(2u32).unwrap();
        assert!(a.partial_cmp(&b) == Some(Ordering::Greater));
    }

    #[test]
    fn arc_partial_ord_nan() {
        let a = Arc::try_new(f64::NAN).unwrap();
        let b = Arc::try_new(f64::NAN).unwrap();
        assert!(a.partial_cmp(&b).is_none());
    }

    #[test]
    fn arc_partial_ord_equal() {
        let a = Arc::try_new(5u32).unwrap();
        let b = Arc::try_new(5u32).unwrap();
        assert_eq!(a.cmp(&b), Ordering::Equal);
    }

    #[test]
    fn arc_ord_sorted() {
        let a = Arc::try_new(3u32).unwrap();
        let b = Arc::try_new(1u32).unwrap();
        let c = Arc::try_new(2u32).unwrap();
        // Verify Ord ordering via cmp
        assert!(a.cmp(&b) > Ordering::Equal);
        assert!(c.cmp(&b) > Ordering::Equal);
        assert!(c.cmp(&a) < Ordering::Equal);
    }

    // -- Hash ---------------------------------------------------------------

    #[test]
    fn arc_hash_consistent_with_payload() {
        let a = Arc::try_new(String::from("hello")).unwrap();
        let b = Arc::try_new(String::from("hello")).unwrap();

        let mut hasher_a = std::collections::hash_map::DefaultHasher::new();
        let mut hasher_b = std::collections::hash_map::DefaultHasher::new();
        (*a).hash(&mut hasher_a);
        (*b).hash(&mut hasher_b);
        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }

    #[test]
    fn arc_hash_different_payloads_differ() {
        let a = Arc::try_new(1u64).unwrap();
        let b = Arc::try_new(2u64).unwrap();

        let mut ha = std::collections::hash_map::DefaultHasher::new();
        let mut hb = std::collections::hash_map::DefaultHasher::new();
        (*a).hash(&mut ha);
        (*b).hash(&mut hb);
        // Not guaranteed for all hashers, but DefaultHasher is deterministic
        // and different inputs almost certainly produce different hashes.
        assert_ne!(ha.finish(), hb.finish());
    }
}
