//! Query methods for [`BTreeSet`]: lookups and size accessors.
//!
//! Thin facades over the underlying [`BTreeMap`]; each delegates straight to
//! the corresponding map method, hiding the zero-sized value marker.

use super::BTreeSet;
use crate::alloc::Allocator;
use core::borrow::Borrow;

impl<T: Ord, A: Allocator> BTreeSet<T, A> {
    /// Returns true if the set contains no values.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the number of values in the set.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Gets a reference to the stored value matching the probe, or `None` if
    /// absent.
    ///
    /// The probe type `Q` need not be identical to the set's element type `T`;
    /// it only has to borrow-compare against it (e.g. querying a
    /// `BTreeSet<String>` with a `&str`).
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        // The set stores its elements as keys of the underlying map, so a
        // successful lookup yields a key — i.e. the stored value itself.
        self.map.get_key_value(value).map(|(k, _)| k)
    }

    /// Returns `true` if the set contains the given value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(value).is_some()
    }

    /// Gets the smallest value in the set, or `None` if the set is empty.
    pub fn first(&self) -> Option<&T> {
        self.map.first_key_value().map(|(k, _)| k)
    }

    /// Gets the greatest value in the set, or `None` if the set is empty.
    pub fn last(&self) -> Option<&T> {
        self.map.last_key_value().map(|(k, _)| k)
    }

    /// Returns a reference to the allocator this set is using.
    pub fn allocator(&self) -> &A {
        self.map.allocator()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::alloc::Global;
    use std::string::String;

    #[test]
    fn empty_set_is_empty_and_len_zero() {
        let set: BTreeSet<i32> = BTreeSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn len_tracks_insertions() {
        let mut set = BTreeSet::new();
        assert!(set.is_empty());
        set.try_insert(1).unwrap();
        assert!(!set.is_empty());
        assert_eq!(set.len(), 1);
        set.try_insert(2).unwrap();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn get_found_and_missing() {
        let mut set = BTreeSet::new();
        assert_eq!(set.get(&5), None);
        set.try_insert(5).unwrap();
        set.try_insert(9).unwrap();
        assert_eq!(set.get(&5), Some(&5));
        assert_eq!(set.get(&9), Some(&9));
        assert_eq!(set.get(&42), None);
    }

    #[test]
    fn contains_found_and_missing() {
        let mut set = BTreeSet::new();
        assert!(!set.contains(&1));
        set.try_insert(1).unwrap();
        assert!(set.contains(&1));
        assert!(!set.contains(&2));
    }

    /// Probes a `String`-valued set with a bare `&str`, exercising the real
    /// `Q != T` path of the `Borrow<Q>` bound (not just the reflexive case).
    #[test]
    fn get_borrowed_str_value() {
        let mut set: BTreeSet<String> = BTreeSet::new();
        set.try_insert(String::from("apple")).unwrap();
        set.try_insert(String::from("banana")).unwrap();
        set.try_insert(String::from("cherry")).unwrap();
        // Present value via &str probe returns the owned String.
        let found = set.get("banana").expect("should find banana");
        assert_eq!(found.as_str(), "banana");
        // Absent &str probe misses.
        assert_eq!(set.get("durian"), None);
        assert!(!set.contains("durian"));
    }

    #[test]
    fn get_multilevel_tree() {
        let mut set = BTreeSet::new();
        for i in 0..80u32 {
            set.try_insert(i).unwrap();
        }
        assert_eq!(set.len(), 80);
        assert_eq!(set.get(&0), Some(&0));
        assert_eq!(set.get(&79), Some(&79));
        assert_eq!(set.get(&40), Some(&40));
        assert_eq!(set.get(&1000), None);
    }

    #[test]
    fn first_last_empty_set() {
        let set: BTreeSet<i32> = BTreeSet::new();
        assert_eq!(set.first(), None);
        assert_eq!(set.last(), None);
    }

    #[test]
    fn first_last_single_element() {
        let mut set = BTreeSet::new();
        set.try_insert(42).unwrap();
        assert_eq!(set.first(), Some(&42));
        assert_eq!(set.last(), Some(&42));
    }

    #[test]
    fn first_last_multiple_elements() {
        let mut set = BTreeSet::new();
        for v in [7, 3, 9, 1, 5] {
            set.try_insert(v).unwrap();
        }
        assert_eq!(set.first(), Some(&1));
        assert_eq!(set.last(), Some(&9));
    }

    #[test]
    fn first_last_multilevel_tree() {
        // Descending insertion forces a multi-level tree.
        let mut set = BTreeSet::new();
        for i in (0..80u32).rev() {
            set.try_insert(i).unwrap();
        }
        assert_eq!(set.len(), 80);
        assert_eq!(set.first(), Some(&0));
        assert_eq!(set.last(), Some(&79));
    }

    #[test]
    fn allocator_returns_reference() {
        let set: BTreeSet<i32> = BTreeSet::new_in(Global);
        let _alloc: &Global = set.allocator();
    }
}
