//! Query methods for `BTreeMap`: lookups, first/last accessors, and allocator introspection.

use core::borrow::Borrow;

use super::map::BTreeMap;
use super::search::SearchResult;
use crate::alloc::Allocator;

impl<K: Ord, V, A: Allocator> BTreeMap<K, V, A> {
    /// Returns true if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Gets the immutable reference to the value corresponding to the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get_key_value(key).map(|(_k, v)| v)
    }

    /// Gets the mutable reference to the value corresponding to the key.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get_key_value_mut(key).map(|(_k, v)| v)
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Gets the given key's corresponding key-value pair in the map.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let node = self.root.as_ref()?.reborrow();
        match node.search_tree(key) {
            SearchResult::Found(kv) => Some(kv.into_kv()),
            SearchResult::GoDown(_) => None,
        }
    }

    /// Gets the given key's corresponding key-value pair in the map, with mutable access to the value.
    pub fn get_key_value_mut<Q>(&mut self, key: &Q) -> Option<(&K, &mut V)>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let node = self.root.as_mut()?.borrow_valmut();
        match node.search_tree(key) {
            SearchResult::Found(kv) => Some(kv.into_kv_valmut()),
            SearchResult::GoDown(_) => None,
        }
    }

    /// Gets the first key-value pair in the map.
    ///
    /// The keys are ordered by their ord comparison.
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        let root = self.root.as_ref()?.reborrow();
        let leaf_edge = root.first_leaf_edge();
        Some(leaf_edge.next_kv().ok()?.into_kv())
    }

    /// Gets the last key-value pair in the map.
    ///
    /// The keys are ordered by their ord comparison.
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        let root = self.root.as_ref()?.reborrow();
        let leaf_edge = root.last_leaf_edge();
        Some(leaf_edge.next_back_kv().ok()?.into_kv())
    }

    /// Returns a reference to the allocator this map is using.
    pub fn allocator(&self) -> &A {
        &self.alloc
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
    fn contains_key_found_and_missing() {
        let mut map = BTreeMap::new_in(Global);
        assert!(!map.contains_key(&1));
        map.try_insert(1, 10).unwrap();
        assert!(map.contains_key(&1));
        assert!(!map.contains_key(&2));
    }

    #[test]
    fn get_key_value_found_and_missing() {
        let mut map = BTreeMap::new_in(Global);
        assert_eq!(map.get_key_value(&1), None);
        map.try_insert(3, 30).unwrap();
        assert_eq!(map.get_key_value(&3), Some((&3, &30)));
        assert_eq!(map.get_key_value(&99), None);
    }

    /// Probes a `String`-keyed map with a bare `&str`, exercising the real
    /// `Q != K` path of the `Borrow<Q>` bound (not just the reflexive case).
    #[test]
    fn get_key_value_borrowed_str_key() {
        let mut map: BTreeMap<String, u32> = BTreeMap::new_in(Global);
        map.try_insert(String::from("apple"), 1).unwrap();
        map.try_insert(String::from("banana"), 2).unwrap();
        map.try_insert(String::from("cherry"), 3).unwrap();
        // Present key via &str probe returns the owned String and its value.
        let (k, v) = map.get_key_value("banana").expect("should find banana");
        assert_eq!(k.as_str(), "banana");
        assert_eq!(*v, 2);
        // Absent &str probe misses.
        assert_eq!(map.get_key_value("durian"), None);
    }

    #[test]
    fn get_key_value_mut_found_and_missing() {
        let mut map = BTreeMap::new_in(Global);
        assert_eq!(map.get_key_value_mut(&1), None);
        map.try_insert(3, 30).unwrap();
        // Missing key returns None.
        assert_eq!(map.get_key_value_mut(&99), None);
        // Found key yields (&K, &mut V); the value is actually mutable.
        let (k, v) = map.get_key_value_mut(&3).expect("key should be present");
        assert_eq!(*k, 3);
        *v = 300;
        assert_eq!(map.get(&3), Some(&300));
    }

    #[test]
    fn get_key_value_mut_does_not_disturb_neighbours() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        // Mutate the middle entry only.
        let (k, v) = map.get_key_value_mut(&2).expect("middle key present");
        assert_eq!(*k, 2);
        *v = 999;
        // Neighbours untouched.
        assert_eq!(map.get(&0), Some(&0));
        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&2), Some(&999));
        assert_eq!(map.get(&3), Some(&30));
        assert_eq!(map.get(&4), Some(&40));
        assert_eq!(map.len(), 5);
    }

    #[test]
    fn get_key_value_mut_first_and_last() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10u32 {
            map.try_insert(i, i).unwrap();
        }
        // Mutate the minimum.
        let (k, v) = map.get_key_value_mut(&0).expect("min present");
        assert_eq!(*k, 0);
        *v = 100;
        assert_eq!(map.first_key_value(), Some((&0, &100)));
        // Mutate the maximum.
        let (k, v) = map.get_key_value_mut(&9).expect("max present");
        assert_eq!(*k, 9);
        *v = 200;
        assert_eq!(map.last_key_value(), Some((&9, &200)));
    }

    #[test]
    fn get_key_value_mut_repeated_calls_stable() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(7, 70).unwrap();
        // Multiple sequential mutable borrows must all succeed and see prior writes.
        {
            let (_, v) = map.get_key_value_mut(&7).unwrap();
            *v += 1;
        }
        {
            let (_, v) = map.get_key_value_mut(&7).unwrap();
            *v += 1;
        }
        assert_eq!(map.get(&7), Some(&72));
    }

    #[test]
    fn first_last_key_value_single() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(5, 50).unwrap();
        assert_eq!(map.first_key_value(), Some((&5, &50)));
        assert_eq!(map.last_key_value(), Some((&5, &50)));
    }

    #[test]
    fn first_last_key_value_multiple() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10 {
            map.try_insert(i, i * 10).unwrap();
        }
        assert_eq!(map.first_key_value(), Some((&0, &0)));
        assert_eq!(map.last_key_value(), Some((&9, &90)));
    }

    #[test]
    fn first_last_key_value_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        assert_eq!(map.first_key_value(), Some((&0, &0)));
        assert_eq!(map.last_key_value(), Some((&79, &158)));
    }

    #[test]
    fn get_key_value_mut_empty() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        assert_eq!(map.get_key_value_mut(&1), None);
    }

    /// Probes a `String`-keyed map with a bare `&str` and mutates through it,
    /// exercising the real `Q != K` path of the `Borrow<Q>` bound.
    #[test]
    fn get_key_value_mut_borrowed_str_key() {
        let mut map: BTreeMap<String, u32> = BTreeMap::new_in(Global);
        map.try_insert(String::from("alpha"), 1).unwrap();
        map.try_insert(String::from("beta"), 2).unwrap();
        map.try_insert(String::from("gamma"), 3).unwrap();
        // Absent &str probe misses.
        assert_eq!(map.get_key_value_mut("delta"), None);
        // Present key via &str yields (&String, &mut u32); mutate the value.
        let (k, v) = map.get_key_value_mut("beta").expect("should find beta");
        assert_eq!(k.as_str(), "beta");
        *v = 99;
        assert_eq!(map.get("beta"), Some(&99));
        // Neighbours untouched.
        assert_eq!(map.get("alpha"), Some(&1));
        assert_eq!(map.get("gamma"), Some(&3));
    }

    #[test]
    fn allocator_returns_reference() {
        let map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        // Just verify it compiles and returns something usable.
        let _alloc: &Global = map.allocator();
    }
}
