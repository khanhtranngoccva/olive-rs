//! Deterministic pseudo-random number generator for reproducible tests.
//!
//! Uses the SplitMix64 algorithm: fast, well-distributed, and trivially
//! seedable. Each test constructs its own [`TestRng`] with a fixed seed so
//! that randomized test sequences are fully deterministic across runs and
//! platforms.

extern crate std;

/// A minimal deterministic PRNG (SplitMix64) for reproducible test sequences.
///
/// # Example
/// ```ignore
/// let rng = TestRng::new(0x1234_5678);
/// for key in rng.permuted(0..200u32) {
///     map.remove(&key);
/// }
/// ```
#[derive(Debug)]
pub struct TestRng {
    state: u64,
}

impl TestRng {
    /// Creates a new RNG seeded with `seed`. The same seed always produces
    /// the same sequence.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the internal state and returns the next 64-bit value.
    ///
    /// Exposed for tests that need raw randomness. Calling this
    /// shifts the permutation, so use with care within a test.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Consumes this RNG and returns an iterator yielding all values in
    /// `range` in a deterministic pseudo-random order (a uniform random
    /// permutation). The same seed and range always produce the same
    /// ordering.
    ///
    /// Internally performs a Fisher-Yates shuffle over the collected range,
    /// so the range must be finite and reasonably sized (this is a test
    /// helper). Because `self` is consumed, the RNG cannot be reused after
    /// this call — construct a new `TestRng` if you need another permutation.
    pub fn permuted<R>(mut self, range: R) -> std::vec::IntoIter<u32>
    where
        R: IntoIterator<Item = u32>,
    {
        let mut v: std::vec::Vec<u32> = range.into_iter().collect();
        self.shuffle_in_place(&mut v);
        v.into_iter()
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Returns a uniformly distributed `u64` in `[0, limit)` with zero
    /// modulo bias.
    #[inline]
    fn below(&mut self, limit: u64) -> u64 {
        assert!(limit > 0, "TestRng::below called with limit == 0");
        // Powers of two divide 2^64 evenly, so masking is already unbiased.
        if limit.is_power_of_two() {
            return self.next_u64() & (limit - 1);
        }
        // Size of the uneven tail: m = 2^64 % limit.
        // Since 2^64 overflows u64, compute via (u64::MAX % limit) + 1.
        let rem = u64::MAX % limit + 1;
        debug_assert_ne!(rem, limit, "power-of-two limit should have hit fast path");
        loop {
            let r = self.next_u64();
            // Reject the top `rem` values.
            // r < 2 ^ 64 - rem, !r = 2 ^ 64 - 1 - r
            // => 2 ^ 64 - 1 - r < rem
            // => r > 2 ^ 64 - 1 - rem
            // => r >= 2 ^ 64 - rem
            if !r < rem {
                continue;
            }
            return r % limit;
        }
    }

    /// In-place Fisher-Yates shuffle using this RNG's state.
    fn shuffle_in_place<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        for i in (1..len).rev() {
            let j = self.below((i + 1) as u64) as usize;
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = TestRng::new(42);
        let mut b = TestRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_different_sequences() {
        let mut a = TestRng::new(1);
        let mut b = TestRng::new(2);
        // Extremely unlikely to collide on first draw.
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn permuted_yields_full_permutation() {
        let rng = TestRng::new(7);
        let collected: std::vec::Vec<u32> = rng.permuted(0..30).collect();
        assert_eq!(collected.len(), 30);
        let mut sorted = collected.clone();
        sorted.sort();
        assert_eq!(sorted, (0..30).collect::<std::vec::Vec<_>>());
    }

    #[test]
    fn permuted_accepts_any_into_iterator() {
        let rng = TestRng::new(42);
        // Works with a non-zero-based range.
        let collected: std::vec::Vec<u32> = rng.permuted(10..25).collect();
        assert_eq!(collected.len(), 15);
        let mut sorted = collected.clone();
        sorted.sort();
        assert_eq!(sorted, (10..25).collect::<std::vec::Vec<_>>());
    }

    #[test]
    fn permuted_deterministic_for_same_seed() {
        let a = TestRng::new(999);
        let b = TestRng::new(999);
        let seq_a: std::vec::Vec<u32> = a.permuted(0..20).collect();
        let seq_b: std::vec::Vec<u32> = b.permuted(0..20).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn permuted_consumes_rng() {
        // Compile-time guarantee: after calling permuted, the RNG is moved.
        // This test just exercises the happy path to confirm ownership works.
        let rng = TestRng::new(123);
        let iter = rng.permuted(0..10);
        // `rng` is no longer accessible here — the compiler enforces this.
        assert_eq!(iter.count(), 10);
    }
}
