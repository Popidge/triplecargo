use super::tt::{TranspositionTable, TTEntry};

/// Fixed-size direct-mapped transposition table.
/// - Capacity must be a power of two
/// - Index = (key as usize) & mask
/// - Replacement: depth-preferred (replace if new.depth >= old.depth)
/// - Stores full 128-bit key for verification
pub struct FixedTT {
    mask: usize,
    keys: Vec<u128>,
    entries: Vec<TTEntry>,
    count: usize,
}

impl FixedTT {
    #[inline]
    pub fn with_capacity_pow2(cap_pow2: usize) -> Self {
        assert!(cap_pow2.is_power_of_two(), "TT capacity must be a power of two");
        let keys = vec![0u128; cap_pow2];
        let entries = vec![
            TTEntry {
                value: 0,
                depth: 0,
                flag: super::tt::Bound::Exact,
                best_move: None,
            };
            cap_pow2
        ];
        Self {
            mask: cap_pow2 - 1,
            keys,
            entries,
            count: 0,
        }
    }

    #[inline]
    fn index(&self, key: u128) -> usize {
        // Use low bits; Zobrist is well-mixed.
        (key as u64 as usize) & self.mask
    }
}

impl TranspositionTable for FixedTT {
    #[inline]
    fn get(&self, key: u128) -> Option<TTEntry> {
        let idx = self.index(key);
        let k = self.keys[idx];
        if k == key {
            Some(self.entries[idx])
        } else {
            None
        }
    }

    #[inline]
    fn put(&mut self, key: u128, entry: TTEntry) {
        let idx = self.index(key);
        let slot_key = self.keys[idx];
        if slot_key == 0 {
            // Empty slot
            self.keys[idx] = key;
            self.entries[idx] = entry;
            self.count += 1;
            return;
        }
        if slot_key == key {
            // Depth-preferred replacement
            if entry.depth >= self.entries[idx].depth {
                self.entries[idx] = entry;
            }
            return;
        }
        // Collision: direct-mapped replacement with depth preference
        // Replace only if the incoming depth is >= resident depth
        if entry.depth >= self.entries[idx].depth {
            self.keys[idx] = key;
            self.entries[idx] = entry;
            // count unchanged
        }
    }

    #[inline]
    fn clear(&mut self) {
        for k in &mut self.keys {
            *k = 0;
        }
        for e in &mut self.entries {
            *e = TTEntry {
                value: 0,
                depth: 0,
                flag: super::tt::Bound::Exact,
                best_move: None,
            };
        }
        self.count = 0;
    }

    #[inline]
    fn len(&self) -> usize {
        self.count
    }
}