use std::collections::HashMap;

/// Minimal entry for a transposition table. For now we only keep the exact value.
/// Later we can extend with depth, bound type, best move, etc.
#[derive(Debug, Clone, Copy)]
pub struct TTEntry {
    pub value: i8,   // exact game-theoretic value from the perspective of player to move
    pub depth: u8,   // remaining depth when stored (for replacement policy)
}

pub trait TranspositionTable {
    fn get(&self, key: u128) -> Option<TTEntry>;
    fn put(&mut self, key: u128, entry: TTEntry);
    fn clear(&mut self);
    fn len(&self) -> usize;
}

/// Simple in-memory hash map implementation (stub friendly).
#[derive(Debug, Default)]
pub struct InMemoryTT {
    map: HashMap<u128, TTEntry>,
}

impl InMemoryTT {
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self { map: HashMap::with_capacity(cap) }
    }
}

impl TranspositionTable for InMemoryTT {
    #[inline]
    fn get(&self, key: u128) -> Option<TTEntry> {
        self.map.get(&key).copied()
    }

    #[inline]
    fn put(&mut self, key: u128, entry: TTEntry) {
        // Simple store; replacement strategy can be added later (e.g., deeper depth wins)
        self.map.insert(key, entry);
    }

    #[inline]
    fn clear(&mut self) {
        self.map.clear();
    }

    #[inline]
    fn len(&self) -> usize {
        self.map.len()
    }
}