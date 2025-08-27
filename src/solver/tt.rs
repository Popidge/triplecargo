use std::collections::HashMap;

use crate::state::Move;

/// Bound type used for alpha-beta aware TT entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower, // value is a lower bound (alpha)
    Upper, // value is an upper bound (beta)
}

/// Transposition table entry storing value bounds, depth, and the best move for ordering/PV.
#[derive(Debug, Clone, Copy)]
pub struct TTEntry {
    pub value: i8,            // value from side-to-move perspective
    pub depth: u8,            // remaining search depth when stored
    pub flag: Bound,          // Exact / Lower / Upper
    pub best_move: Option<Move>,
}

pub trait TranspositionTable {
    fn get(&self, key: u128) -> Option<TTEntry>;
    fn put(&mut self, key: u128, entry: TTEntry);
    fn clear(&mut self);
    fn len(&self) -> usize;
}

/// Simple in-memory hash map implementation with depth-preferred replacement.
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
        // Depth-preferred replacement: replace if new.depth >= old.depth
        let replace = match self.map.get(&key) {
            Some(old) => entry.depth >= old.depth,
            None => true,
        };
        if replace {
            self.map.insert(key, entry);
        }
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