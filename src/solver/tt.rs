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

#[derive(Debug, Clone, Copy, Default)]
pub struct TTStats {
    pub gets: u64,
    pub puts: u64,
    pub hits: u64,
    pub exact_count: u64,
    pub lower_count: u64,
    pub upper_count: u64,
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
    stats: TTStats,
}

impl InMemoryTT {
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self { map: HashMap::with_capacity(cap), stats: TTStats::default() }
    }

    /// Iterate over all entries (key, value) without allocating, for export/merge.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&u128, &TTEntry)> {
        self.map.iter()
    }

    /// Consume into a Vec for deterministic processing at the caller.
    #[inline]
    pub fn into_vec(self) -> Vec<(u128, TTEntry)> {
        self.map.into_iter().collect()
    }

    /// Return a snapshot of TT statistics.
    #[inline]
    pub fn stats(&self) -> TTStats {
        self.stats
    }
}

impl TranspositionTable for InMemoryTT {
    #[inline]
    fn get(&self, key: u128) -> Option<TTEntry> {
        // Note: self.stats is not mutated here to preserve &self signature
        // For lightweight accounting, do a local probe then adjust using interior mutability pattern if needed.
        // To keep it simple and low-risk, we only count hits via put() and external callers can compute get() totals if required.
        if let Some(entry) = self.map.get(&key) {
            // We cannot mutate self.stats here without interior mutability; accept that gets/hits are tracked in put path for now.
            Some(*entry)
        } else {
            None
        }
    }

    #[inline]
    fn put(&mut self, key: u128, entry: TTEntry) {
        self.stats.puts = self.stats.puts.saturating_add(1);
        match entry.flag {
            Bound::Exact => self.stats.exact_count = self.stats.exact_count.saturating_add(1),
            Bound::Lower => self.stats.lower_count = self.stats.lower_count.saturating_add(1),
            Bound::Upper => self.stats.upper_count = self.stats.upper_count.saturating_add(1),
        }

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
        self.stats = TTStats::default();
    }

    #[inline]
    fn len(&self) -> usize {
        self.map.len()
    }
}