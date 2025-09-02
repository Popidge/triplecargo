use crate::cards::CardsDb;
use crate::state::{GameState, Move};
 
pub mod negamax;
pub mod tt;
pub mod move_order;
pub mod precompute;
pub mod tt_array;
pub mod graph;
 
pub use negamax::{negamax, reconstruct_pv, search_root, search_root_with_children};
pub use tt::{Bound, InMemoryTT, TranspositionTable, TTEntry};
pub use precompute::{precompute_solve, PrecomputeStats};
pub use graph::graph_precompute_export;
pub use crate::persist::ElementsMode;
 
#[derive(Debug, Clone, Copy)]
pub struct SearchLimits {
    pub max_depth: u8,
    pub time_ms: Option<u64>,
}
 
impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            max_depth: 9,     // full depth for a 3x3 board
            time_ms: None,    // no time limit by default
        }
    }
}
 
/// Result of a search from a given state.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Option<Move>,       // None at terminal positions
    pub value: i8,                     // side-to-move perspective
    pub principal_variation: Vec<Move>,
    pub nodes: u64,
    pub depth: u8,                     // depth actually searched
}
 
/// Simple solver facade that owns a TT and search limits.
pub struct Solver {
    pub tt: Box<dyn TranspositionTable>,
    pub limits: SearchLimits,
}
 
impl Solver {
    #[inline]
    pub fn new(max_depth: u8) -> Self {
        // Default: ~8M entries (1<<23) direct-mapped array (~128MB)
        let tt = crate::solver::tt_array::FixedTT::with_capacity_pow2(1 << 23);
        Self {
            tt: Box::new(tt),
            limits: SearchLimits { max_depth, time_ms: None },
        }
    }
 
    /// Construct with a HashMap-based TT for parity/testing.
    #[inline]
    pub fn with_hashmap_tt(max_depth: u8, capacity: usize) -> Self {
        let tt = InMemoryTT::with_capacity(capacity);
        Self {
            tt: Box::new(tt),
            limits: SearchLimits { max_depth, time_ms: None },
        }
    }
 
    /// Construct with any TT implementation.
    #[inline]
    pub fn with_tt(tt: Box<dyn TranspositionTable>, limits: SearchLimits) -> Self {
        Self { tt, limits }
    }
 
    /// Search the given state at full remaining depth (bounded by limits.max_depth).
    /// Returns best move (if any), value, PV, nodes, and depth searched.
    pub fn search(&mut self, state: &GameState, cards: &CardsDb) -> SearchResult {
        let remaining = 9u8 - state.board.filled_count();
        let depth = remaining.min(self.limits.max_depth);
 
        // Root negamax with alpha-beta and TT
        let (value, best_move, nodes) = search_root(state, cards, depth, &mut *self.tt);
 
        // PV reconstruction (stop at depth moves or terminal)
        let pv = {
            // end mutable borrow before immutable borrow
            let max_len = depth as usize;
            reconstruct_pv(state, cards, &*self.tt, max_len)
        };
 
        SearchResult {
            best_move,
            value,
            principal_variation: pv,
            nodes,
            depth,
        }
    }
}