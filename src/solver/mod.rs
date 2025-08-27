use crate::cards::CardsDb;
use crate::state::{GameState, Move};

pub mod negamax;
pub mod tt;

pub use negamax::negamax;
pub use tt::{InMemoryTT, TranspositionTable};

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