use crate::cards::CardsDb;
use crate::engine::score::score;
use crate::hash::zobrist_key;
use crate::state::{is_terminal, legal_moves, GameState, Move};

use super::SearchLimits;
use super::tt::{TranspositionTable, TTEntry};

/// Negamax stub with memoisation (transposition table).
/// This is a placeholder to be implemented in the Solver milestone.
/// Current behavior:
/// - If terminal: return final score from Player A's perspective.
/// - Otherwise: returns 0 without searching, storing a TT entry with depth 0.
/// API will remain stable to ease later replacement with full solver.
///
/// Conventions:
/// - Values are from Player A's perspective: positive == advantage for A.
/// - When it's B's turn, value is negated accordingly in a full negamax.
pub fn negamax(state: &GameState, _cards: &CardsDb, _limits: SearchLimits, tt: &mut dyn TranspositionTable) -> i8 {
    let key = zobrist_key(state);
    if let Some(entry) = tt.get(key) {
        return entry.value;
    }

    // Terminal leaf: exact score
    if is_terminal(state) {
        let val = score(state);
        tt.put(key, TTEntry { value: val, depth: 0 });
        return val;
    }

    // Non-terminal: placeholder (no actual search yet)
    let val = 0i8;
    tt.put(key, TTEntry { value: val, depth: 0 });
    val
}

/// Optional helper for future: choose best move using the (future) solver.
/// For now, returns the first legal move if any.
pub fn choose_move_stub(state: &GameState, _cards: &CardsDb) -> Option<Move> {
    let mut moves = legal_moves(state);
    moves.into_iter().next()
}