use crate::cards::CardsDb;
use crate::engine::apply::{apply_move, make_move, unmake_move};
use crate::engine::score::score;
use crate::hash::zobrist_key;
use crate::state::{is_terminal, legal_moves, GameState, Move};
use crate::types::Owner;

use super::SearchLimits;
use super::tt::{Bound, TranspositionTable, TTEntry};
use super::move_order::order_moves;

const ALPHA_INIT: i8 = -127;
const BETA_INIT: i8 = 127;

/// Negamax driver retained for backward compatibility.
/// Values are from side-to-move perspective.
pub fn negamax(state: &GameState, cards: &CardsDb, _limits: SearchLimits, tt: &mut dyn TranspositionTable) -> i8 {
    let depth = 9 - state.board.filled_count();
    let (val, _bm, _nodes) = search_root(state, cards, depth, tt);
    val
}

/// Root search that performs full-depth negamax with alpha-beta and TT.
/// Returns (value, best_move, nodes).
pub fn search_root(state: &GameState, cards: &CardsDb, depth: u8, tt: &mut dyn TranspositionTable) -> (i8, Option<Move>, u64) {
    let mut nodes: u64 = 0;
    let mut scratch = state.clone();
 
    // Terminal shortcut
    if is_terminal(&scratch) {
        let val = terminal_value(&scratch);
        tt.put(
            zobrist_key(&scratch),
            TTEntry {
                value: val,
                depth: 0,
                flag: Bound::Exact,
                best_move: None,
            },
        );
        return (val, None, nodes);
    }
 
    let key = zobrist_key(&scratch);
    let mut tt_best: Option<Move> = None;
    if let Some(entry) = tt.get(key) {
        // Early-out at root: if we already have an Exact at sufficient depth
        if entry.depth >= depth {
            if let Bound::Exact = entry.flag {
                return (entry.value, entry.best_move, nodes);
            }
        }
        tt_best = entry.best_move;
    }
 
    let mut moves = legal_moves(&scratch);
    order_moves(&mut moves, tt_best);
 
    let mut alpha = ALPHA_INIT;
    let beta = BETA_INIT;
 
    let mut best_val = ALPHA_INIT;
    let mut best_move: Option<Move> = None;
 
    for mv in moves {
        if let Ok(undo) = make_move(&mut scratch, cards, mv) {
            let val = -negamax_inner(&mut scratch, depth.saturating_sub(1), -beta, -alpha, tt, cards, &mut nodes);
            unmake_move(&mut scratch, undo);
            if val > best_val {
                best_val = val;
                best_move = Some(mv);
            }
            if best_val > alpha {
                alpha = best_val;
            }
            if alpha >= beta {
                break;
            }
        }
    }
 
    // Store root as Exact with chosen best move
    tt.put(
        key,
        TTEntry {
            value: best_val,
            depth,
            flag: Bound::Exact,
            best_move,
        },
    );
 
    (best_val, best_move, nodes)
}
/// Root search that also returns per-child Q values for all legal moves at the root.
/// Returns (best_value, best_move, child_values[(Move, i8)], nodes).
pub fn search_root_with_children(
    state: &GameState,
    cards: &CardsDb,
    depth: u8,
    tt: &mut dyn TranspositionTable,
) -> (i8, Option<Move>, Vec<(Move, i8)>, u64) {
    let mut nodes: u64 = 0;
    let mut scratch = state.clone();

    // Terminal shortcut
    if is_terminal(&scratch) {
        let val = terminal_value(&scratch);
        tt.put(
            zobrist_key(&scratch),
            TTEntry {
                value: val,
                depth: 0,
                flag: Bound::Exact,
                best_move: None,
            },
        );
        return (val, None, Vec::new(), nodes);
    }

    let key = zobrist_key(&scratch);
    let mut tt_best: Option<Move> = None;
    if let Some(entry) = tt.get(key) {
        // Use TT best move only for ordering; still evaluate children to obtain Q values
        tt_best = entry.best_move;
    }

    let mut moves = legal_moves(&scratch);
    order_moves(&mut moves, tt_best);

    // Evaluate every child independently to obtain Q for all legal moves
    // Use full bounds for each to avoid cross-sibling pruning eliminating Q visibility.
    let mut best_val = ALPHA_INIT;
    let mut best_move: Option<Move> = None;
    let mut child_vals: Vec<(Move, i8)> = Vec::with_capacity(moves.len());

    for mv in moves {
        if let Ok(undo) = make_move(&mut scratch, cards, mv) {
            let val = -negamax_inner(
                &mut scratch,
                depth.saturating_sub(1),
                -BETA_INIT,
                -ALPHA_INIT,
                tt,
                cards,
                &mut nodes,
            );
            unmake_move(&mut scratch, undo);

            child_vals.push((mv, val));
            if val > best_val {
                best_val = val;
                best_move = Some(mv);
            }
        }
    }

    // Store root as Exact with chosen best move
    tt.put(
        key,
        TTEntry {
            value: best_val,
            depth,
            flag: Bound::Exact,
            best_move,
        },
    );

    (best_val, best_move, child_vals, nodes)
}

fn negamax_inner(
    state: &mut GameState,
    depth: u8,
    mut alpha: i8,
    mut beta: i8,
    tt: &mut dyn TranspositionTable,
    cards: &CardsDb,
    nodes: &mut u64,
) -> i8 {
    *nodes += 1;
 
    // Terminal or depth limit
    if is_terminal(state) {
        let val = terminal_value(state);
        tt.put(
            zobrist_key(state),
            TTEntry {
                value: val,
                depth: 0,
                flag: Bound::Exact,
                best_move: None,
            },
        );
        return val;
    }
    if depth == 0 {
        return terminal_value(state);
    }
 
    let key = zobrist_key(state);
    let mut tt_best: Option<Move> = None;
 
    // TT probe
    if let Some(entry) = tt.get(key) {
        if entry.depth >= depth {
            match entry.flag {
                Bound::Exact => return entry.value,
                Bound::Lower => {
                    if entry.value > alpha {
                        alpha = entry.value;
                    }
                }
                Bound::Upper => {
                    if entry.value < beta {
                        beta = entry.value;
                    }
                }
            }
            if alpha >= beta {
                return entry.value;
            }
        }
        tt_best = entry.best_move;
    }
 
    // Generate and order moves
    let mut moves = legal_moves(state);
    order_moves(&mut moves, tt_best);
 
    let alpha_orig = alpha;
    let mut best_val = ALPHA_INIT;
    let mut best_mv: Option<Move> = None;
 
    for mv in moves {
        if let Ok(undo) = make_move(state, cards, mv) {
            let val = -negamax_inner(state, depth - 1, -beta, -alpha, tt, cards, nodes);
            unmake_move(state, undo);
            if val > best_val {
                best_val = val;
                best_mv = Some(mv);
            }
            if best_val > alpha {
                alpha = best_val;
            }
            if alpha >= beta {
                break;
            }
        }
    }
 
    // Determine bound type for storage
    let flag = if best_val <= alpha_orig {
        Bound::Upper
    } else if best_val >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
 
    tt.put(
        key,
        TTEntry {
            value: best_val,
            depth,
            flag,
            best_move: best_mv,
        },
    );
 
    best_val
}

#[inline]
fn terminal_value(state: &GameState) -> i8 {
    let mut val = score(state);
    if state.next == Owner::B {
        val = -val;
    }
    val
}

/// Principal variation reconstruction by following TT best_move entries deterministically.
pub fn reconstruct_pv(state: &GameState, cards: &CardsDb, tt: &dyn TranspositionTable, max_len: usize) -> Vec<Move> {
    let mut pv: Vec<Move> = Vec::new();
    let mut cur = state.clone();
    for _ in 0..max_len {
        let key = zobrist_key(&cur);
        let Some(entry) = tt.get(key) else { break; };
        let Some(mv) = entry.best_move else { break; };
        let Ok(ns) = apply_move(&cur, cards, mv) else { break; };
        pv.push(mv);
        cur = ns;
        if is_terminal(&cur) {
            break;
        }
    }
    pv
}

/// Optional helper for future: choose best move using the (future) solver.
/// For now, returns the first legal move if any.
pub fn choose_move_stub(state: &GameState, _cards: &CardsDb) -> Option<Move> {
    let moves = legal_moves(state);
    moves.into_iter().next()
}