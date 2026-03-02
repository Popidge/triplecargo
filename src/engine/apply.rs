use std::collections::VecDeque;

use crate::board::{Board, Slot};
use crate::cards::{Card, CardsDb};
use crate::hash::{z_token_board, z_token_hand, z_token_next};
use crate::rules::Rules;
use crate::state::{GameState, Move};
use crate::types::{Element, Owner};

#[inline]
fn clamp_side(v: i16) -> u8 {
    v.clamp(0, 11) as u8
}

#[inline]
fn raw_sides(card: &Card) -> [u8; 4] {
    [card.top, card.right, card.bottom, card.left]
}

#[inline]
fn elemental_delta(cell_elem: Option<Element>, card_elem: Option<Element>) -> i16 {
    match cell_elem {
        None => 0,
        Some(e) => {
            if card_elem == Some(e) {
                1
            } else {
                // spec: different element OR no-element card ⇒ −1
                -1
            }
        }
    }
}

/// Adjust all four sides of a card for the element on the given cell.
/// If rules.elemental is false, returns raw sides unchanged.
/// Sides order: [top, right, bottom, left]
fn adjusted_sides_for_cell(card: &Card, cell_idx: u8, board: &Board, rules: &Rules) -> [u8; 4] {
    if !rules.elemental {
        return raw_sides(card);
    }
    let delta = elemental_delta(board.cell_element(cell_idx), card.element) as i16;
    if delta == 0 {
        return raw_sides(card);
    }
    let sides = raw_sides(card);
    [
        clamp_side(sides[0] as i16 + delta),
        clamp_side(sides[1] as i16 + delta),
        clamp_side(sides[2] as i16 + delta),
        clamp_side(sides[3] as i16 + delta),
    ]
}

#[inline]
fn owner_after_mask(
    board: &Board,
    idx: u8,
    placed_owner: Owner,
    capture_mask: &[bool; 9],
) -> Option<Owner> {
    board.get(idx).map(|slot| {
        if capture_mask[idx as usize] {
            placed_owner
        } else {
            slot.owner
        }
    })
}

/// Build the full capture mask for a placed card without mutating ownership incrementally.
/// Captures are resolved with precedence: Same -> Plus -> Basic/Combo BFS.
fn build_capture_mask(board: &Board, cards: &CardsDb, rules: &Rules, placed_idx: u8) -> [bool; 9] {
    let mut capture_mask = [false; 9];

    let Some(placed_slot) = board.get(placed_idx) else {
        return capture_mask;
    };
    let placed_owner = placed_slot.owner;
    let Some(placed_card) = cards.get(placed_slot.card_id) else {
        return capture_mask;
    };
    // Same/Plus/Same Wall use raw printed ranks (Elemental does not apply).
    let placed_raw_sides = raw_sides(placed_card);
    let placed_neighbors = board.neighbors(placed_idx);

    // Gather occupied cardinal neighbors once.
    let mut occ_idx: [Option<u8>; 4] = [None; 4];
    let mut occ_owner: [Owner; 4] = [placed_owner; 4];
    let mut occ_touch: [u8; 4] = [0; 4];
    let mut opp_count = 0u8;

    for i in 0..4 {
        let Some(nidx) = placed_neighbors[i] else {
            continue;
        };
        let Some(nslot) = board.get(nidx) else {
            continue;
        };
        let Some(ncard) = cards.get(nslot.card_id) else {
            continue;
        };
        let n_sides = raw_sides(ncard);

        occ_idx[i] = Some(nidx);
        occ_owner[i] = nslot.owner;
        occ_touch[i] = n_sides[(i + 2) % 4];
        if nslot.owner != placed_owner {
            opp_count = opp_count.saturating_add(1);
        }
    }

    // If there are no opposing neighbors, no captures occur.
    if opp_count == 0 {
        return capture_mask;
    }

    let mut same_captured_dirs = [false; 4];
    let mut plus_captured_dirs = [false; 4];

    // Same precedence (per-opponent-neighbor), with optional Same Wall contributor support.
    if rules.same {
        let mut eq_on_occupied = [false; 4];
        let mut eq_on_wall = [false; 4];

        for i in 0..4 {
            if occ_idx[i].is_some() {
                eq_on_occupied[i] = placed_raw_sides[i] == occ_touch[i];
            } else if rules.same_wall && placed_neighbors[i].is_none() && placed_raw_sides[i] == 10
            {
                eq_on_wall[i] = true;
            }
        }

        for i in 0..4 {
            if occ_owner[i] == placed_owner || !eq_on_occupied[i] {
                continue;
            }
            let mut has_other_equality = false;
            for j in 0..4 {
                if i == j {
                    continue;
                }
                if eq_on_occupied[j] || eq_on_wall[j] {
                    has_other_equality = true;
                    break;
                }
            }
            if has_other_equality {
                same_captured_dirs[i] = true;
                if let Some(idx) = occ_idx[i] {
                    capture_mask[idx as usize] = true;
                }
            }
        }
    }

    let same_triggered = same_captured_dirs.iter().any(|&v| v);

    // Plus runs only if Same did not capture at least one card.
    if !same_triggered && rules.plus {
        let mut sums = [0u8; 4];
        for i in 0..4 {
            if occ_idx[i].is_some() {
                sums[i] = placed_raw_sides[i] + occ_touch[i];
            }
        }

        for i in 0..4 {
            if occ_owner[i] == placed_owner || occ_idx[i].is_none() {
                continue;
            }
            let mut has_other_equal_sum = false;
            for j in 0..4 {
                if i == j || occ_idx[j].is_none() {
                    continue;
                }
                if sums[j] == sums[i] {
                    has_other_equal_sum = true;
                    break;
                }
            }
            if has_other_equal_sum {
                plus_captured_dirs[i] = true;
                if let Some(idx) = occ_idx[i] {
                    capture_mask[idx as usize] = true;
                }
            }
        }
    }

    // Placed-card Basic applies when Same did not trigger.
    // These captures are not combo-eligible and therefore are never BFS sources.
    if !same_triggered {
        let placed_sides = adjusted_sides_for_cell(placed_card, placed_idx, board, rules);
        for i in 0..4 {
            let Some(nidx) = occ_idx[i] else {
                continue;
            };
            let neigh_owner = owner_after_mask(board, nidx, placed_owner, &capture_mask)
                .expect("neighbor card must exist for placed-card basic");
            if neigh_owner == placed_owner {
                continue;
            }
            let Some(nslot) = board.get(nidx) else {
                continue;
            };
            let Some(ncard) = cards.get(nslot.card_id) else {
                continue;
            };
            let n_sides = adjusted_sides_for_cell(ncard, nidx, board, rules);
            if placed_sides[i] > n_sides[(i + 2) % 4] {
                capture_mask[nidx as usize] = true;
            }
        }
    }

    // Combo expansion using BFS from Same/Plus captures only.
    let mut q: VecDeque<u8> = VecDeque::new();
    let mut enqueued = [false; 9];

    if same_triggered {
        for i in 0..4 {
            if same_captured_dirs[i] {
                let idx = occ_idx[i].expect("same capture direction must have occupied neighbor");
                if !enqueued[idx as usize] {
                    enqueued[idx as usize] = true;
                    q.push_back(idx);
                }
            }
        }
    } else {
        for i in 0..4 {
            if plus_captured_dirs[i] {
                let idx = occ_idx[i].expect("plus capture direction must have occupied neighbor");
                if !enqueued[idx as usize] {
                    enqueued[idx as usize] = true;
                    q.push_back(idx);
                }
            }
        }
    }

    while let Some(src_idx) = q.pop_front() {
        let Some(src_slot) = board.get(src_idx) else {
            continue;
        };
        let Some(src_card) = cards.get(src_slot.card_id) else {
            continue;
        };
        let src_owner = owner_after_mask(board, src_idx, placed_owner, &capture_mask)
            .expect("source card must exist during BFS");
        let src_sides = adjusted_sides_for_cell(src_card, src_idx, board, rules);
        let src_neighbors = board.neighbors(src_idx);

        for i in 0..4 {
            let Some(nidx) = src_neighbors[i] else {
                continue;
            };
            let Some(nslot) = board.get(nidx) else {
                continue;
            };
            let neigh_owner = owner_after_mask(board, nidx, placed_owner, &capture_mask)
                .expect("neighbor card must exist during BFS");
            if neigh_owner == src_owner {
                continue;
            }

            let Some(ncard) = cards.get(nslot.card_id) else {
                continue;
            };
            let n_sides = adjusted_sides_for_cell(ncard, nidx, board, rules);
            if src_sides[i] > n_sides[(i + 2) % 4] && !capture_mask[nidx as usize] {
                capture_mask[nidx as usize] = true;
                if !enqueued[nidx as usize] {
                    enqueued[nidx as usize] = true;
                    q.push_back(nidx);
                }
            }
        }
    }

    capture_mask
}

/// Apply a move as a pure transform: returns a new GameState on success.
/// Validates: cell empty, card present in current hand.
/// Implements Elemental, Same, Same Wall, Plus, and Combo per spec.
pub fn apply_move(state: &GameState, cards: &CardsDb, mv: Move) -> Result<GameState, String> {
    let mut ns = state.clone();
    make_move(&mut ns, cards, mv)?;
    Ok(ns)
}

/// Undo information to restore a state exactly after make_move.
#[derive(Debug, Clone)]
pub struct UndoInfo {
    pub placed_cell: u8,
    pub placed_card: u16,
    pub placed_owner: Owner,
    pub restored_hand_slot: u8,
    pub flips: Vec<(u8, Owner)>, // (cell index, old_owner)
}

/// In-place make_move with incremental Zobrist updates.
pub fn make_move(state: &mut GameState, cards: &CardsDb, mv: Move) -> Result<UndoInfo, String> {
    if mv.cell >= 9 {
        return Err("Cell index out of range".to_string());
    }
    if !state.board.is_empty(mv.cell) {
        return Err("Cell is not empty".to_string());
    }

    // Validate card present in current hand and capture slot index
    let mut slot_idx: Option<u8> = None;
    {
        let hand = state.current_hand_mut();
        for (i, slot) in hand.iter_mut().enumerate() {
            if slot.map_or(false, |id| id == mv.card_id) {
                *slot = None;
                slot_idx = Some(i as u8);
                break;
            }
        }
    }
    if slot_idx.is_none() {
        return Err("Card not in current player's hand".to_string());
    }
    // Validate card exists in DB
    if cards.get(mv.card_id).is_none() {
        return Err(format!("Card id {} not found in CardsDb", mv.card_id));
    }

    let placed_owner = state.next;

    // Incremental Z: remove from hand
    state.zobrist ^= z_token_hand(placed_owner, mv.card_id);

    // Place on board
    let placed_slot = Slot {
        owner: placed_owner,
        card_id: mv.card_id,
    };
    state.board.set(mv.cell, Some(placed_slot));
    // Incremental Z: add board token for placed
    state.zobrist ^= z_token_board(mv.cell, placed_owner, mv.card_id);

    let capture_mask = build_capture_mask(&state.board, cards, &state.rules, mv.cell);
    let mut flips: Vec<(u8, Owner)> = Vec::new();
    for idx in 0u8..9u8 {
        if !capture_mask[idx as usize] {
            continue;
        }
        if let Some(mut slot) = state.board.get(idx) {
            let old_owner = slot.owner;
            if old_owner == placed_owner {
                continue;
            }
            // remove old owner token, add new owner token
            state.zobrist ^= z_token_board(idx, old_owner, slot.card_id);
            slot.owner = placed_owner;
            state.board.set(idx, Some(slot));
            state.zobrist ^= z_token_board(idx, slot.owner, slot.card_id);
            flips.push((idx, old_owner));
        }
    }

    // Next player toggle
    let old_next = state.next;
    state.zobrist ^= z_token_next(old_next);
    state.next = state.next.other();
    state.zobrist ^= z_token_next(state.next);

    Ok(UndoInfo {
        placed_cell: mv.cell,
        placed_card: mv.card_id,
        placed_owner: old_next,
        restored_hand_slot: slot_idx.unwrap(),
        flips,
    })
}

/// In-place unmake using UndoInfo. Restores state bit-for-bit, including zobrist.
pub fn unmake_move(state: &mut GameState, undo: UndoInfo) {
    // Toggle next back
    let cur_next = state.next;
    state.zobrist ^= z_token_next(cur_next);
    state.next = cur_next.other();
    state.zobrist ^= z_token_next(state.next);

    // Unflip in reverse order
    for (idx, old_owner) in undo.flips.iter().rev() {
        if let Some(mut slot) = state.board.get(*idx) {
            let cur_owner = slot.owner;
            let card_id = slot.card_id;
            // remove current token, add old token
            state.zobrist ^= z_token_board(*idx, cur_owner, card_id);
            slot.owner = *old_owner;
            state.board.set(*idx, Some(slot));
            state.zobrist ^= z_token_board(*idx, *old_owner, card_id);
        }
    }

    // Remove placed card from board
    if let Some(slot) = state.board.get(undo.placed_cell) {
        debug_assert_eq!(slot.card_id, undo.placed_card);
        debug_assert_eq!(slot.owner, undo.placed_owner);
        state.zobrist ^= z_token_board(undo.placed_cell, slot.owner, slot.card_id);
        state.board.set(undo.placed_cell, None);
    } else {
        debug_assert!(false, "placed cell should be occupied during unmake");
    }

    // Restore card into the correct hand slot for player to move (which is undo.placed_owner)
    {
        debug_assert_eq!(state.next, undo.placed_owner);
        let hand = state.current_hand_mut();
        let idx = undo.restored_hand_slot as usize;
        debug_assert!(hand[idx].is_none());
        hand[idx] = Some(undo.placed_card);
        state.zobrist ^= z_token_hand(state.next, undo.placed_card);
    }
}
