use std::collections::VecDeque;

use crate::board::{Board, Slot};
use crate::cards::{Card, CardsDb};
use crate::rules::Rules;
use crate::state::{GameState, Move};
use crate::types::{Element, Owner};

#[inline]
fn clamp_side(v: i16) -> u8 {
    v.clamp(1, 10) as u8
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
        return [card.top, card.right, card.bottom, card.left];
    }
    let delta = elemental_delta(board.cell_element(cell_idx), card.element) as i16;
    if delta == 0 {
        return [card.top, card.right, card.bottom, card.left];
    }
    let sides = [card.top, card.right, card.bottom, card.left];
    [
        clamp_side(sides[0] as i16 + delta),
        clamp_side(sides[1] as i16 + delta),
        clamp_side(sides[2] as i16 + delta),
        clamp_side(sides[3] as i16 + delta),
    ]
}

/// Apply the Basic capture rule from an origin card at origin_idx.
/// Flips only strictly-less adjacent opponent cards. No cascades here.
/// Returns a deduplicated, sorted list of indices that flipped.
fn apply_basic_from(
    board: &mut Board,
    cards: &CardsDb,
    rules: &Rules,
    origin_idx: u8,
) -> Vec<u8> {
    let origin_slot = match board.get(origin_idx) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let origin_owner = origin_slot.owner;
    let origin_card = match cards.get(origin_slot.card_id) {
        Some(c) => c,
        None => return Vec::new(),
    };
    let o_sides = adjusted_sides_for_cell(origin_card, origin_idx, board, rules);

    let neighs = board.neighbors(origin_idx);
    let mut flipped: Vec<u8> = Vec::new();

    for (i, opt_nidx) in neighs.iter().enumerate() {
        let Some(nidx) = opt_nidx else { continue };
        let Some(mut nslot) = board.get(*nidx) else { continue };
        if nslot.owner == origin_owner {
            continue;
        }
        let Some(ncard) = cards.get(nslot.card_id) else { continue };
        let n_sides = adjusted_sides_for_cell(ncard, *nidx, board, rules);

        let placed_side = o_sides[i];
        // Opposite side index mapping: 0<->2, 1<->3
        let opp_idx = (i + 2) % 4;
        let neigh_side = n_sides[opp_idx];

        if placed_side > neigh_side {
            nslot.owner = origin_owner; // flip
            board.set(*nidx, Some(nslot));
            flipped.push(*nidx);
        }
    }

    // Deterministic order
    flipped.sort_unstable();
    flipped.dedup();
    flipped
}

/// Evaluate Same and Plus triggers for the just-placed card at placed_idx.
/// Performs the initial flips if either rule triggers. Returns the set of newly flipped neighbors.
fn apply_same_plus_if_any(
    board: &mut Board,
    cards: &CardsDb,
    rules: &Rules,
    placed_idx: u8,
) -> Vec<u8> {
    let slot = match board.get(placed_idx) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let owner = slot.owner;
    let card = match cards.get(slot.card_id) {
        Some(c) => c,
        None => return Vec::new(),
    };
    let p_sides = adjusted_sides_for_cell(card, placed_idx, board, rules);
    let neighs = board.neighbors(placed_idx);

    // Collect opponent neighbors and their touching side values.
    // For direction i in [0..4): placed side index i; neighbor side index opp (i+2)%4
    let mut opp_idxs: [Option<u8>; 4] = [None; 4];
    let mut opp_touch_vals: [u8; 4] = [0; 4];
    let mut equal_dirs_for_same: [bool; 4] = [false; 4];
    let mut wall_equalities = 0usize;

    for i in 0..4 {
        match neighs[i] {
            Some(nidx) => {
                if let Some(nslot) = board.get(nidx) {
                    if nslot.owner != owner {
                        if let Some(nc) = cards.get(nslot.card_id) {
                            let ns = adjusted_sides_for_cell(nc, nidx, board, rules);
                            opp_idxs[i] = Some(nidx);
                            opp_touch_vals[i] = ns[(i + 2) % 4];

                            // Same equality check (only opponent neighbors count)
                            if rules.same && p_sides[i] == opp_touch_vals[i] {
                                equal_dirs_for_same[i] = true;
                            }
                        }
                    }
                }
            }
            None => {
                // Possible Same Wall contribution
                if rules.same && rules.same_wall && p_sides[i] == 10 {
                    wall_equalities += 1;
                }
            }
        }
    }

    // Determine Same flips
    let same_count = equal_dirs_for_same.iter().filter(|&b| *b).count() + wall_equalities;
    let mut to_flip: Vec<u8> = Vec::new();

    if rules.same && same_count >= 2 {
        for i in 0..4 {
            if equal_dirs_for_same[i] {
                if let Some(nidx) = opp_idxs[i] {
                    to_flip.push(nidx);
                }
            }
        }
    }

    // Determine Plus flips
    if rules.plus {
        // Count sums; we only consider opponent neighbors (non-wall).
        // Sums range 2..=20
        let mut counts = [0u8; 21];
        let mut sums: [u8; 4] = [0; 4];
        let mut active: [bool; 4] = [false; 4];
        for i in 0..4 {
            if let Some(_nidx) = opp_idxs[i] {
                let s = p_sides[i] + opp_touch_vals[i];
                sums[i] = s;
                active[i] = true;
                counts[s as usize] = counts[s as usize].saturating_add(1);
            }
        }
        let mut any_pair = false;
        for cnt in counts.iter() {
            if *cnt >= 2 {
                any_pair = true;
                break;
            }
        }
        if any_pair {
            for i in 0..4 {
                if active[i] && counts[sums[i] as usize] >= 2 {
                    if let Some(nidx) = opp_idxs[i] {
                        to_flip.push(nidx);
                    }
                }
            }
        }
    }

    // Perform the initial flips if any; deterministically ordered, unique.
    to_flip.sort_unstable();
    to_flip.dedup();

    if !to_flip.is_empty() {
        for nidx in &to_flip {
            if let Some(mut nslot) = board.get(*nidx) {
                nslot.owner = owner;
                board.set(*nidx, Some(nslot));
            }
        }
    }

    to_flip
}

/// Apply a move as a pure transform: returns a new GameState on success.
/// Validates: cell empty, card present in current hand.
/// Implements Elemental, Same, Same Wall, Plus, and Combo per spec.
pub fn apply_move(
    state: &GameState,
    cards: &CardsDb,
    mv: Move,
) -> Result<GameState, String> {
    if mv.cell >= 9 {
        return Err("Cell index out of range".to_string());
    }
    if !state.board.is_empty(mv.cell) {
        return Err("Cell is not empty".to_string());
    }

    // Validate card in hand
    let mut found = false;
    for slot in state.current_hand().iter() {
        if slot.map_or(false, |id| id == mv.card_id) {
            found = true;
            break;
        }
    }
    if !found {
        return Err("Card not in current player's hand".to_string());
    }
    // Validate card exists in DB
    if cards.get(mv.card_id).is_none() {
        return Err(format!("Card id {} not found in CardsDb", mv.card_id));
    }

    // Clone and mutate
    let mut ns = state.clone();

    // Remove card from current hand and place on board
    if !ns.take_from_current_hand(mv.card_id) {
        return Err("Internal error: failed to take card from hand".to_string());
    }
    let placed_slot = Slot {
        owner: ns.next,
        card_id: mv.card_id,
    };
    ns.board.set(mv.cell, Some(placed_slot));

    // Same/Plus initial evaluation
    let initial_flips = apply_same_plus_if_any(&mut ns.board, cards, &ns.rules, mv.cell);

    if initial_flips.is_empty() {
        // Apply Basic once from placed card only
        let _ = apply_basic_from(&mut ns.board, cards, &ns.rules, mv.cell);
    } else {
        // Combo cascades: BFS over newly flipped cards, applying Basic only
        let mut q: VecDeque<u8> = VecDeque::new();
        for idx in initial_flips {
            q.push_back(idx);
        }
        while let Some(idx) = q.pop_front() {
            let flips = apply_basic_from(&mut ns.board, cards, &ns.rules, idx);
            for f in flips {
                q.push_back(f);
            }
        }
    }

    // Next player
    ns.next = ns.next.other();

    Ok(ns)
}