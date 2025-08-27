use std::path::Path;

use triplecargo::{
    load_cards_from_json,
    solver::{search_root, reconstruct_pv, InMemoryTT},
    GameState, Move, Rules,
};

// Bring FixedTT
use triplecargo::solver::tt_array::FixedTT;

fn cards_db() -> triplecargo::CardsDb {
    let path = Path::new("data/cards.json");
    load_cards_from_json(path).expect("failed to load cards.json")
}

#[test]
fn tt_array_and_hashmap_agree_on_values_and_pv_len() {
    let cards = cards_db();
    let rules = Rules::default();

    // Create a small midgame position: 2 plies played deterministically
    let mut s = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    // A plays first legal
    let mv_a = triplecargo::legal_moves(&s).into_iter().next().unwrap();
    let ns = triplecargo::apply_move(&s, &cards, mv_a).unwrap();
    s = ns;

    // B plays first legal
    let mv_b = triplecargo::legal_moves(&s).into_iter().next().unwrap();
    let ns = triplecargo::apply_move(&s, &cards, mv_b).unwrap();
    s = ns;

    // Remaining depth
    let remaining = 9u8 - s.board.filled_count();

    // HashMap TT
    let mut tt_hm = InMemoryTT::default();
    let (val_hm, _bm_hm, _nodes_hm) = search_root(&s, &cards, remaining, &mut tt_hm);
    let pv_hm = reconstruct_pv(&s, &cards, &tt_hm, remaining as usize);

    // Fixed array TT (use a reasonable size)
    let mut tt_fx = FixedTT::with_capacity_pow2(1 << 20);
    let (val_fx, _bm_fx, _nodes_fx) = search_root(&s, &cards, remaining, &mut tt_fx);
    let pv_fx = reconstruct_pv(&s, &cards, &tt_fx, remaining as usize);

    // Primary requirement: values agree
    assert_eq!(val_fx, val_hm, "TT value mismatch between array and hashmap");

    // PV length sanity matches bounds (not necessarily identical sequence)
    assert!(
        pv_hm.len() <= remaining as usize && pv_fx.len() <= remaining as usize,
        "PV lengths exceed remaining plies: hm={}, fx={}, remaining={}",
        pv_hm.len(),
        pv_fx.len(),
        remaining
    );
}