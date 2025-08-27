use std::path::Path;

use triplecargo::{
    engine::apply::{make_move, unmake_move},
    hash::{recompute_zobrist},
    load_cards_from_json,
    GameState, Move, Rules,
};

fn cards_db() -> triplecargo::CardsDb {
    let path = Path::new("data/cards.json");
    load_cards_from_json(path).expect("failed to load cards.json")
}

#[test]
fn incremental_zobrist_matches_recompute_on_make_unmake() {
    let cards = cards_db();
    let rules = Rules::default();
    let mut state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    // initial cached key equals recompute
    let full0 = recompute_zobrist(&state);
    assert_eq!(state.zobrist, full0, "initial zobrist mismatch");

    // Try first few legal moves to validate incremental updates
    let moves = triplecargo::legal_moves(&state);
    for mv in moves.into_iter().take(10) {
        let mut s2 = state.clone();

        // After make_move, cached key equals recompute
        let undo = make_move(&mut s2, &cards, mv).expect("make_move");
        let full_after = recompute_zobrist(&s2);
        assert_eq!(s2.zobrist, full_after, "incremental != recompute after make_move: mv={:?}", (mv.card_id, mv.cell));

        // Unmake restores bit-for-bit (including zobrist)
        unmake_move(&mut s2, undo);
        assert_eq!(s2, state, "state not restored exactly after unmake");
        assert_eq!(s2.zobrist, state.zobrist, "zobrist not restored after unmake");
    }
}

#[test]
fn make_unmake_longer_sequence_restores_exact_state() {
    let cards = cards_db();
    let rules = Rules::default();
    let mut s = GameState::with_hands(rules, [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], None);

    let z0 = s.zobrist;
    let s0 = s.clone();

    // Play a short deterministic sequence using first legal move each ply
    let mut undos = Vec::new();
    for _ply in 0..4 {
        let mv = triplecargo::legal_moves(&s).into_iter().next().expect("legal move");
        let u = make_move(&mut s, &cards, mv).expect("make_move");
        undos.push(u);
        // Invariant: cached z == recompute
        assert_eq!(s.zobrist, recompute_zobrist(&s), "cached != recompute during sequence");
    }

    // Unmake in reverse
    while let Some(u) = undos.pop() {
        unmake_move(&mut s, u);
        assert_eq!(s.zobrist, recompute_zobrist(&s), "after unmake cached != recompute");
    }

    // Exact restoration
    assert_eq!(s, s0, "state struct not restored");
    assert_eq!(s.zobrist, z0, "zobrist not restored");
}