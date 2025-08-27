use std::path::Path;

use triplecargo::{
    apply_move, is_terminal, legal_moves, load_cards_from_json, score, Board, GameState, Move,
    Owner, Rules,
};

fn cards_db() -> triplecargo::CardsDb {
    let path = Path::new("data/cards.json");
    load_cards_from_json(path).expect("failed to load cards.json")
}

#[test]
fn legal_moves_ordering_basic() {
    let cards = cards_db();
    let rules = Rules::default(); // all features off
    let state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    let moves = legal_moves(&state);
    // 9 empty cells * 5 cards = 45 moves
    assert_eq!(moves.len(), 45);

    // First moves must be cell 0 with smallest card_id first, then increasing card ids
    for i in 0..5 {
        assert_eq!(moves[i].cell, 0);
    }
    assert!(moves[0].card_id < moves[1].card_id);
    assert!(moves[1].card_id < moves[2].card_id);
}

#[test]
fn apply_move_basic_no_flip_on_tie() {
    let cards = cards_db();
    let rules = Rules::basic_only();
    // A and B hands (contents don't matter much as long as A has card 1 available)
    let mut state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    // Pre-place an opponent card to the right of center with left side equal to placed right side
    // Choose: Place A: card 1 Geezard (right=4) at cell 4, and B: a neighbor with left=4 at cell 5.
    // From data: card 35 Vysage has left=5 so not equal; card 28 Ochu left=3; find left=4:
    // Many cards have left=4; for determinism we pick id 29 SAM08G (left=4).
    // Set B's card at cell 5 (r=1,c=2)
    state.board.set(5, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 29 }));

    // Now A plays Geezard (id=1) at center cell 4 (r=1,c=1). Right=4 vs neighbor left=4 => tie -> no flip.
    let ns = apply_move(&state, &cards, Move { card_id: 1, cell: 4 }).expect("apply_move");
    // Neighbor should remain B
    let neigh = ns.board.get(5).expect("neighbor present");
    assert_eq!(neigh.owner, Owner::B);

    // Score should show A owns exactly 1 (the placed card), B owns 1 (the neighbor)
    assert_eq!(score(&ns), 0);
}

#[test]
fn apply_move_basic_flip_strictly_greater() {
    let cards = cards_db();
    let rules = Rules::basic_only();
    let mut state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    // Opponent neighbor with left=1: id 7 Gesper has left=1
    state.board.set(5, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 7 }));

    // A plays Geezard id=1 at center (cell 4), right=4 vs left=1 -> should flip neighbor at cell 5.
    let ns = apply_move(&state, &cards, Move { card_id: 1, cell: 4 }).expect("apply_move");
    let neigh = ns.board.get(5).expect("neighbor present");
    assert_eq!(neigh.owner, Owner::A);
}

#[test]
fn elemental_adjustment_causes_flip() {
    let cards = cards_db();
    // Enable Elemental only
    let rules = Rules::new(true, false, false, false);

    // Prepare board with an opponent card above center so placed top compares to neighbor bottom.
    // Use Geezard id=1 at (0,1) as opponent: bottom=5
    let mut elements = [None; 9];
    // Center cell (4) has Earth element
    elements[4] = Some(triplecargo::Element::Earth);

    let mut state = GameState::with_hands(rules, [87, 2, 3, 4, 5], [1, 7, 8, 9, 10], Some(elements));
    // Pre-place opponent at (0,1)=cell 1
    state.board.set(1, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 1 }));

    // Sacred id=87 has element Earth and top=5. On Earth cell, sides +1 => top becomes 6.
    // Compare placed top (6) vs neighbor bottom (5) -> flip.
    let ns = apply_move(&state, &cards, Move { card_id: 87, cell: 4 }).expect("apply_move");
    let neigh = ns.board.get(1).expect("neighbor present");
    assert_eq!(neigh.owner, Owner::A);
}

#[test]
fn terminal_and_scoring_progression() {
    let cards = cards_db();
    let rules = Rules::default();
    let mut state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);
    assert!(!is_terminal(&state));

    // Play 9 legal moves deterministically (the first available each time)
    for _ in 0..9 {
        let mv = legal_moves(&state).into_iter().next().expect("some move");
        state = apply_move(&state, &cards, mv).expect("apply_move");
    }
    assert!(is_terminal(&state));
    // Score is defined; no panic
    let _ = score(&state);
}