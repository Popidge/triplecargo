use std::path::Path;

use triplecargo::{
    apply_move, legal_moves, load_cards_from_json, score, Board, GameState, Move, Owner, Rules,
};

fn cards_db() -> triplecargo::CardsDb {
    let path = Path::new("data/cards.json");
    load_cards_from_json(path).expect("failed to load cards.json")
}

/// Same: two opponent neighbors match placed sides -> both flip.
#[test]
fn same_triggers_double_flip() {
    let cards = cards_db();
    // Enable Same only
    let rules = Rules::new(false, true, false, false);

    // A has Cactaur id=31: top=6, right=2, bottom=6, left=3
    // B pre-placed neighbors:
    //  - cell 1 with bottom=6 (Iron Giant id=45)
    //  - cell 3 with right=3 (Mesmerize id=14)
    let mut state = GameState::with_hands(rules, [31, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);
    state.board.set(1, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 45 }));
    state.board.set(3, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 14 }));

    // A plays Cactaur at center
    let ns = apply_move(&state, &cards, Move { card_id: 31, cell: 4 }).expect("apply_move");

    // Both neighbors should flip due to Same
    assert_eq!(ns.board.get(1).unwrap().owner, Owner::A, "top neighbor should flip by Same");
    assert_eq!(ns.board.get(3).unwrap().owner, Owner::A, "left neighbor should flip by Same");
    // Placed + 2 flips -> at least 3 owned by A on board
    let s = score(&ns); // A - B
    assert!(s >= 1, "expected A advantage after double flip, got {}", s);
}

/// Same Wall: wall (value 10) contributes one equality when placed side equals 10.
/// Combine with one neighbor equality to trigger Same, flipping that neighbor.
#[test]
fn same_wall_contributes_equality() {
    let cards = cards_db();
    // Enable Same + Same Wall
    let rules = Rules::new(false, true, false, true);

    // Pre-place B at cell 2 (top-right) with left=10 to match Squall's right=10.
    // Iguion id=61 has left=2 so not suitable; use a left=10 card, e.g., Cerberus id=94 has left=10.
    let mut state = GameState::with_hands(rules, [110, 2, 3, 4, 5], [94, 7, 8, 9, 10], None);

    // B: cell 2 with left=10 (Cerberus id=94)
    state.board.set(2, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 94 }));

    // A plays Squall id=110 at cell 1 (top-middle): top=10 vs wall and right=10 vs neighbor left=10
    let ns = apply_move(&state, &cards, Move { card_id: 110, cell: 1 }).expect("apply_move");

    // Neighbor at cell 2 should flip due to Same (wall + right equality). Wall does not flip.
    assert_eq!(ns.board.get(2).unwrap().owner, Owner::A, "right neighbor should flip by Same Wall + equality");
}

/// Plus: if two sums equal, both corresponding neighbors flip.
#[test]
fn plus_triggers_double_flip() {
    let cards = cards_db();
    // Enable Same and Plus; only Plus will actually trigger in this setup.
    let rules = Rules::new(false, true, true, false);

    // A will play Quezacotl id=83 at center: top=2, right=9
    // B neighbors:
    //  - cell 1 (top-middle): MinMog id=81 bottom=9 -> sum = 2 + 9 = 11
    //  - cell 5 (right-middle): Iguion id=61 left=2 -> sum = 9 + 2 = 11
    let mut state = GameState::with_hands(rules, [83, 2, 3, 4, 5], [81, 61, 8, 9, 10], None);
    state.board.set(1, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 81 }));
    state.board.set(5, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 61 }));

    let ns = apply_move(&state, &cards, Move { card_id: 83, cell: 4 }).expect("apply_move");

    // Both neighbors should flip by Plus; Same should not trigger here.
    assert_eq!(ns.board.get(1).unwrap().owner, Owner::A, "top neighbor should flip by Plus");
    assert_eq!(ns.board.get(5).unwrap().owner, Owner::A, "right neighbor should flip by Plus");
}

/// Combo cascade after Same/SameWall: newly flipped neighbors apply Basic rule to their neighbors.
#[test]
fn combo_after_same_wall_triggers_basic_chain() {
    let cards = cards_db();
    // Same + Same Wall only
    let rules = Rules::new(false, true, false, true);

    // Setup:
    //  - B at cell 2: Cerberus id=94 (left=10, bottom=6)
    //  - B at cell 5: Gerogero id=60 (top=1)
    // A plays Squall id=110 at cell 1 (top-middle).
    // Same Wall contributes one equality (top=10 vs wall) + right equality (10 vs left=10) -> flip cell 2.
    // Combo: from cell 2, Basic compares bottom=6 vs neighbor at cell 5 top=1 -> flip cell 5.
    let mut state = GameState::with_hands(rules, [110, 2, 3, 4, 5], [94, 60, 8, 9, 10], None);
    state.board.set(2, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 94 })); // top-right
    state.board.set(5, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 60 })); // right-middle

    let ns = apply_move(&state, &cards, Move { card_id: 110, cell: 1 }).expect("apply_move");

    // Both cells should now be A after combo cascade.
    assert_eq!(ns.board.get(2).unwrap().owner, Owner::A, "cell 2 flips by Same Wall + equality");
    assert_eq!(ns.board.get(5).unwrap().owner, Owner::A, "cell 5 flips in combo by Basic");
}

/// Cross-rule interaction: enabling Same alongside Plus does not affect a Plus-only trigger scenario.
#[test]
fn cross_rule_same_plus_interaction_stable() {
    let cards = cards_db();
    // Same + Plus enabled
    let rules = Rules::new(false, true, true, false);

    // Reuse Plus setup from above
    let mut state = GameState::with_hands(rules, [83, 2, 3, 4, 5], [81, 61, 8, 9, 10], None);
    state.board.set(1, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 81 }));
    state.board.set(5, Some(triplecargo::board::Slot { owner: Owner::B, card_id: 61 }));

    let ns = apply_move(&state, &cards, Move { card_id: 83, cell: 4 }).expect("apply_move");

    // Outcome should match Plus-only expectations
    assert_eq!(ns.board.get(1).unwrap().owner, Owner::A);
    assert_eq!(ns.board.get(5).unwrap().owner, Owner::A);
}

/// Zero-allocation-ish assertion on legal_moves capacity: vector is preallocated to exact size.
/// This helps ensure no reallocation occurs while pushing moves.
#[test]
fn legal_moves_preallocates_exact_capacity() {
    let cards = cards_db();
    let rules = Rules::basic_only();
    let state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    let moves = legal_moves(&state);
    let expected = 9usize * 5usize;
    assert_eq!(moves.len(), expected, "expected 45 legal moves on empty board with 5 cards");
    assert_eq!(
        moves.capacity(),
        expected,
        "legal_moves should allocate exact capacity to avoid reallocation"
    );

    // Sanity: first 5 moves are cell 0 with ascending card ids (determinism)
    for i in 0..5 {
        assert_eq!(moves[i].cell, 0);
        if i > 0 {
            assert!(moves[i - 1].card_id < moves[i].card_id);
        }
    }
}