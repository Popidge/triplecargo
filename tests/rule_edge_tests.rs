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
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 45,
        }),
    );
    state.board.set(
        3,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 14,
        }),
    );

    // A plays Cactaur at center
    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 31,
            cell: 4,
        },
    )
    .expect("apply_move");

    // Both neighbors should flip due to Same
    assert_eq!(
        ns.board.get(1).unwrap().owner,
        Owner::A,
        "top neighbor should flip by Same"
    );
    assert_eq!(
        ns.board.get(3).unwrap().owner,
        Owner::A,
        "left neighbor should flip by Same"
    );
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
    state.board.set(
        2,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 94,
        }),
    );

    // A plays Squall id=110 at cell 1 (top-middle): top=10 vs wall and right=10 vs neighbor left=10
    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 110,
            cell: 1,
        },
    )
    .expect("apply_move");

    // Neighbor at cell 2 should flip due to Same (wall + right equality). Wall does not flip.
    assert_eq!(
        ns.board.get(2).unwrap().owner,
        Owner::A,
        "right neighbor should flip by Same Wall + equality"
    );
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
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 81,
        }),
    );
    state.board.set(
        5,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 61,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 83,
            cell: 4,
        },
    )
    .expect("apply_move");

    // Both neighbors should flip by Plus; Same should not trigger here.
    assert_eq!(
        ns.board.get(1).unwrap().owner,
        Owner::A,
        "top neighbor should flip by Plus"
    );
    assert_eq!(
        ns.board.get(5).unwrap().owner,
        Owner::A,
        "right neighbor should flip by Plus"
    );
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
    state.board.set(
        2,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 94,
        }),
    ); // top-right
    state.board.set(
        5,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 60,
        }),
    ); // right-middle

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 110,
            cell: 1,
        },
    )
    .expect("apply_move");

    // Both cells should now be A after combo cascade.
    assert_eq!(
        ns.board.get(2).unwrap().owner,
        Owner::A,
        "cell 2 flips by Same Wall + equality"
    );
    assert_eq!(
        ns.board.get(5).unwrap().owner,
        Owner::A,
        "cell 5 flips in combo by Basic"
    );
}

/// Cross-rule interaction: enabling Same alongside Plus does not affect a Plus-only trigger scenario.
#[test]
fn cross_rule_same_plus_interaction_stable() {
    let cards = cards_db();
    // Same + Plus enabled
    let rules = Rules::new(false, true, true, false);

    // Reuse Plus setup from above
    let mut state = GameState::with_hands(rules, [83, 2, 3, 4, 5], [81, 61, 8, 9, 10], None);
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 81,
        }),
    );
    state.board.set(
        5,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 61,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 83,
            cell: 4,
        },
    )
    .expect("apply_move");

    // Outcome should match Plus-only expectations
    assert_eq!(ns.board.get(1).unwrap().owner, Owner::A);
    assert_eq!(ns.board.get(5).unwrap().owner, Owner::A);
}

/// Same contributors include any occupied neighbor ownership, not only opponents.
#[test]
fn same_uses_ally_neighbor_as_equality_contributor() {
    let cards = cards_db();
    let rules = Rules::new(false, true, false, false);

    // A plays Cactaur id=31 at center.
    // Opponent top neighbor matches (top=6 vs bottom=6): card 45 at cell 1 (B).
    // Ally left neighbor also matches (left=3 vs right=3): card 14 at cell 3 (A).
    // Same should capture the top opponent card because ally equality contributes.
    let mut state = GameState::with_hands(rules, [31, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 45,
        }),
    );
    state.board.set(
        3,
        Some(triplecargo::board::Slot {
            owner: Owner::A,
            card_id: 14,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 31,
            cell: 4,
        },
    )
    .expect("apply_move");

    assert_eq!(ns.board.get(1).unwrap().owner, Owner::A);
}

/// Plus contributors include any occupied neighbor ownership, not only opponents.
#[test]
fn plus_uses_ally_neighbor_as_sum_contributor() {
    let cards = cards_db();
    let rules = Rules::new(false, false, true, false);

    // A plays Quezacotl id=83 at center.
    // Opponent top contributes sum 2 + 9 = 11 (card 81 at cell 1).
    // Ally right contributes sum 9 + 2 = 11 (card 61 at cell 5).
    // Plus should capture the top opponent card with ally as contributor.
    let mut state = GameState::with_hands(rules, [83, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 81,
        }),
    );
    state.board.set(
        5,
        Some(triplecargo::board::Slot {
            owner: Owner::A,
            card_id: 61,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 83,
            cell: 4,
        },
    )
    .expect("apply_move");

    assert_eq!(ns.board.get(1).unwrap().owner, Owner::A);
}

/// If Same captures at least one card, Plus and placed-card Basic are suppressed.
#[test]
fn same_suppresses_plus_and_placed_basic() {
    let cards = cards_db();
    let rules = Rules::new(false, true, true, false);

    // A plays Cactaur id=31 at center.
    // Same trigger:
    // - top opponent equality (cell 1, id 45, bottom=6)
    // - left ally equality contributor (cell 3, id 14, right=3)
    // Plus would otherwise capture right opponent (cell 5, id 94) with equal sum 12 to top.
    // Placed-card Basic would otherwise capture bottom opponent (cell 7, id 60, top=1).
    let mut state = GameState::with_hands(rules, [31, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 45,
        }),
    );
    state.board.set(
        3,
        Some(triplecargo::board::Slot {
            owner: Owner::A,
            card_id: 14,
        }),
    );
    state.board.set(
        5,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 94,
        }),
    );
    state.board.set(
        7,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 60,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 31,
            cell: 4,
        },
    )
    .expect("apply_move");

    assert_eq!(
        ns.board.get(1).unwrap().owner,
        Owner::A,
        "Same capture applies"
    );
    assert_eq!(
        ns.board.get(5).unwrap().owner,
        Owner::B,
        "Plus is suppressed when Same triggers"
    );
    assert_eq!(
        ns.board.get(7).unwrap().owner,
        Owner::B,
        "placed-card Basic is suppressed when Same triggers"
    );
}

/// When Same does not trigger, Plus captures and Basic can both apply in one move.
#[test]
fn plus_and_basic_can_both_apply() {
    let cards = cards_db();
    let rules = Rules::new(false, true, true, false);

    // A plays Quezacotl id=83 at center.
    // Plus captures top and right (equal sums 11).
    // Basic from placed card captures bottom (4 > 1).
    let mut state = GameState::with_hands(rules, [83, 2, 3, 4, 5], [81, 61, 60, 9, 10], None);
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 81,
        }),
    );
    state.board.set(
        5,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 61,
        }),
    );
    state.board.set(
        7,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 60,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 83,
            cell: 4,
        },
    )
    .expect("apply_move");

    assert_eq!(
        ns.board.get(1).unwrap().owner,
        Owner::A,
        "Plus captured top"
    );
    assert_eq!(
        ns.board.get(5).unwrap().owner,
        Owner::A,
        "Plus captured right"
    );
    assert_eq!(
        ns.board.get(7).unwrap().owner,
        Owner::A,
        "Basic from placed card captured bottom"
    );
}

/// Basic captures from the placed card do not seed combo cascades.
#[test]
fn placed_basic_capture_does_not_trigger_combo_chain() {
    let cards = cards_db();
    let rules = Rules::basic_only();

    // A plays Pandemonia id=93 at cell 5.
    // Basic capture: top (cell 2, Cerberus id=94) flips because 10 > 6.
    // If that basic capture incorrectly seeded Combo, id=94 would then flip cell 1
    // (left=10 vs right<=10). Correct behavior: cell 1 remains B.
    let mut state = GameState::with_hands(rules, [93, 2, 3, 4, 5], [94, 60, 8, 9, 10], None);
    state.board.set(
        2,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 94,
        }),
    );
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 60,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 93,
            cell: 5,
        },
    )
    .expect("apply_move");

    assert_eq!(
        ns.board.get(2).unwrap().owner,
        Owner::A,
        "basic captured top"
    );
    assert_eq!(
        ns.board.get(1).unwrap().owner,
        Owner::B,
        "basic capture must not start combo chain"
    );
}

/// Elemental modifiers do not affect Same checks.
#[test]
fn elemental_does_not_affect_same() {
    let cards = cards_db();
    let rules = Rules::new(true, true, false, false);

    // A plays Sacred id=87 at center: top=5, left=9 (raw ranks for Same).
    // Top opponent Bomb id=37 has bottom=6 raw; on Earth cell it would become 5 if Elemental were
    // (incorrectly) used for Same. Left ally Seifer id=109 contributes another raw equality (9 == 9).
    // Correct behavior: Same ignores elemental, so top remains non-equal and no capture occurs.
    let mut elements = [None; 9];
    elements[1] = Some(triplecargo::Element::Earth); // mismatch for Bomb(Fire)

    let mut state =
        GameState::with_hands(rules, [87, 2, 3, 4, 5], [37, 7, 8, 9, 10], Some(elements));
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 37,
        }),
    );
    state.board.set(
        3,
        Some(triplecargo::board::Slot {
            owner: Owner::A,
            card_id: 109,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 87,
            cell: 4,
        },
    )
    .expect("apply_move");

    assert_eq!(
        ns.board.get(1).unwrap().owner,
        Owner::B,
        "top card should not flip via Same when only elemental-adjusted values match"
    );
}

/// Elemental modifiers do not affect Plus checks.
#[test]
fn elemental_does_not_affect_plus() {
    let cards = cards_db();
    let rules = Rules::new(true, false, true, false);

    // A plays Sacred id=87 at center. Raw sums:
    // - top opponent Bomb id=37: 5 + 6 = 11
    // - left ally Fungar id=2:   9 + 1 = 10
    // If Elemental were (incorrectly) used for Plus, Bomb bottom on Earth would be 5 and sums tie at 10.
    // Correct behavior: Plus ignores elemental, so no Plus capture.
    let mut elements = [None; 9];
    elements[1] = Some(triplecargo::Element::Earth); // mismatch for Bomb(Fire)

    let mut state =
        GameState::with_hands(rules, [87, 2, 3, 4, 5], [37, 7, 8, 9, 10], Some(elements));
    state.board.set(
        1,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 37,
        }),
    );
    state.board.set(
        3,
        Some(triplecargo::board::Slot {
            owner: Owner::A,
            card_id: 2,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 87,
            cell: 4,
        },
    )
    .expect("apply_move");

    assert_eq!(
        ns.board.get(1).unwrap().owner,
        Owner::B,
        "top card should not flip via Plus when only elemental-adjusted sums match"
    );
}

/// Elemental modifiers do not affect Same Wall checks.
#[test]
fn elemental_does_not_affect_same_wall() {
    let cards = cards_db();
    let rules = Rules::new(true, true, false, true);

    // A plays Minotaur id=88 at top-middle (cell 1), on Earth element.
    // Raw top is 9, but elemental-adjusted would be 10.
    // Right opponent Cerberus id=94 has left=10.
    // Incorrect behavior (elemental used for Same Wall): wall equality + right equality => capture.
    // Correct behavior: Same/Same Wall use raw ranks (top=9, right=9), so no capture.
    let mut elements = [None; 9];
    elements[1] = Some(triplecargo::Element::Earth);

    let mut state =
        GameState::with_hands(rules, [88, 2, 3, 4, 5], [94, 7, 8, 9, 10], Some(elements));
    state.board.set(
        2,
        Some(triplecargo::board::Slot {
            owner: Owner::B,
            card_id: 94,
        }),
    );

    let ns = apply_move(
        &state,
        &cards,
        Move {
            card_id: 88,
            cell: 1,
        },
    )
    .expect("apply_move");

    assert_eq!(
        ns.board.get(2).unwrap().owner,
        Owner::B,
        "right card should not flip via Same Wall from elemental-adjusted top=10"
    );
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
    assert_eq!(
        moves.len(),
        expected,
        "expected 45 legal moves on empty board with 5 cards"
    );
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
