use triplecargo::{GameState, Rules};
use triplecargo::board::Slot;
use triplecargo::types::Owner;
use triplecargo::solver::graph::fast_hands_from_board;

#[test]
fn fast_hands_basic() {
    // initial hands (sorted ascending, zeros not present)
    let initial_a: [u16; 5] = [1, 1, 2, 3, 5];
    let initial_b: [u16; 5] = [4, 6, 7, 8, 9];

    // Build a GameState with those hands and place three cards on the board:
    // cells 0,1,2 contain card_ids 1,4,9 (owners chosen arbitrarily to ensure owner independence)
    let mut s = GameState::with_hands(Rules::default(), initial_a, initial_b, None);
    s.board.set(0, Some(Slot { owner: Owner::A, card_id: 1 }));
    s.board.set(1, Some(Slot { owner: Owner::B, card_id: 4 }));
    s.board.set(2, Some(Slot { owner: Owner::A, card_id: 9 }));

    let (ha, hb) = fast_hands_from_board(&s.board, initial_a, initial_b);

    // Expect A lost one '1' (duplicate), B lost 4 and 9 => remaining:
    // A: [1,2,3,5], B: [6,7,8]
    assert_eq!(ha, vec![1u16, 2, 3, 5]);
    assert_eq!(hb, vec![6u16, 7, 8]);
}