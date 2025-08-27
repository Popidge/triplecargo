use std::path::Path;

use triplecargo::{
    apply_move, is_terminal, legal_moves, load_cards_from_json, score, GameState, Move, Owner, Rules,
};

fn cards_db() -> triplecargo::CardsDb {
    let path = Path::new("data/cards.json");
    load_cards_from_json(path).expect("failed to load cards.json")
}

#[test]
fn terminal_value_matches_solver() {
    let cards = cards_db();
    let rules = Rules::default();
    let mut state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    // Play to terminal deterministically by always picking the first legal move
    for _ in 0..9 {
        let mv = legal_moves(&state).into_iter().next().expect("legal move");
        state = apply_move(&state, &cards, mv).expect("apply_move");
    }
    assert!(is_terminal(&state));

    // Expected value from side-to-move perspective at terminal
    let sc = score(&state); // A - B
    let expected = if state.next == Owner::B { -sc } else { sc };

    // Run solver
    let mut solver = triplecargo::solver::Solver::new(9);
    let res = solver.search(&state, &cards);

    assert_eq!(res.value, expected, "terminal value mismatch");
    assert!(res.best_move.is_none(), "no best move at terminal");
    assert!(
        res.principal_variation.is_empty(),
        "no PV expected at terminal"
    );
    assert_eq!(res.depth, 0u8, "depth at terminal should be 0");
}

#[test]
fn solver_determinism_same_state_same_pv_and_value() {
    let cards = cards_db();
    let rules = Rules::default();
    let mut state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    // Create a midgame state (2 plies): A plays then B plays first legal moves
    let mv_a = legal_moves(&state).into_iter().next().expect("A move");
    state = apply_move(&state, &cards, mv_a).expect("apply A");

    let mv_b = legal_moves(&state).into_iter().next().expect("B move");
    state = apply_move(&state, &cards, mv_b).expect("apply B");

    assert!(!is_terminal(&state));

    // Two independent solver instances should return identical results
    let mut solver1 = triplecargo::solver::Solver::new(9);
    let res1 = solver1.search(&state, &cards);

    let mut solver2 = triplecargo::solver::Solver::new(9);
    let res2 = solver2.search(&state, &cards);

    assert_eq!(res1.value, res2.value, "values differ");
    assert_eq!(res1.best_move, res2.best_move, "best_move differs");
    assert_eq!(
        res1.principal_variation, res2.principal_variation,
        "principal variation differs"
    );

    // Sanity checks
    let remaining = 9u8 - state.board.filled_count();
    assert!(
        res1.principal_variation.len() <= remaining as usize,
        "PV longer than remaining plies"
    );
}