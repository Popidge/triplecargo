use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use triplecargo::solver::{graph_precompute_export, search_root, InMemoryTT};
use triplecargo::{apply_move, legal_moves, load_cards_from_json, zobrist_key, GameState, Rules};

fn cards_db() -> triplecargo::CardsDb {
    let path = Path::new("data/cards.json");
    load_cards_from_json(path).expect("failed to load cards.json")
}

#[derive(Debug, Deserialize)]
struct ExportProbe {
    state_hash: String,
    value_target: i8,
}

fn export_values_by_hash(state: &GameState, cards: &triplecargo::CardsDb) -> HashMap<String, i8> {
    let rem = 9 - state.board.filled_count();
    let mut out: Vec<u8> = Vec::new();
    graph_precompute_export(state, cards, Some(rem), &mut out).expect("graph_precompute_export");

    let mut values = HashMap::new();
    for line in out.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
        let probe: ExportProbe = serde_json::from_slice(line).expect("decode export line");
        let prev = values.insert(probe.state_hash, probe.value_target);
        assert!(prev.is_none(), "duplicate state_hash in export");
    }
    values
}

fn build_state(cards: &triplecargo::CardsDb, move_choice_sequence: &[usize]) -> GameState {
    let mut state =
        GameState::with_hands(Rules::default(), [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    for &choice in move_choice_sequence {
        let moves = legal_moves(&state);
        assert!(
            !moves.is_empty(),
            "expected legal moves while building state"
        );
        let mv = moves[choice % moves.len()];
        state = apply_move(&state, cards, mv).expect("apply_move in deterministic sequence");
    }
    state
}

#[test]
fn graph_export_root_value_target_matches_negamax_sign() {
    let cards = cards_db();

    // Deterministic, lightweight sample: near-terminal states to keep runtime fast.
    let samples: [&[usize]; 4] = [
        &[0, 0, 0, 0, 0, 0],
        &[1, 2, 3, 4, 5, 6],
        &[2, 1, 4, 3, 6, 5, 8],
        &[3, 1, 7, 2, 5, 4, 9],
    ];

    for seq in samples {
        let state = build_state(&cards, seq);
        let rem = 9 - state.board.filled_count();
        assert!(rem > 0, "sample must be non-terminal");

        let values = export_values_by_hash(&state, &cards);
        let root_hash = format!("{:032x}", zobrist_key(&state));
        let graph_root = values
            .get(&root_hash)
            .copied()
            .expect("root hash missing from graph export");

        let mut tt = InMemoryTT::default();
        let (search_value, _best_move, _nodes) = search_root(&state, &cards, rem, &mut tt);
        let search_sign = search_value.signum();

        assert_eq!(
            graph_root, search_sign,
            "oracle mismatch for root {} (remaining={})",
            root_hash, rem
        );
    }
}
