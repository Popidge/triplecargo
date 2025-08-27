use std::path::Path;

use triplecargo::{
    is_terminal, legal_moves, load_cards_from_json, zobrist_key, GameState, Move, Rules,
};

/// Precompute driver stub:
/// - Loads cards
/// - Constructs a small set of seed states
/// - Iterates their legal moves to demonstrate enumeration and hashing
/// - No solving or persistence yet (reserved for future milestones)
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cards_path = Path::new("data/cards.json");
    let cards = load_cards_from_json(cards_path).map_err(|e| format!("Cards load error: {e}"))?;
    println!(
        "[precompute] Loaded {} cards (max id {}).",
        cards.len(),
        cards.max_id()
    );

    let rules = Rules::default();

    // Seed states: trivial example with two disjoint hands just to drive enumeration
    let seeds: Vec<( [u16; 5], [u16; 5] )> = vec![
        ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        ([11, 12, 13, 14, 15], [16, 17, 18, 19, 20]),
    ];

    let mut total_states = 0usize;
    let mut total_moves = 0usize;

    for (hand_a, hand_b) in seeds {
        let mut state = GameState::with_hands(rules, hand_a, hand_b, None);
        let key = zobrist_key(&state);
        println!(
            "[precompute] Seed state key {:032x}, next={:?}",
            key, state.next
        );

        // Enumerate plies up to depth 2 as a placeholder
        let moves_0 = legal_moves(&state);
        total_moves += moves_0.len();
        for mv0 in moves_0 {
            let ns = match triplecargo::apply_move(&state, &cards, mv0) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[precompute] apply_move error at depth 0: {e}");
                    continue;
                }
            };
            total_states += 1;
            if is_terminal(&ns) {
                continue;
            }
            let moves_1 = legal_moves(&ns);
            total_moves += moves_1.len();
            // Do not proceed deeper; this is a stub
        }
    }

    println!(
        "[precompute] Enumeration stub completed. Visited ~{} states, saw {} legal moves.",
        total_states, total_moves
    );
    println!("[precompute] Persistence and full solver hookup come in later milestones.");

    Ok(())
}