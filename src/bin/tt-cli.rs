use std::path::Path;

use triplecargo::{
    apply_move, is_terminal, legal_moves, load_cards_from_json, score, zobrist_key, GameState,
    Move, Rules,
};

fn print_board(state: &GameState) {
    println!("Board (r,c -> Owner:CardId):");
    for r in 0..3 {
        for c in 0..3 {
            let idx = (r * 3 + c) as u8;
            match state.board.get(idx) {
                Some(slot) => {
                    let owner = match slot.owner {
                        triplecargo::Owner::A => 'A',
                        triplecargo::Owner::B => 'B',
                    };
                    print!("{owner}:{:>3}  ", slot.card_id);
                }
                None => {
                    print!(" .      ");
                }
            }
        }
        println!();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load cards database from JSON
    let cards_path = Path::new("data/cards.json");
    let cards = load_cards_from_json(cards_path).map_err(|e| format!("Cards load error: {e}"))?;
    println!("Loaded {} cards (max id {}).", cards.len(), cards.max_id());

    // Default rules (per spec: all toggles off by default)
    let rules = Rules::default();

    // Prepare a simple deterministic demo: hands are first 5 ids for A, next 5 for B.
    // This is just a CLI demo; proper gameplay/validation lives in library functions.
    let hand_a = [1u16, 2, 3, 4, 5];
    let hand_b = [6u16, 7, 8, 9, 10];

    let mut state = GameState::with_hands(rules, hand_a, hand_b, None);
    println!("Initial key: {:032x}", zobrist_key(&state));

    // Show legal moves for A
    let a_moves = legal_moves(&state);
    println!("A has {} legal moves. First few:", a_moves.len());
    for mv in a_moves.iter().take(6) {
        println!("  A can play card {} at cell {}", mv.card_id, mv.cell);
    }

    // Script a few moves deterministically for demo purposes
    let script = [
        Move { card_id: 1, cell: 0 },  // A
        Move { card_id: 6, cell: 1 },  // B
        Move { card_id: 2, cell: 4 },  // A
        Move { card_id: 7, cell: 3 },  // B
        Move { card_id: 3, cell: 8 },  // A
    ];

    for (turn, mv) in script.into_iter().enumerate() {
        let who = match state.next {
            triplecargo::Owner::A => "A",
            triplecargo::Owner::B => "B",
        };
        println!("Turn {}: {} plays card {} at cell {}", turn + 1, who, mv.card_id, mv.cell);
        state = apply_move(&state, &cards, mv).map_err(|e| format!("apply_move failed: {e}"))?;
        print_board(&state);
        println!("Score (A - B) = {}", score(&state));
        println!("Key: {:032x}", zobrist_key(&state));
        if is_terminal(&state) {
            println!("Game over.");
            break;
        }
    }

    Ok(())
}