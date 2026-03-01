use crate::state::GameState;
use crate::types::Owner;

/// Compute ownership delta over board plus unplayed hands.
/// Score = (A board + A hand) - (B board + B hand).
#[inline]
pub fn score(state: &GameState) -> i8 {
    let mut a: i8 = 0;
    let mut b: i8 = 0;
    for cell in 0u8..9u8 {
        if let Some(slot) = state.board.get(cell) {
            match slot.owner {
                Owner::A => a += 1,
                Owner::B => b += 1,
            }
        }
    }

    for slot in &state.hands_a {
        if slot.is_some() {
            a += 1;
        }
    }
    for slot in &state.hands_b {
        if slot.is_some() {
            b += 1;
        }
    }

    a - b
}
