use crate::state::GameState;
use crate::types::Owner;

/// Compute board score = (#A owned) - (#B owned).
/// Per spec, scoring is by board ownership only (after 9 turns the board is full).
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
    a - b
}