use crate::rules::Rules;
use crate::state::GameState;

/// Simple SplitMix64-based mixer to build a deterministic 128-bit key without precomputed tables.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn mix_into(acc_a: &mut u64, acc_b: &mut u64, data: u64, salt: u64) {
    let m1 = splitmix64(data ^ salt);
    let m2 = splitmix64(m1 ^ 0xA5A5_A5A5_A5A5_A5A5);
    *acc_a ^= m1.rotate_left(17);
    *acc_b = acc_b.rotate_left(13) ^ m2;
}

/// Returns a u128 Zobrist-like key for memoisation.
/// Components:
/// - Board slots with (idx, owner, card_id)
/// - Hands as unordered multisets per player (each present card_id contributes once)
/// - Next player
/// - Rule toggles bitfield
#[inline]
pub fn zobrist_key(state: &GameState) -> u128 {
    let mut a: u64 = 0xC0FF_EE00_D15E_CAFE;
    let mut b: u64 = 0xDEAD_BEEF_F00D_FACE;

    // Board
    for idx in 0u8..9 {
        if let Some(slot) = state.board.get(idx) {
            let owner_bit: u64 = match slot.owner {
                crate::types::Owner::A => 0,
                crate::types::Owner::B => 1,
            };
            let data = (idx as u64)
                | (owner_bit << 8)
                | ((slot.card_id as u64) << 16)
                | (0x01u64 << 63); // domain tag: board
            mix_into(&mut a, &mut b, data, 0xB0A2_1D5E_0000_0001);
        }
    }

    // Hands (unordered): each present card contributes
    for slot in state.hands_a {
        if let Some(id) = slot {
            let data = (id as u64) | (0x02u64 << 60); // domain: hand A
            mix_into(&mut a, &mut b, data, 0xB0A2_1D5E_0000_00A0);
        }
    }
    for slot in state.hands_b {
        if let Some(id) = slot {
            let data = (id as u64) | (0x03u64 << 60); // domain: hand B
            mix_into(&mut a, &mut b, data, 0xB0A2_1D5E_0000_00B0);
        }
    }

    // Next player
    let next_bit: u64 = match state.next {
        crate::types::Owner::A => 0,
        crate::types::Owner::B => 1,
    };
    let data = next_bit | (0x04u64 << 60);
    mix_into(&mut a, &mut b, data, 0xB0A2_1D5E_0000_00C0);

    // Rules toggles
    let Rules {
        elemental,
        same,
        plus,
        same_wall,
    } = state.rules;
    let toggles: u64 = (elemental as u64)
        | ((same as u64) << 1)
        | ((plus as u64) << 2)
        | ((same_wall as u64) << 3);
    let data = toggles | (0x05u64 << 60);
    mix_into(&mut a, &mut b, data, 0xB0A2_1D5E_0000_00D0);

    ((a as u128) << 64) | (b as u128)
}