use crate::rules::Rules;
use crate::state::GameState;
use crate::types::Owner;

/// SplitMix64 PRNG step for stable, fast token generation.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn token128_from_seed(seed: u64) -> u128 {
    // Two rounds to build 128 bits deterministically.
    let lo = splitmix64(seed ^ 0xC0FF_EE00_D15E_CAFE);
    let hi = splitmix64(seed ^ 0xDEAD_BEEF_F00D_FACE ^ lo.rotate_left(17));
    ((hi as u128) << 64) | (lo as u128)
}

// Domain tags (arbitrary but fixed)
const DOM_BOARD: u64 = 0xB0A2_1D5E_0000_0001;
const DOM_HAND:  u64 = 0xB0A2_1D5E_0000_00A0;
const DOM_NEXT:  u64 = 0xB0A2_1D5E_0000_00C0;
const DOM_RULES: u64 = 0xB0A2_1D5E_0000_00D0;

/// Public Zobrist tokens for incremental maintenance

#[inline]
pub fn z_token_board(cell: u8, owner: Owner, card_id: u16) -> u128 {
    let owner_bit: u64 = match owner {
        Owner::A => 0,
        Owner::B => 1,
    };
    let seed = DOM_BOARD
        ^ (cell as u64)
        ^ (owner_bit << 8)
        ^ ((card_id as u64) << 16);
    token128_from_seed(seed)
}

#[inline]
pub fn z_token_hand(owner: Owner, card_id: u16) -> u128 {
    let owner_bit: u64 = match owner {
        Owner::A => 0,
        Owner::B => 1,
    };
    let seed = DOM_HAND ^ owner_bit ^ ((card_id as u64) << 8);
    token128_from_seed(seed)
}

#[inline]
pub fn z_token_next(owner: Owner) -> u128 {
    let owner_bit: u64 = match owner {
        Owner::A => 0,
        Owner::B => 1,
    };
    let seed = DOM_NEXT ^ owner_bit;
    token128_from_seed(seed)
}

#[inline]
pub fn z_token_rules(rules: Rules) -> u128 {
    let toggles: u64 = (rules.elemental as u64)
        | ((rules.same as u64) << 1)
        | ((rules.plus as u64) << 2)
        | ((rules.same_wall as u64) << 3);
    let seed = DOM_RULES ^ toggles;
    token128_from_seed(seed)
}

/// Full recomputation from state components. Used to initialize and to validate
/// incremental updates during tests.
#[inline]
pub fn recompute_zobrist(state: &GameState) -> u128 {
    let mut z: u128 = 0;

    // Board occupants
    for idx in 0u8..9 {
        if let Some(slot) = state.board.get(idx) {
            z ^= z_token_board(idx, slot.owner, slot.card_id);
        }
    }

    // Hands as unordered multisets
    for s in state.hands_a {
        if let Some(id) = s {
            z ^= z_token_hand(Owner::A, id);
        }
    }
    for s in state.hands_b {
        if let Some(id) = s {
            z ^= z_token_hand(Owner::B, id);
        }
    }

    // Side to move
    z ^= z_token_next(state.next);

    // Rules toggles
    z ^= z_token_rules(state.rules);

    z
}

/// Accessor kept for API stability: now returns the cached, incrementally
/// maintained key stored in GameState.
#[inline]
pub fn zobrist_key(state: &GameState) -> u128 {
    state.zobrist
}