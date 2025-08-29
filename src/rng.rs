use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Deterministic RNG factory for a given (seed, game_id, turn) triple.
///
/// Implementation detail:
/// - Derives a per-state 64-bit seed as `seed ^ game_id ^ turn`.
/// - Uses PCG 64-bit generator (rand_pcg::Pcg64) for reproducible sequences.
/// - Returned RNG is deterministic and reproducible across runs when inputs are equal.
#[inline]
pub fn rng_for_state(seed: u64, game_id: u64, turn: u8) -> impl Rng {
    let derived: u64 = seed ^ game_id ^ (turn as u64);
    Pcg64::seed_from_u64(derived)
}