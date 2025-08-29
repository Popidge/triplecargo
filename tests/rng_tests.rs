use rand::Rng;
use triplecargo::rng_for_state;

fn sample(seq_len: usize, seed: u64, game_id: u64, turn: u8) -> Vec<u64> {
    let mut rng = rng_for_state(seed, game_id, turn);
    (0..seq_len).map(|_| rng.gen::<u64>()).collect()
}

#[test]
fn rng_stability_same_triple() {
    let a = sample(16, 0xDEAD_BEEFu64, 0xCAFE_BABEu64, 7);
    let b = sample(16, 0xDEAD_BEEFu64, 0xCAFE_BABEu64, 7);
    assert_eq!(a, b, "rng_for_state must produce stable sequences for identical (seed, game_id, turn)");
}

#[test]
fn rng_diff_for_different_triples() {
    let base_seed: u64 = 0x00C0_FFEEu64;
    let s1 = sample(16, base_seed, 1001, 3);
    let s2 = sample(16, base_seed, 1001, 4);
    let s3 = sample(16, base_seed.wrapping_add(1), 1001, 3);
    let s4 = sample(16, base_seed, 1002, 3);
    assert_ne!(s1, s2, "changing turn should alter sequence");
    assert_ne!(s1, s3, "changing seed should alter sequence");
    assert_ne!(s1, s4, "changing game_id should alter sequence");
}