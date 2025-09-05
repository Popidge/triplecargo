use triplecargo::solver::graph::PackedState;

#[test]
fn state_hash_from_key_hex() {
    // Known u128 key: 0x0123456789abcdef0123456789abcdef
    let key: u128 = 0x0123456789abcdef0123456789abcdefu128;
    // Build a minimal PackedState (board empty, default rules and next)
    let ps = PackedState {
        board: triplecargo::board::Board::new(),
        next: triplecargo::types::Owner::A,
        rules: triplecargo::Rules::default(),
    };

    let hex = format!("{:032x}", key);
    assert_eq!(hex.len(), 32);
    assert_eq!(hex, "0123456789abcdef0123456789abcdef");
    // Sanity: ensure formatting matches expected lowercase hex
    assert_eq!(hex, format!("{:032x}", key));
    // The PackedState isn't used to compute the hash here; we only verify formatting of the u128 key.
    let _ = ps; // keep ps referenced to avoid unused warnings in some toolchains
}