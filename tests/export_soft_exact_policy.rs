use serde_json::Value;
use std::fs;
use tempfile::tempdir;

fn parse_jsonl(path: &std::path::Path) -> Vec<Value> {
    let text = fs::read_to_string(path).expect("read jsonl");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("valid json line"))
        .collect()
}

fn assert_move_mapping(rec: &Value, key: &str) {
    let Some(mv) = rec.get(key) else {
        return;
    };
    if mv.is_null() {
        return;
    }

    let action_id = mv
        .get("action_id")
        .and_then(Value::as_u64)
        .expect("move action_id must be u64") as usize;
    let cell = mv
        .get("cell")
        .and_then(Value::as_u64)
        .expect("move cell must be u64") as usize;
    let card_id = mv
        .get("card_id")
        .and_then(Value::as_u64)
        .expect("move card_id must be u64") as u16;

    assert_eq!(action_id % 9, cell, "{key} action_id must encode cell");

    let mask = rec
        .get("legal_moves_mask")
        .and_then(Value::as_array)
        .expect("legal_moves_mask array present");
    assert_eq!(mask.len(), 45, "legal_moves_mask length must be 45");
    let legal_bit = mask[action_id]
        .as_u64()
        .expect("legal_moves_mask values must be integers");
    assert_eq!(legal_bit, 1, "{key} action must be legal per mask");

    let to_move = rec
        .get("to_move")
        .and_then(Value::as_str)
        .expect("to_move present");
    let hand_key = if to_move == "A" { "A" } else { "B" };
    let hand = rec
        .get("hands")
        .and_then(|h| h.get(hand_key))
        .and_then(Value::as_array)
        .expect("hands for side to move present");
    assert!(
        hand.iter().any(|v| v.as_u64() == Some(card_id as u64)),
        "{key} card_id must exist in side-to-move hand"
    );
}

#[test]
fn soft_exact_probabilities_are_valid_and_legal() {
    let tmp = tempdir().expect("tmpdir");
    let p = tmp.path().join("soft_exact_probs.jsonl");

    let mut cmd = assert_cmd::Command::cargo_bin("precompute").expect("bin");
    cmd.arg("--export").arg(&p);
    cmd.args([
        "--export-mode",
        "trajectory",
        "--games",
        "2",
        "--seed",
        "123",
        "--hand-strategy",
        "random",
        "--rules",
        "none",
        "--elements",
        "none",
        "--policy-format",
        "soft_exact",
        "--soft-exact-temperature",
        "0.5",
        "--soft-exact-qmode",
        "margin",
        "--soft-exact-topk",
        "8",
        "--soft-exact-epsilon",
        "0.01",
        "--value-mode",
        "winloss",
        "--threads",
        "1",
    ]);
    cmd.assert().success();

    let lines = parse_jsonl(&p);
    assert!(!lines.is_empty(), "export must contain records");

    for rec in &lines {
        let mask = rec
            .get("legal_moves_mask")
            .and_then(Value::as_array)
            .expect("legal_moves_mask array present");
        assert_eq!(mask.len(), 45, "legal_moves_mask length must be 45");
        for bit in mask {
            let b = bit.as_u64().expect("mask entry integer");
            assert!(b == 0 || b == 1, "legal_moves_mask values must be 0/1");
        }

        let policy = rec.get("policy_target").expect("policy_target present");
        let format = policy
            .get("format")
            .and_then(Value::as_str)
            .expect("soft_exact format present");
        assert_eq!(format, "soft_exact");

        let entries = policy
            .get("entries")
            .and_then(Value::as_array)
            .expect("entries present");
        let mut sum_p = 0.0f64;
        for ent in entries {
            let action_id = ent
                .get("action_id")
                .and_then(Value::as_u64)
                .expect("entry action_id must be u64") as usize;
            let p = ent
                .get("p")
                .and_then(Value::as_f64)
                .expect("entry p must be f64");
            assert!(p.is_finite(), "soft_exact p must be finite");
            assert!(p >= 0.0, "soft_exact p must be non-negative");
            assert_eq!(
                mask[action_id].as_u64().expect("mask value integer"),
                1,
                "soft_exact entry action must be legal"
            );
            sum_p += p;
        }
        assert!(
            (sum_p - 1.0).abs() <= 1e-6,
            "soft_exact probabilities must sum to 1 (got {sum_p})"
        );
    }
}

#[test]
fn soft_exact_action_mapping_consistent_for_emitted_moves() {
    let tmp = tempdir().expect("tmpdir");
    let p = tmp.path().join("soft_exact_mapping.jsonl");

    let mut cmd = assert_cmd::Command::cargo_bin("precompute").expect("bin");
    cmd.arg("--export").arg(&p);
    cmd.args([
        "--export-mode",
        "trajectory",
        "--games",
        "1",
        "--seed",
        "42",
        "--hand-strategy",
        "random",
        "--rules",
        "none",
        "--elements",
        "none",
        "--policy-format",
        "soft_exact",
        "--soft-exact-temperature",
        "0.5",
        "--soft-exact-qmode",
        "margin",
        "--value-mode",
        "winloss",
        "--threads",
        "1",
    ]);
    cmd.assert().success();

    let lines = parse_jsonl(&p);
    assert!(!lines.is_empty(), "export must contain records");

    for rec in &lines {
        assert_move_mapping(rec, "best_move");
        assert_move_mapping(rec, "action_taken");
    }
}
