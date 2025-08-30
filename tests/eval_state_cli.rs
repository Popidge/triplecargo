use assert_cmd::prelude::*;
use predicates::prelude::*;
use serde::Deserialize;
use std::io::Write;
use std::process::{Command, Stdio};

use triplecargo::{Rules, GameState, zobrist_key};
use triplecargo::cards::load_cards_from_json;
use triplecargo::solver::search_root;
use triplecargo::solver::tt_array::FixedTT;

#[derive(Deserialize)]
struct EvalMoveOut {
    card_id: u16,
    cell: u8,
}

#[derive(Deserialize)]
struct EvalOut {
    #[serde(default)]
    best_move: Option<EvalMoveOut>,
    value: i8,
    margin: i8,
    pv: Vec<EvalMoveOut>,
    nodes: u64,
    depth: u8,
    state_hash: String,
}

fn mk_state_basic(hand_a: [u16; 5], hand_b: [u16; 5]) -> GameState {
    let rules = Rules::basic_only();
    GameState::with_hands(rules, hand_a, hand_b, None)
}

fn run_with_stdin(input: &str, args: &[&str]) -> std::process::Output {
    let mut cmd = Command::cargo_bin("precompute").expect("binary exists");
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn");
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin.write_all(input.as_bytes()).expect("write stdin");
    }
    child.wait_with_output().expect("wait output")
}

#[test]
fn test_eval_known_pv_matches_solver() {
    // Simple opening: Basic rules, no elements
    let hand_a = [1, 2, 3, 4, 5];
    let hand_b = [6, 7, 8, 9, 10];
    let state = mk_state_basic(hand_a, hand_b);

    // Expected from in-process solver
    let cards = load_cards_from_json("data/cards.json").expect("cards load");
    let depth = 9 - state.board.filled_count();
    let mut tt = FixedTT::with_capacity_pow2(FixedTT::capacity_for_budget_bytes(8 * 1024 * 1024));
    let (val, bm, _nodes) = search_root(&state, &cards, depth, &mut tt);
    let expected_value = if val > 0 { 1 } else if val < 0 { -1 } else { 0 };
    let expected_bm = bm.expect("non-terminal opening must have a best move");

    // Prepare JSON input matching export schema subset
    let input_json = serde_json::json!({
        "board": [
            {"cell":0,"card_id":null,"owner":null},
            {"cell":1,"card_id":null,"owner":null},
            {"cell":2,"card_id":null,"owner":null},
            {"cell":3,"card_id":null,"owner":null},
            {"cell":4,"card_id":null,"owner":null},
            {"cell":5,"card_id":null,"owner":null},
            {"cell":6,"card_id":null,"owner":null},
            {"cell":7,"card_id":null,"owner":null},
            {"cell":8,"card_id":null,"owner":null}
        ],
        "hands": { "A": hand_a, "B": hand_b },
        "to_move": "A",
        "turn": 0,
        "rules": {
            "elemental": false,
            "same": false,
            "plus": false,
            "same_wall": false
        }
    }).to_string();

    // Run CLI
    let output = run_with_stdin(&input_json, &["--eval-state", "--cards", "data/cards.json", "--tt-bytes", "8"]);
    assert!(output.status.success(), "process must succeed");
    let stdout = String::from_utf8(output.stdout.clone()).expect("utf8 stdout");

    // Exactly one JSON object line
    assert!(predicate::str::is_match(r#"^\{.*\}\r?\n?$"#).unwrap().eval(&stdout));
    assert!(!String::from_utf8(output.stderr.clone()).unwrap().contains("[eval]"));

    // Parse stdout JSON
    let eval: EvalOut = serde_json::from_str(&stdout).expect("json parse output");

    // Validate
    assert_eq!(eval.value, expected_value, "win/draw/loss value must match solver");
    let bm_out = eval.best_move.expect("best_move must be present at non-terminal");
    assert_eq!(bm_out.card_id, expected_bm.card_id);
    assert_eq!(bm_out.cell, expected_bm.cell);

    // Validate state_hash
    let expected_hash = format!("{:032x}", zobrist_key(&state));
    assert_eq!(eval.state_hash, expected_hash, "state_hash must match zobrist of input state");
    assert_eq!(eval.depth, depth);
}

#[test]
fn test_eval_determinism_two_runs_identical() {
    let input_json = serde_json::json!({
      "board": [
        {"cell":0,"card_id":null,"owner":null},
        {"cell":1,"card_id":null,"owner":null},
        {"cell":2,"card_id":null,"owner":null},
        {"cell":3,"card_id":null,"owner":null},
        {"cell":4,"card_id":null,"owner":null},
        {"cell":5,"card_id":null,"owner":null},
        {"cell":6,"card_id":null,"owner":null},
        {"cell":7,"card_id":null,"owner":null},
        {"cell":8,"card_id":null,"owner":null}
      ],
      "hands": { "A":[1,2,3,4,5], "B":[6,7,8,9,10] },
      "to_move": "A",
      "turn": 0,
      "rules": { "elemental": false, "same": false, "plus": false, "same_wall": false }
    }).to_string();

    let out1 = run_with_stdin(&input_json, &["--eval-state", "--cards", "data/cards.json", "--tt-bytes", "8"]);
    assert!(out1.status.success(), "run1 must succeed");
    let out2 = run_with_stdin(&input_json, &["--eval-state", "--cards", "data/cards.json", "--tt-bytes", "8"]);
    assert!(out2.status.success(), "run2 must succeed");

    let s1 = String::from_utf8(out1.stdout).unwrap();
    let s2 = String::from_utf8(out2.stdout).unwrap();
    assert_eq!(s1, s2, "identical input must produce identical output");
}

#[test]
fn test_eval_terminal_state_omits_best_move() {
    // Terminal board: fill 9 cells; hands empty. Owners arbitrary; rules basic.
    let input_json = serde_json::json!({
      "board": [
        {"cell":0,"card_id":1,"owner":"A"},
        {"cell":1,"card_id":2,"owner":"B"},
        {"cell":2,"card_id":3,"owner":"A"},
        {"cell":3,"card_id":4,"owner":"B"},
        {"cell":4,"card_id":5,"owner":"A"},
        {"cell":5,"card_id":6,"owner":"B"},
        {"cell":6,"card_id":7,"owner":"A"},
        {"cell":7,"card_id":8,"owner":"B"},
        {"cell":8,"card_id":9,"owner":"A"}
      ],
      "hands": { "A": [], "B": [] },
      "to_move": "A",
      "turn": 9,
      "rules": { "elemental": false, "same": false, "plus": false, "same_wall": false }
    }).to_string();

    let output = run_with_stdin(&input_json, &["--eval-state", "--cards", "data/cards.json"]);
    assert!(output.status.success(), "terminal eval must succeed");
    let s = String::from_utf8(output.stdout).unwrap();
    let eval: serde_json::Value = serde_json::from_str(&s).expect("json");

    // best_move must be absent
    assert!(eval.get("best_move").is_none(), "best_move must be omitted at terminal");
}

#[test]
fn test_eval_invalid_json_exit_1() {
    let bad = r#"{ "board": "oops", "#; // malformed
    let output = run_with_stdin(bad, &["--eval-state", "--cards", "data/cards.json"]);
    assert!(!output.status.success(), "invalid json must fail");
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.to_lowercase().contains("invalid json") || err.to_lowercase().contains("error"), "stderr should contain parse error, got: {}", err);
}