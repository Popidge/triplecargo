use assert_cmd::prelude::*;
use std::fs;
use tempfile::tempdir;

fn count_lines(path: &std::path::Path) -> usize {
    let bytes = fs::read(path).expect("read file");
    // Count newline characters; each JSONL line ends with '\n'
    bytes.iter().filter(|&b| *b == b'\n').count()
}

#[test]
fn determinism_across_chunk_sizes_and_line_count() {
    let tmp = tempdir().expect("tmpdir");
    let p_def = tmp.path().join("out_default.jsonl");
    let p_c1 = tmp.path().join("out_chunk1.jsonl");
    let p_c64 = tmp.path().join("out_chunk64.jsonl");

    let games = 2usize;

    let base_args = [
        "--export-mode", "trajectory",
        "--games", "2",
        "--seed", "42",
        "--hand-strategy", "random",
        "--rules", "none",
        "--elements", "none",
        "--policy-format", "onehot",
        "--value-mode", "winloss",
        "--threads", "3",       // ensure multi-worker
        "--max-depth", "2",     // keep test fast; ignored in trajectory but harmless
    ];

    // Default (no --chunk-size flag; uses default 32)
    let mut cmd_def = assert_cmd::Command::cargo_bin("precompute").expect("bin");
    cmd_def.arg("--export").arg(&p_def);
    cmd_def.args(&base_args);
    let out_def = cmd_def.assert().success().get_output().clone();

    // --chunk-size 1
    let mut cmd_c1 = assert_cmd::Command::cargo_bin("precompute").expect("bin");
    cmd_c1.arg("--export").arg(&p_c1);
    cmd_c1.args(&base_args);
    cmd_c1.arg("--chunk-size").arg("1");
    let out_c1 = cmd_c1.assert().success().get_output().clone();

    // --chunk-size 64
    let mut cmd_c64 = assert_cmd::Command::cargo_bin("precompute").expect("bin");
    cmd_c64.arg("--export").arg(&p_c64);
    cmd_c64.args(&base_args);
    cmd_c64.arg("--chunk-size").arg("64");
    let out_c64 = cmd_c64.assert().success().get_output().clone();

    // Files must be byte-identical
    let f_def = fs::read(&p_def).expect("read default");
    let f_c1 = fs::read(&p_c1).expect("read chunk1");
    let f_c64 = fs::read(&p_c64).expect("read chunk64");
    assert_eq!(f_def, f_c1, "default chunk-size output must equal chunk-size=1 output");
    assert_eq!(f_def, f_c64, "default chunk-size output must equal chunk-size=64 output");

    // Line count invariant: 9 × games
    let expected_lines = 9 * games;
    assert_eq!(count_lines(&p_def), expected_lines, "default chunk-size line count mismatch");
    assert_eq!(count_lines(&p_c1), expected_lines, "chunk-size=1 line count mismatch");
    assert_eq!(count_lines(&p_c64), expected_lines, "chunk-size=64 line count mismatch");

    // Startup log includes chunk_size
    let s_def_out = String::from_utf8_lossy(&out_def.stdout);
    let s_c1_out = String::from_utf8_lossy(&out_c1.stdout);
    let s_c64_out = String::from_utf8_lossy(&out_c64.stdout);
    assert!(s_def_out.contains("[export] chunk_size="), "expected chunk_size log for default run");
    assert!(s_c1_out.contains("[export] chunk_size="), "expected chunk_size log for chunk-size=1 run");
    assert!(s_c64_out.contains("[export] chunk_size="), "expected chunk_size log for chunk-size=64 run");

    // Determinism across repeated run with same flags (chunk-size=64)
    let p_c64_b = tmp.path().join("out_chunk64_b.jsonl");
    let mut cmd_c64_b = assert_cmd::Command::cargo_bin("precompute").expect("bin");
    cmd_c64_b.arg("--export").arg(&p_c64_b);
    cmd_c64_b.args(&base_args);
    cmd_c64_b.arg("--chunk-size").arg("64");
    let _ = cmd_c64_b.assert().success();

    let f_c64_b = fs::read(&p_c64_b).expect("read chunk64_b");
    assert_eq!(f_c64, f_c64_b, "repeat runs must be byte-identical with same seed and flags");
}