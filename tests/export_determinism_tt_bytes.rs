use assert_cmd::prelude::*;
use std::fs;
use tempfile::tempdir;

#[test]
fn determinism_across_tt_bytes() {
    let tmp = tempdir().expect("tmpdir");
    let p1 = tmp.path().join("out16.jsonl");
    let p2 = tmp.path().join("out64.jsonl");

    let base_args = [
        "--export-mode", "trajectory",
        "--games", "1",
        "--seed", "42",
        "--hand-strategy", "random",
        "--rules", "none",
        "--elements", "none",
        "--policy-format", "onehot",
        "--value-mode", "winloss",
        "--threads", "1",
        "--max-depth", "2",
    ];

    // Run with 16 MiB
    let mut cmd1 = assert_cmd::Command::cargo_bin("precompute").expect("bin");
    cmd1.arg("--export").arg(&p1);
    cmd1.args(&base_args);
    cmd1.arg("--tt-bytes").arg("16");
    let out1 = cmd1.assert().success().get_output().clone();

    // Run with 64 MiB
    let mut cmd2 = assert_cmd::Command::cargo_bin("precompute").expect("bin");
    cmd2.arg("--export").arg(&p2);
    cmd2.args(&base_args);
    cmd2.arg("--tt-bytes").arg("64");
    let out2 = cmd2.assert().success().get_output().clone();

    let f1 = fs::read(&p1).expect("read out16");
    let f2 = fs::read(&p2).expect("read out64");
    assert_eq!(f1, f2, "exported JSONL must be identical for different tt sizes");

    let s1 = String::from_utf8_lossy(&out1.stderr);
    let s2 = String::from_utf8_lossy(&out2.stderr);
    assert!(s1.contains("TT target="), "expected TT log in stderr for tt-bytes=16");
    assert!(s2.contains("TT target="), "expected TT log in stderr for tt-bytes=64");
}