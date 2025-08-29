use assert_cmd::prelude::*;
use std::process::Command;

fn parse_log_numbers(s: &str) -> Option<(usize, usize, f64)> {
    // Expected line:
    // [worker 0] TT target=32 MiB capacity=8388608 entries ≈31.9 MiB
    // or for full mode:
    // [full] TT target=32 MiB capacity=8388608 entries ≈31.9 MiB
    let tgt_idx = s.find("target=")?;
    let after_tgt = &s[tgt_idx + "target=".len()..];
    let mi_idx = after_tgt.find(" MiB")?;
    let target_mib_str = &after_tgt[..mi_idx];
    let target_mib: usize = target_mib_str.trim().parse().ok()?;

    let cap_idx = s.find("capacity=")?;
    let after_cap = &s[cap_idx + "capacity=".len()..];
    let cap_end = after_cap.find(|c: char| !c.is_ascii_digit()).unwrap_or(after_cap.len());
    let capacity_str = &after_cap[..cap_end];
    let capacity: usize = capacity_str.parse().ok()?;

    let approx_idx = s.find("entries ≈")?;
    let after_approx = &s[approx_idx + "entries ≈".len()..];
    let approx_end = after_approx.find(" MiB").unwrap_or(after_approx.len());
    let approx_str = &after_approx[..approx_end];
    let approx_mib: f64 = approx_str.trim().parse().ok()?;

    Some((target_mib, capacity, approx_mib))
}

fn is_power_of_two(x: usize) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

#[test]
fn tt_capacity_respects_budget_trajectory_16_and_64() {
    // Trajectory mode with threads=1 so only one worker prints a line
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
    ];

    // 16 MiB
    let tmp16 = tempfile::NamedTempFile::new().expect("tmp file");
    let mut cmd16 = Command::cargo_bin("precompute").expect("bin");
    cmd16.arg("--export").arg(tmp16.path());
    cmd16.args(&base_args);
    cmd16.arg("--tt-bytes").arg("16");
    let out16 = cmd16.output().expect("run precompute 16");
    assert!(out16.status.success(), "precompute exit != 0 for 16 MiB");

    let stderr16 = String::from_utf8_lossy(&out16.stderr);
    let (tgt16, cap16, approx16) = parse_log_numbers(&stderr16)
        .expect("failed to parse TT log for 16 MiB");
    assert_eq!(tgt16, 16, "target MiB mismatch in log for 16");
    assert!(is_power_of_two(cap16), "capacity must be power-of-two");
    assert!(approx16 > 0.0, "approx MiB must be positive");
    assert!(approx16 <= 16.0 + 0.01, "approx MiB must be <= target");

    // 64 MiB
    let tmp64 = tempfile::NamedTempFile::new().expect("tmp file");
    let mut cmd64 = Command::cargo_bin("precompute").expect("bin");
    cmd64.arg("--export").arg(tmp64.path());
    cmd64.args(&base_args);
    cmd64.arg("--tt-bytes").arg("64");
    let out64 = cmd64.output().expect("run precompute 64");
    assert!(out64.status.success(), "precompute exit != 0 for 64 MiB");

    let stderr64 = String::from_utf8_lossy(&out64.stderr);
    let (tgt64, cap64, approx64) = parse_log_numbers(&stderr64)
        .expect("failed to parse TT log for 64 MiB");
    assert_eq!(tgt64, 64, "target MiB mismatch in log for 64");
    assert!(is_power_of_two(cap64), "capacity must be power-of-two");
    assert!(approx64 > 0.0, "approx MiB must be positive");
    assert!(approx64 <= 64.0 + 0.01, "approx MiB must be <= target");

    // Monotonicity sanity: capacity for 64 should be >= capacity for 16
    assert!(cap64 >= cap16, "larger budget should not reduce capacity");
}

/* #[test]
fn tt_capacity_full_mode_uses_budget() {
    // Full mode ignores games; run with threads not applicable here
    let tmp = tempfile::NamedTempFile::new().expect("tmp file");
    let mut cmd = Command::cargo_bin("precompute").expect("bin");
    cmd.arg("--export").arg(tmp.path());
    cmd.args([
        "--export-mode", "full",
        "--seed", "42",
        "--hand-strategy", "random",
        "--rules", "none",
        "--elements", "none",
        "--policy-format", "onehot",
        "--value-mode", "winloss",
        "--tt-bytes", "32",
        "--max-depth", "2",
    ]);
    let out = cmd.output().expect("run precompute full");
    assert!(out.status.success(), "precompute exit != 0 in full mode");

    let stderr = String::from_utf8_lossy(&out.stderr);
    let (tgt, cap, approx) = parse_log_numbers(&stderr)
        .expect("failed to parse TT log for full mode");
    assert_eq!(tgt, 32, "target MiB mismatch in full mode log");
    assert!(is_power_of_two(cap), "capacity must be power-of-two");
    assert!(approx > 0.0 && approx <= 32.0 + 0.01, "approx MiB must be within budget");
} */
