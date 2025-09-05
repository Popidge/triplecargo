use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use tempfile::tempdir;
use std::path::Path;

#[test]
fn shards_two_creates_shards_and_index() {
    // Create a temp dir for export
    let td = tempdir().expect("tempdir");
    let export_dir = td.path().join("export");
    // Run precompute in graph mode with small frames and 2 shards
    let mut cmd = assert_cmd::Command::cargo_bin("precompute").expect("bin");
    cmd.arg("--export").arg(&export_dir);
    cmd.arg("--export-mode").arg("graph");
    // keep workload small / deterministic
    cmd.arg("--shards").arg("2");
    // Small frames so writer emits multiple frames quickly
    cmd.arg("--zstd-frame-bytes").arg("1024");
    cmd.arg("--zstd-frame-lines").arg("16");
    // Force single-stage compression to reduce threads / nondet behavior
    cmd.arg("--zstd-workers").arg("0");
    // Avoid final fsync to speed up tests
    cmd.arg("--sync-mode").arg("none");
    // Ensure writer queue small for quick backpressure behavior
    cmd.arg("--writer-queue-frames").arg("2");

    cmd.assert().success();

    // Validate outputs
    // Expect nodes_000.jsonl.zst and nodes_001.jsonl.zst and shared index nodes.idx.jsonl
    let n0 = export_dir.join("nodes_000.jsonl.zst");
    let n1 = export_dir.join("nodes_001.jsonl.zst");
    let idx = export_dir.join("nodes.idx.jsonl");
    let manifest = export_dir.join("graph.manifest.json");

    assert!(n0.exists(), "expected shard file {}", n0.display());
    assert!(n1.exists(), "expected shard file {}", n1.display());
    assert!(idx.exists(), "expected shared index {}", idx.display());
    assert!(manifest.exists(), "expected manifest {}", manifest.display());

    // Index lines should include "shard" field in at least one line
    let idx_bytes = fs::read(&idx).expect("read index");
    let idx_s = String::from_utf8_lossy(&idx_bytes);
    assert!(idx_s.contains("\"shard\""), "index should include shard field: {}", idx_s);

    // Manifest should list nodes as array and include nodes_sha256_list when present
    let mf = fs::read_to_string(&manifest).expect("read manifest");
    assert!(mf.contains("\"nodes\": ["), "manifest nodes should be an array: {}", mf);
    // Integrity per-shard key
    assert!(mf.contains("nodes_sha256") || mf.contains("nodes_sha256_list"), "expected nodes integrity in manifest: {}", mf);

    // Quick sanity: shard files are non-empty (relaxed under fast_tests)
    if cfg!(feature = "fast_tests") {
        // Under fast_tests we may write small stub files; just assert they exist and index/manifest are present.
        // Already asserted above.
    } else {
        let s0 = fs::metadata(&n0).expect("meta n0").len();
        let s1 = fs::metadata(&n1).expect("meta n1").len();
        assert!(s0 > 0, "shard0 should be non-empty");
        assert!(s1 > 0, "shard1 should be non-empty");
    }
}