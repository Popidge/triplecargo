use std::fs::{self, File};
use tempfile::tempdir;

use triplecargo::solver::{AsyncShardedZstdFramesJsonlWriter, GraphJsonlSink};

#[test]
fn sharded_writer_simple_smoke() {
    // Small workspace
    let td = tempdir().expect("tmpdir");
    let dir = td.path();

    // Create shard files and shared index
    let shard0 = dir.join("nodes_000.jsonl.zst");
    let shard1 = dir.join("nodes_001.jsonl.zst");
    let idx = dir.join("nodes.idx.jsonl");

    let f0 = File::create(&shard0).expect("create shard0");
    let f1 = File::create(&shard1).expect("create shard1");
    let idxf = File::create(&idx).expect("create idx");

    // Small buf cap and tiny frame limits so we exercise multiple frames easily
    let buf_cap = 1 * 1024 * 1024; // 1 MiB
    let zstd_level = 1;
    let zstd_threads = 1;
    let frame_lines = 1;
    let frame_bytes = 512; // tiny so each line forms a frame
    let writer_queue_frames = 4;
    let sync_final = false;

    let mut writer = AsyncShardedZstdFramesJsonlWriter::new(
        vec![f0, f1],
        Some(idxf),
        buf_cap,
        zstd_level,
        zstd_threads,
        frame_lines,
        frame_bytes,
        writer_queue_frames,
        sync_final,
        true, // hashing enabled
    );

    // Emit a small number of lines (10) so both shards get some frames
    for i in 0..10u8 {
        let line = format!(r#"{{"i":{}}}"#, i);
        writer.write_line(line.as_bytes(), (i % 3) as u8).expect("write_line");
    }

    let stats = writer.finish_mut().expect("finish");

    // Expect some frames and per-shard digests
    assert!(stats.frame_count > 0, "frame_count should be > 0");
    assert!(stats.total_lines >= 10, "total_lines should be >= 10");
    assert!(stats.nodes_sha256_list.is_some(), "expected per-shard digests");

    // Files exist and non-empty
    let s0 = fs::metadata(&shard0).expect("meta0").len();
    let s1 = fs::metadata(&shard1).expect("meta1").len();
    let sidx = fs::metadata(&idx).expect("metaidx").len();

    assert!(s0 > 0, "shard0 non-empty");
    assert!(s1 > 0, "shard1 non-empty");
    assert!(sidx > 0, "index non-empty");
}