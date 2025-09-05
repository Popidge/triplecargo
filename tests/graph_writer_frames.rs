use std::fs::File;
use tempfile::tempdir;

use triplecargo::solver::{GraphJsonlSink, ZstdFramesJsonlWriter};

fn make_line(len: usize, ch: u8) -> Vec<u8> {
    vec![ch; len]
}

#[test]
fn zstd_writer_frame_bytes_cap_triggers_multiple_frames() {
    let td = tempdir().expect("temp dir");
    let nodes_path = td.path().join("nodes.jsonl.zst");
    let idx_path = td.path().join("nodes.idx.jsonl");

    let nodes = File::create(&nodes_path).expect("create nodes");
    let idx = Some(File::create(&idx_path).expect("create idx"));

    // Small cap: 1 MiB to force multiple frames under a few MiB of uncompressed input
    let frame_bytes = 1 * 1024 * 1024;
    let frame_lines = usize::MAX / 2; // effectively disable line-based cap
    let buf_cap = triplecargo::solver::graph_writer::BUF_WRITER_CAP_BYTES;

    let mut w = ZstdFramesJsonlWriter::new(
        nodes,
        idx,
        buf_cap,
        3,           // zstd level
        1,           // zstd threads
        frame_lines,
        frame_bytes, // frame max bytes (soft cap, uncompressed)
        false,       // sync_final
        true,        // hashing enabled
    );

    // Write ~3.2 MiB uncompressed payload: 800 lines of 4096 bytes (+ newline added by writer)
    let line = make_line(4096, b'a');
    for _ in 0..800 {
        w.write_line(&line, 0).expect("write_line");
    }
    let stats = w.finish_mut().expect("finish");

    // Expect multiple frames due to 1 MiB cap
    assert!(
        stats.frame_count >= 2,
        "expected at least 2 frames, got {}",
        stats.frame_count
    );
    assert!(stats.total_lines == 800, "total_lines must match input");
}

#[test]
fn zstd_writer_large_frame_bytes_cap_aggregates_into_single_frame() {
    let td = tempdir().expect("temp dir");
    let nodes_path = td.path().join("nodes2.jsonl.zst");
    let idx_path = td.path().join("nodes2.idx.jsonl");

    let nodes = File::create(&nodes_path).expect("create nodes");
    let idx = Some(File::create(&idx_path).expect("create idx"));

    // Large cap: 128 MiB (default), with only ~4 MiB uncompressed input, should be 1 frame
    let frame_bytes = 128 * 1024 * 1024;
    let frame_lines = usize::MAX / 2; // effectively disable line-based cap
    let buf_cap = triplecargo::solver::graph_writer::BUF_WRITER_CAP_BYTES;

    let mut w = ZstdFramesJsonlWriter::new(
        nodes,
        idx,
        buf_cap,
        3,            // zstd level
        1,            // zstd threads
        frame_lines,
        frame_bytes,  // large cap
        false,        // sync_final
        true,         // hashing enabled
    );

    // ~4 MiB uncompressed: 1024 lines of 4096 bytes (+ newline)
    let line = make_line(4096, b'b');
    for _ in 0..1024 {
        w.write_line(&line, 0).expect("write_line");
    }
    let stats = w.finish_mut().expect("finish");

    assert_eq!(stats.frame_count, 1, "expected a single frame under 128 MiB cap");
    assert!(stats.total_lines == 1024, "total_lines must match input");
}