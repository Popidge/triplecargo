use std::fs;
use std::fs::File;
use tempfile::tempdir;

use triplecargo::solver::{
    AsyncZstdFramesJsonlWriter,
    GraphJsonlSink,
};
use serde_json::Value;

fn write_n_json_lines(
    sink: &mut dyn GraphJsonlSink,
    n: usize,
) {
    for i in 0..n {
        let line = format!("{{\"i\":{}}}", i);
        // turn field can be constant for these tests
        sink.write_line(line.as_bytes(), 0).expect("write_line");
    }
}

#[test]
fn async_ordering_preserved_with_workers_2_and_index_enabled() {
    let td = tempdir().expect("temp dir");
    let nodes_path = td.path().join("nodes.zst");
    let idx_path = td.path().join("nodes.idx.jsonl");

    let nodes = File::create(&nodes_path).expect("create nodes");
    let idx = Some(File::create(&idx_path).expect("create idx"));

    // Frame per line to stress ordering
    let frame_lines = 1usize;
    let frame_bytes = 1usize << 20; // 1 MiB
    let buf_cap = triplecargo::solver::graph_writer::BUF_WRITER_CAP_BYTES;

    let mut w = AsyncZstdFramesJsonlWriter::new(
        nodes,
        idx,
        buf_cap,
        3,                 // zstd level
        1,                 // zstd threads per encoder
        frame_lines,
        frame_bytes,
        4,                 // raw frames queue
        2,                 // zstd_workers (pool)
        4,                 // compressed frames queue
        false,             // sync_final
    );

    let n = 64usize;
    write_n_json_lines(&mut w, n);
    let stats = w.finish_mut().expect("finish");

    assert_eq!(stats.frame_count as usize, n, "one frame per line");
    assert_eq!(stats.total_lines as usize, n, "line accounting");

    // Parse index and confirm strict frame id and line_start ordering
    let idx_bytes = fs::read(&idx_path).expect("read idx");
    let s = String::from_utf8(idx_bytes).expect("utf8 idx");
    let mut expected_frame = 0u64;
    let mut expected_line_start = 0u64;
    for line in s.lines() {
        let v: Value = serde_json::from_str(line).expect("json idx line");
        let f = v.get("frame").and_then(|x| x.as_u64()).expect("frame");
        let ls = v.get("line_start").and_then(|x| x.as_u64()).expect("line_start");
        let lines = v.get("lines").and_then(|x| x.as_u64()).expect("lines");
        assert_eq!(f, expected_frame, "frame must be sequential");
        assert_eq!(ls, expected_line_start, "line_start must be sequential");
        assert_eq!(lines, 1u64, "frame_lines_target=1 means 1 line per frame");
        expected_frame += 1;
        expected_line_start += 1;
    }
    assert_eq!(expected_frame as usize, n);
    assert_eq!(expected_line_start as usize, n);
}

#[test]
fn async_workers0_and_workers2_yield_identical_outputs_small_payload() {
    let td0 = tempdir().expect("td0");
    let td2 = tempdir().expect("td2");

    let p0_nodes = td0.path().join("nodes.zst");
    let p0_idx = td0.path().join("nodes.idx.jsonl");
    let p2_nodes = td2.path().join("nodes.zst");
    let p2_idx = td2.path().join("nodes.idx.jsonl");

    let buf_cap = triplecargo::solver::graph_writer::BUF_WRITER_CAP_BYTES;
    let frame_lines = 4usize;
    let frame_bytes = 128usize * 1024 * 1024;

    // workers=0 (single-stage)
    {
        let nodes = File::create(&p0_nodes).expect("nodes0");
        let idx = Some(File::create(&p0_idx).expect("idx0"));
        let mut w0 = AsyncZstdFramesJsonlWriter::new(
            nodes,
            idx,
            buf_cap,
            3,
            1,
            frame_lines,
            frame_bytes,
            8,  // raw queue
            0,  // workers=0 => single-stage
            4,  // compressed queue unused
            false,
        );
        write_n_json_lines(&mut w0, 32);
        let _ = w0.finish_mut().expect("finish0");
    }

    // workers=2 (two-stage)
    {
        let nodes = File::create(&p2_nodes).expect("nodes2");
        let idx = Some(File::create(&p2_idx).expect("idx2"));
        let mut w2 = AsyncZstdFramesJsonlWriter::new(
            nodes,
            idx,
            buf_cap,
            3,
            1,
            frame_lines,
            frame_bytes,
            8,
            2,  // workers=2
            4,
            false,
        );
        write_n_json_lines(&mut w2, 32);
        let _ = w2.finish_mut().expect("finish2");
    }

    // Compare files byte-for-byte
    let n0 = fs::read(&p0_nodes).expect("r0");
    let n2 = fs::read(&p2_nodes).expect("r2");
    assert_eq!(n0, n2, "nodes.zst should be byte-identical for same inputs");

    let i0 = fs::read(&p0_idx).expect("ri0");
    let i2 = fs::read(&p2_idx).expect("ri2");
    assert_eq!(i0, i2, "index must match exactly");
}

#[test]
fn async_no_deadlock_with_index_disabled() {
    let td = tempdir().expect("temp dir");
    let nodes_path = td.path().join("nodes.zst");
    let nodes = File::create(&nodes_path).expect("nodes");

    let buf_cap = triplecargo::solver::graph_writer::BUF_WRITER_CAP_BYTES;

    // Small queues to exercise back-pressure; index disabled (None)
    let mut w = AsyncZstdFramesJsonlWriter::new(
        nodes,
        None,
        buf_cap,
        3,
        1,
        1,                  // frame_lines_target
        1 * 1024 * 1024,    // frame cap
        2,                  // raw frames queue
        2,                  // workers
        1,                  // compressed queue
        false,
    );
    write_n_json_lines(&mut w, 128);
    let stats = w.finish_mut().expect("finish");
    assert_eq!(stats.total_lines as usize, 128);
    // No index digest expected
    assert!(stats.index_sha256_hex.is_none());
}