use std::fs::File;
use tempfile::tempdir;

use triplecargo::solver::{PlainJsonlWriter, GraphJsonlSink};
use triplecargo::solver::graph_writer::BUF_WRITER_CAP_BYTES;

#[test]
fn plain_writer_nohash_finish_returns_none_digest() {
    let td = tempdir().expect("tempdir");
    let nodes_path = td.path().join("nodes.jsonl");
    let f = File::create(&nodes_path).expect("create nodes");
    let buf_cap = BUF_WRITER_CAP_BYTES;
    // sync_final = false, hashing = false
    let mut w = PlainJsonlWriter::new(f, buf_cap, false, false);
    w.write_line(b"{\"x\":1}", 0).expect("write_line");
    let stats = w.finish_mut().expect("finish");

    assert_eq!(stats.total_lines, 1);
    assert!(stats.nodes_sha256_hex.is_none(), "expected None when hashing disabled");
    assert!(stats.index_sha256_hex.is_none());
    assert!(stats.nodes_sha256_list.is_none());

    // file should exist and be non-empty
    let meta = std::fs::metadata(&nodes_path).expect("meta");
    assert!(meta.len() > 0);
}