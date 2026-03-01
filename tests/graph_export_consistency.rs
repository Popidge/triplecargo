use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use serde::Deserialize;
use triplecargo::solver::{
    graph_precompute_export, graph_precompute_export_with_sink, GraphJsonlSink, GraphSinkStats,
};
use triplecargo::{apply_move, legal_moves, load_cards_from_json, GameState, Rules};

fn cards_db() -> triplecargo::CardsDb {
    let path = Path::new("data/cards.json");
    load_cards_from_json(path).expect("failed to load cards.json")
}

#[derive(Default)]
struct CollectSink {
    lines: Vec<Vec<u8>>,
}

impl GraphJsonlSink for CollectSink {
    fn write_line(&mut self, json_line: &[u8], _turn: u8) -> io::Result<()> {
        self.lines.push(json_line.to_vec());
        Ok(())
    }

    fn finish_mut(&mut self) -> io::Result<GraphSinkStats> {
        Ok(GraphSinkStats {
            total_lines: self.lines.len() as u64,
            frame_count: 0,
            nodes_sha256_hex: None,
            index_sha256_hex: None,
            nodes_sha256_list: None,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ExportProbe {
    state_hash: String,
    value_target: i8,
}

fn collect_values_from_bytes(buf: &[u8]) -> BTreeMap<String, i8> {
    let mut out: BTreeMap<String, i8> = BTreeMap::new();
    for line in buf.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let probe: ExportProbe = serde_json::from_slice(line).expect("decode export line");
        let prev = out.insert(probe.state_hash, probe.value_target);
        assert!(prev.is_none(), "duplicate state_hash in export output");
    }
    out
}

fn collect_values_from_sink(sink: &CollectSink) -> BTreeMap<String, i8> {
    let mut out: BTreeMap<String, i8> = BTreeMap::new();
    for line in &sink.lines {
        let probe: ExportProbe = serde_json::from_slice(line).expect("decode sink export line");
        let prev = out.insert(probe.state_hash, probe.value_target);
        assert!(prev.is_none(), "duplicate state_hash in sink output");
    }
    out
}

#[test]
fn graph_export_entrypoints_consistent_values() {
    let cards = cards_db();
    let rules = Rules::default();
    let mut state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

    // Keep this test fast by moving to a near-terminal position.
    for _ in 0..7 {
        let mv = legal_moves(&state).into_iter().next().expect("legal move");
        state = apply_move(&state, &cards, mv).expect("apply_move");
    }
    let rem = 9 - state.board.filled_count();
    assert_eq!(rem, 2, "expected two plies remaining");

    let mut direct_out: Vec<u8> = Vec::new();
    graph_precompute_export(&state, &cards, Some(rem), &mut direct_out)
        .expect("direct graph export");

    let mut sink = CollectSink::default();
    let outcome = graph_precompute_export_with_sink(&state, &cards, Some(rem), &mut sink)
        .expect("sink graph export");
    let sink_stats = sink.finish_mut().expect("finish sink");

    let direct_values = collect_values_from_bytes(&direct_out);
    let sink_values = collect_values_from_sink(&sink);

    assert_eq!(
        direct_values, sink_values,
        "value_target mismatch between entrypoints"
    );
    assert_eq!(
        sink_stats.total_lines, outcome.totals_states,
        "state count mismatch"
    );
}
