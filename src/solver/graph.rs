// Two-phase full-graph precompute:
// Phase A: Parallel-ready enumeration with shared visited set
// Phase B: Retrograde exact solve by depth (9 -> 0)
//
// This module provides core types and entrypoints for the --export-mode graph pipeline.

use std::error::Error;
use std::hash::BuildHasherDefault;
use std::sync::{Arc, Mutex};
use std::io::Write;

use hashbrown::HashSet as HbHashSet;
use hashbrown::HashMap as HbHashMap;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use serde::Serialize;
 
use crate::board::{Board};
use crate::cards::CardsDb;
use crate::engine::apply::apply_move;
use crate::engine::score::score;
use crate::hash::{recompute_zobrist, zobrist_key};
use crate::rules::Rules;
use crate::state::{GameState, legal_moves};
use crate::types::{Owner};

use crate::solver::graph_writer::GraphJsonlSink;
use sha2::{Digest, Sha256};
type FastHasher = BuildHasherDefault<ahash::AHasher>;
type FastSet = HbHashSet<u128, FastHasher>;

/// Compact state for storage.
/// Hands are not stored; they are reconstructed exactly from the initial hands and board contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedState {
    pub board: Board,
    pub next: Owner,
    pub rules: Rules,
}

impl PackedState {
    #[inline]
    pub fn from_state(s: &GameState) -> Self {
        Self {
            board: s.board.clone(),
            next: s.next,
            rules: s.rules,
        }
    }

    /// Reconstruct a full GameState using the initial hands (A,B) that produced this position.
    /// Important: remove placed cards from the original owner determined by initial hand membership,
    /// NOT the current board owner (cards may have been flipped).
    #[inline]
    pub fn to_state(&self, initial_a: [u16; 5], initial_b: [u16; 5]) -> GameState {
        // Build multiset counts of how many of each initial card remain for each player
        fn to_counts(hand: [u16; 5]) -> hashbrown::HashMap<u16, u8, FastHasher> {
            let mut m: hashbrown::HashMap<u16, u8, FastHasher> = hashbrown::HashMap::with_hasher(FastHasher::default());
            for id in hand {
                if id != 0 {
                    *m.entry(id).or_insert(0) += 1;
                }
            }
            m
        }
        let mut cnt_a = to_counts(initial_a);
        let mut cnt_b = to_counts(initial_b);

        // For every placed card on the board, decrement from the initial owner who had that card id.
        for cell in 0u8..9 {
            if let Some(slot) = self.board.get(cell) {
                let id = slot.card_id;
                if let Some(e) = cnt_a.get_mut(&id) {
                    if *e > 0 {
                        *e -= 1;
                        continue;
                    }
                }
                if let Some(e) = cnt_b.get_mut(&id) {
                    if *e > 0 {
                        *e -= 1;
                        continue;
                    }
                }
                // If neither had this id, the position is inconsistent with the provided initial hands.
                // We allow it to proceed with best-effort (hands won't include this id).
            }
        }

        // Rebuild hands arrays from remaining counts, sorted ascending for determinism.
        fn counts_to_hand(mut cnt: hashbrown::HashMap<u16, u8, FastHasher>) -> [Option<u16>; 5] {
            let mut v: Vec<u16> = Vec::with_capacity(5);
            for (id, n) in cnt.drain() {
                for _ in 0..n {
                    v.push(id);
                }
            }
            v.sort_unstable();
            let mut out = [None; 5];
            for (i, id) in v.into_iter().enumerate() {
                if i < 5 {
                    out[i] = Some(id);
                }
            }
            out
        }

        let hands_a = counts_to_hand(cnt_a);
        let hands_b = counts_to_hand(cnt_b);

        let mut s = GameState {
            board: self.board.clone(),
            hands_a,
            hands_b,
            next: self.next,
            rules: self.rules,
            zobrist: 0,
        };
        s.zobrist = recompute_zobrist(&s);
        s
    }
}

/// Sharded concurrent visited set keyed by zobrist (u128).
/// try_insert() returns true only the first time a key is observed.
pub struct SharedVisited {
    shards: Vec<Mutex<FastSet>>,
    mask: usize,
}

impl SharedVisited {
    /// Create with shard_count rounded up to next power of two.
    pub fn new(shard_count: usize) -> Self {
        let sc = shard_count.next_power_of_two().max(1);
        let mut shards = Vec::with_capacity(sc);
        for _ in 0..sc {
            shards.push(Mutex::new(HbHashSet::with_hasher(FastHasher::default())));
        }
        Self { shards, mask: sc - 1 }
    }

    #[inline]
    fn shard_index(&self, key: u128) -> usize {
        (key as u64 as usize) & self.mask
    }

    /// Returns true if the key was not present and is inserted now.
    #[inline]
    pub fn try_insert(&self, key: u128) -> bool {
        let idx = self.shard_index(key);
        let mut guard = self.shards[idx].lock().unwrap();
        guard.insert(key)
    }

    /// Rough count across shards (not precise).
    pub fn len_approx(&self) -> usize {
        self.shards
            .iter()
            .map(|m| m.lock().unwrap().len())
            .sum()
    }
}

/// Buckets of graph states by filled cells (depth).
/// Each layer holds (zobrist, PackedState) pairs.
pub struct GraphBuckets {
    pub layers: Vec<Vec<(u128, PackedState)>>,
}

impl GraphBuckets {
    #[inline]
    pub fn total_states(&self) -> usize {
        self.layers.iter().map(|v| v.len()).sum()
    }
}

/// Phase A: Enumerate all reachable states from the initial position up to an optional remaining-plies cap.
/// Uses a shared visited set to deduplicate states.
/// BFS over ply depth ensures stable layering.
pub fn enumerate_graph(
    initial: &GameState,
    cards: &CardsDb,
    max_depth: Option<u8>,
) -> GraphBuckets {
    // Buckets per ply depth 0..=9
    let mut layers: Vec<Vec<(u128, PackedState)>> = vec![Vec::new(); 10];
    let visited = Arc::new(SharedVisited::new(256));

    let start_filled = initial.board.filled_count();
    let rem_full = 9 - start_filled;
    let rem_cap = max_depth.map(|c| c.min(rem_full)).unwrap_or(rem_full);
    let end_depth = start_filled + rem_cap;

    // Root
    let root_key = zobrist_key(initial);
    if visited.try_insert(root_key) {
        layers[start_filled as usize].push((root_key, PackedState::from_state(initial)));
    }

    // BFS by depth, with parallel expansion per layer
    let mut current: Vec<GameState> = vec![initial.clone()];
    for d in start_filled..end_depth {
        let visited_ref = Arc::clone(&visited);
        let results: Vec<(Vec<(u128, PackedState)>, Vec<GameState>)> = current
            .par_iter()
            .map(|s| {
                let mut recs: Vec<(u128, PackedState)> = Vec::new();
                let mut next_local: Vec<GameState> = Vec::new();
                for mv in legal_moves(s) {
                    if let Ok(ns) = apply_move(s, cards, mv) {
                        let k = zobrist_key(&ns);
                        if visited_ref.try_insert(k) {
                            recs.push((k, PackedState::from_state(&ns)));
                            if (d + 1) < end_depth && !ns.is_terminal() {
                                next_local.push(ns);
                            }
                        }
                    }
                }
                (recs, next_local)
            })
            .collect();

        let mut next: Vec<GameState> = Vec::new();
        let layer_idx = (d + 1) as usize;
        for (recs, next_local) in results {
            if !recs.is_empty() {
                layers[layer_idx].extend(recs);
            }
            if !next_local.is_empty() {
                next.extend(next_local);
            }
        }

        current = next;
    }

    GraphBuckets { layers }
}

/// Retrograde result for a state.
#[derive(Debug, Clone, Copy)]
struct RetroEntry {
    value: i8,
    best_move: Option<crate::state::Move>,
}

#[inline]
fn extract_initial_hands(initial: &GameState) -> ([u16; 5], [u16; 5]) {
    let mut a: [u16; 5] = [0; 5];
    let mut b: [u16; 5] = [0; 5];
    for i in 0..5 {
        a[i] = initial.hands_a[i].unwrap_or(0);
        b[i] = initial.hands_b[i].unwrap_or(0);
    }
    (a, b)
}

/// Phase B: exact retrograde solve, assuming full-depth enumeration to ply 9.
/// Returns per-depth hashmaps of (zobrist -> RetroEntry) for depths 0..9.
fn retrograde_solve(
    initial: &GameState,
    cards: &CardsDb,
    buckets: &GraphBuckets,
) -> Vec<HbHashMap<u128, RetroEntry, FastHasher>> {
    let (initial_a, initial_b) = extract_initial_hands(initial);

    // Per-depth value maps for fast child lookup and entries maps for best_move
    let mut entries_by_depth: Vec<HbHashMap<u128, RetroEntry, FastHasher>> =
        (0..10).map(|_| HbHashMap::with_hasher(FastHasher::default())).collect();

    // Depth 9: terminals
    // Build depth-9 map from a parallel Vec, then insert (hashbrown HashMap does not implement FromParallelIterator)
    let computed9: Vec<(u128, RetroEntry)> = buckets.layers[9]
        .par_iter()
        .map(|(key, packed)| {
            let s = packed.to_state(initial_a, initial_b);
            debug_assert!(s.is_terminal(), "Non-terminal at depth 9 encountered");
            let mut v = score(&s);
            if s.next == Owner::B {
                v = -v;
            }
            (*key, RetroEntry { value: v, best_move: None })
        })
        .collect();
    let mut map9: HbHashMap<u128, RetroEntry, FastHasher> = HbHashMap::with_hasher(FastHasher::default());
    map9.reserve(computed9.len());
    for (k, e) in computed9 {
        map9.insert(k, e);
    }
    entries_by_depth[9] = map9;

    // Depths 8..0 retrograde
    for d in (0..=8).rev() {
        let child_map = &entries_by_depth[d + 1];
        let layer = &buckets.layers[d];

        // Compute entries for layer d in parallel, then collect
        let computed: Vec<(u128, RetroEntry)> = layer
            .par_iter()
            .map(|(key, packed)| {
                let s = packed.to_state(initial_a, initial_b);
                let moves = s.legal_moves();
                // There should be legal moves when not terminal; however handle empty defensively
                if moves.is_empty() {
                    // Treat as terminal fallback
                    let mut v = score(&s);
                    if s.next == Owner::B {
                        v = -v;
                    }
                    return (*key, RetroEntry { value: v, best_move: None });
                }

                let mut best_val: Option<i8> = None;
                let mut best_mv: Option<crate::state::Move> = None;

                for mv in moves {
                    // Apply move to get child state and lookup its value at d+1
                    if let Ok(ns) = apply_move(&s, cards, mv) {
                        let ck = zobrist_key(&ns);
                        if let Some(child) = child_map.get(&ck) {
                            let cv = child.value;
                            match (s.next, best_val) {
                                (Owner::A, None) => {
                                    best_val = Some(cv);
                                    best_mv = Some(mv);
                                }
                                (Owner::A, Some(cur)) => {
                                    if cv > cur {
                                        best_val = Some(cv);
                                        best_mv = Some(mv);
                                    }
                                    // tie: keep first encountered to satisfy lexicographic order
                                }
                                (Owner::B, None) => {
                                    best_val = Some(cv);
                                    best_mv = Some(mv);
                                }
                                (Owner::B, Some(cur)) => {
                                    if cv < cur {
                                        best_val = Some(cv);
                                        best_mv = Some(mv);
                                    }
                                    // tie: keep first encountered
                                }
                            }
                        } else {
                            // Child not found: this indicates an enumeration gap; fall back to terminal eval
                            let mut v = score(&ns);
                            if ns.next == Owner::B {
                                v = -v;
                            }
                            match (s.next, best_val) {
                                (Owner::A, None) => {
                                    best_val = Some(v);
                                    best_mv = Some(mv);
                                }
                                (Owner::A, Some(cur)) => {
                                    if v > cur {
                                        best_val = Some(v);
                                        best_mv = Some(mv);
                                    }
                                }
                                (Owner::B, None) => {
                                    best_val = Some(v);
                                    best_mv = Some(mv);
                                }
                                (Owner::B, Some(cur)) => {
                                    if v < cur {
                                        best_val = Some(v);
                                        best_mv = Some(mv);
                                    }
                                }
                            }
                        }
                    }
                }

                let (val, bm) = (best_val.unwrap_or_else(|| {
                    // If no child evaluated (shouldn't happen), fallback to terminal-like eval
                    let mut v = score(&s);
                    if s.next == Owner::B {
                        v = -v;
                    }
                    v
                }), best_mv);

                (*key, RetroEntry { value: val, best_move: bm })
            })
            .collect();

        // Build depth map
        let mut map: HbHashMap<u128, RetroEntry, FastHasher> = HbHashMap::with_hasher(FastHasher::default());
        map.reserve(computed.len());
        for (k, e) in computed {
            map.insert(k, e);
        }
        entries_by_depth[d] = map;
    }

    entries_by_depth
}

/// Phase A + Phase B entrypoint:
/// - Validate full-depth requirement
/// - Enumerate graph to ply 9
/// - Retrograde exact solve 9 -> 0
#[derive(Debug, Serialize)]
struct ExportBoardCell {
    cell: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    card_id: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    owner: Option<char>,
    #[serde(skip_serializing_if = "Option::is_none")]
    element: Option<Option<char>>,
}

#[derive(Debug, Serialize)]
struct ExportHands {
    #[serde(rename = "A")]
    a: Vec<u16>,
    #[serde(rename = "B")]
    b: Vec<u16>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ExportPolicy {
    Move { card_id: u16, cell: u8 },
    Dist(serde_json::Value),
}

#[derive(Debug, Serialize)]
struct ExportLine {
    game_id: usize,
    state_idx: u8,
    board: Vec<ExportBoardCell>,
    hands: ExportHands,
    to_move: char,
    turn: u8,
    rules: Rules,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_target: Option<ExportPolicy>,
    value_target: i8,
    value_mode: String,
    off_pv: bool,
    state_hash: String,
}

#[inline]
fn element_letter_local(e: crate::types::Element) -> char {
    match e {
        crate::types::Element::Fire => 'F',
        crate::types::Element::Ice => 'I',
        crate::types::Element::Thunder => 'T',
        crate::types::Element::Water => 'W',
        crate::types::Element::Earth => 'E',
        crate::types::Element::Poison => 'P',
        crate::types::Element::Holy => 'H',
        crate::types::Element::Wind => 'L',
    }
}

pub fn graph_precompute_export(
    initial: &GameState,
    cards: &CardsDb,
    max_depth: Option<u8>,
    out: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    // Enforce full-depth only for Graph mode
    let start_filled = initial.board.filled_count();
    let rem_full = 9 - start_filled;
    if let Some(cap) = max_depth {
        if cap < rem_full {
            return Err(format!(
                "Graph mode requires full-depth enumeration (remaining plies = {}). Provided --max-depth = {} is insufficient.",
                rem_full, cap
            ).into());
        }
    }

    // Progress: enumeration spinner
    let en_pb = ProgressBar::new_spinner();
    en_pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] enumerate {spinner} {msg}")
            .unwrap()
    );
    en_pb.enable_steady_tick(std::time::Duration::from_millis(120));
    en_pb.set_message(format!("starting (remaining plies = {})", rem_full));
 
    let t_en = std::time::Instant::now();
    let buckets = enumerate_graph(initial, cards, Some(rem_full));
    let en_elapsed = t_en.elapsed();
 
    let mut total = 0usize;
    for d in 0..=9 {
        let n = buckets.layers[d].len();
        total += n;
        eprintln!("[graph] depth {}: {}", d, n);
    }
    en_pb.finish_and_clear();
    eprintln!("[graph] enumeration done: states={} elapsed_ms={}", total, en_elapsed.as_millis());
 
    // Sanity: depth 9 must be terminals
    if buckets.layers[9].is_empty() {
        return Err("Graph enumeration did not reach depth 9; cannot retrograde solve".into());
    }
 
    // Retrograde exact solve (with per-depth progress bars)
    let entries_by_depth = {
        let mp = MultiProgress::new();

        // Depth 9 progress
        {
            let total9 = buckets.layers[9].len() as u64;
            let pb9 = mp.add(ProgressBar::new(total9));
            pb9.set_style(
                ProgressStyle::with_template("[{elapsed_precise}] retro d=9 {bar:40.cyan/blue} {pos}/{len}")
                    .unwrap()
                    .progress_chars("=>-")
            );
            // Build map with a per-item tick via a temporary collection
            let (initial_a, initial_b) = extract_initial_hands(initial);
            let computed9: Vec<(u128, RetroEntry)> = buckets.layers[9]
                .par_iter()
                .map(|(key, packed)| {
                    let s = packed.to_state(initial_a, initial_b);
                    debug_assert!(s.is_terminal(), "Non-terminal at depth 9 encountered");
                    let mut v = score(&s);
                    if s.next == Owner::B {
                        v = -v;
                    }
                    pb9.inc(1);
                    (*key, RetroEntry { value: v, best_move: None })
                })
                .collect();
            pb9.finish_and_clear();

            // Seed entries_by_depth with depth 9 content
            let mut entries_by_depth: Vec<HbHashMap<u128, RetroEntry, FastHasher>> =
                (0..10).map(|_| HbHashMap::with_hasher(FastHasher::default())).collect();
            let mut map9: HbHashMap<u128, RetroEntry, FastHasher> = HbHashMap::with_hasher(FastHasher::default());
            map9.reserve(computed9.len());
            for (k, e) in computed9 {
                map9.insert(k, e);
            }
            entries_by_depth[9] = map9;

            // Depths 8..0 with progress bars
            for d in (0..=8).rev() {
                let layer_len = buckets.layers[d].len() as u64;
                let pbd = mp.add(ProgressBar::new(layer_len));
                pbd.set_style(
                    ProgressStyle::with_template(&format!("[{{elapsed_precise}}] retro d={} {{bar:40.cyan/blue}} {{pos}}/{{len}}", d))
                        .unwrap()
                        .progress_chars("=>-")
                );
                let child_map = &entries_by_depth[d + 1];
                let layer = &buckets.layers[d];
                let (initial_a, initial_b) = extract_initial_hands(initial);

                let computed: Vec<(u128, RetroEntry)> = layer
                    .par_iter()
                    .map(|(key, packed)| {
                        let s = packed.to_state(initial_a, initial_b);
                        let moves = s.legal_moves();
                        if moves.is_empty() {
                            let mut v = score(&s);
                            if s.next == Owner::B {
                                v = -v;
                            }
                            pbd.inc(1);
                            return (*key, RetroEntry { value: v, best_move: None });
                        }
                        let mut best_val: Option<i8> = None;
                        let mut best_mv: Option<crate::state::Move> = None;
                        for mv in moves {
                            if let Ok(ns) = apply_move(&s, cards, mv) {
                                let ck = zobrist_key(&ns);
                                let v = if let Some(child) = child_map.get(&ck) {
                                    child.value
                                } else {
                                    // Fallback: should be rare; compute terminal-like
                                    let mut tv = score(&ns);
                                    if ns.next == Owner::B {
                                        tv = -tv;
                                    }
                                    tv
                                };
                                match (s.next, best_val) {
                                    (Owner::A, None) => { best_val = Some(v); best_mv = Some(mv); }
                                    (Owner::A, Some(cur)) => {
                                        if v > cur { best_val = Some(v); best_mv = Some(mv); }
                                    }
                                    (Owner::B, None) => { best_val = Some(v); best_mv = Some(mv); }
                                    (Owner::B, Some(cur)) => {
                                        if v < cur { best_val = Some(v); best_mv = Some(mv); }
                                    }
                                }
                            }
                        }
                        pbd.inc(1);
                        let val = best_val.unwrap_or_else(|| {
                            let mut v = score(&s);
                            if s.next == Owner::B {
                                v = -v;
                            }
                            v
                        });
                        (*key, RetroEntry { value: val, best_move: best_mv })
                    })
                    .collect();

                let mut map: HbHashMap<u128, RetroEntry, FastHasher> = HbHashMap::with_hasher(FastHasher::default());
                map.reserve(computed.len());
                for (k, e) in computed {
                    map.insert(k, e);
                }
                entries_by_depth[d] = map;
                pbd.finish_and_clear();
            }

            entries_by_depth
        }
    };

    // Root summary
    let root_key = zobrist_key(initial);
    if let Some(root) = entries_by_depth[start_filled as usize].get(&root_key) {
        eprintln!(
            "[graph] root depth={} value={} best_move={:?}",
            9 - start_filled,
            root.value,
            root.best_move.map(|m| (m.card_id, m.cell))
        );
    } else {
        eprintln!("[graph] root not found in retrograde results (unexpected)");
    }

    // JSONL export: iterate layers in increasing depth (0..9) and emit one line per state.
    // Value target is winloss (side-to-move perspective), policy_target is onehot best_move when non-terminal.
    let (initial_a, initial_b) = extract_initial_hands(initial);
    let mut lines_written: usize = 0;

    for d in 0..=9 {
        let layer = &buckets.layers[d];
        let map = &entries_by_depth[d];
        for (key, packed) in layer {
            let state = packed.to_state(initial_a, initial_b);

            // Hands snapshot vectors (compact ascending)
            let mut ha: Vec<u16> = Vec::with_capacity(5);
            let mut hb: Vec<u16> = Vec::with_capacity(5);
            for &o in state.hands_a.iter() {
                if let Some(id) = o {
                    ha.push(id);
                }
            }
            for &o in state.hands_b.iter() {
                if let Some(id) = o {
                    hb.push(id);
                }
            }

            // Board cells
            let mut board_vec: Vec<ExportBoardCell> = Vec::with_capacity(9);
            let elemental_enabled = state.rules.elemental;
            for cell in 0u8..9 {
                let slot = state.board.get(cell);
                let (card_id, owner) = match slot {
                    Some(s) => (Some(s.card_id), Some(match s.owner { Owner::A => 'A', Owner::B => 'B' })),
                    None => (None, None),
                };
                let element_field: Option<Option<char>> = if elemental_enabled {
                    Some(state.board.cell_element(cell).map(element_letter_local))
                } else {
                    None
                };
                board_vec.push(ExportBoardCell { cell, card_id, owner, element: element_field });
            }

            // to_move
            let to_move = match state.next { Owner::A => 'A', Owner::B => 'B' };

            // Lookup retrograde entry for this state
            let entry = match map.get(key) {
                Some(e) => *e,
                None => {
                    // Should not happen; skip or compute fallback
                    let mut v = score(&state);
                    if state.next == Owner::B {
                        v = -v;
                    }
                    RetroEntry { value: v, best_move: None }
                }
            };

            // policy_target: best_move when non-terminal
            let policy_target = if !state.is_terminal() {
                entry.best_move.map(|m| ExportPolicy::Move { card_id: m.card_id, cell: m.cell })
            } else {
                None
            };

            let line = ExportLine {
                game_id: 0,
                state_idx: state.board.filled_count(),
                board: board_vec,
                hands: ExportHands { a: ha, b: hb },
                to_move,
                turn: state.board.filled_count(),
                rules: state.rules,
                policy_target,
                value_target: if entry.value > 0 { 1 } else if entry.value < 0 { -1 } else { 0 },
                value_mode: "winloss".to_string(),
                off_pv: false,
                state_hash: format!("{:032x}", zobrist_key(&state)),
            };

            let s = serde_json::to_string(&line)?;
            out.write_all(s.as_bytes())?;
            out.write_all(b"\n")?;
            lines_written += 1;
        }
    }

    eprintln!("[graph] export lines written: {}", lines_written);

    Ok(())
}

// High-throughput Graph export with pluggable sink (zstd/plain) and manifest-friendly stats
#[derive(Debug, Clone)]
pub struct GraphExportOutcome {
    pub totals_by_depth: [u64; 10],
    pub totals_states: u64,
    pub totals_terminals: u64,
    pub logical_checksum_hex: String,
}

pub fn graph_precompute_export_with_sink(
    initial: &GameState,
    cards: &CardsDb,
    max_depth: Option<u8>,
    sink: &mut dyn crate::solver::graph_writer::GraphJsonlSink,
) -> Result<GraphExportOutcome, Box<dyn Error>> {
    // Enforce full-depth only for Graph mode
    let start_filled = initial.board.filled_count();
    let rem_full = 9 - start_filled;
    if let Some(cap) = max_depth {
        if cap < rem_full {
            return Err(format!(
                "Graph mode requires full-depth enumeration (remaining plies = {}). Provided --max-depth = {} is insufficient.",
                rem_full, cap
            ).into());
        }
    }

    // Progress: enumeration spinner
    let en_pb = ProgressBar::new_spinner();
    en_pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] enumerate {spinner} {msg}")
            .unwrap()
    );
    en_pb.enable_steady_tick(std::time::Duration::from_millis(120));
    en_pb.set_message(format!("starting (remaining plies = {})", rem_full));

    let t_en = std::time::Instant::now();
    let buckets = enumerate_graph(initial, cards, Some(rem_full));
    let en_elapsed = t_en.elapsed();

    let mut totals_by_depth: [u64; 10] = [0; 10];
    let mut totals_states: u64 = 0;
    for d in 0..=9 {
        let n = buckets.layers[d].len() as u64;
        totals_by_depth[d] = n;
        totals_states = totals_states.saturating_add(n);
        eprintln!("[graph] depth {}: {}", d, n);
    }
    en_pb.finish_and_clear();
    eprintln!("[graph] enumeration done: states={} elapsed_ms={}", totals_states, en_elapsed.as_millis());

    // Sanity: depth 9 must be terminals
    if buckets.layers[9].is_empty() {
        return Err("Graph enumeration did not reach depth 9; cannot retrograde solve".into());
    }
    let totals_terminals: u64 = buckets.layers[9].len() as u64;

    // Retrograde exact solve (with per-depth progress bars)
    let entries_by_depth = {
        let mp = MultiProgress::new();

        // Depth 9 progress
        {
            let total9 = buckets.layers[9].len() as u64;
            let pb9 = mp.add(ProgressBar::new(total9));
            pb9.set_style(
                ProgressStyle::with_template("[{elapsed_precise}] retro d=9 {bar:40.cyan/blue} {pos}/{len}")
                    .unwrap()
                    .progress_chars("=>-")
            );
            // Build map with a per-item tick via a temporary collection
            let (initial_a, initial_b) = extract_initial_hands(initial);
            let computed9: Vec<(u128, RetroEntry)> = buckets.layers[9]
                .par_iter()
                .map(|(key, packed)| {
                    let s = packed.to_state(initial_a, initial_b);
                    debug_assert!(s.is_terminal(), "Non-terminal at depth 9 encountered");
                    let mut v = score(&s);
                    if s.next == Owner::B {
                        v = -v;
                    }
                    pb9.inc(1);
                    (*key, RetroEntry { value: v, best_move: None })
                })
                .collect();
            pb9.finish_and_clear();

            // Seed entries_by_depth with depth 9 content
            let mut entries_by_depth: Vec<HbHashMap<u128, RetroEntry, FastHasher>> =
                (0..10).map(|_| HbHashMap::with_hasher(FastHasher::default())).collect();
            let mut map9: HbHashMap<u128, RetroEntry, FastHasher> = HbHashMap::with_hasher(FastHasher::default());
            map9.reserve(computed9.len());
            for (k, e) in computed9 {
                map9.insert(k, e);
            }
            entries_by_depth[9] = map9;

            // Depths 8..0 with progress bars
            for d in (0..=8).rev() {
                let layer_len = buckets.layers[d].len() as u64;
                let pbd = mp.add(ProgressBar::new(layer_len));
                pbd.set_style(
                    ProgressStyle::with_template(&format!("[{{elapsed_precise}}] retro d={} {{bar:40.cyan/blue}} {{pos}}/{{len}}", d))
                        .unwrap()
                        .progress_chars("=>-")
                );
                let child_map = &entries_by_depth[d + 1];
                let layer = &buckets.layers[d];
                let (initial_a, initial_b) = extract_initial_hands(initial);

                let computed: Vec<(u128, RetroEntry)> = layer
                    .par_iter()
                    .map(|(key, packed)| {
                        let s = packed.to_state(initial_a, initial_b);
                        let moves = s.legal_moves();
                        if moves.is_empty() {
                            let mut v = score(&s);
                            if s.next == Owner::B {
                                v = -v;
                            }
                            pbd.inc(1);
                            return (*key, RetroEntry { value: v, best_move: None });
                        }
                        let mut best_val: Option<i8> = None;
                        let mut best_mv: Option<crate::state::Move> = None;
                        for mv in moves {
                            if let Ok(ns) = apply_move(&s, cards, mv) {
                                let ck = zobrist_key(&ns);
                                let v = if let Some(child) = child_map.get(&ck) {
                                    child.value
                                } else {
                                    // Fallback: should be rare; compute terminal-like
                                    let mut tv = score(&ns);
                                    if ns.next == Owner::B {
                                        tv = -tv;
                                    }
                                    tv
                                };
                                match (s.next, best_val) {
                                    (Owner::A, None) => { best_val = Some(v); best_mv = Some(mv); }
                                    (Owner::A, Some(cur)) => {
                                        if v > cur { best_val = Some(v); best_mv = Some(mv); }
                                    }
                                    (Owner::B, None) => { best_val = Some(v); best_mv = Some(mv); }
                                    (Owner::B, Some(cur)) => {
                                        if v < cur { best_val = Some(v); best_mv = Some(mv); }
                                    }
                                }
                            }
                        }
                        pbd.inc(1);
                        let val = best_val.unwrap_or_else(|| {
                            let mut v = score(&s);
                            if s.next == Owner::B {
                                v = -v;
                            }
                            v
                        });
                        (*key, RetroEntry { value: val, best_move: best_mv })
                    })
                    .collect();

                let mut map: HbHashMap<u128, RetroEntry, FastHasher> = HbHashMap::with_hasher(FastHasher::default());
                map.reserve(computed.len());
                for (k, e) in computed {
                    map.insert(k, e);
                }
                entries_by_depth[d] = map;
                pbd.finish_and_clear();
            }

            entries_by_depth
        }
    };

    // Root summary
    let root_key = zobrist_key(initial);
    if let Some(root) = entries_by_depth[start_filled as usize].get(&root_key) {
        eprintln!(
            "[graph] root depth={} value={} best_move={:?}",
            9 - start_filled,
            root.value,
            root.best_move.map(|m| (m.card_id, m.cell))
        );
    } else {
        eprintln!("[graph] root not found in retrograde results (unexpected)");
    }

    // JSONL export through sink: iterate layers in increasing depth (0..9)
    let (initial_a, initial_b) = extract_initial_hands(initial);
    // Writing progress bar
    let write_total = totals_states;
    let write_pb = ProgressBar::new(write_total);
    write_pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] write {bar:40.cyan/blue} {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"),
    );
    let mut lines_written: usize = 0;
    let mut hasher = Sha256::new();

    for d in 0..=9 {
        let layer = &buckets.layers[d];
        let map = &entries_by_depth[d];
        for (key, packed) in layer {
            let state = packed.to_state(initial_a, initial_b);

            // Hands snapshot vectors (compact ascending)
            let mut ha: Vec<u16> = Vec::with_capacity(5);
            let mut hb: Vec<u16> = Vec::with_capacity(5);
            for &o in state.hands_a.iter() {
                if let Some(id) = o {
                    ha.push(id);
                }
            }
            for &o in state.hands_b.iter() {
                if let Some(id) = o {
                    hb.push(id);
                }
            }

            // Board cells
            let mut board_vec: Vec<ExportBoardCell> = Vec::with_capacity(9);
            let elemental_enabled = state.rules.elemental;
            for cell in 0u8..9 {
                let slot = state.board.get(cell);
                let (card_id, owner) = match slot {
                    Some(s) => (Some(s.card_id), Some(match s.owner { Owner::A => 'A', Owner::B => 'B' })),
                    None => (None, None),
                };
                let element_field: Option<Option<char>> = if elemental_enabled {
                    Some(state.board.cell_element(cell).map(element_letter_local))
                } else {
                    None
                };
                board_vec.push(ExportBoardCell { cell, card_id, owner, element: element_field });
            }

            // to_move
            let to_move = match state.next { Owner::A => 'A', Owner::B => 'B' };

            // Lookup retrograde entry for this state
            let entry = match map.get(key) {
                Some(e) => *e,
                None => {
                    // Should not happen; fallback to terminal-like
                    let mut v = score(&state);
                    if state.next == Owner::B {
                        v = -v;
                    }
                    RetroEntry { value: v, best_move: None }
                }
            };

            // policy_target: best_move when non-terminal
            let policy_target = if !state.is_terminal() {
                entry.best_move.map(|m| ExportPolicy::Move { card_id: m.card_id, cell: m.cell })
            } else {
                None
            };

            let vt = if entry.value > 0 { 1 } else if entry.value < 0 { -1 } else { 0 };

            let line = ExportLine {
                game_id: 0,
                state_idx: state.board.filled_count(),
                board: board_vec,
                hands: ExportHands { a: ha, b: hb },
                to_move,
                turn: state.board.filled_count(),
                rules: state.rules,
                policy_target,
                value_target: vt,
                value_mode: "winloss".to_string(),
                off_pv: false,
                state_hash: format!("{:032x}", zobrist_key(&state)),
            };

            // Serialize and write via sink
            let s = serde_json::to_vec(&line)?;
            sink.write_line(&s, state.board.filled_count())?;
            write_pb.inc(1);
            lines_written += 1;

            // Logical checksum update: (state_hash|value|best_move or -)
            let bm_str = match entry.best_move {
                Some(m) => format!("{}-{}", m.card_id, m.cell),
                None => "-".to_string(),
            };
            let tuple = format!("{}|{}|{}\n", line.state_hash, vt, bm_str);
            hasher.update(tuple.as_bytes());
        }
    }
 
    write_pb.finish_and_clear();
    eprintln!("[graph] export lines written: {}", lines_written);
 
    let logical_checksum_hex = hex::encode(hasher.finalize());

    Ok(GraphExportOutcome {
        totals_by_depth,
        totals_states,
        totals_terminals,
        logical_checksum_hex,
    })
}