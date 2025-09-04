use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::sync::atomic::{Ordering, AtomicU64};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use serde::{Serialize, Deserialize};
use serde_json;
use std::io::{self, BufRead};
use std::process;

use triplecargo::{
    load_cards_from_json, zobrist_key, Element, GameState, Owner, Rules, apply_move, legal_moves, score, rng_for_state,
    solver::{search_root, search_root_with_children},
};
use triplecargo::solver::tt_array::FixedTT;

#[derive(Debug, Clone, ValueEnum)]
enum ElementsOpt {
    None,
    Random,
}

#[derive(Debug, Clone, ValueEnum)]
enum HandStrategyOpt {
    Random,
    Stratified,
}

#[derive(Debug, Clone, ValueEnum)]
enum ExportModeOpt {
    Trajectory,
    Full,
    Graph,
}

#[derive(Debug, Clone, ValueEnum)]
enum PolicyFormatOpt {
    Onehot,
    Mcts,
}

#[derive(Debug, Clone, ValueEnum)]
enum ValueModeOpt {
    Winloss,
    Margin,
}

#[derive(Debug, Clone, ValueEnum)]
enum SyncModeOpt {
    #[clap(name = "none")]
    None,
    #[clap(name = "final")]
    Final,
}

#[derive(Debug, Clone, ValueEnum)]
enum OffPvStrategyOpt {
    Random,
    Weighted,
    Mcts,
}

#[derive(Debug, Clone, ValueEnum)]
enum PlayStrategyOpt {
    Pv,
    Mcts,
    Heuristic,
    Mix,
}

#[derive(Debug, Parser)]
#[command(name = "precompute_new", about = "Triplecargo export driver (parallel skeleton)")]
struct Args {
    /// Rules toggles as comma-separated list: elemental,same,plus,same_wall (or 'none')
    #[arg(long, default_value = "none")]
    rules: String,

    /// Elements mode: none | random
    #[arg(long, value_enum, default_value_t = ElementsOpt::None)]
    elements: ElementsOpt,

    /// Seed used for deterministic RNG (elements + sampling)
    #[arg(long, default_value_t = 0x00C0FFEEu64)]
    seed: u64,

    /// Optional cap on search depth (plies remaining), 1..=9; omit for full depth.
    #[arg(long)]
    max_depth: Option<u8>,

    /// Cards JSON path (defaults to data/cards.json)
    #[arg(long, default_value = "data/cards.json")]
    cards: String,

    /// Export JSONL file path (one object per line).
    #[arg(long)]
    export: Option<PathBuf>,

    /// Export mode: trajectory | full (default: trajectory)
    #[arg(long, value_enum, default_value_t = ExportModeOpt::Trajectory)]
    export_mode: ExportModeOpt,

    /// Number of games to sample when --export is used (trajectory only).
    #[arg(long, default_value_t = 1000)]
    games: usize,

    /// Number of worker threads for export (default: available_parallelism-1, min 1)
    #[arg(long)]
    threads: Option<usize>,
    
    /// Chunk size per worker request (number of games fetched at once). If omitted, defaults to min(32, max(1, games/threads)).
    #[arg(long = "chunk-size")]
    chunk_size: Option<usize>,
    
    /// Hand sampling strategy: random | stratified
    #[arg(long, value_enum, default_value_t = HandStrategyOpt::Random)]
    hand_strategy: HandStrategyOpt,

    /// Policy export format: onehot | mcts
    #[arg(long, value_enum, default_value_t = PolicyFormatOpt::Onehot)]
    policy_format: PolicyFormatOpt,

    /// Rollouts for MCTS policy (only used when --policy-format mcts)
    #[arg(long, default_value_t = 100)]
    mcts_rollouts: usize,

    /// Off-PV sampling rate [0.0..1.0] (0 disables)
    #[arg(long, default_value_t = 0.0)]
    off_pv_rate: f32,

    /// Off-PV sampling strategy: random | weighted | mcts
    #[arg(long, value_enum, default_value_t = OffPvStrategyOpt::Weighted)]
    off_pv_strategy: OffPvStrategyOpt,

    /// Play strategy for advancing the trajectory: pv | mcts | heuristic | mix
    #[arg(long, value_enum, default_value_t = PlayStrategyOpt::Mix)]
    play_strategy: PlayStrategyOpt,

    /// Progressive mix: heuristic weight at early game (turn=0)
    #[arg(long, default_value_t = 0.65)]
    mix_heuristic_early: f32,

    /// Progressive mix: heuristic weight at late game (turn=8)
    #[arg(long, default_value_t = 0.10)]
    mix_heuristic_late: f32,

    /// Progressive mix: MCTS fixed weight (only when --policy-format mcts)
    #[arg(long, default_value_t = 0.25)]
    mix_mcts: f32,

    /// Heuristic weights
    #[arg(long = "heur-w-corner", default_value_t = 1.0)]
    heur_w_corner: f32,
    #[arg(long = "heur-w-edge", default_value_t = 0.3)]
    heur_w_edge: f32,
    #[arg(long = "heur-w-center", default_value_t = -0.2)]
    heur_w_center: f32,
    #[arg(long = "heur-w-greedy", default_value_t = 0.8)]
    heur_w_greedy: f32,
    #[arg(long = "heur-w-defense", default_value_t = 0.6)]
    heur_w_defense: f32,
    #[arg(long = "heur-w-element", default_value_t = 0.6)]
    heur_w_element: f32,

    /// Value target mode: winloss | margin
    #[arg(
        long = "value-mode",
        short = 'm',
        value_enum,
        default_value_t = ValueModeOpt::Winloss,
        value_name = "MODE",
        visible_aliases = ["value_mode", "valuemode", "vm"]
    )]
    value_mode: ValueModeOpt,

    /// Transposition table size per worker in MiB (rounded down to a power-of-two capacity under the budget)
    #[arg(long = "tt-bytes", default_value_t = 32, value_name = "MiB")]
    tt_bytes: usize,

    /// Evaluate a single state from stdin; emits exactly one JSON object to stdout
    #[arg(long = "eval-state")]
    eval_state: bool,

    /// Verbose diagnostics to stderr (eval mode only)
    #[arg(long)]
    verbose: bool,

    // Graph mode input flags (lightweight)
    /// When true, Graph mode uses explicit hands/elements flags instead of sampling
    #[arg(long = "graph-input", default_value_t = false)]
    graph_input: bool,

    /// Graph mode: initial hand for A as comma-separated ids, e.g. "12,34,56,78,90" (required when --graph-input)
    #[arg(long = "graph-hand-a")]
    graph_hand_a: Option<String>,

    /// Graph mode: initial hand for B as comma-separated ids, e.g. "22,33,44,55,66" (required when --graph-input)
    #[arg(long = "graph-hand-b")]
    graph_hand_b: Option<String>,

    /// Graph mode: per-cell elements as 9 comma-separated entries using letters F,I,T,W,E,P,H,L or '-' for none; no duplicates allowed among elements
    #[arg(long = "graph-elements")]
    graph_elements: Option<String>,

    // ---- Graph mode compression and indexing flags ----

    /// Enable zstd compression of node streams (default: true)
    #[arg(long = "zstd", default_value_t = true)]
    zstd: bool,

    /// Alias for disabling compression (equivalent to --zstd=false)
    #[arg(long = "no-compress", default_value_t = false, visible_alias = "nocompress")]
    no_compress: bool,

    /// Zstd compression level (1..=10, default 3)
    #[arg(long = "zstd-level", default_value_t = 3)]
    zstd_level: i32,

    /// Zstd worker threads (default: min(4, available_parallelism()))
    #[arg(long = "zstd-threads")]
    zstd_threads: Option<usize>,

    /// Target number of JSONL records per zstd frame (default: 131072)
    #[arg(long = "zstd-frame-lines", default_value_t = 131072)]
    zstd_frame_lines: usize,

    /// Target maximum uncompressed bytes per zstd frame (soft cap; default: 128 MiB)
    #[arg(long = "zstd-frame-bytes", default_value_t = 134217728)]
    zstd_frame_bytes: usize,
 
    /// Emit nodes.idx.jsonl (one line per frame) when zstd is enabled (default: true)
    #[arg(long = "zstd-index", default_value_t = true)]
    zstd_index: bool,

    /// Sync policy for graph export files: none | final (default: final)
    #[arg(long = "sync-mode", value_enum, default_value_t = SyncModeOpt::Final)]
    sync_mode: SyncModeOpt,

    /// Max in-flight frames in writer queue (default: 8)
    #[arg(long = "writer-queue-frames", default_value_t = 8)]
    writer_queue_frames: usize,

    /// Async compression workers for stage-1 (0 = compress in writer thread; default: 2)
    #[arg(long = "zstd-workers", default_value_t = 2)]
    zstd_workers: usize,

    /// Bounded queue size for compressed frames handed to the writer (default: 4)
    #[arg(long = "writer-queue-compressed", default_value_t = 4)]
    writer_queue_compressed: usize,

    /// Number of shard output files (1 = no sharding). When >1 nodes files will be nodes_000.jsonl.zst ... nodes_{N-1}.jsonl.zst
    #[arg(long = "shards", default_value_t = 1)]
    shards: usize,
}

fn parse_rules(s: &str) -> Rules {
    let mut r = Rules::default();
    let s = s.trim();
    if s.eq_ignore_ascii_case("none") || s.is_empty() {
        return r;
    }
    for tok in s.split(',') {
        match tok.trim().to_ascii_lowercase().as_str() {
            "elemental" => r.elemental = true,
            "same" => r.same = true,
            "plus" => r.plus = true,
            "same_wall" | "samewall" => r.same_wall = true,
            "" => {}
            other => eprintln!("[precompute_new] Warning: ignoring unknown rule token '{other}'"),
        }
    }
    r
}

// Minimal SplitMix64 for deterministic local RNG (no rand dependency)
#[inline]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gen_elements(seed: u64) -> [Option<Element>; 9] {
    let mut s = seed;
    let to_elem = |k: u8| -> Option<Element> {
        match k {
            0 => None,
            1 => Some(Element::Earth),
            2 => Some(Element::Fire),
            3 => Some(Element::Water),
            4 => Some(Element::Poison),
            5 => Some(Element::Holy),
            6 => Some(Element::Thunder),
            7 => Some(Element::Wind),
            _ => Some(Element::Ice), // 8
        }
    };
    let mut arr: [Option<Element>; 9] = [None; 9];
    for i in 0..9 {
        let r = splitmix64(&mut s);
        let k = (r % 9) as u8; // uniform over 0..=8: None + 8 elements
        arr[i] = to_elem(k);
    }
    arr
}

#[derive(Debug, Serialize)]
struct BoardCell {
    cell: u8,
    card_id: Option<u16>,
    owner: Option<char>, // 'A' | 'B'
    #[serde(skip_serializing_if = "Option::is_none")]
    element: Option<Option<char>>,
}

#[derive(Debug, Serialize)]
struct HandsRecord {
    #[serde(rename = "A")]
    a: Vec<u16>,
    #[serde(rename = "B")]
    b: Vec<u16>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum PolicyOut {
    Move { card_id: u16, cell: u8 },
    Dist(BTreeMap<String, f32>),
}

#[derive(Serialize)]
struct ExportRecord {
    game_id: usize,
    state_idx: u8,
    board: Vec<BoardCell>,
    hands: HandsRecord,
    to_move: char,
    turn: u8,
    rules: Rules,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_target: Option<PolicyOut>,
    value_target: i8,
    value_mode: String,
    off_pv: bool,
    state_hash: String,
}

#[derive(Debug, Clone)]
struct ExportConfig {
    policy_format: PolicyFormatOpt,
    value_mode: ValueModeOpt,
    mcts_rollouts: usize,
    off_pv_rate: f32,
    off_pv_strategy: OffPvStrategyOpt,
    base_seed: u64,
}

// Helpers for sampling
fn all_card_ids_sorted(cards: &triplecargo::CardsDb) -> Vec<u16> {
    let mut ids: Vec<u16> = cards.iter().map(|c| c.id).collect();
    ids.sort_unstable();
    ids
}

fn build_level_bands(cards: &triplecargo::CardsDb) -> [Vec<u16>; 5] {
    let mut bands: [Vec<u16>; 5] = Default::default();
    for c in cards.iter() {
        let band = match c.level {
            1 | 2 => 0,
            3 | 4 => 1,
            5 | 6 => 2,
            7 | 8 => 3,
            _ => 4, // 9 | 10
        };
        bands[band].push(c.id);
    }
    for b in bands.iter_mut() {
        b.sort_unstable();
    }
    bands
}

fn sample_hand_random(ids_pool: &mut Vec<u16>, rng: &mut u64) -> [u16; 5] {
    let mut hand = [0u16; 5];
    for i in 0..5 {
        let r = splitmix64(rng);
        let idx = (r as usize) % ids_pool.len();
        hand[i] = ids_pool.swap_remove(idx);
    }
    hand
}

fn sample_hand_stratified(bands: &mut [Vec<u16>; 5], rng: &mut u64) -> [u16; 5] {
    let mut hand = [0u16; 5];
    for (i, band) in bands.iter_mut().enumerate() {
        let r = splitmix64(rng);
        let idx = (r as usize) % band.len();
        hand[i] = band.swap_remove(idx);
    }
    hand
}

// Mapping required by spec: F, I, T, W, E, P, H, L
#[inline]
fn element_letter(e: Element) -> char {
    match e {
        Element::Fire => 'F',
        Element::Ice => 'I',
        Element::Thunder => 'T',
        Element::Water => 'W',
        Element::Earth => 'E',
        Element::Poison => 'P',
        Element::Holy => 'H',
        Element::Wind => 'L',
    }
}

#[inline]
fn value_mode_str(vm: &ValueModeOpt) -> &'static str {
    match vm {
        ValueModeOpt::Winloss => "winloss",
        ValueModeOpt::Margin => "margin",
    }
}

#[inline]
fn compute_value_target(vm: &ValueModeOpt, side_value: i8, to_move: Owner) -> i8 {
    match vm {
        ValueModeOpt::Winloss => {
            if side_value > 0 { 1 } else if side_value < 0 { -1 } else { 0 }
        }
        ValueModeOpt::Margin => {
            match to_move {
                Owner::A => side_value,
                Owner::B => -side_value,
            }
        }
    }
}

fn root_perspective_value(root_next: Owner, terminal: &GameState) -> i8 {
    let mut v = score(terminal);
    if root_next == Owner::B {
        v = -v;
    }
    v
}

/// Deterministic shallow MCTS at root to produce a soft policy distribution.
fn mcts_policy_distribution(
    state: &GameState,
    cards: &triplecargo::CardsDb,
    rollouts: usize,
    mut seed: u64,
) -> BTreeMap<String, f32> {
    let moves = legal_moves(state);
    let n = moves.len();
    let mut dist: BTreeMap<String, f32> = BTreeMap::new();
    if n == 0 || rollouts == 0 {
        return dist;
    }

    let root_next = state.next;

    let mut counts: Vec<usize> = vec![0; n];
    let mut sums: Vec<f64> = vec![0.0; n];

    let total_rollouts = rollouts.max(n);

    for t in 0..total_rollouts {
        let mut idx: usize = 0;
        if t < n {
            idx = t;
        } else {
            let total: f64 = (t as f64).max(1.0);
            let ln_total = total.ln();
            let c = 1.41421356237_f64;
            let mut best = f64::NEG_INFINITY;
            for i in 0..n {
                let ci = counts[i].max(1) as f64;
                let mean = if counts[i] == 0 { 0.0 } else { sums[i] / (counts[i] as f64) };
                let ucb = mean + c * (ln_total / ci).sqrt();
                if ucb > best {
                    best = ucb;
                    idx = i;
                }
            }
        }

        // Simulate from chosen child using random playouts
        let child = match apply_move(state, cards, moves[idx]) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut sim = child.clone();

        while !sim.is_terminal() {
            let legals = legal_moves(&sim);
            if legals.is_empty() {
                break;
            }
            let r = splitmix64(&mut seed);
            let pick = (r as usize) % legals.len();
            let mv = legals[pick];
            match apply_move(&sim, cards, mv) {
                Ok(ns) => { sim = ns; }
                Err(_) => break,
            }
        }

        let v = root_perspective_value(root_next, &sim) as f64;
        counts[idx] += 1;
        sums[idx] += v;
    }

    let total_visits: usize = counts.iter().sum();
    if total_visits == 0 {
        let p = 1.0 / (n as f32);
        for mv in moves {
            dist.insert(format!("{}-{}", mv.card_id, mv.cell), p);
        }
        return dist;
    }

    for (i, mv) in moves.iter().enumerate() {
        let p = (counts[i] as f32) / (total_visits as f32);
        dist.insert(format!("{}-{}", mv.card_id, mv.cell), p);
    }
    dist
}

fn pick_random_non_pv_move(
    state: &GameState,
    pv: Option<triplecargo::Move>,
    seed: u64,
    game_id: u64,
    turn: u8,
) -> Option<triplecargo::Move> {
    let mut candidates = legal_moves(state);
    if let Some(pv_mv) = pv {
        candidates.retain(|m| !(m.card_id == pv_mv.card_id && m.cell == pv_mv.cell));
    }
    if candidates.is_empty() {
        return pv;
    }
    let mut rng = rng_for_state(seed, game_id, turn);
    let idx = rng.gen_range(0..candidates.len());
    Some(candidates[idx])
}

fn pick_weighted_non_pv_move(
    child_vals: &[(triplecargo::Move, i8)],
    pv: Option<triplecargo::Move>,
    seed: u64,
    game_id: u64,
    turn: u8,
) -> Option<triplecargo::Move> {
    let mut cands: Vec<(triplecargo::Move, i8)> = Vec::new();
    match pv {
        Some(pv_mv) => {
            for &(m, q) in child_vals {
                if !(m.card_id == pv_mv.card_id && m.cell == pv_mv.cell) {
                    cands.push((m, q));
                }
            }
        }
        None => {
            cands.extend_from_slice(child_vals);
        }
    }
    if cands.is_empty() {
        return pv;
    }
    let max_q = cands.iter().map(|(_, q)| *q as f64).fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = Vec::with_capacity(cands.len());
    let mut sum_w: f64 = 0.0;
    for &(_, q) in &cands {
        let w = (q as f64 - max_q).exp();
        weights.push(w);
        sum_w += w;
    }
    if !(sum_w.is_finite()) || sum_w <= 0.0 {
        let mut rng = rng_for_state(seed, game_id, turn);
        let idx = rng.gen_range(0..cands.len());
        return Some(cands[idx].0);
    }
    let mut rng = rng_for_state(seed, game_id, turn);
    let r: f64 = rng.gen::<f64>();
    let mut acc: f64 = 0.0;
    for (i, &(mv, _)) in cands.iter().enumerate() {
        acc += weights[i] / sum_w;
        if r < acc {
            return Some(mv);
        }
    }
    Some(cands.last().unwrap().0)
}

fn parse_move_key(key: &str) -> Option<triplecargo::Move> {
    let mut parts = key.splitn(2, '-');
    let card_s = parts.next()?;
    let cell_s = parts.next()?;
    let card_id: u16 = card_s.parse().ok()?;
    let cell_u: u8 = cell_s.parse().ok()?;
    Some(triplecargo::Move { card_id, cell: cell_u })
}

#[derive(Clone, Copy)]
struct HeurWeights {
    corner: f32,
    edge: f32,
    center: f32,
    greedy: f32,
    defense: f32,
    element: f32,
}

#[inline]
fn cell_category_local(cell: u8) -> u8 {
    match cell {
        0 | 2 | 6 | 8 => 0, // corners
        4 => 2,             // center
        _ => 1,             // edges
    }
}

#[inline]
fn elemental_delta_local(cell_elem: Option<Element>, card_elem: Option<Element>) -> i16 {
    match cell_elem {
        None => 0,
        Some(e) => {
            if card_elem == Some(e) { 1 } else { -1 }
        }
    }
}

#[inline]
fn clamp_side_local(v: i16) -> u8 {
    v.clamp(1, 10) as u8
}

#[inline]
fn adjusted_sides_for_cell_local(
    card: &triplecargo::Card,
    cell_idx: u8,
    board: &triplecargo::Board,
    rules: &Rules,
) -> [u8; 4] {
    if !rules.elemental {
        return [card.top, card.right, card.bottom, card.left];
    }
    let delta = elemental_delta_local(board.cell_element(cell_idx), card.element) as i16;
    if delta == 0 {
        return [card.top, card.right, card.bottom, card.left];
    }
    let sides = [card.top, card.right, card.bottom, card.left];
    [
        clamp_side_local(sides[0] as i16 + delta),
        clamp_side_local(sides[1] as i16 + delta),
        clamp_side_local(sides[2] as i16 + delta),
        clamp_side_local(sides[3] as i16 + delta),
    ]
}

fn heuristic_scores_for_legal(
    state: &GameState,
    cards: &triplecargo::CardsDb,
    weights: &HeurWeights,
) -> (Vec<triplecargo::Move>, Vec<f32>) {
    let moves = legal_moves(state);
    if moves.is_empty() {
        return (moves, Vec::new());
    }

    let before_score = score(state);
    let to_move = state.next;

    let mut scores: Vec<f32> = Vec::with_capacity(moves.len());

    for mv in &moves {
        // Positional category
        let cat = cell_category_local(mv.cell);
        let pos_bonus = match cat {
            0 => weights.corner,
            1 => weights.edge,
            _ => weights.center,
        };

        // Element synergy (only when elemental rules)
        let elem_bonus = if state.rules.elemental {
            match state.board.cell_element(mv.cell) {
                Some(e) => {
                    // match => +w_element, mismatch or no-element card => -w_element
                    match cards.get(mv.card_id) {
                        Some(c) => {
                            if c.element == Some(e) { weights.element } else { -weights.element }
                        }
                        None => 0.0,
                    }
                }
                None => 0.0,
            }
        } else {
            0.0
        };

        // Greedy gain: simulate move and compute delta margin from mover's perspective
        let greedy_gain = match apply_move(state, cards, *mv) {
            Ok(sim) => {
                let after_score = score(&sim);
                let delta = (after_score as i16 - before_score as i16) as i8;
                if to_move == Owner::A { delta as f32 } else { -(delta as f32) }
            }
            Err(_) => 0.0,
        };

        // Defensive exposure: penalize low exposed sides and immediate threats from existing neighbors
        let defense_penalty = match cards.get(mv.card_id) {
            Some(card) => {
                let placed_sides = adjusted_sides_for_cell_local(card, mv.cell, &state.board, &state.rules);
                let neighs = state.board.neighbors(mv.cell);
                let mut pen: f32 = 0.0;
                for (i, opt_nidx) in neighs.iter().enumerate() {
                    let our_side = placed_sides[i] as i16;
                    match opt_nidx {
                        Some(nidx) => {
                            if let Some(nslot) = state.board.get(*nidx) {
                                // If neighbor occupied by opponent, see if its touching side beats ours
                                if nslot.owner != to_move {
                                    if let Some(nc) = cards.get(nslot.card_id) {
                                        let nsides = adjusted_sides_for_cell_local(nc, *nidx, &state.board, &state.rules);
                                        let opp_idx = (i + 2) % 4;
                                        let neigh_side = nsides[opp_idx] as i16;
                                        if neigh_side > our_side {
                                            pen += ((neigh_side - our_side) as f32) / 10.0;
                                        }
                                    }
                                }
                            } else {
                                // Empty neighbor: general exposure penalty inversely proportional to our side
                                pen += ((10 - (our_side as i16).clamp(1, 10)) as f32) / 10.0;
                            }
                        }
                        None => {
                            // Wall: no exposure
                        }
                    }
                }
                pen
            }
            None => 0.0,
        };

        let score_total =
            pos_bonus +
            weights.greedy * greedy_gain +
            (-weights.defense) * defense_penalty +
            elem_bonus;

        scores.push(score_total);
    }

    (moves, scores)
}

fn softmax_probs(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mut maxv = f32::NEG_INFINITY;
    for &s in scores {
        if s.is_finite() && s > maxv {
            maxv = s;
        }
    }
    if !maxv.is_finite() {
        // fallback to uniform
        let n = scores.len();
        return vec![1.0 / (n as f32); n];
    }
    let mut expv: Vec<f32> = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for &s in scores {
        let e = (s - maxv).exp();
        expv.push(e);
        sum += e;
    }
    if sum <= 0.0 || !sum.is_finite() {
        let n = scores.len();
        return vec![1.0 / (n as f32); n];
    }
    expv.into_iter().map(|e| e / sum).collect()
}

fn map_mcts_to_probs(moves: &[triplecargo::Move], dist: &BTreeMap<String, f32>) -> Vec<f32> {
    if moves.is_empty() {
        return Vec::new();
    }
    let mut probs: Vec<f32> = Vec::with_capacity(moves.len());
    let mut sum = 0.0f32;
    for mv in moves {
        let key = format!("{}-{}", mv.card_id, mv.cell);
        let p = *dist.get(&key).unwrap_or(&0.0);
        probs.push(p.max(0.0));
        sum += p.max(0.0);
    }
    if sum > 0.0 && sum.is_finite() {
        probs.iter().map(|p| *p / sum).collect()
    } else {
        // fallback to uniform
        vec![1.0 / (moves.len() as f32); moves.len()]
    }
}

fn pv_probs(moves: &[triplecargo::Move], bm: Option<triplecargo::Move>) -> Vec<f32> {
    if moves.is_empty() {
        return Vec::new();
    }
    if let Some(b) = bm {
        if let Some(pos) = moves.iter().position(|m| *m == b) {
            let mut v = vec![0.0f32; moves.len()];
            v[pos] = 1.0;
            return v;
        }
    }
    // fallback: uniform
    vec![1.0 / (moves.len() as f32); moves.len()]
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn mix_probs(
    pv: &[f32],
    mcts: Option<&[f32]>,
    heur: &[f32],
    turn: u8,
    mix_heuristic_early: f32,
    mix_heuristic_late: f32,
    mix_mcts: f32,
    mcts_available: bool,
) -> Vec<f32> {
    let n = pv.len();
    let mut out = vec![0.0f32; n];
    if n == 0 {
        return out;
    }
    let frac = (turn as f32 / 8.0).clamp(0.0, 1.0);
    let mut w_heur = lerp(mix_heuristic_early, mix_heuristic_late, frac).clamp(0.0, 1.0);
    let mut w_mcts = if mcts_available { mix_mcts } else { 0.0 };
    w_mcts = w_mcts.clamp(0.0, 1.0);
    let mut w_pv = 1.0 - w_heur - w_mcts;
    if w_pv < 0.0 {
        // Renormalize proportionally
        let sum_pos = (w_heur.max(0.0)) + (w_mcts.max(0.0));
        if sum_pos > 0.0 {
            w_heur = w_heur.max(0.0) / sum_pos;
            w_mcts = w_mcts.max(0.0) / sum_pos;
            w_pv = 0.0;
        } else {
            // all zero or negative, default to PV
            w_heur = 0.0;
            w_mcts = 0.0;
            w_pv = 1.0;
        }
    }

    for i in 0..n {
        let mut p = w_pv * pv[i] + w_heur * heur[i];
        if let Some(m) = mcts {
            p += w_mcts * m[i];
        }
        out[i] = p.max(0.0);
    }
    // Normalize
    let mut s = 0.0f32;
    for &p in &out {
        s += p;
    }
    if s > 0.0 && s.is_finite() {
        for p in &mut out {
            *p /= s;
        }
    } else {
        // fallback to PV
        return pv.to_vec();
    }
    out
}

fn sample_from_probs(
    moves: &[triplecargo::Move],
    probs: &[f32],
    seed: u64,
    game_id: u64,
    turn: u8,
) -> Option<triplecargo::Move> {
    if moves.is_empty() || probs.is_empty() || moves.len() != probs.len() {
        return None;
    }
    let mut rng = rng_for_state(seed, game_id, turn);
    let r: f64 = rng.gen::<f64>();
    let mut acc: f64 = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p as f64;
        if r < acc {
            return Some(moves[i]);
        }
    }
    Some(moves[moves.len() - 1])
}

fn choose_next_move_heuristic(
    state: &GameState,
    cards: &triplecargo::CardsDb,
    weights: &HeurWeights,
    seed: u64,
    game_id: u64,
    turn: u8,
) -> Option<triplecargo::Move> {
    let (moves, scores) = heuristic_scores_for_legal(state, cards, weights);
    if moves.is_empty() {
        return None;
    }
    let probs = softmax_probs(&scores);
    sample_from_probs(&moves, &probs, seed, game_id, turn)
}

fn choose_next_move_mcts(
    state: &GameState,
    mcts_dist: &BTreeMap<String, f32>,
    seed: u64,
    game_id: u64,
    turn: u8,
) -> Option<triplecargo::Move> {
    let moves = legal_moves(state);
    if moves.is_empty() {
        return None;
    }
    let probs = map_mcts_to_probs(&moves, mcts_dist);
    sample_from_probs(&moves, &probs, seed, game_id, turn)
}

fn choose_next_move_mixed(
    state: &GameState,
    cards: &triplecargo::CardsDb,
    pv: Option<triplecargo::Move>,
    mcts_dist: Option<&BTreeMap<String, f32>>,
    seed: u64,
    game_id: u64,
    turn: u8,
    mix_heuristic_early: f32,
    mix_heuristic_late: f32,
    mix_mcts: f32,
    weights: &HeurWeights,
    mcts_available: bool,
) -> Option<triplecargo::Move> {
    let moves = legal_moves(state);
    if moves.is_empty() {
        return None;
    }

    // PV distribution
    let pv_p = pv_probs(&moves, pv);

    // Heuristic distribution
    let (_m2, scores) = heuristic_scores_for_legal(state, cards, weights);
    let heur_p = softmax_probs(&scores);

    // Optional MCTS distribution
    let mcts_p_vec = mcts_dist.map(|d| map_mcts_to_probs(&moves, d));

    // Mix
    let mixed = mix_probs(
        &pv_p,
        mcts_p_vec.as_deref(),
        &heur_p,
        turn,
        mix_heuristic_early,
        mix_heuristic_late,
        mix_mcts,
        mcts_available && mcts_p_vec.is_some(),
    );

    sample_from_probs(&moves, &mixed, seed, game_id, turn)
}
fn pick_mcts_non_pv_move_from_dist(
    dist: &BTreeMap<String, f32>,
    pv: Option<triplecargo::Move>,
    seed: u64,
    game_id: u64,
    turn: u8,
) -> Option<triplecargo::Move> {
    let mut cands: Vec<(triplecargo::Move, f32)> = Vec::new();
    for (k, &p) in dist.iter() {
        if let Some(mv) = parse_move_key(k) {
            if let Some(pv_mv) = pv {
                if mv.card_id == pv_mv.card_id && mv.cell == pv_mv.cell {
                    continue;
                }
            }
            cands.push((mv, p.max(0.0)));
        }
    }
    if cands.is_empty() {
        return pv;
    }
    let sum_p: f64 = cands.iter().map(|(_, p)| *p as f64).sum();
    if sum_p > 0.0 && sum_p.is_finite() {
        let mut rng = rng_for_state(seed, game_id, turn);
        let r: f64 = rng.gen::<f64>();
        let mut acc: f64 = 0.0;
        for (mv, p) in &cands {
            acc += (*p as f64) / sum_p;
            if r < acc {
                return Some(*mv);
            }
        }
        return Some(cands.last().unwrap().0);
    }
    // Fallback: PV had all mass -> pick highest-prob non-PV deterministically (by value then key order)
    let mut best: Option<(triplecargo::Move, f32)> = None;
    for &(mv, p) in &cands {
        best = match best {
            None => Some((mv, p)),
            Some((bmv, bp)) => {
                if p > bp { Some((mv, p)) } else { Some((bmv, bp)) }
            }
        };
    }
    Some(best.unwrap().0)
}

fn format_hms(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

// ---- Eval-state mode: input schema, helpers, and driver ----

#[derive(Debug, Deserialize)]
struct EvalInputHands {
    #[serde(rename = "A")]
    a: Vec<u16>,
    #[serde(rename = "B")]
    b: Vec<u16>,
}

#[derive(Debug, Deserialize)]
struct EvalInputCell {
    cell: u8,
    #[serde(default)]
    card_id: Option<u16>,
    #[serde(default)]
    owner: Option<char>, // 'A' | 'B'
    // When rules.elemental=true, export lines may include element per cell; accept here too.
    #[serde(default)]
    element: Option<char>,
}

#[derive(Debug, Deserialize)]
struct ApplyReq {
    card_id: u16,
    cell: u8,
}

#[derive(Debug, Deserialize)]
struct EvalInput {
    board: Vec<EvalInputCell>,
    hands: EvalInputHands,
    to_move: char,                   // 'A' | 'B'
    #[serde(default)]
    turn: Option<u8>,                // optional; if present must match board filled count
    rules: Rules,
    // Optional; if present, must match board[].element
    #[serde(default)]
    board_elements: Option<Vec<Option<char>>>,
    // Optional apply step; when present, perform a single move and return the new state
    #[serde(default)]
    apply: Option<ApplyReq>,
}

#[derive(Debug, Serialize)]
struct EvalMoveOut {
    card_id: u16,
    cell: u8,
}

#[derive(Debug, Serialize)]
struct EvalOut {
    #[serde(skip_serializing_if = "Option::is_none")]
    best_move: Option<EvalMoveOut>,
    value: i8,           // {-1,0,1} from side-to-move perspective
    margin: i8,          // A_cards − B_cards at terminal
    pv: Vec<EvalMoveOut>,
    nodes: u64,
    depth: u8,           // remaining plies searched
    state_hash: String,  // 128-bit zobrist hex
}

#[derive(Debug, Serialize)]
struct ApplyStateOut {
    board: Vec<BoardCell>,
    hands: HandsRecord,
    to_move: char,
    turn: u8,
    rules: Rules,
}

#[derive(Debug, Serialize)]
struct OutcomeOut {
    mode: String,            // "winloss"
    value: i8,               // {-1,0,1} from A perspective
    #[serde(skip_serializing_if = "Option::is_none")]
    winner: Option<char>,    // 'A' | 'B' | null for draw / non-terminal
}

#[derive(Debug, Serialize)]
struct ApplyOut {
    state: ApplyStateOut,
    done: bool,
    outcome: OutcomeOut,
    state_hash: String,
}

#[derive(Debug, Serialize)]
struct ErrorOut {
    error: String,
}

#[inline]
fn elem_from_letter(ch: char) -> Result<Element, String> {
    match ch {
        'F' => Ok(Element::Fire),
        'I' => Ok(Element::Ice),
        'T' => Ok(Element::Thunder),
        'W' => Ok(Element::Water),
        'E' => Ok(Element::Earth),
        'P' => Ok(Element::Poison),
        'H' => Ok(Element::Holy),
        'L' => Ok(Element::Wind),
        other => Err(format!("Invalid element letter '{other}' (expected one of F,I,T,W,E,P,H,L)")),
    }
}

fn build_state_from_eval_input(inp: &EvalInput, cards: &triplecargo::CardsDb) -> Result<GameState, String> {
    // Derive per-cell elements from board_elements (preferred) or from board[].element
    let mut derived_from_cells: Option<[Option<Element>; 9]> = None;
    if inp.board.iter().any(|c| c.element.is_some()) {
        let mut arr: [Option<Element>; 9] = [None; 9];
        for cell in &inp.board {
            if let Some(el) = cell.element {
                let e = elem_from_letter(el)?;
                if (cell.cell as usize) >= 9 {
                    return Err(format!("Board cell index {} out of range 0..8", cell.cell));
                }
                arr[cell.cell as usize] = Some(e);
            }
        }
        derived_from_cells = Some(arr);
    }

    let mut from_top: Option<[Option<Element>; 9]> = None;
    if let Some(v) = &inp.board_elements {
        if v.len() != 9 {
            return Err(format!("board_elements must have length 9, got {}", v.len()));
        }
        let mut arr: [Option<Element>; 9] = [None; 9];
        for i in 0..9usize {
            arr[i] = match v[i] {
                Some(ch) => Some(elem_from_letter(ch)?),
                None => None,
            };
        }
        from_top = Some(arr);
    }

    // If both present, they must match
    let cell_elements: Option<[Option<Element>; 9]> = match (from_top, derived_from_cells) {
        (Some(a), Some(b)) => {
            for i in 0..9 {
                if a[i] != b[i] {
                    return Err(format!("Element mismatch at cell {} between board_elements and board[].element", i));
                }
            }
            Some(a)
        }
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    // Start state with or without elements
    let mut s = if let Some(elems) = cell_elements {
        GameState::with_elements(inp.rules, elems)
    } else {
        GameState::new_empty(inp.rules)
    };

    // Populate board occupants
    for cell in &inp.board {
        if let (Some(card_id), Some(owner_ch)) = (cell.card_id, cell.owner) {
            if (cell.cell as usize) >= 9 {
                return Err(format!("Board cell index {} out of range 0..8", cell.cell));
            }
            if !s.board.is_empty(cell.cell) {
                return Err(format!("Duplicate occupant for cell {}", cell.cell));
            }
            if cards.get(card_id).is_none() {
                return Err(format!("Card id {} not found in cards DB", card_id));
            }
            let owner = match owner_ch {
                'A' => Owner::A,
                'B' => Owner::B,
                _ => return Err(format!("Invalid owner '{}' (expected 'A' or 'B')", owner_ch)),
            };
            let slot = triplecargo::board::Slot { owner, card_id };
            s.board.set(cell.cell, Some(slot));
        }
    }

    // Populate hands (sorted ascending for stability)
    let mut a_sorted = inp.hands.a.clone();
    let mut b_sorted = inp.hands.b.clone();
    a_sorted.sort_unstable();
    b_sorted.sort_unstable();
    if a_sorted.len() > 5 || b_sorted.len() > 5 {
        return Err("Hands arrays must be length <= 5".to_string());
    }
    s.hands_a = [None; 5];
    s.hands_b = [None; 5];
    for (i, id) in a_sorted.iter().copied().enumerate() {
        s.hands_a[i] = Some(id);
    }
    for (i, id) in b_sorted.iter().copied().enumerate() {
        s.hands_b[i] = Some(id);
    }

    // Side to move
    s.next = match inp.to_move {
        'A' => Owner::A,
        'B' => Owner::B,
        other => return Err(format!("Invalid to_move '{}' (expected 'A' or 'B')", other)),
    };

    // Recompute zobrist to ensure consistency
    s.zobrist = triplecargo::hash::recompute_zobrist(&s);

    // Optional validation for turn
    if let Some(t) = inp.turn {
        let filled = s.board.filled_count();
        if t != filled {
            return Err(format!("turn mismatch: provided {}, but board has {} occupied cells", t, filled));
        }
    }

    Ok(s)
}

fn eval_state_main(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // Mutual exclusivity check (basic)
    if args.export.is_some() {
        eprintln!("--eval-state is mutually exclusive with --export");
        std::process::exit(2);
    }

    // Preload cards once
    let cards = match load_cards_from_json(&args.cards) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Cards load error: {e}");
            process::exit(1);
        }
    };

    // Allocate a TT once (reused across requests)
    let budget_bytes = args.tt_bytes.saturating_mul(1024 * 1024);
    let cap = FixedTT::capacity_for_budget_bytes(budget_bytes);
    let mut tt = FixedTT::with_capacity_pow2(cap);

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    while let Some(line_res) = lines.next() {
        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                eprintln!("stdin read error: {e}");
                process::exit(1);
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        // Parse one request per line
        let inp: EvalInput = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Invalid JSON: {e}");
                process::exit(1);
            }
        };

        // Build input state
        let state0 = match build_state_from_eval_input(&inp, &cards) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("State build error: {e}");
                process::exit(1);
            }
        };

        if let Some(app) = inp.apply {
            // Apply request
            let mv = triplecargo::Move { card_id: app.card_id, cell: app.cell };
            match apply_move(&state0, &cards, mv) {
                Ok(state1) => {
                    // Prepare board vector (include element when elemental is on)
                    let mut board_vec: Vec<BoardCell> = Vec::with_capacity(9);
                    let elemental_enabled = state1.rules.elemental;
                    for cell in 0u8..9 {
                        let slot = state1.board.get(cell);
                        let (card_id, owner): (Option<u16>, Option<char>) = match slot {
                            Some(s) => (Some(s.card_id), Some(match s.owner { Owner::A => 'A', Owner::B => 'B' })),
                            None => (None, None),
                        };
                        let element_field: Option<Option<char>> = if elemental_enabled {
                            Some(state1.board.cell_element(cell).map(element_letter))
                        } else {
                            None
                        };
                        board_vec.push(BoardCell { cell, card_id, owner, element: element_field });
                    }

                    // Hands snapshot (compact)
                    let mut ha: Vec<u16> = Vec::with_capacity(5);
                    let mut hb: Vec<u16> = Vec::with_capacity(5);
                    for &o in state1.hands_a.iter() { if let Some(id) = o { ha.push(id); } }
                    for &o in state1.hands_b.iter() { if let Some(id) = o { hb.push(id); } }
                    let hands = HandsRecord { a: ha, b: hb };

                    let to_move = match state1.next { Owner::A => 'A', Owner::B => 'B' };
                    let turn = state1.board.filled_count();

                    let done = state1.is_terminal();
                    let (out_value, winner) = if done {
                        let m = score(&state1);
                        let v = if m > 0 { 1 } else if m < 0 { -1 } else { 0 };
                        let w = if m > 0 { Some('A') } else if m < 0 { Some('B') } else { None };
                        (v, w)
                    } else {
                        (0, None)
                    };

                    let resp = ApplyOut {
                        state: ApplyStateOut {
                            board: board_vec,
                            hands,
                            to_move,
                            turn,
                            rules: state1.rules,
                        },
                        done,
                        outcome: OutcomeOut {
                            mode: "winloss".to_string(),
                            value: out_value,
                            winner,
                        },
                        state_hash: format!("{:032x}", zobrist_key(&state1)),
                    };

                    println!("{}", serde_json::to_string(&resp).expect("serialize apply json"));
                    let _ = io::stdout().flush();
                }
                Err(e) => {
                    // Produce an error object for invalid apply without breaking the stream
                    let err = ErrorOut { error: format!("apply error: {e}") };
                    println!("{}", serde_json::to_string(&err).expect("serialize error json"));
                    let _ = io::stdout().flush();
                }
            }
        } else {
            // Eval request (search)
            let depth: u8 = 9u8 - state0.board.filled_count();
            let (val, bm, nodes) = search_root(&state0, &cards, depth, &mut tt);

            // Reconstruct PV up to remaining depth and ensure terminal by continuing perfect play
            let mut pv_moves = triplecargo::solver::reconstruct_pv(&state0, &cards, &tt, depth as usize);

            let mut sim = state0.clone();
            for mv in &pv_moves {
                match apply_move(&sim, &cards, *mv) {
                    Ok(ns) => sim = ns,
                    Err(e) => {
                        eprintln!("Internal error applying PV move: {e}");
                        process::exit(1);
                    }
                }
            }
            while !sim.is_terminal() {
                let rem = 9u8 - sim.board.filled_count();
                let (_v2, bm2, _n2) = search_root(&sim, &cards, rem, &mut tt);
                match bm2 {
                    Some(mv) => {
                        pv_moves.push(mv);
                        match apply_move(&sim, &cards, mv) {
                            Ok(ns) => sim = ns,
                            Err(e) => {
                                eprintln!("Internal error applying move while finishing PV: {e}");
                                process::exit(1);
                            }
                        }
                    }
                    None => break,
                }
            }

            let margin = score(&sim);

            let best_move_out = bm.map(|m| EvalMoveOut { card_id: m.card_id, cell: m.cell });
            let pv_out: Vec<EvalMoveOut> = pv_moves
                .into_iter()
                .map(|m| EvalMoveOut { card_id: m.card_id, cell: m.cell })
                .collect();

            let out = EvalOut {
                best_move: best_move_out, // omitted at terminal
                value: if val > 0 { 1 } else if val < 0 { -1 } else { 0 },
                margin,
                pv: pv_out,
                nodes,
                depth,
                state_hash: format!("{:032x}", zobrist_key(&state0)),
            };

            if args.verbose {
                eprintln!("[eval] nodes={} depth={}", nodes, depth);
            }

            println!("{}", serde_json::to_string(&out).expect("serialize eval json"));
            let _ = io::stdout().flush();
        }
    }

    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Eval-state fast path: strict stdin -> stdout JSON, no extra stdout logs
    if args.eval_state {
        return eval_state_main(&args);
    }

    let t0 = Instant::now();
 
    // Load cards
    let cards = load_cards_from_json(&args.cards).map_err(|e| format!("Cards load error: {e}"))?;
    println!("[precompute_new] Loaded {} cards (max id {}).", cards.len(), cards.max_id());

    let rules = parse_rules(&args.rules);

    let Some(export_path) = args.export.as_ref() else {
        eprintln!("[precompute_new] This binary only supports JSONL export. Re-run with --export PATH");
        return Ok(());
    };


    // Static resources for sampling
    let all_ids_sorted = all_card_ids_sorted(&cards);
    let bands_master = build_level_bands(&cards);

    // Progress counters (shared)
    let nodes_total = Arc::new(AtomicU64::new(0));
    let written_states = Arc::new(AtomicU64::new(0));
    let policy_info: String = match args.policy_format {
        PolicyFormatOpt::Mcts => format!("mcts@{}", args.mcts_rollouts),
        PolicyFormatOpt::Onehot => "onehot".to_string(),
    };

    match args.export_mode {
        ExportModeOpt::Trajectory => {
            let games = args.games;
            let total_states: u64 = (games as u64) * 9;
            let pb = ProgressBar::new(total_states);
            pb.set_style(
                ProgressStyle::with_template("[{elapsed_precise}] traj {bar:40.cyan/blue} {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            // Progress updater thread: updates PB message with floating rates and ETA every second.
            let pb_updater = pb.clone();
            let nodes_upd = Arc::clone(&nodes_total);
            let written_upd = Arc::clone(&written_states);
            let total_states_upd = total_states;
            let policy_info_upd = policy_info.clone();
            let start_instant = Instant::now();
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    let elapsed = start_instant.elapsed().as_secs_f64().max(1e-6);
                    let states_done_u = written_upd.load(Ordering::Relaxed);
                    let states_done = states_done_u as f64;
                    let states_per_sec = states_done / elapsed;
                    let nodes = nodes_upd.load(Ordering::Relaxed) as f64;
                    let nodes_per_sec = nodes / elapsed;
                    let nodes_m = nodes / 1.0e6;
                    let remaining = (total_states_upd.saturating_sub(states_done_u)) as f64;
                    let eta_secs = if states_per_sec > 1e-9 { (remaining / states_per_sec).round() as u64 } else { 0u64 };
                    let _nodes_per_sec = nodes / elapsed;
                    let msg = format!(
                        "states/s {:.1} | nodes {:.2}M | ETA {} | pol {}",
                        states_per_sec,
                        nodes_m,
                        format_hms(eta_secs),
                        policy_info_upd,
                    );
                    pb_updater.set_position(states_done_u);
                    pb_updater.set_message(msg);
                }
            });

            println!(
                "[export] Mode=trajectory games={} hand_strategy={:?} seed={:#x} policy_format={:?} value_mode={:?} play_strategy={:?} off_pv_rate={} off_pv_strategy={:?}",
                games, args.hand_strategy, args.seed, args.policy_format, args.value_mode, args.play_strategy, args.off_pv_rate, args.off_pv_strategy
            );

            // Determine worker count
            let worker_count = args.threads.unwrap_or_else(|| {
                let n = std::thread::available_parallelism().map(|nz| nz.get()).unwrap_or(1);
                if n <= 1 { 1 } else { n.saturating_sub(1) }
            }).max(1);

            println!(
                "[export] trajectory workers={} (default = available_parallelism-1)",
                worker_count
            );
            // Compute effective chunk size when not specified: min(32, max(1, floor(games/threads)))
            let effective_chunk_size: usize = match args.chunk_size {
                Some(v) => v.max(1),
                None => {
                    let per = (games / worker_count).max(1);
                    per.min(32)
                }
            };
            println!("[export] chunk_size={}", effective_chunk_size);

            // Create output JSONL file for trajectory export
            let file = File::create(export_path).map_err(|e| format!("Failed to create export file: {e}"))?;
            let writer = BufWriter::new(file);

            // Shared resources
            let cards_arc = Arc::new(cards);
            let bands_arc = Arc::new(bands_master);
            let all_ids_arc = Arc::new(all_ids_sorted);

            // Result channel
            let (res_tx, res_rx) = mpsc::channel::<(usize, Vec<String>)>();

            // Chunked dispatcher: deterministic round-robin across workers (no global job counter)

            // Move writer + receiver to writer thread to ensure single-writer semantics
            let pb_clone = pb.clone();
            let nodes_total_c = Arc::clone(&nodes_total);
            let written_states_c = Arc::clone(&written_states);
            let writer_handle = std::thread::spawn(move || {
                let mut writer = writer;
                let mut pending: BTreeMap<usize, Vec<String>> = BTreeMap::new();
                let mut next_write: usize = 0;
 
                while let Ok((gid, lines)) = res_rx.recv() {
                    // Track incoming work size for progress rates
                    let batch_len: u64 = lines.len() as u64;
                    nodes_total_c.fetch_add(0, Ordering::Relaxed); // placeholder if needed
                    written_states_c.fetch_add(batch_len, Ordering::Relaxed);
 
                    pending.insert(gid, lines);
                    // Flush contiguous games
                    while let Some(lines) = pending.remove(&next_write) {
                        for line in lines {
                            writer.write_all(line.as_bytes()).expect("export write error");
                            writer.write_all(b"\n").expect("export write error");
                        }
                        next_write = next_write.saturating_add(1);
                        pb_clone.set_position((next_write as u64) * 9);
                    }
                }
                // Drain (should be empty)
                for (_gid, lines) in pending.into_iter() {
                    for line in lines {
                        writer.write_all(line.as_bytes()).expect("export write error");
                        writer.write_all(b"\n").expect("export write error");
                    }
                }
                let _ = writer.flush();
            });

            // Spawn workers
            let mut worker_handles = Vec::with_capacity(worker_count);
            for worker_id in 0..worker_count {
                let res_tx = res_tx.clone();
                let cards_c = Arc::clone(&cards_arc);
                let bands_c = Arc::clone(&bands_arc);
                let all_ids_c = Arc::clone(&all_ids_arc);
                let nodes_total_c = Arc::clone(&nodes_total);

                let hand_strategy = args.hand_strategy.clone();
                let elements_opt = args.elements.clone();
                let policy_format = args.policy_format.clone();
                let mcts_rollouts = args.mcts_rollouts;
                let off_pv_rate = args.off_pv_rate;
                let off_pv_strategy = args.off_pv_strategy.clone();
                let value_mode = args.value_mode.clone();
                let base_seed = args.seed;
                let max_games = games;
                let tt_mib = args.tt_bytes;
                let chunk_size = effective_chunk_size;

                // New: play strategy and heuristic/mix parameters
                let play_strategy = args.play_strategy.clone();
                let mix_heuristic_early = args.mix_heuristic_early;
                let mix_heuristic_late = args.mix_heuristic_late;
                let mix_mcts = args.mix_mcts;
                let heur_w_corner = args.heur_w_corner;
                let heur_w_edge = args.heur_w_edge;
                let heur_w_center = args.heur_w_center;
                let heur_w_greedy = args.heur_w_greedy;
                let heur_w_defense = args.heur_w_defense;
                let heur_w_element = args.heur_w_element;

                let handle = std::thread::spawn(move || {
                    // Allocate a per-worker FixedTT and keep it warm across all games
                    let budget_bytes = tt_mib.saturating_mul(1024 * 1024);
                    let cap = FixedTT::capacity_for_budget_bytes(budget_bytes);
                    let approx = FixedTT::approx_bytes_for_capacity(cap);
                    eprintln!(
                        "[worker {}] TT target={} MiB capacity={} entries ≈{:.1} MiB",
                        worker_id,
                        tt_mib,
                        cap,
                        (approx as f64) / (1024.0 * 1024.0)
                    );
                    let mut tt = FixedTT::with_capacity_pow2(cap);

                    // Deterministic chunked scheduling: round-robin chunks across workers
                    let chunk_size = chunk_size.max(1);
                    let mut round: usize = 0;
                    loop {
                        let chunk_idx = round.saturating_mul(worker_count) + worker_id;
                        let start = chunk_idx.saturating_mul(chunk_size);
                        if start >= max_games {
                            break;
                        }
                        let end = (start + chunk_size).min(max_games);
                        for i in start..end {
                            // Per-game deterministic RNG
                            let mut rng_state: u64 = base_seed ^ ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    
                            // Sample hands deterministically
                            let (hand_a, hand_b) = match hand_strategy {
                                HandStrategyOpt::Random => {
                                    let mut pool = (*all_ids_c).clone();
                                    let ha = sample_hand_random(&mut pool, &mut rng_state);
                                    let hb = sample_hand_random(&mut pool, &mut rng_state);
                                    (ha, hb)
                                }
                                HandStrategyOpt::Stratified => {
                                    let mut bands = (*bands_c).clone();
                                    let ha = sample_hand_stratified(&mut bands, &mut rng_state);
                                    let hb = sample_hand_stratified(&mut bands, &mut rng_state);
                                    (ha, hb)
                                }
                            };
    
                            // Elements per game
                            let cell_elements = match elements_opt {
                                ElementsOpt::None => None,
                                ElementsOpt::Random => {
                                    let seed_for_elems = splitmix64(&mut rng_state);
                                    Some(gen_elements(seed_for_elems))
                                }
                            };
    
                            let mut state = GameState::with_hands(rules, hand_a, hand_b, cell_elements);

                            // New: per-game heuristic weights snapshot
                            let heur_weights = HeurWeights {
                                corner: heur_w_corner,
                                edge: heur_w_edge,
                                center: heur_w_center,
                                greedy: heur_w_greedy,
                                defense: heur_w_defense,
                                element: heur_w_element,
                            };
    
                            // Per-game off-PV activation
                            let off_pv_game = matches!(off_pv_strategy, OffPvStrategyOpt::Random | OffPvStrategyOpt::Weighted | OffPvStrategyOpt::Mcts)
                                && off_pv_rate > 0.0
                                && {
                                    let mut grng = rng_for_state(base_seed, i as u64, 0);
                                    grng.gen::<f32>() < off_pv_rate
                                };
    
                            let mut lines: Vec<String> = Vec::with_capacity(9);
    
                            let mut nodes_sum_local: u64 = 0;
                            for _ply in 0..9 {
                                let eff_depth = 9 - state.board.filled_count();
                                // Always search full remaining depth (trajectory semantics)
                                let (val, bm, child_vals, nodes) = search_root_with_children(&state, &cards_c, eff_depth, &mut tt);
                                nodes_sum_local = nodes_sum_local.saturating_add(nodes);
    
                                // Board vector
                                let mut board_vec: Vec<BoardCell> = Vec::with_capacity(9);
                                let elemental_enabled = state.rules.elemental;
                                for cell in 0u8..9 {
                                    let slot = state.board.get(cell);
                                    let (card_id, owner) = match slot {
                                        Some(s) => (Some(s.card_id), Some(match s.owner { Owner::A => 'A', Owner::B => 'B' })),
                                        None => (None, None),
                                    };
                                    let element_field: Option<Option<char>> = if elemental_enabled {
                                        Some(state.board.cell_element(cell).map(element_letter))
                                    } else {
                                        None
                                    };
                                    board_vec.push(BoardCell { cell, card_id, owner, element: element_field });
                                }
    
                                // Hands snapshot
                                let mut hands_a_vec: Vec<u16> = Vec::with_capacity(5);
                                let mut hands_b_vec: Vec<u16> = Vec::with_capacity(5);
                                for &o in state.hands_a.iter() { if let Some(id) = o { hands_a_vec.push(id); } }
                                for &o in state.hands_b.iter() { if let Some(id) = o { hands_b_vec.push(id); } }
                                let hands_rec = HandsRecord { a: hands_a_vec, b: hands_b_vec };
    
                                // Optional root MCTS distribution cache for off-PV re-use
                                let mut mcts_root_dist: Option<BTreeMap<String, f32>> = None;
    
                                // Policy output
                                let policy_out: Option<PolicyOut> = match policy_format {
                                    PolicyFormatOpt::Onehot => {
                                        bm.map(|m| PolicyOut::Move { card_id: m.card_id, cell: m.cell })
                                    }
                                    PolicyFormatOpt::Mcts => {
                                        let seed_for_rollouts = splitmix64(&mut rng_state);
                                        let dist = mcts_policy_distribution(&state, &cards_c, mcts_rollouts, seed_for_rollouts);
                                        mcts_root_dist = Some(dist.clone());
                                        Some(PolicyOut::Dist(dist))
                                    }
                                };
    
                                let to_move = match state.next { Owner::A => 'A', Owner::B => 'B' };
                                let vt = compute_value_target(&value_mode, val, state.next);
    
                                let rec = ExportRecord {
                                    game_id: i,
                                    state_idx: state.board.filled_count(),
                                    board: board_vec,
                                    hands: hands_rec,
                                    to_move,
                                    turn: state.board.filled_count(),
                                    rules: state.rules,
                                    policy_target: policy_out,
                                    value_target: vt,
                                    value_mode: value_mode_str(&value_mode).to_string(),
                                    off_pv: off_pv_game,
                                    state_hash: format!("{:032x}", zobrist_key(&state)),
                                };
                                let line = serde_json::to_string(&rec).expect("serialize JSONL record");
                                lines.push(line);
    
                                // Choose next move based on play strategy (mix overrides off-PV)
                                let next_mv = match play_strategy {
                                    PlayStrategyOpt::Mix => {
                                        // Use MCTS distribution only when policy_format is Mcts
                                        let dist_opt = if matches!(policy_format, PolicyFormatOpt::Mcts) {
                                            mcts_root_dist.as_ref()
                                        } else {
                                            None
                                        };
                                        choose_next_move_mixed(
                                            &state,
                                            &cards_c,
                                            bm,
                                            dist_opt,
                                            base_seed,
                                            i as u64,
                                            state.board.filled_count(),
                                            mix_heuristic_early,
                                            mix_heuristic_late,
                                            mix_mcts,
                                            &heur_weights,
                                            matches!(policy_format, PolicyFormatOpt::Mcts),
                                        )
                                    }
                                    PlayStrategyOpt::Pv => {
                                        // Preserve off-PV behavior when not mixing
                                        if off_pv_game {
                                            match off_pv_strategy {
                                                OffPvStrategyOpt::Random => {
                                                    pick_random_non_pv_move(&state, bm, base_seed, i as u64, state.board.filled_count())
                                                }
                                                OffPvStrategyOpt::Weighted => {
                                                    pick_weighted_non_pv_move(&child_vals, bm, base_seed, i as u64, state.board.filled_count())
                                                }
                                                OffPvStrategyOpt::Mcts => {
                                                    if matches!(policy_format, PolicyFormatOpt::Mcts) {
                                                        let dist_ref = mcts_root_dist.as_ref().expect("mcts_root_dist present");
                                                        pick_mcts_non_pv_move_from_dist(dist_ref, bm, base_seed, i as u64, state.board.filled_count())
                                                    } else {
                                                        let seed_for_mcts_step = splitmix64(&mut rng_state);
                                                        let dist2 = mcts_policy_distribution(&state, &cards_c, mcts_rollouts, seed_for_mcts_step);
                                                        pick_mcts_non_pv_move_from_dist(&dist2, bm, base_seed, i as u64, state.board.filled_count())
                                                    }
                                                }
                                            }
                                        } else {
                                            bm
                                        }
                                    }
                                    PlayStrategyOpt::Mcts => {
                                        if matches!(policy_format, PolicyFormatOpt::Mcts) {
                                            match mcts_root_dist.as_ref() {
                                                Some(dist) => choose_next_move_mcts(&state, dist, base_seed, i as u64, state.board.filled_count()),
                                                None => bm,
                                            }
                                        } else {
                                            // No MCTS policy available; fall back to PV/off-PV behavior as-is
                                            if off_pv_game {
                                                match off_pv_strategy {
                                                    OffPvStrategyOpt::Random => {
                                                        pick_random_non_pv_move(&state, bm, base_seed, i as u64, state.board.filled_count())
                                                    }
                                                    OffPvStrategyOpt::Weighted => {
                                                        pick_weighted_non_pv_move(&child_vals, bm, base_seed, i as u64, state.board.filled_count())
                                                    }
                                                    OffPvStrategyOpt::Mcts => bm,
                                                }
                                            } else {
                                                bm
                                            }
                                        }
                                    }
                                    PlayStrategyOpt::Heuristic => {
                                        choose_next_move_heuristic(&state, &cards_c, &heur_weights, base_seed, i as u64, state.board.filled_count())
                                    }
                                };
    
                                if let Some(mv) = next_mv {
                                    if let Ok(ns) = apply_move(&state, &cards_c, mv) {
                                        state = ns;
                                    } else {
                                        break;
                                    }
                                } else {
                                    break;
                                }
    
                                // Throttled progress update message (in main would need shared state; skip here in worker)
                            } // ply loop
    
                            let _ = nodes_total_c.fetch_add(nodes_sum_local, Ordering::Relaxed);
                            let _ = res_tx.send((i, lines));
                        } // for i in chunk
                        round = round.saturating_add(1);
                    } // chunked worker loop
                });
                worker_handles.push(handle);
            }

            // Drop main res_tx
            drop(res_tx);

            // Wait for workers
            for h in worker_handles {
                let _ = h.join();
            }
            // Writer exits when channel is closed
            let _ = writer_handle.join();

            // Finish progress bar
            pb.set_position(total_states);
            pb.finish_and_clear();
            let elapsed = t0.elapsed();
            println!(
                "[export] Done. mode=trajectory games={} lines={} elapsed_ms={}",
                games,
                games * 9,
                elapsed.as_millis()
            );

            Ok(())
        }
        ExportModeOpt::Full => {
            // Deterministic full-state export for a single sampled hand pair (ignores --games)
            // Reuse the single-thread path for simplicity/determinism
            let mut rng_state: u64 = args.seed;
            let (hand_a, hand_b) = match args.hand_strategy {
                HandStrategyOpt::Random => {
                    let mut pool = all_ids_sorted.clone();
                    let ha = sample_hand_random(&mut pool, &mut rng_state);
                    let hb = sample_hand_random(&mut pool, &mut rng_state);
                    (ha, hb)
                }
                HandStrategyOpt::Stratified => {
                    let mut bands = bands_master.clone();
                    let ha = sample_hand_stratified(&mut bands, &mut rng_state);
                    let hb = sample_hand_stratified(&mut bands, &mut rng_state);
                    (ha, hb)
                }
            };

            let cell_elements = match args.elements {
                ElementsOpt::None => None,
                ElementsOpt::Random => {
                    let seed_for_elems = splitmix64(&mut rng_state);
                    Some(gen_elements(seed_for_elems))
                }
            };

            let initial = GameState::with_hands(rules, hand_a, hand_b, cell_elements);

            // Create output JSONL file for full export
            let file = File::create(export_path).map_err(|e| format!("Failed to create export file: {e}"))?;
            let mut writer = BufWriter::new(file);

            // Enumerate via DFS, computing labels with search_root
            let mut out: Vec<ExportRecord> = Vec::new();
            let mut visited: hashbrown::HashSet<u128> = hashbrown::HashSet::default();
            // Single FixedTT sized by --tt-bytes for the entire full export
            let budget_bytes = args.tt_bytes.saturating_mul(1024 * 1024);
            let cap = FixedTT::capacity_for_budget_bytes(budget_bytes);
            let approx = FixedTT::approx_bytes_for_capacity(cap);
            eprintln!(
                "[full] TT target={} MiB capacity={} entries ≈{:.1} MiB",
                args.tt_bytes,
                cap,
                (approx as f64) / (1024.0 * 1024.0)
            );
            let mut tt = FixedTT::with_capacity_pow2(cap);
            let cards_ref = &cards;

            fn eff_depth_for(state: &GameState, cap: Option<u8>) -> u8 {
                let full = 9 - state.board.filled_count();
                match cap {
                    Some(c) => full.min(c),
                    None => full,
                }
            }

            fn push_record_full(
                state: &GameState,
                cards: &triplecargo::CardsDb,
                tt: &mut dyn triplecargo::solver::TranspositionTable,
                cap: Option<u8>,
                out: &mut Vec<ExportRecord>,
                export_cfg: &ExportConfig,
            ) {
                let depth = eff_depth_for(state, cap);
                let (val, bm, _nodes) = search_root(state, cards, depth, tt);

                let mut board_vec: Vec<BoardCell> = Vec::with_capacity(9);
                let elemental_enabled = state.rules.elemental;
                for cell in 0u8..9 {
                    let slot = state.board.get(cell);
                    let (card_id, owner) = match slot {
                        Some(s) => (Some(s.card_id), Some(match s.owner { Owner::A => 'A', Owner::B => 'B' })),
                        None => (None, None),
                    };
                    let element_field: Option<Option<char>> = if elemental_enabled {
                        Some(state.board.cell_element(cell).map(element_letter))
                    } else {
                        None
                    };
                    board_vec.push(BoardCell { cell, card_id, owner, element: element_field });
                }

                let mut ha: Vec<u16> = Vec::with_capacity(5);
                let mut hb: Vec<u16> = Vec::with_capacity(5);
                for &o in state.hands_a.iter() { if let Some(id) = o { ha.push(id); } }
                for &o in state.hands_b.iter() { if let Some(id) = o { hb.push(id); } }
                let hands = HandsRecord { a: ha, b: hb };

                let policy_out: Option<PolicyOut> = match export_cfg.policy_format {
                    PolicyFormatOpt::Onehot => bm.map(|m| PolicyOut::Move { card_id: m.card_id, cell: m.cell }),
                    PolicyFormatOpt::Mcts => {
                        let z = zobrist_key(state);
                        let seed = export_cfg.base_seed ^ (z as u64) ^ ((z >> 64) as u64);
                        let dist = mcts_policy_distribution(state, cards, export_cfg.mcts_rollouts, seed);
                        Some(PolicyOut::Dist(dist))
                    }
                };

                let to_move = match state.next { Owner::A => 'A', Owner::B => 'B' };
                let vt = compute_value_target(&export_cfg.value_mode, val, state.next);

                let rec = ExportRecord {
                    game_id: 0,
                    state_idx: state.board.filled_count(),
                    board: board_vec,
                    hands,
                    to_move,
                    turn: state.board.filled_count(),
                    rules: state.rules,
                    policy_target: policy_out,
                    value_target: vt,
                    value_mode: value_mode_str(&export_cfg.value_mode).to_string(),
                    off_pv: false,
                    state_hash: format!("{:032x}", zobrist_key(state)),
                };
                out.push(rec);
            }

            fn dfs(
                state: &GameState,
                cards: &triplecargo::CardsDb,
                tt: &mut dyn triplecargo::solver::TranspositionTable,
                visited: &mut hashbrown::HashSet<u128>,
                cap: Option<u8>,
                out: &mut Vec<ExportRecord>,
                export_cfg: &ExportConfig,
            ) {
                let key = zobrist_key(state);
                if !visited.insert(key) {
                    return;
                }
                push_record_full(state, cards, tt, cap, out, export_cfg);
                if state.is_terminal() {
                    return;
                }
                let moves = legal_moves(state);
                for mv in moves {
                    if let Ok(ns) = apply_move(state, cards, mv) {
                        dfs(&ns, cards, tt, visited, cap, out, export_cfg);
                    }
                }
            }

            let export_cfg = ExportConfig {
                policy_format: args.policy_format.clone(),
                value_mode: args.value_mode.clone(),
                mcts_rollouts: args.mcts_rollouts,
                off_pv_rate: args.off_pv_rate,
                off_pv_strategy: args.off_pv_strategy.clone(),
                base_seed: args.seed,
            };

            dfs(&initial, cards_ref, &mut tt, &mut visited, args.max_depth, &mut out, &export_cfg);

            let mut total_lines = 0usize;
            for mut rec in out {
                rec.game_id = 0;
                let line = serde_json::to_string(&rec).map_err(|e| format!("serialize JSONL record error: {e}"))?;
                writer.write_all(line.as_bytes()).map_err(|e| format!("export write error: {e}"))?;
                writer.write_all(b"\n").map_err(|e| format!("export write error: {e}"))?;
                total_lines += 1;
                if total_lines % 10_000 == 0 {
                    let _ = writer.flush();
                }
            }
            let _ = writer.flush();

            let elapsed = t0.elapsed();
            println!(
                "[export] Done. mode=full lines={} elapsed_ms={}",
                total_lines,
                elapsed.as_millis()
            );

            Ok(())
        }
        ExportModeOpt::Graph => {
            // Graph mode: enumerate full game graph for a single initial hand pair + optional elements,
            // then retrograde solve (two-phase pipeline inside solver::graph).
            // UX:
            // - If --graph-input is false (default), sample hands/elements deterministically using --hand-strategy and --seed.
            // - If --graph-input is true, require --graph-hand-a and --graph-hand-b; optional --graph-elements.
            //   Each hand must contain 5 unique ids (per hand), ids must exist in cards DB.
            //   --graph-elements (if provided) must have 9 entries and no duplicate element letters.

            // Helpers (local to arm)
            fn parse_hand_list(s: &str) -> Result<[u16; 5], String> {
                let mut out: [u16; 5] = [0; 5];
                let mut seen: hashbrown::HashSet<u16> = hashbrown::HashSet::new();
                let parts: Vec<&str> = s.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()).collect();
                if parts.len() != 5 {
                    return Err(format!("expected exactly 5 card ids, got {}", parts.len()));
                }
                for (i, tok) in parts.iter().enumerate() {
                    let id: u16 = tok.parse().map_err(|_| format!("invalid card id '{}'", tok))?;
                    if !seen.insert(id) {
                        return Err(format!("duplicate card id {} in the same hand", id));
                    }
                    out[i] = id;
                }
                Ok(out)
            }
            fn parse_graph_elements_arg(s: &str) -> Result<[Option<Element>; 9], String> {
                let mut out: [Option<Element>; 9] = [None; 9];
                let mut seen_elem: hashbrown::HashSet<Element> = hashbrown::HashSet::new();
                let parts: Vec<&str> = s.split(',').map(|t| t.trim()).collect();
                if parts.len() != 9 {
                    return Err(format!("--graph-elements must have exactly 9 comma-separated entries, got {}", parts.len()));
                }
                for (i, tok) in parts.iter().enumerate() {
                    let opt = if tok.eq_ignore_ascii_case("-") || tok.eq_ignore_ascii_case("none") || tok.eq_ignore_ascii_case("null") || tok.is_empty() {
                        None
                    } else {
                        let ch = tok.chars().next().ok_or_else(|| format!("invalid element token '{}'", tok))?;
                        let el = elem_from_letter(ch).map_err(|e| format!("invalid element at position {}: {}", i, e))?;
                        if !seen_elem.insert(el) {
                            return Err(format!("duplicate element '{}' in --graph-elements", tok));
                        }
                        Some(el)
                    };
                    out[i] = opt;
                }
                Ok(out)
            }

            let mut rng_state: u64 = args.seed;

            // Decide hands + elements
            let (hand_a, hand_b, cell_elements): ([u16; 5], [u16; 5], Option<[Option<Element>; 9]>) = if args.graph_input {
                // Require both hands
                let ha_s = args.graph_hand_a.as_ref().ok_or_else(|| "--graph-input requires --graph-hand-a".to_string())
                    .map_err(|e| format!("{}", e)).unwrap();
                let hb_s = args.graph_hand_b.as_ref().ok_or_else(|| "--graph-input requires --graph-hand-b".to_string())
                    .map_err(|e| format!("{}", e)).unwrap();

                let ha = parse_hand_list(ha_s).map_err(|e| format!("--graph-hand-a error: {e}"))?;
                let hb = parse_hand_list(hb_s).map_err(|e| format!("--graph-hand-b error: {e}"))?;

                // Validate against cards DB (ids exist)
                for id in ha.iter().chain(hb.iter()) {
                    if cards.get(*id).is_none() {
                        return Err(format!("Card id {} not found in cards DB", id).into());
                    }
                }

                // Optional per-cell elements, with uniqueness constraint
                let elems = match &args.graph_elements {
                    Some(es) => Some(parse_graph_elements_arg(es).map_err(|e| format!("--graph-elements error: {e}"))?),
                    None => None,
                };

                (ha, hb, elems)
            } else {
                // Deterministic sampling
                let (ha, hb) = match args.hand_strategy {
                    HandStrategyOpt::Random => {
                        let mut pool = all_ids_sorted.clone();
                        let ha = sample_hand_random(&mut pool, &mut rng_state);
                        let hb = sample_hand_random(&mut pool, &mut rng_state);
                        (ha, hb)
                    }
                    HandStrategyOpt::Stratified => {
                        let mut bands = bands_master.clone();
                        let ha = sample_hand_stratified(&mut bands, &mut rng_state);
                        let hb = sample_hand_stratified(&mut bands, &mut rng_state);
                        (ha, hb)
                    }
                };
                let elems = match args.elements {
                    ElementsOpt::None => None,
                    ElementsOpt::Random => {
                        let seed_for_elems = splitmix64(&mut rng_state);
                        Some(gen_elements(seed_for_elems))
                    }
                };
                (ha, hb, elems)
            };

            let initial = GameState::with_hands(rules, hand_a, hand_b, cell_elements);

            // Determine export directory semantics for Graph mode
            let export_dir = export_path;
            // Verify directory or create it
            if export_dir.exists() {
                let meta = std::fs::metadata(export_dir).map_err(|e| format!("stat export path error: {e}"))?;
                if !meta.is_dir() {
                    return Err(format!("--export path '{}' exists and is not a directory. Graph mode requires a directory.", export_dir.display()).into());
                }
            } else {
                std::fs::create_dir_all(export_dir).map_err(|e| format!("create export directory error: {e}"))?;
            }

            // Resolve compression settings
            let mut zstd_enabled = args.zstd && !args.no_compress;
            let zstd_level = args.zstd_level.clamp(1, 10);
            let zstd_threads: usize = match args.zstd_threads {
                Some(v) => v.max(1),
                None => {
                    let n = std::thread::available_parallelism().map(|nz| nz.get()).unwrap_or(1);
                    n.min(4).max(1)
                }
            };
            let frame_lines = args.zstd_frame_lines.max(1);
            let index_enabled = args.zstd_index && zstd_enabled;

            // Build file paths
            let nodes_name = if zstd_enabled { "nodes.jsonl.zst" } else { "nodes.jsonl" };
            let nodes_path = export_dir.join(nodes_name);
            // Open nodes file
            let nodes_file = File::create(&nodes_path).map_err(|e| format!("create nodes file error: {e}"))?;

            let idx_path_opt = if index_enabled {
                Some(export_dir.join("nodes.idx.jsonl"))
            } else {
                None
            };
            let idx_file_opt = match idx_path_opt.as_ref() {
                Some(p) => Some(File::create(p).map_err(|e| format!("create index file error: {e}"))?),
                None => None,
            };

            // Construct sink
            let sync_final = matches!(args.sync_mode, SyncModeOpt::Final);
            let buf_cap = triplecargo::solver::graph_writer::BUF_WRITER_CAP_BYTES;
            let frame_bytes = args.zstd_frame_bytes;
            
            // Track whether we created sharded node outputs and their names for manifest building
            let mut nodes_sharded: bool = false;
            let mut nodes_shard_names: Vec<String> = Vec::new();
            
            let mut sink_box: Box<dyn triplecargo::solver::GraphJsonlSink> = if zstd_enabled && args.shards > 1 {
                // Create per-shard node files nodes_000.jsonl.zst ... nodes_{N-1}.jsonl.zst
                let shards = args.shards;
                let mut shard_files: Vec<File> = Vec::with_capacity(shards);
                for i in 0..shards {
                    let name = format!("nodes_{:03}.jsonl.zst", i);
                    let path = export_dir.join(&name);
                    let f = File::create(&path).map_err(|e| format!("create shard nodes file {} error: {e}", path.display()))?;
                    shard_files.push(f);
                    nodes_shard_names.push(name);
                }
                nodes_sharded = true;
                // Shared index file (if enabled) is already prepared in idx_file_opt
                Box::new(triplecargo::solver::AsyncShardedZstdFramesJsonlWriter::new(
                    shard_files,
                    idx_file_opt,
                    buf_cap,            // BufWriter capacity (32 MiB)
                    zstd_level,
                    zstd_threads,
                    frame_lines,
                    frame_bytes,        // frame max bytes (soft cap)
                    args.writer_queue_frames, // raw frames queue
                    sync_final,
                ))
            } else if zstd_enabled {
                // Single-file zstd writer (non-sharded)
                Box::new(triplecargo::solver::AsyncZstdFramesJsonlWriter::new(
                    nodes_file,
                    idx_file_opt,
                    buf_cap,            // BufWriter capacity (32 MiB)
                    zstd_level,
                    zstd_threads,
                    frame_lines,
                    frame_bytes,                       // frame max bytes (soft cap)
                    args.writer_queue_frames,          // raw frames queue
                    args.zstd_workers,                 // compression workers (0 = single-stage)
                    args.writer_queue_compressed,      // compressed frames queue
                    sync_final,
                ))
            } else {
                Box::new(triplecargo::solver::PlainJsonlWriter::new(
                    nodes_file,
                    buf_cap,
                    sync_final,
                ))
            };

            // Run high-throughput export through sink
            let t_start = Instant::now();
            let outcome_res = triplecargo::solver::graph_precompute_export_with_sink(&initial, &cards, args.max_depth, &mut *sink_box);

            // If zstd was requested but failed (e.g., codec unavailable), fall back to plain JSONL
            let (outcome, stats, final_nodes_name, final_index_name, compression_enabled) = match outcome_res {
                Ok(o) => {
                    let stats = triplecargo::solver::GraphJsonlSink::finish_mut(&mut *sink_box)
                        .map_err(|e| format!("finalize sink error: {e}"))?;
                    let idx_name = if index_enabled && stats.index_sha256_hex.is_some() { Some("nodes.idx.jsonl".to_string()) } else { None };
                    (o, stats, nodes_name.to_string(), idx_name, zstd_enabled)
                }
                Err(e) if zstd_enabled => {
                    eprintln!("[graph] zstd writer failed: {e}. Falling back to uncompressed JSONL.");
                    // Re-open plain writer and rerun
                    let plain_nodes_name = "nodes.jsonl".to_string();
                    let plain_nodes_path = export_dir.join(&plain_nodes_name);
                    let plain_nodes_file = File::create(&plain_nodes_path).map_err(|e| format!("create plain nodes file error: {e}"))?;
                    zstd_enabled = false;
                    let mut plain_sink = triplecargo::solver::PlainJsonlWriter::new(
                        plain_nodes_file,
                        buf_cap,
                        sync_final,
                    );
                    let outcome2 = triplecargo::solver::graph_precompute_export_with_sink(&initial, &cards, args.max_depth, &mut plain_sink)
                        .map_err(|e| format!("graph export retry (plain) error: {e}"))?;
                    let stats2 = triplecargo::solver::GraphJsonlSink::finish_mut(&mut plain_sink)
                        .map_err(|e| format!("finalize plain sink error: {e}"))?;
                    (outcome2, stats2, plain_nodes_name, None, false)
                }
                Err(e) => {
                    return Err(Box::<dyn std::error::Error>::from(format!("graph precompute error: {e}")));
                }
            };

            let elapsed = t_start.elapsed();

            // Build manifest JSON
            let manifest_name = "graph.manifest.json";
            let manifest_path = export_dir.join(manifest_name);

            let compression_obj = if compression_enabled {
                serde_json::json!({
                    "codec": "zstd",
                    "enabled": true,
                    "level": zstd_level,
                    "threads": zstd_threads,
                    "frame_lines": frame_lines as u64,
                    "frame_count": stats.frame_count,
                    "indexed": stats.index_sha256_hex.is_some(),
                })
            } else {
                serde_json::json!({
                    "codec": "none",
                    "enabled": false,
                    "level": serde_json::Value::Null,
                    "threads": serde_json::Value::Null,
                    "frame_lines": serde_json::Value::Null,
                    "frame_count": 0,
                    "indexed": false,
                })
            };

            // Files listing: use shard array when sharded, otherwise single filename
            let files_obj = if nodes_sharded {
                serde_json::json!({
                    "nodes": nodes_shard_names,
                    "index": final_index_name,
                    "manifest": manifest_name,
                })
            } else {
                serde_json::json!({
                    "nodes": final_nodes_name,
                    "index": final_index_name,
                    "manifest": manifest_name,
                })
            };
            
            // Integrity: when sharded, include per-shard node digests; otherwise single nodes digest
            let integrity_obj = if nodes_sharded {
                serde_json::json!({
                    "nodes_sha256_list": stats.nodes_sha256_list.clone().unwrap_or_default(),
                    "index_sha256": stats.index_sha256_hex,
                })
            } else {
                serde_json::json!({
                    "nodes_sha256": stats.nodes_sha256_hex,
                    "index_sha256": stats.index_sha256_hex,
                })
            };

            let totals_obj = serde_json::json!({
                "states": outcome.totals_states,
                "terminals": outcome.totals_terminals,
                "by_depth": outcome.totals_by_depth,
            });

            let manifest = serde_json::json!({
                "files": files_obj,
                "compression": compression_obj,
                "totals": totals_obj,
                "integrity": integrity_obj,
                "logical_checksum": outcome.logical_checksum_hex,
            });

            // Write manifest
            {
                let mut mf = File::create(&manifest_path).map_err(|e| format!("create manifest file error: {e}"))?;
                let buf = serde_json::to_vec_pretty(&manifest).map_err(|e| format!("serialize manifest error: {e}"))?;
                mf.write_all(&buf).map_err(|e| format!("write manifest error: {e}"))?;
            }

            println!(
                "[export] Done. mode=graph elapsed_ms={} nodes_file={} indexed={} frames={}",
                elapsed.as_millis(),
                export_dir.join(&final_nodes_name).display(),
                compression_enabled && stats.index_sha256_hex.is_some(),
                stats.frame_count
            );
            Ok(())
        }
    }
}