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
use serde::Serialize;
use serde_json;

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
enum OffPvStrategyOpt {
    Random,
    Weighted,
    Mcts,
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

#[derive(Serialize)]
struct BoardCell {
    cell: u8,
    card_id: Option<u16>,
    owner: Option<char>, // 'A' | 'B'
    #[serde(skip_serializing_if = "Option::is_none")]
    element: Option<Option<char>>,
}

#[derive(Serialize)]
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let t0 = Instant::now();

    // Load cards
    let cards = load_cards_from_json(&args.cards).map_err(|e| format!("Cards load error: {e}"))?;
    println!("[precompute_new] Loaded {} cards (max id {}).", cards.len(), cards.max_id());

    let rules = parse_rules(&args.rules);

    let Some(export_path) = args.export.as_ref() else {
        eprintln!("[precompute_new] This binary only supports JSONL export. Re-run with --export PATH");
        return Ok(());
    };

    // Output file
    let file = File::create(export_path).map_err(|e| format!("Failed to create export file: {e}"))?;
    let mut writer = BufWriter::new(file);

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
                "[export] Mode=trajectory games={} hand_strategy={:?} seed={:#x} policy_format={:?} value_mode={:?} off_pv_rate={} off_pv_strategy={:?}",
                games, args.hand_strategy, args.seed, args.policy_format, args.value_mode, args.off_pv_rate, args.off_pv_strategy
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
    
                                // Choose next move based on off-PV setting
                                let next_mv = if off_pv_game {
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
    }
}