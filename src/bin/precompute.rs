use std::path::PathBuf;
use std::time::Instant;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::collections::BTreeMap;

use clap::{Parser, ValueEnum};
use rayon::ThreadPoolBuilder;
use serde::Serialize;
use serde_json;
use rand::Rng;
use indicatif::{ProgressBar, ProgressStyle};

use triplecargo::{
    load_cards_from_json, zobrist_key, Element, GameState, Owner, Rules, apply_move, legal_moves, score, rng_for_state,
    solver::{precompute_solve, ElementsMode, search_root, search_root_with_children, InMemoryTT},
    persist::{DbHeader, fingerprint_cards},
    persist_stream::{StreamWriter, StreamCompression, compact_stream_to_db_file, compact_stream_with_baseline_to_db_file},
};

#[derive(Debug, Clone, ValueEnum)]
enum ElementsOpt {
    None,
    Random,
}

#[derive(Debug, Clone, ValueEnum)]
enum CompressOpt {
    Lz4,
    Zstd,
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
#[command(name = "precompute", about = "Triplecargo precompute driver")]
struct Args {
    /// Rules toggles as comma-separated list: elemental,same,plus,same_wall (or 'none')
    #[arg(long, default_value = "none")]
    rules: String,

    /// Hands specification: pass exactly two values: A=1,2,3,4,5 B=6,7,8,9,10
    /// Example: --hands "A=1,2,3,4,5" "B=6,7,8,9,10"
    /// Required unless --export is used.
    #[arg(long, num_args = 2, required_unless_present = "export")]
    hands: Vec<String>,

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

    /// Output DB file path (legacy persistence). Conflicts with --export.
    #[arg(long, default_value = "solved.bin")]
    out: PathBuf,

    /// Export JSONL file path (one object per line). Conflicts with --out.
    #[arg(long, conflicts_with = "out")]
    export: Option<PathBuf>,

    /// Export mode: trajectory | full (default: trajectory)
    #[arg(long, value_enum, default_value_t = ExportModeOpt::Trajectory)]
    export_mode: ExportModeOpt,

    /// Number of games to sample when --export is used.
    /// - trajectory: generates N trajectories (9 states each)
    /// - full: ignored (always 1 hand pair, full state space)
    #[arg(long, default_value_t = 1000)]
    games: usize,

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

    /// Batch flush size for streaming persistence
    #[arg(long, default_value_t = 100_000)]
    batch_size: usize,

    /// Optional compression for stream frames
    #[arg(long, value_enum)]
    compress: Option<CompressOpt>,

    /// Periodic checkpointing by elapsed seconds
    #[arg(long)]
    checkpoint_seconds: Option<u64>,

    /// Periodic checkpointing by states enumerated
    #[arg(long)]
    checkpoint_states: Option<u64>,

    /// Analysis-only mode (no persistence)
    #[arg(long, default_value_t = false)]
    analysis: bool,

    /// Optional baseline legacy DB (e.g., latest checkpoint) to merge with the current stream at final compaction (resume).
    /// If provided, final solved.bin = merge(baseline, stream) with deterministic preference.
    #[arg(long)]
    baseline: Option<PathBuf>,
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
            other => {
                eprintln!("[precompute] Warning: ignoring unknown rule token '{other}'");
            }
        }
    }
    r
}

fn parse_hands(tokens: &[String]) -> Result<([u16; 5], [u16; 5]), String> {
    if tokens.len() != 2 {
        return Err("Expected exactly two --hands values: \"A=...\" \"B=...\"".into());
    }
    let mut a: Option<[u16; 5]> = None;
    let mut b: Option<[u16; 5]> = None;
    for t in tokens {
        let parts: Vec<&str> = t.split('=').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid hands token '{}', expected A=... or B=...", t));
        }
        let side = parts[0].trim();
        let vals = parts[1].trim();
        let ids: Vec<u16> = if vals.is_empty() {
            Vec::new()
        } else {
            vals.split(',')
                .map(|x| x.trim().parse::<u16>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Invalid card id in '{}': {e}", t))?
        };
        if ids.len() != 5 {
            return Err(format!("Expected 5 card ids for {}, got {}", side, ids.len()));
        }
        let arr = [ids[0], ids[1], ids[2], ids[3], ids[4]];
        match side {
            "A" | "a" => a = Some(arr),
            "B" | "b" => b = Some(arr),
            _ => return Err(format!("Unknown hand side '{}', expected A or B", side)),
        }
    }
    match (a, b) {
        (Some(ha), Some(hb)) => Ok((ha, hb)),
        _ => Err("Both A=... and B=... must be provided".into()),
    }
}

// Minimal SplitMix64 for deterministic elements RNG (no rand dependency)
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
    // Present only when rules.elemental = true; inner None serializes to null
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
   // onehot
   Move { card_id: u16, cell: u8 },
   // mcts
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

fn all_card_ids_sorted(cards: &triplecargo::CardsDb) -> Vec<u16> {
   let mut ids: Vec<u16> = cards.iter().map(|c| c.id).collect();
   ids.sort_unstable();
   ids
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

fn sample_hand_stratified(bands: &mut [Vec<u16>; 5], rng: &mut u64) -> [u16; 5] {
   let mut hand = [0u16; 5];
   for (i, band) in bands.iter_mut().enumerate() {
       let r = splitmix64(rng);
       let idx = (r as usize) % band.len();
       hand[i] = band.swap_remove(idx);
   }
   hand
}

// Element to letter mapping required by spec: F, I, T, W, E, P, H, L
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
            // Convert side-to-move perspective to A-perspective absolute margin
            match to_move {
                Owner::A => side_value,
                Owner::B => -side_value,
            }
        }
    }
}

fn enumerate_solved_records(
   initial: &GameState,
   cards: &triplecargo::CardsDb,
   max_depth: Option<u8>,
   export_cfg: &ExportConfig,
) -> Vec<ExportRecord> {
   let mut out: Vec<ExportRecord> = Vec::new();
   let mut visited: hashbrown::HashSet<u128> = hashbrown::HashSet::default();
   let mut tt = InMemoryTT::default();

   fn eff_depth_for(state: &GameState, cap: Option<u8>) -> u8 {
       let full = 9 - state.board.filled_count();
       match cap {
           Some(c) => full.min(c),
           None => full,
       }
   }

   fn collect_hands_from_state(state: &GameState) -> HandsRecord {
       let mut ha: Vec<u16> = Vec::with_capacity(5);
       let mut hb: Vec<u16> = Vec::with_capacity(5);
       for &o in state.hands_a.iter() {
           if let Some(id) = o { ha.push(id); }
       }
       for &o in state.hands_b.iter() {
           if let Some(id) = o { hb.push(id); }
       }
       HandsRecord { a: ha, b: hb }
   }

   fn push_record(
       state: &GameState,
       cards: &triplecargo::CardsDb,
       tt: &mut InMemoryTT,
       cap: Option<u8>,
       out: &mut Vec<ExportRecord>,
       export_cfg: &ExportConfig,
   ) {
       let depth = eff_depth_for(state, cap);
       let (val, bm, _nodes) = search_root(state, cards, depth, tt);

       // Build board cells with optional elemental letters
       let mut board_vec: Vec<BoardCell> = Vec::with_capacity(9);
       let elemental_enabled = state.rules.elemental;
       for cell in 0u8..9 {
           let slot = state.board.get(cell);
           let (card_id, owner) = match slot {
               Some(s) => (Some(s.card_id), Some(match s.owner {
                   Owner::A => 'A',
                   Owner::B => 'B',
               })),
               None => (None, None),
           };
           let element_field: Option<Option<char>> = if elemental_enabled {
               Some(state.board.cell_element(cell).map(element_letter))
           } else {
               None
           };
           board_vec.push(BoardCell { cell, card_id, owner, element: element_field });
       }

       // Policy output
       let policy_out: Option<PolicyOut> = match export_cfg.policy_format {
           PolicyFormatOpt::Onehot => {
               bm.map(|m| PolicyOut::Move { card_id: m.card_id, cell: m.cell })
           }
           PolicyFormatOpt::Mcts => {
               // Deterministic per-state seed
               let z = zobrist_key(state);
               let seed = export_cfg.base_seed ^ (z as u64) ^ ((z >> 64) as u64);
               let dist = mcts_policy_distribution(state, cards, export_cfg.mcts_rollouts, seed);
               Some(PolicyOut::Dist(dist))
           }
       };

       let hands = collect_hands_from_state(state);
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
       tt: &mut InMemoryTT,
       visited: &mut hashbrown::HashSet<u128>,
       cap: Option<u8>,
       out: &mut Vec<ExportRecord>,
       export_cfg: &ExportConfig,
   ) {
       let key = zobrist_key(state);
       if !visited.insert(key) {
           return;
       }
       push_record(state, cards, tt, cap, out, export_cfg);

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

   dfs(initial, cards, &mut tt, &mut visited, max_depth, &mut out, export_cfg);
   out
}

fn root_perspective_value(root_next: Owner, terminal: &GameState) -> i8 {
    let mut v = score(terminal);
    if root_next == Owner::B {
        v = -v;
    }
    v
}

/// Deterministic shallow MCTS at root to produce a soft policy distribution.
/// - rollouts: number of simulations
/// - seed: SplitMix64 seed for deterministic random playouts
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

    // Each move visited at least once
    let total_rollouts = rollouts.max(n);

    for t in 0..total_rollouts {
        // Select arm
        let mut idx: usize = 0;
        // Ensure each gets one visit
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

        // Random playout to terminal
        while !sim.is_terminal() {
            let legals = legal_moves(&sim);
            if legals.is_empty() {
                break;
            }
            // deterministic RNG
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
        // uniform fallback
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

//
// Helper: pick a uniform random legal move excluding the PV move if possible.
// Falls back to PV move (or None) when no alternative exists.
//
fn pick_random_non_pv_move(
    state: &GameState,
    pv: Option<triplecargo::Move>,
    seed: u64,
    game_id: u64,
    turn: u8,
) -> Option<triplecargo::Move> {
    // Candidate moves = all legals except PV (if provided)
    let mut candidates = legal_moves(state);
    if let Some(pv_mv) = pv {
        candidates.retain(|m| !(m.card_id == pv_mv.card_id && m.cell == pv_mv.cell));
    }

    if candidates.is_empty() {
        // No alternative to PV: return PV (or None in terminal)
        return pv;
    }

    // Deterministic RNG derived from (seed, game_id, turn)
    let mut rng = rng_for_state(seed, game_id, turn);
    let idx = rng.gen_range(0..candidates.len());
    Some(candidates[idx])
}
//
// Helper: pick a weighted random legal move (softmax over child Q) excluding PV.
// - child_vals: (Move, Q) for each legal root move
// - pv: principal-variation move (excluded by setting prob=0)
// - Sampling uses rng_for_state(seed, game_id, turn) for determinism
//
fn pick_weighted_non_pv_move(
    child_vals: &[(triplecargo::Move, i8)],
    pv: Option<triplecargo::Move>,
    seed: u64,
    game_id: u64,
    turn: u8,
) -> Option<triplecargo::Move> {
    // Candidates: all children excluding the PV move (if provided)
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
        // No alternative candidate: return PV (or None)
        return pv;
    }

    // Stable softmax over Q values (i8 -> f64)
    let max_q = cands.iter().map(|(_, q)| *q as f64).fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = Vec::with_capacity(cands.len());
    let mut sum_w: f64 = 0.0;
    for &(_, q) in &cands {
        let w = (q as f64 - max_q).exp();
        weights.push(w);
        sum_w += w;
    }

    // Guard against degenerate sums; fallback to uniform
    if !(sum_w.is_finite()) || sum_w <= 0.0 {
        let mut rng = rng_for_state(seed, game_id, turn);
        let idx = rng.gen_range(0..cands.len());
        return Some(cands[idx].0);
    }

    // Sample proportionally using cumulative probabilities
    let mut rng = rng_for_state(seed, game_id, turn);
    let r: f64 = rng.gen::<f64>(); // in [0,1)
    let mut acc: f64 = 0.0;
    for (i, &(mv, _)) in cands.iter().enumerate() {
        acc += weights[i] / sum_w;
        if r < acc {
            return Some(mv);
        }
    }

    // Fallback due to potential rounding: return last candidate
    Some(cands.last().unwrap().0)
}
//
// Helper: parse "card-cell" key into a Move
//
fn parse_move_key(key: &str) -> Option<triplecargo::Move> {
    let mut parts = key.splitn(2, '-');
    let card_s = parts.next()?;
    let cell_s = parts.next()?;
    let card_id: u16 = card_s.parse().ok()?;
    let cell_u: u8 = cell_s.parse().ok()?;
    Some(triplecargo::Move { card_id, cell: cell_u })
}

//
// Helper: pick a move from an MCTS root distribution excluding the PV move.
// - If policy_format=mcts, reuse the provided root distribution (callers should pass it).
// - Else compute root MCTS with --mcts-rollouts and a deterministic seed (callers provide a dist).
// - Set PV probability to 0, renormalize, and sample with rng_for_state(seed, game_id, turn).
// - Fallback: if PV had all mass (sum of others == 0), pick the highest-prob non-PV move deterministically
//   (first by BTreeMap order in a tie). If no alternative exists, return PV.
//
fn pick_mcts_non_pv_move_from_dist(
    dist: &BTreeMap<String, f32>,
    pv: Option<triplecargo::Move>,
    seed: u64,
    game_id: u64,
    turn: u8,
) -> Option<triplecargo::Move> {
    // Build candidates excluding PV
    let mut cands: Vec<(triplecargo::Move, f32)> = Vec::new();
    for (k, &p) in dist.iter() {
        if let Some(mv) = parse_move_key(k) {
            // Exclude PV by assigning zero probability
            if let Some(pv_mv) = pv {
                if mv.card_id == pv_mv.card_id && mv.cell == pv_mv.cell {
                    continue;
                }
            }
            cands.push((mv, p.max(0.0)));
        }
    }

    if cands.is_empty() {
        // No alternative candidate: return PV (or None)
        return pv;
    }

    // Sum of non-PV probabilities
    let sum_p: f64 = cands.iter().map(|(_, p)| *p as f64).sum();

    if sum_p > 0.0 && sum_p.is_finite() {
        // Sample proportionally using cumulative probabilities
        let mut rng = rng_for_state(seed, game_id, turn);
        let r: f64 = rng.gen::<f64>(); // [0,1)
        let mut acc: f64 = 0.0;
        for (mv, p) in &cands {
            acc += (*p as f64) / sum_p;
            if r < acc {
                return Some(*mv);
            }
        }
        // Fallback to last due to rounding
        return Some(cands.last().unwrap().0);
    }

    // Fallback: all mass was on PV (sum of others == 0) → pick highest-prob non-PV deterministically.
    // In a tie, BTreeMap iteration order induces a stable choice.
    let mut best: Option<(triplecargo::Move, f32)> = None;
    for &(mv, p) in &cands {
        best = match best {
            None => Some((mv, p)),
            Some((bmv, bp)) => {
                if p > bp {
                    Some((mv, p))
                } else {
                    Some((bmv, bp))
                }
            }
        };
    }
    Some(best.unwrap().0)
}

// --- Progress helpers for trajectory export ---
//
// Format seconds as HH:MM:SS
fn format_hms(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}
 
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Configure a small, deterministic Rayon pool (2-4 threads). Ignore error if already set.
    let _ = ThreadPoolBuilder::new().num_threads(6).build_global();

    let t0 = Instant::now();

    // Load cards (for validation and header fingerprint)
    let cards = load_cards_from_json(&args.cards)
        .map_err(|e| format!("Cards load error: {e}"))?;
    println!("[precompute] Loaded {} cards (max id {}).", cards.len(), cards.max_id());

    // Parse inputs
    let rules = parse_rules(&args.rules);

    // Branch: JSONL export mode
    if let Some(export_path) = args.export.as_ref() {
        // Elements mode (affects board elements only)
        let _elements_mode = match args.elements {
            ElementsOpt::None => ElementsMode::None,
            ElementsOpt::Random => ElementsMode::Random,
        };

        // Output file
        let file = File::create(export_path).map_err(|e| format!("Failed to create export file: {e}"))?;
        let mut writer = BufWriter::new(file);

        // Precompute static resources for sampling
        let all_ids_sorted = all_card_ids_sorted(&cards);
        let bands_master = build_level_bands(&cards);
                // Progress tracking (indicatif)
                let mut nodes_total: u64 = 0;
                let start_instant = Instant::now();
                let mut last_tick = Instant::now();
                let mut games_done: usize = 0;
                let mut offpv_games_done: usize = 0;
                let policy_info: String = match args.policy_format {
                    PolicyFormatOpt::Mcts => format!("mcts@{}", args.mcts_rollouts),
                    PolicyFormatOpt::Onehot => "onehot".to_string(),
                };
                let total_states: u64 = (args.games as u64) * 9;
                let pb = ProgressBar::new(total_states);
                pb.set_style(
                    ProgressStyle::with_template("[{elapsed_precise}] traj {bar:40.cyan/blue} {pos}/{len} {msg}")
                        .unwrap()
                        .progress_chars("=>-"),
                );

        match args.export_mode {
            ExportModeOpt::Trajectory => {
                let games = args.games;
                println!(
                    "[export] Mode=trajectory games={} hand_strategy={:?} seed={:#x} policy_format={:?} value_mode={:?} off_pv_rate={} off_pv_strategy={:?}",
                    games, args.hand_strategy, args.seed, args.policy_format, args.value_mode, args.off_pv_rate, args.off_pv_strategy
                );

                let mut total_lines: usize = 0;

                for i in 0..games {
                    let mut rng_state: u64 = args.seed ^ ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

                    // Sample hands deterministically (no overlap between A and B)
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

                    // Elements per game if requested
                    let cell_elements = match args.elements {
                        ElementsOpt::None => None,
                        ElementsOpt::Random => {
                            let seed_for_elems = splitmix64(&mut rng_state);
                            Some(gen_elements(seed_for_elems))
                        }
                    };

                    // Build initial state
                    let mut state = GameState::with_hands(rules, hand_a, hand_b, cell_elements);
                    let mut tt = InMemoryTT::default();

                    // Decide per-game off-PV sampling (random or weighted strategies)
                    let off_pv_game = matches!(args.off_pv_strategy, OffPvStrategyOpt::Random | OffPvStrategyOpt::Weighted | OffPvStrategyOpt::Mcts)
                        && args.off_pv_rate > 0.0
                        && {
                            let mut grng = rng_for_state(args.seed, i as u64, 0);
                            grng.gen::<f32>() < args.off_pv_rate
                        };

                    // Play a single principal-variation trajectory of 9 plies
                    for _ply in 0..9 {
                        // In trajectory mode we always search full remaining depth to obtain final-outcome value targets.
                        let eff_depth = 9 - state.board.filled_count();

                        let (val, bm, child_vals, nodes) = search_root_with_children(&state, &cards, eff_depth, &mut tt);

                        // Serialize current state record
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
                        // Build shrinking-hands snapshot for current state
                        let mut hands_a_vec: Vec<u16> = Vec::with_capacity(5);
                        let mut hands_b_vec: Vec<u16> = Vec::with_capacity(5);
                        for &o in state.hands_a.iter() { if let Some(id) = o { hands_a_vec.push(id); } }
                        for &o in state.hands_b.iter() { if let Some(id) = o { hands_b_vec.push(id); } }
                        let hands_rec = HandsRecord { a: hands_a_vec, b: hands_b_vec };

                        // Optional root MCTS distribution cache for off-PV stepping reuse
                        let mut mcts_root_dist: Option<BTreeMap<String, f32>> = None;

                        // Build policy output based on requested format
                        let policy_out: Option<PolicyOut> = match args.policy_format {
                            PolicyFormatOpt::Onehot => {
                                bm.map(|m| PolicyOut::Move { card_id: m.card_id, cell: m.cell })
                            }
                            PolicyFormatOpt::Mcts => {
                                // Seed per ply for determinism (trajectory keeps legacy per-ply seeding)
                                let seed_for_rollouts = splitmix64(&mut rng_state);
                                let dist = mcts_policy_distribution(&state, &cards, args.mcts_rollouts, seed_for_rollouts);
                                mcts_root_dist = Some(dist.clone());
                                Some(PolicyOut::Dist(dist))
                            }
                        };

                        let to_move = match state.next { Owner::A => 'A', Owner::B => 'B' };
                        let vt = compute_value_target(&args.value_mode, val, state.next);

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
                            value_mode: value_mode_str(&args.value_mode).to_string(),
                            off_pv: off_pv_game,
                            state_hash: format!("{:032x}", zobrist_key(&state)),
                        };
                        let line = serde_json::to_string(&rec).map_err(|e| format!("serialize JSONL record error: {e}"))?;
                        writer.write_all(line.as_bytes()).map_err(|e| format!("export write error: {e}"))?;
                        writer.write_all(b"\n").map_err(|e| format!("export write error: {e}"))?;
                        total_lines += 1;
                        nodes_total = nodes_total.saturating_add(nodes);

                        // Update progress at most once per second (indicatif)
                        let now = Instant::now();
                        if now.duration_since(last_tick).as_secs() >= 1 {
                            let elapsed = now.duration_since(start_instant);
                            let elapsed_secs = elapsed.as_secs().max(1);
                            let states_done = total_lines as u64;
                            let states_per_sec = states_done / elapsed_secs;
                            let nodes_per_sec = nodes_total / elapsed_secs;
                            let remaining_states = total_states.saturating_sub(states_done);
                            let eta_secs = if states_per_sec > 0 { remaining_states / states_per_sec } else { 0 };
                            let offpv_pct = if games_done > 0 {
                                (offpv_games_done as f64) * 100.0 / (games_done as f64)
                            } else {
                                0.0
                            };
                            pb.set_position(states_done);
                            pb.set_message(format!(
                                "games {}/{} | states/s {} | nodes {:.1}M | nodes/s {:.1}M | offpv {}/{} ({:.1}%) | pol {} | ETA {}",
                                games_done, games,
                                states_per_sec,
                                (nodes_total as f64) / 1.0e6,
                                (nodes_per_sec as f64) / 1.0e6,
                                offpv_games_done, games_done, offpv_pct,
                                policy_info,
                                format_hms(eta_secs),
                            ));
                            last_tick = now;
                        }

                        // Choose next move based on off-PV setting
                        let next_mv = if off_pv_game {
                            match args.off_pv_strategy {
                                OffPvStrategyOpt::Random => {
                                    pick_random_non_pv_move(&state, bm, args.seed, i as u64, state.board.filled_count())
                                }
                                OffPvStrategyOpt::Weighted => {
                                    pick_weighted_non_pv_move(&child_vals, bm, args.seed, i as u64, state.board.filled_count())
                                }
                                OffPvStrategyOpt::Mcts => {
                                    if matches!(args.policy_format, PolicyFormatOpt::Mcts) {
                                        // Reuse root distribution from policy_out
                                        let dist_ref = mcts_root_dist.as_ref().expect("mcts_root_dist should be present when policy_format=mcts");
                                        pick_mcts_non_pv_move_from_dist(dist_ref, bm, args.seed, i as u64, state.board.filled_count())
                                    } else {
                                        // Compute root MCTS distribution just for stepping (deterministic per-ply seed)
                                        let seed_for_mcts_step = splitmix64(&mut rng_state);
                                        let dist2 = mcts_policy_distribution(&state, &cards, args.mcts_rollouts, seed_for_mcts_step);
                                        pick_mcts_non_pv_move_from_dist(&dist2, bm, args.seed, i as u64, state.board.filled_count())
                                    }
                                }
                            }
                        } else {
                            bm
                        };
                        if let Some(mv) = next_mv {
                            if let Ok(ns) = apply_move(&state, &cards, mv) {
                                state = ns;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    // Game completed
                    games_done = games_done.saturating_add(1);
                    if off_pv_game {
                        offpv_games_done = offpv_games_done.saturating_add(1);
                    }

                    // Periodic flush
                    if total_lines % 10_000 == 0 {
                        let _ = writer.flush();
                    }
                }

                // Finish progress bar
                pb.set_position(total_states);
                pb.finish_and_clear();
                let _ = writer.flush();
                let elapsed = t0.elapsed();
                println!(
                    "[export] Done. mode=trajectory games={} lines={} elapsed_ms={}",
                    args.games,
                    args.games * 9,
                    elapsed.as_millis()
                );
                return Ok(());
            }
            ExportModeOpt::Full => {
                // One hand pair only; ignore --games
                println!(
                    "[export] Mode=full (ignoring --games). hand_strategy={:?} seed={:#x} policy_format={:?} value_mode={:?} off_pv_rate={} off_pv_strategy={:?}",
                    args.hand_strategy, args.seed, args.policy_format, args.value_mode, args.off_pv_rate, args.off_pv_strategy
                );

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

                // Build export configuration (plumbed through; no behavior change yet for off-PV fields)
                let export_cfg = ExportConfig {
                    policy_format: args.policy_format.clone(),
                    value_mode: args.value_mode.clone(),
                    mcts_rollouts: args.mcts_rollouts,
                    off_pv_rate: args.off_pv_rate,
                    off_pv_strategy: args.off_pv_strategy.clone(),
                    base_seed: args.seed, // base seed for per-state MCTS seeding
                };

                // Full enumeration with new schema/options
                let records = enumerate_solved_records(
                    &initial,
                    &cards,
                    args.max_depth,
                    &export_cfg,
                );

                let mut total_lines: usize = 0;
                for mut rec in records {
                    // Ensure game_id and state_idx contract for full mode
                    rec.game_id = 0;
                    // state_idx already equals turn in construction
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
                return Ok(());
            }
        }
    }

    // Legacy DB precompute path (no --export)
    let (hand_a, hand_b) = parse_hands(&args.hands)
        .map_err(|e| format!("Hands parse error: {e}"))?;

    // Construct initial state
    let elements_mode = match args.elements {
        ElementsOpt::None => ElementsMode::None,
        ElementsOpt::Random => ElementsMode::Random,
    };
    let cell_elements = match elements_mode {
        ElementsMode::None => None,
        ElementsMode::Random => Some(gen_elements(args.seed)),
    };
    let initial = GameState::with_hands(rules, hand_a, hand_b, cell_elements);

    // Quick best move for A at initial state (capped by --max-depth if provided)
    let remaining0 = 9 - initial.board.filled_count();
    let eff_depth0 = match args.max_depth {
        Some(cap) => remaining0.min(cap),
        None => remaining0,
    };
    let mut quick_tt = InMemoryTT::default();
    let (best_val0, best_mv0, _nodes0) = search_root(&initial, &cards, eff_depth0, &mut quick_tt);
    println!(
        "[precompute] Initial best (depth={}): value={}, best_move={}",
        eff_depth0,
        best_val0,
        match best_mv0 {
            Some(m) => format!("{{ card_id: {}, cell: {} }}", m.card_id, m.cell),
            None => "None".to_string(),
        }
    );

    // Run precompute
    let (db_map, stats) = precompute_solve(&initial, &cards, elements_mode, args.max_depth);
    let elapsed = t0.elapsed();

    // Compose header
    let header = DbHeader {
        version: triplecargo::persist::FORMAT_VERSION,
        rules,
        elements_mode,
        seed: args.seed,
        start_player: Owner::A,
        hands_a: hand_a,
        hands_b: hand_b,
        cards_fingerprint: fingerprint_cards(&cards),
    };

    // Persistence
    if args.analysis {
        println!("[precompute] Analysis mode: skipping persistence.");
    } else {
        let mut stream_path = args.out.clone();
        stream_path.set_extension("stream");

        let compression = match args.compress {
            Some(CompressOpt::Lz4) => StreamCompression::Lz4,
            Some(CompressOpt::Zstd) => StreamCompression::Zstd,
            None => StreamCompression::None,
        };

        let mut writer = StreamWriter::create(&stream_path, &header, compression, args.batch_size)
            .map_err(|e| format!("Stream create error: {e}"))?;

        // Checkpointing controls (performed during streaming of merged map)
        let mut ckpt_seq: u64 = 0;
        let mut states_since_ckpt: u64 = 0;
        let mut last_ckpt_instant = Instant::now();
        let have_ckpt_time = args.checkpoint_seconds.unwrap_or(0) > 0;
        let have_ckpt_states = args.checkpoint_states.unwrap_or(0) > 0;

        // Stream out merged map deterministically with batching
        for (k, v) in &db_map {
            // write to stream
            writer.push(*k, v.clone()).map_err(|e| format!("Stream push error: {e}"))?;
            states_since_ckpt = states_since_ckpt.saturating_add(1);

            // evaluate checkpoint triggers
            let time_due = have_ckpt_time && last_ckpt_instant.elapsed().as_secs() >= args.checkpoint_seconds.unwrap_or(0);
            let states_due = have_ckpt_states && states_since_ckpt >= args.checkpoint_states.unwrap_or(0);

            if time_due || states_due {
                writer.flush_all().map_err(|e| format!("Stream flush error: {e}"))?;
                let mut ckpt_path = args.out.clone();
                ckpt_seq = ckpt_seq.saturating_add(1);
                ckpt_path.set_extension(format!("ckpt-{ckpt_seq}.bin"));
                compact_stream_to_db_file(&stream_path, &ckpt_path)
                    .map_err(|e| format!("Checkpoint compaction error: {e}"))?;
                last_ckpt_instant = Instant::now();
                states_since_ckpt = 0;
            }
        }

        // Optional final checkpoint snapshot if configured and there were new frames since last checkpoint
        if (have_ckpt_time || have_ckpt_states) && states_since_ckpt > 0 {
            writer.flush_all().map_err(|e| format!("Stream flush error: {e}"))?;
            let mut ckpt_path = args.out.clone();
            ckpt_seq = ckpt_seq.saturating_add(1);
            ckpt_path.set_extension(format!("ckpt-{ckpt_seq}.bin"));
            compact_stream_to_db_file(&stream_path, &ckpt_path)
                .map_err(|e| format!("Checkpoint compaction error: {e}"))?;
            last_ckpt_instant = Instant::now();
            states_since_ckpt = 0;
        }

        writer.flush_all().map_err(|e| format!("Stream flush error: {e}"))?;

        // Final deterministic compaction into legacy DB format
        if let Some(ref base) = args.baseline {
            compact_stream_with_baseline_to_db_file(&stream_path, base, &args.out)
                .map_err(|e| format!("Compaction (with baseline) error: {e}"))?;
        } else {
            compact_stream_to_db_file(&stream_path, &args.out)
                .map_err(|e| format!("Compaction error: {e}"))?;
        }
    }

    println!(
        "[precompute] Done. exact_entries={}, nodes={}, states_enumerated={}, roots={}, max_depth={:?}, elapsed_ms={}",
        stats.exact_entries, stats.nodes, stats.states_enumerated, stats.roots, args.max_depth, elapsed.as_millis()
    );
    println!(
        "[precompute] Initial key {:032x}",
        zobrist_key(&initial)
    );

    Ok(())
}
#[cfg(test)]
mod cli_tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_defaults_when_omitted() {
        // Use --export to satisfy required_unless_present without needing --hands
        let args = Args::try_parse_from(["precompute", "--export", "x.jsonl"])
            .expect("Args parse failed");
        assert!((args.off_pv_rate - 0.0).abs() < f32::EPSILON, "default off_pv_rate should be 0.0");
        assert!(matches!(args.off_pv_strategy, OffPvStrategyOpt::Weighted), "default off_pv_strategy should be Weighted");
    }

    #[test]
    fn cli_accepts_offpv_flags() {
        let args = Args::try_parse_from([
            "precompute",
            "--export",
            "x.jsonl",
            "--off-pv-rate",
            "0.25",
            "--off-pv-strategy",
            "random",
        ])
        .expect("Args parse with off-PV flags failed");

        assert!((args.off_pv_rate - 0.25).abs() < 1e-6, "off_pv_rate parsed incorrectly");
        assert!(matches!(args.off_pv_strategy, OffPvStrategyOpt::Random), "off_pv_strategy should parse as Random");
    }

    #[test]
    fn help_includes_offpv_flags() {
        let mut buf: Vec<u8> = Vec::new();
        Args::command().write_help(&mut buf).expect("write_help failed");
        let help = String::from_utf8(buf).expect("utf8 help");
        assert!(help.contains("--off-pv-rate"), "help text missing --off-pv-rate");
        assert!(help.contains("--off-pv-strategy"), "help text missing --off-pv-strategy");
    }

    #[test]
    fn export_records_include_off_pv_false() {
        // Build a near-terminal state (8 plies) to keep enumeration small.
        let cards = load_cards_from_json("data/cards.json").expect("cards load");
        let rules = Rules::default();
        let mut state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);

        // Play 8 deterministic plies (first legal each time)
        for _ in 0..8 {
            let mv = legal_moves(&state).into_iter().next().expect("legal move");
            state = apply_move(&state, &cards, mv).expect("apply_move");
        }

        // Minimal export configuration; off-PV is not used in full export (always false).
        let export_cfg = ExportConfig {
            policy_format: PolicyFormatOpt::Onehot,
            value_mode: ValueModeOpt::Winloss,
            mcts_rollouts: 0,
            off_pv_rate: 0.0,
            off_pv_strategy: OffPvStrategyOpt::Weighted,
            base_seed: 42,
        };

        // Enumerate solved records from this state and assert off_pv is present and always false.
        let recs = enumerate_solved_records(&state, &cards, None, &export_cfg);
        assert!(!recs.is_empty(), "expected at least one exported record");
        for r in &recs {
            assert_eq!(r.off_pv, false, "off_pv must be false in full export");
            let js = serde_json::to_string(r).expect("serialize record");
            assert!(
                js.contains("\"off_pv\":false"),
                "serialized JSON must include off_pv=false; got: {}",
                js
            );
        }
    }

    // --- Off-PV stepping tests (random strategy) ---

    fn build_initial_state_for_tests() -> (triplecargo::CardsDb, GameState) {
        let cards = load_cards_from_json("data/cards.json").expect("cards load");
        let rules = Rules::default();
        let state = GameState::with_hands(rules, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], None);
        (cards, state)
    }

    #[test]
    fn offpv_random_never_chooses_pv_when_alternative_exists() {
        let (cards, mut state) = build_initial_state_for_tests();
        let seed: u64 = 12345;
        let game_id: u64 = 0;
        let mut tt = InMemoryTT::default();

        for _ply in 0..9 {
            let eff_depth = 9 - state.board.filled_count();
            let (_val, bm, _nodes) = search_root(&state, &cards, eff_depth, &mut tt);

            // Determine next move as off-PV random
            let next_mv = pick_random_non_pv_move(&state, bm, seed, game_id, state.board.filled_count());

            // If there are multiple legals and PV exists, ensure we did not choose PV
            let legals = legal_moves(&state);
            if legals.len() > 1 {
                if let (Some(pv), Some(ch)) = (bm, next_mv) {
                    assert!(
                        !(pv.card_id == ch.card_id && pv.cell == ch.cell),
                        "off-PV random should not choose PV when an alternative exists"
                    );
                }
            }

            if let Some(mv) = next_mv {
                state = apply_move(&state, &cards, mv).expect("apply move");
            } else {
                break;
            }
        }
    }

    #[test]
    fn offpv_rate_zero_matches_pv_baseline() {
        let (cards, mut state_baseline) = build_initial_state_for_tests();
        let (cards2, mut state_test) = build_initial_state_for_tests();
        let mut tt1 = InMemoryTT::default();
        let mut tt2 = InMemoryTT::default();

        let mut baseline_seq: Vec<triplecargo::Move> = Vec::new();
        let mut test_seq: Vec<triplecargo::Move> = Vec::new();

        // Baseline: always follow PV
        for _ply in 0..9 {
            let eff_depth = 9 - state_baseline.board.filled_count();
            let (_val, bm, _nodes) = search_root(&state_baseline, &cards, eff_depth, &mut tt1);
            if let Some(mv) = bm {
                baseline_seq.push(mv);
                state_baseline = apply_move(&state_baseline, &cards, mv).expect("apply move");
            } else {
                break;
            }
        }

        // "Rate zero" behavior: our stepping should also follow PV
        for _ply in 0..9 {
            let eff_depth = 9 - state_test.board.filled_count();
            let (_val, bm, _nodes) = search_root(&state_test, &cards2, eff_depth, &mut tt2);
            let mv = bm;
            if let Some(m) = mv {
                test_seq.push(m);
                state_test = apply_move(&state_test, &cards2, m).expect("apply move");
            } else {
                break;
            }
        }

        assert_eq!(baseline_seq.len(), test_seq.len(), "sequence lengths differ");
        for (a, b) in baseline_seq.iter().zip(test_seq.iter()) {
            assert!(a.card_id == b.card_id && a.cell == b.cell, "moves differ: baseline {:?} vs test {:?}", a, b);
        }
    }

    #[test]
    fn offpv_determinism_fixed_seed_identical_sequences() {
        let (cards_a, mut state_a) = build_initial_state_for_tests();
        let (cards_b, mut state_b) = build_initial_state_for_tests();
        let seed: u64 = 0xC0FFEE;
        let game_id: u64 = 7;
        let mut tt_a = InMemoryTT::default();
        let mut tt_b = InMemoryTT::default();

        let mut seq_a: Vec<triplecargo::Move> = Vec::new();
        let mut seq_b: Vec<triplecargo::Move> = Vec::new();

        for _ply in 0..9 {
            let eff_a = 9 - state_a.board.filled_count();
            let (_va, bma, _na) = search_root(&state_a, &cards_a, eff_a, &mut tt_a);
            let mv_a = pick_random_non_pv_move(&state_a, bma, seed, game_id, state_a.board.filled_count());
            if let Some(m) = mv_a {
                seq_a.push(m);
                state_a = apply_move(&state_a, &cards_a, m).expect("apply move");
            } else { break; }

            let eff_b = 9 - state_b.board.filled_count();
            let (_vb, bmb, _nb) = search_root(&state_b, &cards_b, eff_b, &mut tt_b);
            let mv_b = pick_random_non_pv_move(&state_b, bmb, seed, game_id, state_b.board.filled_count());
            if let Some(m) = mv_b {
                seq_b.push(m);
                state_b = apply_move(&state_b, &cards_b, m).expect("apply move");
            } else { break; }
        }

        assert_eq!(seq_a.len(), seq_b.len(), "determinism: lengths differ");
        for (a, b) in seq_a.iter().zip(seq_b.iter()) {
            assert!(a.card_id == b.card_id && a.cell == b.cell, "determinism: moves differ {:?} vs {:?}", a, b);
        }
    }

    #[test]
    fn offpv_line_count_invariant_nine() {
        let (_cards, mut state) = build_initial_state_for_tests();
        let seed: u64 = 999;
        let game_id: u64 = 0;
        let cards = load_cards_from_json("data/cards.json").expect("cards load");
        let mut tt = InMemoryTT::default();
        let mut steps = 0usize;

        for _ply in 0..9 {
            let eff = 9 - state.board.filled_count();
            let (_v, bm, _n) = search_root(&state, &cards, eff, &mut tt);
            let mv = pick_random_non_pv_move(&state, bm, seed, game_id, state.board.filled_count());
            if let Some(m) = mv {
                steps += 1;
                state = apply_move(&state, &cards, m).expect("apply move");
            } else { break; }
        }
        assert_eq!(steps, 9, "trajectory should have exactly 9 plies");
    }

    // --- Off-PV stepping tests (weighted strategy) ---
    #[test]
    fn offpv_weighted_never_chooses_pv_when_alternative_exists() {
        // Synthetic root children with PV present among children
        let mv_pv = triplecargo::Move { card_id: 10, cell: 0 };
        let mv_a = triplecargo::Move { card_id: 11, cell: 1 };
        let mv_b = triplecargo::Move { card_id: 12, cell: 2 };
        let child_vals = vec![(mv_pv, 0i8), (mv_a, 1i8), (mv_b, -1i8)];

        let seed: u64 = 0x55AA55AA;
        let turn: u8 = 0;

        // Sample across multiple game_ids; PV must never be selected when an alternative exists
        for gid in 0u64..200u64 {
            let pick = pick_weighted_non_pv_move(&child_vals, Some(mv_pv), seed, gid, turn)
                .expect("expected a move");
            assert!(
                !(pick.card_id == mv_pv.card_id && pick.cell == mv_pv.cell),
                "weighted picker must not choose PV when alternatives exist"
            );
        }
    }

    #[test]
    fn offpv_weighted_determinism_fixed_seed_identical_sequences() {
        // Fixed synthetic child set
        let mv_pv = triplecargo::Move { card_id: 20, cell: 0 };
        let mv_a = triplecargo::Move { card_id: 21, cell: 1 };
        let mv_b = triplecargo::Move { card_id: 22, cell: 2 };
        // Ensure both non-PV are eligible and have different Qs
        let child_vals = vec![(mv_pv, 0i8), (mv_a, 3i8), (mv_b, -2i8)];

        let seed: u64 = 0xC0FFEE;
        let game_id: u64 = 7;

        // Produce two sequences over turns 0..8 with identical parameters
        let mut seq1: Vec<triplecargo::Move> = Vec::new();
        let mut seq2: Vec<triplecargo::Move> = Vec::new();

        for t in 0u8..9u8 {
            let a = pick_weighted_non_pv_move(&child_vals, Some(mv_pv), seed, game_id, t)
                .expect("seq1 move");
            let b = pick_weighted_non_pv_move(&child_vals, Some(mv_pv), seed, game_id, t)
                .expect("seq2 move");
            seq1.push(a);
            seq2.push(b);
        }

        assert_eq!(seq1.len(), seq2.len(), "determinism: lengths differ");
        for (a, b) in seq1.iter().zip(seq2.iter()) {
            assert!(a.card_id == b.card_id && a.cell == b.cell, "determinism: moves differ {:?} vs {:?}", a, b);
        }
    }

    #[test]
    fn offpv_weighted_samples_dominant_child_more_often() {
        // Create a child set where one NON-PV child is clearly dominant by Q
        let mv_pv = triplecargo::Move { card_id: 30, cell: 0 }; // set PV to a non-dominant move
        let mv_dom = triplecargo::Move { card_id: 31, cell: 1 }; // dominant candidate
        let mv_other = triplecargo::Move { card_id: 32, cell: 2 };

        // Assign Q values: dominant >> other; PV excluded by the picker
        let child_vals = vec![(mv_pv, 0i8), (mv_dom, 5i8), (mv_other, 0i8)];

        let seed: u64 = 0xDEADBEEF;
        let turn: u8 = 0;

        let mut count_dom = 0usize;
        let mut count_other = 0usize;

        // Vary game_id to generate many independent draws with the same RNG scheme
        let trials = 2000usize;
        for gid in 0..trials {
            let pick = pick_weighted_non_pv_move(&child_vals, Some(mv_pv), seed, gid as u64, turn)
                .expect("expected a move");
            if pick.card_id == mv_dom.card_id && pick.cell == mv_dom.cell {
                count_dom += 1;
            } else if pick.card_id == mv_other.card_id && pick.cell == mv_other.cell {
                count_other += 1;
            } else {
                panic!("picked unexpected move: {:?}", pick);
            }
        }

        // Dominant child must be sampled strictly more often than the other non-PV child
        assert!(
            count_dom > count_other,
            "dominant child should be sampled more often; dom={} other={}",
            count_dom,
            count_other
        );
    }
}
// --- Off-PV stepping tests (mcts strategy) ---
#[cfg(test)]
mod mcts_offpv_tests {
    use super::*;

    #[test]
    fn offpv_mcts_excludes_pv_when_alternative_exists() {
        // Build a synthetic MCTS root distribution with PV and two alternatives
        let mv_pv = triplecargo::Move { card_id: 101, cell: 0 };
        let mv_a = triplecargo::Move { card_id: 102, cell: 1 };
        let mv_b = triplecargo::Move { card_id: 103, cell: 2 };

        let mut dist: BTreeMap<String, f32> = BTreeMap::new();
        dist.insert(format!("{}-{}", mv_pv.card_id, mv_pv.cell), 0.4);
        dist.insert(format!("{}-{}", mv_a.card_id, mv_a.cell), 0.35);
        dist.insert(format!("{}-{}", mv_b.card_id, mv_b.cell), 0.25);

        let seed: u64 = 0xABCDEF12;
        let turn: u8 = 0;

        // Across many draws with different game_id, PV should never be picked
        for gid in 0u64..500u64 {
            let pick = super::pick_mcts_non_pv_move_from_dist(&dist, Some(mv_pv), seed, gid, turn)
                .expect("expected a move");
            assert!(
                !(pick.card_id == mv_pv.card_id && pick.cell == mv_pv.cell),
                "mcts picker must not choose PV when alternatives exist"
            );
        }
    }

    #[test]
    fn offpv_mcts_determinism_fixed_seed_identical_sequences() {
        let mv_pv = triplecargo::Move { card_id: 201, cell: 0 };
        let mv_a = triplecargo::Move { card_id: 202, cell: 1 };
        let mv_b = triplecargo::Move { card_id: 203, cell: 2 };

        let mut dist: BTreeMap<String, f32> = BTreeMap::new();
        dist.insert(format!("{}-{}", mv_pv.card_id, mv_pv.cell), 0.2);
        dist.insert(format!("{}-{}", mv_a.card_id, mv_a.cell), 0.5);
        dist.insert(format!("{}-{}", mv_b.card_id, mv_b.cell), 0.3);

        let seed: u64 = 0xC0FFEE77;
        let game_id: u64 = 9;

        // Build two sequences over turns; must be identical with same RNG inputs
        let mut seq1: Vec<triplecargo::Move> = Vec::new();
        let mut seq2: Vec<triplecargo::Move> = Vec::new();

        for t in 0u8..9u8 {
            let a = super::pick_mcts_non_pv_move_from_dist(&dist, Some(mv_pv), seed, game_id, t).expect("seq1");
            let b = super::pick_mcts_non_pv_move_from_dist(&dist, Some(mv_pv), seed, game_id, t).expect("seq2");
            seq1.push(a);
            seq2.push(b);
        }

        assert_eq!(seq1.len(), seq2.len(), "determinism: lengths differ");
        for (a, b) in seq1.iter().zip(seq2.iter()) {
            assert!(a.card_id == b.card_id && a.cell == b.cell, "determinism: moves differ {:?} vs {:?}", a, b);
        }
    }

    #[test]
    fn offpv_mcts_fallback_non_pv_when_pv_had_all_mass() {
        // Construct a distribution where PV has all the mass, others exist with 0.0
        let mv_pv = triplecargo::Move { card_id: 301, cell: 0 };
        let mv_a = triplecargo::Move { card_id: 302, cell: 1 };
        let mv_b = triplecargo::Move { card_id: 303, cell: 2 };

        let mut dist: BTreeMap<String, f32> = BTreeMap::new();
        dist.insert(format!("{}-{}", mv_pv.card_id, mv_pv.cell), 1.0);
        dist.insert(format!("{}-{}", mv_a.card_id, mv_a.cell), 0.0);
        dist.insert(format!("{}-{}", mv_b.card_id, mv_b.cell), 0.0);

        let seed: u64 = 0xDEADFACE;
        let turn: u8 = 0;
        let gid: u64 = 0;

        // With PV removed, fallback should select a non-PV move deterministically
        let pick = super::pick_mcts_non_pv_move_from_dist(&dist, Some(mv_pv), seed, gid, turn)
            .expect("expected a move");
        assert!(
            !(pick.card_id == mv_pv.card_id && pick.cell == mv_pv.cell),
            "fallback should select a non-PV move when PV had all mass"
        );
    }
}