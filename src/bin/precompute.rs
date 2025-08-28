use std::path::PathBuf;
use std::time::Instant;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::collections::BTreeMap;

use clap::{Parser, ValueEnum};
use rayon::ThreadPoolBuilder;
use serde::Serialize;
use serde_json;

use triplecargo::{
    load_cards_from_json, zobrist_key, Element, GameState, Owner, Rules, apply_move, legal_moves, score,
    solver::{precompute_solve, ElementsMode, search_root, InMemoryTT},
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

    /// Value target mode: winloss | margin
    #[arg(long, value_enum, default_value_t = ValueModeOpt::Winloss)]
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
   state_hash: String,
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
   policy_format: PolicyFormatOpt,
   value_mode: ValueModeOpt,
   mcts_rollouts: usize,
   base_seed: u64,
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
       policy_format: &PolicyFormatOpt,
       value_mode: &ValueModeOpt,
       mcts_rollouts: usize,
       base_seed: u64,
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
       let policy_out: Option<PolicyOut> = match policy_format {
           PolicyFormatOpt::Onehot => {
               bm.map(|m| PolicyOut::Move { card_id: m.card_id, cell: m.cell })
           }
           PolicyFormatOpt::Mcts => {
               // Deterministic per-state seed
               let z = zobrist_key(state);
               let seed = base_seed ^ (z as u64) ^ ((z >> 64) as u64);
               let dist = mcts_policy_distribution(state, cards, mcts_rollouts, seed);
               Some(PolicyOut::Dist(dist))
           }
       };

       let hands = collect_hands_from_state(state);
       let to_move = match state.next { Owner::A => 'A', Owner::B => 'B' };
       let vt = compute_value_target(value_mode, val, state.next);
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
           value_mode: value_mode_str(value_mode).to_string(),
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
       policy_format: &PolicyFormatOpt,
       value_mode: &ValueModeOpt,
       mcts_rollouts: usize,
       base_seed: u64,
   ) {
       let key = zobrist_key(state);
       if !visited.insert(key) {
           return;
       }
       push_record(state, cards, tt, cap, out, policy_format, value_mode, mcts_rollouts, base_seed);

       if state.is_terminal() {
           return;
       }
       let moves = legal_moves(state);
       for mv in moves {
           if let Ok(ns) = apply_move(state, cards, mv) {
               dfs(&ns, cards, tt, visited, cap, out, policy_format, value_mode, mcts_rollouts, base_seed);
           }
       }
   }

   dfs(initial, cards, &mut tt, &mut visited, max_depth, &mut out, &policy_format, &value_mode, mcts_rollouts, base_seed);
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Configure a small, deterministic Rayon pool (2-4 threads). Ignore error if already set.
    let _ = ThreadPoolBuilder::new().num_threads(4).build_global();

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

        match args.export_mode {
            ExportModeOpt::Trajectory => {
                let games = args.games;
                println!(
                    "[export] Mode=trajectory games={} hand_strategy={:?} seed={:#x} policy_format={:?} value_mode={:?}",
                    games, args.hand_strategy, args.seed, args.policy_format, args.value_mode
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

                    // Play a single principal-variation trajectory of 9 plies
                    for _ply in 0..9 {
                        // In trajectory mode we always search full remaining depth to obtain final-outcome value targets.
                        let eff_depth = 9 - state.board.filled_count();

                        let (val, bm, _nodes) = search_root(&state, &cards, eff_depth, &mut tt);

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

                        // Build policy output based on requested format
                        let policy_out: Option<PolicyOut> = match args.policy_format {
                            PolicyFormatOpt::Onehot => {
                                bm.map(|m| PolicyOut::Move { card_id: m.card_id, cell: m.cell })
                            }
                            PolicyFormatOpt::Mcts => {
                                // Seed per ply for determinism (trajectory keeps legacy per-ply seeding)
                                let seed_for_rollouts = splitmix64(&mut rng_state);
                                let dist = mcts_policy_distribution(&state, &cards, args.mcts_rollouts, seed_for_rollouts);
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
                            state_hash: format!("{:032x}", zobrist_key(&state)),
                        };
                        let line = serde_json::to_string(&rec).map_err(|e| format!("serialize JSONL record error: {e}"))?;
                        writer.write_all(line.as_bytes()).map_err(|e| format!("export write error: {e}"))?;
                        writer.write_all(b"\n").map_err(|e| format!("export write error: {e}"))?;
                        total_lines += 1;

                        // Apply best move; if None (shouldn't happen before full), stop
                        if let Some(mv) = bm {
                            if let Ok(ns) = apply_move(&state, &cards, mv) {
                                state = ns;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    // Periodic flush
                    if total_lines % 10_000 == 0 {
                        let _ = writer.flush();
                    }
                }

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
                    "[export] Mode=full (ignoring --games). hand_strategy={:?} seed={:#x} policy_format={:?} value_mode={:?}",
                    args.hand_strategy, args.seed, args.policy_format, args.value_mode
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

                // Full enumeration with new schema/options
                let records = enumerate_solved_records(
                    &initial,
                    &cards,
                    args.max_depth,
                    args.policy_format.clone(),
                    args.value_mode.clone(),
                    args.mcts_rollouts,
                    args.seed, // base seed for per-state MCTS seeding
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