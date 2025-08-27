use std::collections::BTreeMap;
use std::hash::BuildHasherDefault;
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use hashbrown::HashSet as HbHashSet;

use crate::cards::CardsDb;
use crate::engine::apply::apply_move;
use crate::hash::zobrist_key;
use crate::persist::{ElementsMode, SolvedEntry};
use crate::state::{is_terminal, legal_moves, GameState};
use crate::solver::move_order::order_moves;
use crate::solver::negamax::search_root;
use crate::solver::tt::{Bound, InMemoryTT, TTEntry, TranspositionTable, TTStats};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

type FastHasher = BuildHasherDefault<ahash::AHasher>;
type FastSet = HbHashSet<u128, FastHasher>;

#[derive(Debug, Clone, Copy)]
pub struct PrecomputeStats {
    pub nodes: u64,
    pub states_enumerated: u64,
    pub exact_entries: usize,
    pub roots: usize,
}

#[inline]
fn ttentry_to_solved(e: &TTEntry) -> SolvedEntry {
    SolvedEntry {
        value: e.value,
        best_move: e.best_move,
        depth: e.depth,
    }
}

/// Deterministic tie-breaking for merging entries with the same key.
/// Preference order:
/// - Higher depth wins
/// - If equal depth, prefer presence of best_move (Some over None)
/// - If both Some, prefer lexicographically smaller (cell, then card_id)
/// - If still equal or both None, prefer smaller value to keep order stable
fn prefer_new_over_old(new: &TTEntry, old: &SolvedEntry) -> bool {
    if new.depth > old.depth {
        return true;
    }
    if new.depth < old.depth {
        return false;
    }
    match (new.best_move, old.best_move) {
        (Some(a), Some(b)) => {
            if a.cell != b.cell {
                return a.cell < b.cell;
            }
            if a.card_id != b.card_id {
                return a.card_id < b.card_id;
            }
            new.value < old.value
        }
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => new.value < old.value,
    }
}

fn dfs_enumerate_and_solve(
    state: &GameState,
    cards: &CardsDb,
    tt: &mut InMemoryTT,
    visited: &mut FastSet,
    nodes_acc: &mut u64,
    max_depth: Option<u8>,
    nodes_total: &Arc<AtomicU64>,
    states_total: &Arc<AtomicU64>,
    pb: &ProgressBar,
) {
    let key = zobrist_key(state);
    if !visited.insert(key) {
        return;
    }
    // Throttled: no per-state progress bar increment
    states_total.fetch_add(1, Ordering::Relaxed);

    let full = 9 - state.board.filled_count();
    let eff_depth = match max_depth {
        Some(cap) => full.min(cap),
        None => full,
    };
    // Throttled: avoid per-state message updates
    // Skip re-solving if we already have an Exact entry with sufficient depth
    let mut need_solve = true;
    if let Some(entry) = tt.get(key) {
        if entry.flag == Bound::Exact && entry.depth >= eff_depth {
            need_solve = false;
        }
    }
    if need_solve {
        let (_val, _bm, nodes) = search_root(state, cards, eff_depth, tt);
        *nodes_acc += nodes;
        nodes_total.fetch_add(nodes, Ordering::Relaxed);
    }

    if is_terminal(state) {
        return;
    }

    let moves = legal_moves(state);
    for mv in moves {
        if let Ok(ns) = apply_move(state, cards, mv) {
            dfs_enumerate_and_solve(&ns, cards, tt, visited, nodes_acc, max_depth, nodes_total, states_total, pb);
        }
    }
}

/// Bulk solve reachable state space from the provided initial state.
///
/// Strategy:
/// - Enumerate root moves deterministically
/// - Parallelize by root move (each worker has its own TT)
/// - For each encountered state, invoke a full-depth root search to cache an Exact entry
/// - Merge per-root TTs deterministically, persisting Exact entries only
pub fn precompute_solve(
    initial: &GameState,
    cards: &CardsDb,
    _elements_mode: ElementsMode,
    max_depth: Option<u8>,
) -> (BTreeMap<u128, SolvedEntry>, PrecomputeStats) {
    // Solve and include the initial state's entry itself
    let mut tt0 = InMemoryTT::default();
    let remaining0 = 9 - initial.board.filled_count();
    let depth0 = match max_depth {
        Some(cap) => remaining0.min(cap),
        None => remaining0,
    };
    let (_val0, _bm0, _nodes0) = search_root(initial, cards, depth0, &mut tt0);

    // Deterministically order root moves for better load balance
    let mut roots = legal_moves(initial);
    order_moves(&mut roots, None);
 
    // Multi-progress bars
    let mp = MultiProgress::new();
    let root_pb = mp.add(ProgressBar::new(roots.len() as u64));
    root_pb
        .set_style(ProgressStyle::with_template("[{elapsed_precise}] roots {bar:40.cyan/blue} {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"));
 
    let states_pb = mp.add(ProgressBar::new_spinner());
    states_pb
        .set_style(ProgressStyle::with_template("[{elapsed_precise}] states ~{pos} {msg}")
            .unwrap());
    states_pb.enable_steady_tick(std::time::Duration::from_millis(100));
 
    let nodes_pb = mp.add(ProgressBar::new_spinner());
    nodes_pb
        .set_style(ProgressStyle::with_template("[{elapsed_precise}] nodes ~{pos} {msg}")
            .unwrap());
    nodes_pb.enable_steady_tick(std::time::Duration::from_millis(250));
 
    // Shared counters for rates
    let start = Instant::now();
    let nodes_total = Arc::new(AtomicU64::new(0));
    let states_total = Arc::new(AtomicU64::new(0));
 
    // Background rate updater
    {
        let states_pb = states_pb.clone();
        let nodes_pb = nodes_pb.clone();
        let nodes_total_c = Arc::clone(&nodes_total);
        let states_total_c = Arc::clone(&states_total);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let elapsed = start.elapsed().as_secs_f64().max(1e-6);
                let s = states_total_c.load(Ordering::Relaxed);
                let n = nodes_total_c.load(Ordering::Relaxed);
                states_pb.set_message(format!("{:.0}/s", (s as f64 / elapsed)));
                states_pb.set_position(s);
                nodes_pb.set_message(format!("{:.0}/s", (n as f64 / elapsed)));
                nodes_pb.set_position(n);
            }
        });
    }
 
    // Capture depth cap for parallel workers
    let cap = max_depth;
 
    let results: Vec<(Vec<(u128, TTEntry)>, u64, u64, TTStats)> = roots
        .par_iter()
        .map(|mv| {
            let mut nodes: u64 = 0;
            let mut visited: FastSet = HbHashSet::default();
            let mut tt = InMemoryTT::default();
 
            // Clone progress bars and counters for this worker (they share internal state)
            let local_states = states_pb.clone();
            let local_root = root_pb.clone();
            let local_nodes_total = Arc::clone(&nodes_total);
            let local_states_total = Arc::clone(&states_total);
 
            if let Ok(child) = apply_move(initial, cards, *mv) {
                dfs_enumerate_and_solve(
                    &child,
                    cards,
                    &mut tt,
                    &mut visited,
                    &mut nodes,
                    cap,
                    &local_nodes_total,
                    &local_states_total,
                    &local_states,
                );
            }
            local_root.inc(1);
            let enumerated = visited.len() as u64;
            let stats = tt.stats();
            let items = tt.into_vec();
            (items, nodes, enumerated, stats)
        })
        .collect();
 
    // Finish progress bars
    root_pb.finish_and_clear();
    states_pb.finish_and_clear();
    nodes_pb.finish_and_clear();

    let mut merged: BTreeMap<u128, SolvedEntry> = BTreeMap::new();
    let mut nodes_sum: u64 = 0;
    let mut states_enum_sum: u64 = 0;

    let init_key = zobrist_key(initial);
    if let Some(e0) = tt0.get(init_key) {
        if e0.flag == Bound::Exact {
            merged.insert(init_key, ttentry_to_solved(&e0));
        }
    }

    // Aggregate per-root TT stats
    let mut sum_stats = TTStats::default();

    for (items, nodes, enum_count, stats) in results {
        nodes_sum = nodes_sum.saturating_add(nodes);
        states_enum_sum = states_enum_sum.saturating_add(enum_count);

        sum_stats.puts = sum_stats.puts.saturating_add(stats.puts);
        sum_stats.exact_count = sum_stats.exact_count.saturating_add(stats.exact_count);
        sum_stats.lower_count = sum_stats.lower_count.saturating_add(stats.lower_count);
        sum_stats.upper_count = sum_stats.upper_count.saturating_add(stats.upper_count);

        for (k, e) in items {
            if e.flag != Bound::Exact {
                continue;
            }
            let se_new = ttentry_to_solved(&e);
            match merged.get(&k) {
                None => {
                    merged.insert(k, se_new);
                }
                Some(old) => {
                    if prefer_new_over_old(&e, old) {
                        merged.insert(k, se_new);
                    }
                }
            }
        }
    }

    println!(
        "[precompute] TT stats: puts={}, exact={}, lower={}, upper={}",
        sum_stats.puts, sum_stats.exact_count, sum_stats.lower_count, sum_stats.upper_count
    );

    let stats = PrecomputeStats {
        nodes: nodes_sum,
        states_enumerated: states_enum_sum,
        exact_entries: merged.len(),
        roots: roots.len(),
    };

    (merged, stats)
}