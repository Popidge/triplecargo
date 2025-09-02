# ♠️ Triplecargo — Triple Triad Assistant & Solver (FF8)

Deterministic, high-performance Rust implementation of Final Fantasy VIII's Triple Triad ruleset. Designed for exact solving, large-scale precomputation, analysis experiments, training-data export, and AI/ML integration.

Key capabilities:
- Perfect-play solving (Negamax + αβ + transposition table).
- Full state-space precomputation for instant queries.
- Deterministic, reproducible exports for training and analysis.
- Lightweight assistants and integration points for ML workflows.

---

## Quickstart

1. Build (release):
   - cargo build --release

2. Binaries
   - Precompute / export: [`target/release/precompute`](target/release/precompute:1)
   - Query / lookup: [`target/release/query`](target/release/query:1)
   - Demo CLI: [`target/release/tt-cli`](target/release/tt-cli:1)

3. Card data
   - Card definitions are loaded from [`data/cards.json`](data/cards.json:1).

---

## Overview & Design Goals

Triplecargo is both a research-grade solver and the foundation for a superhuman Triple Triad assistant — combining exact computation with modern AI techniques. Design priorities:

- Correctness and determinism (fixed move ordering, reproducible cascades, stable hashing).
- High throughput for precomputation (multi-threaded, per-worker caches).
- Clean, append-only persistence for database writes and deterministic compaction.
- Export formats suitable for ML training (JSONL trajectory or full state space).

---

## Binaries and Modes

- Precompute/export: [`target/release/precompute`](target/release/precompute:1)
  - Exports training data in three modes: trajectory, full state-space, or graph (two-phase full-graph).
  - Single-state evaluation mode: see the "Single-state evaluation" section and [`src/bin/precompute.rs:555`](src/bin/precompute.rs:555).

- Query service: [`target/release/query`](target/release/query:1)
  - Planned CLI/HTTP lookup against precomputed DBs.

- Demo CLI: [`target/release/tt-cli`](target/release/tt-cli:1)

Source entry points:
- Precompute implementation and play/selector logic: [`src/bin/precompute.rs:381`](src/bin/precompute.rs:381), [`src/bin/precompute.rs:877`](src/bin/precompute.rs:877).
- Root search + child Q values: [`src/solver/negamax.rs:98`](src/solver/negamax.rs:98).
- Heuristic simulation helpers: [`src/engine/apply.rs:224`](src/engine/apply.rs:224), [`src/engine/apply.rs:33`](src/engine/apply.rs:33).
- Board cell elements: [`src/board.rs:56`](src/board.rs:56).
- Graph pipeline: [rust.enumerate_graph()](src/solver/graph.rs:175), [rust.retrograde_solve()](src/solver/graph.rs:256), [rust.graph_precompute_export()](src/solver/graph.rs:396).

---

## CLI Reference — Export / Precompute

The following documents the flags accepted by the precompute/export binary (`[`target/release/precompute`](target/release/precompute:1)`). Flags are grouped by purpose and described with defaults and semantics.

Top-level export
- --export PATH
  - JSONL output file path (required for export runs).
- --export-mode trajectory|full|graph
  - trajectory (default): emit a single 9-state PV trajectory per sampled game.
  - full: emit the entire reachable state space for one sampled hand pair (exhaustive).
  - graph: two-phase full-graph pipeline. Phase A enumerates all reachable states with a sharded visited set; Phase B performs a bottom‑up retrograde solve (depth 9 → 0) to compute exact values and best_move for every state deterministically. Requires full depth; if `--max-depth` would truncate, Graph mode returns an error.
- --games N
  - Number of sampled games (trajectory mode only; ignored in full and graph modes).
Graph explicit input flags (only for --export-mode graph)
- --graph-input
  - When true, Graph mode uses explicit hands/elements flags instead of sampling.
  - When false (default), Graph mode samples hands/elements deterministically using --hand-strategy and --seed (mirrors Full mode sampling).
- --graph-hand-a "id1,id2,id3,id4,id5"
  - Required when --graph-input is true. Exactly 5 card ids, unique within the hand; validated against cards DB.
- --graph-hand-b "id1,id2,id3,id4,id5"
  - Required when --graph-input is true. Exactly 5 card ids, unique within the hand; validated against cards DB.
  - Note: A and B hands may contain the same card id (cross-hand duplicates allowed); only intra-hand duplicates are rejected.
- --graph-elements "E0,E1,E2,E3,E4,E5,E6,E7,E8"
  - Optional when --graph-input is true. Exactly 9 comma-separated tokens using letters F,I,T,W,E,P,H,L or '-' (none).
  - Constraint: the 8 element letters must not repeat across the 9 cells (no duplicate element letters). '-' entries are ignored for this check.

Parallelism & scheduling
- --threads N
  - Number of worker threads (default: available_parallelism()-1, min 1). Workers + single writer model; deterministically claim game indices.
- --chunk-size N
  - Number of games each worker fetches at once. If omitted, defaults to effective_chunk_size = min(32, max(1, floor(games/threads))). Scheduling only; affects locality and latency.

Sampling & determinism
- --seed U64
  - Master RNG seed used for hand sampling, per-ply RNG (off-PV), and MCTS rollouts. Deterministic across runs with identical flags.
- --hand-strategy random|stratified
  - How hands are sampled from the card pool.
- --rules elemental,same,plus,same_wall (or 'none')
  - Comma-separated rule toggles. Example: --rules elemental,same
- --elements none|random
  - Board element layout. random is deterministic per game (driven by --seed).

Policy & value targets
- --policy-format onehot|mcts
  - onehot: single-best-move export (object).
  - mcts: distribution over legal moves (map).
- --mcts-rollouts N
  - Rollouts used when --policy-format mcts (default 100).
- --value-mode winloss|margin
  - winloss: {-1,0,1} from side-to-move perspective.
  - margin: integer final score difference (A_cards − B_cards) from A's perspective.

Off-principal-variation (off-PV) stepping (trajectory-only)
- --off-pv-rate FLOAT
  - Per-game probability (0..1) that a game uses off-PV stepping for all plies. 0 disables.
- --off-pv-strategy random|weighted|mcts
  - random: uniform over legal non-PV moves (fallback to PV when none).
  - weighted: softmax over root child negamax Q values; PV mass zeroed and renormalised.
  - mcts: reuse root MCTS (when --policy-format=mcts) or compute root MCTS with --mcts-rollouts otherwise.

Transposition table configuration
- --tt-bytes N
  - TT size per worker in MiB (default 32). Rounded down to nearest power-of-two capacity that fits the budget. Total TT memory ≈ workers × tt-bytes.
  - Use to control memory footprint for large parallel runs.

Logging & verbosity
- --verbose
  - Emit a compact diagnostic line during eval-style runs and additional worker-start logs.

Play strategy & heuristic mixing (trajectory)
- --play-strategy pv|mcts|heuristic|mix (default: mix)
  - pv: always play the principal-variation move.
  - mcts: sample from MCTS root distribution when available.
  - heuristic: sample from a lightweight heuristic distribution.
  - mix: per-ply mixture of PV, heuristic, and optional MCTS; early plies are heuristic-heavier, late plies PV-heavier.
- --mix-heuristic-early FLOAT (default 0.65)
- --mix-heuristic-late FLOAT (default 0.10)
- --mix-mcts FLOAT (default 0.25 when --policy-format mcts, else ignored)

Heuristic feature weights
- --heur-w-corner FLOAT (default 1.0)
- --heur-w-edge FLOAT (default 0.3)
- --heur-w-center FLOAT (default -0.2)
- --heur-w-greedy FLOAT (default 0.8)
  - Immediate margin gain after simulating the move (helper in [`src/engine/apply.rs:224`](src/engine/apply.rs:224)).
- --heur-w-defense FLOAT (default 0.6)
  - Exposure/vulnerability vs neighbours (uses adjusted sides logic in [`src/engine/apply.rs:33`](src/engine/apply.rs:33)).
- --heur-w-element FLOAT (default 0.6)
  - Element synergy penalties/bonuses (see [`src/board.rs:56`](src/board.rs:56)).

Progress & determinism notes
- The writer thread buffers out-of-order arrivals and writes games in strict increasing game_id order to guarantee byte-for-byte identical JSONL output across different --threads and --chunk-size values when the seed + flags are identical.
- Worker behavior and scheduling are deterministic functions of (seed, game_id, worker_id, chunk_size).
## Graph mode (two-phase full-graph)

Graph mode builds an exact, deterministic value/best_move label for every reachable state from a single sampled initial hand pair (+ optional elements). It proceeds in two phases:

- Phase A — Enumeration: parallel BFS by ply with a sharded visited set to avoid duplicates. See [rust.enumerate_graph()](src/solver/graph.rs:175).
- Phase B — Retrograde solve: bottom-up dynamic programming from ply 9 (terminal) back to the root (0), computing exact values and the lexicographically-first optimal move at each state. See [rust.retrograde_solve()](src/solver/graph.rs:256).

Properties:
- Deterministic by construction (stable move order; tie-break is “first in legal_moves order”).
- Requires full depth (to ply 9). If `--max-depth` would truncate, Graph mode errors out early.
- Current output: per-depth counts and a root summary printed to stderr. JSONL emission using retrograde labels is planned.

CLI summary:
- `--export-mode graph`
- `--games` is ignored (graph solves exactly one sampled initial pair).
- `--hand-strategy`, `--rules`, `--elements`, and `--seed` control the initial state.
- `--tt-bytes` has no effect during retrograde (no search is used); it still applies in other modes.

Implementation references:
- Entry: [rust.graph_precompute_export()](src/solver/graph.rs:396)
- Enumeration: [rust.enumerate_graph()](src/solver/graph.rs:175)
- Retrograde: [rust.retrograde_solve()](src/solver/graph.rs:256)

Single-state evaluation
- Mode: read exactly one JSON object from stdin and write exactly one JSON object to stdout.
- Invoke: [`target/release/precompute --eval-state`](target/release/precompute:1) (suppress progress; use --verbose for diagnostics).
- Configure TT size with --tt-bytes to control memory during eval.

Input schema (single-state eval — subset of a JSONL line)
{
  "board": [ { "cell":0,"card_id":12,"owner":"A","element":"F" }, ... ],
  "hands": { "A":[34,56,78,90,11], "B":[22,33,44,55,66] },
  "to_move": "A",
  "turn": 0,
  "rules": { "elemental":true,"same":true,"plus":false,"same_wall":false },
  "board_elements": ["F","I",null,...]   // optional; if present must match board[].element
}

Output schema (single-state eval)
{
  "best_move": { "card_id": 34, "cell": 0 },  // omitted at terminal
  "value": 1,                                  // {-1,0,1} from side-to-move perspective
  "margin": 3,                                 // A_cards − B_cards at terminal
  "pv": [ { "card_id":34,"cell":0 }, ... ],
  "nodes": 123456,
  "depth": 9,
  "state_hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
}
- See the eval-mode entry in source: [`src/bin/precompute.rs:555`](src/bin/precompute.rs:555).

---

## JSONL Export Schema (per-line)

Each exported line represents the state before a move is taken.

Core fields
- game_id: sequential per sampled game (trajectory); 0 in full mode.
- state_idx: 0..8 within a trajectory (equals "turn").
- board: list of 9 cell objects:
  - {"cell":0,"card_id":12,"owner":"A","element":"F"} — element present only when rules.elemental=true.
- hands: "A" and "B" lists (shrinking across the trajectory).
- to_move: "A" or "B"
- turn: integer 0..8
- rules: object with booleans (elemental, same, plus, same_wall)
- policy_target:
  - onehot → single best-move object: {"card_id":34,"cell":0}
  - mcts → map of "cardId-cell" → probability: {"34-0":0.7,"56-1":0.3}
  - Terminal states: onehot → omitted, mcts → {}
- value_target:
  - winloss: {-1,0,+1} (side-to-move perspective)
  - margin: integer final score difference (A − B)
- off_pv: boolean, true for exported lines belonging to off-PV games
- state_hash: 128-bit Zobrist hash (hex string)

Example JSONL line structure:
{
  "game_id": 12,
  "state_idx": 0,
  "board": [...],
  "hands": { "A": [...], "B": [...] },
  "to_move": "A",
  "turn": 0,
  "rules": { "elemental": true, "same": true, "plus": false, "same_wall": false },
  "policy_target": {"34-0": 0.7, "56-1": 0.3},
  "value_target": 1,
  "value_mode": "winloss",
  "off_pv": false,
  "state_hash": "..."
}

---

## Export Semantics & Scheduling

Chunked scheduling
- Each worker acquires chunks of game indices deterministically: a round‑robin mapping of (worker_id, worker_count, chunk_size) to game ranges.
- Default chunk-size behavior: effective_chunk_size = min(32, max(1, floor(games/threads))).
- Chunking affects only scheduling and locality; it does not change exported content or ordering because the single writer emits lines in increasing game_id order.

Per-worker transposition table (TT)
- Each worker owns a fixed-size direct-mapped TT (no sharing or locks).
- Defaults:
  - --tt-bytes 32 (MiB) per worker (rounded down to power-of-two capacity).
- Warming:
  - In trajectory mode, each worker reuses its TT across the games it processes for improved locality.
  - In full mode, a single TT is used for the entire DFS export.
- Determinism:
  - With identical seed + flags, labels are byte-for-byte identical regardless of --tt-bytes or --threads.

Determinism guarantee
- With identical seed and flags, JSONL output is byte-for-byte identical across runs regardless of thread count and chunk size. The writer enforces ordering and workers derive sampling deterministically.

---

## Play selection & Off-PV behaviour

Play strategies
- --play-strategy pv|mcts|heuristic|mix
  - mix (default): deterministic per-ply mixture (progressive schedule), described by --mix-heuristic-early/late and optionally --mix-mcts.
- --mix-heuristic-early, --mix-heuristic-late
  - Control the progressive probability schedule (defaults: 0.65 early, 0.10 late).

Off-PV stepping (trajectory only)
- Controlled by --off-pv-rate and --off-pv-strategy.
- Off-PV sampling is deterministic using an RNG derived from (seed, game_id, turn).

Implementation references
- Mixed selector logic: [`src/bin/precompute.rs:877`](src/bin/precompute.rs:877)
- MCTS policy distribution: [`src/bin/precompute.rs:381`](src/bin/precompute.rs:381)
- Root Q-values used for policy/value: [`src/solver/negamax.rs:98`](src/solver/negamax.rs:98)

---

## Rules Implemented

- Basic capture: placed card flips weaker adjacent opponents.
- Elemental: +1/−1 adjustments based on cell element.
- Same: 2+ equalities trigger flips.
- Same Wall: walls count as 10 for Same.
- Plus: equal sums trigger flips.
- Combo: cascades that apply Basic rule only.

References: rules engine and apply logic are implemented across [`src/rules.rs`](src/rules.rs:1), [`src/engine/apply.rs:224`](src/engine/apply.rs:224).

---

## Data model & Hashing

- Cards: validated, loaded from [`data/cards.json`](data/cards.json:1).
- Board: fixed 3×3 array; optional per-cell element field when elemental rules are enabled.
- Hands: 5 cards per player (engine uses fixed arrays; exports show shrinking lists).
- Rules: independent boolean toggles.
- Hashing: XOR-based Zobrist with incremental updates and 128-bit keys for deterministic state identity.

---

## Performance & Implementation Notes

- Solver: Negamax with αβ pruning and a transposition table (depth-preferred replacement).
- Move ordering: corners > edges > center, then cell index, then card id.
- Cascade processing: BFS on queue with ascending indices for deterministic cascades.
- Throughput: ~10M nodes/sec, ~1M states/sec on a Ryzen 3600 (observed; config-dependent).
- Training data export: JSONL trajectory or full state-space, deterministic with a fixed seed.
- Persistence: append-only batch writes, deterministic compaction, optional compression and checkpoint/resume.

---

## Examples

Trajectory (onehot), 10 games, no elements:
- [`target/release/precompute`](target/release/precompute:1) --export export.jsonl --export-mode trajectory --games 10 --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss

Trajectory (mcts policy, 256 rollouts), stratified hands, Elemental+Same, margin:
- [`target/release/precompute`](target/release/precompute:1) --export export_mcts.jsonl --export-mode trajectory --games 10 --seed 42 --hand-strategy stratified --rules elemental,same --elements random --policy-format mcts --mcts-rollouts 256 --value-mode margin

Trajectory with off-PV weighted stepping @ 20%:
- [`target/release/precompute`](target/release/precompute:1) --export export_weighted.jsonl --export-mode trajectory --games 100 --seed 123 --hand-strategy random --rules none --elements none --policy-format onehot --off-pv-rate 0.2 --off-pv-strategy weighted

Full state-space export (exhaustive) for one sampled hand pair:
- [`target/release/precompute`](target/release/precompute:1) --export export_full.jsonl --export-mode full --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss

Graph (two-phase full-graph) for one sampled hand pair + elements (prints per-depth counts and root summary):
- [`target/release/precompute`](target/release/precompute:1) --export export_graph.jsonl --export-mode graph --seed 42 --hand-strategy random --rules none --elements random

Single-state evaluation (stdin → stdout) example:
- echo '<single-state-json>' | [`target/release/precompute`](target/release/precompute:1) --eval-state

---

## Testing & Validation

- Engine tests: correctness, rule edge cases, allocation-free hot paths (see tests/).
- Solver tests: terminal values and PV determinism.
- Persistence tests: batch flush, compression, checkpoint/resume determinism.
- Determinism tests include cross-thread JSONL equality for small sample runs.

See test harness and targeted tests in the repository tests/ directory (example files visible in the project root).

---

## Project Layout (high level)

- Core engine: rules, state, hashing, scoring (in [`src/`](src/:1)).
- Solver: negamax + αβ + TT, PV reconstruction (in [`src/solver/`](src/solver/:1)).
- Precompute: state enumeration, parallel solve, persistence (in [`src/bin/precompute.rs`](src/bin/precompute.rs:1) and [`src/solver/precompute.rs`](src/solver/precompute.rs:1)).
- Graph pipeline: enumeration and retrograde solver (in [`src/solver/graph.rs`](src/solver/graph.rs:1)).
- Persistence: batch writes, compression, checkpointing (see [`src/persist.rs`](src/persist.rs:1), [`src/persist_stream.rs`](src/persist_stream.rs:1)).
- Binaries: precompute, query, tt-cli (see [`src/bin/`](src/bin/:1)).

---

## Roadmap

Near-term
- Query API: CLI/HTTP service for instant best-move lookups from solved DBs.
- Analysis experiments: first-player advantage, card/hand Elo, rule impact studies.

Mid-term
- AI/ML integration: compact NPZ exports, PyTorch dataset utilities, small policy/value nets, AlphaZero-lite self-play.
- Imperfect-information play: Expectimax/MCTS with hidden hands and opponent models.

Long-term
- Superhuman assistant with multiple playstyles and human-like modes.
- Meta-analysis: best-hand search, card tier lists, balance insights.

---

## License & Attribution

- Repository code: open license (TBD). Triple Triad rules belong to Square Enix (Final Fantasy VIII). This repository is a clean‑room reimplementation for research and educational purposes.

---

## Acknowledgements

- FF8’s Triple Triad ruleset (Square Enix)
- Classic engine patterns: negamax, αβ, Zobrist hashing, transposition tables
- Inspiration & help from GPT-5/Kilo Code