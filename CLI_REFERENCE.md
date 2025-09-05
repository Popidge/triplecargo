# CLI Reference — precompute/export

This document describes the command-line flags accepted by the precompute/export binary ([`target/release/precompute`](target/release/precompute:1)). Flags are grouped by purpose and described with defaults and semantics.

Usage overview

- Export output: --export PATH (required for export runs)
- Mode: --export-mode trajectory|full|graph (default: trajectory)
- Single-state eval: --eval-state (stdin→stdout)

Top-level export flags

- --export PATH
  - JSONL output file path (required for export runs).
- --export-mode trajectory|full|graph
  - trajectory (default): emit a single 9-state PV trajectory per sampled game.
  - full: emit the entire reachable state space for one sampled hand pair (exhaustive).
  - graph: two-phase full-graph pipeline. Phase A enumerates reachable states; Phase B performs bottom-up retrograde solve to compute exact values and best_move for every state deterministically. Requires full depth; if --max-depth would truncate, Graph mode errors.
- --games N
  - Number of sampled games (trajectory mode only; ignored in full and graph modes).

Graph explicit input flags (graph-only)

- --graph-input
  - When true, Graph mode uses explicit hands/elements flags instead of sampling.
- --graph-hand-a "id1,id2,id3,id4,id5"
  - Required when --graph-input is true. Exactly 5 card ids, unique within the hand; validated against cards DB (`data/cards.json`).
- --graph-hand-b "id1,id2,id3,id4,id5"
  - Required when --graph-input is true. Exactly 5 card ids, unique within the hand; validated against cards DB.
  - Note: cross-hand duplicates allowed; only intra-hand duplicates are rejected.
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

Off-principal-variation (off-PV) stepping (trajectory only)

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

Graph mode (two-phase full-graph)

Graph mode builds an exact, deterministic value/best_move label for every reachable state from a single sampled initial hand pair (+ optional elements). It proceeds in two phases:

- Phase A — Enumeration: parallel BFS by ply with a sharded visited set to avoid duplicates. See [`src/solver/graph.rs:186`](src/solver/graph.rs:186).
- Phase B — Retrograde solve: bottom-up dynamic programming from ply 9 (terminal) back to the root (0), computing exact values and the lexicographically-first optimal move at each state. See [`src/solver/graph.rs:376`](src/solver/graph.rs:376).

Properties:
- Deterministic by construction (stable move order; tie-break is “first in legal_moves order”).
- Requires full depth (to ply 9). If --max-depth would truncate, Graph mode errors out early.
- Current output: per-depth counts and a root summary printed to stderr. JSONL emission using retrograde labels is planned.

CLI summary:
- `--export-mode graph`
- `--games` is ignored (graph solves exactly one sampled initial pair).
- `--hand-strategy`, `--rules`, `--elements`, and `--seed` control the initial state.
- `--tt-bytes` has no effect during retrograde (no search is used); it still applies in other modes.

Implementation references

- Entry: [`src/solver/graph.rs:580`](src/solver/graph.rs:580)
- Enumeration: [`src/solver/graph.rs:186`](src/solver/graph.rs:186)
- Retrograde: [`src/solver/graph.rs:376`](src/solver/graph.rs:376)

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
- See the eval-mode entry in source: [`src/bin/precompute.rs:1275`](src/bin/precompute.rs:1275).

Examples

- Trajectory (onehot), 10 games, no elements:
  - [`target/release/precompute`](target/release/precompute:1) --export export.jsonl --export-mode trajectory --games 10 --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss

- Trajectory (mcts policy, 256 rollouts), stratified hands, Elemental+Same, margin:
  - [`target/release/precompute`](target/release/precompute:1) --export export_mcts.jsonl --export-mode trajectory --games 10 --seed 42 --hand-strategy stratified --rules elemental,same --elements random --policy-format mcts --mcts-rollouts 256 --value-mode margin

- Trajectory with off-PV weighted stepping @ 20%:
  - [`target/release/precompute`](target/release/precompute:1) --export export_weighted.jsonl --export-mode trajectory --games 100 --seed 123 --hand-strategy random --rules none --elements none --policy-format onehot --off-pv-rate 0.2 --off-pv-strategy weighted

- Full state-space export (exhaustive) for one sampled hand pair:
  - [`target/release/precompute`](target/release/precompute:1) --export export_full.jsonl --export-mode full --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss

- Graph (two-phase full-graph) for one sampled hand pair + elements (prints per-depth counts and root summary):
  - [`target/release/precompute`](target/release/precompute:1) --export export_graph.jsonl --export-mode graph --seed 42 --hand-strategy random --rules none --elements random

- Single-state evaluation (stdin → stdout) example:
  - echo '<single-state-json>' | [`target/release/precompute`](target/release/precompute:1) --eval-state