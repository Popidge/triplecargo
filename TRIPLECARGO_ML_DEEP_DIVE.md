# Triplecargo ML Deep Dive

## Executive overview

`triplecargo` is a deterministic Triple Triad engine/solver/export toolkit implemented in Rust. It combines exact search (Negamax + alpha-beta + transposition tables), deterministic graph retrograde solving, and JSONL export pipelines intended for downstream training and analysis.

For ML engineering, this gives a rare combination:

- Exact state labels (`value_target`, `policy_target`) from a perfect solver.
- Deterministic replayable dataset generation (`--seed`, deterministic scheduling/writer ordering).
- Multiple export modes spanning imitation-style trajectories and exhaustive state-space exports.
- A strict stdin/stdout eval/apply API suitable for environment-serving and online data generation.

Core entrypoints live in `src/bin/precompute.rs`, `src/solver/negamax.rs`, `src/solver/graph.rs`, and `src/solver/graph_writer.rs`.

---

## How to install/build/run triplecargo

### Prerequisites

- Rust toolchain (stable), Cargo.
- Linux/macOS/Windows where Rust `cargo` builds are supported.
- Card database at `data/cards.json`.

### Build

```bash
cargo build --release
```

Release binaries (from `Cargo.toml`):

- `target/release/precompute`
- `target/release/query`
- `target/release/tt-cli`
- `target/release/tt-gui`

References: `Cargo.toml:33`, `Cargo.toml:37`, `Cargo.toml:41`, `Cargo.toml:45`.

### Basic runs

```bash
# Export trajectory training data
target/release/precompute --export export.jsonl --export-mode trajectory --games 100 --seed 42

# Query solved DB entry (legacy bincode DB path)
target/release/query --db solved.bin --hands "A=1,2,3,4,5" "B=6,7,8,9,10"

# Demo CLI game/scripted solver output
target/release/tt-cli

# Desktop GUI
target/release/tt-gui
```

References: `README.md:19`, `README.md:33`, `src/bin/query.rs:16`, `src/bin/tt-cli.rs:30`, `src/bin/tt-gui.rs:994`.

---

## System architecture and core modules

### High-level flow

1. Parse cards/rules/hands/elements into `GameState`.
2. Solve via exact search (`search_root` / `search_root_with_children`) or full graph retrograde.
3. Export JSONL line(s) with deterministic ordering and stable hashes.
4. Optionally consume via eval/apply stdin/stdout for service-style integration.

### Core module map

- Engine/state/rules/types:
  - `src/state.rs` (`GameState`, `Move`, legal move generation)
  - `src/engine/apply.rs` (Basic/Same/Plus/Same Wall/Combo move application)
  - `src/rules.rs`, `src/types.rs`, `src/board.rs`, `src/engine/score.rs`
- Hashing/determinism:
  - `src/hash.rs` (incremental Zobrist, 128-bit state identity)
  - `src/rng.rs` (deterministic RNG factory)
- Solver stack:
  - `src/solver/negamax.rs` (exact minimax search, PV reconstruction)
  - `src/solver/tt.rs`, `src/solver/tt_array.rs` (TT interfaces + fixed array TT)
  - `src/solver/move_order.rs`
- Graph/retrograde/export-at-scale:
  - `src/solver/graph.rs` (enumeration + retrograde + line emission)
  - `src/solver/graph_writer.rs` (plain/zstd async/sharded sinks, integrity)
- Binaries:
  - `src/bin/precompute.rs` (main export/eval/apply surface)
  - `src/bin/query.rs`, `src/bin/tt-cli.rs`, `src/bin/tt-gui.rs`

Library export surface: `src/lib.rs:5`.

---

## Triple Triad state/rules/move/outcome representation

### State

`GameState` (`src/state.rs:14`) includes:

- `board`: 3x3 fixed grid (`Board`) with optional occupied slots.
- `hands_a`, `hands_b`: fixed `[Option<u16>; 5]` card slots.
- `next`: side to move (`Owner::A`/`Owner::B`).
- `rules`: booleans (`elemental`, `same`, `plus`, `same_wall`).
- `zobrist`: cached 128-bit incremental hash.

### Move

`Move` is `(card_id, cell)` with deterministic legal move ordering by `cell` then `card_id` (`src/state.rs:107`).

### Rules and resolution

- Basic capture, Same, Plus, Same Wall, and Combo are implemented in `src/engine/apply.rs:224`.
- Elemental side adjustment uses per-cell element and card element (`src/engine/apply.rs:16`).
- Score is board ownership difference `A - B` (`src/engine/score.rs:4`).

### Outcome conventions

- Search values are side-to-move perspective in negamax (`src/solver/negamax.rs:16`).
- Export `value_target` is either win/loss sign or margin based on `value_mode` (`src/bin/precompute.rs:438`).

---

## CLI usage guide (`precompute`, `query`, `tt-cli`, `tt-gui`)

## `precompute`

Main export/eval binary (despite clap label `precompute_new` in source). Supports modes `trajectory`, `full`, `graph`, and `--eval-state`.

### Trajectory export examples

```bash
# One-hot expert policy + win/loss values
target/release/precompute \
  --export out_traj.jsonl \
  --export-mode trajectory \
  --games 10000 \
  --seed 42 \
  --hand-strategy stratified \
  --rules elemental,same \
  --elements random \
  --policy-format onehot \
  --value-mode winloss \
  --threads 8 --chunk-size 32

# MCTS soft policy labels
target/release/precompute \
  --export out_traj_mcts.jsonl \
  --export-mode trajectory \
  --games 5000 \
  --seed 42 \
  --policy-format mcts --mcts-rollouts 256
```

### Full export example

```bash
target/release/precompute \
  --export out_full.jsonl \
  --export-mode full \
  --seed 42 \
  --hand-strategy random \
  --rules none --elements none \
  --policy-format onehot --value-mode winloss
```

### Graph export example

```bash
target/release/precompute \
  --export graph_out \
  --export-mode graph \
  --seed 42 \
  --hand-strategy random \
  --rules none --elements random \
  --zstd --zstd-index --shards 2
```

References: `src/bin/precompute.rs:104`, `src/bin/precompute.rs:1497`, `src/bin/precompute.rs:1959`, `src/bin/precompute.rs:2137`.

## `query`

Looks up a solved entry in a persisted bincode DB (`load_db`) by reconstructed initial state key:

```bash
target/release/query \
  --db solved.bin \
  --rules none \
  --hands "A=1,2,3,4,5" "B=6,7,8,9,10" \
  --elements none
```

Reference: `src/bin/query.rs:16`, `src/persist.rs:126`.

## `tt-cli`

Scripted/demo CLI that loads cards, plays deterministic script, prints board/score/hash, then runs solver once.

```bash
target/release/tt-cli
```

Reference: `src/bin/tt-cli.rs:30`.

## `tt-gui`

Desktop app with selectable difficulty/rules/seed and solver-driven computer opponent.

```bash
cargo run --bin tt-gui
```

Reference: `src/bin/tt-gui.rs:3`, `src/bin/tt-gui.rs:994`.

---

## Export modes (`trajectory`, `full`, `graph`) and what each emits

## `trajectory`

- Emits sampled games, 9 pre-move states per game.
- Uses exact solver labels each ply; can vary behavior with `--play-strategy`, off-PV stepping, heuristic/mix, optional MCTS distribution export.
- Writer enforces deterministic game-id ordering across workers/chunk sizes.

Implementation anchors: `src/bin/precompute.rs:1498`, `src/bin/precompute.rs:1630`, `src/bin/precompute.rs:1820`.

## `full`

- Single sampled initial hand pair; DFS over reachable states.
- Emits one JSONL record per unique visited state.
- Labels each state via `search_root` using configured `value_mode`/`policy_format`.

Implementation anchors: `src/bin/precompute.rs:1959`, `src/bin/precompute.rs:2079`.

## `graph`

- Two-phase exact pipeline:
  1. Phase A BFS enumeration with dedupe.
  2. Phase B retrograde solve from ply 9 to root.
- Exports JSONL nodes via sink abstraction (plain/zstd/async/sharded) and emits `graph.manifest.json` with totals/checksums.

Implementation anchors: `src/solver/graph.rs:186`, `src/solver/graph.rs:396`, `src/solver/graph.rs:1005`, `src/bin/precompute.rs:2343`, `src/bin/precompute.rs:2439`.

---

## Data schemas and field-by-field ML interpretation

Primary export line schema is in `ExportRecord` (`src/bin/precompute.rs:343`) and `ExportLine` (`src/solver/graph.rs:591`).

### Common fields

- `game_id` (int)
  - Trajectory game index; always `0` in full/graph.
  - ML use: grouping/sequencing, split strategy (by game).
- `state_idx` / `turn` (0..9)
  - Ply index from initial state in that export context.
  - ML use: curriculum by game stage, temporal conditioning.
- `board` (9 cells)
  - Per cell: `cell`, `card_id|null`, `owner|null`, optional `element`.
  - ML use: board tensorization, occupancy/channel encoding.
- `hands` (`A`, `B` arrays)
  - Remaining card ids.
  - ML use: action masking, hidden-card considerations if partial observability introduced.
- `to_move` (`A`/`B`)
  - ML use: side-to-move plane/embedding.
- `rules` (booleans)
  - ML use: conditional policy/value head context.
- `policy_target`
  - `onehot`: `{card_id,cell}` best move.
  - `mcts`: map `"card-cell" -> p`.
  - ML use: supervised policy head target.
- `value_target`
  - Win/loss sign or margin (depending on `value_mode`).
  - ML use: scalar value head target.
- `value_mode` (`winloss`|`margin`)
  - ML use: decoding target semantics.
- `off_pv` (bool)
  - Game-level indicator in trajectory generation.
  - ML use: filter for on-policy-like vs exploratory trajectories.
- `state_hash` (32-char hex, 128-bit)
  - Deterministic identity key.
  - ML use: dedupe, leakage prevention, reproducibility auditing.

Reference docs: `DATA_MODEL.md:11`, `DATA_MODEL.md:101`.

### ML-specific caveat on target semantics

`value_target` for `margin` is converted relative to side-to-move (`src/bin/precompute.rs:444`), while score helper is `A-B` (`src/engine/score.rs:4`). Keep this consistent in training/evaluation.

---

## Eval/apply stdin-stdout API and use as environment service

`precompute --eval-state` supports line-delimited JSON requests over stdin with one JSON response per request.

Reference: `src/bin/precompute.rs:1275`.

### Eval request (search)

Input keys: `board`, `hands`, `to_move`, optional `turn`, `rules`, optional `board_elements`.

Output keys: `best_move?`, `value`, `margin`, `pv`, `nodes`, `depth`, `state_hash`.

### Apply request (transition)

If input includes `apply: {"card_id":...,"cell":...}`, output is state transition payload:

- `state` (new board/hands/to_move/turn/rules)
- `done`
- `outcome` (`mode`, `value`, `winner?`)
- `state_hash`

Implementation: `src/bin/precompute.rs:1093`, `src/bin/precompute.rs:1371`.

### Service usage pattern

```bash
# one eval line
printf '%s\n' '{"board":[{"cell":0,"card_id":null,"owner":null},{"cell":1,"card_id":null,"owner":null},{"cell":2,"card_id":null,"owner":null},{"cell":3,"card_id":null,"owner":null},{"cell":4,"card_id":null,"owner":null},{"cell":5,"card_id":null,"owner":null},{"cell":6,"card_id":null,"owner":null},{"cell":7,"card_id":null,"owner":null},{"cell":8,"card_id":null,"owner":null}],"hands":{"A":[1,2,3,4,5],"B":[6,7,8,9,10]},"to_move":"A","turn":0,"rules":{"elemental":false,"same":false,"plus":false,"same_wall":false}}' | target/release/precompute --eval-state
```

For production integration, keep process warm and stream newline-delimited requests over pipes/sockets via wrapper.

---

## Inventory of ML-useful signals available now

- Exact minimax value sign and best action (`search_root`).
- Optional soft policy from deterministic shallow MCTS (`policy_format=mcts`).
- Full principal variation in eval output (`pv`).
- Search effort stats (`nodes`, `depth`) in eval output.
- Deterministic state identity (`state_hash`).
- Rule flags and board elements as context variables.
- Off-PV indicator for trajectory-wide exploratory behavior.
- Graph-mode global signals in manifest: depth totals, terminal count, logical checksum, file integrity digests.

References: `src/solver/negamax.rs:25`, `src/bin/precompute.rs:461`, `src/bin/precompute.rs:1105`, `src/bin/precompute.rs:2440`, `src/solver/graph.rs:821`.

---

## Mapping to modern ML paradigms

## Imitation learning (behavior cloning)

- Use `trajectory` + `policy_format=onehot` for strict expert policy imitation.
- Add `policy_format=mcts` for richer soft targets.
- Stratify by rules/elements to avoid domain collapse.

## Value learning

- Train scalar head on `value_target` (`winloss` for bounded output, `margin` for richer signal).
- Use `full` or `graph` for denser unique-state coverage and lower trajectory correlation.

## Self-play RL

- Use eval/apply mode as deterministic environment service.
- Start with exact solver bootstrap targets then anneal toward model-driven self-play.
- Preserve exact solver as periodic oracle evaluator.

## Offline RL

- Feasible with trajectory/full exports, but current logs are missing behavior action/probability fields; treat as conservative imitation/value dataset unless schema is extended.

## Distillation

- Distill exact solver policy/value into compact NN for fast inference.
- Use `query`/solver oracle as teacher, NN as student, keep solver fallback for correctness-critical paths.

---

## End-to-end training pipeline recommendations

1. **Data generation**
   - Generate multiple seeds across rule mixes.
   - Prefer `graph` for exhaustive per-initial-state labels; use `trajectory` for gameplay-like sequences.
2. **Normalization/schema lock**
   - Freeze parser against current schema; include commit SHA in dataset metadata.
3. **Feature encoding**
   - Board occupancy/owner/card embeddings, hand card sets, to-move, rule flags, optional element planes.
4. **Targets**
   - Policy: onehot or mcts map over legal action mask.
   - Value: winloss sign or normalized margin.
5. **Train/val/test split**
   - Split by `state_hash` (or game/seed blocks) to avoid leakage.
6. **Validation**
   - Compare model move/value against solver on held-out states.
   - Track top-1 action match and value sign agreement.
7. **Deployment strategy**
   - Use NN for prior/move ordering; keep exact solver as final arbiter in strict mode.

---

## Dataset generation at scale, determinism, and reproducibility best practices

- Fix and log full command line, git SHA, Rust toolchain version.
- Keep `--seed` explicit; avoid implicit defaults in production pipelines.
- For trajectory determinism, keep writer ordering guarantees (`src/bin/precompute.rs:1632`).
- Validate byte-identical reproducibility with different `--threads`/`--chunk-size` in CI (already tested).
- For graph exports, keep manifest and integrity hashes with artifacts (`graph.manifest.json`).
- Prefer sharded zstd for large exports (`--shards`, `--zstd-frame-lines`, `--zstd-frame-bytes`).
- Deduplicate by `state_hash` when combining trajectories to reduce repeated supervision.

References: `tests/export_determinism_chunk_size.rs:12`, `tests/export_determinism_tt_bytes.rs:6`, `src/bin/precompute.rs:2390`.

---

## Benchmarks/tests/validation harnesses in repo and how to use them for model validation

No dedicated `benches/` currently, but test suite provides strong correctness/determinism harnesses.

### Core correctness

- Engine/rules: `tests/engine_tests.rs`, `tests/rule_edge_tests.rs`
- Solver behavior: `tests/solver_tests.rs`, `tests/tt_parity.rs`
- Incremental hash/make-unmake invariants: `tests/incremental_tests.rs`

### Export and API correctness

- Eval API behavior: `tests/eval_state_cli.rs`
- Export determinism: `tests/export_determinism_chunk_size.rs`, `tests/export_determinism_tt_bytes.rs`
- Graph parity/consistency: `tests/oracle_parity_graph_negamax.rs`, `tests/graph_export_consistency.rs`

### I/O pipeline correctness

- Graph writer/frame/sharding: `tests/graph_writer_frames.rs`, `tests/async_writer_tests.rs`, `tests/sharded_writer_unit.rs`, `tests/sharded_nodes_integration.rs`
- Stream persistence and CRC: `tests/persist_stream_tests.rs`

### Commands

```bash
# full suite
cargo test

# focused parity tests useful for ML label trust
cargo test --test oracle_parity_graph_negamax
cargo test --test graph_export_consistency
cargo test --test eval_state_cli
```

For model validation, add an offline harness that samples held-out JSONL states, calls solver (`--eval-state`), and compares NN logits/value against oracle outputs.

---

## Known limitations/gaps and high-impact extensions

### Required caveats from current analysis

1. `--quiet` appears declared but not wired in precompute path.
   - Declared at `src/bin/precompute.rs:195`; no downstream usage in that file.
2. Graph docs have stale wording in one place vs implementation.
   - `CLI_REFERENCE.md:122` says JSONL emission is planned, but graph JSONL + manifest are implemented (`src/solver/graph.rs:1005`, `src/bin/precompute.rs:2439`).
3. No explicit export schema version field currently.
   - Documented as missing: `DATA_MODEL.md:113`; export records have no schema/version field (`src/bin/precompute.rs:343`).
4. No explicit `action_taken` or behavior probability in exported trajectory records.
   - Records contain supervision targets (`policy_target`) but not the actual sampled step action/prob under play strategy (`src/bin/precompute.rs:1820`, move selection happens later at `src/bin/precompute.rs:1837`).

### Additional practical gaps

- `query` consumes legacy solved DB (`persist::load_db`) while current main workflow emphasizes JSONL export; docs/examples can confuse this boundary.
- No built-in benchmark harness for throughput/latency regressions (`benches/` absent).
- Export metadata is in command context/manifest, not embedded per record (except graph manifest).
- Trajectory-level off-policy details (strategy branch taken, sampling temperature/probability) are not emitted.

### High-impact extensions

- Add `schema_version` and `generator_metadata` to every record (or file header).
- Emit `action_taken`, `behavior_policy` (or sampled action probability), and `legal_moves_mask`.
- Add batched eval mode over stdin for high-throughput model-vs-oracle comparisons.
- Add canonical benchmark targets (solver NPS, export lines/s, eval latency).

---

## Practical roadmap for integrating a neural net while preserving exact solver behavior

### Phase 1: Non-invasive integration

- Train policy/value network from exported labels.
- Integrate NN only as move-ordering prior in negamax (`order_moves` augmentation), keeping exact search result authoritative.
- Keep existing TT and exact terminal evaluation untouched.

### Phase 2: Hybrid acceleration

- Use NN value for shallow cutoff heuristics under optional mode flags.
- Keep strict mode that always does full exact solve (baseline correctness path).
- Track discrepancy metrics between NN estimate and exact value.

### Phase 3: Productionized dual-path runtime

- Fast path: NN-only or NN-guided shallow search for responsiveness.
- Exact path: full negamax/graph for verification-critical calls.
- Add confidence gating: if NN confidence low or state out-of-distribution, fallback to exact solver.

### Phase 4: Continuous validation loop

- Nightly parity checks against oracle on stratified held-out states.
- Regression gates on action agreement/value sign agreement.
- Data refresh jobs with fixed seeds and manifest checksums to keep reproducibility.

This roadmap preserves current deterministic exact behavior as the source of truth while enabling incremental NN speedups.

---

## Appendix: concrete command set for ML dataset operations

```bash
# 1) Build
cargo build --release

# 2) Trajectory dataset (onehot)
target/release/precompute --export data/traj_onehot.jsonl --export-mode trajectory --games 200000 --seed 20260301 --threads 16 --chunk-size 32 --hand-strategy stratified --rules elemental,same,plus,same_wall --elements random --policy-format onehot --value-mode winloss

# 3) Trajectory dataset (soft policy)
target/release/precompute --export data/traj_mcts.jsonl --export-mode trajectory --games 100000 --seed 20260301 --threads 16 --hand-strategy stratified --rules elemental,same --elements random --policy-format mcts --mcts-rollouts 256 --value-mode winloss

# 4) Exhaustive graph snapshot (single initial state) with sharding
target/release/precompute --export data/graph_run --export-mode graph --seed 20260301 --hand-strategy random --rules none --elements random --shards 8 --zstd --zstd-index --zstd-frame-lines 131072 --zstd-frame-bytes 134217728

# 5) Eval service smoke test
printf '%s\n' '{"board":[{"cell":0,"card_id":null,"owner":null},{"cell":1,"card_id":null,"owner":null},{"cell":2,"card_id":null,"owner":null},{"cell":3,"card_id":null,"owner":null},{"cell":4,"card_id":null,"owner":null},{"cell":5,"card_id":null,"owner":null},{"cell":6,"card_id":null,"owner":null},{"cell":7,"card_id":null,"owner":null},{"cell":8,"card_id":null,"owner":null}],"hands":{"A":[1,2,3,4,5],"B":[6,7,8,9,10]},"to_move":"A","turn":0,"rules":{"elemental":false,"same":false,"plus":false,"same_wall":false}}' | target/release/precompute --eval-state --tt-bytes 64

# 6) Core validation subset
cargo test --test oracle_parity_graph_negamax --test graph_export_consistency --test eval_state_cli --test export_determinism_chunk_size --test export_determinism_tt_bytes
```
