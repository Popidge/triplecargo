# ♠️ Triplecargo — Triple Triad Assistant & Solver (FF8)

Deterministic, high-performance Rust implementation of Final Fantasy VIII's Triple Triad ruleset. Designed for exact solving, large-scale precomputation, analysis experiments, training-data export, and AI/ML integration.

Key capabilities:
- Perfect-play solving (Negamax + αβ + transposition table).
- Full state-space precomputation for instant queries.
- Deterministic, reproducible exports for training and analysis.
- Lightweight assistants and integration points for ML workflows.

---

Quickstart

1. Build (release):
   - cargo build --release

2. Binaries
   - Precompute / export: [`target/release/precompute`](target/release/precompute:1)
   - Query / lookup: [`target/release/query`](target/release/query:1)
   - Demo CLI: [`target/release/tt-cli`](target/release/tt-cli:1)

3. Card data
   - Card definitions are loaded from [`data/cards.json`](data/cards.json:1).

---

Short CLI summary

The primary user-facing tool is the precompute/export binary. Full CLI documentation (flags, defaults, examples) is in the dedicated [`CLI_REFERENCE.md`](CLI_REFERENCE.md:1). The data and export schemas are documented separately in [`DATA_MODEL.md`](DATA_MODEL.md:1).

Common invocations:
- Export training data: [`target/release/precompute`](target/release/precompute:1) --export export.jsonl --export-mode trajectory --games 100 --seed 42
- Single-state evaluation (stdin → stdout): echo '<state-json>' | [`target/release/precompute`](target/release/precompute:1) --eval-state
- Run the demo CLI: [`target/release/tt-cli`](target/release/tt-cli:1)

---

Overview & design goals

Triplecargo combines exact computation with tooling for ML research. Main priorities are correctness, determinism, high-throughput precomputation, and export formats suitable for downstream training and analysis.

---

Project layout (high level)

- Core engine: rules, state, hashing, scoring (in [`src/`](src/:1)).
- Solver: negamax + αβ + TT, PV reconstruction (in [`src/solver/`](src/solver/:1)).
- Precompute/export: driver and orchestration in [`src/bin/precompute.rs`](src/bin/precompute.rs:1) and [`src/solver/precompute.rs`](src/solver/precompute.rs:1).
- Graph pipeline: enumeration and retrograde solver (in [`src/solver/graph.rs`](src/solver/graph.rs:1)).
- Persistence: batch writes and streams (see [`src/persist.rs`](src/persist.rs:1), [`src/persist_stream.rs`](src/persist_stream.rs:1)).
- Binaries: precompute, query, tt-cli (see [`src/bin/`](src/bin/:1)).

---

Examples (abridged)

Trajectory (onehot), 10 games:
- [`target/release/precompute`](target/release/precompute:1) --export export.jsonl --export-mode trajectory --games 10 --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss

Full state-space export (exhaustive) for one sampled hand pair:
- [`target/release/precompute`](target/release/precompute:1) --export export_full.jsonl --export-mode full --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss

Graph (two-phase full-graph) for one sampled hand pair:
- [`target/release/precompute`](target/release/precompute:1) --export export_graph.jsonl --export-mode graph --seed 42 --hand-strategy random --rules none --elements random

---

Contributing / docs

See [`CLI_REFERENCE.md`](CLI_REFERENCE.md:1) for full CLI details and [`DATA_MODEL.md`](DATA_MODEL.md:1) for export and eval schemas. To propose documentation changes, open a PR targeting these files.

Roadmap (high level)

- Query API for instant best-move lookups.
- AI/ML integration: compact exports and model utilities.

---

License & Attribution

- Repository code: open license (TBD). Triple Triad rules belong to Square Enix (Final Fantasy VIII). This repository is a clean‑room reimplementation for research and educational purposes.

---

Acknowledgements

- FF8’s Triple Triad ruleset (Square Enix)
- Classic engine patterns: negamax, αβ, Zobrist hashing, transposition tables
- Inspiration & help from GPT-5/Kilo Code