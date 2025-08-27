# Triplecargo — Triple Triad Engine (FF8)

Deterministic, high‑performance Rust implementation of the Final Fantasy VIII Triple Triad ruleset, designed to support perfect play solving, full state-space precomputation, and instant best‑move queries.

- Core language: Rust 2021
- Rule toggles: Elemental, Same, Plus, Same Wall (independent switches)
- Determinism: Fixed move ordering, deterministic cascades, reproducible state hashing
- Data source: [data/cards.json](data/cards.json)
- Library‑first design with small binaries for CLI and precompute

Goals
- Correct, complete game engine with configurable rules
- Deterministic and allocation‑free hot paths suitable for exhaustive search
- Full solver implemented (Negamax + αβ + TT) with deterministic move ordering and PV
- Precompute phase stub prepared for future milestone


## Project layout

Core modules
- Types and shared enums: [src/types.rs](src/types.rs)
- Rule toggles configuration: [src/rules.rs](src/rules.rs)
- Cards loader and DB: [src/cards.rs](src/cards.rs)
- Board representation and neighbors: [src/board.rs](src/board.rs)
- Game state and legal moves: [src/state.rs](src/state.rs)
- Engine — rule application and cascades: [src/engine/apply.rs](src/engine/apply.rs)
- Engine — scoring: [src/engine/score.rs](src/engine/score.rs)
- State hashing for memoisation: [src/hash.rs](src/hash.rs)
- Public surface and re‑exports: [src/lib.rs](src/lib.rs)

Solver and precompute
- Solver module facade, limits, and SearchResult: [src/solver/mod.rs](src/solver/mod.rs)
- Negamax with αβ pruning, TT integration, and PV reconstruction: [src/solver/negamax.rs](src/solver/negamax.rs)
- Transposition table (Bound, TTEntry, depth‑preferred replacement): [src/solver/tt.rs](src/solver/tt.rs)
- Deterministic move ordering heuristics: [src/solver/move_order.rs](src/solver/move_order.rs)

Binaries and tests
- CLI demo: [src/bin/tt-cli.rs](src/bin/tt-cli.rs)
- Precompute driver stub: [src/bin/precompute.rs](src/bin/precompute.rs)
- Engine tests: [tests/engine_tests.rs](tests/engine_tests.rs)
- Solver tests: [tests/solver_tests.rs](tests/solver_tests.rs)
- Rule edge tests (Same, Same Wall, Plus, Combo, allocation assertions): [tests/rule_edge_tests.rs](tests/rule_edge_tests.rs)


## Data model (high level)

- Cards: Loaded at runtime from [data/cards.json](data/cards.json). Each card has top/right/bottom/left side values 1–10 and an optional element. JSON validation enforces side ranges and uniqueness by id and name.
- Board: 3×3 grid stored as a fixed 9‑slot array with optional occupant and optional per‑cell element. Deterministic neighbor order: Up, Right, Down, Left.
- Hands: Each player starts with 5 cards (no duplicates in a hand). Current implementation stores hands as fixed small arrays with Option slots for removed cards.
- Next player: Owner A or B, toggles each move.
- Rules: Independent boolean toggles for elemental, same, plus, same_wall with spec defaults set to off.


## Rules implemented

Core structure
- Players: 2 (A and B)
- Board: 3×3, 9 turns total, first player defaults to A
- Hands: 5 cards each, one placement per turn, alternate turns

Basic capture
- When a card is placed, compare its touching side against each orthogonally adjacent opponent card’s touching side.
- If the placed card’s side is strictly greater, the neighbor flips to the placer’s ownership.
- In the absence of Same/Plus flips, only the placed card can flip neighbors and there is no cascade.

Elemental (toggle)
- Some board cells may have an element.
- For each card on an elemental cell:
  - If the card has the same element: all sides +1 (capped at 10).
  - If the card has a different element: all sides −1 (floored at 1).
  - If the card has no element: all sides −1 (floored at 1).
- If the cell has no element: no change.
- Adjustments apply to both the placed card and any neighbors, each relative to the element of the cell they occupy.

Same (toggle)
- If two or more sides of the placed card are equal to the touching sides of adjacent opponent cards, all those matched neighbors flip.

Same Wall (toggle)
- If enabled, treat the board edge as a “wall” with value 10 for Same equality checks only.
- If the placed card’s side equals 10 and matches a wall, it contributes one equality toward the Same trigger.
- Walls never flip and are not considered for any other rule.

Plus (toggle)
- If the sum of the placed card’s side and an opponent neighbor’s touching side equals the sum of the placed card’s side and another opponent neighbor’s touching side, both those neighbors flip.
- Only opponent neighbors are considered; walls are excluded from Plus.

Combo / cascades
- If Same or Plus flips occur, enqueue those newly flipped cards and process a cascade:
  - For each dequeued newly flipped card, apply the Basic greater‑than rule to its neighbors.
  - Cascades repeat until no more flips.
  - Important: cascades use only the Basic rule, not Same/Plus again.

Scoring and end of game
- After 9 turns, all cells are filled.
- Score = (#A owned) − (#B owned).
- Positive score: A wins; negative: B wins; zero: draw.


## Determinism and performance

- Deterministic move ordering: legal moves are generated first by ascending board index, then by ascending card id in the current player’s hand.
- Deterministic cascade order: BFS queue over newly flipped indices sorted ascending to ensure reproducible cascade sequences.
- Elemental adjustments are computed per occupant per cell before comparisons; strictly greater/tie behavior is consistent and deterministic.
- Hot path is allocation‑free aside from small, bounded temporaries; core data structures are fixed‑size arrays for cache locality.
- State hashing returns a 128‑bit key produced from the board, the unordered hands, next player bit, and rule toggles for stable memoisation across runs. See [src/hash.rs](src/hash.rs).


## Elements and cards

Elements supported (matching JSON): Earth, Fire, Water, Poison, Holy, Thunder, Wind, Ice.

Cards are loaded at runtime from [data/cards.json](data/cards.json). The loader validates side ranges (1..=10), ensures unique ids and names, and builds:
- Dense id index for O(1) fetch by id
- Name‑to‑id map for direct lookups (case‑sensitive, as stored)


## Building, testing, and running

Prerequisites
- Rust toolchain (stable recommended)

Build and test
- Run unit tests:
  - cargo test

Run the CLI demo (loads JSON, constructs simple hands, applies a scripted sequence, prints board and score):
- cargo run --bin tt-cli

Run the precompute stub (demonstrates enumerating states and computing keys, without solving/persisting yet):
- cargo run --bin precompute


## Public usage

This crate exposes a minimal, stable surface that re‑exports types and helpers from the internal modules via [src/lib.rs](src/lib.rs). Typical external usage flow:
- Load cards from [data/cards.json](data/cards.json)
- Construct a game state (empty, with elements, or with hands)
- Enumerate legal moves deterministically
- Apply a move to get a new state
- Compute the score or check for terminal state
- Compute a zobrist‑like key for memoisation

See:
- State and legal moves: [src/state.rs](src/state.rs)
- Move application (rules): [src/engine/apply.rs](src/engine/apply.rs)
- Scoring: [src/engine/score.rs](src/engine/score.rs)
- Hashing: [src/hash.rs](src/hash.rs)


## Testing scope

Engine tests: [tests/engine_tests.rs](tests/engine_tests.rs)
- Legal move ordering determinism
- Basic capture strictness (no flips on ties, flips on strictly greater)
- Elemental adjustments (same element +1 with cap 10; different or no element −1 with floor 1) applied to both placed and neighbor cards based on their own cells
- End‑to‑end progression to terminal state and scoring integrity

Rule edge tests: [tests/rule_edge_tests.rs](tests/rule_edge_tests.rs)
- Same triggers flipping of all matched opponent neighbors
- Same Wall contributes equality at value 10; walls do not flip
- Plus triggers double flips when two sums match; walls excluded
- Combo cascades apply Basic rule only from newly flipped neighbors (deterministic BFS)
- Zero‑allocation assertion: legal_moves preallocates exact capacity to avoid reallocation

Solver tests: [tests/solver_tests.rs](tests/solver_tests.rs)
- Terminal value correctness from side‑to‑move perspective
- Determinism: identical value and principal variation across runs for the same state


## Roadmap and future milestones

Not implemented yet (prepared via stubs and structure):
- Precomputation
  - Full enumeration of reachable states for typical card pools/rule sets
  - Bulk solve and caching of outcomes
- Persistence
  - On‑disk database for solved states (format TBD)
  - Loader that maps solved DB into memory for instant queries
- Query API
  - Fast CLI or HTTP service that returns best moves from solved DB
- Analysis
  - Experiments for first‑player advantage, card ratings, rule interactions


## Design notes and decisions

- Reproducibility first: move order and cascade order are pinned; hashing is stable; default starting player is A.
- Modular toggles: Elemental, Same, Plus, and Same Wall are represented independently; they can be combined arbitrarily and are evaluated exactly as per the specification.
- Elemental semantics: When a cell has an element and the card does not, the card’s sides are reduced by 1 (floored at 1). On a non‑elemental cell, there is no adjustment.
- Walls participate only in Same when Same Wall is enabled, contributing equality at value 10; walls never flip and never participate in Plus.
- Basic rule is the only rule used during cascades; Same/Plus never re‑trigger within a cascade.
- Memory layout: fixed arrays and small value types along hot paths to support future exhaustive search performance.


## Solver (Negamax + αβ + TT)

The project includes a full-depth, deterministic solver based on Negamax with alpha–beta pruning and a transposition table (TT).

- Value convention: side-to-move perspective using i8. At terminal, score is A−B; if next is B, the terminal value is negated.
- Search depth: 9 − filled_count (full remaining plies).
- Transposition table: depth-preferred replacement with bounds (Exact/Lower/Upper) and optional best_move for ordering/PV.
- Deterministic move ordering: Corners > Edges > Center; then cell index; then card id. If TT provides a best_move, it is tried first.

Files
- Facade, limits, and SearchResult: [src/solver/mod.rs](src/solver/mod.rs)
- Negamax + αβ + TT and PV reconstruction: [src/solver/negamax.rs](src/solver/negamax.rs)
- Transposition table (Bound, TTEntry, InMemoryTT): [src/solver/tt.rs](src/solver/tt.rs)
- Move ordering heuristics: [src/solver/move_order.rs](src/solver/move_order.rs)
- CLI demo integration: [src/bin/tt-cli.rs](src/bin/tt-cli.rs)
- Tests for solver correctness and determinism: [tests/solver_tests.rs](tests/solver_tests.rs)

Usage outline
- Construct a GameState and load cards as usual.
- Instantiate a Solver and call its search method to get best move, value, PV, nodes, and depth. See [src/solver/mod.rs](src/solver/mod.rs) for the facade and [src/bin/tt-cli.rs](src/bin/tt-cli.rs) for a demonstration.

Determinism
- Stable move ordering and cascades, stable hashing ([src/hash.rs](src/hash.rs)), and depth-preferred TT replacement provide reproducible results.

Testing
- Engine tests continue to verify rules and determinism.
- Additional tests exercise solver terminal values and determinism: [tests/solver_tests.rs](tests/solver_tests.rs).

## License

The code in this repository is provided under an open license to be decided for distribution. If you intend to redistribute or use parts of it, ensure to add an explicit license file and headers as appropriate for your project or organisation.


## Acknowledgements

Triple Triad belongs to Square Enix (Final Fantasy VIII). This repository provides a clean‑room implementation of the game rules for research and educational purposes, including AI search and game analysis.