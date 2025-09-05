# Data Model & Export Schema

This document describes the JSONL export format produced by the precompute/export binary ([`target/release/precompute`](target/release/precompute:1)) and the single-state evaluation (stdin→stdout) schema. It also documents the in-memory data model primitives (cards, board, hands, rules) and hashing notes.

---

JSONL Export (per-line)

Each exported line represents the game state immediately before a move is taken. Lines are independent JSON objects (JSONL).

Core fields
- game_id: integer — sequential per sampled game (trajectory mode). 0 in full mode.
- state_idx: integer — 0..8 within a trajectory (equals "turn").
- board: array[9] — list of 9 cell objects describing the board.
  - Cell object example:
    - {"cell":0,"card_id":12,"owner":"A","element":"F"} — element present only when rules.elemental=true.
- hands: object — maps "A" and "B" to arrays of remaining card ids (lists shrink across the trajectory).
- to_move: string — "A" or "B".
- turn: integer — 0..8 (number of moves played so far).
- rules: object — booleans for toggles (elemental, same, plus, same_wall).
- policy_target: object or omitted — policy label:
  - onehot → single best-move object: {"card_id":34,"cell":0}
  - mcts → map of "cardId-cell" → probability: {"34-0":0.7,"56-1":0.3}
  - Terminal states: onehot → omitted, mcts → {}.
- value_target: integer — label value (see value_mode).
- value_mode: string — "winloss" or "margin".
- off_pv: boolean — true for exported lines belonging to off‑PV games.
- state_hash: string — 128-bit Zobrist hash (hex string) for deterministic state identity.

Example JSONL object
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

Single-state evaluation (stdin → stdout)

This mode reads exactly one JSON object from stdin and writes exactly one JSON object to stdout. Invoke with [`target/release/precompute --eval-state`](target/release/precompute:1).

Input schema (subset of a JSONL line)
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

---

Data model primitives

Cards
- Validated and loaded from [`data/cards.json`](data/cards.json:1).
- Each card has an integer id used in exports and engine internals.
- Consumers should treat card ids as opaque integers and validate against the project's card DB when necessary.

Board
- Fixed 3×3 array (9 cells). Cells are indexed 0..8 in row-major order (top-left = 0).
- Each cell entry in exports contains:
  - cell: integer (0..8)
  - card_id: integer (id of the occupying card) or null for empty
  - owner: "A" | "B" | null
  - element: optional letter (F,I,T,W,E,P,H,L) present only when elemental rules are active
- Board elements may be provided separately via board_elements in the eval input; when present they must match any per-cell element fields in board[].

Hands
- Each player (A and B) has 5 cards at the start. Engine uses fixed arrays internally; exports show shrinking lists as cards are played.
- Hand order is unspecified for consumers; selection is by card_id.

Rules
- Independent boolean toggles: elemental, same, plus, same_wall
- Example representation in JSON: { "elemental": true, "same": false, "plus": true, "same_wall": false }
- Rule semantics and references are implemented across [`src/rules.rs`](src/rules.rs:1) and [`src/engine/apply.rs:224`](src/engine/apply.rs:224).

Hashing & state identity
- Zobrist hashing with XOR-based incremental updates.
- 128-bit keys (hex strings) are emitted as state_hash for deterministic state identity across runs.
- Hash updates are deterministic and depend on card positions, owners, elements, hands, rules, to_move, and turn.
- Consumers should use state_hash to de-duplicate or verify state identity across exports.

Determinism guarantees & ordering notes
- Export writer ensures byte-for-byte identical JSONL when seed + flags are identical, regardless of --threads or --chunk-size (ordering enforced by the single writer).
- The writer buffers out-of-order arrivals and emits lines in strict increasing game_id order.
- For machine consumers, prefer stable parsing (streaming JSONL readers) and validate state_hash when reproducibility matters.

Backwards compatibility & versioning
- At present there is no explicit export schema version field. Consumers should pin to a commit/sha or add schema versioning client-side.
- Planned: add schema/version field in a future update and emit a header/meta JSONL line when needed.

Implementation references
- Export driver & labels: [`src/bin/precompute.rs:343`](src/bin/precompute.rs:343)
- Graph/retrograde references: [`src/solver/graph.rs:186`](src/solver/graph.rs:186), [`src/solver/graph.rs:376`](src/solver/graph.rs:376)
- Persistence writer: [`src/persist_stream.rs`](src/persist_stream.rs:1)

---

Contact & notes
- If you find inconsistencies between the emitted JSONL and this document, please open an issue or PR.
- For implementation questions refer to the source files linked above.