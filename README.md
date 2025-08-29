♠️ Triplecargo — Triple Triad Assistant & Solver (FF8)


Deterministic, high‑performance Rust implementation of the Final Fantasy VIII Triple Triad ruleset, designed to support:


- Perfect play solving (Negamax + αβ + TT).
- Full state‑space precomputation for instant queries.
- Analysis experiments (first‑player advantage, card ratings, rule impacts).
- AI/ML integration for lightweight move suggestion and imperfect‑information play.
Triplecargo is both a research‑grade solver and the foundation for a superhuman Triple Triad assistant — combining exact computation with modern AI techniques.


---

✨ Features

- Full ruleset: Basic, Elemental, Same, Plus, Same Wall, Combo cascades.
- Deterministic engine: fixed move ordering, reproducible cascades, stable hashing.
- High‑performance solver: Negamax + αβ pruning + transposition table.
- Precompute driver: bulk solve all reachable states for a given hand/ruleset.
- Persistence: append‑only batch writes, deterministic compaction, optional compression, checkpoint/resume.
- Analysis mode: RAM‑only runs for experiments (no I/O bottleneck).
- Throughput: ~10M nodes/sec, ~1M states/sec on a Ryzen 3600.
- Training data export (JSONL): trajectory or full state‑space, policy_format onehot|mcts, deterministic with fixed seed.

---

📂 Project layout

- Core engine: rules, state, hashing, scoring.
- Solver: negamax + αβ + TT, PV reconstruction.
- Precompute: state enumeration, parallel solve, persistence.
- Persistence: batch writes, compression, checkpointing.
- Binaries:
	- tt-cli: demo CLI.
	- precompute: bulk solver + DB writer.
- Tests: engine correctness, rule edge cases, solver determinism.

---

🧮 Data model

- Cards: loaded from data/cards.json, validated for ranges and uniqueness.
- Board: 3×3 fixed array, optional per‑cell element.
- Hands: 5 cards per player (engine: fixed arrays; export: shrinking lists as cards are played).
- Rules: independent toggles for Elemental, Same, Plus, Same Wall.
- Hashing: XOR‑based Zobrist, incremental updates, 128‑bit keys.

---
 
📤 Training data export (JSONL)

- Modes (via precompute):
  - --export-mode trajectory (default): exports a single 9‑state principal‑variation trajectory per sampled game.
  - --export-mode full: exports the entire reachable state space for one sampled hand pair (exhaustive).

- Determinism:
  - --seed controls sampling (hands/elements) and, if mcts is selected, the rollout RNG.
  - Same seed + flags → identical JSONL output (including Elemental layout and policy targets).

- Value targets:
  - --value-mode winloss (default): sign(value) from side‑to‑move perspective ∈ {-1, 0, +1}.
  - --value-mode margin: final A‑perspective score margin (A_cards − B_cards) ∈ [-9, +9].

- Hand sampling:
  - --hand-strategy random: uniform without replacement from all 110 cards.
  - --hand-strategy stratified: one card from each level band [1–2], [3–4], [5–6], [7–8], [9–10].

- Elements:
  - --elements none|random with deterministic per‑game element RNG when random is chosen.
  - Element letter codes on export: F (Fire), I (Ice), T (Thunder), W (Water), E (Earth), P (Poison), H (Holy), L (Wind).

- Policy targets:
  - --policy-format onehot (default): exported as a single move object {"card_id":..., "cell":...}.
  - --policy-format mcts: exported as a {"cardId-cell": prob} distribution over legal moves; --mcts-rollouts N (default 100) controls simulations.
  - In terminal states: onehot → policy_target omitted (null), mcts → {} empty map.

- JSONL writing:
  - Lines are appended as they are generated (streaming), with periodic flushes; no buffering of entire games.

CLI examples
- Trajectory export (onehot), 10 games, no elements:
  - target/release/precompute --export export.jsonl --export-mode trajectory --games 10 --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss
- Trajectory export (MCTS with 256 rollouts), stratified hands, with Elemental+Same and margin targets:
  - target/release/precompute --export export_mcts.jsonl --export-mode trajectory --games 10 --seed 42 --hand-strategy stratified --rules elemental,same --elements random --policy-format mcts --mcts-rollouts 256 --value-mode margin
- Full state‑space export (exhaustive) for one sampled hand pair:
  - target/release/precompute --export export_full.jsonl --export-mode full --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss

JSONL schema
Each line contains one state. Fields:

{
  "game_id": 12,                // sequential per sampled game (trajectory); full mode uses 0
  "state_idx": 0,               // 0..8 within a game trajectory (equals "turn")
  "board": [
    {"cell":0,"card_id":12,"owner":"A","element":"F"}, // element present only when rules.elemental=true; otherwise omitted
    {"cell":1,"card_id":null,"owner":null,"element":null},
    {"cell":2,"card_id":null,"owner":null,"element":"W"}
  ],
  "hands": {
    "A": [34,56,78,90,11],      // shrinking lists (5 → 0 across a trajectory)
    "B": [22,33,44,55,66]
  },
  "to_move": "A",
  "turn": 0,
  "rules": {
    "elemental": true,
    "same": true,
    "plus": false,
    "same_wall": false
  },

  // Policy target semantics:
  // onehot → single best move object:
  //   "policy_target": {"card_id":34,"cell":0}
  // mcts → distribution over legal moves:
  //   "policy_target": {"34-0":0.7,"56-1":0.3}
  "policy_target": {"34-0": 0.7, "56-1": 0.3},

  // Value target semantics controlled by --value-mode:
  // - winloss → {-1,0,+1} from side-to-move perspective (sign of solver value)
  // - margin  → integer margin A_cards − B_cards (A-perspective)
  "value_target": 1,
  "value_mode": "winloss",

  // Off-principal-variation sampling flag; true for all lines of off-PV games, false otherwise
  "off_pv": false,

  // 128-bit Zobrist hash of the state (hex string)
  "state_hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
}

Notes
- In trajectory mode, the solver searches full remaining depth at each ply to produce value_target consistent with final outcome.
- In full mode, enumeration exports all reachable states for the single sampled hand pair (useful for analysis). For mcts in full mode, rollout RNG is derived from (seed XOR state Zobrist) for traversal‑order invariance.

📑 Export Semantics

- 
State timing


	- Each JSONL line represents the state before a move is made.
	- At turn = t, the board has t occupied cells, and the side in to_move is about to play their (t+1)‑th card.
	- Hands shrink accordingly, but at the final turn = 8 there is still one card left in the mover’s hand and a policy_target showing where it will be placed.
- 
Policy targets


	- --policy-format onehot:
		- Exported as a single move object:

	"policy_target": {"card_id": 34, "cell": 0}



	- --policy-format mcts:
		- Exported as a distribution over legal moves:

	"policy_target": {"34-0": 0.7, "56-1": 0.3}



	- Terminal states:
		- onehot → policy_target omitted (null).
		- mcts → empty map {}.
- 
	- Off-PV stepping (strategy=mcts):
		- When --off-pv-strategy mcts is active:
			- If --policy-format mcts is selected, the root MCTS distribution computed for policy_target is reused for stepping.
			- Otherwise, a root MCTS distribution is computed with --mcts-rollouts at each ply using a deterministic per-ply seed.
			- The PV move’s probability is set to 0 and the distribution is renormalised; the next move is sampled deterministically via a per-ply RNG tied to (seed, game_id, turn).
			- Fallback: if the PV move had all probability mass (sum of non-PV probs == 0), the highest-probability non-PV move is selected deterministically.
Value targets


	- Controlled by --value-mode:
		- winloss: value ∈ {-1, 0, +1}, always from the side‑to‑move perspective.
			- +1 = side‑to‑move wins under perfect play.
			- 0 = draw.
			- −1 = side‑to‑move loses.
		- margin: integer final score difference (A_cards − B_cards), always from A’s perspective, independent of side‑to‑move.
	- Values are computed by solving the full remaining depth at each ply, so they are consistent with the final outcome.
- 
Elemental layout


	- When rules.elemental = true, each board cell includes an "element" field with one of:
		- "F" (Fire), "I" (Ice), "T" (Thunder), "W" (Water), "E" (Earth), "P" (Poison), "H" (Holy), "L" (Wind).
	- Otherwise, the "element" field is omitted.
- 
Metadata


	- game_id: sequential per sampled game in trajectory mode; fixed 0 in full mode.
	- state_idx: 0..8 within a trajectory (equals turn).
	- off_pv: boolean flag indicating off-principal-variation sampling; true for all lines of off-PV games, false otherwise.
	- state_hash: 128‑bit Zobrist hash of the state, hex string.

🕹️ Rules implemented

- Basic capture: placed card flips weaker adjacent opponents.
- Elemental: +1/−1 adjustments based on cell element.
- Same: 2+ equalities trigger flips.
- Same Wall: walls count as 10 for Same.
- Plus: equal sums trigger flips.
- Combo: cascades apply Basic rule only.

---

⚡ Performance & determinism

- Move ordering: Corners > Edges > Center, then cell index, then card id.
- Cascade order: BFS queue, ascending indices.
- TT: fixed‑size array, depth‑preferred replacement, full key verification.
- Progress: MultiProgress bars, states/sec, nodes/sec.
- Determinism: same state → same result, reproducible DBs.

---

🔮 Roadmap

Near‑term

- Query API: CLI/HTTP service for instant best‑move lookups from solved DBs.
- Analysis experiments:
	- First‑player advantage quantification.
	- Card/hand Elo ratings.
	- Elemental variance studies.
	- Rule interaction balance.

Mid‑term

- AI/ML integration:
	- NPZ export option for compact storage.
	- PyTorch Dataset utilities to load JSONL → tensors.
	- Train small policy/value nets for lightweight move suggestion.
	- Reinforced self‑play (AlphaZero‑lite) to refine policies.
- Imperfect‑info play:
	- Expectimax/MCTS with hidden hands.
	- Opponent modelling.

Long‑term

- Superhuman assistant:
	- Instant move suggestions in Open and Closed formats.
	- Configurable playstyles: Optimal, Robust, Exploitative.
	- Human‑like play modes for fun.
- Meta‑analysis:
	- “Best hand” search for each ruleset.
	- Card tier lists and balance insights.
	- Quantitative impact of Elemental RNG.

---

🧪 Testing scope

- Engine tests: rule correctness, determinism, allocation‑free hot paths.
- Solver tests: terminal values, PV determinism.
- Rule edge tests: Same, Plus, Same Wall, Combo cascades.
- Persistence tests: batch flush, compression, checkpoint/resume determinism.

---

📜 License


The code in this repository is provided under an open license (TBD).
Triple Triad belongs to Square Enix (Final Fantasy VIII). This is a clean‑room reimplementation for research and educational purposes.


---

🙏 Acknowledgements

- FF8’s Triple Triad ruleset by Square Enix.
- Chess/Go engine design patterns (negamax, αβ, Zobrist, TT).
- OpenAI GPT‑5 + Kilo Code for collaborative design and implementation.