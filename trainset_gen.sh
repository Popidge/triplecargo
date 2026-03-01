THREADS=11
CHUNK=32
ROLLOUTS=256

# NONE
target/release/precompute \
  --export data/exports/traj_train_none.jsonl \
  --export-mode trajectory \
  --games 200000 \
  --seed 20260301 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules none --elements none \
  --policy-format mcts --mcts-rollouts $ROLLOUTS \
  --value-mode winloss

# ELEMENTAL
target/release/precompute \
  --export data/exports/traj_train_elemental.jsonl \
  --export-mode trajectory \
  --games 200000 \
  --seed 20260301 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules elemental --elements random \
  --policy-format mcts --mcts-rollouts $ROLLOUTS \
  --value-mode winloss

# SAME
target/release/precompute \
  --export data/exports/traj_train_same.jsonl \
  --export-mode trajectory \
  --games 200000 \
  --seed 20260301 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules same --elements none \
  --policy-format mcts --mcts-rollouts $ROLLOUTS \
  --value-mode winloss

# SAME+PLUS
target/release/precompute \
  --export data/exports/traj_train_same_plus.jsonl \
  --export-mode trajectory \
  --games 200000 \
  --seed 20260301 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules same,plus --elements none \
  --policy-format mcts --mcts-rollouts $ROLLOUTS \
  --value-mode winloss

# SAME+ELEMENTAL
target/release/precompute \
  --export data/exports/traj_train_same_elemental.jsonl \
  --export-mode trajectory \
  --games 200000 \
  --seed 20260301 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules same,elemental --elements random \
  --policy-format mcts --mcts-rollouts $ROLLOUTS \
  --value-mode winloss

# PLUS+ELEMENTAL
target/release/precompute \
  --export data/exports/traj_train_plus_elemental.jsonl \
  --export-mode trajectory \
  --games 200000 \
  --seed 20260301 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules plus,elemental --elements random \
  --policy-format mcts --mcts-rollouts $ROLLOUTS \
  --value-mode winloss

# SAME+PLUS+ELEMENTAL
target/release/precompute \
  --export data/exports/traj_train_same_plus_elemental.jsonl \
  --export-mode trajectory \
  --games 200000 \
  --seed 20260301 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules same,plus,elemental --elements random \
  --policy-format mcts --mcts-rollouts $ROLLOUTS \
  --value-mode winloss
