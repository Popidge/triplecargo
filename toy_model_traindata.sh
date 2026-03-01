# Build
cargo build --release
mkdir -p data/exports

THREADS=11
CHUNK=32

RULES="same,plus,elemental"
ELEMENTS="random"

TAU=0.5
QMODE="margin"
VALUE_MODE="winloss"

# Train
target/release/precompute \
  --export data/exports/traj_train_spe_soft_exact.jsonl \
  --export-mode trajectory \
  --games 2000 \
  --seed 20260301 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules $RULES --elements $ELEMENTS \
  --policy-format soft_exact \
  --soft-exact-temperature $TAU \
  --soft-exact-qmode $QMODE \
  --value-mode $VALUE_MODE

# Val
target/release/precompute \
  --export data/exports/traj_val_spe_soft_exact.jsonl \
  --export-mode trajectory \
  --games 500 \
  --seed 20260302 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules $RULES --elements $ELEMENTS \
  --policy-format soft_exact \
  --soft-exact-temperature $TAU \
  --soft-exact-qmode $QMODE \
  --value-mode $VALUE_MODE

# Test
target/release/precompute \
  --export data/exports/traj_test_spe_soft_exact.jsonl \
  --export-mode trajectory \
  --games 500 \
  --seed 20260303 \
  --threads $THREADS --chunk-size $CHUNK \
  --hand-strategy stratified \
  --rules $RULES --elements $ELEMENTS \
  --policy-format soft_exact \
  --soft-exact-temperature $TAU \
  --soft-exact-qmode $QMODE \
  --value-mode $VALUE_MODE
