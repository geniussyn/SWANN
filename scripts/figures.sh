#!/usr/bin/env bash
# Redraw every figure the paper includes, from the result roots named here.
# One place, so a re-measurement is followed by one command and not by memory.
set -eu
cd "$(dirname "$0")/.."
OUT=${OUT:-/root/swann-paper/figures}
CORPORA="gist-960 msmarco-1m openai-1536 qwen-2560 openai-3072 qwen-4096"
TITLES=("GIST 1M" "MS MARCO 1M" "OpenAI-1536 1M" "Qwen-2560 100K" "OpenAI-3072 1M" "Qwen-4096 100K")

# 1. the frontier, over every system's own knobs
ARGS=()
while IFS= read -r line; do
  [ "$line" = "--panel" ] && continue
  ARGS+=(--panel "$line")
done < <(python3 bench/fig_panels.py --root clean --only-ready)
python3 bench/fig_recall_qps.py --cols 5 --width 1.52 --xlo 0.90 \
  "${ARGS[@]}" --out "$OUT/recall_qps"

# 2. batch scaling: latency and throughput
python3 bench/fig_batch.py --root results/batch --corpora $CORPORA \
  --titles "${TITLES[@]}" --targets 0.95 --rows latency_ms qps --out "$OUT/batch"

# 3. what a query costs the memory system: rate, bytes, arithmetic
python3 bench/fig_batch.py --root results/batch --corpora $CORPORA \
  --titles "${TITLES[@]}" --targets 0.95 \
  --rows gb_per_s bytes_per_query flops_per_query --out "$OUT/memory"

# 4. the three mechanisms, one at a time
python3 bench/fig_ablate.py --root results/ablate --out "$OUT/ablate"

# 5. build time and storage
python3 bench/fig_cost.py --root results/buildstore2 --corpus msmarco-1m --out "$OUT/cost"

# 6. the architecture, which is a drawing and reads no results
python3 bench/fig_arch.py --out "$OUT/arch"
echo "FIGURES DONE"
