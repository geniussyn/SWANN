#!/usr/bin/env bash
# Every figure in the paper, from whatever reproduce.sh left in results/.
set -u
OUT="${OUT:-results}"
FIG="${FIG:-figures}"
mkdir -p "$FIG"

ARGS=()
while IFS= read -r line; do
  [ "$line" = "--panel" ] && continue
  ARGS+=(--panel "$line")
done < <(python3 bench/fig_panels.py --root "$(basename "$OUT")/clean" --only-ready)

python3 bench/fig_recall_qps.py --cols 5 --width 1.52 --xlo 0.90 \
  "${ARGS[@]}" --out "$FIG/recall_qps"
python3 bench/fig_speedup.py --root "$OUT/clean" --out "$FIG/speedup"
python3 bench/fig_ablate.py --root "$OUT/ablate" --out "$FIG/ablate" 2>/dev/null || \
  echo "  (no ablation data; run the mechanism sweep to draw it)"
python3 bench/fig_cost.py --root "$OUT/buildstore" --out "$FIG/cost"
python3 bench/fig_arch.py --out "$FIG/arch"

SIX="gist-960 msmarco-1m openai-1536 openai-3072 qwen-2560 qwen-4096"
TITLES=("GIST 1M" "MS MARCO 1M" "OpenAI-1536 1M" "OpenAI-3072 1M" "Qwen-2560 100K" "Qwen-4096 100K")
python3 bench/fig_batch.py --root "$OUT/batch" --corpora $SIX --titles "${TITLES[@]}" \
  --targets 0.95 --rows qps latency_ms --width 1.5 --out "$FIG/batch"
python3 bench/fig_batch.py --root "$OUT/batch" --corpora $SIX --titles "${TITLES[@]}" \
  --targets 0.95 --rows gb_per_s gflop_per_s bytes_per_query flops_per_query \
  --width 1.5 --out "$FIG/memory" \
  --note "Counted for SWANN; derived from the index structure for the IVF family. CAGRA's iteration count is internal and is not estimated."
echo "FIGURES DONE -> $FIG"
