#!/usr/bin/env bash
# Index build time across corpora, then the figure.
#
# Portable on purpose: point it at a directory of prepared corpora and it writes
# one JSON per corpus plus a figure. Nothing else in this repository has to have
# been run first. On a machine where cuVS lives in its own tree, export its path
# before running:
#
#     CUVS_PY=/opt/cuvsenv bash bench/run_build_time.sh /dev/shm/gpuann
#
# The device name is recorded in every JSON, so results measured on different
# machines stay distinguishable when they are drawn together.
set -u
ROOT="${1:-/dev/shm/gpuann}"
OUT="${OUT:-results/buildtime}"
CUVS_PY="${CUVS_PY:-/opt/cuvsenv}"
mkdir -p "$OUT"

# label:directory, in the order the figure should read left to right
CORPORA=(
  "SIFT:sift-128"
  "GloVe:glove-100"
  "GIST:gist-960"
  "MS MARCO:msmarco-1m"
  "OpenAI-1536:openai-1536"
  "Qwen-2560:qwen-2560"
  "OpenAI-3072:openai-3072"
  "Qwen-4096:qwen-4096"
  "arXiv-4096:arxiv-4096"
)

for spec in "${CORPORA[@]}"; do
  label="${spec%%:*}"; dir="${spec##*:}"
  [ -f "$ROOT/$dir/manifest.json" ] || { echo "skip $label (not prepared)"; continue; }
  echo "=== $label"
  PYTHONPATH="$CUVS_PY" python3 bench/build_time.py \
    --data "$ROOT/$dir" --corpus "$label" \
    --out "$OUT/${dir}.json" || echo "  ($label failed, continuing)"
done

python3 bench/fig_build_time.py --glob "$OUT/*.json" \
  --order SIFT GloVe GIST "MS MARCO" OpenAI-1536 Qwen-2560 OpenAI-3072 \
          Qwen-4096 arXiv-4096 \
  --out "${FIG:-/root/swann-paper/figures/build_time}"
