#!/usr/bin/env bash
# Everything the paper measures, in the order it has to happen.
#
# Point CORPORA at prepared corpus directories (offline/prep_*.py build them)
# and run this on an otherwise idle GPU. Every timing in the paper is best of
# five on a device with nothing else on it; a concurrent job silently inflates
# all of them, which is a mistake we made once and will not repeat.
set -u
ROOT="${ROOT:-/dev/shm/swann}"
OUT="${OUT:-results}"
CUVS_PY="${CUVS_PY:-/opt/cuvsenv}"
CORPORA="${CORPORA:-sift-128 glove-100 nytimes-256 gist-960 msmarco-1m fever-1m openai-1536 qwen-2560 openai-3072 qwen-4096}"
PROBES="1 2 4 8 16 32 64 128 256 512 1024"
EPS="0.01 0.02 0.05 0.08 0.12 0.20 0.30"
mkdir -p "$OUT/clean"

idle () {           # refuse to measure while anything else holds the device
  while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do
    echo "  waiting for the GPU to be free"; sleep 30
  done
}

for DS in $CORPORA; do
  [ -f "$ROOT/$DS/G_d64.npy" ] || { echo "skip $DS (no graph; build it first)"; continue; }
  echo "###### $DS"; idle
  POOL=nest4k; case $DS in qwen-*) POOL=random4k;; esac

  for EP in $EPS; do
    for HALF in "" "--half"; do
      python3 bench/entry_arms.py --data "$ROOT/$DS" --graph G_d64.npy \
        --split test --nq 0 --k 10 --pool $POOL --P 4096 --L 12 --B 64 --team 16 \
        --stop-eps $EP --cache --bound --exclusive --elide-visited --fanout branch \
        --arms distance --V 1 2 4 8 16 32 64 128 --repeats 5 $HALF \
        --out "$OUT/clean/${DS}_swann_e${EP}${HALF:+_h}.json" >/dev/null 2>&1 || true
    done
  done
  for EP in 0.01 0.05 0.20; do
    for HALF in "" "--half"; do
      python3 bench/entry_arms.py --data "$ROOT/$DS" --graph G_d64.npy \
        --split test --nq 0 --k 10 --pool $POOL --P 4096 --L 12 --B 16 --team 16 \
        --stop-eps $EP --cache --bound --exclusive --elide-visited --fanout branch \
        --arms distance --V 1 2 4 8 16 32 --repeats 5 $HALF \
        --out "$OUT/clean/${DS}_swann_B16_e${EP}${HALF:+_h}.json" >/dev/null 2>&1 || true
    done
  done

  for B in 3 5 8; do
    python3 bench/cuvs_rabitq.py --data "$ROOT/$DS" --lists 4096 --bits $B \
      --mode 0 1 --probes $PROBES --repeats 5 \
      --out "$OUT/clean/${DS}_rabitq_b$B.json" >/dev/null 2>&1 || true
  done
  PYTHONPATH="$CUVS_PY" python3 bench/cuvs_ivf.py --data "$ROOT/$DS" \
    --lists 4096 --probes $PROBES --arms ivf_flat ivf_pq --refine 2.0 4.0 \
    --repeats 5 --out "$OUT/clean/${DS}_ivf.json" >/dev/null 2>&1 || true

  # CAGRA at both precisions: it takes a half dataset too, and comparing our
  # fp16 arm against its fp32 one inflates every ratio.
  CM=inner_product; case $DS in sift-128|gist-960) CM=sqeuclidean;; esac
  for H in "" "--half"; do
    SUF=""; [ -n "$H" ] && SUF="_h"
    PYTHONPATH="$CUVS_PY" python3 bench/cagra_native.py --data "$ROOT/$DS" \
      --split test --k 10 --graph G_d64.npy --metric $CM --arms built $H \
      --itopk 32 64 96 128 192 256 384 512 768 1024 --width 1 2 4 \
      --algo single_cta multi_cta --repeats 5 \
      --out "$OUT/clean/${DS}_cagra${SUF}.json" >/dev/null 2>&1 || true
  done

  M=cosine; case $DS in sift-128|gist-960) M=euclidean;; esac
  for KB in 24 32 64; do
    python3 bench/ggnn_native.py --data "$ROOT/$DS" --split test --k 10 \
      --k-build $KB --measure $M --repeats 5 \
      --taus 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.5 2.0 3.0 \
      --out "$OUT/clean/${DS}_ggnn_k$KB.json" >/dev/null 2>&1 || true
  done
  python3 bench/hnsw_cpu.py --data "$ROOT/$DS" --split test --k 10 \
    --M 32 --efc 200 --threads 10 --repeats 5 \
    --efs 16 24 32 48 64 96 128 192 256 384 512 768 \
    --out "$OUT/clean/${DS}_hnsw.json" >/dev/null 2>&1 || true
done

echo "###### batch sweep"
python3 bench/pick_settings.py --root "$OUT/clean" --corpora $CORPORA \
  --targets 0.90 0.95 0.99 --out "$OUT/batch/settings.json"
for DS in $CORPORA; do
  idle
  POOL=nest4k; case $DS in qwen-*) POOL=random4k;; esac
  python3 bench/batch_torch.py --data "$ROOT/$DS" --corpus $DS --pool $POOL \
    --settings "$OUT/batch/settings.json" --batches 1 10 100 1000 10000 \
    --repeats 5 --out "$OUT/batch/${DS}_torch.json" >/dev/null 2>&1 || true
  PYTHONPATH="$CUVS_PY" python3 bench/batch_cuvs.py --data "$ROOT/$DS" \
    --corpus $DS --settings "$OUT/batch/settings.json" \
    --batches 1 10 100 1000 10000 --repeats 5 \
    --out "$OUT/batch/${DS}_cuvs.json" >/dev/null 2>&1 || true
done

echo "###### build time and storage"
for DS in $CORPORA; do
  idle
  PYTHONPATH="$CUVS_PY" python3 bench/build_storage.py --data "$ROOT/$DS" \
    --corpus $DS --out "$OUT/buildstore/${DS}_cuvs.json" >/dev/null 2>&1 || true
  python3 bench/build_time.py --data "$ROOT/$DS" --corpus $DS --methods SWANN \
    --out "$OUT/buildstore/${DS}_swann.json" >/dev/null 2>&1 || true
done
echo "REPRODUCE DONE -- draw with scripts/figures.sh"
