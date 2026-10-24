#!/usr/bin/env bash
# CUDA timeline and emitted PTX for one pass over shared token-ID fixtures.
set -euo pipefail
MODEL=${1:?usage: trace-requests.sh MODEL OUTDIR [--tune KEY=VALUE ...]}
OUT=${2:?usage: trace-requests.sh MODEL OUTDIR [--tune KEY=VALUE ...]}
shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=$(cd "$MODEL" && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
BIN=${BIN:-$ROOT/target/release/herbert-cuda-bench}
NSYS=${NSYS:-nsys}
"$NSYS" --version > "$OUT/nsys-version.txt"
nvidia-smi --query-gpu=name,clocks.sm,pstate --format=csv > "$OUT/clocks.csv"
sha256sum "$BIN" > "$OUT/binary.sha256"
"$BIN" "$MODEL" --prepare --cases "$OUT/cases.json" --prompts "$ROOT/bench/prompts" --labels "${LABELS:-128,256,1024,4000}"
"$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o "$OUT/trace" \
  "$BIN" "$MODEL" --cases "$OUT/cases.json" --out "$OUT/requests.json" --rounds 1 --warmup 0 \
  --outputs 1 --ctx "${CTX:-4096}" --dump-ptx "$OUT/ptx" "$@" > "$OUT/nsys.log" 2>&1
"$NSYS" export --type sqlite --force-overwrite=true --output "$OUT/trace.sqlite" "$OUT/trace.nsys-rep" > "$OUT/export.log" 2>&1
python3 "$ROOT/bench/_nsys.py" "$OUT/trace.sqlite" | tee "$OUT/kernel-gaps.txt"
echo ALLDONE-TRACE-REQUESTS
