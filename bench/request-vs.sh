#!/usr/bin/env bash
# Shared-ID greedy request timing. vLLM before/after; Herbert variants alternate.
set -euo pipefail
OUT=${1:?usage: request-vs.sh OUTDIR [ROUNDS]}
ROUNDS=${2:-20}
VLLM_ROUNDS=${VLLM_ROUNDS:-$ROUNDS}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
BIN=${BIN:-$ROOT/target/release/herbert-cuda-bench}
VLLMPY=${VLLMPY:-$HOME/venvs/bench-vllm-cu/bin/python}
MODELS=${MODELS:-$HOME/models}
MODEL_SIZES=${MODEL_SIZES:-"0.8B 4B"}
LABELS=${LABELS:-128,256,512,1024,2048,4000}
OUTPUTS=${OUTPUTS:-1}
CTX=${CTX:-4096}
VARIANTS=${VARIANTS:-"auto:gemm.auto=1;candidate:gemm.auto=3 gemm.layout=xor prefill.block=auto"}
WARMUP=${WARMUP:-2}
REUSE=${REUSE:-0}
IFS=';' read -ra variants <<< "$VARIANTS"
if [[ "$REUSE" == 1 && ${#variants[@]} != 1 ]]; then
  echo 'REUSE=1 requires a single Herbert variant' >&2
  exit 1
fi
nvidia-smi --query-gpu=name,clocks.sm,pstate --format=csv > "$OUT/clocks-start.csv"
sha256sum "$BIN" > "$OUT/binary.sha256"
find "$ROOT/src" "$ROOT/bench" -type f \( -name '*.rs' -o -name '*.py' -o -name '*.sh' \) -exec sha256sum {} + > "$OUT/source.sha256"
sha256sum "$ROOT/Cargo.toml" "$ROOT/Cargo.lock" >> "$OUT/source.sha256"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv > "$OUT/gpu.csv"
uname -a > "$OUT/host.txt"
printf 'LABELS=%s\nOUTPUTS=%s\nCTX=%s\nROUNDS=%s\nVLLM_ROUNDS=%s\nWARMUP=%s\nREUSE=%s\nHONLY=%s\n' "$LABELS" "$OUTPUTS" "$CTX" "$ROUNDS" "$VLLM_ROUNDS" "$WARMUP" "$REUSE" "${HONLY:-0}" > "$OUT/command.txt"
printf '%s\n' "$VARIANTS" > "$OUT/variants.txt"
for model in $MODEL_SIZES; do
  cases="$OUT/cases-$model.json"
  "$BIN" "$MODELS/Qwen3.5-$model" --prepare --cases "$cases" --prompts "$ROOT/bench/prompts" --labels "$LABELS"
  if [[ ${HONLY:-0} != 1 ]]; then
    echo "vLLM before · $model · $(date -Is)"
    VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLMPY" "$ROOT/bench/_vllm_requests.py" --model "$MODELS/Qwen3.5-$model" \
      --cases "$cases" --out "$OUT/vllm-$model-before.json" --rounds "$VLLM_ROUNDS" --warmup "$WARMUP" --outputs "$OUTPUTS" --ctx "$CTX" \
      > "$OUT/vllm-$model-before.log" 2>&1
  fi
  processes=$ROUNDS
  sample_rounds=1
  if [[ "$REUSE" == 1 ]]; then processes=1; sample_rounds=$ROUNDS; fi
  if [[ "$REUSE" == 2 ]]; then
    python3 - "$VARIANTS" "$OUT/variants.json" <<'PYVAR'
import json, sys
variants = []
for v in sys.argv[1].split(';'):
    name, tune = v.split(':', 1)
    variants.append(dict(name=name, tune=tune.split()))
json.dump(variants, open(sys.argv[2], 'w'), indent=2)
PYVAR
    # Allocate the largest block once. All variants keep automatic N padding.
    allocate=$(python3 - "$OUT/variants.json" <<'PYVAR'
import json, sys
blocks = [int(key.split('=')[1]) for v in json.load(open(sys.argv[1])) for key in v['tune'] if key.startswith('prefill.block=')]
print(max(blocks or [1024]))
PYVAR
)
    "$BIN" "$MODELS/Qwen3.5-$model" --cases "$cases" --out "$OUT" --variants-file "$OUT/variants.json" \
      --rounds "$ROUNDS" --warmup "$WARMUP" --outputs "$OUTPUTS" --ctx "$CTX" \
      --tune gemm.auto=1 --tune "prefill.block=$allocate" > "$OUT/herbert-$model-variants.log" 2>&1
  else
  for round in $(seq 1 "$processes"); do
    echo "Herbert $model round $round/$ROUNDS · $(date -Is)"
    for ((vi=0; vi<${#variants[@]}; vi++)); do
      index=$vi
      (( round % 2 == 0 )) && index=$((${#variants[@]} - 1 - vi))
      variant=${variants[$index]}
      name=${variant%%:*}; opts=${variant#*:}
      tune=()
      if [[ -n "$opts" ]]; then
        read -ra keys <<< "$opts"
        for key in "${keys[@]}"; do tune+=(--tune "$key"); done
      fi
      "$BIN" "$MODELS/Qwen3.5-$model" --cases "$cases" --out "$OUT/herbert-$model-$name-r$round.json" \
        --rounds "$sample_rounds" --warmup "$WARMUP" --outputs "$OUTPUTS" --ctx "$CTX" "${tune[@]}" > "$OUT/herbert-$model-$name-r$round.log" 2>&1
    done
  done
  fi
  if [[ ${HONLY:-0} != 1 ]]; then
    echo "vLLM after · $model · $(date -Is)"
    VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLMPY" "$ROOT/bench/_vllm_requests.py" --model "$MODELS/Qwen3.5-$model" \
      --cases "$cases" --out "$OUT/vllm-$model-after.json" --rounds "$VLLM_ROUNDS" --warmup "$WARMUP" --outputs "$OUTPUTS" --ctx "$CTX" \
      > "$OUT/vllm-$model-after.log" 2>&1
  fi
done
nvidia-smi --query-gpu=name,clocks.sm,pstate --format=csv > "$OUT/clocks-end.csv"
python3 "$ROOT/bench/_requests.py" "$OUT" | tee "$OUT/summary.txt"
echo ALLDONE-REQUEST-VS
