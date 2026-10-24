#!/usr/bin/env bash
# profile-prefill.sh OUTDIR — profile every exact prompt fixture with the
# current CUDA binary; use as diagnosis, not as a throughput benchmark.
set -euo pipefail
OUT=${1:?usage: profile-prefill.sh OUTDIR}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HVBIN=${HVBIN:-$ROOT/target/release/herbert-cuda-chat}
MODELS=${MODELS:-$HOME/models}
DEVICE=${DEVICE:-0}
SIZES=${SIZES:-4B}
PROMPTS=${PROMPTS:-"128 256 512 1024 2048 4000"}
VARIANTS=${VARIANTS:-default:}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,pstate --format=csv > "$OUT/clocks-start.txt"
IFS=';' read -ra variants <<< "$VARIANTS"
for model in $SIZES; do
  for label in $PROMPTS; do
    ask=$(cat "$ROOT/bench/prompts/pp$label.txt")
    for variant in "${variants[@]}"; do
      name=${variant%%:*}
      tune=${variant#*:}
      suffix=""
      [[ "$name" == default ]] || suffix="-$name"
      tuning=()
      if [[ -n "$tune" ]]; then
        read -r -a tune_opts <<< "$tune"
        for opt in "${tune_opts[@]}"; do tuning+=(--tune "$opt"); done
      fi
      "$HVBIN" "$MODELS/Qwen3.5-$model" --device "$DEVICE" --no-system --no-think \
        --max-tokens 1 --warmup 1 --profile "${tuning[@]}" --ask "$ask" \
        > "$OUT/$model-pp$label$suffix-profile.log" 2>&1
    done
  done
done
nvidia-smi --query-gpu=name,clocks.sm,pstate --format=csv > "$OUT/clocks-end.txt"
echo "ALLDONE-PREFILL-PROFILE"
