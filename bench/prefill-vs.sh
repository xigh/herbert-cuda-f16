#!/usr/bin/env bash
# prefill-vs.sh OUTDIR [ROUNDS] — exact-length, single-request prefill on the
# NVIDIA device against vLLM's own latency bench. 512/4000 fixtures tokenize to
# 513/4001 because the chat template contributes a token; vLLM uses those exact
# lengths. Full logs and clocks are preserved in OUTDIR.
set -euo pipefail
OUT=${1:?usage: prefill-vs.sh OUTDIR [ROUNDS]}
ROUNDS=${2:-5}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROMPTS="$ROOT/bench/prompts"
HVBIN=${HVBIN:-$ROOT/target/release/herbert-cuda-chat}
MODELS=${MODELS:-$HOME/models}
VLLM=${VLLM:-$HOME/venvs/bench-vllm-cu/bin/vllm}
DEVICE=${DEVICE:-0}
VARIANTS=${VARIANTS:-default:}
MODEL_SIZES=${MODEL_SIZES:-"0.8B 4B"}
PROMPT_LABELS=${PROMPT_LABELS:-"128 256 512 1024 2048 4000"}
mkdir -p "$OUT"

if [[ ! -x "$HVBIN" || ! -x "$VLLM" ]]; then
  echo "missing executable: HVBIN=$HVBIN VLLM=$VLLM" >&2
  exit 2
fi

nvidia-smi --query-gpu=name,clocks.sm,pstate --format=csv > "$OUT/clocks-start.txt"
"$VLLM" --version > "$OUT/vllm-version.txt" 2>&1
if [[ -n "${HERBERT_REVISION:-}" ]]; then
  printf '%s\n' "$HERBERT_REVISION" > "$OUT/herbert-revision.txt"
else
  (git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown) > "$OUT/herbert-revision.txt"
fi
printf 'engine\tmodel\tvariant\tprompt_tokens\therbert_tok_s\tvllm_tok_s\n' > "$OUT/summary.tsv"
IFS=';' read -ra variants <<< "$VARIANTS"

for round in $(seq 1 "$ROUNDS"); do
  echo "herbert round $round · $(uptime)"
  for model in $MODEL_SIZES; do
    for label in $PROMPT_LABELS; do
      case "$label" in
        128) actual=128 ;;
        256) actual=256 ;;
        512) actual=513 ;;
        1024) actual=1024 ;;
        2048) actual=2048 ;;
        4000) actual=4001 ;;
      esac
      prompt="$PROMPTS/pp$label.txt"
      [[ -f "$prompt" ]] || { echo "missing prompt fixture: $prompt" >&2; exit 2; }
      ask=$(cat "$prompt")
      for variant in "${variants[@]}"; do
        name=${variant%%:*}
        tune=${variant#*:}
        suffix=""
        [[ "$name" == default ]] || suffix="-$name"
        log="$OUT/r$round-$model-herbert-cuda$suffix-pp$label.log"
        tuning=()
        if [[ -n "$tune" ]]; then
          read -r -a tune_opts <<< "$tune"
          for opt in "${tune_opts[@]}"; do tuning+=(--tune "$opt"); done
        fi
        "$HVBIN" "$MODELS/Qwen3.5-$model" --device "$DEVICE" --no-system --no-think --max-tokens 1 --warmup 1 "${tuning[@]}" --ask "$ask" > "$log" 2>&1
        read -r got rate < <(python3 - "$log" <<'PY'
import re, sys
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("[turn"):
        m = re.search(r"(\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", line)
        print(m.group(1), m.group(2)); break
else:
    raise SystemExit("missing [turn] line")
PY
        )
        [[ "$got" == "$actual" ]] || { echo "$prompt token count $got != expected $actual" >&2; exit 1; }
        printf 'herbert\t%s\t%s\t%s\t%s\t-\n' "$model" "$name" "$got" "$rate" >> "$OUT/summary.tsv"
      done
    done
  done
done

echo "vLLM own latency bench · $(uptime)"
for model in $MODEL_SIZES; do
  for label in $PROMPT_LABELS; do
    case "$label" in
      128) actual=128 ;;
      256) actual=256 ;;
      512) actual=513 ;;
      1024) actual=1024 ;;
      2048) actual=2048 ;;
      4000) actual=4001 ;;
    esac
    VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLM" bench latency \
      --model "$MODELS/Qwen3.5-$model" --dtype bfloat16 --input-len "$actual" --output-len 1 \
      --batch-size 1 --num-iters-warmup 2 --num-iters 5 --max-model-len 4096 \
      --max-num-seqs 8 --gpu-memory-utilization 0.85 \
      --output-json "$OUT/vllm-$model-pp$label.json" > "$OUT/vllm-$model-pp$label.log" 2>&1
    python3 - "$OUT/vllm-$model-pp$label.json" "$model" "$label" "$actual" >> "$OUT/summary.tsv" <<'PY'
import json, sys
lat = json.load(open(sys.argv[1]))["avg_latency"]
print(f"vllm\t{sys.argv[2]}\t-\t{sys.argv[4]}\t-\t{int(sys.argv[4])/lat:.1f}")
PY
  done
done

python3 "$ROOT/bench/_pp.py" "$OUT" > "$OUT/summary.txt"
nvidia-smi --query-gpu=name,clocks.sm,pstate --format=csv > "$OUT/clocks-end.txt"
echo "ALLDONE-PREFILL-VS"
