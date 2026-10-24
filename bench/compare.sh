#!/usr/bin/env bash
# compare.sh MODEL OUTDIR DEVICE... — runs every prompt of prompts/ on the CPU
# oracle (herbert-bf16-chat) and on each CUDA device (herbert-cuda-chat),
# greedy, no system prompt; the full outputs go to OUTDIR/<prompt>-<engine>.log.
# p3 runs with thinking on (the template's <think> block), the others without.
# VARIANTS="name:--tune a=1 --tune b=2;other:..." runs each CUDA device once
# per variant (default: one variant "default" with no option); the CPU runs once.
# A variant's options starting with @BINARY run that build instead.
set -euo pipefail
MODEL=$1; OUT=$2; shift 2
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
MODEL=$(cd "$MODEL" && pwd)
cd "$(dirname "$0")"
VARIANTS=${VARIANTS:-default:}
PROMPTS=${PROMPTS:-"prompts/*.txt"}
for p in $PROMPTS; do
  name=$(basename "$p" .txt)
  ask=$(cat "$p")
  think="--no-think"; max=200
  case $name in p3-*) think=""; max=256;; esac
  # Both chat drivers retain the first ID with max=2; the CPU driver
  # truncates it with max=1. This fits the 4001-token fixture in 4096.
  case $name in pp*) max=2;; esac
  if [ ! -f "$OUT/$name-cpu.log" ]; then
    ~/herbert-jit-bf16/target/release/herbert-bf16-chat "$MODEL" --no-system $think --max-tokens $max --ask "$ask" > "$OUT/$name-cpu.log" 2>&1
    echo "$name cpu exit $?"
  fi
  IFS=';' read -ra vs <<< "$VARIANTS"
  for v in "${vs[@]}"; do
    vname=${v%%:*}; vargs=${v#*:}
    for d in "$@"; do
      # shellcheck disable=SC2086
      bin=../target/release/herbert-cuda-chat
      # A variant "name:@BINARY options" runs another build.
      if [[ $vargs == @* ]]; then bin=${vargs%% *}; bin=${bin#@}; vargs=${vargs#* }; [[ $vargs == @* ]] && vargs=""; fi
      "$bin" "$MODEL" --device "$d" --no-system $think --max-tokens $max $vargs --ask "$ask" > "$OUT/$name-cuda-$d-$vname.log" 2>&1
      echo "$name cuda-$d-$vname exit $?"
    done
  done
done
python3 _diff.py "$OUT"
