#!/usr/bin/env bash
# speed.sh MODEL OUTDIR ROUNDS DEVICE... — p3-long (759 prompt tokens, 255
# generated, thinking on, greedy) on the CPU oracle and on each CUDA device
# and variant, interleaved round after round; full outputs in
# OUTDIR/r<round>-<engine>.log. VARIANTS as in compare.sh; NOCPU=1 skips the CPU;
# PROMPT=p4-xlong takes prompts/p4-xlong.txt instead. A variant's options
# starting with @BINARY run that build instead; a variant's own --device
# replaces DEVICE (e.g. herbert-vk-chat's "nvidia").
set -u
MODEL=$1; OUT=$2; ROUNDS=$3; shift 3
mkdir -p "$OUT"
cd "$(dirname "$0")"
ask=$(cat prompts/${PROMPT:-p3-long}.txt)
VARIANTS=${VARIANTS:-default:}
IFS=';' read -ra vs <<< "$VARIANTS"
for r in $(seq 1 "$ROUNDS"); do
  echo "round $r · $(uptime)"
  if [ "${NOCPU:-0}" != 1 ]; then
    ~/herbert-jit-bf16/target/release/herbert-bf16-chat "$MODEL" --no-system --max-tokens 256 --ask "$ask" > "$OUT/r$r-cpu.log" 2>&1
  fi
  for v in "${vs[@]}"; do
    vname=${v%%:*}; vargs=${v#*:}
    for d in "$@"; do
      # shellcheck disable=SC2086
      bin=../target/release/herbert-cuda-chat
      # A variant "name:@BINARY options" runs another build .
      if [[ $vargs == @* ]]; then bin=${vargs%% *}; bin=${bin#@}; vargs=${vargs#* }; [[ $vargs == @* ]] && vargs=""; fi
      # A variant that names its own --device (another engine's selector) keeps it.
      dev=(--device "$d"); [[ " $vargs " == *" --device "* ]] && dev=()
      "$bin" "$MODEL" "${dev[@]}" --no-system --max-tokens 256 $vargs --ask "$ask" > "$OUT/r$r-cuda-$d-$vname.log" 2>&1
    done
  done
done
echo done
