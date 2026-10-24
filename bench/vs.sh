#!/usr/bin/env bash
# vs.sh OUTDIR ROUNDS — herbert-cuda against llama.cpp (CUDA and Vulkan, commit
# c841aeeb8) at the shape of p3: a prefill of 759 tokens and a decode of 255
# tokens at a depth of 759, Qwen3.5 in bf16, interleaved round after round;
# then, with VLLM=1, vLLM's own latency bench (its start-up is too long to
# interleave). Full outputs in OUTDIR.
#
# Per machine, from the environment:
#   LLCUDA   llama-bench built with CUDA ("" = none)      LLVK  llama-bench built with Vulkan
#   GGUF     directory of Qwen3.5-<size>-BF16.gguf        MODELS  directory of Qwen3.5-<size>
#   PAIRS    "llama_vulkan_device:herbert_device:label …" (e.g. "Vulkan1:nvidia:nv Vulkan0:amd:amd")
#   CUDAPAIR "herbert_device:label" compared with llama.cpp CUDA0 (e.g. "nvidia:nv")
#   SIZES    "0.8B 4B" by default;  TUNE  herbert --tune options of the "tuned" run ("" = none)
#   HVBIN    herbert-cuda-chat binary (default ../target/release/herbert-cuda-chat; the bf16
#            build: ../target-bf16/release/herbert-cuda-chat)
#   WARM=1   also both engines at the same temperature: herbert after a warm-up turn
#            (--warmup 1, "-warm"; llama-bench warms up by default) and llama-bench's
#            prefill with --no-warmup ("-cold", like herbert's default run)
set -u
OUT=$1; ROUNDS=$2
mkdir -p "$OUT"
cd "$(dirname "$0")"
SIZES=${SIZES:-0.8B 4B}
TUNE=${TUNE:-}
ask=$(cat prompts/p3-long.txt)
lb() { # tag binary device gguf
  "$2" -m "$4" -dev "$3" -ngl 99 -p 759 -n 0 -r 1 -o json > "$OUT/$1-pp.json" 2> "$OUT/$1-pp.err"
  "$2" -m "$4" -dev "$3" -ngl 99 -p 0 -n 255 -d 759 -r 1 -o json > "$OUT/$1-tg.json" 2> "$OUT/$1-tg.err"
}
hv() { # tag device size [options]
  local tag=$1 dev=$2 m=$3; shift 3
  "${HVBIN:-../target/release/herbert-cuda-chat}" "$MODELS/Qwen3.5-$m" --device "$dev" --no-system --max-tokens 256 "$@" --ask "$ask" > "$OUT/$tag.log" 2>&1
}
for r in $(seq 1 "$ROUNDS"); do
  echo "round $r · $(uptime)"
  for m in $SIZES; do
    gguf=$GGUF/Qwen3.5-$m-BF16.gguf
    if [ -n "${LLCUDA:-}" ] && [ -n "${CUDAPAIR:-}" ]; then
      lb "r$r-$m-llama-cuda-${CUDAPAIR#*:}" "$LLCUDA" CUDA0 "$gguf"
    fi
    for p in $PAIRS; do
      ldev=${p%%:*}; rest=${p#*:}; hdev=${rest%%:*}; label=${rest#*:}
      lb "r$r-$m-llama-vk-$label" "$LLVK" "$ldev" "$gguf"
      hv "r$r-$m-vk-$label-default" "$hdev" "$m"
      if [ "${WARM:-0}" = 1 ]; then
        "$LLVK" -m "$gguf" -dev "$ldev" -ngl 99 -p 759 -n 0 -r 1 --no-warmup -o json > "$OUT/r$r-$m-llama-vk-$label-cold-pp.json" 2> "$OUT/r$r-$m-llama-vk-$label-cold-pp.err"
        hv "r$r-$m-vk-$label-warm" "$hdev" "$m" --warmup 1
      fi
      if [ -n "$TUNE" ]; then
        # shellcheck disable=SC2086
        hv "r$r-$m-vk-$label-tuned" "$hdev" "$m" $TUNE
      fi
    done
  done
done
if [ "${VLLM:-0}" = 1 ]; then
  echo "vllm · $(uptime)"
  for m in $SIZES; do
    for len in 255 1; do
      VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/bench-vllm-cu/bin/vllm bench latency --model "$MODELS/Qwen3.5-$m" --dtype bfloat16 --input-len 759 --output-len $len \
        --batch-size 1 --num-iters-warmup 2 --num-iters 5 --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.85 \
        --output-json "$OUT/vllm-$m-out$len.json" > "$OUT/vllm-$m-out$len.log" 2>&1
      echo "vllm $m out$len exit $?"
    done
  done
fi
echo ALLDONE
