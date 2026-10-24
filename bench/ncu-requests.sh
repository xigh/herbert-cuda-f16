#!/usr/bin/env bash
# Bounded counter capture of real 4B prefill GEMMs, after one whole warmup.
# Diagnostic only: request timings from this process are instrumented.
set -euo pipefail
MODEL=${1:?usage: ncu-requests.sh MODEL OUTDIR LABEL [--tune KEY=VALUE ...]}
OUT=${2:?usage: ncu-requests.sh MODEL OUTDIR LABEL [--tune KEY=VALUE ...]}
LABEL=${3:?usage: ncu-requests.sh MODEL OUTDIR LABEL [--tune KEY=VALUE ...]}
shift 3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=$(cd "$MODEL" && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
BIN=${BIN:-$ROOT/target/release/herbert-cuda-bench}
NCU=${NCU:-/opt/nvidia/nsight-compute/2025.1.1/ncu}
BLOCK=${BLOCK:-1024}
CACHE_CONTROL=${CACHE_CONTROL:-all}
REPLAY_MODE=${REPLAY_MODE:-kernel}
COUNT=${COUNT:-9}
CLOCK_CONTROL=${CLOCK_CONTROL:-none}
METRICS=${METRICS:-}
[[ $CLOCK_CONTROL == none || $CLOCK_CONTROL == base ]] || exit 2
[[ $BLOCK =~ ^[1-9][0-9]*$ && $COUNT =~ ^[1-9][0-9]*$ ]] || exit 2
for arg in "$@"; do
  [[ $arg != prefill.block=* ]] || { echo 'Use BLOCK to set prefill.block and the matching warmup skip.' >&2; exit 2; }
done
"$BIN" "$MODEL" --prepare --cases "$OUT/cases.json" --prompts "$ROOT/bench/prompts" --labels "$LABEL"
SKIP=$(python3 - "$MODEL/config.json" "$OUT/cases.json" "$BLOCK" "$COUNT" "$CACHE_CONTROL" "$REPLAY_MODE" "$CLOCK_CONTROL" "$OUT/capture.json" <<'PY'
import json, sys
config, cases_path, block, count, cache, replay, clock, output = sys.argv[1:]
c = json.load(open(config))
c = c.get('text_config', c)
assert c['hidden_size'] == 2560 and c['intermediate_size'] == 9216 and c['num_hidden_layers'] == 32, 'This filter targets Qwen3.5-4B only.'
cases = json.load(open(cases_path))
assert len(cases) == 1
m = len(cases[0]['ids'])
block = int(block)
blocks = (m + block - 1) // block
# 32 gate_up + 32 down + 24 DeltaNet projections per block.
# Skip the whole warmup, then the first two (DeltaNet) layers.
skip = blocks * 88 + 6
json.dump(dict(schema=1, diagnostic_only=True, model='Qwen3.5-4B',
    case=cases[0]['name'], prompt_tokens=m, block=block, first_block_m=min(m, block),
    warmup=1, rounds=1, matched_kernels_per_block=88, launch_skip=skip,
    launch_count=int(count), cache_control=cache, replay_mode=replay,
    clock_control=clock, requested_sm_clock_mhz=2805 if clock == 'none' else None,
    note='Nine matched launches: gate_up/down in layers 2,3,4; DeltaNet projection in layers 2,4,5. Timings under profiling are not benchmark latencies.'),
    open(output, 'w'), indent=2)
print(skip)
PY
)
[[ $(sed -n 's/^RmProfilingAdminOnly: //p' /proc/driver/nvidia/params) == 0 ]] || {
  echo 'GPU counters require driver permission (RmProfilingAdminOnly=0).' >&2; exit 2;
}
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw --format=csv > "$OUT/clocks-start.csv"
if [[ $CLOCK_CONTROL == none ]]; then
 [[ $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -n 1) == 2805 ]] || {
  echo 'Restore the campaign clock: sudo nvidia-smi -lgc 2805,2805' >&2; exit 2;
 }
fi
"$NCU" --version > "$OUT/ncu-version.txt"
"$NCU" --list-sections > "$OUT/ncu-sections.txt"
sha256sum "$BIN" "$ROOT/bench/ncu-requests.sh" > "$OUT/binary-tool.sha256"
cat /proc/driver/nvidia/params > "$OUT/nvidia-params.txt"
COLLECT=(--section LaunchStats --section Occupancy --section SpeedOfLight
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
  --section SchedulerStats --section WarpStateStats --section SourceCounters)
if [[ -n $METRICS ]]; then COLLECT=(--metrics "$METRICS"); fi
CMD=("$NCU" --clock-control "$CLOCK_CONTROL" --cache-control "$CACHE_CONTROL" --replay-mode "$REPLAY_MODE"
  "${COLLECT[@]}"
  --kernel-name 'regex:^(gemm_2560x(18432|12416)_bf16_|gemm_9216x2560_res_)'
  --launch-skip "$SKIP" --launch-count "$COUNT"
  --export "$OUT/counters" --force-overwrite
  "$BIN" "$MODEL" --cases "$OUT/cases.json" --out "$OUT/instrumented-requests.json"
  --rounds 1 --warmup 1 --outputs 1 --ctx 4096
  "$@" --tune "prefill.block=$BLOCK")
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
"${CMD[@]}" > "$OUT/ncu.log" 2>&1
"$NCU" --import "$OUT/counters.ncu-rep" --page raw --csv --print-units base > "$OUT/metrics.csv" 2> "$OUT/export-raw.log"
"$NCU" --import "$OUT/counters.ncu-rep" --page details > "$OUT/details.txt" 2> "$OUT/export-details.log"
"$NCU" --import "$OUT/counters.ncu-rep" --page source --print-source sass > "$OUT/driver-sass.txt" 2> "$OUT/export-sass.log"
"$NCU" --import "$OUT/counters.ncu-rep" --page source --print-source sass --csv > "$OUT/source.csv" 2> "$OUT/export-source-csv.log"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw --format=csv > "$OUT/clocks-end.csv"
echo ALLDONE-NCU-REQUESTS
