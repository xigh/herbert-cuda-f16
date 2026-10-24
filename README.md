# herbert-cuda-f16 — Qwen3.5 in IEEE f16 on NVIDIA GPUs, every kernel emitted as PTX

The IEEE f16 variant of [herbert-cuda-bf16](https://github.com/xigh/herbert-cuda-bf16): the same engine (Qwen3.5-0.8B and 4B, text, block prefill on the tensor cores, decode replayed as a CUDA graph), the same emitter, the same tools, with every 16-bit value in IEEE f16 instead of bf16. No CUDA C, no nvcc, no NVRTC, no cuBLAS: every compute kernel is PTX emitted at start-up by Rust (`src/ptx`) and handed to the driver, which is loaded at run time (`libcuda.so.1`, `nvcuda.dll` on Windows). Nothing of the CUDA toolkit is needed to build or to run.

Author: Philippe Anel. License: MIT, see `LICENSE`.

## The numerical contract

- **Weights**: read from the same bf16 Hugging Face checkpoint and converted to IEEE f16 at load, round to nearest even (`Checkpoint::h16`); the two f32 tensors of the checkpoint (`linear_attn.A_log`, `linear_attn.dt_bias`) stay in f32. The RoPE tables are converted the same way.
- **Activations, KV cache, DeltaNet states**: f16. Inside a kernel the arithmetic is f32: f16 is widened by `cvt.f32.f16` (exact) and each output is narrowed once by `cvt.rn` (round to nearest even; beyond 65 504, ±inf). `--tune act=f32` keeps the DeltaNet activations and states in f32 (the key keeps its inherited name: `act=bf16`, the default, means the engine's 16-bit format, f16 here).
- **Accumulators**: f32 by default. The prefill GEMMs run `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
- **`--tune gemm.f16=1`**: another contract. The prefill GEMMs accumulate in f16 over each K tile (`mma.sync … .f16.f16.f16.f16`) and add each tile's sum into the f32 accumulators; the decode keeps f32 accumulation (its matvecs switch to a variant module, `_wh`, with the same f16 × f16 → f32 arithmetic). It is faster at prefill and gives other tokens. It requires `gemm.wmma=0` and `gemm.schedule=0`.

This is no longer the contract of the CPU oracle ([herbert-jit-bf16](https://github.com/xigh/herbert-jit-bf16), bf16): correctness is judged by the first divergence of the greedy tokens against it (`bench/compare.sh`), knowing that some divergence is expected.

## The attention of this version

herbert-cuda-f16 was forked from herbert-cuda-bf16 **before** the attention corrections of herbert-cuda-bf16 v0.1.0 (`sdpa.gqa`, `sdpa.split=64`, `sdpa.pipe`, `sdpa.rows`). It has the former attention: the decode attention runs per query head with `sdpa.split=8` (`decode::sdpa_split`), and each prefill attention tile waits for its K and V copies before computing. The keys `sdpa.gqa`, `sdpa.pipe` and `sdpa.rows` do not exist here (`src/cuda/tuning.rs` rejects them as unknown). So its decode at very long context falls the way herbert-cuda-bf16's did before v0.1.0. On the RTX 4080, phase 5b of [herbert-cuda-bench](https://github.com/xigh/herbert-cuda-bench) measured the 0.8B decode of 128 tokens at 299.1 tok/s after a 10 000-token prompt and 133.5 tok/s after 60 000 tokens, the same as the bf16 engine of that time.

## Versions

This is the tree of development commit `0c7d88d`, the copy measured in phase 5 of herbert-cuda-bench (units `hc-f16` and `hc-f16-acc16`). Phase 4 of the campaign (bf16 against f16, short prompts) ran on `048c7da`; between the two, the engine changes only by `--ask-file` in `herbert-cuda-chat` (`git diff 048c7da 0c7d88d --stat`: `src/bin/herbert-cuda-chat.rs`, plus a campaign script and a notes file that are not part of this repository). As in herbert-cuda-bf16 since 2026-10-19, the default prefill is `gemm.auto=3`, `gemm.layout=xor`, `prefill.block=auto`.

From phase 5b of herbert-cuda-bench (`raw/2026-10-19-cuda-bf16-phase5b-prefill-long-herbert-contre-comparateurs/summary.txt`), prefill in tok/s:

| model, prompt | herbert-cuda-bf16 | herbert-cuda-f16 | herbert-cuda-f16 `gemm.f16=1` |
|---|---|---|---|
| 0.8B, 10 000 tokens | 53 141 | 52 903 | 59 635 |
| 0.8B, 60 000 tokens | 26 646 | 26 578 | 28 059 |
| 4B, 10 000 tokens | 10 599 | 10 547 | 12 635 |
| 4B, 60 000 tokens | 6 904 | 6 874 | 7 643 |

## Build and run

```sh
cargo check --bins --locked                   # anywhere, no CUDA SDK needed
cargo build --release --locked                # on the GPU machine
cargo build --release --locked --features serve   # adds herbert-cuda-serve
```

The binaries (`target/release/`):

```sh
herbert-cuda-probe                            # what the device declares, then measurements with emitted kernels
herbert-cuda-probe --emit /tmp/ptx            # no device: every kernel of the probe written as /tmp/ptx/<name>.ptx
herbert-cuda-chat ~/models/Qwen3.5-0.8B --no-system --no-think --ask 'What is the capital of France?'
herbert-cuda-chat ~/models/Qwen3.5-4B --tune gemm.f16=1 --no-system --no-think --ask 'What is the capital of France?'
herbert-cuda-chat ~/models/Qwen3.5-4B --ctx 61440 --no-system --no-think --max-tokens 128 --ask-file long-prompt.txt
herbert-cuda-serve ~/models/Qwen3.5-4B --ctx 4608 --port 8080      # feature serve
herbert-cuda-bench ~/models/Qwen3.5-4B --cases cases.json --out run.json   # shared-ID request timing
herbert-cuda-absmax ~/models/Qwen3.5-4B --out absmax.json             # the f16 range, see below
```

`MODEL_DIR` is the Hugging Face checkpoint of Qwen3.5 (`config.json`, `tokenizer.json`, the bf16 safetensors); the conversion to f16 happens at load.

`herbert-cuda-chat` options: `--device N` (default 0), `--ctx N` (positions reserved, default 4096), `--no-think`, `--max-tokens N` (default 1024), `--system TEXT | --no-system`, `--ask TEXT` and `--ask-file PATH` (repeatable, in order), `--temperature T`, `--dump-ptx DIR`, `--tune KEY=VALUE` (repeatable), `--profile` (every launch timed on the GPU, no graph), `--warmup N`, `--pause MS`. On stderr, per turn: `[ids]` in the format of the CPU oracle's `herbert-bf16-chat`, `[decoded]` and `[turn]` (counts and speeds). `herbert-cuda-serve` (feature `serve`) answers the Anthropic Messages API (`POST /v1/messages`), one request at a time, every request from an empty cache (`--device`, `--ctx` default 4608, `--tune`, `--no-think`, `--logits`, `--host` default 127.0.0.1, `--port` default 8080, `--name`). `herbert-cuda-bench` times greedy requests on pretokenized ids (`--prepare --cases FILE --prompts DIR --labels A,B`, then `--cases FILE --out FILE [--outputs N,…] [--rounds N] [--warmup N] [--ctx N] [--tune …]`).

The keys of `--tune` (`src/cuda/tuning.rs`): `matvec.rows`, `matvec.warps`, `matvec.depth`, `matvec.cache`, `sdpa.split` (default 8), `act`, `graph`, `prefill.block`, `gemm.tile`, `gemm.bk`, `gemm.stages`, `gemm.auto`, `gemm.carveout`, `gemm.layout`, `gemm.schedule`, `gemm.pipeline`, `gemm.down_short`, `gemm.ldmatrix`, `gemm.copy`, `gemm.wmma`, `gemm.f16`, `sdpa.bk`, `sdpa.softmax`, `rec.tc`. Their meaning is the one documented in the README of [herbert-cuda-bf16](https://github.com/xigh/herbert-cuda-bf16); the former default prefill is `--tune gemm.auto=0 --tune gemm.layout=pad --tune prefill.block=1024`.

## herbert-cuda-absmax: the f16 range of the model

Before porting a model to f16, one has to know whether its values fit: f16 overflows beyond 65 504, and loses precision below its normal range (2⁻¹⁴) down to its smallest subnormal (2⁻²⁴). `herbert-cuda-absmax MODEL_DIR --out FILE.json [--prompts DIR] [--labels p1-short,…] [--tune K=V]…` reports, for every weight tensor of the checkpoint (as stored, bf16 or f32) and for the activations of this engine's block prefill on real prompts (the activation buffers the kernels write, per layer, read back after each launch), the largest magnitude, the kernel that produced it, the count above 65 504, the nonzero counts below 2⁻¹⁴ and below 2⁻²⁴, and the non-finite values. It needs block prefill and the default `act`. `python3 bench/_absmax_summary.py FILE.json…` prints the summary per buffer. The tool was first run on the bf16 engine, before any f16 kernel existed (its header comment says so); in this repository it reads the f16 activations of this engine.

## What the emitted PTX looks like

`cargo run --release --locked --bin herbert-cuda-probe -- --emit DIR` runs on a machine without an NVIDIA GPU (it was run on an ARM Mac for this README) and writes every kernel of the probe as text: 480 kernels in 432 distinct files for the shapes of Qwen3.5-0.8B and 4B. Below, 26 consecutive lines of `gemm_2560x18432_bf16_64x128x32_w2x2_s2.ptx` (lines 229–254), the prefill GEMM of the 4B's fused gate/up projection (K = 2 560, N = 18 432) with 64×128 tiles, K = 32 per step, 2×2 warps and two stages, as the probe emits it (padded shared layout; the engine's default XOR layout permutes the same addresses). The main loop issues the `cp.async` copies of the next stage, then loads the fragments of the current one with `ldmatrix` and feeds them to `mma` in f16 with f32 accumulators. The `bf16` in module names is the tag of the 16-bit output inherited from herbert-cuda-bf16: in this engine it is f16, as the `mma` line shows.

```
	cp.async.cg.shared.global [%r87], [%rd25], 16;
	add.u32 %r88, %r59, %r80;
	cp.async.cg.shared.global [%r88], [%rd29], 16;
	cp.async.commit_group;
	add.s64 %rd9, %rd9, 64;
	add.s64 %rd13, %rd13, 64;
	add.s64 %rd17, %rd17, 64;
	add.s64 %rd21, %rd21, 64;
	add.s64 %rd25, %rd25, 64;
	add.s64 %rd29, %rd29, 64;
L1:
	not.pred %p3, %p2;
	@%p3 cp.async.commit_group;
	add.u32 %r80, %r80, 15360;
	setp.eq.u32 %p4, %r80, 30720;
	@%p4 mov.u32 %r80, 0;
	add.u32 %r89, %r67, %r79;
	add.u32 %r90, %r78, %r79;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r91, %r92, %r93, %r94}, [%r89];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r95, %r96, %r97, %r98}, [%r89+1280];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r99, %r100, %r101, %r102}, [%r90];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r103, %r104, %r105, %r106}, [%r90+1280];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r107, %r108, %r109, %r110}, [%r90+2560];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r111, %r112, %r113, %r114}, [%r90+3840];
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0, %f1, %f2, %f3}, {%r91, %r92, %r93, %r94}, {%r99, %r100}, {%f0, %f1, %f2, %f3};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f4, %f5, %f6, %f7}, {%r91, %r92, %r93, %r94}, {%r101, %r102}, {%f4, %f5, %f6, %f7};
```

To check every module with the toolkit's assembler: `herbert-cuda-probe --emit /tmp/ptx && for f in /tmp/ptx/*.ptx; do ptxas -arch=sm_89 -o /dev/null $f; done`.

## Measured on

An RTX 4080 16 GB (sm_89, 76 SM) in a Ryzen 9 9900X machine under Linux, NVIDIA driver 595.99.02, SM clock locked at 2 805 MHz and memory at 11 201 MHz, the card held in P0 under CUDA by the application profile `CudaNoStablePerfLimit` (without it the driver keeps the card in P2 during any CUDA context). Built with the stable Rust toolchain (1.98), release profile `opt-level=3`, no LTO. On macOS the crate compiles (`cargo check`) and the probe emits PTX, nothing runs. The emitted PTX targets sm_89 (PTX 8.5); the driver compiles it forward for newer GPUs, and on sm_80 … sm_88 the target line is lowered to the device's. Other GPUs are untested; the automatic GEMM policy and the per-model prefill block apply only to sm_89 with 76 SM.

## The code and the bench scripts

The modules are those of herbert-cuda-bf16 (`core`, `model`, `chat`, `ptx`, `cuda`, `serve`); the f16 port changes the emitter's 16-bit primitives (`src/ptx`), the kernels (`src/cuda/decode.rs`, `prefill.rs`, `gemm.rs`, `kernels.rs`, `dntc.rs`), the loader (`src/model`) and the engine (`src/cuda/engine.rs`). Some doc comments and identifiers still say bf16 where the value is now f16: they are inherited, the arithmetic is the one described above.

`bench/` holds the same tools as herbert-cuda-bf16: `compare.sh` (tokens against the CPU oracle, expected in `~/herbert-jit-bf16`), `speed.sh`, `vs.sh`, `prefill-vs.sh`, `request-vs.sh`, `profile-prefill.sh`, `trace-requests.sh`, `ncu-requests.sh` and their Python summaries, plus `_absmax_summary.py`. Models are looked up in `$MODELS` (default `~/models`). The campaign that measured this engine against llama.cpp, vLLM, SGLang and transformers is [herbert-cuda-bench](https://github.com/xigh/herbert-cuda-bench).
