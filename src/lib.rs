//! herbert-cuda-bf16 — Qwen3.5 in bf16 on NVIDIA GPUs through the CUDA driver,
//! every compute kernel emitted as PTX at start-up.
//!
//! | module  | what it does |
//! |---------|--------------|
//! | `core`  | the model's `config.json`, bf16 conversions, the error type, the sampler |
//! | `model` | the safetensors checkpoint, read into bf16 row-major tensors |
//! | `chat`  | the Qwen3.5 template, tokens, streaming detokenization |
//! | `ptx`   | the PTX emitter every kernel is written with |
//! | `cuda`  | the driver (loaded at run time), the device, the kernels, the engine, the probe |
//! | `serve` | the Anthropic Messages API over HTTP, for the common bench client (feature `serve`) |
//!
//! Numerical contract, the CPU engine's: weights bf16, activations bf16,
//! accumulators f32, KV cache bf16. Nothing is quantized.

pub mod chat;
pub mod core;
pub mod cuda;
pub mod model;
pub mod ptx;
#[cfg(feature = "serve")]
pub mod serve;
