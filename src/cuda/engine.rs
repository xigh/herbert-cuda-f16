//! The engine: the model on the device, one decode step captured once as a
//! CUDA graph and replayed for every token.
//!
//! What `load` does: reads the checkpoint, uploads the weights (the
//! projections of one input fused into one matrix: `[q | k | v]` of the
//! attention, `[qkv | z | a | b]` of the DeltaNet, `[gate | up]`), allocates
//! every state and activation once, emits and loads the kernels with the
//! [`Tuning`], and lists the step of one token as launches: the embedding,
//! every layer, the head, the argmax. The token and its position are read by
//! the kernels from the `state` buffer, so the same list serves every step:
//! captured into a graph (`graph=1`), or launched one by one, with an event
//! after each launch when profiling.
//!
//! The prompt goes by blocks of `prefill.block` rows: the projections are
//! GEMMs on the tensor cores (`cuda/gemm.rs`), the row-wise kernels are the
//! decode's in their `rows` mode, the DeltaNet walks the block's rows in
//! order and the attention is causal FlashAttention (`cuda/prefill.rs`); the
//! head runs on the last row only. With `prefill.block=1` the prompt goes
//! token by token through the decode step.

use super::decode as dc;
use super::dntc;
use super::gemm;
use super::prefill as pf;
use super::driver::CuDevicePtr;
use super::gpu::{Arena, Arg, Event, Graph, Gpu, Kernel};
use super::kernels::{self as kn, Out};
use super::tuning::Tuning;
use super::types::{AttnShape, DeltaShape, STATE_ARGMAX};
use crate::core::bf16::Bf16;
use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use crate::model::Checkpoint;
use crate::ptx::Emitted;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Options {
    /// The device ordinal.
    pub device: i32,
    /// Positions the KV cache holds: the ceiling of a conversation.
    pub capacity: usize,
    pub dump_ptx: Option<PathBuf>,
    /// `key=value` overrides of the [`Tuning`].
    pub tune: Vec<String>,
    /// Time every launch of the step with events.
    pub profile: bool,
    /// The caller reads the logits (sampling, `HERBERT_VK_TOP`): copied back
    /// after every step.
    pub logits: bool,
}

type Ptr = CuDevicePtr;

enum Mixer {
    Attention { qkv: Ptr, q_norm: Ptr, k_norm: Ptr, o: Ptr, k_cache: Ptr, v_cache: Ptr },
    Delta { proj: Ptr, conv_w: Ptr, a_log: Ptr, dt_bias: Ptr, norm: Ptr, out: Ptr, conv_state: Ptr, rec_state: Ptr },
}

struct Layer {
    input_norm: Ptr,
    post_norm: Ptr,
    gate_up: Ptr,
    down: Ptr,
    mixer: Mixer,
}

struct Weights {
    embed: Ptr,
    lm_head: Ptr,
    final_norm: Ptr,
    cos: Ptr,
    sin: Ptr,
    layers: Vec<Layer>,
}

/// Activations and scratch of one step.
struct Acts {
    state: Ptr,
    x: Ptr,
    xn: Ptr,
    qkv: Ptr,
    scores: Ptr,
    attn_out: Ptr,
    sdpa_partial: Ptr,
    gu: Ptr,
    h: Ptr,
    proj: Ptr,
    gates: Ptr,
    dn_out: Ptr,
    dn_y: Ptr,
    logits: Ptr,
    partial: Ptr,
}

/// Activations of a prefill block, `rows` rows each.
struct Block {
    rows: usize,
    allocated_rows: usize,
    tokens: Ptr,
    x: Ptr,
    xn: Ptr,
    qkv: Ptr,
    attn_out: Ptr,
    gu: Ptr,
    h: Ptr,
    proj: Ptr,
    conv_out: Ptr,
    gates: Ptr,
    dn_out: Ptr,
    dn_y: Ptr,
    /// `rec.tc`: the chunk operators (bf16) and `Γ_C` (f32) of every (chunk, value head).
    tc_mat: Ptr,
    tc_vec: Ptr,
}

/// The kernels of a prefill block.
#[derive(Clone, Copy)]
struct GemmChoice { kernel: usize, tile: gemm::Tile }

#[derive(Clone, Copy)]
struct GemmKernels { base: GemmChoice, small: GemmChoice, exact128: GemmChoice, short: GemmChoice, exact256: GemmChoice, medium: GemmChoice, wide: GemmChoice }

impl GemmKernels {
    fn select(self, m: u32) -> GemmChoice {
        if m == 128 { self.exact128 } else if m <= 128 { self.small } else if m == 256 { self.exact256 } else if m <= 256 { self.short }
        else if m >= 1024 { self.wide } else if m >= 512 { self.medium } else { self.base }
    }
}

#[derive(Clone, Copy)]
struct AttentionKernels { qkv: GemmKernels, rope: usize, flash: usize, out: GemmKernels }
#[derive(Clone, Copy)]
struct DeltaKernels { proj: GemmKernels, conv: usize, conv_state: usize, recurrence: usize, gated_norm: usize, out: GemmKernels }

#[derive(Clone, Copy)]
struct BlockKernels {
    embed: usize,
    norm: usize,
    swiglu: usize,
    gate_up: GemmKernels,
    down: GemmKernels,
    attn: Option<AttentionKernels>,
    delta: Option<DeltaKernels>,
    /// `rec.tc`: the chunk preparation and the state pass, replacing the recurrence.
    tc: Option<(usize, usize)>,
    /// The decode's norm and head, on the last row.
    last_norm: usize,
    head: usize,
    arg1: usize,
    arg2: usize,
}

/// One launch of the step.
struct Op {
    kernel: usize,
    grid: (u32, u32, u32),
    args: Vec<Arg>,
}

/// The per-launch GPU times of the profiled steps.
struct Profile {
    events: Vec<Event>,
    /// Per kernel name, the time of each step (summed over its launches), seconds.
    steps: HashMap<String, Vec<f64>>,
    calls: HashMap<String, usize>,
    totals: Vec<f64>,
    walls: Vec<f64>,
}

/// What `prefill_absmax` records, per (layer, buffer): the largest magnitude, the elements above the f16
/// maximum (65 504), the nonzero elements below the f16 normal range (2^-14) and below its smallest
/// subnormal (2^-24, flushed to zero in f16), the non-finite ones, and the count; and the kernel that
/// last raised the maximum.
#[derive(Default)]
pub struct AbsMax {
    pub rows: std::collections::BTreeMap<(Option<usize>, String), AbsRow>,
}

#[derive(Default, Clone)]
pub struct AbsRow {
    pub max: f32,
    pub over_f16: u64,
    pub below_normal: u64,
    pub below_subnormal: u64,
    pub non_finite: u64,
    pub count: u64,
    pub max_kernel: String,
}

impl AbsMax {
    fn add(&mut self, layer: Option<usize>, name: &str, kernel: &str, v: &[u16]) {
        let r = self.rows.entry((layer, name.to_string())).or_default();
        for &e in v {
            let x = half::f16::from_bits(e).to_f32().abs();
            if !x.is_finite() { r.non_finite += 1; continue; }
            if x > r.max { r.max = x; r.max_kernel = kernel.to_string(); }
            if x > 65504.0 { r.over_f16 += 1; }
            if x != 0.0 && x < 6.103_515_6e-5 { r.below_normal += 1; }
            if x != 0.0 && x < 5.960_464_5e-8 { r.below_subnormal += 1; }
        }
        r.count += v.len() as u64;
    }
}

pub struct Engine {
    config: Config,
    tuning: Tuning,
    pub attn: AttnShape,
    pub delta: DeltaShape,
    capacity: usize,
    w: Weights,
    a: Acts,
    _arena: Arena,
    /// (pointer, bytes) of every state a new conversation zeroes.
    states: Vec<(Ptr, usize)>,
    kernels: Vec<Kernel>,
    ops: Vec<Op>,
    blk: Option<(Block, BlockKernels)>,
    graph: Option<Graph>,
    profile: Option<Profile>,
    /// The prefill blocks' profile (`--profile`), one entry per block.
    profile_prefill: Option<Profile>,
    pos: usize,
    argmax: u32,
    need_logits: bool,
    logits_cpu: Vec<u16>,
    pub load_seconds: f64,
    pub mount_seconds: f64,
    /// Driver module loading/JIT, within mount_seconds; cache hits included.
    pub module_load_seconds: f64,
    /// Last: the fields drop in order, and the modules, graph, events and
    /// buffers above need the context the device holds.
    gpu: Gpu,
}

fn concat(parts: Vec<Vec<Bf16>>) -> Vec<Bf16> {
    let mut v = Vec::with_capacity(parts.iter().map(Vec::len).sum());
    for p in parts {
        v.extend_from_slice(&p);
    }
    v
}

/// The kernels of the engine, loaded once per distinct name.
struct Loader<'a> {
    gpu: &'a Gpu,
    list: Vec<Kernel>,
    by_name: HashMap<String, usize>,
    dump: Option<PathBuf>,
    gemm_carveout: Option<u32>,
    gemm_down_short: Option<(gemm::Tile, gemm::Tile)>,
    module_load_seconds: f64,
}

impl Loader<'_> {
    fn get(&mut self, e: Emitted) -> Result<usize> {
        if let Some(&i) = self.by_name.get(&e.name) {
            return Ok(i);
        }
        if let Some(dir) = &self.dump {
            e.dump(dir)?;
        }
        let module_started = Instant::now();
        let k = self.gpu.load(&e)?;
        self.module_load_seconds += module_started.elapsed().as_secs_f64();
        if e.name.starts_with("gemm_") {
            if let Some(percent) = self.gemm_carveout { self.gpu.set_shared_carveout(&k, percent)?; }
            if self.gemm_carveout.is_some() || self.dump.is_some() { eprintln!("[launch] {}: {}", k.name, self.gpu.launch_resources(&k)?); }
        }
        if k.local > 0 {
            eprintln!("[ptxas] {}: {} registers, {} bytes of spill per thread", k.name, k.regs, k.local);
        }
        self.list.push(k);
        self.by_name.insert(e.name.clone(), self.list.len() - 1);
        Ok(self.list.len() - 1)
    }
}

fn load_gemm(ld: &mut Loader<'_>, model_h: u32, k: u32, n: u32, out: Out, tile: gemm::Tile, auto: u32) -> Result<GemmKernels> {
    // Policy 3 was measured on the 76-SM RTX 4080. Other hardware/models
    // retain policy 1 rather than extrapolating the 4B warp choice.
    let measured_4b = matches!(auto, 3..=6) && model_h == 2560 && ld.gpu.cc == (8, 9) && ld.gpu.sms == 76;
    let experimental_08b = auto == 6 && model_h == 1024 && ld.gpu.cc == (8, 9) && ld.gpu.sms == 76;
    let measured_08b = auto == 3 && model_h == 1024 && ld.gpu.cc == (8, 9) && ld.gpu.sms == 76;
    let swizzle = tile.swizzle;
    let (schedule, pipeline) = (tile.schedule, tile.pipeline);
    let (ldmatrix, async_copy, wmma, f16) = (tile.ldmatrix, tile.async_copy, tile.wmma, tile.f16);
    let down_short = if model_h == 2560 && k == 9216 && n == 2560 && out == Out::Residual { ld.gemm_down_short } else { None };
    let mut load = |mut tile: gemm::Tile| -> Result<GemmChoice> {
        tile.swizzle = swizzle;
        tile.ldmatrix = ldmatrix;
        tile.async_copy = async_copy;
        tile.wmma = wmma;
        tile.f16 = f16;
        if auto == 0 || (model_h == 2560 && tile.bm == 64 && tile.bn == 128) {
            tile.schedule = schedule;
            tile.pipeline = pipeline;
        }
        Ok(GemmChoice { kernel: ld.get(gemm::gemm(k, n, out, tile))?, tile })
    };
    let base = load(tile)?;
    if auto != 0 {
        let small = load(gemm::Tile::T64_K64_S2)?;
        let output_4b = n == 2560 && out == Out::Residual && matches!(k, 4096 | 9216);
        let short_tile = if measured_4b && auto == 4 {
                if k == 2560 && n == 18432 { gemm::Tile::T64X128_K64_S2 }
                else if output_4b { gemm::Tile::T64 }
                else { gemm::Tile::T64_K64_S2 }
            }
            else if measured_4b { gemm::Tile::T64_W1X4_K64_S2 }
            else if auto == 2 && output_4b { gemm::Tile::T64_K32_S2 }
            else if auto == 2 && k == 2560 && n == 10240 { gemm::Tile::T64 }
            else { gemm::Tile::T64_K64_S2 };
        let medium_tile = if measured_4b { gemm::Tile::T64X128_W2X2_S2 }
            else if auto == 2 && output_4b { gemm::Tile::T64 } else { gemm::Tile::T64X128 };
        let wide_tile = if auto == 6 && (measured_4b || experimental_08b) {
            if n % 256 == 0 && n >= 2560 { gemm::Tile::T64X256_S2 }
            else { gemm::Tile::T64X128_W2X2_S2 }
        } else if measured_08b { gemm::Tile::T64X128_W2X2_S2 } else { medium_tile };
        let short = load(short_tile)?;
        // XOR removes the shared-memory padding: the original K64 tile
        // then wins at 256, whereas padded weights favor the larger tile.
        let exact256 = if measured_4b && auto == 3 && swizzle { small }
        else if measured_4b && (auto == 3 || auto >= 5) {
            load(if k == 2560 && matches!(n, 18432 | 12416) { gemm::Tile::T128_W4X4_K64_S2 }
                 else if output_4b { gemm::Tile::T64 }
                 else { gemm::Tile::T64_K64_S2 })?
        } else { short };
        let wide = load(wide_tile)?;
        let exact128 = if let Some((tile, _)) = down_short { load(tile)? } else { small };
        let exact256 = if let Some((_, tile)) = down_short { load(tile)? } else { exact256 };
        Ok(GemmKernels { base, small, exact128, short, exact256, medium: load(medium_tile)?, wide })
    } else {
        let exact128 = if let Some((tile, _)) = down_short { load(tile)? } else { base };
        let exact256 = if let Some((_, tile)) = down_short { load(tile)? } else { base };
        Ok(GemmKernels { base, small: base, exact128, short: base, exact256, medium: base, wide: base })
    }
}

impl Engine {
    pub fn load(model_dir: &Path, o: &Options) -> Result<Engine> {
        let started = Instant::now();
        let gpu = Gpu::open(o.device)?;
        eprintln!("[cuda] {} — sm_{}{}, {} SMs, driver {}.{}", gpu.name, gpu.cc.0, gpu.cc.1, gpu.sms, gpu.driver_version / 1000, (gpu.driver_version % 1000) / 10);
        let mut tuning = Tuning::with(&o.tune)?;
        let mut ck = Checkpoint::load(model_dir)?;
        let c = ck.config.clone();
        if tuning.prefill_block == 0 {
            tuning.prefill_block = if gpu.cc == (8, 9) && gpu.sms == 76 {
                match c.hidden_size { 2560 => 2048, 1024 => 4096, _ => 1024 }
            } else { 1024 };
        }
        eprintln!("[tuning] {}", tuning.summary());
        eprintln!(
            "[loader] {} — hidden {}, {} layers, {} heads / {} kv × {}, intermediate {}, vocab {}, rotary {}, linear k {}×{} v {}×{}, tied head {}",
            c.model_type,
            c.hidden_size,
            c.num_layers,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.intermediate_size,
            c.vocab_size,
            c.rotary_ndims,
            c.linear_num_key_heads,
            c.linear_key_head_dim,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.tie_word_embeddings
        );
        if !c.attn_output_gate {
            return Err(HerbertError::Config("this engine serves the gated attention of Qwen3.5 (attn_output_gate)".into()));
        }
        let (h, inter, vocab) = (c.hidden_size, c.intermediate_size, c.vocab_size);
        let cap = o.capacity.min(c.max_position_embeddings);
        let attn = AttnShape::from_config(&c, cap);
        let tile = tuning.tile()?;
        let blocks = tuning.prefill_block > 1;
        // `gemm.f16` (herbert-cuda-bf16's f16 experiment) keeps its kernel variants: with f16 everywhere
        // its 16-bit stores are the default ones, and `gemm.f16=1` still selects the f16 accumulation.
        let f16 = tuning.gemm_f16;
        // Every weight is already IEEE f16 (`Checkpoint::h16`); `to_w` stays for the call sites.
        let to_w = |v: Vec<Bf16>| -> Vec<Bf16> { v };
        // The GEMM's N is a multiple of its tile: the DeltaNet projection padded.
        let delta = DeltaShape::from_config(&c, tuning.act).padded(if blocks { if tuning.gemm_auto != 0 { 128 } else { tile.bn } } else { 1 });
        let has_delta = (0..c.num_layers).any(|l| c.is_deltanet_layer(l));
        let act = delta.act.bytes() as usize;

        // ── weights ──
        let mut arena = Arena::default();
        let embed_host = ck.h16("embed_tokens", vocab * h)?;
        let embed = arena.upload(&gpu, &embed_host)?;
        drop(embed_host);
        let lm_head = if ck.has("lm_head") {
            let w = ck.h16("lm_head", vocab * h)?;
            arena.upload(&gpu, &w)?
        } else if c.tie_word_embeddings {
            embed
        } else {
            return Err(HerbertError::ModelLoad("no lm_head and tie_word_embeddings is false".into()));
        };
        let final_norm = arena.upload(&gpu, &ck.h16("norm", h)?)?;
        let mut layers = Vec::with_capacity(c.num_layers);
        let mut states = Vec::new();
        for l in 0..c.num_layers {
            let p = |f: &str| format!("{l}.{f}");
            let input_norm = arena.upload(&gpu, &ck.h16(&p("input_layernorm.weight"), h)?)?;
            let post_norm = arena.upload(&gpu, &ck.h16(&p("post_attention_layernorm.weight"), h)?)?;
            let gate = ck.h16(&p("mlp.gate_proj.weight"), inter * h)?;
            let upw = ck.h16(&p("mlp.up_proj.weight"), inter * h)?;
            let gate_up = arena.upload(&gpu, &to_w(concat(vec![gate, upw])))?;
            let down = arena.upload(&gpu, &to_w(ck.h16(&p("mlp.down_proj.weight"), h * inter)?))?;
            let mixer = if c.is_deltanet_layer(l) {
                let (value, nv) = (delta.value() as usize, delta.v_heads as usize);
                let conv_dim = delta.conv_dim() as usize;
                let qkv = ck.h16(&p("linear_attn.in_proj_qkv.weight"), conv_dim * h)?;
                let z = ck.h16(&p("linear_attn.in_proj_z.weight"), value * h)?;
                let a = ck.h16(&p("linear_attn.in_proj_a.weight"), nv * h)?;
                let b = ck.h16(&p("linear_attn.in_proj_b.weight"), nv * h)?;
                let proj = arena.upload(&gpu, &to_w(concat(vec![qkv, z, a, b, vec![0; delta.pad as usize * h]])))?;
                let conv_w = arena.upload(&gpu, &ck.h16(&p("linear_attn.conv1d.weight"), conv_dim * delta.conv as usize)?)?;
                let a_log = arena.upload(&gpu, &ck.f32(&p("linear_attn.A_log"), nv)?)?;
                let dt_bias = arena.upload(&gpu, &ck.f32(&p("linear_attn.dt_bias"), nv)?)?;
                let norm = arena.upload(&gpu, &ck.h16(&p("linear_attn.norm.weight"), delta.v_dim as usize)?)?;
                let out = arena.upload(&gpu, &to_w(ck.h16(&p("linear_attn.out_proj.weight"), h * value)?))?;
                let cs_bytes = conv_dim * delta.conv as usize * act;
                let rs_bytes = delta.state_len() as usize * act;
                let conv_state = arena.alloc(&gpu, cs_bytes)?;
                let rec_state = arena.alloc(&gpu, rs_bytes)?;
                states.extend([(conv_state, cs_bytes), (rec_state, rs_bytes)]);
                Mixer::Delta { proj, conv_w, a_log, dt_bias, norm, out, conv_state, rec_state }
            } else {
                let (qw, kvw) = (attn.q_width() as usize, attn.kv_width() as usize);
                let q = ck.h16(&p("self_attn.q_proj.weight"), qw * h)?;
                let k = ck.h16(&p("self_attn.k_proj.weight"), kvw * h)?;
                let v = ck.h16(&p("self_attn.v_proj.weight"), kvw * h)?;
                let qkv = arena.upload(&gpu, &to_w(concat(vec![q, k, v])))?;
                let q_norm = arena.upload(&gpu, &ck.h16(&p("self_attn.q_norm.weight"), c.head_dim)?)?;
                let k_norm = arena.upload(&gpu, &ck.h16(&p("self_attn.k_norm.weight"), c.head_dim)?)?;
                let o = arena.upload(&gpu, &to_w(ck.h16(&p("self_attn.o_proj.weight"), h * attn.out_width() as usize)?))?;
                let cache_bytes = kvw * cap * 2;
                let k_cache = arena.alloc(&gpu, cache_bytes)?;
                let v_cache = arena.alloc(&gpu, cache_bytes)?;
                states.extend([(k_cache, cache_bytes), (v_cache, cache_bytes)]);
                Mixer::Attention { qkv, q_norm, k_norm, o, k_cache, v_cache }
            };
            layers.push(Layer { input_norm, post_norm, gate_up, down, mixer });
        }
        drop(ck);

        // RoPE tables of the reserved positions, f32 then rounded to bf16 once.
        let half = c.rotary_ndims / 2;
        let inv_freq: Vec<f32> = (0..half).map(|i| 1.0 / c.rope_theta.powf((2 * i) as f32 / c.rotary_ndims as f32)).collect();
        let mut cos = Vec::with_capacity(cap * half);
        let mut sin = Vec::with_capacity(cap * half);
        for pos in 0..cap {
            for &f in &inv_freq {
                let angle = pos as f32 * f;
                cos.push(half::f16::from_f32(angle.cos()).to_bits());
                sin.push(half::f16::from_f32(angle.sin()).to_bits());
            }
        }
        let cos = arena.upload(&gpu, &cos)?;
        let sin = arena.upload(&gpu, &sin)?;
        let load_seconds = started.elapsed().as_secs_f64();
        let w = Weights { embed, lm_head, final_norm, cos, sin, layers };

        // ── activations ──
        let mounted = Instant::now();
        let splits = tuning.sdpa_split as usize;
        let arg_groups = dc::argmax_groups(vocab as u32) as usize;
        let small = 16;
        let a = Acts {
            state: arena.alloc(&gpu, 16)?,
            x: arena.alloc(&gpu, h * 2)?,
            xn: arena.alloc(&gpu, h * 2)?,
            qkv: arena.alloc(&gpu, attn.qkv_width() as usize * 2)?,
            scores: arena.alloc(&gpu, attn.heads as usize * cap * 4)?,
            attn_out: arena.alloc(&gpu, attn.out_width() as usize * 2)?,
            sdpa_partial: arena.alloc(&gpu, attn.heads as usize * splits * (attn.head_dim as usize + 2) * 4)?,
            gu: arena.alloc(&gpu, 2 * inter * 2)?,
            h: arena.alloc(&gpu, inter * 2)?,
            proj: arena.alloc(&gpu, if has_delta { delta.proj_width() as usize * act } else { small })?,
            gates: arena.alloc(&gpu, if has_delta { 2 * delta.v_heads as usize * act } else { small })?,
            dn_out: arena.alloc(&gpu, if has_delta { delta.value() as usize * act } else { small })?,
            dn_y: arena.alloc(&gpu, if has_delta { delta.value() as usize * 2 } else { small })?,
            logits: arena.alloc(&gpu, vocab * 2)?,
            partial: arena.alloc(&gpu, arg_groups * 8)?,
        };
        states.push((a.state, 16));
        let bm = if blocks { (tuning.prefill_block as usize).min(cap) } else { 0 };
        let blk_acts = if blocks {
            let dn = |n: usize| if has_delta { bm * n } else { small };
            Some(Block {
                rows: bm,
                allocated_rows: bm,
                tokens: arena.alloc(&gpu, bm * 4)?,
                x: arena.alloc(&gpu, bm * h * 2)?,
                xn: arena.alloc(&gpu, bm * h * 2)?,
                qkv: arena.alloc(&gpu, bm * attn.qkv_width() as usize * 2)?,
                attn_out: arena.alloc(&gpu, bm * attn.out_width() as usize * 2)?,
                gu: arena.alloc(&gpu, bm * 2 * inter * 2)?,
                h: arena.alloc(&gpu, bm * inter * 2)?,
                proj: arena.alloc(&gpu, dn(delta.proj_width() as usize * act))?,
                conv_out: arena.alloc(&gpu, dn(delta.conv_dim() as usize * act))?,
                gates: arena.alloc(&gpu, dn(2 * delta.v_heads as usize * act))?,
                dn_out: arena.alloc(&gpu, dn(delta.value() as usize * act))?,
                dn_y: arena.alloc(&gpu, dn(delta.value() as usize * 2))?,
                tc_mat: arena.alloc(&gpu, if has_delta && tuning.rec_tc != 0 { dntc::chunks(bm as u32) as usize * delta.v_heads as usize * dntc::mat_len(delta) as usize * 2 } else { small })?,
                tc_vec: arena.alloc(&gpu, if has_delta && tuning.rec_tc != 0 { dntc::chunks(bm as u32) as usize * delta.v_heads as usize * 4 } else { small })?,
            })
        } else {
            None
        };
        let (free, total) = gpu.mem_info()?;
        eprintln!(
            "[cuda] {:.2} GiB allocated, KV for {cap} positions; {:.1} of {:.1} GiB free",
            arena.bytes as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64
        );

        // ── kernels and the step ──
        let mut ld = Loader { gpu: &gpu, list: Vec::new(), by_name: HashMap::new(), dump: o.dump_ptx.clone(), gemm_carveout: tuning.gemm_carveout, gemm_down_short: tuning.down_short_tiles()?, module_load_seconds: 0.0 };
        let mv = tuning.matvec;
        // The projections' matvecs read f16 weights under `gemm.f16`; the head stays bf16.
        let mvw = kn::Matvec { w16: f16, ..mv };
        let (h32, i32_) = (h as u32, inter as u32);
        let eps = c.rms_norm_eps;
        let plus_one = c.norm_type == crate::core::config::NormType::GemmaRMSNorm;
        let mut ops: Vec<Op> = Vec::new();
        let mut op = |kernel: usize, grid: u32, args: Vec<Arg>| ops.push(Op { kernel, grid: (grid, 1, 1), args });
        let p = Arg::Ptr;
        let k_embed = ld.get(dc::embed(h32, false))?;
        let k_norm = ld.get(dc::rmsnorm(h32, eps, plus_one, false, false))?;
        let k_gate_up = ld.get(kn::matvec(h32, 2 * i32_, Out::Bf16, mvw))?;
        let k_swiglu = ld.get(dc::swiglu(i32_, false, false))?;
        let k_down = ld.get(kn::matvec(i32_, h32, Out::Residual, mvw))?;
        let k_head = ld.get(kn::matvec(h32, vocab as u32, Out::Bf16, mv))?;
        let k_arg1 = ld.get(dc::argmax_partial(vocab as u32))?;
        let k_arg2 = ld.get(dc::argmax_final(arg_groups as u32))?;
        op(k_embed, 1, vec![p(a.state), p(w.embed), p(a.x)]);
        for l in &w.layers {
            op(k_norm, 1, vec![p(a.x), p(l.input_norm), p(a.xn)]);
            match l.mixer {
                Mixer::Attention { qkv, q_norm, k_norm: kn_w, o, k_cache, v_cache } => {
                    let k_qkv = ld.get(kn::matvec(h32, attn.qkv_width(), Out::Bf16, mvw))?;
                    let k_rope = ld.get(dc::qk_rope_append(attn, false))?;
                    let k_split = ld.get(dc::sdpa_split(attn, tuning.sdpa_split))?;
                    let k_comb = ld.get(dc::sdpa_combine(attn, tuning.sdpa_split))?;
                    let k_o = ld.get(kn::matvec(attn.out_width(), h32, Out::Residual, mvw))?;
                    op(k_qkv, mv.groups(attn.qkv_width()), vec![p(qkv), p(a.xn), p(a.qkv)]);
                    op(k_rope, attn.heads + 2 * attn.kv_heads, vec![p(a.state), p(a.qkv), p(q_norm), p(kn_w), p(w.cos), p(w.sin), p(k_cache), p(v_cache)]);
                    op(k_split, attn.heads * tuning.sdpa_split, vec![p(a.state), p(a.qkv), p(k_cache), p(v_cache), p(a.scores), p(a.sdpa_partial)]);
                    op(k_comb, attn.heads, vec![p(a.qkv), p(a.sdpa_partial), p(a.attn_out)]);
                    op(k_o, mv.groups(h32), vec![p(o), p(a.attn_out), p(a.x)]);
                }
                Mixer::Delta { proj, conv_w, a_log, dt_bias, norm, out, conv_state, rec_state } => {
                    let proj_out = match delta.act {
                        super::types::Act::Bf16 => Out::Bf16,
                        super::types::Act::F32 => Out::F32,
                    };
                    let k_proj = ld.get(kn::matvec(h32, delta.proj_width(), proj_out, mvw))?;
                    let k_conv = ld.get(dc::dn_conv_l2_gates(delta))?;
                    let k_rec = ld.get(dc::dn_recurrence(delta))?;
                    let k_gn = ld.get(dc::dn_gated_norm(delta, false, false))?;
                    let k_out = ld.get(kn::matvec(delta.value(), h32, Out::Residual, mvw))?;
                    op(k_proj, mv.groups(delta.proj_width()), vec![p(proj), p(a.xn), p(a.proj)]);
                    op(k_conv, 2 * delta.k_heads + delta.v_heads + 1, vec![p(a.proj), p(conv_state), p(conv_w), p(a_log), p(dt_bias), p(a.gates)]);
                    op(k_rec, delta.v_heads, vec![p(a.proj), p(a.gates), p(rec_state), p(a.dn_out)]);
                    op(k_gn, delta.v_heads, vec![p(a.dn_out), p(a.proj), p(norm), p(a.dn_y)]);
                    op(k_out, mv.groups(h32), vec![p(out), p(a.dn_y), p(a.x)]);
                }
            }
            op(k_norm, 1, vec![p(a.x), p(l.post_norm), p(a.xn)]);
            op(k_gate_up, mv.groups(2 * i32_), vec![p(l.gate_up), p(a.xn), p(a.gu)]);
            op(k_swiglu, i32_.div_ceil(512), vec![p(a.gu), p(a.h)]);
            op(k_down, mv.groups(h32), vec![p(l.down), p(a.h), p(a.x)]);
        }
        op(k_norm, 1, vec![p(a.x), p(w.final_norm), p(a.xn)]);
        op(k_head, mv.groups(vocab as u32), vec![p(w.lm_head), p(a.xn), p(a.logits)]);
        op(k_arg1, arg_groups as u32, vec![p(a.logits), p(a.partial)]);
        op(k_arg2, 1, vec![p(a.partial), p(a.state)]);
        let blk = match blk_acts {
            Some(b) => {
                let has_attn = (0..c.num_layers).any(|l| !c.is_deltanet_layer(l));
                let proj_out = match delta.act {
                    super::types::Act::Bf16 => Out::Bf16,
                    super::types::Act::F32 => Out::F32,
                };
                let bk = BlockKernels {
                    embed: ld.get(dc::embed(h32, true))?,
                    norm: ld.get(dc::rmsnorm(h32, eps, plus_one, true, f16))?,
                    swiglu: ld.get(dc::swiglu(i32_, true, f16))?,
                    gate_up: load_gemm(&mut ld, h32, h32, 2 * i32_, Out::Bf16, tile, tuning.gemm_auto)?,
                    down: load_gemm(&mut ld, h32, i32_, h32, Out::Residual, tile, tuning.gemm_auto)?,
                    attn: if has_attn {
                        Some(AttentionKernels {
                            qkv: load_gemm(&mut ld, h32, h32, attn.qkv_width(), Out::Bf16, tile, tuning.gemm_auto)?,
                            rope: ld.get(dc::qk_rope_append(attn, true))?,
                            flash: ld.get(pf::sdpa_flash(attn, tuning.sdpa_bk, tuning.sdpa_smem_softmax, f16))?,
                            out: load_gemm(&mut ld, h32, attn.out_width(), h32, Out::Residual, tile, tuning.gemm_auto)?,
                        })
                    } else {
                        None
                    },
                    delta: if has_delta {
                        Some(DeltaKernels {
                            proj: load_gemm(&mut ld, h32, h32, delta.proj_width(), proj_out, tile, tuning.gemm_auto)?,
                            conv: ld.get(pf::dn_conv_l2_gates_rows(delta))?,
                            conv_state: ld.get(pf::dn_conv_state(delta))?,
                            recurrence: ld.get(pf::dn_recurrence_rows(delta))?,
                            gated_norm: ld.get(dc::dn_gated_norm(delta, true, f16))?,
                            out: load_gemm(&mut ld, h32, delta.value(), h32, Out::Residual, tile, tuning.gemm_auto)?,
                        })
                    } else {
                        None
                    },
                    tc: if has_delta && tuning.rec_tc != 0 {
                        if delta.act != super::types::Act::Bf16 {
                            return Err(HerbertError::Config("--tune rec.tc=1 needs act=bf16".into()));
                        }
                        Some((ld.get(dntc::dn_tc_prep(delta))?, ld.get(dntc::dn_tc_state(delta))?))
                    } else {
                        None
                    },
                    last_norm: k_norm,
                    head: k_head,
                    arg1: k_arg1,
                    arg2: k_arg2,
                };
                Some((b, bk))
            }
            None => None,
        };
        let module_load_seconds = ld.module_load_seconds;
        let kernels = ld.list;
        eprintln!("[cuda] {} kernels, {} launches per step", kernels.len(), ops.len());

        let mut e = Engine {
            gpu,
            config: c,
            tuning,
            attn,
            delta,
            capacity: cap,
            w,
            a,
            _arena: arena,
            states,
            kernels,
            ops,
            blk,
            graph: None,
            profile: None,
            profile_prefill: None,
            pos: 0,
            argmax: 0,
            need_logits: o.logits,
            logits_cpu: vec![0; vocab],
            load_seconds,
            mount_seconds: 0.0,
            module_load_seconds,
        };
        if o.profile {
            let events = (0..=e.ops.len()).map(|_| e.gpu.event()).collect::<Result<_>>()?;
            e.profile = Some(Profile { events, steps: HashMap::new(), calls: HashMap::new(), totals: Vec::new(), walls: Vec::new() });
            e.profile_prefill = Some(Profile { events: Vec::new(), steps: HashMap::new(), calls: HashMap::new(), totals: Vec::new(), walls: Vec::new() });
        } else if e.tuning.graph != 0 {
            let g = e.gpu.capture(|| e.launch_all())?;
            e.graph = Some(g);
        }
        e.mount_seconds = mounted.elapsed().as_secs_f64();
        Ok(e)
    }

    /// Switch GEMM experiments between requests while reusing model weights.
    /// Decode graphs and weight padding stay fixed. All JIT work is outside
    /// benchmark timers; the caller still resets every measured request.
    pub fn retune_gemm(&mut self, overrides: &[String]) -> Result<()> {
        for setting in overrides {
            let key = setting.split('=').next().unwrap_or("");
            if !matches!(key, "gemm.auto" | "gemm.layout" | "gemm.schedule" | "gemm.pipeline" | "gemm.down_short" | "prefill.block") {
                return Err(HerbertError::Config(format!("retune_gemm cannot change {key}")));
            }
        }
        let parsed = Tuning::with(overrides)?;
        if parsed.gemm_auto == 0 || self.tuning.gemm_auto == 0 {
            return Err(HerbertError::Config("retune_gemm requires automatic GEMM policies and fixed 128-column weight padding".into()));
        }
        let mut tuning = self.tuning.clone();
        tuning.gemm_auto = parsed.gemm_auto;
        tuning.gemm_swizzle = parsed.gemm_swizzle;
        tuning.gemm_schedule = parsed.gemm_schedule;
        tuning.gemm_pipeline = parsed.gemm_pipeline;
        tuning.gemm_down_short = parsed.gemm_down_short;
        tuning.prefill_block = parsed.prefill_block;
        let (block, previous) = self.blk.as_ref().ok_or_else(|| HerbertError::Config("retune_gemm requires block prefill".into()))?;
        if tuning.prefill_block <= 1 || tuning.prefill_block as usize > block.allocated_rows {
            return Err(HerbertError::Config("retune_gemm block size exceeds allocated rows (or is not numeric)".into()));
        }
        let mut next = *previous;
        let tile = tuning.tile()?;
        self.gpu.sync()?;
        let by_name = self.kernels.iter().enumerate().map(|(i, k)| (k.name.clone(), i)).collect();
        let mut ld = Loader { gpu: &self.gpu, list: std::mem::take(&mut self.kernels), by_name,
            dump: None, gemm_carveout: tuning.gemm_carveout,
            gemm_down_short: tuning.down_short_tiles()?, module_load_seconds: 0.0 };
        let h = self.config.hidden_size as u32;
        let inter = self.config.intermediate_size as u32;
        let result = (|| -> Result<()> {
            next.gate_up = load_gemm(&mut ld, h, h, 2 * inter, Out::Bf16, tile, tuning.gemm_auto)?;
            next.down = load_gemm(&mut ld, h, inter, h, Out::Residual, tile, tuning.gemm_auto)?;
            if let Some(attn) = next.attn.as_mut() {
                attn.qkv = load_gemm(&mut ld, h, h, self.attn.qkv_width(), Out::Bf16, tile, tuning.gemm_auto)?;
                attn.out = load_gemm(&mut ld, h, self.attn.out_width(), h, Out::Residual, tile, tuning.gemm_auto)?;
            }
            if let Some(delta) = next.delta.as_mut() {
                let out = match self.delta.act { super::types::Act::Bf16 => Out::Bf16, super::types::Act::F32 => Out::F32 };
                delta.proj = load_gemm(&mut ld, h, h, self.delta.proj_width(), out, tile, tuning.gemm_auto)?;
                delta.out = load_gemm(&mut ld, h, self.delta.value(), h, Out::Residual, tile, tuning.gemm_auto)?;
            }
            Ok(())
        })();
        self.module_load_seconds += ld.module_load_seconds;
        self.kernels = ld.list; // Retain original module indices even on a JIT error.
        result?;
        let (block, kernels) = self.blk.as_mut().unwrap();
        block.rows = tuning.prefill_block as usize;
        *kernels = next;
        self.tuning = tuning;
        Ok(())
    }

    fn launch_all(&self) -> Result<()> {
        for op in &self.ops {
            self.gpu.launch(&self.kernels[op.kernel], op.grid, &op.args)?;
        }
        Ok(())
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Back to an empty conversation: KV caches and DeltaNet states zeroed.
    pub fn reset(&mut self) -> Result<()> {
        self.gpu.sync()?;
        for &(p, bytes) in &self.states {
            self.gpu.zero_raw(p, bytes)?;
        }
        self.pos = 0;
        Ok(())
    }

    /// One token through the whole model at the current position.
    fn step(&mut self, token: u32) -> Result<()> {
        if self.pos >= self.capacity {
            return Err(HerbertError::Backend(format!("the conversation reached the {} reserved positions", self.capacity)));
        }
        let started = Instant::now();
        let io = [token, self.pos as u32];
        self.gpu.write_raw(self.a.state, crate::core::bf16::as_bytes(&io))?;
        match (&self.graph, &self.profile) {
            (Some(g), _) => self.gpu.replay(g)?,
            (None, Some(p)) => {
                for (i, op) in self.ops.iter().enumerate() {
                    self.gpu.record(&p.events[i])?;
                    self.gpu.launch(&self.kernels[op.kernel], op.grid, &op.args)?;
                }
                self.gpu.record(&p.events[self.ops.len()])?;
            }
            (None, None) => self.launch_all()?,
        }
        let mut state = [0u32; 4];
        self.gpu.read_raw(self.a.state, &mut state)?;
        self.argmax = state[STATE_ARGMAX as usize];
        if self.need_logits {
            let mut l = std::mem::take(&mut self.logits_cpu);
            self.gpu.read_raw(self.a.logits, &mut l)?;
            self.logits_cpu = l;
        }
        self.pos += 1;
        let wall = started.elapsed().as_secs_f64();
        if self.profile.is_some() {
            self.collect(wall)?;
        }
        Ok(())
    }

    fn collect(&mut self, wall: f64) -> Result<()> {
        let names: Vec<String> = self.ops.iter().map(|op| self.kernels[op.kernel].name.clone()).collect();
        Self::collect_into(&self.gpu, self.profile.as_mut().expect("profile"), &names, wall)
    }

    fn collect_into(gpu: &Gpu, p: &mut Profile, names: &[String], wall: f64) -> Result<()> {
        let mut per: HashMap<String, f64> = HashMap::new();
        let mut calls: HashMap<String, usize> = HashMap::new();
        let mut total = 0.0;
        for (i, name) in names.iter().enumerate() {
            let ms = gpu.elapsed(&p.events[i], &p.events[i + 1])?;
            *per.entry(name.clone()).or_default() += ms / 1e3;
            *calls.entry(name.clone()).or_default() += 1;
            total += ms / 1e3;
        }
        for (name, t) in per {
            p.steps.entry(name).or_default().push(t);
        }
        p.calls = calls;
        p.totals.push(total);
        p.walls.push(wall);
        Ok(())
    }

    /// Forget the profiled steps so far (after a warm-up).
    pub fn clear_profile(&mut self) {
        for p in [&mut self.profile, &mut self.profile_prefill].into_iter().flatten() {
            p.steps.clear();
            p.totals.clear();
            p.walls.clear();
        }
    }

    /// Per kernel: launches per step, median GPU time per step, share of the
    /// median step; the host's wall time beside. `None` when not profiling.
    pub fn profile_report(&self) -> Option<String> {
        let mut out = String::new();
        if let Some(p) = &self.profile_prefill {
            out.push_str(&Self::report_of(p, "prefill blocks", "block"));
        }
        if let Some(p) = &self.profile {
            out.push_str(&Self::report_of(p, "decode steps", "step"));
        }
        (!out.is_empty()).then_some(out)
    }

    fn report_of(p: &Profile, what: &str, unit: &str) -> String {
        if p.totals.is_empty() {
            return String::new();
        }
        let median = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(f64::total_cmp);
            s[s.len() / 2]
        };
        let total = median(&p.totals);
        let mut rows: Vec<(String, f64)> = p.steps.iter().map(|(n, v)| (n.clone(), median(v))).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut out = format!("[profile] {} {what}: GPU {:.1} µs per {unit} (median), host wall {:.1} µs\n", p.totals.len(), total * 1e6, median(&p.walls) * 1e6);
        for (name, t) in rows {
            out.push_str(&format!("[profile]   {:>8.1} µs {:>5.1} %  ×{:<3} {name}\n", t * 1e6, 100.0 * t / total, p.calls.get(&name).copied().unwrap_or(0)));
        }
        out
    }

    /// Append the tokens; the greedy token of the last one. By blocks of
    /// `prefill.block` rows, or one decode step each with `prefill.block=1`.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32> {
        if self.pos + ids.len() > self.capacity {
            return Err(HerbertError::Backend(format!("the conversation reached the {} reserved positions", self.capacity)));
        }
        let Some(rows) = self.blk.as_ref().map(|b| b.0.rows) else {
            for &t in ids {
                self.step(t)?;
            }
            return Ok(self.argmax);
        };
        let n = ids.len().div_ceil(rows);
        for (i, block) in ids.chunks(rows).enumerate() {
            let ops = self.block_ops(block.len() as u32, i + 1 == n);
            let b = &self.blk.as_ref().expect("block").0;
            self.gpu.write_raw(b.tokens, crate::core::bf16::as_bytes(block))?;
            let io = [block[block.len() - 1], self.pos as u32];
            self.gpu.write_raw(self.a.state, crate::core::bf16::as_bytes(&io))?;
            let started = Instant::now();
            if let Some(p) = self.profile_prefill.as_mut() {
                while p.events.len() <= ops.len() {
                    p.events.push(self.gpu.event()?);
                }
                for (i, op) in ops.iter().enumerate() {
                    self.gpu.record(&p.events[i])?;
                    self.gpu.launch(&self.kernels[op.kernel], op.grid, &op.args)?;
                }
                self.gpu.record(&p.events[ops.len()])?;
            } else {
                for op in &ops {
                    self.gpu.launch(&self.kernels[op.kernel], op.grid, &op.args)?;
                }
            }
            self.gpu.sync()?;
            self.pos += block.len();
            if self.profile_prefill.is_some() {
                let names: Vec<String> = ops.iter().map(|op| self.kernels[op.kernel].name.clone()).collect();
                let wall = started.elapsed().as_secs_f64();
                Self::collect_into(&self.gpu, self.profile_prefill.as_mut().expect("profile"), &names, wall)?;
            }
        }
        let mut state = [0u32; 4];
        self.gpu.read_raw(self.a.state, &mut state)?;
        self.argmax = state[STATE_ARGMAX as usize];
        if self.need_logits {
            let mut l = std::mem::take(&mut self.logits_cpu);
            self.gpu.read_raw(self.a.logits, &mut l)?;
            self.logits_cpu = l;
        }
        Ok(self.argmax)
    }

    /// The prefill of `prefill`, one launch at a time, for the f16 question: after each launch every
    /// activation buffer among its arguments (the block's, the DeltaNet states, the chunk operators) is
    /// read back and its range recorded per layer and buffer in `rec` — the largest magnitude, and the
    /// values an IEEE f16 could not hold (above 65 504) or would hold below its normal range. The
    /// buffers hold the engine's 16-bit element type (IEEE f16 in herbert-cuda-f16); nothing else changes.
    pub fn prefill_absmax(&mut self, ids: &[u32], rec: &mut AbsMax) -> Result<u32> {
        if self.delta.act != crate::cuda::types::Act::Bf16 {
            return Err(HerbertError::Config("prefill_absmax reads 16-bit activations: act=bf16 (f16 here)".into()));
        }
        let Some(rows) = self.blk.as_ref().map(|b| b.0.rows) else {
            return Err(HerbertError::Config("prefill_absmax needs block prefill".into()));
        };
        if self.pos + ids.len() > self.capacity {
            return Err(HerbertError::Backend(format!("the conversation reached the {} reserved positions", self.capacity)));
        }
        let n = ids.len().div_ceil(rows);
        for (i, block) in ids.chunks(rows).enumerate() {
            let m = block.len() as u32;
            let ops = self.block_ops(m, i + 1 == n);
            let named = self.absmax_buffers(m);
            let b = &self.blk.as_ref().expect("block").0;
            self.gpu.write_raw(b.tokens, crate::core::bf16::as_bytes(block))?;
            let io = [block[block.len() - 1], self.pos as u32];
            self.gpu.write_raw(self.a.state, crate::core::bf16::as_bytes(&io))?;
            let mut layer: Option<usize> = None;
            for op in &ops {
                if let Some(Arg::Ptr(first)) = op.args.get(1) {
                    if let Some(l) = self.w.layers.iter().position(|l| l.input_norm == *first) { layer = Some(l); }
                }
                self.gpu.launch(&self.kernels[op.kernel], op.grid, &op.args)?;
                self.gpu.sync()?;
                for arg in &op.args {
                    let Arg::Ptr(ptr) = arg else { continue };
                    let Some((name, len, per_layer)) = named.iter().find(|(p, ..)| *p == *ptr).map(|(_, n, l, pl)| (*n, *l, *pl)) else { continue };
                    let mut v = vec![0u16; len];
                    self.gpu.read_raw(*ptr, &mut v)?;
                    let at = if per_layer || layer.is_some() { layer } else { None };
                    rec.add(at, name, &self.kernels[op.kernel].name, &v);
                }
            }
            self.pos += block.len();
        }
        let mut state = [0u32; 4];
        self.gpu.read_raw(self.a.state, &mut state)?;
        self.argmax = state[STATE_ARGMAX as usize];
        Ok(self.argmax)
    }

    /// (pointer, name, 16-bit elements, per layer) of the buffers `prefill_absmax` reads, for `m` rows.
    fn absmax_buffers(&self, m: u32) -> Vec<(Ptr, &'static str, usize, bool)> {
        let (b, k) = self.blk.as_ref().expect("block");
        let (attn, delta) = (self.attn, self.delta);
        let (h, inter, m) = (self.config.hidden_size, self.config.intermediate_size, m as usize);
        let mut v = vec![
            (b.x, "x", m * h, false), (b.xn, "xn", m * h, false), (b.qkv, "qkv", m * attn.qkv_width() as usize, false),
            (b.attn_out, "attn_out", m * attn.out_width() as usize, false), (b.gu, "gate_up", m * 2 * inter, false),
            (b.h, "swiglu", m * inter, false),
        ];
        if self.config.layer_types.contains(&crate::core::config::LayerType::LinearAttention) {
            v.extend([
                (b.proj, "dn_proj", m * delta.proj_width() as usize, false), (b.conv_out, "dn_conv", m * delta.conv_dim() as usize, false),
                (b.gates, "dn_gates", m * 2 * delta.v_heads as usize, false), (b.dn_out, "dn_out", m * delta.value() as usize, false),
                (b.dn_y, "dn_y", m * delta.value() as usize, false),
            ]);
            if k.tc.is_some() {
                v.push((b.tc_mat, "dn_tc_mat", (dntc::chunks(m as u32) * delta.v_heads * dntc::mat_len(delta)) as usize, false));
            }
            for l in &self.w.layers {
                if let Mixer::Delta { rec_state, conv_state, .. } = l.mixer {
                    v.push((rec_state, "dn_state", delta.state_len() as usize, true));
                    v.push((conv_state, "dn_conv_state", (delta.conv_dim() * delta.conv) as usize, true));
                }
            }
        }
        v
    }

    /// The launches of a prefill block of `m` rows; the head on the last row
    /// of the `last` block only.
    fn block_ops(&self, m: u32, last: bool) -> Vec<Op> {
        let (b, k) = self.blk.as_ref().expect("block kernels");
        let (w, a, attn, delta) = (&self.w, &self.a, self.attn, self.delta);
        let (h32, i32_) = (self.config.hidden_size as u32, self.config.intermediate_size as u32);
        let vocab = self.config.vocab_size as u32;
        let mv = self.tuning.matvec;
        let p = Arg::Ptr;
        let mu = Arg::U32(m);
        let mut ops = Vec::new();
        let mut op = |kernel: usize, grid: (u32, u32, u32), args: Vec<Arg>| ops.push(Op { kernel, grid, args });
        op(k.embed, (1, m, 1), vec![p(b.tokens), p(w.embed), p(b.x)]);
        for l in &w.layers {
            op(k.norm, (1, m, 1), vec![p(b.x), p(l.input_norm), p(b.xn)]);
            match l.mixer {
                Mixer::Attention { qkv, q_norm, k_norm, o, k_cache, v_cache } => {
                    let kernels = k.attn.as_ref().expect("attention kernels");
                    let g = kernels.qkv.select(m);
                    op(g.kernel, g.tile.grid(m, attn.qkv_width()), vec![p(b.xn), p(qkv), p(b.qkv), mu]);
                    op(kernels.rope, (attn.heads + 2 * attn.kv_heads, m, 1), vec![p(a.state), p(b.qkv), p(q_norm), p(k_norm), p(w.cos), p(w.sin), p(k_cache), p(v_cache)]);
                    op(kernels.flash, pf::flash_grid(attn, m), vec![p(a.state), p(b.qkv), p(k_cache), p(v_cache), p(b.attn_out), mu]);
                    let g = kernels.out.select(m);
                    op(g.kernel, g.tile.grid(m, h32), vec![p(b.attn_out), p(o), p(b.x), mu]);
                }
                Mixer::Delta { proj, conv_w, a_log, dt_bias, norm, out, conv_state, rec_state } => {
                    let kernels = k.delta.as_ref().expect("DeltaNet kernels");
                    let g = kernels.proj.select(m);
                    op(g.kernel, g.tile.grid(m, delta.proj_width()), vec![p(b.xn), p(proj), p(b.proj), mu]);
                    op(kernels.conv, (2 * delta.k_heads + delta.v_heads + 1, m, 1), vec![p(b.proj), p(conv_state), p(conv_w), p(a_log), p(dt_bias), p(b.conv_out), p(b.gates)]);
                    op(kernels.conv_state, (delta.conv_dim() / delta.k_dim, 1, 1), vec![p(b.proj), p(conv_state), mu]);
                    match k.tc {
                        Some((k_prep, k_state)) => {
                            op(k_prep, (dntc::chunks(m) * delta.v_heads, 1, 1), vec![p(b.conv_out), p(b.gates), p(b.tc_mat), p(b.tc_vec), mu]);
                            op(k_state, (delta.v_heads * delta.v_dim / dntc::DVS, 1, 1), vec![p(b.conv_out), p(b.tc_mat), p(b.tc_vec), p(rec_state), p(b.dn_out), mu]);
                        }
                        None => op(kernels.recurrence, (delta.v_heads, 1, 1), vec![p(b.conv_out), p(b.gates), p(rec_state), p(b.dn_out), mu]),
                    }
                    op(kernels.gated_norm, (delta.v_heads, m, 1), vec![p(b.dn_out), p(b.proj), p(norm), p(b.dn_y)]);
                    let g = kernels.out.select(m);
                    op(g.kernel, g.tile.grid(m, h32), vec![p(b.dn_y), p(out), p(b.x), mu]);
                }
            }
            op(k.norm, (1, m, 1), vec![p(b.x), p(l.post_norm), p(b.xn)]);
            let g = k.gate_up.select(m);
            op(g.kernel, g.tile.grid(m, 2 * i32_), vec![p(b.xn), p(l.gate_up), p(b.gu), mu]);
            op(k.swiglu, (i32_.div_ceil(512), m, 1), vec![p(b.gu), p(b.h)]);
            let g = k.down.select(m);
            op(g.kernel, g.tile.grid(m, h32), vec![p(b.h), p(l.down), p(b.x), mu]);
        }
        if last {
            let x_last = b.x + (m as u64 - 1) * h32 as u64 * 2;
            op(k.last_norm, (1, 1, 1), vec![p(x_last), p(w.final_norm), p(a.xn)]);
            op(k.head, (mv.groups(vocab), 1, 1), vec![p(w.lm_head), p(a.xn), p(a.logits)]);
            op(k.arg1, (dc::argmax_groups(vocab), 1, 1), vec![p(a.logits), p(a.partial)]);
            op(k.arg2, (1, 1, 1), vec![p(a.partial), p(a.state)]);
        }
        ops
    }

    /// Decode from `first`: after each token the callback receives the argmax
    /// and the logits and returns the next token, or `None` to stop. At
    /// `max_steps` the callback's last token is not fed. Returns the tokens fed.
    pub fn generate(&mut self, first: u32, max_steps: usize, f: &mut dyn FnMut(u32, &[u16]) -> Result<Option<u32>>) -> Result<usize> {
        let mut token = first;
        let mut fed = 0;
        while fed < max_steps {
            self.step(token)?;
            fed += 1;
            match f(self.argmax, &self.logits_cpu)? {
                Some(next) => token = next,
                None => break,
            }
        }
        Ok(fed)
    }

    /// The logits of the last step — zeros when the engine was loaded without
    /// [`Options::logits`] (greedy decoding reads only the argmax).
    pub fn logits(&self) -> &[u16] {
        &self.logits_cpu
    }
}
