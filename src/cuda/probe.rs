//! The probe: what the device declares, then measurements with emitted
//! kernels, timed with events. Values, not verdicts:
//! - `selftest`: the emitter end to end (loads, bf16 widening and rounding,
//!   FMA, warp shuffles, predicated stores), checked element by element;
//! - read bandwidth: a grid-stride loop over 16-byte loads, per cache path;
//! - `mma`: tensor core throughput, bf16 → f32 against f16 → f32 and f16 → f16,
//!   operands in registers;
//! - the bf16 matvec of the decode at the shapes of Qwen3.5-0.8B and 4B,
//!   against an f64 reference, in effective GB/s of weights read (the weights
//!   rotate over copies larger than the L2, as a decode step reads them);
//! - the prefill GEMM on tensor cores at the same shapes, against f64 on
//!   sampled entries, in TFLOPS.
//!
//! Errors are counted in bf16 ulps of the f64 reference: one rounding of an
//! f32 sum close to the reference gives at most 1, unless the sum cancels
//! (|Σ a·w| ≪ Σ |a·w|), where the f32 accumulation error alone exceeds an ulp
//! of the result. So each error is also given against the magnitude of the
//! terms: `|err| / (Σ |a·w| · 2⁻⁸)`, ≤ 0.5 for a correctly rounded result and
//! independent of the cancellation.

use super::gemm::{self, Tile};
use super::gpu::{Arg, Buf, Gpu};
use super::kernels::{self as kn, Matvec, Out};
use crate::core::bf16::{bf16_to_f32, f32_to_bf16, Bf16};
use crate::core::error::{HerbertError, Result};
use crate::ptx::{Emitted, Ld};

/// The projections of a layer and the head: (name, K, N).
pub const SHAPES_0_8B: &[(&str, u32, u32)] = &[
    ("dn.proj", 1024, 8256),
    ("dn.proj128", 1024, 8320),
    ("dn.proj256", 1024, 8448),
    ("dn.out", 2048, 1024),
    ("attn.qkv", 1024, 5120),
    ("attn.o", 2048, 1024),
    ("mlp.gate_up", 1024, 7168),
    ("mlp.down", 3584, 1024),
    ("head", 1024, 248320),
];
pub const SHAPES_4B: &[(&str, u32, u32)] = &[
    ("dn.proj", 2560, 12352),
    ("dn.proj128", 2560, 12416),
    ("dn.proj256", 2560, 12544),
    ("dn.out", 4096, 2560),
    ("attn.qkv", 2560, 10240),
    ("attn.o", 4096, 2560),
    ("mlp.gate_up", 2560, 18432),
    ("mlp.down", 9216, 2560),
    ("head", 2560, 248320),
];

/// Every kernel the probe runs and the decode kernels of 0.8B and 4B, for
/// `ptxas` offline.
pub fn all_kernels() -> Vec<Emitted> {
    use super::decode as dc;
    use super::types::{Act, AttnShape, DeltaShape};
    let mut v = vec![kn::selftest()];
    for (h, inter, attn, delta) in [
        (1024, 3584, AttnShape::QWEN35_0_8B, DeltaShape::QWEN35_0_8B),
        (2560, 9216, AttnShape { heads: 16, kv_heads: 4, ..AttnShape::QWEN35_0_8B }, DeltaShape { v_heads: 32, ..DeltaShape::QWEN35_0_8B }),
    ] {
        v.push(dc::embed(h, false));
        v.push(dc::rmsnorm(h, 1e-6, true, false, false));
        v.push(dc::swiglu(inter, false, false));
        v.push(dc::qk_rope_append(attn, false));
        v.push(dc::qk_rope_append(attn, true));
        v.push(dc::embed(h, true));
        v.push(dc::rmsnorm(h, 1e-6, true, true, false));
        v.push(dc::swiglu(inter, true, false));
        v.push(super::prefill::sdpa_flash(attn, 32, false, false));
        v.push(super::prefill::sdpa_flash(attn, 32, true, false));
        v.push(gemm::gemm(inter, h, Out::Residual, Tile::T64));
        v.push(dc::sdpa_split(attn, 8));
        v.push(dc::sdpa_combine(attn, 8));
        for act in [Act::Bf16, Act::F32] {
            let d = DeltaShape { act, ..delta };
            v.push(dc::dn_conv_l2_gates(d));
            v.push(dc::dn_recurrence(d));
            v.push(dc::dn_gated_norm(d, false, false));
            v.push(dc::dn_gated_norm(d, true, false));
            v.push(super::prefill::dn_conv_l2_gates_rows(d));
            v.push(super::prefill::dn_conv_state(d));
            v.push(super::prefill::dn_recurrence_rows(d));
            if act == Act::Bf16 {
                v.push(super::dntc::dn_tc_prep(d));
                v.push(super::dntc::dn_tc_state(d));
            }
        }
        v.push(kn::matvec(inter, h, Out::Residual, Matvec::DEFAULT));
    }
    for kind in kn::Mma::ALL {
        for chains in [4, 8] {
            v.push(kn::mma_peak(kind, chains, 4));
        }
    }
    v.push(dc::argmax_partial(248320));
    v.push(dc::argmax_final(dc::argmax_groups(248320)));
    for c in [Ld::Ca, Ld::Nc, Ld::NcNoL1, Ld::Cg, Ld::Cs] {
        v.push(kn::read(c, 4));
    }
    for &(_, kd, n) in SHAPES_0_8B.iter().chain(SHAPES_4B) {
        for rows in [1, 2, 4] {
            v.push(kn::matvec(kd, n, Out::Bf16, Matvec { rows, ..Matvec::DEFAULT }));
        }
        if n % 64 == 0 {
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T32X64));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T32X64_K64_S2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64_K32_S2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64_K64_S2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64_K64_S3));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64_W4X1));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64_W1X4));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64_W1X4_K64_S2));
        }
        if n % 128 == 0 {
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64X128));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64X128_S2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64X128_K64_S2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64X128_K64_S3));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64X128_W4X2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64X128_W2X2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64X128_W2X2_S2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T128));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T128_K64_S2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T128_W4X4_K64_S2));
            v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T128_W4X2_K64_S2));
        }
        if n % 256 == 0 { v.push(gemm::gemm(kd, n, Out::Bf16, Tile::T64X256_S2)); }
    }
    v
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-scale, scale), rounded to bf16.
    fn bf16(&mut self, scale: f32) -> Bf16 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        f32_to_bf16((2.0 * u - 1.0) * scale)
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<Bf16> {
        (0..n).map(|_| self.bf16(scale)).collect()
    }
}

/// The distance from `got` to `want` in bf16 ulps of `want` (its exponent,
/// floored at 2⁻¹⁰⁰ for values near zero).
fn ulps(got: f32, want: f64) -> f64 {
    let e = want.abs().max(1e-30).log2().floor() as i32;
    (got as f64 - want).abs() / 2f64.powi(e - 7)
}

/// Tally of errors in ulps.
#[derive(Default)]
struct Errors {
    n: usize,
    max: f64,
    over1: usize,
    /// Max of `|err| / (Σ |a·w| · 2⁻⁸)`.
    mag: f64,
    nonfinite: usize,
}
impl Errors {
    fn add(&mut self, got: f32, want: f64, terms: f64) {
        self.n += 1;
        if !got.is_finite() {
            self.nonfinite += 1;
            return;
        }
        self.mag = self.mag.max((got as f64 - want).abs() / (terms.max(1e-30) / 256.0));
        let u = ulps(got, want);
        self.max = self.max.max(u);
        if u > 1.0 + 1e-9 {
            self.over1 += 1;
        }
    }
    fn line(&self) -> String {
        format!("max {:.3} ulp, {} > 1 ulp, {:.4} of Σ|aw|·2⁻⁸, {} non-finite / {}", self.max, self.over1, self.mag, self.nonfinite, self.n)
    }
}

/// Milliseconds per launch: `warm` launches, then `reps` between two events.
fn timed(gpu: &Gpu, warm: u32, reps: u32, mut launch: impl FnMut(u32) -> Result<()>) -> Result<f64> {
    for i in 0..warm {
        launch(i)?;
    }
    let (a, b) = (gpu.event()?, gpu.event()?);
    gpu.record(&a)?;
    for i in 0..reps {
        launch(i)?;
    }
    gpu.record(&b)?;
    Ok(gpu.elapsed(&a, &b)? / reps as f64)
}

pub fn report(gpu: &Gpu) -> Result<()> {
    use super::driver::attr;
    let (free, total) = gpu.mem_info()?;
    println!("== {} — sm_{}{}, {} SMs, driver {}.{}", gpu.name, gpu.cc.0, gpu.cc.1, gpu.sms, gpu.driver_version / 1000, (gpu.driver_version % 1000) / 10);
    println!(
        "   clock {} MHz, memory {} MHz × {} bits, L2 {} MiB, {:.1}/{:.1} GiB free",
        gpu.attr(attr::CLOCK_RATE)? / 1000,
        gpu.attr(attr::MEMORY_CLOCK_RATE)? / 1000,
        gpu.attr(attr::GLOBAL_MEMORY_BUS_WIDTH)?,
        gpu.l2 >> 20,
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    println!(
        "   shared {} KiB per block ({} opt-in), {} KiB per SM; {} registers and {} threads per SM",
        gpu.attr(attr::MAX_SHARED_PER_BLOCK)? >> 10,
        gpu.smem_optin >> 10,
        gpu.attr(attr::MAX_SHARED_PER_SM)? >> 10,
        gpu.attr(attr::MAX_REGISTERS_PER_SM)?,
        gpu.attr(attr::MAX_THREADS_PER_SM)?
    );
    println!("   driver reserves {} bytes of shared memory per block", gpu.attr(attr::RESERVED_SHARED_PER_BLOCK)?);
    Ok(())
}

fn ptxas_line(k: &super::gpu::Kernel) -> String {
    format!("{} regs, {} B spill, {} B static shared, {} B dynamic", k.regs, k.local, k.static_shared, k.dyn_shared)
}

pub fn selftest(gpu: &Gpu) -> Result<()> {
    let n = 1000u32;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let x: Vec<Bf16> = rng.vec(2 * n as usize, 4.0);
    let e = kn::selftest();
    let k = gpu.load(&e)?;
    let xb = gpu.upload(&x)?;
    let yb = gpu.alloc(4 * n as usize)?;
    let warps = n.div_ceil(256) * 8;
    let sb = gpu.alloc(4 * warps as usize)?;
    gpu.launch(&k, (n.div_ceil(256), 1, 1), &[(&xb).into(), (&yb).into(), (&sb).into(), Arg::U32(n)])?;
    let mut y = vec![0u16; 2 * n as usize];
    gpu.read(&yb, 0, &mut y)?;
    let mut s = vec![0f32; warps as usize];
    gpu.read(&sb, 0, &mut s)?;
    let mut bad = 0;
    for i in 0..2 * n as usize {
        let want = f32_to_bf16(bf16_to_f32(x[i]).mul_add(2.0, 1.0));
        if y[i] != want {
            bad += 1;
        }
    }
    let mut serr = 0f64;
    for (w, &got) in s.iter().enumerate() {
        let want: f64 = (w * 32..(w * 32 + 32).min(n as usize)).map(|i| bf16_to_f32(x[2 * i]) as f64 + bf16_to_f32(x[2 * i + 1]) as f64).sum();
        serr = serr.max((got as f64 - want).abs());
    }
    println!("selftest: {bad} wrong bf16 of {}, warp sums max |err| {serr:.2e} ({})", 2 * n, ptxas_line(&k));
    if bad > 0 || serr > 1e-3 {
        return Err(HerbertError::Backend("selftest failed".into()));
    }
    Ok(())
}

pub fn bandwidth(gpu: &Gpu) -> Result<()> {
    let bytes = 1usize << 30;
    let src = gpu.alloc(bytes)?;
    gpu.zero(&src)?;
    let blocks = gpu.sms * 8;
    let sink = gpu.alloc(4 * (blocks * 256) as usize)?;
    for c in [Ld::Ca, Ld::Nc, Ld::NcNoL1, Ld::Cg, Ld::Cs] {
        for unroll in [1, 4, 8] {
            let k = gpu.load(&kn::read(c, unroll))?;
            let ms = timed(gpu, 2, 10, |_| gpu.launch(&k, (blocks, 1, 1), &[(&src).into(), (&sink).into(), Arg::U32((bytes / 16) as u32)]))?;
            println!("read {:<28} {:7.1} GB/s ({})", k.name, bytes as f64 / ms / 1e6, ptxas_line(&k));
        }
    }
    Ok(())
}

/// Tensor core throughput per operand/accumulator type (`kn::mma_peak`):
/// TFLOPS counted as 2·16·8·16 per `mma`, for 1 to 8 blocks of 4 warps per SM.
pub fn mma(gpu: &Gpu) -> Result<()> {
    let unroll = 4;
    let max_blocks = gpu.sms * 8;
    let out = gpu.alloc(4 * (max_blocks * 128) as usize)?;
    for kind in kn::Mma::ALL {
        for chains in [4, 8] {
            let k = gpu.load(&kn::mma_peak(kind, chains, unroll))?;
            for per_sm in [1, 2, 4, 8] {
                let blocks = gpu.sms * per_sm;
                let per_pass = blocks as u64 * 4 * (chains * unroll) as u64;
                let iters = (250_000_000u64 / per_pass).max(1) as u32;
                let ms = timed(gpu, 2, 5, |_| gpu.launch(&k, (blocks, 1, 1), &[(&out).into(), Arg::U32(iters)]))?;
                let tflops = (per_pass * iters as u64) as f64 * 4096.0 / (ms * 1e-3) / 1e12;
                println!("mma {:<8} chains {chains} unroll {unroll} warps/SM {:2}: {tflops:7.2} TFLOPS ({ms:.3} ms, {})", kind.tag(), per_sm * 4, ptxas_line(&k));
            }
        }
    }
    Ok(())
}

/// The decode matvec at every shape, `Matvec` variants side by side.
pub fn matvec(gpu: &Gpu, variants: &[Matvec]) -> Result<()> {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for (model, shapes) in [("0.8B", SHAPES_0_8B), ("4B", SHAPES_4B)] {
        for &(what, kd, n) in shapes {
            let w = rng.vec(kd as usize * n as usize, 0.05);
            let x = rng.vec(kd as usize, 1.0);
            let dot = |r: usize, abs: bool| -> f64 {
                (0..kd as usize)
                    .map(|c| {
                        let t = bf16_to_f32(w[r * kd as usize + c]) as f64 * bf16_to_f32(x[c]) as f64;
                        if abs {
                            t.abs()
                        } else {
                            t
                        }
                    })
                    .sum()
            };
            let want: Vec<(f64, f64)> = (0..n as usize).map(|r| (dot(r, false), dot(r, true))).collect();
            let wbytes = w.len() * 2;
            // Copies of W rotated through so that every launch reads DRAM.
            let copies = (256usize << 20).div_ceil(wbytes).max(2);
            let wb: Vec<Buf> = (0..copies).map(|_| gpu.upload(&w)).collect::<Result<_>>()?;
            let xb = gpu.upload(&x)?;
            let yb = gpu.alloc(n as usize * 2)?;
            for &t in variants {
                let k = gpu.load(&kn::matvec(kd, n, Out::Bf16, t))?;
                gpu.zero(&yb)?;
                gpu.launch(&k, (t.groups(n), 1, 1), &[(&wb[0]).into(), (&xb).into(), (&yb).into()])?;
                let mut y = vec![0u16; n as usize];
                gpu.read(&yb, 0, &mut y)?;
                let mut err = Errors::default();
                for (r, &h) in y.iter().enumerate() {
                    err.add(bf16_to_f32(h), want[r].0, want[r].1);
                }
                let ms = timed(gpu, 3, 50, |i| gpu.launch(&k, (t.groups(n), 1, 1), &[(&wb[i as usize % copies]).into(), (&xb).into(), (&yb).into()]))?;
                println!(
                    "matvec {model:>4} {what:<12} {kd:>5}×{n:<6} r{} w{} d{} {:?}: {:8.2} µs {:7.1} GB/s — {} — {}",
                    t.rows,
                    t.warps,
                    t.depth,
                    t.cache,
                    ms * 1e3,
                    wbytes as f64 / ms / 1e6,
                    err.line(),
                    ptxas_line(&k)
                );
            }
        }
    }
    Ok(())
}

/// The prefill GEMM at every shape and `m` rows, tiles side by side.
pub struct GemmBench<'a> {
    pub model: Option<&'a str>, pub shape: Option<&'a str>, pub rounds: u32,
    pub rotate: bool, pub out: Out,
}

/// Offline emission uses the same dimensions, epilogues and variant tags as the GPU probe.
pub fn gemm_kernels(tiles: &[Tile], model: Option<&str>, shape: Option<&str>, out: Out) -> Vec<crate::ptx::Emitted> {
    let mut kernels = Vec::new();
    for (name, shapes) in [("0.8B", SHAPES_0_8B), ("4B", SHAPES_4B)] {
        if model.is_some_and(|v| v != name) { continue; }
        for &(what, kd, n) in shapes {
            if shape.is_some_and(|v| v != what) { continue; }
            for &t in tiles {
                if n % t.bn == 0 && kd % t.bk == 0 { kernels.push(gemm::gemm(kd, n, out, t)); }
            }
        }
    }
    kernels
}

pub fn gemm(gpu: &Gpu, ms_rows: &[u32], tiles: &[Tile], options: GemmBench<'_>) -> Result<()> {
    if ms_rows.is_empty() || ms_rows.contains(&0) || options.rounds == 0 || options.out == Out::F32 {
        return Err(HerbertError::Config("GEMM probe: positive M/rounds, bf16 or residual output required".into()));
    }
    if gemm_kernels(tiles, options.model, options.shape, options.out).is_empty() {
        return Err(HerbertError::Config("GEMM probe: filters select no compatible kernels".into()));
    }
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for (model, shapes) in [("0.8B", SHAPES_0_8B), ("4B", SHAPES_4B)] {
        if options.model.is_some_and(|v| v != model) { continue; }
        for &(what, kd, n) in shapes {
            if options.shape.is_some_and(|v| v != what) { continue; }
            let mmax = *ms_rows.iter().max().unwrap();
            let w = rng.vec(kd as usize * n as usize, 0.05);
            let a = rng.vec(kd as usize * mmax as usize, 1.0);
            let copies = if options.rotate { (2 * gpu.l2 as usize).div_ceil(w.len() * 2).clamp(1, 16) } else { 1 };
            let wb = gpu.upload(&w.repeat(copies))?;
            let ab = gpu.upload(&a)?;
            // f16 modes: the same values in IEEE f16 (exact unless subnormal in f16).
            let to16 = |v: &[Bf16]| v.iter().map(|&x| half::f16::from_f32(bf16_to_f32(x)).to_bits()).collect::<Vec<u16>>();
            let f16_bufs = if tiles.iter().any(|t| t.f16 != 0) { Some((gpu.upload(&to16(&w).repeat(copies))?, gpu.upload(&to16(&a))?)) } else { None };
            let yb = gpu.alloc(mmax as usize * n as usize * 2)?;
            let initial = if options.out == Out::Residual { rng.vec(mmax as usize * n as usize, 0.3) } else { Vec::new() };
            // Sampled entries: corners, the last row, and random ones.
            let mut samples: Vec<(usize, usize)> = Vec::new();
            for &m in ms_rows {
                let m = m as usize;
                for &(r, c) in &[(0, 0), (0, n as usize - 1), (m - 1, 0), (m - 1, n as usize - 1)] {
                    samples.push((r, c));
                }
            }
            for _ in 0..2048 {
                let r = (rng.next() % mmax as u64) as usize;
                let c = (rng.next() % n as u64) as usize;
                samples.push((r, c));
            }
            for round in 0..options.rounds {
            for ti in 0..tiles.len() {
                let t = tiles[if round % 2 == 0 { ti } else { tiles.len() - 1 - ti }];
                if n % t.bn != 0 || kd % t.bk != 0 {
                    continue;
                }
                let k = gpu.load(&gemm::gemm(kd, n, options.out, t))?;
                let pad_control = if t.swizzle { Some(gpu.load(&gemm::gemm(kd, n, options.out, Tile { swizzle: false, ..t }))?) } else { None };
                let order_control = if t.schedule != 0 || t.pipeline != 0 { Some(gpu.load(&gemm::gemm(kd, n, options.out, Tile { schedule: 0, pipeline: 0, ..t }))?) } else { None };
                let (wbt, abt) = match (&f16_bufs, t.f16) { (Some((w16, a16)), 1..) => (w16, a16), _ => (&wb, &ab) };
                let f16_control = if t.f16 != 0 { Some(gpu.load(&gemm::gemm(kd, n, options.out, Tile { f16: 0, ..t }))?) } else { None };
                let path_control = if !t.ldmatrix || !t.async_copy || t.wmma { Some(gpu.load(&gemm::gemm(kd, n, options.out, Tile { ldmatrix: true, async_copy: true, wmma: false, ..t }))?) } else { None };
                println!("resources {}: {}", t.tag(), gpu.launch_resources(&k)?);
                for &m in ms_rows {
                    if options.out == Out::Residual { gpu.write(&yb, 0, crate::core::bf16::as_bytes(&initial))?; } else { gpu.zero(&yb)?; }
                    let grid = t.grid(m, n);
                    let args = [abt.into(), wbt.into(), (&yb).into(), Arg::U32(m)];
                    gpu.launch(&k, grid, &args)?;
                    let mut y = vec![0u16; m as usize * n as usize];
                    gpu.read(&yb, 0, &mut y)?;
                    let mut err = Errors::default();
                    for &(r, c) in samples.iter().filter(|s| s.0 < m as usize) {
                        let terms = (0..kd as usize).map(|i| bf16_to_f32(a[r * kd as usize + i]) as f64 * bf16_to_f32(w[c * kd as usize + i]) as f64);
                        let (mut want, mut mag) = terms.fold((0.0, 0.0), |(s, m), t| (s + t, m + t.abs()));
                        if options.out == Out::Residual {
                            let old = bf16_to_f32(initial[r * n as usize + c]);
                            want = bf16_to_f32(f32_to_bf16(old + bf16_to_f32(f32_to_bf16(want as f32)))) as f64;
                            mag += old.abs() as f64;
                        }
                        err.add(bf16_to_f32(y[r * n as usize + c]), want, mag);
                    }
                    if err.nonfinite != 0 { return Err(HerbertError::Backend("GEMM produced non-finite values".into())); }
                    if let Some(control) = &pad_control {
                        if options.out == Out::Residual { gpu.write(&yb, 0, crate::core::bf16::as_bytes(&initial))?; } else { gpu.zero(&yb)?; }
                        gpu.launch(control, grid, &args)?;
                        let mut pad = vec![0u16; y.len()];
                        gpu.read(&yb, 0, &mut pad)?;
                        if let Some(at) = pad.iter().zip(&y).position(|(a, b)| a != b) {
                            return Err(HerbertError::Backend(format!("XOR vs padded layout mismatch at {at}, M={m}, {}", t.tag())));
                        }
                        println!("layout-check {} M {m}: {} BF16 entries identical to pad", t.tag(), y.len());
                    }
                    if let Some(control) = &order_control {
                        if options.out == Out::Residual { gpu.write(&yb, 0, crate::core::bf16::as_bytes(&initial))?; } else { gpu.zero(&yb)?; }
                        gpu.launch(control, grid, &args)?;
                        let mut original = vec![0u16; y.len()];
                        gpu.read(&yb, 0, &mut original)?;
                        if let Some(at) = original.iter().zip(&y).position(|(a, b)| a != b) {
                            return Err(HerbertError::Backend(format!("schedule/pipeline mismatch at {at}, M={m}, {}", t.tag())));
                        }
                        println!("order-check {} M {m}: {} BF16 entries identical to schedule0/pipeline0", t.tag(), y.len());
                    }
                    if let Some(control) = &path_control {
                        if options.out == Out::Residual { gpu.write(&yb, 0, crate::core::bf16::as_bytes(&initial))?; } else { gpu.zero(&yb)?; }
                        gpu.launch(control, grid, &args)?;
                        let mut original = vec![0u16; y.len()];
                        gpu.read(&yb, 0, &mut original)?;
                        if let Some(at) = original.iter().zip(&y).position(|(a, b)| a != b) {
                            return Err(HerbertError::Backend(format!("ldmatrix/cp.async/mma path mismatch at {at}, M={m}, {}", t.tag())));
                        }
                        println!("path-check {} M {m}: {} BF16 entries identical to ldmatrix/cp.async/mma", t.tag(), y.len());
                    }
                    let reps = (2e10 / (2.0 * m as f64 * n as f64 * kd as f64)).clamp(3.0, 200.0) as u32;
                    if let Some(control) = &f16_control {
                        // Against the bf16 kernel on the bf16 data: a count, not a check.
                        if options.out == Out::Residual { gpu.write(&yb, 0, crate::core::bf16::as_bytes(&initial))?; } else { gpu.zero(&yb)?; }
                        gpu.launch(control, grid, &[(&ab).into(), (&wb).into(), (&yb).into(), Arg::U32(m)])?;
                        let mut base = vec![0u16; y.len()];
                        gpu.read(&yb, 0, &mut base)?;
                        let differ = base.iter().zip(&y).filter(|(a, b)| a != b).count();
                        let worst = base.iter().zip(&y).map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs()).max().unwrap_or(0);
                        println!("f16-check {} M {m}: {differ} of {} BF16 entries differ from the bf16 kernel, at most {worst} bf16 steps", t.tag(), y.len());
                    }
                    let ms = timed(gpu, 2, reps, |i| gpu.launch(&k, grid, &[abt.into(), Arg::Ptr(wbt.ptr + (i as usize % copies * w.len() * 2) as u64), (&yb).into(), Arg::U32(m)]))?;
                    let flops = 2.0 * m as f64 * n as f64 * kd as f64;
                    println!(
                        "gemm {model:>4} {what:<12} M {m:<5} {kd:>5}×{n:<6} {}: {:9.2} µs {:6.1} TFLOPS — {} — {} — {} round {} weight copies {}",
                        t.tag(),
                        ms * 1e3,
                        flops / ms / 1e9,
                        err.line(),
                        ptxas_line(&k), options.out.tag(), round + 1, copies
                    );
                }
            }
            }
        }
    }
    Ok(())
}
