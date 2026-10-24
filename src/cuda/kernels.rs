//! The emitted kernels of one decode step (M = 1), and the probe's two
//! measuring kernels (selftest, read bandwidth).
//!
//! Numerical contract, the CPU engine's: bf16 in memory, f32 inside a kernel,
//! one rounding to nearest even at the output. Every dimension is an
//! immediate of the module; each emitter returns an [`Emitted`] whose
//! parameters are listed in its doc, in order.

use crate::ptx::{Cmp, Emitted, Kernel, Ld, D, F, R};

/// Where a projection writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// `y[n] = bf16(acc)`.
    Bf16,
    /// `y[n] = acc` in f32.
    F32,
    /// `y[n] = bf16(y[n] + bf16(acc))`: the projection and the residual add.
    Residual,
}

impl Out {
    pub fn tag(self) -> &'static str {
        match self {
            Out::Bf16 => "bf16",
            Out::F32 => "f32",
            Out::Residual => "res",
        }
    }
}

// ── probe kernels ────────────────────────────────────────────────────────────

/// The emitter checked end to end: params `x (u32 pairs of bf16), y (u32
/// pairs), s (f32 per warp), n (pairs)`; `y = bf16(2·x + 1)` per element and
/// `s[warp] = Σ` over the warp of `lo + hi` (f32). 256 threads, `ceil(n/256)` blocks.
pub fn selftest() -> Emitted {
    let mut k = Kernel::new("selftest", 256);
    let x = k.param_ptr();
    let y = k.param_ptr();
    let s = k.param_ptr();
    let n = k.param_u32();
    let tid = k.tid();
    let b = k.ctaid_x();
    let i = k.mad(b, 256u32, tid);
    let inside = k.setp(Cmp::Lt, i, n);
    let sum = k.fmov(0.0);
    k.if_(inside, |k| {
        let a = k.index(x, i, 4);
        let pair = k.ld_u32(Ld::Nc, a, 0);
        let lo = k.h_lo(pair);
        let hi = k.h_hi(pair);
        let lo2 = k.ffma(lo, 2.0, 1.0);
        let hi2 = k.ffma(hi, 2.0, 1.0);
        let packed = k.pack_h(lo2, hi2);
        let o = k.index(y, i, 4);
        k.st_u32(o, 0, packed);
        k.fadd_to(sum, lo, hi);
    });
    let total = k.warp_sum(sum);
    let lane = k.and(tid, 31u32);
    let first = k.setp(Cmp::Eq, lane, 0u32);
    let warp = k.shr(i, 5u32);
    let o = k.index(s, warp, 4);
    k.when(first);
    k.st_f32(o, 0, total);
    Emitted::of(k, 0)
}

/// The operand and accumulator types of one `mma.m16n8k16`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mma {
    /// bf16 × bf16 → f32, the engine's.
    Bf16F32,
    /// f16 × f16 → f32.
    F16F32,
    /// f16 × f16 → f16.
    F16F16,
}

impl Mma {
    pub const ALL: [Mma; 3] = [Mma::Bf16F32, Mma::F16F32, Mma::F16F16];
    pub fn tag(self) -> &'static str {
        match self {
            Mma::Bf16F32 => "bf16_f32",
            Mma::F16F32 => "f16_f32",
            Mma::F16F16 => "f16_f16",
        }
    }
}

/// Tensor core throughput: every warp runs `chains` independent
/// accumulators, `unroll` `mma.m16n8k16` each per pass, `iters` passes, on
/// operands held in registers (no memory in the loop). The operands are
/// 2⁻¹⁰ (bf16 or f16, the lowest bit flipped on odd lanes) so that the
/// accumulators stay finite over millions of products; the accumulators are
/// folded by xor and stored, so the products cannot be dropped. Params `out
/// (u32 per thread), iters`; 128 threads.
pub fn mma_peak(kind: Mma, chains: u32, unroll: u32) -> Emitted {
    let threads = 128;
    let mut k = Kernel::new(format!("mma_peak_{}_c{chains}_u{unroll}", kind.tag()), threads);
    let out = k.param_ptr();
    let iters = k.param_u32();
    let tid = k.tid();
    let b = k.ctaid_x();
    let gid = k.mad(b, threads, tid);
    let odd = k.and(tid, 1u32);
    let one = match kind { Mma::Bf16F32 => 0x3a80_3a80u32, Mma::F16F32 | Mma::F16F16 => 0x1400_1400 };
    let a: [R; 4] = std::array::from_fn(|_| k.xor(odd, one));
    let bb: [R; 2] = std::array::from_fn(|_| k.xor(odd, one));
    let acc32: Vec<[F; 4]> = if kind == Mma::F16F16 { Vec::new() } else {
        (0..chains).map(|_| [k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)]).collect()
    };
    let acc16: Vec<[R; 2]> = if kind == Mma::F16F16 { (0..chains).map(|_| [k.mov(0u32), k.mov(0u32)]).collect() } else { Vec::new() };
    k.for_(0u32, iters, 1, |k, _| {
        for _ in 0..unroll {
            for c in 0..chains as usize {
                match kind {
                    Mma::Bf16F32 => k.mma_h(&acc32[c], &a, &bb),
                    Mma::F16F32 => k.mma_f16(&acc32[c], &a, &bb),
                    Mma::F16F16 => k.mma_f16_f16(&acc16[c], &a, &bb),
                }
            }
        }
    });
    let fold = k.mov(0u32);
    for acc in &acc32 {
        for &v in acc {
            k.ins(format!("xor.b32 {fold}, {fold}, {v}"));
        }
    }
    for acc in &acc16 {
        for &v in acc {
            k.ins(format!("xor.b32 {fold}, {fold}, {v}"));
        }
    }
    let o = k.index(out, gid, 4);
    k.st_u32(o, 0, fold);
    Emitted::of(k, 0)
}

/// Read bandwidth: every thread strides over `n16` chunks of 16 bytes and
/// folds them by xor; the fold is stored only if it equals a value the data
/// never produces (the loads cannot be dropped). Params `src, sink, n16`.
pub fn read(cache: Ld, unroll: u32) -> Emitted {
    let tag = match cache {
        Ld::Ca => "ca",
        Ld::Nc => "nc",
        Ld::NcNoL1 => "nc_nol1",
        Ld::Cg => "cg",
        Ld::Cs => "cs",
    };
    let threads = 256;
    let mut k = Kernel::new(format!("read_{tag}_u{unroll}"), threads);
    let src = k.param_ptr();
    let sink = k.param_ptr();
    let n16 = k.param_u32();
    let tid = k.tid();
    let b = k.ctaid_x();
    let nb = k.nctaid_x();
    let first = k.mad(b, threads, tid);
    let stride = k.mul(nb, threads);
    let acc = k.mov(0u32);
    // `unroll` loads per pass; the pass runs while its last chunk is inside.
    let span = k.mul(stride, unroll);
    let last = k.sub(n16, span);
    let room = k.setp(Cmp::Ge, n16, span);
    let i = k.mov(first);
    k.if_(room, |k| {
        let (top, out) = (k.label(), k.label());
        let p = k.setp(Cmp::Gt, i, last);
        k.bra_if(p, out);
        k.place(top);
        let mut words = Vec::new();
        let base = k.index(src, i, 16);
        let step = k.mul(stride, 16u32);
        let step64 = k.d();
        k.ins(format!("cvt.u64.u32 {step64}, {step}"));
        let mut a = base;
        for u in 0..unroll {
            if u > 0 {
                let n = k.d();
                k.ins(format!("add.s64 {n}, {a}, {step64}"));
                a = n;
            }
            words.push(k.ld_v4(cache, a, 0));
        }
        for v in words {
            for w in v {
                k.ins(format!("xor.b32 {acc}, {acc}, {w}"));
            }
        }
        k.add_to(i, i, span);
        let p = k.setp(Cmp::Le, i, last);
        k.bra_if(p, top);
        k.place(out);
    });
    // The rest, one chunk at a time.
    k.for_(i, n16, 0, |k, j| {
        let a = k.index(src, j, 16);
        let v = k.ld_v4(cache, a, 0);
        for w in v {
            k.ins(format!("xor.b32 {acc}, {acc}, {w}"));
        }
        k.add_to(j, j, stride);
    });
    let hit = k.setp(Cmp::Eq, acc, 0x5EED_F00Du32);
    let o = k.index(sink, first, 4);
    k.when(hit);
    k.st_u32(o, 0, acc);
    Emitted::of(k, 0)
}

// ── projections ──────────────────────────────────────────────────────────────

/// How a matvec is cut and read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Matvec {
    /// Rows of `W` per warp (they share the loads of `x`).
    pub rows: u32,
    /// Warps per block.
    pub warps: u32,
    /// Chunks of 16 bytes in flight per row before the first FMA of a batch.
    pub depth: u32,
    /// How the weights are loaded.
    pub cache: Ld,
    /// The weights in IEEE f16 instead of bf16 (`gemm.f16`); `x` stays bf16.
    pub w16: bool,
}

impl Matvec {
    pub const DEFAULT: Matvec = Matvec { rows: 2, warps: 4, depth: 8, cache: Ld::Nc, w16: false };

    pub fn groups(&self, n: u32) -> u32 {
        n.div_ceil(self.rows * self.warps)
    }
}

/// `y = W·x`, `W` `[n][k]` bf16 row-major, `x` `[k]` bf16, f32 accumulation.
/// Each warp reduces `rows` consecutive rows, the lanes striding over `k` by
/// 8 bf16 (16 bytes); the loop over `k` is unrolled at emission (`k` is
/// known), its loads issued `depth` chunks ahead of the FMAs.
/// Params `W, x, y`; blocks [`Matvec::groups`].
pub fn matvec(kdim: u32, n: u32, out: Out, t: Matvec) -> Emitted {
    assert!(kdim.is_multiple_of(8), "K = {kdim} is not a multiple of 8");
    let cache = match t.cache {
        Ld::Ca => "ca",
        Ld::Nc => "nc",
        Ld::NcNoL1 => "nol1",
        Ld::Cg => "cg",
        Ld::Cs => "cs",
    };
    let name = format!("matvec_{kdim}x{n}_{}_r{}w{}d{}{cache}{}", out.tag(), t.rows, t.warps, t.depth, if t.w16 { "_wh" } else { "" });
    let mut k = Kernel::new(name, t.warps * 32);
    let w = k.param_ptr();
    let x = k.param_ptr();
    let y = k.param_ptr();
    let tid = k.tid();
    let warp = k.shr(tid, 5u32);
    let lane = k.and(tid, 31u32);
    let b = k.ctaid_x();
    let slot = k.mad(b, t.warps, warp);
    let row0 = k.mul(slot, t.rows);
    let outside = k.setp(Cmp::Ge, row0, n);
    let end = k.label();
    k.bra_if(outside, end);

    let chunks = kdim / 8;
    let passes = chunks.div_ceil(32);
    let tail = chunks % 32; // lanes of the last pass, 0 = all
    let lane16 = k.mul(lane, 16u32);
    // Rows past the end read the last row and are not written.
    let bases: Vec<D> = (0..t.rows)
        .map(|r| {
            let row = k.add(row0, r);
            let row = k.min(row, n - 1);
            let at = k.index(w, row, kdim * 2);
            k.index(at, lane16, 1)
        })
        .collect();
    let xb = k.index(x, lane16, 1);
    let accs: Vec<F> = (0..t.rows).map(|_| k.fmov(0.0)).collect();
    let in_tail = if tail != 0 { Some(k.setp(Cmp::Lt, lane, tail)) } else { None };

    // One pass: x's chunk and each row's chunk, then the 8 FMAs per row.
    let fma = |k: &mut Kernel, xv: [R; 4], wv: &[[R; 4]]| {
        let xs: Vec<(F, F)> = xv.iter().map(|&p| (k.h_lo(p), k.h_hi(p))).collect();
        for (r, wr) in wv.iter().enumerate() {
            for (i, &p) in wr.iter().enumerate() {
                if t.w16 {
                    let (lo, hi) = k.f16x2_to_f32(p);
                    k.ffma_to(accs[r], lo, xs[i].0, accs[r]);
                    k.ffma_to(accs[r], hi, xs[i].1, accs[r]);
                    continue;
                }
                let lo = k.h_lo(p);
                k.ffma_to(accs[r], lo, xs[i].0, accs[r]);
                let hi = k.h_hi(p);
                k.ffma_to(accs[r], hi, xs[i].1, accs[r]);
            }
        }
    };
    let full = if tail != 0 { passes - 1 } else { passes };
    let mut p = 0;
    while p < full {
        let batch = t.depth.min(full - p);
        let mut loads = Vec::new();
        for q in p..p + batch {
            let off = (q * 32 * 16) as i32;
            let xv = k.ld_v4(Ld::Nc, xb, off);
            let wv: Vec<[R; 4]> = bases.iter().map(|&a| k.ld_v4(t.cache, a, off)).collect();
            loads.push((xv, wv));
        }
        for (xv, wv) in loads {
            fma(&mut k, xv, &wv);
        }
        p += batch;
    }
    if let Some(pt) = in_tail {
        k.if_(pt, |k| {
            let off = (full * 32 * 16) as i32;
            let xv = k.ld_v4(Ld::Nc, xb, off);
            let wv: Vec<[R; 4]> = bases.iter().map(|&a| k.ld_v4(t.cache, a, off)).collect();
            fma(k, xv, &wv);
        });
    }

    let first = k.setp(Cmp::Eq, lane, 0u32);
    for (r, &a) in accs.iter().enumerate() {
        let sum = k.warp_sum(a);
        let row = k.add(row0, r as u32);
        let inside = k.setp(Cmp::Lt, row, n);
        let write = k.pand(first, inside);
        match out {
            Out::F32 => {
                let o = k.index(y, row, 4);
                k.when(write);
                k.st_f32(o, 0, sum);
            }
            Out::Bf16 => {
                let h = k.f_to_h(sum);
                let o = k.index(y, row, 2);
                k.when(write);
                k.st_b16(o, 0, h);
            }
            Out::Residual => {
                let o = k.index(y, row, 2);
                k.if_(write, |k| {
                    let r = k.round_h(sum);
                    let old = k.ld_b16(Ld::Ca, o, 0);
                    let of = k.h_to_f(old);
                    let s = k.fadd(of, r);
                    let h = k.f_to_h(s);
                    k.st_b16(o, 0, h);
                });
            }
        }
    }
    k.place(end);
    Emitted::of(k, 0)
}
