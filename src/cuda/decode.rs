//! The emitted kernels of one decode step besides the projections
//! (`kernels::matvec`), in the order of a layer: embedding, RMSNorm, the
//! attention (head norms + RoPE + KV append, split-K attention and its
//! combination), the MLP's SwiGLU, the GatedDeltaNet (conv + L2 + gates, the
//! delta rule, the gated norm) and the argmax of the head.
//!
//! The semantics are those of herbert-vk-bf16's `vk/kernels.rs`, itself
//! checked against the CPU engine: same operations in the same order, f32
//! inside, one rounding to nearest even at each output. Every dimension is an
//! immediate; the position and the token are read from the `state` buffer
//! (`types::STATE_*`), so one recorded step (a CUDA graph) serves every token.

use super::types::{AttnShape, DeltaShape, STATE_ARGMAX, STATE_POS, STATE_TOKEN};
use crate::ptx::{Cmp, Emitted, Kernel, Ld, D, F, R};

/// The position of the token fed (of the first row of a prefill block).
pub(crate) fn pos(k: &mut Kernel, state: D) -> R {
    k.ld_u32(Ld::Ca, state, (4 * STATE_POS) as i32)
}

/// With `rows` (a prefill block), the row of this block, `ctaid.y`.
fn row_of(k: &mut Kernel, rows: bool) -> Option<R> {
    rows.then(|| k.ctaid_y())
}

/// `base + row·bytes`, or `base` without a row.
fn at_row(k: &mut Kernel, base: D, row: Option<R>, bytes: u32) -> D {
    match row {
        Some(r) => k.index(base, r, bytes),
        None => base,
    }
}

fn rows_tag(rows: bool) -> &'static str {
    if rows {
        "_rows"
    } else {
        ""
    }
}

/// `y = bf16((x·inv)·(1+w))` (or `·w`), `inv = 1/√(Σx²/d + ε)`, for one row of
/// `d` held by `threads` threads, two elements per pass; `x`, `w`, `y` point
/// at the row. Shared `scratch` holds one f32 per warp.
#[allow(clippy::too_many_arguments)]
pub(crate) fn norm_row(k: &mut Kernel, x: D, w: D, y: D, d: u32, eps: f32, plus_one: bool, scratch: R, tid: R, h16: bool) {
    let threads = k.threads();
    let pairs = d / 2;
    let passes = pairs.div_ceil(threads);
    let mut vals = Vec::new();
    let ss = k.fmov(0.0);
    for p in 0..passes {
        let i = k.add(tid, p * threads);
        let inside = k.setp(Cmp::Lt, i, pairs);
        let a = k.index(x, i, 4);
        let pair = k.mov(0u32);
        k.when(inside);
        k.ins(format!("ld.global.u32 {pair}, [{a}]"));
        let lo = k.h_lo(pair);
        let hi = k.h_hi(pair);
        k.ffma_to(ss, lo, lo, ss);
        k.ffma_to(ss, hi, hi, ss);
        vals.push((i, inside, lo, hi));
    }
    let total = k.block_sum(ss, scratch);
    let m = k.fmul(total, 1.0 / d as f32);
    let m = k.fadd(m, eps);
    let inv = k.finv_sqrt(m);
    for (i, inside, lo, hi) in vals {
        k.if_(inside, |k| {
            let wa = k.index(w, i, 4);
            let wp = k.ld_u32(Ld::Nc, wa, 0);
            let mut wl = k.h_lo(wp);
            let mut wh = k.h_hi(wp);
            if plus_one {
                wl = k.fadd(wl, 1.0);
                wh = k.fadd(wh, 1.0);
            }
            let nl = k.fmul(lo, inv);
            let ol = k.fmul(nl, wl);
            let nh = k.fmul(hi, inv);
            let oh = k.fmul(nh, wh);
            let p = k.pack16(ol, oh, h16);
            let ya = k.index(y, i, 4);
            k.st_u32(ya, 0, p);
        });
    }
}

// ── embedding and norms ──────────────────────────────────────────────────────

/// `x = embed[token]`, 16 bytes per thread. Params `state, table, x`; one
/// block. With `rows`: params `tokens (u32 [m]), table, x`, blocks `(1, m)`.
pub fn embed(hidden: u32, rows: bool) -> Emitted {
    assert!(hidden.is_multiple_of(8));
    let threads = hidden / 8;
    let mut k = Kernel::new(format!("embed_{hidden}{}", rows_tag(rows)), threads);
    let state = k.param_ptr();
    let table = k.param_ptr();
    let x = k.param_ptr();
    let row = row_of(&mut k, rows);
    let x = at_row(&mut k, x, row, hidden * 2);
    let tid = k.tid();
    let token = match row {
        Some(r) => {
            let ta = k.index(state, r, 4);
            k.ld_u32(Ld::Ca, ta, 0)
        }
        None => k.ld_u32(Ld::Ca, state, (4 * STATE_TOKEN) as i32),
    };
    let row = k.index(table, token, hidden * 2);
    let src = k.index(row, tid, 16);
    let v = k.ld_v4(Ld::Nc, src, 0);
    let dst = k.index(x, tid, 16);
    k.st_v4(dst, 0, v);
    Emitted::of(k, 0)
}

/// RMSNorm of one vector of `d` (bf16 in and out). Params `x, w, y`; one
/// block (`rows`: blocks `(1, m)`, one per row).
/// `h16`: the output in IEEE f16 (the f16 experiment, `gemm.f16`: it feeds a GEMM).
pub fn rmsnorm(d: u32, eps: f32, plus_one: bool, rows: bool, h16: bool) -> Emitted {
    let mut k = Kernel::new(format!("rmsnorm_{d}{}{}{}", if plus_one { "_p1" } else { "" }, rows_tag(rows), if h16 { "_h16" } else { "" }), 256);
    let x = k.param_ptr();
    let w = k.param_ptr();
    let y = k.param_ptr();
    let row = row_of(&mut k, rows);
    let x = at_row(&mut k, x, row, d * 2);
    let y = at_row(&mut k, y, row, d * 2);
    let scratch = k.shared(32 * 4);
    let tid = k.tid();
    norm_row(&mut k, x, w, y, d, eps, plus_one, scratch, tid, h16);
    Emitted::of(k, 0)
}

// ── full attention ───────────────────────────────────────────────────────────

/// Head norms, partial RoPE and the KV append, one block per head: blocks
/// `heads` (Q, in place), `kv_heads` (K, into the cache), `kv_heads` (V,
/// copied into the cache). Per head: `y = bf16((x·inv)·(1+w))`, then on the
/// first `rotary` dims the pairs `(i, i + rotary/2)` rotate, `a' = a·c − b·s`,
/// `b' = a·s + b·c`, with the bf16 tables of the position.
/// Params `state, qkv, q_norm, k_norm, cos, sin, k_cache, v_cache`. With
/// `rows`: blocks `(…, m)`, row r at position `pos + r`.
pub fn qk_rope_append(s: AttnShape, rows: bool) -> Emitted {
    let hd = s.head_dim;
    let half = s.rotary / 2;
    let mut k = Kernel::new(format!("qk_rope_append_h{}kv{}d{hd}{}", s.heads, s.kv_heads, rows_tag(rows)), hd);
    let state = k.param_ptr();
    let qkv = k.param_ptr();
    let qn = k.param_ptr();
    let kn = k.param_ptr();
    let cos = k.param_ptr();
    let sin = k.param_ptr();
    let kc = k.param_ptr();
    let vc = k.param_ptr();
    let scratch = k.shared(32 * 4);
    let row = k.shared(hd * 4);
    let r = row_of(&mut k, rows);
    let qkv = at_row(&mut k, qkv, r, s.qkv_width() * 2);
    let g = k.ctaid_x();
    let lid = k.tid();
    let p = pos(&mut k, state);
    let p = match r {
        Some(r) => k.add(p, r),
        None => p,
    };
    let is_q = k.setp(Cmp::Lt, g, s.heads);
    let is_qk = k.setp(Cmp::Lt, g, s.heads + s.kv_heads);
    let (v_path, end) = (k.label(), k.label());
    k.bra_unless(is_qk, v_path);
    {
        // Q head g, or K head g − heads.
        let kvh = k.sub(g, s.heads);
        let q_at = k.mul(g, 2 * hd);
        let kv_at = k.mad(kvh, hd, s.q_width());
        let src = k.selp(q_at, kv_at, is_q);
        let at = k.add(src, lid);
        let xa = k.index(qkv, at, 2);
        let xh = k.ld_b16(Ld::Ca, xa, 0);
        let x = k.h_to_f(xh);
        let x2 = k.fmul(x, x);
        let ss = k.block_sum(x2, scratch);
        let m = k.fmul(ss, 1.0 / hd as f32);
        let m = k.fadd(m, s.eps);
        let inv = k.finv_sqrt(m);
        let wq = k.index(qn, lid, 2);
        let wk = k.index(kn, lid, 2);
        let wqh = k.ld_b16(Ld::Nc, wq, 0);
        let wkh = k.ld_b16(Ld::Nc, wk, 0);
        let wqf = k.h_to_f(wqh);
        let wkf = k.h_to_f(wkh);
        let wf = k.fselp(wqf, wkf, is_q);
        let wf = k.fadd(wf, 1.0);
        let n = k.fmul(x, inv);
        let y = k.fmul(n, wf);
        let y = k.round_h(y);
        let ra = k.mad(lid, 4u32, row);
        k.sts_f32(ra, 0, y);
        k.bar();
        let out = k.fmov(y);
        let in_rot = k.setp(Cmp::Lt, lid, s.rotary);
        k.if_(in_rot, |k| {
            let first = k.setp(Cmp::Lt, lid, half);
            let j_hi = k.sub(lid, half);
            let j = k.selp(lid, j_hi, first);
            let t = k.mad(p, half, j);
            let ca = k.index(cos, t, 2);
            let sa = k.index(sin, t, 2);
            let ch = k.ld_b16(Ld::Nc, ca, 0);
            let sh = k.ld_b16(Ld::Nc, sa, 0);
            let c = k.h_to_f(ch);
            let sn = k.h_to_f(sh);
            let aa = k.mad(j, 4u32, row);
            let a = k.lds_f32(aa, 0);
            let b = k.lds_f32(aa, (half * 4) as i32);
            let ac = k.fmul(a, c);
            let bs = k.fmul(b, sn);
            let lo = k.fsub(ac, bs);
            let as_ = k.fmul(a, sn);
            let bc = k.fmul(b, c);
            let hi = k.fadd(as_, bc);
            let r = k.fselp(lo, hi, first);
            k.fmov_to(out, r);
        });
        let b = k.f_to_h(out);
        k.if_(is_q, |k| k.st_b16(xa, 0, b));
        let not_q = k.pnot(is_q);
        k.if_(not_q, |k| {
            let base = k.mul(kvh, s.capacity * hd);
            let a = k.mad(p, hd, base);
            let a = k.add(a, lid);
            let ka = k.index(kc, a, 2);
            k.st_b16(ka, 0, b);
        });
        k.bra(end);
    }
    k.place(v_path);
    {
        // V head g − heads − kv_heads: copied as is.
        let vh = k.sub(g, s.heads + s.kv_heads);
        let src = k.mad(vh, hd, s.q_width() + s.kv_width());
        let src = k.add(src, lid);
        let va = k.index(qkv, src, 2);
        let v = k.ld_b16(Ld::Ca, va, 0);
        let base = k.mul(vh, s.capacity * hd);
        let a = k.mad(p, hd, base);
        let a = k.add(a, lid);
        let dst = k.index(vc, a, 2);
        k.st_b16(dst, 0, v);
    }
    k.place(end);
    Emitted::of(k, 0)
}

/// Split-K decode attention, first pass: `splits` blocks per query head, each
/// over its share of the `pos + 1` positions. Scores by one warp per position
/// (each lane 8 dims: one 16-byte load of K), `q·k / √hd` in f32 into the
/// `scores` scratch; then `e = exp(s − m)` with the block's max `m`, their sum
/// `l`, and per dimension the unnormalized `Σ e·v`, in f32.
/// Params `state, qkv, k_cache, v_cache, scores, partial`; blocks
/// `heads · splits`. `partial` = `[heads·splits][hd]` sums, then
/// `[heads·splits][2]` (m, l).
pub fn sdpa_split(s: AttnShape, splits: u32) -> Emitted {
    let hd = s.head_dim;
    assert_eq!(hd, 256, "one lane per 8 dims, one thread per dim");
    let threads = 256;
    let warps = threads / 32;
    let mut k = Kernel::new(format!("sdpa_split_h{}kv{}d{hd}_s{splits}", s.heads, s.kv_heads), threads);
    let state = k.param_ptr();
    let qkv = k.param_ptr();
    let kc = k.param_ptr();
    let vc = k.param_ptr();
    let scores = k.param_ptr();
    let partial = k.param_ptr();
    let scratch = k.shared(32 * 4);
    let g = k.ctaid_x();
    let h = k.div(g, splits);
    let sp = k.rem(g, splits);
    let tid = k.tid();
    let lane = k.and(tid, 31u32);
    let warp = k.shr(tid, 5u32);
    let p = pos(&mut k, state);
    let len = k.add(p, 1u32);
    let lp = k.add(len, splits - 1);
    let chunk = k.div(lp, splits);
    let t0 = k.mul(sp, chunk);
    let t1 = k.add(t0, chunk);
    let t1 = k.min(t1, len);
    let kvh = k.div(h, s.heads / s.kv_heads);
    // q: 8 dims per lane.
    let q_base = k.mul(h, 2 * hd);
    let qd = k.mad(lane, 8u32, q_base);
    let qa = k.index(qkv, qd, 2);
    let qv = k.ld_v4(Ld::Ca, qa, 0);
    let q: Vec<F> = qv.iter().flat_map(|&w| [k.h_lo(w), k.h_hi(w)]).collect();
    let k_rows = k.mul(kvh, s.capacity);
    let row_base = k.mul(h, s.capacity);
    let scale = 1.0 / (hd as f32).sqrt();
    let local_max = k.fmov(f32::MIN);
    let first_t = k.add(t0, warp);
    let lane16 = k.mul(lane, 16u32);
    let lane0 = k.setp(Cmp::Eq, lane, 0u32);
    k.for_(first_t, t1, warps, |k, t| {
        let r = k.add(k_rows, t);
        let ka = k.index(kc, r, hd * 2);
        let ka = k.index(ka, lane16, 1);
        let kv = k.ld_v4(Ld::Ca, ka, 0);
        let dot = k.fmov(0.0);
        for (i, &w) in kv.iter().enumerate() {
            let lo = k.h_lo(w);
            k.ffma_to(dot, q[2 * i], lo, dot);
            let hi = k.h_hi(w);
            k.ffma_to(dot, q[2 * i + 1], hi, dot);
        }
        let sum = k.warp_sum(dot);
        let sc = k.fmul(sum, scale);
        let at = k.add(row_base, t);
        let sa = k.index(scores, at, 4);
        k.when(lane0);
        k.st_f32(sa, 0, sc);
        let m = k.fmax(local_max, sc);
        k.fmov_to(local_max, m);
    });
    let mx = k.block_max(local_max, scratch);
    let local_sum = k.fmov(0.0);
    let tl = k.add(t0, tid);
    k.for_(tl, t1, threads, |k, t| {
        let at = k.add(row_base, t);
        let sa = k.index(scores, at, 4);
        let sc = k.ld_f32(Ld::Ca, sa, 0);
        let d = k.fsub(sc, mx);
        let e = k.fexp(d);
        k.st_f32(sa, 0, e);
        k.fadd_to(local_sum, local_sum, e);
    });
    let total = k.block_sum(local_sum, scratch);
    let slot = k.mad(h, splits, sp);
    // One thread per dimension.
    let acc = k.fmov(0.0);
    k.for_(t0, t1, 1, |k, t| {
        let at = k.add(row_base, t);
        let sa = k.index(scores, at, 4);
        let e = k.ld_f32(Ld::Ca, sa, 0);
        let r = k.add(k_rows, t);
        let va = k.index(vc, r, hd * 2);
        let va = k.index(va, tid, 2);
        let vh = k.ld_b16(Ld::Ca, va, 0);
        let vf = k.h_to_f(vh);
        k.ffma_to(acc, e, vf, acc);
    });
    let oi = k.mad(slot, hd, tid);
    let oa = k.index(partial, oi, 4);
    k.st_f32(oa, 0, acc);
    let first = k.setp(Cmp::Eq, tid, 0u32);
    k.if_(first, |k| {
        let at = k.mad(slot, 2u32, s.heads * splits * hd);
        let ma = k.index(partial, at, 4);
        k.st_f32(ma, 0, mx);
        k.st_f32(ma, 4, total);
    });
    Emitted::of(k, 0)
}

/// Split-K decode attention, second pass, one block per head, one thread per
/// dimension: the partials rescaled to the global max, `o = Σ e^(m_i−M)·o_i /
/// Σ e^(m_i−M)·l_i`, rounded to bf16, then the output gate `bf16(o·σ(gate))`.
/// Params `qkv, partial, out`; blocks `heads`.
pub fn sdpa_combine(s: AttnShape, splits: u32) -> Emitted {
    let hd = s.head_dim;
    let mut k = Kernel::new(format!("sdpa_combine_h{}d{hd}_s{splits}", s.heads), hd);
    let qkv = k.param_ptr();
    let partial = k.param_ptr();
    let out = k.param_ptr();
    let h = k.ctaid_x();
    let d = k.tid();
    let ml0 = s.heads * splits * hd;
    let slot0 = k.mul(h, splits);
    let ma = k.mad(slot0, 2u32, ml0);
    let ma = k.index(partial, ma, 4);
    let ms: Vec<(F, F)> = (0..splits).map(|i| (k.ld_f32(Ld::Ca, ma, (8 * i) as i32), k.ld_f32(Ld::Ca, ma, (8 * i + 4) as i32))).collect();
    let mut mx = ms[0].0;
    for &(m, _) in &ms[1..] {
        mx = k.fmax(mx, m);
    }
    let mut den = k.fmov(0.0);
    let mut ws = Vec::new();
    for &(m, l) in &ms {
        let dm = k.fsub(m, mx);
        let w = k.fexp(dm);
        den = k.ffma(w, l, den);
        ws.push(w);
    }
    let inv = k.fdiv(1.0, den);
    let oi = k.mad(slot0, hd, d);
    let oa = k.index(partial, oi, 4);
    let mut num = k.fmov(0.0);
    for (i, &w) in ws.iter().enumerate() {
        let o = k.ld_f32(Ld::Ca, oa, (i as u32 * hd * 4) as i32);
        num = k.ffma(w, o, num);
    }
    let o = k.fmul(num, inv);
    let o = k.round_h(o);
    let q_base = k.mul(h, 2 * hd);
    let gi = k.add(q_base, hd);
    let gi = k.add(gi, d);
    let ga = k.index(qkv, gi, 2);
    let gh = k.ld_b16(Ld::Ca, ga, 0);
    let gf = k.h_to_f(gh);
    let sg = k.fsigmoid(gf);
    let r = k.fmul(o, sg);
    let b = k.f_to_h(r);
    let yi = k.mad(h, hd, d);
    let ya = k.index(out, yi, 2);
    k.st_b16(ya, 0, b);
    Emitted::of(k, 0)
}

// ── MLP ──────────────────────────────────────────────────────────────────────

/// `h = bf16(g·σ(g)·u)` over the fused `[gate (n) | up (n)]`, two elements per
/// thread. Params `gate_up, h`; blocks `ceil(n / 512)` (`rows`: `(…, m)`).
/// `h16`: the output in IEEE f16 (`gemm.f16`).
pub fn swiglu(n: u32, rows: bool, h16: bool) -> Emitted {
    assert!(n.is_multiple_of(2));
    let mut k = Kernel::new(format!("swiglu_{n}{}{}", rows_tag(rows), if h16 { "_h16" } else { "" }), 256);
    let gu = k.param_ptr();
    let h = k.param_ptr();
    let r = row_of(&mut k, rows);
    let gu = at_row(&mut k, gu, r, 4 * n);
    let h = at_row(&mut k, h, r, 2 * n);
    let tid = k.tid();
    let b = k.ctaid_x();
    let i = k.mad(b, 256u32, tid);
    let inside = k.setp(Cmp::Lt, i, n / 2);
    k.if_(inside, |k| {
        let ga = k.index(gu, i, 4);
        let gp = k.ld_u32(Ld::Ca, ga, 0);
        let up = k.ld_u32(Ld::Ca, ga, (n * 2) as i32);
        let mut o = Vec::new();
        for (g, u) in [(k.h_lo(gp), k.h_lo(up)), (k.h_hi(gp), k.h_hi(up))] {
            let s = k.fsigmoid(g);
            let gs = k.fmul(g, s);
            o.push(k.fmul(gs, u));
        }
        let p = k.pack16(o[0], o[1], h16);
        let ha = k.index(h, i, 4);
        k.st_u32(ha, 0, p);
    });
    Emitted::of(k, 0)
}

// ── GatedDeltaNet ────────────────────────────────────────────────────────────

/// The causal conv + SiLU, the L2 norms and the gates of one decode step:
/// blocks `2·k_heads` (a q or k head: conv of its `dk` channels, then the L2
/// norm `x·1/√(Σx² + 1e-6)`), `v_heads` (the conv of a value head's `dv`
/// channels), and one for the gates `β = σ(b)`, `g = −exp(A_log)·softplus(a +
/// dt_bias)` (softplus(x) = x past 20), stored as `g` (bf16 activations) or
/// `exp(g)` (f32). The conv state of a channel holds its last `conv` inputs,
/// the current one last. Params `proj, conv_state, conv_w, a_log, dt_bias,
/// gates`; blocks `2·k_heads + v_heads + 1`.
pub fn dn_conv_l2_gates(s: DeltaShape) -> Emitted {
    let (dk, dv, kk, act) = (s.k_dim, s.v_dim, s.conv, s.act);
    assert_eq!(dk, dv, "one thread per channel of a head");
    let mut k = Kernel::new(format!("dn_conv_l2_gates_k{}v{}_{}", s.k_heads, s.v_heads, act.tag()), dk);
    let proj = k.param_ptr();
    let st = k.param_ptr();
    let w = k.param_ptr();
    let a_log = k.param_ptr();
    let dt_bias = k.param_ptr();
    let gates = k.param_ptr();
    let scratch = k.shared(32 * 4);
    let g = k.ctaid_x();
    let lid = k.tid();
    let is_conv = k.setp(Cmp::Lt, g, 2 * s.k_heads + s.v_heads);
    let (gate_path, end) = (k.label(), k.label());
    k.bra_unless(is_conv, gate_path);
    {
        let c = k.mad(g, dk, lid);
        let base = k.mul(c, kk);
        let x = act.ld(&mut k, proj, c);
        let mut shifted = Vec::new();
        for j in 0..kk {
            let v = if j + 1 < kk {
                let at = k.add(base, j + 1);
                act.ld(&mut k, st, at)
            } else {
                x
            };
            shifted.push(v);
        }
        let mut acc = k.fmov(0.0);
        for (j, &v) in shifted.iter().enumerate() {
            let at = k.add(base, j as u32);
            act.st(&mut k, st, at, v);
            let wa = k.index(w, at, 2);
            let wh = k.ld_b16(Ld::Nc, wa, 0);
            let wf = k.h_to_f(wh);
            acc = k.ffma(v, wf, acc);
        }
        let sg = k.fsigmoid(acc);
        let y = k.fmul(acc, sg);
        let is_qk = k.setp(Cmp::Lt, g, 2 * s.k_heads);
        let v_head = k.label();
        k.bra_unless(is_qk, v_head);
        // The L2 norm over a q or k head (uniform per block).
        let y2 = k.fmul(y, y);
        let ss = k.block_sum(y2, scratch);
        let m = k.fadd(ss, 1e-6);
        let inv = k.finv_sqrt(m);
        let o = k.fmul(y, inv);
        act.st(&mut k, proj, c, o);
        k.bra(end);
        k.place(v_head);
        act.st(&mut k, proj, c, y);
        k.bra(end);
    }
    k.place(gate_path);
    k.for_(lid, s.v_heads, dk, |k, i| {
        let ai = k.add(i, s.a_at());
        let bi = k.add(i, s.b_at());
        let a = act.ld(k, proj, ai);
        let b = act.ld(k, proj, bi);
        let beta = k.fsigmoid(b);
        let ba = k.index(dt_bias, i, 4);
        let bias = k.ld_f32(Ld::Nc, ba, 0);
        let x = k.fadd(a, bias);
        let ex = k.fexp(x);
        let l = k.fadd(ex, 1.0);
        let sp = k.flog(l);
        let big = k.fsetp(Cmp::Gt, x, 20.0);
        let dt = k.fselp(x, sp, big);
        let la = k.index(a_log, i, 4);
        let al = k.ld_f32(Ld::Nc, la, 0);
        let ea = k.fexp(al);
        let prod = k.fmul(ea, dt);
        let neg = k.fneg(prod);
        // bf16 gates hold g = log(decay) (vLLM keeps g: near 1 a bf16 decay is
        // off by 2⁻⁹, a bf16 g by 2⁻⁹·|g|).
        let decay = match act {
            super::types::Act::Bf16 => neg,
            super::types::Act::F32 => k.fexp(neg),
        };
        act.st(k, gates, i, beta);
        let di = k.add(i, s.v_heads);
        act.st(k, gates, di, decay);
    });
    k.place(end);
    Emitted::of(k, 0)
}

/// The gated delta rule, one block per value head, one thread per value
/// column `j` holding its column of the state `S[h]` (`[dk][dv]`) in
/// registers. In the CPU engine's order: `s *= decay; kv = Σ s·k; δ = (v −
/// kv)·β; s += k⊗δ; o = Σ s·(q·scale)`. With bf16 states the decayed state is
/// rounded as vk-bf16 stores it between its two passes. Params `proj, gates,
/// state, out`; blocks `v_heads`.
pub fn dn_recurrence(s: DeltaShape) -> Emitted {
    let (dk, dv, act) = (s.k_dim, s.v_dim, s.act);
    let mut k = Kernel::new(format!("dn_recurrence_k{}v{}_{}", s.k_heads, s.v_heads, act.tag()), dv);
    let proj = k.param_ptr();
    let gates = k.param_ptr();
    let state = k.param_ptr();
    let out = k.param_ptr();
    let qs = k.shared(dk * 4);
    let ks = k.shared(dk * 4);
    let h = k.ctaid_x();
    let j = k.tid();
    let kh = k.div(h, s.v_heads / s.k_heads);
    let scale = 1.0 / (dk as f32).sqrt();
    k.for_(j, dk, dv, |k, i| {
        let qa = k.mad(kh, dk, i);
        let q = act.ld(k, proj, qa);
        let q = k.fmul(q, scale);
        let sa = k.mad(i, 4u32, qs);
        k.sts_f32(sa, 0, q);
        let ka = k.add(qa, s.key());
        let kv = act.ld(k, proj, ka);
        let sa = k.mad(i, 4u32, ks);
        k.sts_f32(sa, 0, kv);
    });
    k.bar();
    let beta = act.ld(&mut k, gates, h);
    let di = k.add(h, s.v_heads);
    let decay = act.ld(&mut k, gates, di);
    let decay = match act {
        super::types::Act::Bf16 => k.fexp(decay),
        super::types::Act::F32 => decay,
    };
    let vi = k.mad(h, dv, 2 * s.key());
    let vi = k.add(vi, j);
    let v = act.ld(&mut k, proj, vi);
    let sbase = k.mad(h, dk * dv, j);
    let sa = k.index(state, sbase, act.bytes());
    let row = (dv * act.bytes()) as i32;
    let load = |k: &mut Kernel, d: u32| -> F {
        match act {
            super::types::Act::Bf16 => {
                let hh = k.ld_b16(Ld::Ca, sa, row * d as i32);
                k.h_to_f(hh)
            }
            super::types::Act::F32 => k.ld_f32(Ld::Ca, sa, row * d as i32),
        }
    };
    let col: Vec<F> = (0..dk).map(|d| load(&mut k, d)).collect();
    let kv = k.fmov(0.0);
    for d in 0..dk {
        let sv = k.fmul(col[d as usize], decay);
        let kd = k.lds_f32(ks, (4 * d) as i32);
        k.ffma_to(kv, sv, kd, kv);
    }
    let diff = k.fsub(v, kv);
    let delta = k.fmul(diff, beta);
    let o = k.fmov(0.0);
    for d in 0..dk {
        // The decayed state again (the same product), as stored: one column
        // live instead of two.
        let dec = k.fmul(col[d as usize], decay);
        let dec = act.round(&mut k, dec);
        let kd = k.lds_f32_again(ks, (4 * d) as i32);
        let sv = k.ffma(kd, delta, dec);
        match act {
            super::types::Act::Bf16 => {
                let hh = k.f_to_h(sv);
                k.st_b16(sa, row * d as i32, hh);
            }
            super::types::Act::F32 => k.st_f32(sa, row * d as i32, sv),
        }
        let qd = k.lds_f32(qs, (4 * d) as i32);
        k.ffma_to(o, sv, qd, o);
    }
    let oi = k.mad(h, dv, j);
    act.st(&mut k, out, oi, o);
    Emitted::of(k, 0)
}

/// Gated RMSNorm per value head: `y = bf16(((x·inv)·w)·silu(z))`, `inv =
/// 1/√(Σx²/dv + ε)`, `w` plain (not `1 + w`). Params `out, proj (z), w, y`;
/// blocks `v_heads` (`rows`: `(v_heads, m)`).
/// `h16`: the output in IEEE f16 (`gemm.f16`).
pub fn dn_gated_norm(s: DeltaShape, rows: bool, h16: bool) -> Emitted {
    let (dv, act) = (s.v_dim, s.act);
    let mut k = Kernel::new(format!("dn_gated_norm_v{}d{dv}_{}{}{}", s.v_heads, act.tag(), rows_tag(rows), if h16 { "_h16" } else { "" }), dv);
    let out = k.param_ptr();
    let proj = k.param_ptr();
    let w = k.param_ptr();
    let y = k.param_ptr();
    let r = row_of(&mut k, rows);
    let out = at_row(&mut k, out, r, s.value() * act.bytes());
    let proj = at_row(&mut k, proj, r, s.proj_width() * act.bytes());
    let y = at_row(&mut k, y, r, s.value() * 2);
    let scratch = k.shared(32 * 4);
    let h = k.ctaid_x();
    let j = k.tid();
    let at = k.mad(h, dv, j);
    let x = act.ld(&mut k, out, at);
    let x2 = k.fmul(x, x);
    let ss = k.block_sum(x2, scratch);
    let m = k.fmul(ss, 1.0 / dv as f32);
    let m = k.fadd(m, s.eps);
    let inv = k.finv_sqrt(m);
    let wa = k.index(w, j, 2);
    let wh = k.ld_b16(Ld::Nc, wa, 0);
    let wf = k.h_to_f(wh);
    let zi = k.add(at, s.z_at());
    let z = act.ld(&mut k, proj, zi);
    let sz = k.fsigmoid(z);
    let silu = k.fmul(z, sz);
    let n = k.fmul(x, inv);
    let n = k.fmul(n, wf);
    let o = k.fmul(n, silu);
    let b = if h16 { k.f_to_f16(o) } else { k.f_to_h(o) };
    let ya = k.index(y, at, 2);
    k.st_b16(ya, 0, b);
    Emitted::of(k, 0)
}

// ── head ─────────────────────────────────────────────────────────────────────

/// Elements of the logits each argmax block scans.
pub const ARGMAX_CHUNK: u32 = 4096;

pub fn argmax_groups(n: u32) -> u32 {
    n.div_ceil(ARGMAX_CHUNK)
}

/// The best (value, first index) of the block, in thread 0: per thread the
/// strict maximum of its elements (the first on a tie, the elements in
/// increasing order), per warp by shuffles (on a tie the smaller index), then
/// thread 0 over the warps in order.
fn argmax_block(k: &mut Kernel, best: F, best_i: R, shared: R) -> (F, R) {
    let warps = k.threads() / 32;
    let mut v = best;
    let mut i = best_i;
    for m in [16, 8, 4, 2, 1] {
        let ov = k.shfl_xor(v, m);
        let oi = k.shfl_xor_u(i, m);
        let gt = k.fsetp(Cmp::Gt, ov, v);
        let eq = k.fsetp(Cmp::Eq, ov, v);
        let lower = k.setp(Cmp::Lt, oi, i);
        let tie = k.pand(eq, lower);
        let nv = k.fselp(ov, v, gt);
        let ni = k.selp(oi, i, gt);
        let ni = k.selp(oi, ni, tie);
        v = nv;
        i = ni;
    }
    let tid = k.tid();
    let lane = k.and(tid, 31u32);
    let warp = k.shr(tid, 5u32);
    let first = k.setp(Cmp::Eq, lane, 0u32);
    let at = k.mad(warp, 8u32, shared);
    k.when(first);
    k.sts_f32(at, 0, v);
    k.when(first);
    k.sts_u32(at, 4, i);
    k.bar();
    let bv = k.lds_f32(shared, 0);
    let bi = k.lds_u32(shared, 4);
    for w in 1..warps {
        let ov = k.lds_f32(shared, (8 * w) as i32);
        let oi = k.lds_u32(shared, (8 * w + 4) as i32);
        let gt = k.fsetp(Cmp::Gt, ov, bv);
        let eq = k.fsetp(Cmp::Eq, ov, bv);
        let lower = k.setp(Cmp::Lt, oi, bi);
        let tie = k.pand(eq, lower);
        let take = k.p();
        k.ins(format!("or.pred {take}, {gt}, {tie}"));
        k.when(take);
        k.fmov_to(bv, ov);
        k.when(take);
        k.mov_to(bi, oi);
    }
    (bv, bi)
}

/// First pass: each block finds the largest logit of its chunk and its first
/// index. Params `logits (bf16), partial (u32 [2·blocks])`; blocks
/// [`argmax_groups`].
pub fn argmax_partial(n: u32) -> Emitted {
    let threads = 256;
    let mut k = Kernel::new(format!("argmax_partial_{n}"), threads);
    let logits = k.param_ptr();
    let partial = k.param_ptr();
    let shared = k.shared(8 * 8 * 4);
    let g = k.ctaid_x();
    let tid = k.tid();
    let first_i = k.mul(g, ARGMAX_CHUNK);
    let start = k.add(first_i, tid);
    let end = k.add(first_i, ARGMAX_CHUNK);
    let end = k.min(end, n);
    let best = k.fmov(f32::NEG_INFINITY);
    let best_i = k.mov(u32::MAX);
    k.for_(start, end, threads, |k, i| {
        let a = k.index(logits, i, 2);
        let h = k.ld_b16(Ld::Ca, a, 0);
        let v = k.h_to_f(h);
        let better = k.fsetp(Cmp::Gt, v, best);
        k.when(better);
        k.fmov_to(best, v);
        k.when(better);
        k.mov_to(best_i, i);
    });
    let (bv, bi) = argmax_block(&mut k, best, best_i, shared);
    let first = k.setp(Cmp::Eq, tid, 0u32);
    k.if_(first, |k| {
        let pa = k.index(partial, g, 8);
        let bits = k.fbits(bv);
        k.st_u32(pa, 0, bits);
        k.st_u32(pa, 4, bi);
    });
    Emitted::of(k, 0)
}

/// Second pass, one block: the best of the partials (first index on a tie),
/// into `state[ARGMAX]`. Params `partial, state`.
pub fn argmax_final(groups: u32) -> Emitted {
    let threads = 256;
    let mut k = Kernel::new(format!("argmax_final_{groups}"), threads);
    let partial = k.param_ptr();
    let state = k.param_ptr();
    let shared = k.shared(8 * 8 * 4);
    let tid = k.tid();
    let best = k.fmov(f32::NEG_INFINITY);
    let best_i = k.mov(u32::MAX);
    k.for_(tid, groups, threads, |k, g| {
        let pa = k.index(partial, g, 8);
        let vb = k.ld_u32(Ld::Ca, pa, 0);
        let v = k.bits_f(vb);
        let i = k.ld_u32(Ld::Ca, pa, 4);
        let better = k.fsetp(Cmp::Gt, v, best);
        k.when(better);
        k.fmov_to(best, v);
        k.when(better);
        k.mov_to(best_i, i);
    });
    let (_, bi) = argmax_block(&mut k, best, best_i, shared);
    let first = k.setp(Cmp::Eq, tid, 0u32);
    k.when(first);
    k.st_u32(state, (4 * STATE_ARGMAX) as i32, bi);
    Emitted::of(k, 0)
}
