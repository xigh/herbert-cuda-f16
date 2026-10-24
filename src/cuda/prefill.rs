//! The emitted kernels of a prefill block that are not a row-wise copy of a
//! decode kernel (those are the decode emitters with `rows`): the DeltaNet's
//! conv, conv state and recurrence over the block's rows, and the causal
//! attention of the block on the tensor cores.
//!
//! Same numerical contract as the decode: what the decode would compute for
//! each row in turn, f32 inside, one rounding to nearest even at each output;
//! the DeltaNet's states are rounded to their element type after every row,
//! as the decode stores them after every token. A block has `m` rows (a
//! kernel parameter), its first row at position `state[POS]`.

use super::decode::pos;
use super::types::{Act, AttnShape, DeltaShape};
use crate::ptx::{Cmp, Emitted, Kernel, Ld, D, F};

/// The element of `act` at address `a`.
fn ld_at(k: &mut Kernel, act: Act, a: D) -> F {
    match act {
        Act::Bf16 => {
            let h = k.ld_b16(Ld::Ca, a, 0);
            k.h_to_f(h)
        }
        Act::F32 => k.ld_f32(Ld::Ca, a, 0),
    }
}

fn selp64(k: &mut Kernel, a: D, b: D, p: crate::ptx::P) -> D {
    let r = k.d();
    k.ins(format!("selp.b64 {r}, {a}, {b}, {p}"));
    r
}

// ── GatedDeltaNet ────────────────────────────────────────────────────────────

/// The causal conv + SiLU, the L2 norms and the gates of every row: blocks
/// `(2·k_heads + v_heads + 1, m)`, as [`super::decode::dn_conv_l2_gates`] for
/// row `ctaid.y`, its inputs the rows before it in `proj` and, before the
/// block, the conv state (left untouched: [`dn_conv_state`] advances it).
/// Writes the conv outputs `[q | k | v]` (q and k normalized) to `conv_out`
/// `[m][conv_dim]` and the gates to `gates` `[m][2·v_heads]`.
/// Params `proj, conv_state, conv_w, a_log, dt_bias, conv_out, gates`.
pub fn dn_conv_l2_gates_rows(s: DeltaShape) -> Emitted {
    let (dk, kk, act) = (s.k_dim, s.conv, s.act);
    assert_eq!(dk, s.v_dim, "one thread per channel of a head");
    let mut k = Kernel::new(format!("dn_conv_l2_gates_rows_k{}v{}_{}", s.k_heads, s.v_heads, act.tag()), dk);
    let proj = k.param_ptr();
    let st = k.param_ptr();
    let w = k.param_ptr();
    let a_log = k.param_ptr();
    let dt_bias = k.param_ptr();
    let conv_out = k.param_ptr();
    let gates = k.param_ptr();
    let scratch = k.shared(32 * 4);
    let g = k.ctaid_x();
    let r = k.ctaid_y();
    let lid = k.tid();
    let ab = act.bytes();
    let is_conv = k.setp(Cmp::Lt, g, 2 * s.k_heads + s.v_heads);
    let (gate_path, end) = (k.label(), k.label());
    k.bra_unless(is_conv, gate_path);
    {
        let c = k.mad(g, dk, lid);
        let out_at = k.mad(r, s.conv_dim(), c);
        let mut acc = k.fmov(0.0);
        for j in 0..kk {
            // Input r − d: the row of the block, or the state's slot kk − d + r.
            let d = kk - 1 - j;
            let v = if d == 0 {
                let i = k.mad(r, s.proj_width(), c);
                act.ld(&mut k, proj, i)
            } else {
                let inside = k.setp(Cmp::Ge, r, d);
                let ri = k.sub(r, d);
                let ri = k.selp(ri, 0u32, inside);
                let pi = k.mad(ri, s.proj_width(), c);
                let pa = k.index(proj, pi, ab);
                let slot = k.add(r, kk - d);
                let slot = k.min(slot, kk - 1);
                let si = k.mad(c, kk, slot);
                let sa = k.index(st, si, ab);
                let a = selp64(&mut k, pa, sa, inside);
                ld_at(&mut k, act, a)
            };
            let wi = k.mad(c, kk, j);
            let wa = k.index(w, wi, 2);
            let wh = k.ld_b16(Ld::Nc, wa, 0);
            let wf = k.h_to_f(wh);
            acc = k.ffma(v, wf, acc);
        }
        let sg = k.fsigmoid(acc);
        let y = k.fmul(acc, sg);
        let is_qk = k.setp(Cmp::Lt, g, 2 * s.k_heads);
        let v_head = k.label();
        k.bra_unless(is_qk, v_head);
        let y2 = k.fmul(y, y);
        let ss = k.block_sum(y2, scratch);
        let m = k.fadd(ss, 1e-6);
        let inv = k.finv_sqrt(m);
        let o = k.fmul(y, inv);
        act.st(&mut k, conv_out, out_at, o);
        k.bra(end);
        k.place(v_head);
        act.st(&mut k, conv_out, out_at, y);
        k.bra(end);
    }
    k.place(gate_path);
    let pr = k.mul(r, s.proj_width());
    let gr = k.mul(r, 2 * s.v_heads);
    k.for_(lid, s.v_heads, dk, |k, i| {
        let ai = k.add(pr, i);
        let ai = k.add(ai, s.a_at());
        let bi = k.add(pr, i);
        let bi = k.add(bi, s.b_at());
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
        let decay = match act {
            Act::Bf16 => neg,
            Act::F32 => k.fexp(neg),
        };
        let gi = k.add(gr, i);
        act.st(k, gates, gi, beta);
        let di = k.add(gi, s.v_heads);
        act.st(k, gates, di, decay);
    });
    k.place(end);
    Emitted::of(k, 0)
}

/// The conv state after the block: the last `conv` inputs of each channel,
/// from the block's rows or, for a block shorter than that, the old state.
/// Params `proj, conv_state, m`; blocks `conv_dim / k_dim`.
pub fn dn_conv_state(s: DeltaShape) -> Emitted {
    let (dk, kk, act) = (s.k_dim, s.conv, s.act);
    let mut k = Kernel::new(format!("dn_conv_state_c{}_{}", s.conv_dim(), act.tag()), dk);
    let proj = k.param_ptr();
    let st = k.param_ptr();
    let m = k.param_u32();
    let g = k.ctaid_x();
    let lid = k.tid();
    let ab = act.bytes();
    let c = k.mad(g, dk, lid);
    let vals: Vec<F> = (0..kk)
        .map(|j| {
            // Input m − kk + j: a row, or the old slot m + j.
            let back = kk - j;
            let inside = k.setp(Cmp::Ge, m, back);
            let ri = k.sub(m, back);
            let ri = k.selp(ri, 0u32, inside);
            let pi = k.mad(ri, s.proj_width(), c);
            let pa = k.index(proj, pi, ab);
            let slot = k.add(m, j);
            let slot = k.min(slot, kk - 1);
            let si = k.mad(c, kk, slot);
            let sa = k.index(st, si, ab);
            let a = selp64(&mut k, pa, sa, inside);
            ld_at(&mut k, act, a)
        })
        .collect();
    for (j, v) in vals.into_iter().enumerate() {
        let si = k.mad(c, kk, j as u32);
        act.st(&mut k, st, si, v);
    }
    Emitted::of(k, 0)
}

/// The gated delta rule over the block's rows in order: one block per value
/// head, one thread per value column holding its column of the state in
/// registers for the whole block; per row the q and k of the head go through
/// shared memory (two buffers, one barrier per row). Per row, as the decode:
/// `s *= decay; kv = Σ s·k; δ = (v − kv)·β; s += k⊗δ; o = Σ s·(q·scale)`,
/// the state rounded to its element type after the row.
/// Params `conv_out (q, k, v), gates, state, out, m`; blocks `v_heads`.
pub fn dn_recurrence_rows(s: DeltaShape) -> Emitted {
    let (dk, dv, act) = (s.k_dim, s.v_dim, s.act);
    let mut k = Kernel::new(format!("dn_recurrence_rows_k{}v{}_{}", s.k_heads, s.v_heads, act.tag()), dv);
    let qkv = k.param_ptr();
    let gates = k.param_ptr();
    let state = k.param_ptr();
    let out = k.param_ptr();
    let m = k.param_u32();
    let qs = k.shared(2 * dk * 4);
    let ks = k.shared(2 * dk * 4);
    let h = k.ctaid_x();
    let j = k.tid();
    let kh = k.div(h, s.v_heads / s.k_heads);
    let scale = 1.0 / (dk as f32).sqrt();
    let ab = act.bytes();
    let sbase = k.mad(h, dk * dv, j);
    let sa = k.index(state, sbase, ab);
    let row_b = (dv * ab) as i32;
    let col: Vec<F> = (0..dk)
        .map(|d| match act {
            Act::Bf16 => {
                let hh = k.ld_b16(Ld::Ca, sa, row_b * d as i32);
                k.h_to_f(hh)
            }
            Act::F32 => k.ld_f32(Ld::Ca, sa, row_b * d as i32),
        })
        .collect();
    let qk0 = k.mul(kh, dk);
    k.for_(0u32, m, 1, |k, r| {
        let buf = k.and(r, 1u32);
        let buf = k.mul(buf, dk * 4);
        let qb = k.add(qs, buf);
        let kb = k.add(ks, buf);
        let rb = k.mul(r, s.conv_dim());
        let rq = k.add(rb, qk0);
        k.for_(j, dk, dv, |k, i| {
            let qi = k.add(rq, i);
            let q = act.ld(k, qkv, qi);
            let q = k.fmul(q, scale);
            let a = k.mad(i, 4u32, qb);
            k.sts_f32(a, 0, q);
            let ki = k.add(qi, s.key());
            let kv = act.ld(k, qkv, ki);
            let a = k.mad(i, 4u32, kb);
            k.sts_f32(a, 0, kv);
        });
        k.bar();
        let gr = k.mad(r, 2 * s.v_heads, h);
        let beta = act.ld(k, gates, gr);
        let di = k.add(gr, s.v_heads);
        let decay = act.ld(k, gates, di);
        let decay = match act {
            Act::Bf16 => k.fexp(decay),
            Act::F32 => decay,
        };
        let vi = k.mad(h, dv, 2 * s.key());
        let vi = k.add(vi, rb);
        let vi = k.add(vi, j);
        let v = act.ld(k, qkv, vi);
        let kv = k.fmov(0.0);
        for d in 0..dk {
            let sv = k.fmul(col[d as usize], decay);
            let kd = k.lds_f32(kb, (4 * d) as i32);
            k.ffma_to(kv, sv, kd, kv);
        }
        let diff = k.fsub(v, kv);
        let delta = k.fmul(diff, beta);
        let o = k.fmov(0.0);
        for d in 0..dk {
            // The decayed state again (the same product), as stored.
            let sv = k.fmul(col[d as usize], decay);
            let sv = act.round(k, sv);
            let kd = k.lds_f32_again(kb, (4 * d) as i32);
            let nv = k.ffma(kd, delta, sv);
            let qd = k.lds_f32(qb, (4 * d) as i32);
            k.ffma_to(o, nv, qd, o);
            let nr = act.round(k, nv);
            k.fmov_to(col[d as usize], nr);
        }
        let oi = k.mad(r, s.value(), j);
        let oi = k.mad(h, dv, oi);
        act.st(k, out, oi, o);
    });
    for d in 0..dk {
        match act {
            Act::Bf16 => {
                let hh = k.f_to_h(col[d as usize]);
                k.st_b16(sa, row_b * d as i32, hh);
            }
            Act::F32 => k.st_f32(sa, row_b * d as i32, col[d as usize]),
        }
    }
    Emitted::of(k, 0)
}

// ── causal attention on the tensor cores ─────────────────────────────────────

/// Query rows per warp: the M of `mma.m16n8k16`.
pub const QT: u32 = 16;

/// Dynamic shared bytes of [`sdpa_flash`]: the Q tiles of the block's warps,
/// one K and one V tile of `bk` keys, rows padded by 16 bytes; with the
/// softmax in shared memory, per warp S (f32), P (bf16) and two row vectors.
pub fn flash_shared_bytes(s: AttnShape, bk: u32, smem_softmax: bool) -> u32 {
    let g = s.heads / s.kv_heads;
    let stride = (s.head_dim + 8) * 2;
    (g * QT + 2 * bk) * stride + if smem_softmax { g * softmax_warp_bytes(bk) } else { 0 }
}

/// Per warp, the softmax in shared memory: S `QT × bk` f32, P `QT × bk` bf16
/// (rows padded by 16 bytes for `ldmatrix`), the correction and the sum of
/// each row (f32).
fn softmax_warp_bytes(bk: u32) -> u32 {
    QT * bk * 4 + QT * (bk + 8) * 2 + 2 * QT * 4
}

/// Blocks of [`sdpa_flash`] for `m` rows.
pub fn flash_grid(s: AttnShape, m: u32) -> (u32, u32, u32) {
    (m.div_ceil(QT), s.kv_heads, 1)
}

/// The causal attention of the block, FlashAttention's online softmax with
/// `S = Q·Kᵀ` and `O += P·V` on `mma.m16n8k16` (bf16 operands, f32
/// accumulators). One block per (tile of 16 query rows, KV head), one warp per
/// query head of the KV head's group: the warps share the K and V tiles of
/// `bk` keys staged by `cp.async`. The fragment layout being known, the
/// softmax stays in registers: lane t holds rows t/4 and t/4 + 8 of S, its row
/// max and sum reduced over the quad, the correction applied to its own
/// accumulators, P packed to bf16 straight into the A fragments of `P·V`.
/// The scale `1/√hd` multiplies S in f32; keys past a row's position are
/// masked. Output as the decode: `bf16(o / l)` rounded, then the gate
/// `bf16(o·σ(gate))`. Params `state, qkv, k_cache, v_cache, out, m`; blocks
/// [`flash_grid`]; dynamic shared [`flash_shared_bytes`].
///
/// `smem_softmax` (`sdpa.softmax=shared`): the softmax as Vulkan must do it,
/// the fragment layout of its matrices being opaque (`vk/fatc.rs`): S stored
/// to shared memory, two lanes per row for the max, the exponentials and the
/// sum, P written back as bf16 and loaded as A fragments, each row's
/// correction read back from shared memory by the lanes holding that row of
/// O. The sums run in another order than in registers.
pub fn sdpa_flash(s: AttnShape, bk: u32, smem_softmax: bool, h16: bool) -> Emitted {
    let hd = s.head_dim;
    let g = s.heads / s.kv_heads;
    assert!(hd.is_multiple_of(16) && bk.is_multiple_of(16));
    let threads = g * 32;
    let stride = (hd + 8) * 2;
    let chunks = hd / 8; // 16-byte chunks per row
    assert!((bk * chunks).is_multiple_of(threads) && (QT * chunks).is_multiple_of(32));
    let mut k = Kernel::new(format!("sdpa_flash_h{}kv{}d{hd}_k{bk}{}{}", s.heads, s.kv_heads, if smem_softmax { "_smem" } else { "" }, if h16 { "_h16" } else { "" }), threads);
    let state = k.param_ptr();
    let qkv = k.param_ptr();
    let kc = k.param_ptr();
    let vc = k.param_ptr();
    let out = k.param_ptr();
    let m = k.param_u32();
    let smem = k.dyn_shared();
    let tid = k.tid();
    let lane = k.and(tid, 31u32);
    let warp = k.shr(tid, 5u32);
    let qt = k.ctaid_x();
    let kvh = k.ctaid_y();
    let p0 = pos(&mut k, state);
    let r0 = k.mul(qt, QT);
    let h = k.mad(kvh, g, warp);
    let q_smem = k.mad(warp, QT * stride, smem);
    let k_smem = k.add(smem, g * QT * stride);
    let v_smem = k.add(k_smem, bk * stride);
    let m_last = k.sub(m, 1u32);
    // The keys the tile's rows see.
    let rows_end = k.add(r0, QT);
    let rows_end = k.min(rows_end, m);
    let kv_end = k.add(p0, rows_end);

    // ── Q of the warp's head, 16 rows, zeros past m ──
    for x in 0..QT * chunks / 32 {
        let e = k.add(lane, x * 32);
        let i = k.div(e, chunks);
        let c = k.rem(e, chunks);
        let row = k.add(r0, i);
        let inside = k.setp(Cmp::Lt, row, m);
        let row = k.min(row, m_last);
        let qi = k.mad(row, s.qkv_width(), 0u32);
        let qi = k.mad(h, 2 * hd, qi);
        let qa = k.index(qkv, qi, 2);
        let c16 = k.mul(c, 16u32);
        let src = k.index(qa, c16, 1);
        let bytes = k.selp(16u32, 0u32, inside);
        let dst = k.mad(i, stride, c16);
        let dst = k.add(dst, q_smem);
        k.cp_async(dst, 0, src, Some(bytes));
    }
    k.cp_commit();

    // ── per-lane fragment addresses ──
    let l15 = k.and(lane, 15u32);
    let l16 = k.shr(lane, 4u32);
    let a_off = k.mul(l15, stride);
    let a_k = k.mul(l16, 16u32);
    let a_off = k.add(a_off, a_k);
    let a_frag = k.add(q_smem, a_off);
    // K (as B of S, not transposed): key (l&7) + 8·(l>>4), dims +8·((l>>3)&1).
    let l7 = k.and(lane, 7u32);
    let l3 = k.shr(lane, 3u32);
    let l3 = k.and(l3, 1u32);
    let kr = k.shl(l16, 3u32);
    let kr = k.add(kr, l7);
    let kd = k.mul(l3, 16u32);
    let k_off = k.mad(kr, stride, kd);
    let k_frag = k.add(k_smem, k_off);
    // V (as B of P·V, transposed): key (l&7) + 8·((l>>3)&1), dims +8·(l>>4).
    let vr = k.shl(l3, 3u32);
    let vr = k.add(vr, l7);
    let vd = k.mul(l16, 16u32);
    let v_off = k.mad(vr, stride, vd);
    let v_frag = k.add(v_smem, v_off);
    // The softmax in shared memory: the warp's S, P, correction and sum rows;
    // lane t owns row t/2, keys (t%2)·bk/2 … of every tile of bk keys.
    let (s_stride, p_stride) = (bk * 4, (bk + 8) * 2);
    let sm = smem_softmax.then(|| {
        let base = k.add(v_smem, bk * stride);
        let w = k.mad(warp, softmax_warp_bytes(bk), base);
        let p_base = k.add(w, QT * s_stride);
        let c_base = k.add(p_base, QT * p_stride);
        let l_base = k.add(c_base, QT * 4);
        let rr = k.shr(lane, 1u32);
        let hh = k.and(lane, 1u32);
        let col0 = k.mul(hh, bk / 2);
        let s_row = k.mad(rr, s_stride, w);
        let s_own = k.mad(col0, 4u32, s_row);
        let p_row = k.mad(rr, p_stride, p_base);
        let p_own = k.mad(col0, 2u32, p_row);
        let c_own = k.mad(rr, 4u32, c_base);
        let l_own = k.mad(rr, 4u32, l_base);
        let row = k.add(r0, rr);
        let qpos = k.add(p0, row);
        let inr = k.setp(Cmp::Lt, row, m);
        let first = k.setp(Cmp::Eq, hh, 0u32);
        let m_row = k.fmov(f32::MIN);
        let l_row = k.fmov(0.0);
        // Fragment addresses: S as stored by the accumulators, P as A of P·V.
        let gq = k.shr(lane, 2u32);
        let q8 = k.and(lane, 3u32);
        let q8 = k.shl(q8, 3u32);
        let s_frag = k.mad(gq, s_stride, w);
        let s_frag = k.add(s_frag, q8);
        let p_frag = k.mad(l15, p_stride, p_base);
        let p16 = k.mul(l16, 16u32);
        let p_frag = k.add(p_frag, p16);
        let c_frag = k.mad(gq, 4u32, c_base);
        let l_frag = k.mad(gq, 4u32, l_base);
        (s_own, p_own, c_own, l_own, col0, qpos, inr, first, m_row, l_row, s_frag, p_frag, c_frag, l_frag)
    });

    let gq = k.shr(lane, 2u32);
    let q4 = k.and(lane, 3u32);
    let q2 = k.shl(q4, 1u32);
    // Positions of the lane's two rows; rows past m are masked entirely.
    let row_g = k.add(r0, gq);
    let row_g8 = k.add(row_g, 8u32);
    let qpos_g = k.add(p0, row_g);
    let qpos_g8 = k.add(p0, row_g8);
    let in_g = k.setp(Cmp::Lt, row_g, m);
    let in_g8 = k.setp(Cmp::Lt, row_g8, m);

    let nt = hd / 8;
    let o: Vec<[F; 4]> = (0..nt).map(|_| [k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)]).collect();
    let m_run = [k.fmov(f32::MIN), k.fmov(f32::MIN)];
    let l_run = [k.fmov(0.0), k.fmov(0.0)];
    let scale = 1.0 / (hd as f32).sqrt();
    let kv_rows = k.mul(kvh, s.capacity);
    let kv_last = k.sub(kv_end, 1u32);

    let k0 = k.mov(0u32);
    let (top, done) = (k.label(), k.label());
    k.place(top);
    {
        // a. K and V of keys k0 … k0 + bk − 1 (zeros past kv_end).
        k.bar();
        for x in 0..bk * chunks / threads {
            let e = k.add(tid, x * threads);
            let j = k.div(e, chunks);
            let c = k.rem(e, chunks);
            let kp = k.add(k0, j);
            let inside = k.setp(Cmp::Lt, kp, kv_end);
            let kp = k.min(kp, kv_last);
            let r = k.add(kv_rows, kp);
            let c16 = k.mul(c, 16u32);
            let off = k.mad(r, hd * 2, c16);
            let ksrc = k.index(kc, off, 1);
            let vsrc = k.index(vc, off, 1);
            let bytes = k.selp(16u32, 0u32, inside);
            let dst = k.mad(j, stride, c16);
            let kd = k.add(dst, k_smem);
            let vd = k.add(dst, v_smem);
            k.cp_async(kd, 0, ksrc, Some(bytes));
            k.cp_async(vd, 0, vsrc, Some(bytes));
        }
        k.cp_commit();
        k.cp_wait(0);
        k.bar();
        // b. S = Q·Kᵀ, 16 × bk.
        let st = bk / 8;
        let sacc: Vec<[F; 4]> = (0..st).map(|_| [k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)]).collect();
        for kk in 0..hd / 16 {
            let a = k.ldmatrix_x4(a_frag, (kk * 32) as i32, false);
            for jp in 0..st / 2 {
                let b = k.ldmatrix_x4(k_frag, (jp * 16 * stride + kk * 32) as i32, false);
                k.mma_h(&sacc[2 * jp as usize], &a, &[b[0], b[1]]);
                k.mma_h(&sacc[2 * jp as usize + 1], &a, &[b[2], b[3]]);
            }
        }
        if let Some((s_own, p_own, c_own, _, col0, qpos, inr, first, m_row, l_row, s_frag, p_frag, c_frag, _)) = sm {
            // c'. S to shared memory, the softmax of the lane's row half.
            for (j, sa) in sacc.iter().enumerate() {
                for e in 0..4usize {
                    let off = ((e / 2) as u32 * 8 * s_stride + (j as u32 * 8 + (e % 2) as u32) * 4) as i32;
                    k.sts_f32(s_frag, off, sa[e]);
                }
            }
            k.bar_warp();
            let kc0 = k.add(k0, col0);
            let mut vals = Vec::new();
            let mut mx = k.fmov(f32::MIN);
            for c in 0..bk / 2 {
                let kp = k.add(kc0, c);
                let causal = k.setp(Cmp::Le, kp, qpos);
                let valid = k.pand(causal, inr);
                let v = k.lds_f32(s_own, (c * 4) as i32);
                let v = k.fmul(v, scale);
                let v = k.fselp(v, f32::MIN, valid);
                mx = k.fmax(mx, v);
                vals.push(v);
            }
            let o1 = k.shfl_xor(mx, 1);
            mx = k.fmax(mx, o1);
            let m_new = k.fmax(m_row, mx);
            let dm = k.fsub(m_row, m_new);
            let corr = k.fexp(dm);
            k.fmov_to(m_row, m_new);
            let mut sum = k.fmov(0.0);
            for c in (0..bk / 2).step_by(2) {
                let d0 = k.fsub(vals[c as usize], m_new);
                let e0 = k.fexp(d0);
                let d1 = k.fsub(vals[c as usize + 1], m_new);
                let e1 = k.fexp(d1);
                sum = k.fadd(sum, e0);
                sum = k.fadd(sum, e1);
                let pk = k.pack_h(e0, e1);
                k.sts_u32(p_own, (c * 2) as i32, pk);
            }
            let o1 = k.shfl_xor(sum, 1);
            sum = k.fadd(sum, o1);
            let l = k.ffma(l_row, corr, sum);
            k.fmov_to(l_row, l);
            k.when(first);
            k.sts_f32(c_own, 0, corr);
            k.bar_warp();
            // d'. O = diag(corr)·O + P·V, corr and P read back.
            let corr = [k.lds_f32(c_frag, 0), k.lds_f32(c_frag, (8 * 4) as i32)];
            for on in &o {
                for e in 0..4usize {
                    let v = k.fmul(on[e], corr[e / 2]);
                    k.fmov_to(on[e], v);
                }
            }
            for kk in 0..bk / 16 {
                let a = k.ldmatrix_x4(p_frag, (kk * 32) as i32, false);
                for np in 0..nt / 2 {
                    let b = k.ldmatrix_x4(v_frag, (kk * 16 * stride + np * 32) as i32, true);
                    k.mma_h(&o[2 * np as usize], &a, &[b[0], b[1]]);
                    k.mma_h(&o[2 * np as usize + 1], &a, &[b[2], b[3]]);
                }
            }
        } else {
        // c. The online softmax of the lane's two rows.
        let kq = k.add(k0, q2);
        let mut vals: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        let mut mx = [k.fmov(f32::MIN), k.fmov(f32::MIN)];
        for (j, sa) in sacc.iter().enumerate() {
            for e in 0..4usize {
                let half = e / 2;
                let kp = k.add(kq, (j * 8 + e % 2) as u32);
                let (qp, inr) = if half == 0 { (qpos_g, in_g) } else { (qpos_g8, in_g8) };
                let causal = k.setp(Cmp::Le, kp, qp);
                let valid = k.pand(causal, inr);
                let v = k.fmul(sa[e], scale);
                let v = k.fselp(v, f32::MIN, valid);
                mx[half] = k.fmax(mx[half], v);
                vals[half].push(v);
            }
        }
        let mut corr = [mx[0], mx[1]];
        let mut m_new = [mx[0], mx[1]];
        for hf in 0..2 {
            for sh in [1, 2] {
                let o = k.shfl_xor(mx[hf], sh);
                mx[hf] = k.fmax(mx[hf], o);
            }
            m_new[hf] = k.fmax(m_run[hf], mx[hf]);
            let dm = k.fsub(m_run[hf], m_new[hf]);
            corr[hf] = k.fexp(dm);
            k.fmov_to(m_run[hf], m_new[hf]);
        }
        let mut sums = [k.fmov(0.0), k.fmov(0.0)];
        let mut p: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        for hf in 0..2 {
            for &v in &vals[hf] {
                let d = k.fsub(v, m_new[hf]);
                let e = k.fexp(d);
                sums[hf] = k.fadd(sums[hf], e);
                p[hf].push(e);
            }
            for sh in [1, 2] {
                let o = k.shfl_xor(sums[hf], sh);
                sums[hf] = k.fadd(sums[hf], o);
            }
            let l = k.ffma(l_run[hf], corr[hf], sums[hf]);
            k.fmov_to(l_run[hf], l);
        }
        // d. O = diag(corr)·O + P·V.
        for on in &o {
            for e in 0..4usize {
                let c = corr[e / 2];
                let v = k.fmul(on[e], c);
                k.fmov_to(on[e], v);
            }
        }
        for kk in 0..bk / 16 {
            // A fragment of P for keys 16kk … 16kk + 15: S tiles 2kk and 2kk + 1.
            let (j0, j1) = (2 * kk as usize, 2 * kk as usize + 1);
            let a = [
                k.pack_h(p[0][2 * j0], p[0][2 * j0 + 1]),
                k.pack_h(p[1][2 * j0], p[1][2 * j0 + 1]),
                k.pack_h(p[0][2 * j1], p[0][2 * j1 + 1]),
                k.pack_h(p[1][2 * j1], p[1][2 * j1 + 1]),
            ];
            for np in 0..nt / 2 {
                let b = k.ldmatrix_x4(v_frag, (kk * 16 * stride + np * 32) as i32, true);
                k.mma_h(&o[2 * np as usize], &a, &[b[0], b[1]]);
                k.mma_h(&o[2 * np as usize + 1], &a, &[b[2], b[3]]);
            }
        }
        }
        k.add_to(k0, k0, bk);
        let more = k.setp(Cmp::Lt, k0, kv_end);
        k.bra_if(more, top);
    }
    k.place(done);

    // ── out: bf16(o / l), then the gate ──
    let l_out = match sm {
        Some((_, _, _, l_own, _, _, _, first, _, l_row, _, _, _, l_frag)) => {
            k.when(first);
            k.sts_f32(l_own, 0, l_row);
            k.bar_warp();
            [k.lds_f32(l_frag, 0), k.lds_f32(l_frag, (8 * 4) as i32)]
        }
        None => l_run,
    };
    for hf in 0..2usize {
        let inv = k.fdiv(1.0, l_out[hf]);
        let (row, inside) = if hf == 0 { (row_g, in_g) } else { (row_g8, in_g8) };
        let row = k.min(row, m_last);
        let gi = k.mul(row, s.qkv_width());
        let gi = k.mad(h, 2 * hd, gi);
        let gi = k.add(gi, hd);
        let gi = k.add(gi, q2);
        let ga = k.index(qkv, gi, 2);
        let oi = k.mul(row, s.out_width());
        let oi = k.mad(h, hd, oi);
        let oi = k.add(oi, q2);
        let oa = k.index(out, oi, 2);
        for (n, on) in o.iter().enumerate() {
            let off = (n * 16) as i32;
            let gp = k.ld_u32(Ld::Ca, ga, off);
            let mut r = Vec::new();
            for (e, gf) in [(0, k.h_lo(gp)), (1, k.h_hi(gp))] {
                let v = k.fmul(on[2 * hf + e], inv);
                let v = k.round_h(v);
                let sg = k.fsigmoid(gf);
                r.push(k.fmul(v, sg));
            }
            let pk = k.pack16(r[0], r[1], h16);
            k.when(inside);
            k.st_u32(oa, off, pk);
        }
    }
    Emitted::of(k, flash_shared_bytes(s, bk, smem_softmax))
}
