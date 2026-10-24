//! The gated delta rule of a prefill block on the tensor cores (`rec.tc=1`,
//! bf16 activations): the chunked form of FLA (vLLM's kernels), as
//! herbert-vk-bf16's `vk/dntc.rs`, chunks of `C` rows, bf16 operands and f32
//! accumulators.
//!
//! With `G_t = Σ_{i≤t} g_i` (g = log decay, cumulated from the chunk's first
//! row), `Γ_t = exp(G_t)` and the state `S0` (`[dk][dv]`) at the chunk's start:
//! `A_ts = β_t·exp(G_t−G_s)·(k_t·k_s)` for `s < t`, `T = (I + A)⁻¹`;
//! `Δ = T·diag(β)·V − T·diag(β·Γ)·K·S0`;
//! `O = scale·diag(Γ)·Q·S0 + (scale·exp(G_t−G_s)·(q_t·k_s))_{s≤t}·Δ`;
//! `S = Γ_C·S0 + (diag(Γ_C/Γ)·K)ᵀ·Δ` — the row-by-row recurrence summed by
//! chunks. Within a block the state stays in f32 accumulators (its bf16
//! rounding is the operand of the products); it is stored in bf16 at the end.
//!
//! [`dn_tc_prep`] computes per (chunk, value head), in f32 on the SIMT cores
//! (a few hundred thousand products per block), what depends on q, k, β and
//! g only, into a bf16 scratch: `Tβ = T·diag(β)`, `P` (the masked, decayed,
//! scaled `Q·Kᵀ`), `W = bf16(T·diag(β·Γ))·K`, `scale·diag(Γ)·Q`, `KD =
//! diag(Γ_C/Γ)·K`, and `Γ_C` in f32. [`dn_tc_state`] walks the chunks of the
//! block in order per (value head, slice of [`DVS`] columns), its slice of the
//! state in `mma.m16n8k16` accumulators, every operand staged in shared
//! memory and read by `ldmatrix`.

use super::types::DeltaShape;
use crate::ptx::{Cmp, Emitted, Kernel, Ld, F, R};

/// Rows of a chunk.
pub const C: u32 = 32;
/// Columns of the state per block of [`dn_tc_state`].
pub const DVS: u32 = 32;
const THREADS: u32 = 128;

/// bf16 elements of the scratch of one (chunk, value head): `Tβ, P` (`C × C`),
/// then `W, QΓ, KD` (`C × dk`).
pub fn mat_len(s: DeltaShape) -> u32 {
    2 * C * C + 3 * C * s.k_dim
}
const OFF_TB: u32 = 0;
const OFF_P: u32 = C * C;
fn off_w(_s: DeltaShape) -> u32 {
    2 * C * C
}
fn off_qg(s: DeltaShape) -> u32 {
    2 * C * C + C * s.k_dim
}
fn off_kd(s: DeltaShape) -> u32 {
    2 * C * C + 2 * C * s.k_dim
}

pub fn chunks(m: u32) -> u32 {
    m.div_ceil(C)
}

/// Per (chunk, value head): the chunk's operators (module doc). Params
/// `conv_out (bf16 [m][conv_dim]: q, k normalized), gates (bf16 [m][β | g]),
/// mat (bf16 scratch), vec (f32 scratch, Γ_C), m`; blocks `chunks(m) · v_heads`.
pub fn dn_tc_prep(s: DeltaShape) -> Emitted {
    let (dk, nv) = (s.k_dim, s.v_heads);
    assert!(dk == THREADS, "one thread per column of K");
    let mut k = Kernel::new(format!("dn_tc_prep_k{}v{}_c{C}", s.k_heads, nv), THREADS);
    let qkv = k.param_ptr();
    let gates = k.param_ptr();
    let mat = k.param_ptr();
    let vecb = k.param_ptr();
    let m = k.param_u32();
    // K and Q in f32, rows padded to dk + 1 floats (column reads conflict-free).
    let rs = dk + 1;
    let ks = k.shared(C * rs * 4);
    let qs = k.shared(C * rs * 4);
    let kk_s = k.shared(C * C * 4);
    let qk_s = k.shared(C * C * 4);
    let t_s = k.shared(C * C * 4);
    let beta_s = k.shared(C * 4);
    let g_s = k.shared(C * 4);
    let g = k.ctaid_x();
    let ci = k.div(g, nv);
    let h = k.rem(g, nv);
    let kh = k.div(h, nv / s.k_heads);
    let r0 = k.mul(ci, C);
    let t = k.tid();
    let scale = 1.0 / (dk as f32).sqrt();
    let m_last = k.sub(m, 1u32);
    let base = k.mul(g, mat_len(s));
    let mat_g = k.index(mat, base, 2);

    // 1. K and Q rows of the chunk (zero past m), one column per thread.
    let col_q = k.mad(kh, dk, t);
    for r in 0..C {
        let grow = k.add(r0, r);
        let inside = k.setp(Cmp::Lt, grow, m);
        let rr = k.min(grow, m_last);
        let qi = k.mad(rr, s.conv_dim(), col_q);
        let qa = k.index(qkv, qi, 2);
        let qh = k.ld_b16(Ld::Ca, qa, 0);
        let kh_ = k.ld_b16(Ld::Ca, qa, (s.key() * 2) as i32);
        let qf = k.h_to_f(qh);
        let kf = k.h_to_f(kh_);
        let qf = k.fselp(qf, 0.0, inside);
        let kf = k.fselp(kf, 0.0, inside);
        let sa = k.mad(t, 4u32, (r * rs * 4) as u32);
        let sq = k.add(sa, qs);
        let sk = k.add(sa, ks);
        k.sts_f32(sq, 0, qf);
        k.sts_f32(sk, 0, kf);
    }
    // 2. β and g of the chunk's rows (zero past m).
    let in_c = k.setp(Cmp::Lt, t, C);
    k.if_(in_c, |k| {
        let grow = k.add(r0, t);
        let inside = k.setp(Cmp::Lt, grow, m);
        let rr = k.min(grow, m_last);
        let gi = k.mad(rr, 2 * nv, h);
        let ga = k.index(gates, gi, 2);
        let bh = k.ld_b16(Ld::Ca, ga, 0);
        let gh = k.ld_b16(Ld::Ca, ga, (nv * 2) as i32);
        let b = k.h_to_f(bh);
        let gl = k.h_to_f(gh);
        let b = k.fselp(b, 0.0, inside);
        let gl = k.fselp(gl, 0.0, inside);
        let a = k.mad(t, 4u32, beta_s);
        k.sts_f32(a, 0, b);
        let a = k.mad(t, 4u32, g_s);
        k.sts_f32(a, 0, gl);
    });
    k.bar();
    // 3. G, cumulated over the chunk (one thread, in order).
    let first = k.setp(Cmp::Eq, t, 0u32);
    k.if_(first, |k| {
        let acc = k.fmov(0.0);
        for i in 0..C {
            let x = k.lds_f32(g_s, (4 * i) as i32);
            k.fadd_to(acc, acc, x);
            k.sts_f32(g_s, (4 * i) as i32, acc);
        }
    });
    k.bar();
    // 4. K·Kᵀ and Q·Kᵀ: entries t + 128·i, (r, c) = (e / C, e % C).
    for i in 0..C * C / THREADS {
        let e = k.add(t, i * THREADS);
        let r = k.div(e, C);
        let c = k.rem(e, C);
        let ra = k.mul(r, rs * 4);
        let ca = k.mul(c, rs * 4);
        let kr = k.add(ra, ks);
        let qr = k.add(ra, qs);
        let kc = k.add(ca, ks);
        let (dkk, dqk) = (k.fmov(0.0), k.fmov(0.0));
        for d in 0..dk {
            let b = k.lds_f32(kc, (4 * d) as i32);
            let a = k.lds_f32(kr, (4 * d) as i32);
            k.ffma_to(dkk, a, b, dkk);
            let q = k.lds_f32(qr, (4 * d) as i32);
            k.ffma_to(dqk, q, b, dqk);
        }
        let ea = k.mad(e, 4u32, kk_s);
        k.sts_f32(ea, 0, dkk);
        let ea = k.mad(e, 4u32, qk_s);
        k.sts_f32(ea, 0, dqk);
    }
    k.bar();
    // 5. A (into kk_s), P (scratch), T = I to start.
    for i in 0..C * C / THREADS {
        let e = k.add(t, i * THREADS);
        let r = k.div(e, C);
        let c = k.rem(e, C);
        let gra = k.mad(r, 4u32, g_s);
        let gca = k.mad(c, 4u32, g_s);
        let gr = k.lds_f32(gra, 0);
        let gc = k.lds_f32(gca, 0);
        let dg = k.fsub(gr, gc);
        let f = k.fexp(dg);
        let below = k.setp(Cmp::Lt, c, r);
        let diag = k.setp(Cmp::Eq, c, r);
        let upto = k.setp(Cmp::Le, c, r);
        let ba = k.mad(r, 4u32, beta_s);
        let br = k.lds_f32(ba, 0);
        let ea = k.mad(e, 4u32, kk_s);
        let kkv = k.lds_f32(ea, 0);
        let a = k.fmul(br, f);
        let a = k.fmul(a, kkv);
        let a = k.fselp(a, 0.0, below);
        k.sts_f32(ea, 0, a);
        let qa = k.mad(e, 4u32, qk_s);
        let qkv_ = k.lds_f32(qa, 0);
        let p = k.fmul(f, scale);
        let p = k.fmul(p, qkv_);
        let p = k.fselp(p, 0.0, upto);
        let pb = k.f_to_h(p);
        let pa = k.index(mat_g, e, 2);
        k.st_b16(pa, (OFF_P * 2) as i32, pb);
        let tv = k.fselp(1.0, 0.0, diag);
        let ta = k.mad(e, 4u32, t_s);
        k.sts_f32(ta, 0, tv);
    }
    k.bar();
    // 6. T = (I + A)⁻¹ by forward substitution, a row at a time.
    for r in 1..C {
        let col = k.setp(Cmp::Lt, t, r);
        k.if_(col, |k| {
            let acc = k.fmov(0.0);
            for s2 in 0..r {
                let a = k.lds_f32(kk_s, (4 * (r * C + s2)) as i32);
                let ta = k.mad(t, 4u32, t_s);
                let tv = k.lds_f32(ta, (4 * s2 * C) as i32);
                k.ffma_to(acc, a, tv, acc);
            }
            let neg = k.fneg(acc);
            let ta = k.mad(t, 4u32, t_s);
            k.sts_f32(ta, (4 * r * C) as i32, neg);
        });
        k.bar();
    }
    // 7. Tβ into the scratch; bf16(TβΓ) into qk_s for W.
    for i in 0..C * C / THREADS {
        let e = k.add(t, i * THREADS);
        let c = k.rem(e, C);
        let ta = k.mad(e, 4u32, t_s);
        let tv = k.lds_f32(ta, 0);
        let ba = k.mad(c, 4u32, beta_s);
        let bc = k.lds_f32(ba, 0);
        let tbv = k.fmul(tv, bc);
        let ga = k.mad(c, 4u32, g_s);
        let gc = k.lds_f32(ga, 0);
        let gam = k.fexp(gc);
        let tbgv = k.fmul(tbv, gam);
        let x = k.f_to_h(tbv);
        let at = k.index(mat_g, e, 2);
        k.st_b16(at, (OFF_TB * 2) as i32, x);
        let y = k.round_h(tbgv);
        let qa = k.mad(e, 4u32, qk_s);
        k.sts_f32(qa, 0, y);
    }
    k.bar();
    // 8. W = TβΓ·K, 9. QΓ·scale and KD = diag(Γ_C/Γ)·K: thread t = column t.
    let glast = k.lds_f32(g_s, (4 * (C - 1)) as i32);
    let tcol = k.mul(t, 4u32);
    let kcol = k.add(tcol, ks);
    let qcol = k.add(tcol, qs);
    let kcolv: Vec<F> = (0..C).map(|s2| k.lds_f32(kcol, (4 * s2 * rs) as i32)).collect();
    for r in 0..C {
        let acc = k.fmov(0.0);
        for (s2, &kv) in kcolv.iter().enumerate() {
            let tg = k.lds_f32(qk_s, (4 * (r * C + s2 as u32)) as i32);
            k.ffma_to(acc, tg, kv, acc);
        }
        let wb = k.f_to_h(acc);
        let wi = k.add(t, r * dk);
        let wa = k.index(mat_g, wi, 2);
        k.st_b16(wa, (off_w(s) * 2) as i32, wb);
        let gr = k.lds_f32(g_s, (4 * r) as i32);
        let eg = k.fexp(gr);
        let qsc = k.fmul(eg, scale);
        let qv = k.lds_f32(qcol, (4 * r * rs) as i32);
        let x = k.fmul(qv, qsc);
        let xb = k.f_to_h(x);
        k.st_b16(wa, (off_qg(s) * 2) as i32, xb);
        let dg = k.fsub(glast, gr);
        let ksc = k.fexp(dg);
        let y = k.fmul(kcolv[r as usize], ksc);
        let yb = k.f_to_h(y);
        k.st_b16(wa, (off_kd(s) * 2) as i32, yb);
    }
    // 10. Γ_C.
    k.if_(first, |k| {
        let gamma = k.fexp(glast);
        let va = k.index(vecb, g, 4);
        k.st_f32(va, 0, gamma);
    });
    Emitted::of(k, 0)
}

/// A shared tile of bf16 rows padded by 16 bytes: `ldmatrix` reads its eight
/// row addresses from 32 distinct banks.
struct Tile {
    base: R,
    stride: u32,
}

impl Tile {
    fn at(&self, row: u32, col: u32) -> i32 {
        (row * self.stride + col * 2) as i32
    }
}

/// Lane offsets of the `ldmatrix.x4` addressing patterns, bytes for a tile of
/// `stride`: `a` for an A fragment from row-major storage (rows (l&15), cols
/// 8·(l>>4)), `bt` for two n8 B fragments from row-major K×N storage
/// (transposed: rows (l&7) + 8·((l>>3)&1), cols 8·(l>>4)), `at` for an A
/// fragment from K×M storage (transposed: rows (l&7) + 8·(l>>4), cols
/// 8·((l>>3)&1)).
struct Lanes {
    a: (R, R),
    bt: (R, R),
    at: (R, R),
}

fn lane_parts(k: &mut Kernel, lane: R) -> Lanes {
    let l15 = k.and(lane, 15u32);
    let l16 = k.shr(lane, 4u32);
    let l7 = k.and(lane, 7u32);
    let l3 = k.shr(lane, 3u32);
    let l3 = k.and(l3, 1u32);
    let c16 = k.mul(l16, 16u32); // 8 bf16, bytes
    let r_bt = k.mad(l3, 8u32, l7);
    let r_at = k.mad(l16, 8u32, l7);
    let c_at = k.mul(l3, 16u32);
    Lanes { a: (l15, c16), bt: (r_bt, c16), at: (r_at, c_at) }
}

fn lane_addr(k: &mut Kernel, t: &Tile, rc: (R, R)) -> R {
    let a = k.mad(rc.0, t.stride, rc.1);
    k.add(a, t.base)
}

/// Per (value head, slice of [`DVS`] columns): the chunks of the block in
/// order, the slice of the state in f32 accumulators (module doc). Params
/// `conv_out (v), mat, vec, rec_state (bf16 [nv][dk][dv]), out (bf16
/// [m][nv·dv]), m`; blocks `v_heads · dv / DVS`.
pub fn dn_tc_state(s: DeltaShape) -> Emitted {
    let (dk, dv, nv) = (s.k_dim, s.v_dim, s.v_heads);
    assert!(dk == 128 && dv.is_multiple_of(DVS) && DVS == 32 && C == 32, "dk {dk}, dv {dv}");
    let mut k = Kernel::new(format!("dn_tc_state_k{}v{}_c{C}d{DVS}", s.k_heads, nv), THREADS);
    let qkv = k.param_ptr();
    let mat = k.param_ptr();
    let vecb = k.param_ptr();
    let state = k.param_ptr();
    let out = k.param_ptr();
    let m = k.param_u32();
    let sc = (C + 8) * 2; // stride of a C-wide tile
    let sk = (dk + 8) * 2; // stride of a dk-wide tile
    let sd = (DVS + 8) * 2; // stride of a DVS-wide tile
    let tb_s = Tile { base: k.shared(C * sc), stride: sc };
    let p_s = Tile { base: k.shared(C * sc), stride: sc };
    let w_s = Tile { base: k.shared(C * sk), stride: sk };
    let qg_s = Tile { base: k.shared(C * sk), stride: sk };
    let kd_s = Tile { base: k.shared(C * sk), stride: sk };
    let v_s = Tile { base: k.shared(C * sd), stride: sd };
    let s0_s = Tile { base: k.shared(dk * sd), stride: sd };
    let d_s = Tile { base: k.shared(C * sd), stride: sd };
    let g = k.ctaid_x();
    let slices = dv / DVS;
    let h = k.div(g, slices);
    let cs = k.rem(g, slices);
    let c0 = k.mul(cs, DVS);
    let tid = k.tid();
    let lane = k.and(tid, 31u32);
    let warp = k.shr(tid, 5u32);
    let gq = k.shr(lane, 2u32);
    let q4 = k.and(lane, 3u32);
    let q2 = k.shl(q4, 1u32);
    let ln = lane_parts(&mut k, lane);
    let m_last = k.sub(m, 1u32);

    // The warp's state: rows 32·warp + 16i + {g, g+8}, columns 8j + 2q (+1).
    let srow = k.mad(warp, 32u32, gq);
    let s_idx = k.mad(h, dk * dv, c0);
    let s_idx = k.mad(srow, dv, s_idx);
    let s_idx = k.add(s_idx, q2);
    let s_at = k.index(state, s_idx, 2);
    let mut acc: Vec<Vec<[F; 4]>> = Vec::new();
    for i in 0..2u32 {
        let mut row = Vec::new();
        for j in 0..4u32 {
            let mut v = Vec::new();
            for hf in 0..2u32 {
                let off = (((16 * i + 8 * hf) * dv + 8 * j) * 2) as i32;
                let p = k.ld_u32(Ld::Ca, s_at, off);
                v.push(k.h_lo(p));
                v.push(k.h_hi(p));
            }
            row.push([v[0], v[1], v[2], v[3]]);
        }
        acc.push(row);
    }
    // The warp's 16 × 16 tile of Δ and O: rows 16·(warp >> 1), columns 16·(warp & 1).
    let tm = k.shr(warp, 1u32);
    let tm16 = k.mul(tm, 16u32);
    let tn = k.and(warp, 1u32);
    let tn16 = k.mul(tn, 16u32);

    let n_chunks = k.add(m, C - 1);
    let n_chunks = k.div(n_chunks, C);
    k.for_(0u32, n_chunks, 1, |k, ci| {
        let cn = k.mad(ci, nv, h);
        let mbase = k.mul(cn, mat_len(s));
        let mat_c = k.index(mat, mbase, 2);
        let r0 = k.mul(ci, C);
        // a. The chunk's operators and V's slice into shared memory; S0 in bf16.
        let copies: [(u32, &Tile, u32); 5] = [(OFF_TB, &tb_s, C), (OFF_P, &p_s, C), (off_w(s), &w_s, dk), (off_qg(s), &qg_s, dk), (off_kd(s), &kd_s, dk)];
        for &(off, tile, width) in &copies {
            let per_row = width / 8;
            for x in 0..(C * per_row).div_ceil(THREADS) {
                let e = k.add(tid, x * THREADS);
                let inside = k.setp(Cmp::Lt, e, C * per_row);
                let r = k.div(e, per_row);
                let c = k.rem(e, per_row);
                let r = k.min(r, C - 1);
                let src_i = k.mad(r, width, off);
                let c8 = k.mul(c, 8u32);
                let src_i = k.add(src_i, c8);
                let src = k.index(mat_c, src_i, 2);
                let c16 = k.mul(c, 16u32);
                let dst = k.mad(r, tile.stride, c16);
                let dst = k.add(dst, tile.base);
                k.when(inside);
                k.cp_async(dst, 0, src, None);
            }
        }
        {
            // V: rows r0 … r0 + C − 1, columns 2·key + h·dv + c0 …, zeros past m.
            let e = tid; // C rows × 4 chunks = 128
            let r = k.shr(e, 2u32);
            let c = k.and(e, 3u32);
            let grow = k.add(r0, r);
            let inside = k.setp(Cmp::Lt, grow, m);
            let gr = k.min(grow, m_last);
            let vi = k.mad(gr, s.conv_dim(), 2 * s.key());
            let vi = k.mad(h, dv, vi);
            let vi = k.add(vi, c0);
            let c8 = k.mul(c, 8u32);
            let vi = k.add(vi, c8);
            let src = k.index(qkv, vi, 2);
            let bytes = k.selp(16u32, 0u32, inside);
            let c16 = k.mul(c, 16u32);
            let dst = k.mad(r, v_s.stride, c16);
            let dst = k.add(dst, v_s.base);
            k.cp_async(dst, 0, src, Some(bytes));
        }
        k.cp_commit();
        let s0_row = k.mad(warp, 32u32, gq);
        let s0_at = k.mad(s0_row, s0_s.stride, s0_s.base);
        let q2b = k.mul(q2, 2u32);
        let s0_at = k.add(s0_at, q2b);
        for i in 0..2u32 {
            for j in 0..4u32 {
                for hf in 0..2u32 {
                    let a = &acc[i as usize][j as usize];
                    let pk = k.pack_h(a[(2 * hf) as usize], a[(2 * hf + 1) as usize]);
                    k.sts_u32(s0_at, s0_s.at(16 * i + 8 * hf, 8 * j), pk);
                }
            }
        }
        k.cp_wait(0);
        k.bar();
        // b. Δ = Tβ·V − W·S0, the warp's 16 × 16 tile, into d_s (bf16).
        let du = [[k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)], [k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)]];
        let dw = [[k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)], [k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)]];
        let a_tb = lane_addr(k, &tb_s, ln.a);
        let off = k_row_off(k, tm16, tb_s.stride);
        let a_tb = k.add(a_tb, off);
        let b_v = lane_addr(k, &v_s, ln.bt);
        let off = k_col_off(k, tn16);
        let b_v = k.add(b_v, off);
        for kk in 0..C / 16 {
            let a = k.ldmatrix_x4(a_tb, (kk * 32) as i32, false);
            let b = k.ldmatrix_x4(b_v, v_s.at(16 * kk, 0), true);
            k.mma_h(&du[0], &a, &[b[0], b[1]]);
            k.mma_h(&du[1], &a, &[b[2], b[3]]);
        }
        let a_w = lane_addr(k, &w_s, ln.a);
        let off = k_row_off(k, tm16, w_s.stride);
        let a_w = k.add(a_w, off);
        let b_s0 = lane_addr(k, &s0_s, ln.bt);
        let off = k_col_off(k, tn16);
        let b_s0 = k.add(b_s0, off);
        for kk in 0..dk / 16 {
            let a = k.ldmatrix_x4(a_w, (kk * 32) as i32, false);
            let b = k.ldmatrix_x4(b_s0, s0_s.at(16 * kk, 0), true);
            k.mma_h(&dw[0], &a, &[b[0], b[1]]);
            k.mma_h(&dw[1], &a, &[b[2], b[3]]);
        }
        let d_row = k.add(tm16, gq);
        let d_at = k.mad(d_row, d_s.stride, d_s.base);
        let d_at = k.add(d_at, q2b);
        let tn32 = k.mul(tn16, 2u32);
        let d_at = k.add(d_at, tn32);
        for j in 0..2usize {
            for hf in 0..2usize {
                let lo = k.fsub(du[j][2 * hf], dw[j][2 * hf]);
                let hi = k.fsub(du[j][2 * hf + 1], dw[j][2 * hf + 1]);
                let pk = k.pack_h(lo, hi);
                k.sts_u32(d_at, d_s.at(8 * hf as u32, 8 * j as u32), pk);
            }
        }
        k.bar();
        // c. O = QΓ·S0 + P·Δ, the warp's 16 × 16 tile, to `out`.
        let o = [[k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)], [k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)]];
        let a_qg = lane_addr(k, &qg_s, ln.a);
        let off = k_row_off(k, tm16, qg_s.stride);
        let a_qg = k.add(a_qg, off);
        for kk in 0..dk / 16 {
            let a = k.ldmatrix_x4(a_qg, (kk * 32) as i32, false);
            let b = k.ldmatrix_x4(b_s0, s0_s.at(16 * kk, 0), true);
            k.mma_h(&o[0], &a, &[b[0], b[1]]);
            k.mma_h(&o[1], &a, &[b[2], b[3]]);
        }
        let a_p = lane_addr(k, &p_s, ln.a);
        let off = k_row_off(k, tm16, p_s.stride);
        let a_p = k.add(a_p, off);
        let b_d = lane_addr(k, &d_s, ln.bt);
        let off = k_col_off(k, tn16);
        let b_d = k.add(b_d, off);
        for kk in 0..C / 16 {
            let a = k.ldmatrix_x4(a_p, (kk * 32) as i32, false);
            let b = k.ldmatrix_x4(b_d, d_s.at(16 * kk, 0), true);
            k.mma_h(&o[0], &a, &[b[0], b[1]]);
            k.mma_h(&o[1], &a, &[b[2], b[3]]);
        }
        for hf in 0..2u32 {
            let row = k.add(r0, d_row);
            let row = k.add(row, 8 * hf);
            let inside = k.setp(Cmp::Lt, row, m);
            let oi = k.mul(row, s.value());
            let oi = k.mad(h, dv, oi);
            let oi = k.add(oi, c0);
            let oi = k.add(oi, tn16);
            let oi = k.add(oi, q2);
            let oa = k.index(out, oi, 2);
            for j in 0..2usize {
                let pk = k.pack_h(o[j][(2 * hf) as usize], o[j][(2 * hf + 1) as usize]);
                k.when(inside);
                k.st_u32(oa, (16 * j) as i32, pk);
            }
        }
        // d. S = Γ_C·S0 + KDᵀ·Δ.
        let ga = k.index(vecb, cn, 4);
        let gamma = k.ld_f32(Ld::Ca, ga, 0);
        for row in acc.iter() {
            for a in row.iter() {
                for &e in a.iter() {
                    let v = k.fmul(e, gamma);
                    k.fmov_to(e, v);
                }
            }
        }
        let a_kd = lane_addr(k, &kd_s, ln.at);
        let w32 = k.mul(warp, 64u32); // 32 columns of KD (dims), bytes
        let a_kd = k.add(a_kd, w32);
        let b_d2 = lane_addr(k, &d_s, ln.bt);
        for kk in 0..C / 16 {
            let b01 = k.ldmatrix_x4(b_d2, d_s.at(16 * kk, 0), true);
            let b23 = k.ldmatrix_x4(b_d2, d_s.at(16 * kk, 16), true);
            let bs = [[b01[0], b01[1]], [b01[2], b01[3]], [b23[0], b23[1]], [b23[2], b23[3]]];
            for i in 0..2u32 {
                let a = k.ldmatrix_x4(a_kd, kd_s.at(16 * kk, 16 * i), true);
                for j in 0..4usize {
                    k.mma_h(&acc[i as usize][j], &a, &bs[j]);
                }
            }
        }
        // The next chunk overwrites every tile.
        k.bar();
    });
    for i in 0..2u32 {
        for j in 0..4u32 {
            for hf in 0..2u32 {
                let a = &acc[i as usize][j as usize];
                let pk = k.pack_h(a[(2 * hf) as usize], a[(2 * hf + 1) as usize]);
                let off = (((16 * i + 8 * hf) * dv + 8 * j) * 2) as i32;
                k.st_u32(s_at, off, pk);
            }
        }
    }
    Emitted::of(k, 0)
}

/// `rows · stride` bytes.
fn k_row_off(k: &mut Kernel, rows: R, stride: u32) -> R {
    k.mul(rows, stride)
}

/// `cols` bf16 columns, bytes.
fn k_col_off(k: &mut Kernel, cols: R) -> R {
    k.mul(cols, 2u32)
}
