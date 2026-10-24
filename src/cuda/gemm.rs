//! The prefill GEMM on tensor cores: `Y[M][N] = A[M][K] · W[N][K]ᵀ`, bf16 in,
//! f32 accumulators (`mma.sync.m16n8k16`), one rounding at the output.
//!
//! A block computes a `bm × bn` tile of `Y`; its warps (`wm × wn`) each hold a
//! `bm/wm × bn/wn` tile of accumulators in registers. The K loop runs over
//! tiles of `bk`, brought global → shared by `cp.async` (16 bytes per copy,
//! L2 only) `stages − 1` tiles ahead of the one being multiplied; fragments
//! come out of shared memory by `ldmatrix`. Both operands are K-major (A
//! row-major, W row-major = B column-major), which is what `mma .row.col`
//! takes, and what `ldmatrix` without transposition gives for both.
//!
//! The optional XOR layout removes padding and maps 16-byte chunks with
//! `physical = (row * bk * 2 + col_bytes) ^ ((row & 7) << 4)`. Both copies
//! and fragment loads use this permutation, relative to each operand plane.
//! For K32 it exchanges chunks between adjacent rows; for K64 within a row.
//! The default layout is unchanged.
//!
//! Shared rows are padded by 16 bytes (`bk + 8` bf16): eight consecutive rows
//! then start on eight different groups of four banks, and the eight row
//! addresses of each `ldmatrix` matrix hit 32 distinct banks.
//!
//! Rows of A past `M` are filled with zeros (`cp.async` with a source size of
//! 0) and not written. `Out::Residual` adds into `Y`: `y = bf16(y + bf16(acc))`. `N` must be a multiple of `bn` (the loader pads the
//! weights), `K` a multiple of `bk`.
//!
//! Blocks: `(ceil(M / bm), N / bn)`, the M tiles first: consecutive blocks
//! share one tile of W, read once from DRAM and then from L2.

use super::kernels::Out;
use crate::ptx::{Cmp, Emitted, Kernel, Ld, D, F, R};

/// The tiling of a GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    /// Warps along M and along N.
    pub wm: u32,
    pub wn: u32,
    /// Shared buffers of the K pipeline (≥ 2).
    pub stages: u32,
    /// XOR the 16-byte chunk address with low row bits, without row padding.
    pub swizzle: bool,
    /// 0 row-major, 1 column-major, 2 diagonal, 3 streamed B,
    /// 4 one-pair-ahead B, 5 one-row-ahead A. K order is unchanged.
    pub schedule: u32,
    /// 0 early copies, 1 after first K16 MMA, 2 after fragment loads,
    /// 3 per-stage ready/free mbarriers instead of the loop CTA barrier.
    pub pipeline: u32,
    /// Fragments by `ldmatrix` (true) or by one `ld.shared.b32` per register
    /// at the lane's address of the m16n8k16 layout (false), as without it.
    pub ldmatrix: bool,
    /// Global → shared by `cp.async` (true), or through registers (false):
    /// plain `ld.global.v4` before the stage's MMAs, `st.shared.v4` after them.
    pub async_copy: bool,
    /// `wmma.m16n16k16` (fragments in the driver's layout, as Vulkan's
    /// cooperative matrices) instead of `ldmatrix` + `mma.m16n8k16`; the
    /// epilogue goes through shared memory (`wmma.store` has no bf16).
    pub wmma: bool,
    /// Operands and accumulation (probe only; the engine is bf16): 0 bf16
    /// operands, f32 accumulators; 1 f16 operands, f32 accumulators; 2 f16
    /// operands accumulated in f16 over each k tile, added into f32 after it;
    /// 3 f16 operands accumulated in f16 over the whole K.
    pub f16: u32,
}

impl Tile {
    /// Small row tile for short prompts: more independent CTAs when M is
    /// below one SM wave, while retaining the 64-column output tile.
    pub const T32X64: Tile = Tile { bm: 32, bn: 64, bk: 32, wm: 2, wn: 2, stages: 3, swizzle: false, schedule: 0, pipeline: 0, ldmatrix: true, async_copy: true, wmma: false, f16: 0 };
    pub const T32X64_K64_S2: Tile = Tile { bk: 64, stages: 2, ..Self::T32X64 };
    pub const T64: Tile = Tile { bm: 64, bn: 64, ..Self::T32X64 };
    pub const T64X32: Tile = Tile { bn: 32, ..Self::T64 };
    pub const T64X32_K64_S2: Tile = Tile { bk: 64, stages: 2, ..Self::T64X32 };
    pub const T32X64_W1X1_K64_S2: Tile = Tile { wm: 1, wn: 1, bk: 64, stages: 2, ..Self::T32X64 };
    pub const T64_W4X1: Tile = Tile { wm: 4, wn: 1, ..Self::T64 };
    pub const T64_W1X4: Tile = Tile { wm: 1, wn: 4, ..Self::T64 };
    pub const T64_W1X4_K64_S2: Tile = Tile { bk: 64, stages: 2, ..Self::T64_W1X4 };
    /// Fewer pipeline stages with twice the K work per shared-memory tile.
    pub const T64_K64_S2: Tile = Tile { bk: 64, stages: 2, ..Self::T64 };
    pub const T64_K32_S2: Tile = Tile { stages: 2, ..Self::T64 };
    pub const T64_K64_S3: Tile = Tile { stages: 3, ..Self::T64_K64_S2 };
    /// Keep M occupancy at the 64-row tile while sharing twice as many output
    /// columns across the eight warps.
    pub const T64X128: Tile = Tile { bn: 128, wn: 4, ..Self::T64 };
    pub const T64X128_W4X2: Tile = Tile { wm: 4, wn: 2, ..Self::T64X128 };
    pub const T64X128_W2X2: Tile = Tile { wm: 2, wn: 2, ..Self::T64X128 };
    pub const T64X128_W2X2_S2: Tile = Tile { stages: 2, ..Self::T64X128_W2X2 };
    pub const T64X256_S2: Tile = Tile { bn: 256, stages: 2, ..Self::T64X128 };
    pub const T64X128_S2: Tile = Tile { stages: 2, ..Self::T64X128 };
    pub const T64X128_K64_S2: Tile = Tile { bk: 64, stages: 2, ..Self::T64X128 };
    pub const T64X128_K64_S3: Tile = Tile { bk: 64, ..Self::T64X128 };
    pub const T128: Tile = Tile { bm: 128, ..Self::T64X128 };
    pub const T128_K64_S2: Tile = Tile { bk: 64, stages: 2, ..Self::T128 };
    pub const T128_W4X4_K64_S2: Tile = Tile { wm: 4, wn: 4, ..Self::T128_K64_S2 };
    pub const T128_W4X2_K64_S2: Tile = Tile { wm: 4, wn: 2, ..Self::T128_K64_S2 };

    pub fn threads(&self) -> u32 {
        self.wm * self.wn * 32
    }
    fn stride(&self) -> u32 {
        (self.bk + if self.swizzle { 0 } else { 8 }) * 2
    }
    fn stage_bytes(&self) -> u32 {
        (self.bm + self.bn) * self.stride()
    }
    pub fn shared_bytes(&self) -> u32 {
        let pipe = self.stages * self.stage_bytes() + if self.pipeline == 3 { self.stages * 16 } else { 0 };
        // wmma: the epilogue reuses the stages for each warp's f32 tile.
        if self.wmma { pipe.max(self.bm * self.bn * 4) } else { pipe }
    }
    pub fn grid(&self, m: u32, n: u32) -> (u32, u32, u32) {
        (m.div_ceil(self.bm), n / self.bn, 1)
    }
    pub fn tag(&self) -> String {
        let mut tag = format!("{}x{}x{}_w{}x{}_s{}{}", self.bm, self.bn, self.bk, self.wm, self.wn, self.stages, if self.swizzle { "_xor" } else { "" });
        if self.schedule != 0 { tag.push_str(&format!("_sched{}", self.schedule)); }
        if self.pipeline != 0 { tag.push_str(&format!("_pipe{}", self.pipeline)); }
        if !self.ldmatrix { tag.push_str("_lds"); }
        if !self.async_copy { tag.push_str("_sync"); }
        if self.wmma { tag.push_str("_wmma"); }
        if self.f16 != 0 { tag.push_str(&format!("_f16m{}", self.f16)); }
        tag
    }
}

/// Params `A, W, Y, M`; blocks [`Tile::grid`]; dynamic shared [`Tile::shared_bytes`].
pub fn gemm(kdim: u32, n: u32, out: Out, t: Tile) -> Emitted {
    let (wtm, wtn) = (t.bm / t.wm, t.bn / t.wn);
    assert!(wtm % 16 == 0 && wtn % 16 == 0, "warp tile {wtm}×{wtn}: multiples of 16");
    assert!(t.bk % 16 == 0 && t.bk.is_power_of_two() && t.stages >= 2);
    assert!(t.schedule <= 5 && t.pipeline <= 3);
    assert!(!t.swizzle || t.bk >= 32, "XOR layout requires bk >= 32");
    assert!(t.async_copy || t.pipeline == 0, "copies through registers use pipeline 0");
    assert!(t.f16 == 0 || (t.f16 <= 3 && t.schedule == 0 && !t.wmma), "f16 modes: schedule 0, mma.m16n8k16");
    assert!(!t.wmma || (!t.swizzle && t.schedule == 0 && matches!(t.pipeline, 0 | 3)), "wmma reads a linear, padded layout in its own order, early copies");
    assert!(kdim % t.bk == 0, "K = {kdim} is not a multiple of bk = {}", t.bk);
    assert!(n % t.bn == 0, "N = {n} is not a multiple of bn = {}", t.bn);
    let threads = t.threads();
    let (mt, nt) = (wtm / 16, wtn / 8);
    let stride = t.stride();
    let sb = t.stage_bytes();
    let ktiles = kdim / t.bk;
    let row_chunks = t.bk / 8; // 16-byte chunks per tile row
    let a_chunks = t.bm * row_chunks;
    let b_chunks = t.bn * row_chunks;
    assert!(a_chunks % threads == 0 && b_chunks % threads == 0, "tile loads must divide among {threads} threads");

    let name = format!("gemm_{kdim}x{n}_{}_{}", out.tag(), t.tag());
    let mut k = Kernel::new(name, threads);
    let a = k.param_ptr();
    let w = k.param_ptr();
    let y = k.param_ptr();
    let m = k.param_u32();
    let smem = k.dyn_shared();
    let tid = k.tid();
    let lane = k.and(tid, 31u32);
    let warp = k.shr(tid, 5u32);
    let bmi = k.ctaid_x();
    let bni = k.ctaid_y();
    let m0 = k.mul(bmi, t.bm);
    let n0 = k.mul(bni, t.bn);
    let m_last = k.sub(m, 1u32);

    // ── the copies of one k tile: per thread, its chunks of A and of W ──
    let shift = row_chunks.trailing_zeros();
    let mut a_src: Vec<(D, R)> = Vec::new(); // (global address at k = 0, source bytes)
    let mut a_dst: Vec<R> = Vec::new(); // shared offset in a stage
    for j in 0..a_chunks / threads {
        let c = k.add(tid, j * threads);
        let row = k.shr(c, shift);
        let col = k.and(c, row_chunks - 1);
        let gm = k.add(m0, row);
        let inside = k.setp(Cmp::Lt, gm, m);
        let gm = k.min(gm, m_last);
        let at = k.index(a, gm, kdim * 2);
        let col16 = k.mul(col, 16u32);
        let src = k.index(at, col16, 1);
        let bytes = k.selp(16u32, 0u32, inside);
        let so = k.mad(row, stride, col16);
        let so = if t.swizzle { let low = k.and(row, 7u32); let mask = k.shl(low, 4u32); k.xor(so, mask) } else { so };
        let so = k.add(so, smem);
        a_src.push((src, bytes));
        a_dst.push(so);
    }
    let mut b_src: Vec<D> = Vec::new();
    let mut b_dst: Vec<R> = Vec::new();
    for j in 0..b_chunks / threads {
        let c = k.add(tid, j * threads);
        let row = k.shr(c, shift);
        let col = k.and(c, row_chunks - 1);
        let gn = k.add(n0, row);
        let at = k.index(w, gn, kdim * 2);
        let col16 = k.mul(col, 16u32);
        let src = k.index(at, col16, 1);
        let so = k.mad(row, stride, col16);
        let so = if t.swizzle { let low = k.and(row, 7u32); let mask = k.shl(low, 4u32); k.xor(so, mask) } else { so };
        let so = k.add(so, smem);
        let so = k.add(so, t.bm * stride);
        b_src.push(src);
        b_dst.push(so);
    }
    // Ready/free barriers are separate: ready orders asynchronous writes,
    // free orders all consumers before the stage is reused. There is one
    // arrival per CTA thread in each phase of each barrier.
    let ready_base = if t.pipeline == 3 {
        let base = k.add(smem, t.stages * sb);
        let leader = k.setp(Cmp::Eq, tid, 0u32);
        k.if_(leader, |k| {
            for s in 0..t.stages * 2 {
                let at = k.add(base, s * 8);
                k.mbarrier_init(at, threads);
            }
        });
        k.bar();
        Some(base)
    } else { None };
    // Copy k tile `kt` (known at emission) or the tile the sources point at,
    // into the stage at byte `stage` (register) + `fixed`.
    let issue = |k: &mut Kernel, stage: Option<R>, fixed: u32, a_src: &[(D, R)], b_src: &[D], ready: Option<R>| {
        for (i, &(src, bytes)) in a_src.iter().enumerate() {
            let dst = match stage {
                Some(s) => k.add(a_dst[i], s),
                None => a_dst[i],
            };
            k.cp_async(dst, fixed as i32, src, Some(bytes));
        }
        for (i, &src) in b_src.iter().enumerate() {
            let dst = match stage {
                Some(s) => k.add(b_dst[i], s),
                None => b_dst[i],
            };
            k.cp_async(dst, fixed as i32, src, None);
        }
        k.cp_commit();
        if let Some(ready) = ready { k.cp_mbarrier_arrive(ready); }
    };
    // Without cp.async: the same chunks loaded into registers (A rows past M
    // read as zeros), to be stored by `store`.
    let load = |k: &mut Kernel, stage: Option<R>, fixed: u32, a_src: &[(D, R)], b_src: &[D]| -> Vec<(R, i32, [R; 4])> {
        let mut pending = Vec::new();
        for (i, &(src, bytes)) in a_src.iter().enumerate() {
            let dst = match stage { Some(s) => k.add(a_dst[i], s), None => a_dst[i] };
            let v = k.ld_v4(Ld::Ca, src, 0);
            let empty = k.setp(Cmp::Eq, bytes, 0u32);
            for &r in &v { k.when(empty); k.mov_to(r, 0u32); }
            pending.push((dst, fixed as i32, v));
        }
        for (i, &src) in b_src.iter().enumerate() {
            let dst = match stage { Some(s) => k.add(b_dst[i], s), None => b_dst[i] };
            pending.push((dst, fixed as i32, k.ld_v4(Ld::Ca, src, 0)));
        }
        pending
    };
    let store = |k: &mut Kernel, pending: &[(R, i32, [R; 4])]| {
        for &(dst, off, v) in pending { k.sts_v4(dst, off, v); }
    };
    let advance = |k: &mut Kernel, a_src: &mut Vec<(D, R)>, b_src: &mut Vec<D>| {
        for s in a_src.iter_mut() {
            k.ins(format!("add.s64 {}, {}, {}", s.0, s.0, t.bk * 2));
        }
        for s in b_src.iter_mut() {
            k.ins(format!("add.s64 {s}, {s}, {}", t.bk * 2));
        }
    };

    // ── prologue: tiles 0 … stages − 2 ──
    for s in 0..(t.stages - 1).min(ktiles) {
        if t.async_copy {
            let ready = ready_base.map(|base| k.add(base, s * 8));
            issue(&mut k, None, s * sb, &a_src, &b_src, ready);
        } else {
            let pending = load(&mut k, None, s * sb, &a_src, &b_src);
            store(&mut k, &pending);
        }
        advance(&mut k, &mut a_src, &mut b_src);
    }
    if t.async_copy {
        for _ in ktiles..t.stages - 1 {
            k.cp_commit();
        }
    }

    // ── per-lane fragment addresses ──
    let wmi = k.rem(warp, t.wm);
    let wni = k.div(warp, t.wm);
    // A: lanes 0–15 rows 0–15 at k 0, lanes 16–31 rows 0–15 at k 8.
    let l15 = k.and(lane, 15u32);
    let ahalf = k.shr(lane, 4u32);
    let arow = k.mad(wmi, wtm, l15);
    let ak = k.mul(ahalf, 16u32);
    let a_logical = k.mad(arow, stride, ak);
    let a_frag = k.add(a_logical, smem);
    // W: lanes 0–7 rows 0–7 k 0, 8–15 rows 0–7 k 8, 16–23 rows 8–15 k 0, 24–31 rows 8–15 k 8.
    let l7 = k.and(lane, 7u32);
    let bhi = k.shr(lane, 4u32);
    let bhi8 = k.shl(bhi, 3u32);
    let brow = k.add(l7, bhi8);
    let brow = k.mad(wni, wtn, brow);
    let bk8 = k.shr(lane, 3u32);
    let bk8 = k.and(bk8, 1u32);
    let bk16 = k.mul(bk8, 16u32);
    let b_logical = k.mad(brow, stride, bk16);
    let b_frag = k.add(b_logical, smem);
    let b_frag = k.add(b_frag, t.bm * stride);
    let masks = if t.swizzle {
        let ar = k.and(arow, 7u32); let br = k.and(brow, 7u32);
        Some((k.shl(ar, 4u32), k.shl(br, 4u32)))
    } else { None };
    // Without ldmatrix, lane t reads its own fragment words: rows g and g + 8
    // of its warp's tile, bytes 4q and 16 + 4q of each k16 step. The XOR mask
    // of rows g and g + 8 (warp tiles are multiples of 16 rows) is g << 4.
    let lds_lane = if t.ldmatrix { None } else {
        let g = k.shr(lane, 2u32);
        let q = k.and(lane, 3u32);
        let q4 = k.shl(q, 2u32);
        let ar = k.mad(wmi, wtm, g);
        let br = k.mad(wni, wtn, g);
        let al = k.mad(ar, stride, q4);
        let bl = k.mad(br, stride, q4);
        let mask = if t.swizzle { Some(k.shl(g, 4u32)) } else { None };
        Some((al, bl, mask))
    };

    let acc: Vec<Vec<[F; 4]>> = if t.wmma { Vec::new() } else {
        (0..mt).map(|_| (0..nt).map(|_| [k.fmov(0.0), k.fmov(0.0), k.fmov(0.0), k.fmov(0.0)]).collect()).collect()
    };
    // f16 accumulation (modes 2 and 3): one f16x2 pair per row half.
    let acc16: Vec<Vec<[R; 2]>> = if t.f16 >= 2 {
        (0..mt).map(|_| (0..nt).map(|_| [k.mov(0u32), k.mov(0u32)]).collect()).collect()
    } else { Vec::new() };
    let flush = |k: &mut Kernel, acc: &Vec<Vec<[F; 4]>>, acc16: &Vec<Vec<[R; 2]>>| {
        for i in 0..mt as usize { for j in 0..nt as usize {
            for h in 0..2 {
                let (lo, hi) = k.f16x2_to_f32(acc16[i][j][h]);
                k.fadd_to(acc[i][j][2 * h], acc[i][j][2 * h], lo);
                k.fadd_to(acc[i][j][2 * h + 1], acc[i][j][2 * h + 1], hi);
                k.mov_to(acc16[i][j][h], 0u32);
            }
        } }
    };
    let (mw, nw) = (wtm / 16, wtn / 16);
    let wacc: Vec<Vec<[F; 8]>> = if t.wmma {
        (0..mw).map(|_| (0..nw).map(|_| std::array::from_fn(|_| k.fmov(0.0))).collect()).collect()
    } else { Vec::new() };
    // wmma: the warp's tile origins in a stage, A rows and W rows.
    let wmma_at = t.wmma.then(|| {
        let ar = k.mul(wmi, wtm);
        let a0 = k.mad(ar, stride, smem);
        let br = k.mul(wni, wtn);
        let b0 = k.mad(br, stride, smem);
        let b0 = k.add(b0, t.bm * stride);
        (a0, b0)
    });

    // ── the K loop ──
    let rd = k.mov(0u32); // stage being multiplied, bytes
    let wr = k.mov((t.stages - 1) * sb); // stage being filled, bytes
    let kt = k.mov(0u32);
    let barrier_state = ready_base.map(|base| (k.add(base, 0u32), k.add(base, (t.stages - 1) * 8), k.mov(0u32), k.mov(1u32)));
    let top = k.label();
    k.place(top);
    if let Some((read_bar, _, parity, _)) = barrier_state {
        k.mbarrier_wait(read_bar, parity);
    } else {
        if t.async_copy { k.cp_wait(t.stages - 2); }
        k.bar();
    }
    let prefetch = |k: &mut Kernel| -> Option<(crate::ptx::P, Vec<(R, i32, [R; 4])>)> {
        // The first write uses the unoccupied final stage. Subsequent
        // writes wait until every consumer of that stage has finished.
        let next = k.add(kt, t.stages - 1);
        let more = k.setp(Cmp::Lt, next, ktiles);
        let mut pending = Vec::new();
        k.if_(more, |k| {
            if let Some((_, write_bar, _, free_parity)) = barrier_state {
                let reused = k.setp(Cmp::Ne, kt, 0u32);
                k.if_(reused, |k| {
                    let free = k.add(write_bar, t.stages * 8);
                    k.mbarrier_wait(free, free_parity);
                });
            }
            if t.async_copy {
                issue(k, Some(wr), 0, &a_src, &b_src, barrier_state.map(|b| b.1));
            } else {
                pending = load(k, Some(wr), 0, &a_src, &b_src);
            }
            let (mut asrc, mut bsrc) = (a_src.clone(), b_src.clone());
            advance(k, &mut asrc, &mut bsrc);
        });
        if barrier_state.is_none() && t.async_copy {
            let none = k.pnot(more);
            k.when(none);
            k.cp_commit();
        }
        k.add_to(wr, wr, sb);
        let wrap = k.setp(Cmp::Eq, wr, t.stages * sb);
        k.when(wrap);
        k.mov_to(wr, 0u32);
        if let Some((_, write_bar, _, parity)) = barrier_state {
            k.add_to(write_bar, write_bar, 8u32);
            k.if_(wrap, |k| {
                k.mov_to(write_bar, ready_base.unwrap());
                k.ins(format!("xor.b32 {parity}, {parity}, 1"));
            });
        }
        if t.async_copy { None } else { Some((more, pending)) }
    };
    let deferred = if matches!(t.pipeline, 0 | 3) { prefetch(&mut k) } else { None };
    // Multiply the stage at rd.
    let a_at = k.add(a_frag, rd);
    let b_at = k.add(b_frag, rd);
    let lds_at = lds_lane.map(|(al, bl, _)| {
        let a = k.add(al, smem);
        let b = k.add(bl, smem);
        let b = k.add(b, t.bm * stride);
        (k.add(a, rd), k.add(b, rd))
    });
    for kk in 0..t.bk / 16 {
        if let Some((a0, b0)) = wmma_at {
            let ldm = stride / 2;
            let a_at = k.add(a0, rd);
            let b_at = k.add(b0, rd);
            let af: Vec<[R; 4]> = (0..mw).map(|i| { let p = k.add(a_at, i * 16 * stride + kk * 32); k.wmma_load_a_h(p, ldm) }).collect();
            let bf: Vec<[R; 4]> = (0..nw).map(|j| { let p = k.add(b_at, j * 16 * stride + kk * 32); k.wmma_load_b_h(p, ldm) }).collect();
            for i in 0..mw as usize { for j in 0..nw as usize { k.wmma_mma_h(&wacc[i][j], &af[i], &bf[j]); } }
            continue;
        }
        // Row increments here are multiples of 16 and do not change the XOR
        // mask. K offsets must be applied before XOR, then the stage/base.
        let (a_ptr, b_ptr, k_offset) = if let Some((am, bm)) = masks {
            let al = k.add(a_logical, kk * 32); let bl = k.add(b_logical, kk * 32);
            let ax = k.xor(al, am); let bx = k.xor(bl, bm);
            let ap = k.add(ax, smem); let bp = k.add(bx, smem);
            let bp = k.add(bp, t.bm * stride);
            (k.add(ap, rd), k.add(bp, rd), 0)
        } else { (a_at, b_at, kk * 32) };
        // Without ldmatrix: the bases of the two k8 halves, an offset common
        // to both, and the byte step from the first half to the second.
        let lds = lds_lane.map(|(al, bl, mask)| match mask {
            Some(mask) => {
                let mut half = |logical: R, plane: u32, h: u32| {
                    let l = k.add(logical, kk * 32 + h * 16);
                    let x = k.xor(l, mask);
                    let p = k.add(x, smem);
                    let p = if plane > 0 { k.add(p, plane) } else { p };
                    k.add(p, rd)
                };
                let a = [half(al, 0, 0), half(al, 0, 1)];
                let b = [half(bl, t.bm * stride, 0), half(bl, t.bm * stride, 1)];
                (a, b, 0u32, 0u32)
            }
            None => {
                let (a, b) = lds_at.unwrap();
                ([a, a], [b, b], kk * 32, 16u32)
            }
        });
        let load_a = |k: &mut Kernel, i: u32| -> [R; 4] {
            match lds {
                None => k.ldmatrix_x4(a_ptr, (i * 16 * stride + k_offset) as i32, false),
                Some((a, _, off, step)) => {
                    let o = (i * 16 * stride + off) as i32;
                    let r8 = (8 * stride) as i32;
                    [k.lds_u32(a[0], o), k.lds_u32(a[0], o + r8), k.lds_u32(a[1], o + step as i32), k.lds_u32(a[1], o + r8 + step as i32)]
                }
            }
        };
        let load_b = |k: &mut Kernel, j: u32| {
            match lds {
                None => {
                    let v = k.ldmatrix_x4(b_ptr, (j * 16 * stride + k_offset) as i32, false);
                    [[v[0], v[1]], [v[2], v[3]]]
                }
                Some((_, b, off, step)) => {
                    let o = (j * 16 * stride + off) as i32;
                    let r8 = (8 * stride) as i32;
                    let v0 = k.lds_u32(b[0], o);
                    let v1 = k.lds_u32(b[1], o + step as i32);
                    let v2 = k.lds_u32(b[0], o + r8);
                    let v3 = k.lds_u32(b[1], o + r8 + step as i32);
                    [[v0, v1], [v2, v3]]
                }
            }
        };
        let during_loads = |k: &mut Kernel| { if kk == 0 && t.pipeline == 2 { prefetch(k); } };
        if t.schedule <= 2 {
            let af: Vec<[R; 4]> = (0..mt).map(|i| load_a(&mut k, i)).collect();
            let mut bf: Vec<[R; 2]> = Vec::new();
            for j in 0..nt / 2 { bf.extend(load_b(&mut k, j)); }
            during_loads(&mut k);
            if t.schedule == 0 {
                for i in 0..mt as usize { for j in 0..nt as usize {
                    match t.f16 {
                        0 => k.mma_h(&acc[i][j], &af[i], &bf[j]),
                        1 => k.mma_f16(&acc[i][j], &af[i], &bf[j]),
                        _ => k.mma_f16_f16(&acc16[i][j], &af[i], &bf[j]),
                    }
                } }
            } else if t.schedule == 1 {
                for j in 0..nt as usize { for i in 0..mt as usize { k.mma_h(&acc[i][j], &af[i], &bf[j]); } }
            } else {
                for q in 0..nt as usize { for i in 0..mt as usize {
                    let j = (q + i) % nt as usize;
                    k.mma_h(&acc[i][j], &af[i], &bf[j]);
                } }
            }
        } else if matches!(t.schedule, 3 | 4) {
            let af: Vec<[R; 4]> = (0..mt).map(|i| load_a(&mut k, i)).collect();
            let mut pair = load_b(&mut k, 0);
            during_loads(&mut k);
            for j in 0..nt / 2 {
                let next = if t.schedule == 4 && j + 1 < nt / 2 { Some(load_b(&mut k, j + 1)) } else { None };
                for i in 0..mt as usize { for v in 0..2 {
                    k.mma_h(&acc[i][j as usize * 2 + v], &af[i], &pair[v]);
                } }
                if j + 1 < nt / 2 { pair = next.unwrap_or_else(|| load_b(&mut k, j + 1)); }
            }
        } else {
            let mut bf: Vec<[R; 2]> = Vec::new();
            for j in 0..nt / 2 { bf.extend(load_b(&mut k, j)); }
            let mut af = load_a(&mut k, 0);
            during_loads(&mut k);
            for i in 0..mt {
                let next = if i + 1 < mt { Some(load_a(&mut k, i + 1)) } else { None };
                for j in 0..nt as usize { k.mma_h(&acc[i as usize][j], &af, &bf[j]); }
                if let Some(next) = next { af = next; }
            }
        }
        if kk == 0 && t.pipeline == 1 { prefetch(&mut k); }
    }
    if t.f16 == 2 { flush(&mut k, &acc, &acc16); }
    // Copies through registers: the next stage's stores, after this stage's MMAs.
    if let Some((more, pending)) = &deferred {
        k.if_(*more, |k| store(k, pending));
    }
    if let Some((read_bar, _, _, _)) = barrier_state {
        let free = k.add(read_bar, t.stages * 8);
        k.mbarrier_arrive(free);
    }
    k.add_to(rd, rd, sb);
    let wrap = k.setp(Cmp::Eq, rd, t.stages * sb);
    k.when(wrap);
    k.mov_to(rd, 0u32);
    if let Some((read_bar, _, parity, _)) = barrier_state {
        k.add_to(read_bar, read_bar, 8u32);
        k.if_(wrap, |k| {
            k.mov_to(read_bar, ready_base.unwrap());
            k.ins(format!("xor.b32 {parity}, {parity}, 1"));
        });
    }
    k.add_to(kt, kt, 1u32);
    let again = k.setp(Cmp::Lt, kt, ktiles);
    k.bra_if(again, top);
    if t.async_copy { k.cp_wait(0); }

    if t.f16 == 3 { flush(&mut k, &acc, &acc16); }
    let esize = match out {
        Out::F32 => 4,
        _ => 2,
    };
    if t.wmma {
        // ── wmma epilogue: each warp's f32 tile through shared memory ──
        k.bar();
        let wbase = k.mad(warp, wtm * wtn * 4, smem);
        for i in 0..mw { for j in 0..nw {
            let p = k.add(wbase, (i * 16 * wtn + j * 16) * 4);
            k.wmma_store_f32(p, &wacc[i as usize][j as usize], wtn);
        } }
        k.bar_warp();
        let row0 = k.mad(wmi, wtm, m0);
        let col0 = k.mad(wni, wtn, n0);
        for x in 0..wtm * wtn / 2 / 32 {
            let e = k.add(lane, x * 32);
            let r = k.div(e, wtn / 2);
            let c2 = k.rem(e, wtn / 2);
            let c = k.shl(c2, 1u32);
            let sa = k.mad(r, wtn * 4, wbase);
            let sa = k.mad(c, 4u32, sa);
            let lo = k.lds_f32(sa, 0);
            let hi = k.lds_f32(sa, 4);
            let gr = k.add(row0, r);
            let inside = k.setp(Cmp::Lt, gr, m);
            let gc = k.add(col0, c);
            let at = k.mad(gr, n, gc);
            let base = k.index(y, at, esize);
            match out {
                Out::F32 => { k.when(inside); k.st_v2_f32(base, 0, [lo, hi]); }
                Out::Bf16 => { let p = k.pack_h(lo, hi); k.when(inside); k.st_u32(base, 0, p); }
                Out::Residual => k.if_(inside, |k| {
                    let old = k.ld_u32(crate::ptx::Ld::Ca, base, 0);
                    let (ol, oh) = (k.h_lo(old), k.h_hi(old));
                    let (rl, rh) = (k.round_h(lo), k.round_h(hi));
                    let sl = k.fadd(ol, rl);
                    let sh = k.fadd(oh, rh);
                    let p = k.pack_h(sl, sh);
                    k.st_u32(base, 0, p);
                }),
            }
        }
        return Emitted::of(k, t.shared_bytes());
    }

    // ── epilogue: lane t holds rows g, g + 8 and columns 2q, 2q + 1 ──
    let g = k.shr(lane, 2u32);
    let q = k.and(lane, 3u32);
    let row = k.mad(wmi, wtm, g);
    let row = k.add(row, m0);
    let col = k.mad(wni, wtn, n0);
    let q2 = k.shl(q, 1u32);
    let col = k.add(col, q2);
    for i in 0..mt {
        for half in 0..2u32 {
            let r = k.add(row, i * 16 + half * 8);
            let inside = k.setp(Cmp::Lt, r, m);
            let at = k.mad(r, n, col);
            let base = k.index(y, at, esize);
            for j in 0..nt as usize {
                let c = &acc[i as usize][j];
                let (lo, hi) = (c[(half * 2) as usize], c[(half * 2 + 1) as usize]);
                let off = (j as u32 * 8 * esize) as i32;
                match out {
                    Out::F32 => {
                        k.when(inside);
                        k.st_v2_f32(base, off, [lo, hi]);
                    }
                    Out::Bf16 => {
                        let p = k.pack_h(lo, hi);
                        k.when(inside);
                        k.st_u32(base, off, p);
                    }
                    Out::Residual => k.if_(inside, |k| {
                        // y = bf16(y + bf16(acc)), as the decode's matvec.
                        let old = k.ld_u32(crate::ptx::Ld::Ca, base, off);
                        let (ol, oh) = (k.h_lo(old), k.h_hi(old));
                        let (rl, rh) = (k.round_h(lo), k.round_h(hi));
                        let sl = k.fadd(ol, rl);
                        let sh = k.fadd(oh, rh);
                        let p = k.pack_h(sl, sh);
                        k.st_u32(base, off, p);
                    }),
                }
            }
        }
    }
    Emitted::of(k, t.shared_bytes())
}
