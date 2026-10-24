//! The kernel parameters of the engine, overridable by `--tune key=value`.
//!
//! The defaults are the best measured on the RTX 4080 so far; every key keeps
//! the alternatives measurable. Since 2026-10-19 the default is the measured
//! candidate (`gemm.auto=3 gemm.layout=xor prefill.block=auto`); the former
//! default is `gemm.auto=0 gemm.layout=pad prefill.block=1024`.

use super::gemm::Tile;
use super::kernels::Matvec;
use super::types::Act;
use crate::core::error::{HerbertError, Result};
use crate::ptx::Ld;

#[derive(Clone, Debug)]
pub struct Tuning {
    /// `matvec.rows`, `matvec.warps`, `matvec.depth`, `matvec.cache` (ca, nc, nol1, cg, cs).
    pub matvec: Matvec,
    /// `sdpa.split`: blocks per head of the decode attention.
    pub sdpa_split: u32,
    /// `act`: bf16 or f32, the DeltaNet's activations and states.
    pub act: Act,
    /// `graph`: 1 = the decode step replayed as a CUDA graph, 0 = launched one by one.
    pub graph: u32,
    /// `prefill.block`: rows of a prefill block (1 = the prompt token by token
    /// through the decode step). Zero denotes model-dependent `auto`.
    pub prefill_block: u32,
    /// `gemm.tile`: row/column tile and optional warp layout (see `tile`).
    pub gemm_tile: String,
    /// K depth and pipeline stages of the prefill GEMM.
    pub gemm_bk: u32,
    pub gemm_stages: u32,
    /// 0 = manual, 1 = block height, 2/4 = matrix experiments, 3 = measured policy.
    pub gemm_auto: u32,
    /// Optional preferred shared memory percentage (0..100); None = driver.
    pub gemm_carveout: Option<u32>,
    pub gemm_swizzle: bool,
    /// Automatic policies apply these only to 4B 64x128 tiles; manual tiles
    /// apply them directly. Defaults keep the original PTX.
    pub gemm_schedule: u32,
    pub gemm_pipeline: u32,
    /// Explicit down-only tile at exactly M=128/256 (same = normal policy).
    pub gemm_down_short: String,
    /// `gemm.ldmatrix=0`: fragments by `ld.shared.b32` instead of `ldmatrix`.
    pub gemm_ldmatrix: bool,
    /// `gemm.copy=sync`: global → shared through registers instead of `cp.async`.
    pub gemm_async_copy: bool,
    /// `gemm.wmma=1`: `wmma.m16n16k16`, the driver's fragment layout.
    pub gemm_wmma: bool,
    /// `gemm.f16=1`: the f16 experiment. Projection weights in IEEE f16, the
    /// prefill GEMMs' inputs written in f16, f16 accumulation over each k tile
    /// added into f32 (`Tile::f16 = 2`); the decode matvecs read the f16
    /// weights. Another numerical contract than the oracle's.
    pub gemm_f16: bool,
    /// `sdpa.bk`: keys per step of the prefill attention (multiple of 16).
    pub sdpa_bk: u32,
    /// `sdpa.softmax=shared`: the prefill attention's softmax through shared
    /// memory, as with opaque fragments (Vulkan), instead of in registers.
    pub sdpa_smem_softmax: bool,
    /// `rec.tc`: 1 = the prefill's delta rule chunked on the tensor cores
    /// (`cuda/dntc.rs`, bf16 activations), 0 = row by row.
    pub rec_tc: u32,
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning { matvec: Matvec { rows: 2, warps: 4, depth: 8, cache: Ld::NcNoL1, w16: false }, sdpa_split: 8, act: Act::Bf16, graph: 1, prefill_block: 0, gemm_tile: "64".to_string(), gemm_bk: 32, gemm_stages: 3, gemm_auto: 3, gemm_carveout: None, gemm_swizzle: true, gemm_schedule: 0, gemm_pipeline: 0, gemm_down_short: "same".into(), gemm_ldmatrix: true, gemm_async_copy: true, gemm_wmma: false, gemm_f16: false, sdpa_bk: 32, sdpa_smem_softmax: false, rec_tc: 1 }
    }
}

fn cache_name(c: Ld) -> &'static str {
    match c {
        Ld::Ca => "ca",
        Ld::Nc => "nc",
        Ld::NcNoL1 => "nol1",
        Ld::Cg => "cg",
        Ld::Cs => "cs",
    }
}

impl Tuning {
    pub fn with(overrides: &[String]) -> Result<Tuning> {
        let mut t = Tuning::default();
        for o in overrides {
            let (key, value) = o.split_once('=').ok_or_else(|| HerbertError::Config(format!("--tune {o}: key=value expected")))?;
            let num = || value.parse::<u32>().map_err(|e| HerbertError::Config(format!("--tune {o}: {e}")));
            match key {
                "matvec.rows" => t.matvec.rows = num()?,
                "matvec.warps" => t.matvec.warps = num()?,
                "matvec.depth" => t.matvec.depth = num()?,
                "matvec.cache" => {
                    t.matvec.cache = match value {
                        "ca" => Ld::Ca,
                        "nc" => Ld::Nc,
                        "nol1" => Ld::NcNoL1,
                        "cg" => Ld::Cg,
                        "cs" => Ld::Cs,
                        _ => return Err(HerbertError::Config(format!("--tune {o}: ca, nc, nol1, cg or cs"))),
                    }
                }
                "sdpa.split" => t.sdpa_split = num()?.max(1),
                "act" => {
                    t.act = match value {
                        "bf16" => Act::Bf16,
                        "f32" => Act::F32,
                        _ => return Err(HerbertError::Config(format!("--tune {o}: bf16 or f32"))),
                    }
                }
                "graph" => t.graph = num()?,
                "prefill.block" => t.prefill_block = if value == "auto" { 0 } else { num()?.max(1) },
                "gemm.tile" => t.gemm_tile = value.to_string(),
                "gemm.bk" => t.gemm_bk = num()?,
                "gemm.stages" => t.gemm_stages = num()?,
                "gemm.auto" => t.gemm_auto = num()?,
                "gemm.carveout" => t.gemm_carveout = if value == "driver" { None } else { Some(num()?) },
                "gemm.layout" => t.gemm_swizzle = match value { "pad" => false, "xor" => true, _ => return Err(HerbertError::Config("gemm.layout: pad or xor".into())) },
                "gemm.schedule" => t.gemm_schedule = num()?,
                "gemm.pipeline" => t.gemm_pipeline = num()?,
                "gemm.down_short" => t.gemm_down_short = value.to_string(),
                "gemm.ldmatrix" => t.gemm_ldmatrix = num()? != 0,
                "gemm.wmma" => t.gemm_wmma = num()? != 0,
                "gemm.f16" => t.gemm_f16 = num()? != 0,
                "gemm.copy" => t.gemm_async_copy = match value { "async" => true, "sync" => false, _ => return Err(HerbertError::Config("gemm.copy: async or sync".into())) },
                "sdpa.bk" => t.sdpa_bk = num()?,
                "sdpa.softmax" => t.sdpa_smem_softmax = match value { "reg" => false, "shared" => true, _ => return Err(HerbertError::Config("sdpa.softmax: reg or shared".into())) },
                "rec.tc" => t.rec_tc = num()?,
                _ => return Err(HerbertError::Config(format!("--tune {o}: unknown key (matvec.rows, matvec.warps, matvec.depth, matvec.cache, sdpa.split, act, graph, prefill.block, gemm.tile, gemm.bk, gemm.stages, gemm.auto, gemm.carveout, gemm.layout, gemm.schedule, gemm.pipeline, gemm.down_short, gemm.ldmatrix, gemm.copy, gemm.wmma, gemm.f16, sdpa.bk, sdpa.softmax, rec.tc)"))),
            }
        }
        Ok(t)
    }

    pub fn tile(&self) -> Result<Tile> {
        if self.gemm_schedule > 5 || self.gemm_pipeline > 3 {
            return Err(HerbertError::Config("gemm.schedule: 0..5; gemm.pipeline: 0..3".into()));
        }
        self.down_short_tiles()?;
        if !self.gemm_async_copy && self.gemm_pipeline != 0 {
            return Err(HerbertError::Config("gemm.copy=sync requires gemm.pipeline=0".into()));
        }
        if self.gemm_f16 && (self.gemm_wmma || self.gemm_schedule != 0) {
            return Err(HerbertError::Config("gemm.f16=1 requires mma.m16n8k16 (gemm.wmma=0) and gemm.schedule=0".into()));
        }
        if self.gemm_wmma && (self.gemm_swizzle || self.gemm_schedule != 0 || !matches!(self.gemm_pipeline, 0 | 3)) {
            return Err(HerbertError::Config("gemm.wmma=1 requires gemm.layout=pad (the default is xor), gemm.schedule=0 and gemm.pipeline=0 or 3".into()));
        }
        if self.gemm_carveout.is_some_and(|v| v > 100) {
            return Err(HerbertError::Config("gemm.carveout: driver or 0..100".into()));
        }
        if !matches!(self.gemm_auto, 0..=6) {
            return Err(HerbertError::Config(format!("--tune gemm.auto={}: 0 through 6", self.gemm_auto)));
        }
        if self.gemm_auto != 0 && (self.gemm_tile != "64" || self.gemm_bk != 32 || self.gemm_stages != 3) {
            return Err(HerbertError::Config("gemm.auto selects gemm.tile/bk/stages itself (manual tiles: gemm.auto=0)".into()));
        }
        let mut tile = match self.gemm_tile.as_str() {
            "32" => Ok(Tile::T32X64),
            "64" => Ok(Tile::T64),
            "64x32" => Ok(Tile::T64X32),
            "32w1x1" => Ok(Tile::T32X64_W1X1_K64_S2),
            "64w4x1" => Ok(Tile::T64_W4X1),
            "64w1x4" => Ok(Tile::T64_W1X4),
            "64x128" => Ok(Tile::T64X128),
            "64x128w4x2" => Ok(Tile::T64X128_W4X2),
            "64x128w2x2" => Ok(Tile::T64X128_W2X2),
            "64x256" => Ok(Tile::T64X256_S2),
            "128" => Ok(Tile::T128),
            "128w4x4" => Ok(Tile::T128_W4X4_K64_S2),
            "128w4x2" => Ok(Tile::T128_W4X2_K64_S2),
            t => Err(HerbertError::Config(format!("--tune gemm.tile={t}: 32, 32w1x1, 64, 64x32, 64w4x1, 64w1x4, 64x128, 64x128w4x2, 64x128w2x2, 64x256, 128, 128w4x4 or 128w4x2"))),
        }?;
        if !matches!(self.gemm_bk, 32 | 64) {
            return Err(HerbertError::Config(format!("--tune gemm.bk={}: 32 or 64", self.gemm_bk)));
        }
        if !matches!(self.gemm_stages, 2 | 3) {
            return Err(HerbertError::Config(format!("--tune gemm.stages={}: 2 or 3", self.gemm_stages)));
        }
        tile.bk = self.gemm_bk;
        tile.stages = self.gemm_stages;
        tile.swizzle = self.gemm_swizzle;
        tile.schedule = self.gemm_schedule;
        tile.pipeline = self.gemm_pipeline;
        tile.ldmatrix = self.gemm_ldmatrix;
        tile.async_copy = self.gemm_async_copy;
        tile.wmma = self.gemm_wmma;
        tile.f16 = if self.gemm_f16 { 2 } else { 0 };
        Ok(tile)
    }

    /// Exact M=128/256 overrides of the 4B down residual projection only.
    pub fn down_short_tiles(&self) -> Result<Option<(Tile, Tile)>> {
        let tile = match self.gemm_down_short.as_str() {
            "same" => return Ok(None),
            "64x32-split" => return Ok(Some((Tile::T64X32_K64_S2, Tile::T64X32))),
            "32x64-k32-s3" => Tile::T32X64,
            "32x64-k64-s2" => Tile::T32X64_K64_S2,
            "64x32-k32-s3" => Tile::T64X32,
            "64x32-k64-s2" => Tile::T64X32_K64_S2,
            "32x64w1x1-k64-s2" => Tile::T32X64_W1X1_K64_S2,
            _ => return Err(HerbertError::Config("gemm.down_short: same, 64x32-split, 32x64-k32-s3, 32x64-k64-s2, 64x32-k32-s3, 64x32-k64-s2 or 32x64w1x1-k64-s2".into())),
        };
        Ok(Some((tile, tile)))
    }

    pub fn summary(&self) -> String {
        format!(
            "matvec.rows={} matvec.warps={} matvec.depth={} matvec.cache={} sdpa.split={} act={} graph={} prefill.block={} gemm.tile={} gemm.bk={} gemm.stages={} gemm.auto={} gemm.carveout={} gemm.layout={} gemm.schedule={} gemm.pipeline={} gemm.down_short={} gemm.ldmatrix={} gemm.copy={} gemm.wmma={} gemm.f16={} sdpa.bk={} sdpa.softmax={} rec.tc={}",
            self.matvec.rows,
            self.matvec.warps,
            self.matvec.depth,
            cache_name(self.matvec.cache),
            self.sdpa_split,
            self.act.tag(),
            self.graph,
            self.prefill_block,
            self.gemm_tile,
            self.gemm_bk,
            self.gemm_stages,
            self.gemm_auto,
            self.gemm_carveout.map(|v| v.to_string()).unwrap_or_else(|| "driver".into()),
            if self.gemm_swizzle { "xor" } else { "pad" },
            self.gemm_schedule,
            self.gemm_pipeline,
            self.gemm_down_short,
            self.gemm_ldmatrix as u32,
            if self.gemm_async_copy { "async" } else { "sync" },
            self.gemm_wmma as u32,
            self.gemm_f16 as u32,
            self.sdpa_bk,
            if self.sdpa_smem_softmax { "shared" } else { "reg" },
            self.rec_tc
        )
    }
}
