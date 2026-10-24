//! herbert-cuda-probe [--device N] [--only selftest|bw|mma|matvec|gemm] [--m M,M…] [--emit DIR]
//!
//! Prints what the device declares, then measures with emitted kernels: the
//! selftest, the read bandwidth, the decode matvec and the prefill GEMM at
//! the shapes of Qwen3.5-0.8B and 4B (see `cuda/probe.rs`). `--emit DIR`: no
//! device, every kernel of the probe written as `DIR/<name>.ptx` for `ptxas`.

use herbert_cuda::cuda::gemm::Tile;
use herbert_cuda::cuda::gpu::Gpu;
use herbert_cuda::cuda::kernels::{Matvec, Out};
use herbert_cuda::cuda::probe;
use herbert_cuda::ptx::Ld;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut device = 0i32;
    let mut only: Option<String> = None;
    let mut ms: Vec<u32> = vec![128, 1024];
    let mut model = None;
    let mut shape = None;
    let mut rounds = 3;
    let mut rotate = false;
    let mut out = Out::Bf16;
    let mut selected_tiles: Option<Vec<String>> = None;
    let mut schedules = vec![0u32];
    let mut pipelines = vec![0u32];
    let mut emit = None;
    let mut layout = "pad".to_owned();
    let mut frags = vec![true];
    let mut copies = vec![true];
    let mut mmas = vec![false];
    let mut f16s = vec![0u32];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--device" => device = args.next().and_then(|s| s.parse().ok()).expect("--device N"),
            "--only" => only = args.next(),
            "--model" => model = args.next(),
            "--shape" => shape = args.next(),
            "--rounds" => rounds = args.next().expect("--rounds N").parse().expect("rounds"),
            "--gemm-layout" => { layout = args.next().expect("--gemm-layout pad|xor|both"); assert!(["pad", "xor", "both"].contains(&layout.as_str())); },
            "--gemm-tiles" => selected_tiles = Some(args.next().expect("--gemm-tiles TAG,TAG…").split(',').map(str::to_owned).collect()),
            "--gemm-schedules" => schedules = args.next().expect("--gemm-schedules 0,1,…,5").split(',').map(|v| { let n = v.parse::<u32>().expect("schedule"); assert!(n <= 5); n }).collect(),
            "--gemm-pipelines" => pipelines = args.next().expect("--gemm-pipelines 0,1,…,3").split(',').map(|v| { let n = v.parse::<u32>().expect("pipeline"); assert!(n <= 3); n }).collect(),
            "--gemm-frag" => frags = match args.next().as_deref() { Some("ldmatrix") => vec![true], Some("lds") => vec![false], Some("both") => vec![true, false], _ => panic!("--gemm-frag ldmatrix|lds|both") },
            "--gemm-copy" => copies = match args.next().as_deref() { Some("async") => vec![true], Some("sync") => vec![false], Some("both") => vec![true, false], _ => panic!("--gemm-copy async|sync|both") },
            "--gemm-mma" => mmas = match args.next().as_deref() { Some("mma") => vec![false], Some("wmma") => vec![true], Some("both") => vec![false, true], _ => panic!("--gemm-mma mma|wmma|both") },
            "--gemm-f16" => f16s = args.next().expect("--gemm-f16 0,1,2,3").split(',').map(|v| { let n = v.parse::<u32>().expect("f16 mode"); assert!(n <= 3); n }).collect(),
            "--rotate-weights" => rotate = true,
            "--gemm-out" => out = match args.next().as_deref() { Some("bf16") => Out::Bf16, Some("res") => Out::Residual, _ => panic!("--gemm-out bf16|res") },
            "--m" => ms = args.next().expect("--m M,M…").split(',').map(|s| s.parse().expect("M")).collect(),
            "--emit" => emit = Some(PathBuf::from(args.next().expect("--emit DIR"))),
            other => {
                eprintln!("unknown option {other}\nusage: herbert-cuda-probe [--device N] [--only selftest|bw|mma|matvec|gemm] [--m M,M…] [--emit DIR]");
                std::process::exit(2);
            }
        }
    }
    let base = [Tile::T32X64, Tile::T64, Tile::T64_K32_S2, Tile::T64X128, Tile::T64_K64_S2, Tile::T128,
                Tile::T64_K64_S3, Tile::T64X128_S2, Tile::T64X128_K64_S2, Tile::T64X128_K64_S3,
                Tile::T64_W4X1, Tile::T64_W1X4, Tile::T64X128_W4X2, Tile::T32X64_K64_S2, Tile::T64X128_W2X2,
                Tile::T64_W1X4_K64_S2, Tile::T64X128_W2X2_S2, Tile::T128_K64_S2, Tile::T128_W4X4_K64_S2, Tile::T128_W4X2_K64_S2, Tile::T64X256_S2];
    let mut available = base.to_vec();
    available.extend([Tile::T64X32, Tile::T64X32_K64_S2, Tile::T32X64_W1X1_K64_S2]);
    let chosen = if let Some(names) = selected_tiles {
        names.iter().map(|name| *available.iter().find(|t| t.tag() == *name).unwrap_or_else(|| panic!("unknown tile {name}; choose {:?}", available.iter().map(Tile::tag).collect::<Vec<_>>()))).collect::<Vec<_>>()
    } else { base.to_vec() };
    let mut tiles = Vec::new();
    for swizzle in [false, true] {
        if layout == "both" || (layout == "xor") == swizzle {
            for &schedule in &schedules { for &pipeline in &pipelines {
                for &ldmatrix in &frags { for &async_copy in &copies { for &wmma in &mmas {
                    if !async_copy && pipeline != 0 { continue; }
                    // wmma loads its own fragments: no ldmatrix switch, pad only.
                    if wmma && (swizzle || schedule != 0 || !ldmatrix || !matches!(pipeline, 0 | 3)) { continue; }
                    for &f16 in &f16s {
                        if f16 != 0 && (schedule != 0 || wmma) { continue; }
                        tiles.extend(chosen.iter().map(|t| Tile { swizzle, schedule, pipeline, ldmatrix, async_copy, wmma, f16, ..*t }));
                    }
                } } }
            } }
        }
    }
    if let Some(dir) = emit {
        let all = if only.as_deref() == Some("gemm") {
            probe::gemm_kernels(&tiles, model.as_deref(), shape.as_deref(), out)
        } else { probe::all_kernels() };
        assert!(!all.is_empty(), "filters select no compatible kernels");
        for e in &all { e.dump(&dir).expect("write"); }
        println!("{} kernels emitted into {}", all.len(), dir.display());
        return;
    }
    let run = || -> herbert_cuda::core::error::Result<()> {
        for (i, name) in Gpu::list()? {
            println!("device {i}: {name}");
        }
        let gpu = Gpu::open(device)?;
        probe::report(&gpu)?;
        let want = |s: &str| only.as_deref().is_none_or(|o| o == s);
        if want("selftest") {
            probe::selftest(&gpu)?;
        }
        if want("bw") {
            probe::bandwidth(&gpu)?;
        }
        if want("mma") {
            probe::mma(&gpu)?;
        }
        if want("matvec") {
            let mut v = Vec::new();
            for rows in [1, 2, 4] {
                v.push(Matvec { rows, ..Matvec::DEFAULT });
            }
            v.push(Matvec { cache: Ld::NcNoL1, ..Matvec::DEFAULT });
            probe::matvec(&gpu, &v)?;
        }
        if want("gemm") {
            probe::gemm(&gpu, &ms, &tiles,
                probe::GemmBench { model: model.as_deref(), shape: shape.as_deref(), rounds, rotate, out })?;
        }
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
