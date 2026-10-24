//! The f16 range of Qwen3.5, measured before any f16 kernel: the weights as loaded (bf16, the f32 ones
//! apart) and the activations of the bf16 engine's block prefill on real prompts, per layer and buffer.
//! For each: the largest magnitude, the elements above the f16 maximum (65 504), the nonzero ones below
//! the f16 normal range (2^-14) and below its smallest subnormal (2^-24). JSON on --out.
//! usage: herbert-cuda-absmax MODEL_DIR --out FILE.json [--prompts DIR] [--labels p1-short,…] [--tune K=V]…
use herbert_cuda::chat::{template, Tokens};
use herbert_cuda::cuda::engine::{AbsMax, Engine, Options};
use herbert_cuda::model::Checkpoint;
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() { eprintln!("error: {e}"); std::process::exit(1); }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().ok_or("MODEL_DIR required")?);
    let (mut out, mut prompts, mut labels) = (None, PathBuf::from("bench/prompts"), "p1-short,p2-explain,p3-long,p4-xlong".to_string());
    let mut o = Options { device: 0, capacity: 4096, dump_ptx: None, tune: Vec::new(), profile: false, logits: false };
    while let Some(key) = args.next() {
        let value = args.next().ok_or(format!("{key} requires a value"))?;
        match key.as_str() {
            "--out" => out = Some(PathBuf::from(value)),
            "--prompts" => prompts = PathBuf::from(value),
            "--labels" => labels = value,
            "--tune" => o.tune.push(value),
            "--device" => o.device = value.parse()?,
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    let out = out.ok_or("--out FILE required")?;
    // Weights.
    let ck = Checkpoint::load(&model)?;
    let mut weights = Vec::new();
    let mut names: Vec<_> = ck.tensors().collect();
    names.sort_by(|a, b| a.0.cmp(b.0));
    for (name, t) in names {
        let vals: Box<dyn Iterator<Item = f32>> = if t.bf16.is_empty() { Box::new(t.f32.iter().map(|x| x.abs())) } else { Box::new(t.bf16.iter().map(|&e| f32::from_bits((e as u32) << 16).abs())) };
        let (mut max, mut over, mut below_normal, mut below_sub, mut n) = (0f32, 0u64, 0u64, 0u64, 0u64);
        for x in vals {
            max = max.max(x); n += 1;
            if x > 65504.0 { over += 1; }
            if x != 0.0 && x < 6.103_515_6e-5 { below_normal += 1; }
            if x != 0.0 && x < 5.960_464_5e-8 { below_sub += 1; }
        }
        weights.push(serde_json::json!({ "name": name, "dtype": if t.bf16.is_empty() { "f32" } else { "bf16" }, "max": max,
            "over_f16": over, "below_normal": below_normal, "below_subnormal": below_sub, "count": n }));
    }
    drop(ck);
    // Activations.
    let tokens = Tokens::load(&model)?;
    let mut engine = Engine::load(&model, &o)?;
    let mut acts = Vec::new();
    for label in labels.split(',') {
        let text = std::fs::read_to_string(prompts.join(format!("{label}.txt")))?;
        let ids = tokens.encode(&(template::user_turn(&text) + &template::assistant_header(label == "p3-long")))?;
        engine.reset()?;
        let mut rec = AbsMax::default();
        let first = engine.prefill_absmax(&ids, &mut rec)?;
        eprintln!("[absmax] {label}: {} tokens, first token {first}", ids.len());
        for ((layer, buf), r) in &rec.rows {
            acts.push(serde_json::json!({ "prompt": label, "prompt_tokens": ids.len(), "first_token": first, "layer": layer, "buffer": buf,
                "max": r.max, "max_kernel": r.max_kernel, "over_f16": r.over_f16, "below_normal": r.below_normal,
                "below_subnormal": r.below_subnormal, "non_finite": r.non_finite, "count": r.count }));
        }
    }
    let result = serde_json::json!({ "model": model, "tuning": engine.tuning().summary(), "weights": weights, "activations": acts });
    serde_json::to_writer_pretty(std::fs::File::create(&out)?, &result)?;
    Ok(())
}
