//! Single-request benchmark on shared token IDs, excluding model loading,
//! tokenization and detokenization. Fixed greedy output length, including EOS.
use herbert_cuda::chat::{template, Tokens};
use herbert_cuda::cuda::engine::{Engine, Options};
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Instant};

#[derive(Serialize, Deserialize)]
struct Case { name: String, ids: Vec<u32> }

#[derive(Deserialize)]
struct Variant { name: String, tune: Vec<String> }

#[derive(Serialize)]
struct Sample {
    case: String, prompt_tokens: usize, output_tokens: usize, round: usize,
    first_token_seconds: f64, latency_seconds: f64, output_ids: Vec<u32>,
}

fn main() {
    if let Err(e) = run() { eprintln!("error: {e}"); std::process::exit(1); }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().ok_or("MODEL_DIR required")?);
    let mut cases_path = None;
    let mut out_path = None;
    let mut prepare = false;
    let mut variants_path = None;
    let mut rounds = 20usize;
    let mut warmup = 2usize;
    let mut outputs = vec![1usize];
    let mut prompts = PathBuf::from("bench/prompts");
    let mut labels = "128,256,512,1024,2048,4000".to_string();
    let mut o = Options { device: 0, capacity: 4096, dump_ptx: None, tune: Vec::new(), profile: false, logits: false };
    while let Some(key) = args.next() {
        if key == "--prepare" { prepare = true; continue; }
        if key == "--profile" { o.profile = true; continue; }
        let value = args.next().ok_or(format!("{key} requires a value"))?;
        match key.as_str() {
            "--variants-file" => variants_path = Some(PathBuf::from(value)),
            "--cases" => cases_path = Some(PathBuf::from(value)),
            "--out" => out_path = Some(PathBuf::from(value)),
            "--prompts" => prompts = PathBuf::from(value),
            "--labels" => labels = value,
            "--rounds" => rounds = value.parse()?,
            "--warmup" => warmup = value.parse()?,
            "--outputs" => outputs = value.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--ctx" => o.capacity = value.parse()?,
            "--device" => o.device = value.parse()?,
            "--tune" => o.tune.push(value),
            "--dump-ptx" => o.dump_ptx = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    let cases_path = cases_path.ok_or("--cases FILE required")?;
    if prepare {
        let tokens = Tokens::load(&model)?;
        let mut cases = Vec::new();
        for label in labels.split(',') {
            let name = if label.starts_with('p') { label.to_string() } else { format!("pp{label}") };
            let text = std::fs::read_to_string(prompts.join(format!("{name}.txt")))?;
            let ids = tokens.encode(&(template::user_turn(&text) + &template::assistant_header(name == "p3-long")))?;
            if !label.starts_with('p') {
                let expected = match label { "512" => 513, "4000" => 4001, _ => label.parse()? };
                if ids.len() != expected { return Err(format!("{name}: {} tokens != {expected}", ids.len()).into()); }
            }
            cases.push(Case { name, ids });
        }
        serde_json::to_writer_pretty(std::fs::File::create(cases_path)?, &cases)?;
        return Ok(());
    }
    let cases: Vec<Case> = serde_json::from_reader(std::fs::File::open(&cases_path)?)?;
    if cases.is_empty() || rounds == 0 || outputs.is_empty() || outputs.contains(&0) { return Err("empty cases or nonpositive rounds/outputs".into()); }
    for c in &cases {
        if c.ids.is_empty() || c.ids.len() + outputs.iter().max().unwrap() > o.capacity { return Err(format!("{} exceeds context {}", c.name, o.capacity).into()); }
    }
    let mut engine = Engine::load(&model, &o)?;
    if let Some(path) = variants_path {
        let variants: Vec<Variant> = serde_json::from_reader(std::fs::File::open(path)?)?;
        if variants.is_empty() { return Err("empty variants".into()); }
        let dir = out_path.ok_or("--out DIR required for variants")?;
        std::fs::create_dir_all(&dir)?;
        let model_name = if engine.config().hidden_size == 2560 { "4B" } else if engine.config().hidden_size == 1024 { "0.8B" } else { return Err("variant benchmark supports 0.8B/4B".into()); };
        let mut names = std::collections::HashSet::new();
        for v in &variants {
            if v.name.is_empty() || !v.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') || !names.insert(&v.name) {
                return Err("invalid or duplicate variant name".into());
            }
            engine.retune_gemm(&v.tune)?;
        }
        let mut samples: Vec<Vec<Sample>> = (0..variants.len()).map(|_| Vec::new()).collect();
        let mut first_calls: Vec<Vec<Sample>> = (0..variants.len()).map(|_| Vec::new()).collect();
        let mut resolved = vec![String::new(); variants.len()];
        for round in 0..warmup + rounds {
            eprintln!("[bench-variants] round {}/{} ({})", round + 1, warmup + rounds, if round < warmup { "warmup" } else { "measured" });
            for vi in 0..variants.len() {
                let vi = if round % 2 == 0 { vi } else { variants.len() - 1 - vi };
                let variant = &variants[vi];
                engine.retune_gemm(&variant.tune)?;
                resolved[vi] = engine.tuning().summary();
                for ci in 0..cases.len() {
                    let c = &cases[(ci + round) % cases.len()];
                    for &count in &outputs {
                        let started = Instant::now();
                        engine.reset()?;
                        let first = engine.prefill(&c.ids)?;
                        let first_token_seconds = started.elapsed().as_secs_f64();
                        let mut output_ids = vec![first];
                        engine.generate(first, count - 1, &mut |id, _| { output_ids.push(id); Ok(Some(id)) })?;
                        let sample = Sample { case: c.name.clone(), prompt_tokens: c.ids.len(), output_tokens: count,
                            round: round.saturating_sub(warmup), first_token_seconds,
                            latency_seconds: started.elapsed().as_secs_f64(), output_ids };
                        if round >= warmup { samples[vi].push(sample); } else if round == 0 { first_calls[vi].push(sample); }
                    }
                }
            }
        }
        for (vi, variant) in variants.iter().enumerate() {
            let result = serde_json::json!({ "schema": 1, "backend": "herbert-cuda", "model": model,
                "timing": "token_ids_to_output_ids_greedy_ignore_eos", "request_initialization_included": true,
                "context": o.capacity, "tuning": variant.tune, "resolved_tuning": resolved[vi],
                "rounds": rounds, "warmup": warmup, "reuse_model": true, "variant_alternation": true,
                "variant": variant.name, "load_seconds": engine.load_seconds, "mount_seconds": engine.mount_seconds,
                "module_load_seconds": engine.module_load_seconds,
                "first_calls": first_calls[vi], "samples": samples[vi] });
            serde_json::to_writer_pretty(std::fs::File::create(dir.join(format!("herbert-{model_name}-{}-r1.json", variant.name)))?, &result)?;
        }
        return Ok(());
    }
    let mut samples = Vec::new();
    let mut first_calls = Vec::new();
    for round in 0..warmup + rounds {
        eprintln!("[bench] round {}/{} ({})", round + 1, warmup + rounds, if round < warmup { "warmup" } else { "measured" });
        for ci in 0..cases.len() {
            let c = &cases[(ci + round) % cases.len()];
            for &count in &outputs {
                let started = Instant::now();
                // Include initialization of a fresh request, as LLM.generate
                // does for vLLM; model loading and tokenization stay outside.
                engine.reset()?;
                let first = engine.prefill(&c.ids)?;
                let first_token_seconds = started.elapsed().as_secs_f64();
                let mut output_ids = Vec::with_capacity(count);
                output_ids.push(first);
                engine.generate(first, count - 1, &mut |argmax, _| { output_ids.push(argmax); Ok(Some(argmax)) })?;
                let latency_seconds = started.elapsed().as_secs_f64();
                let sample = Sample { case: c.name.clone(), prompt_tokens: c.ids.len(), output_tokens: count, round: round.saturating_sub(warmup), first_token_seconds, latency_seconds, output_ids };
                if round >= warmup { samples.push(sample); } else if round == 0 { first_calls.push(sample); }
            }
        }
    }
    if let Some(report) = engine.profile_report() { eprint!("{report}"); }
    let result = serde_json::json!({ "schema": 1, "backend": "herbert-cuda", "model": model,
        "timing": "token_ids_to_output_ids_greedy_ignore_eos", "request_initialization_included": true, "context": o.capacity,
        "tuning": o.tune, "resolved_tuning": engine.tuning().summary(), "rounds": rounds, "warmup": warmup, "load_seconds": engine.load_seconds,
        "mount_seconds": engine.mount_seconds, "module_load_seconds": engine.module_load_seconds,
        "first_calls": first_calls, "samples": samples });
    serde_json::to_writer_pretty(std::fs::File::create(out_path.ok_or("--out FILE required")?)?, &result)?;
    Ok(())
}
