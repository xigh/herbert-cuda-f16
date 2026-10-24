//! The checkpoint read into host tensors — Qwen3.5, text only.
//!
//! What enters: the model directory (`config.json`, the `*.safetensors`
//! shards). What comes out: the `Config` and every tensor the engine uses,
//! under a normalized name (`embed_tokens`, `norm`, `lm_head`,
//! `<layer>.<field>` such as `3.self_attn.q_proj.weight`), in bf16 row-major —
//! the checkpoint's bytes — or in f32 for the two DeltaNet vectors the CPU
//! engine keeps in f32 (`A_log`, `dt_bias`). The multi-token-prediction head
//! (`mtp.*`) and the vision tower are skipped. The names are those of
//! herbert-jit-bf16's `model/loader.rs`.

use crate::core::bf16::{bf16_to_f32, f32_to_bf16, Bf16};
use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

pub struct Tensor {
    pub shape: Vec<usize>,
    pub bf16: Vec<Bf16>,
    pub f32: Vec<f32>,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub config: Config,
    tensors: HashMap<String, Tensor>,
}

/// A Hugging Face cache directory points at a snapshot; a plain directory is
/// used as is.
pub fn resolve_model_dir(model_dir: &Path) -> Result<PathBuf> {
    if model_dir.join("config.json").exists() {
        return Ok(model_dir.to_path_buf());
    }
    let snapshots = model_dir.join("snapshots");
    if !snapshots.is_dir() {
        return Ok(model_dir.to_path_buf());
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snapshots)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    candidates.sort();
    candidates.pop().ok_or_else(|| HerbertError::ModelLoad(format!("no snapshot under {}", snapshots.display())))
}

fn load_bf16(view: &TensorView<'_>, name: &str) -> Result<Vec<Bf16>> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::BF16 => Ok(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()),
        safetensors::Dtype::F16 => {
            Ok(bytes.chunks_exact(2).map(|b| f32_to_bf16(half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())).collect())
        }
        safetensors::Dtype::F32 => Ok(bytes.chunks_exact(4).map(|b| f32_to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))).collect()),
        d => Err(HerbertError::ModelLoad(format!("unsupported dtype {d:?} for {name}"))),
    }
}

fn load_f32(view: &TensorView<'_>, name: &str) -> Result<Vec<f32>> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()),
        safetensors::Dtype::F16 => Ok(bytes.chunks_exact(2).map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()).collect()),
        safetensors::Dtype::BF16 => Ok(bytes.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()),
        d => Err(HerbertError::ModelLoad(format!("unsupported dtype {d:?} for {name}"))),
    }
}

/// The normalized name of a tensor, `None` when the engine does not use it;
/// and whether it is kept in f32.
fn normalize(name: &str) -> Option<(String, bool)> {
    let rest = name
        .strip_prefix("language_model.model.")
        .or_else(|| name.strip_prefix("model.language_model."))
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);
    match rest {
        "embed_tokens.weight" => return Some(("embed_tokens".into(), false)),
        "norm.weight" => return Some(("norm".into(), false)),
        "lm_head.weight" => return Some(("lm_head".into(), false)),
        _ => {}
    }
    if name == "lm_head.weight" {
        return Some(("lm_head".into(), false));
    }
    let rest = rest.strip_prefix("layers.")?;
    let (index, field) = rest.split_once('.')?;
    let layer: usize = index.parse().ok()?;
    let f32_kept = matches!(field, "linear_attn.A_log" | "linear_attn.dt_bias");
    let used = matches!(
        field,
        "input_layernorm.weight"
            | "post_attention_layernorm.weight"
            | "self_attn.q_proj.weight"
            | "self_attn.k_proj.weight"
            | "self_attn.v_proj.weight"
            | "self_attn.o_proj.weight"
            | "self_attn.q_norm.weight"
            | "self_attn.k_norm.weight"
            | "mlp.gate_proj.weight"
            | "mlp.up_proj.weight"
            | "mlp.down_proj.weight"
            | "linear_attn.in_proj_qkv.weight"
            | "linear_attn.in_proj_z.weight"
            | "linear_attn.in_proj_a.weight"
            | "linear_attn.in_proj_b.weight"
            | "linear_attn.out_proj.weight"
            | "linear_attn.conv1d.weight"
            | "linear_attn.A_log"
            | "linear_attn.dt_bias"
            | "linear_attn.norm.weight"
    );
    used.then(|| (format!("{layer}.{field}"), f32_kept))
}

impl Checkpoint {
    pub fn load(model_dir: &Path) -> Result<Checkpoint> {
        let dir = resolve_model_dir(model_dir)?;
        let config = Config::from_file(&dir.join("config.json"))?;
        let index_path = dir.join("model.safetensors.index.json");
        let shards: Vec<String> = if index_path.exists() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: HashMap<String, String>,
            }
            let index: Index = serde_json::from_slice(&std::fs::read(&index_path)?)?;
            let set: HashSet<String> = index.weight_map.values().cloned().collect();
            let mut v: Vec<String> = set.into_iter().collect();
            v.sort();
            v
        } else {
            vec!["model.safetensors".to_string()]
        };
        let mut tensors = HashMap::new();
        let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
        for shard in &shards {
            let file = std::fs::File::open(dir.join(shard)).map_err(|e| HerbertError::ModelLoad(format!("{shard}: {e}")))?;
            // SAFETY: read-only mapping of a file nobody modifies meanwhile.
            let mapped = unsafe { Mmap::map(&file) }.map_err(|e| HerbertError::ModelLoad(format!("{shard}: {e}")))?;
            let st = SafeTensors::deserialize(&mapped[..])?;
            for (name, view) in st.tensors() {
                let Some((key, keep_f32)) = normalize(&name) else {
                    *skipped.entry(name.split('.').take(2).collect::<Vec<_>>().join(".")).or_default() += 1;
                    continue;
                };
                let t = if keep_f32 {
                    Tensor { shape: view.shape().to_vec(), bf16: Vec::new(), f32: load_f32(&view, &name)? }
                } else {
                    Tensor { shape: view.shape().to_vec(), bf16: load_bf16(&view, &name)?, f32: Vec::new() }
                };
                tensors.insert(key, t);
            }
        }
        if !skipped.is_empty() {
            let list: Vec<String> = skipped.iter().map(|(p, n)| format!("{p}.* ×{n}")).collect();
            eprintln!("[loader] skipped: {}", list.join(", "));
        }
        Ok(Checkpoint { dir, config, tensors })
    }

    /// Every tensor kept by the loader, by normalized name (for the f16 range check of herbert-cuda-absmax).
    pub fn tensors(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.tensors.iter()
    }

    /// Take a tensor out (the host copy is freed once uploaded).
    pub fn take(&mut self, key: &str) -> Result<Tensor> {
        self.tensors.remove(key).ok_or_else(|| HerbertError::ModelLoad(format!("missing tensor {key}")))
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// A layer tensor in bf16, checked against `len` elements.
    pub fn bf16(&mut self, key: &str, len: usize) -> Result<Vec<Bf16>> {
        let t = self.take(key)?;
        if t.bf16.len() != len {
            return Err(HerbertError::ModelLoad(format!("{key}: {} elements, {len} expected ({:?})", t.bf16.len(), t.shape)));
        }
        Ok(t.bf16)
    }

    /// A layer tensor converted from bf16 to IEEE f16 bits (round to nearest even), checked against
    /// `len` elements: what herbert-cuda-f16 uploads.
    pub fn h16(&mut self, key: &str, len: usize) -> Result<Vec<u16>> {
        Ok(self.bf16(key, len)?.into_iter().map(|b| half::f16::from_f32(bf16_to_f32(b)).to_bits()).collect())
    }

    pub fn f32(&mut self, key: &str, len: usize) -> Result<Vec<f32>> {
        let t = self.take(key)?;
        if t.f32.len() != len {
            return Err(HerbertError::ModelLoad(format!("{key}: {} elements, {len} expected ({:?})", t.f32.len(), t.shape)));
        }
        Ok(t.f32)
    }
}
