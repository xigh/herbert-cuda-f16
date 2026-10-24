//! The model's `config.json`, read into the numbers the engine needs.
//!
//! What enters: the JSON of a Qwen3.5 text checkpoint (`model_type` `qwen3_5`
//! or `qwen3_5_text`, the fields possibly nested under `text_config`).
//! What comes out: one flat [`Config`]. Any other model type is refused by
//! name: the emitters below are compiled for this architecture and nothing
//! else.
//!
//! What Qwen3.5-4B declares, and what each number drives:
//! - `hidden_size` 2560, `num_hidden_layers` 32, `intermediate_size` 9216;
//! - `num_attention_heads` 16, `num_key_value_heads` 4, `head_dim` 256:
//!   the attention layers, grouped-query (four query heads per KV head);
//! - `layer_types` or `full_attention_interval` 4: which layers are full
//!   attention (every fourth) and which are GatedDeltaNet;
//! - `linear_num_key_heads` 16, `linear_num_value_heads` 32,
//!   `linear_key_head_dim` 128, `linear_value_head_dim` 128,
//!   `linear_conv_kernel_dim` 4: the GatedDeltaNet mixer;
//! - `rope_parameters.partial_rotary_factor` 0.25 on `head_dim` 256: RoPE
//!   rotates 64 dimensions of each head, `rope_theta` 1e7;
//! - `attn_output_gate`: `q_proj` carries `[Q; gate]` and the attention output
//!   is multiplied by `sigmoid(gate)`;
//! - `rms_norm_eps`, and the norms are zero-centred: `(1 + w) · rms(x)`.

use crate::core::error::{HerbertError, Result};
use serde_json::Value;
use std::path::Path;

/// How a norm weight is applied. Qwen3.5 stores every norm weight zero-centred
/// (`GemmaRMSNorm`): the kernel computes `(1 + w) · rms(x)`, not `w · rms(x)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    RMSNorm,
    GemmaRMSNorm,
}

/// What a layer's mixer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub norm_type: NormType,
    pub rope_theta: f32,
    /// Dimensions of each head rotated by RoPE: `head_dim · partial_rotary_factor`.
    pub rotary_ndims: usize,
    /// Positions the RoPE tables cover: the model's context ceiling.
    pub max_position_embeddings: usize,
    /// The language head is the embedding table.
    pub tie_word_embeddings: bool,
    /// The activation of the MLP and of the DeltaNet conv; `silu` is what the
    /// kernels emit.
    pub hidden_act: String,
    /// `q_proj` is `[Q; gate]` and the attention output is gated by `sigmoid(gate)`.
    pub attn_output_gate: bool,
    /// The mixer of each layer, in order.
    pub layer_types: Vec<LayerType>,
    // ── the GatedDeltaNet mixer ──
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

impl Config {
    /// Reads `config.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Parses the JSON text. The text fields may sit at the top level or under
    /// `text_config` (the multimodal packaging of Qwen3.5); the top level wins
    /// for `tie_word_embeddings` when both are present, as Hugging Face does.
    pub fn from_json(content: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(content).map_err(|e| HerbertError::Config(format!("config.json: {e}")))?;
        let text = root.get("text_config").unwrap_or(&root);
        let field = |key: &str| text.get(key).or_else(|| root.get(key));
        let missing = |key: &str| HerbertError::Config(format!("config.json: missing {key}"));
        let usize_of = |key: &str| -> Result<usize> {
            field(key).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| missing(key))
        };
        let model_type = field("model_type").and_then(Value::as_str).ok_or_else(|| missing("model_type"))?.to_string();
        if model_type != "qwen3_5" && model_type != "qwen3_5_text" {
            return Err(HerbertError::Config(format!(
                "model_type {model_type:?}: this engine serves qwen3_5 (text) only"
            )));
        }
        let head_dim = usize_of("head_dim")?;
        let num_layers = usize_of("num_hidden_layers")?;
        // RoPE: `rope_parameters` on Qwen3.5, the flat keys on older packagings.
        let rope = field("rope_parameters");
        let rope_field = |key: &str| rope.and_then(|r| r.get(key)).or_else(|| field(key)).and_then(Value::as_f64);
        let rope_theta = rope_field("rope_theta").ok_or_else(|| missing("rope_theta"))? as f32;
        let partial_rotary_factor = rope_field("partial_rotary_factor").unwrap_or(1.0);
        let rotary_ndims = (head_dim as f64 * partial_rotary_factor).round() as usize;
        // Which layer is which mixer: the explicit list, or every `interval`-th layer is attention.
        let layer_types: Vec<LayerType> = match field("layer_types").and_then(Value::as_array) {
            Some(list) => list
                .iter()
                .map(|v| match v.as_str() {
                    Some("full_attention") => Ok(LayerType::FullAttention),
                    Some("linear_attention") => Ok(LayerType::LinearAttention),
                    other => Err(HerbertError::Config(format!("config.json: layer_types holds {other:?}"))),
                })
                .collect::<Result<_>>()?,
            None => {
                let interval = usize_of("full_attention_interval").unwrap_or(1).max(1);
                (0..num_layers)
                    .map(|l| if (l + 1) % interval == 0 { LayerType::FullAttention } else { LayerType::LinearAttention })
                    .collect()
            }
        };
        let has_linear = layer_types.contains(&LayerType::LinearAttention);
        let linear = |key: &str| -> Result<usize> { if has_linear { usize_of(key) } else { Ok(0) } };
        let config = Config {
            model_type,
            vocab_size: usize_of("vocab_size")?,
            hidden_size: usize_of("hidden_size")?,
            num_layers,
            num_attention_heads: usize_of("num_attention_heads")?,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim,
            intermediate_size: usize_of("intermediate_size")?,
            rms_norm_eps: field("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            norm_type: NormType::GemmaRMSNorm,
            rope_theta,
            rotary_ndims,
            max_position_embeddings: usize_of("max_position_embeddings")?,
            tie_word_embeddings: root
                .get("tie_word_embeddings")
                .or_else(|| text.get("tie_word_embeddings"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            hidden_act: field("hidden_act").and_then(Value::as_str).unwrap_or("silu").to_string(),
            attn_output_gate: field("attn_output_gate").and_then(Value::as_bool).unwrap_or(false),
            layer_types,
            linear_num_key_heads: linear("linear_num_key_heads")?,
            linear_num_value_heads: linear("linear_num_value_heads")?,
            linear_key_head_dim: linear("linear_key_head_dim")?,
            linear_value_head_dim: linear("linear_value_head_dim")?,
            linear_conv_kernel_dim: linear("linear_conv_kernel_dim")?,
        };
        config.validate()?;
        Ok(config)
    }

    /// Is layer `l` a GatedDeltaNet layer?
    pub fn is_deltanet_layer(&self, l: usize) -> bool {
        self.layer_types.get(l) == Some(&LayerType::LinearAttention)
    }

    /// Rows of `q_proj` without the gate: `num_attention_heads · head_dim`.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Rows of `k_proj` and `v_proj`: `num_key_value_heads · head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    fn validate(&self) -> Result<()> {
        let refuse = |what: &str| Err(HerbertError::Config(format!("config.json: {what}")));
        if self.hidden_size == 0 || self.num_layers == 0 || self.vocab_size == 0 || self.intermediate_size == 0 {
            return refuse("a dimension is zero");
        }
        if self.layer_types.len() != self.num_layers {
            return refuse("layer_types does not have one entry per layer");
        }
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 || !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return refuse("num_attention_heads must be a positive multiple of num_key_value_heads");
        }
        if self.head_dim == 0 || self.rotary_ndims == 0 || self.rotary_ndims > self.head_dim {
            return refuse("rotary_ndims must lie in 1..=head_dim");
        }
        if self.layer_types.contains(&LayerType::LinearAttention)
            && (self.linear_num_key_heads == 0
                || self.linear_num_value_heads == 0
                || !self.linear_num_value_heads.is_multiple_of(self.linear_num_key_heads)
                || self.linear_key_head_dim == 0
                || self.linear_value_head_dim == 0
                || self.linear_conv_kernel_dim == 0)
        {
            return refuse("the linear_* fields of the GatedDeltaNet mixer are missing or inconsistent");
        }
        Ok(())
    }
}
