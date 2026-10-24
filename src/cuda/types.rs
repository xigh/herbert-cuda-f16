//! What the kernels, the engine and the probe share: the layout of the
//! decode state, the element type of the DeltaNet's activations, and the
//! geometry of the two kinds of layer, built from the model's `Config`.

use crate::core::config::Config;
use crate::ptx::{Kernel, Ld, D, F};

/// The decode state (`state` buffer, `u32`): the token fed, its position (the
/// first row's in a prefill block), the argmax of the logits, the rows of the
/// block.
pub const STATE_TOKEN: u32 = 0;
pub const STATE_POS: u32 = 1;
pub const STATE_ARGMAX: u32 = 2;
pub const STATE_ROWS: u32 = 3;

/// The element of what the DeltaNet keeps between kernels and from token to
/// token (projection output, conv state, gates, recurrent state, output):
/// bf16 (vk-bf16's best build on the 4080, the gates then hold `g = log
/// decay` as vLLM does) or f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Act {
    Bf16,
    F32,
}

impl Act {
    pub fn bytes(self) -> u32 {
        match self {
            Act::Bf16 => 2,
            Act::F32 => 4,
        }
    }
    pub fn tag(self) -> &'static str {
        match self {
            Act::Bf16 => "bf16",
            Act::F32 => "f32",
        }
    }
    /// Element `i` of `base` (widened from bf16 exactly).
    pub fn ld(self, k: &mut Kernel, base: D, i: impl Into<crate::ptx::U>) -> F {
        let a = k.index(base, i, self.bytes());
        match self {
            Act::Bf16 => {
                let h = k.ld_b16(Ld::Ca, a, 0);
                k.h_to_f(h)
            }
            Act::F32 => k.ld_f32(Ld::Ca, a, 0),
        }
    }
    /// Store element `i` of `base` (rounded to nearest even for bf16).
    pub fn st(self, k: &mut Kernel, base: D, i: impl Into<crate::ptx::U>, v: F) {
        let a = k.index(base, i, self.bytes());
        match self {
            Act::Bf16 => {
                let h = k.f_to_h(v);
                k.st_b16(a, 0, h);
            }
            Act::F32 => k.st_f32(a, 0, v),
        }
    }
    /// What a store then a load gives.
    pub fn round(self, k: &mut Kernel, v: F) -> F {
        match self {
            Act::Bf16 => k.round_h(v),
            Act::F32 => v,
        }
    }
}

/// Geometry of an attention layer.
#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary: u32,
    /// Positions the KV cache holds.
    pub capacity: u32,
    pub eps: f32,
}

impl AttnShape {
    pub const QWEN35_0_8B: AttnShape = AttnShape { heads: 8, kv_heads: 2, head_dim: 256, rotary: 64, capacity: 4096, eps: 1e-6 };

    pub fn from_config(c: &Config, capacity: usize) -> AttnShape {
        AttnShape {
            heads: c.num_attention_heads as u32,
            kv_heads: c.num_key_value_heads as u32,
            head_dim: c.head_dim as u32,
            rotary: c.rotary_ndims as u32,
            capacity: capacity as u32,
            eps: c.rms_norm_eps,
        }
    }

    /// Width of `q_proj`'s output: Q and its gate, interleaved per head as the
    /// checkpoint stores them, `[Q_h (hd) | gate_h (hd)]`.
    pub fn q_width(&self) -> u32 {
        2 * self.heads * self.head_dim
    }
    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
    /// Width of the fused `[q | k | v]` projection.
    pub fn qkv_width(&self) -> u32 {
        self.q_width() + 2 * self.kv_width()
    }
    /// Width of the attention output, the input of `o_proj`.
    pub fn out_width(&self) -> u32 {
        self.heads * self.head_dim
    }
}

/// Geometry of a GatedDeltaNet layer. The fused projection writes
/// `[q (nk·dk) | k (nk·dk) | v (nv·dv) | z (nv·dv) | a (nv) | b (nv) | pad]`.
#[derive(Clone, Copy, Debug)]
pub struct DeltaShape {
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    pub conv: u32,
    pub eps: f32,
    /// Zero columns after `b`, so the projection's width is a multiple of the
    /// GEMM tile.
    pub pad: u32,
    pub act: Act,
}

impl DeltaShape {
    pub const QWEN35_0_8B: DeltaShape = DeltaShape { k_heads: 16, v_heads: 16, k_dim: 128, v_dim: 128, conv: 4, eps: 1e-6, pad: 0, act: Act::Bf16 };

    pub fn from_config(c: &Config, act: Act) -> DeltaShape {
        DeltaShape {
            k_heads: c.linear_num_key_heads as u32,
            v_heads: c.linear_num_value_heads as u32,
            k_dim: c.linear_key_head_dim as u32,
            v_dim: c.linear_value_head_dim as u32,
            conv: c.linear_conv_kernel_dim as u32,
            eps: c.rms_norm_eps,
            pad: 0,
            act,
        }
    }

    /// The same, the projection's width rounded up to a multiple of `m` (0 or 1: as is).
    pub fn padded(self, m: u32) -> DeltaShape {
        let raw = self.b_at() + self.v_heads;
        let pad = if m > 1 { raw.div_ceil(m) * m - raw } else { 0 };
        DeltaShape { pad, ..self }
    }

    pub fn key(&self) -> u32 {
        self.k_heads * self.k_dim
    }
    pub fn value(&self) -> u32 {
        self.v_heads * self.v_dim
    }
    /// Channels of the causal conv: `[q | k | v]`.
    pub fn conv_dim(&self) -> u32 {
        2 * self.key() + self.value()
    }
    pub fn z_at(&self) -> u32 {
        self.conv_dim()
    }
    pub fn a_at(&self) -> u32 {
        self.z_at() + self.value()
    }
    pub fn b_at(&self) -> u32 {
        self.a_at() + self.v_heads
    }
    /// Width of the fused projection, with its padding.
    pub fn proj_width(&self) -> u32 {
        self.b_at() + self.v_heads + self.pad
    }
    /// Elements of the recurrent state of the layer: `[nv][dk][dv]`.
    pub fn state_len(&self) -> u32 {
        self.v_heads * self.k_dim * self.v_dim
    }
}
