//! Choosing the next token from a row of logits.
//!
//! What enters: the greedy token the generated program already computed, and
//! the bf16 logits of the block. What comes out: the token to feed next.
//! At temperature zero the answer is the argmax and the logits are not even
//! read. Otherwise: widen to f32, divide by the temperature, keep the `top_k`
//! largest, keep the smallest set whose probability mass reaches `top_p`, drop
//! what is below `min_p` times the best, then draw one with a fixed-seed
//! xorshift generator, so that a run repeats.

/// The knobs of one conversation. `temperature == 0.0` is greedy.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    /// `0` = no top-k cut.
    pub top_k: usize,
    /// `1.0` = no top-p cut.
    pub top_p: f32,
    /// `0.0` = no min-p cut.
    pub min_p: f32,
    pub seed: u64,
}

impl Sampling {
    /// Deterministic: the argmax, every time.
    pub fn greedy() -> Self {
        Sampling { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed: 0 }
    }

    /// What Qwen recommends for the thinking mode.
    pub fn qwen_thinking(seed: u64) -> Self {
        Sampling { temperature: 0.6, top_k: 20, top_p: 0.95, min_p: 0.0, seed }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

pub struct Sampler {
    config: Sampling,
    state: u64,
    /// `(logit, token)` of the whole vocabulary, reused at every draw.
    scratch: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(config: Sampling) -> Self {
        Sampler { config, state: config.seed | 1, scratch: Vec::new() }
    }

    pub fn config(&self) -> Sampling {
        self.config
    }

    /// The next token: `argmax` as is when greedy, a draw otherwise.
    pub fn choose(&mut self, argmax: u32, logits_bf16: &[u16]) -> u32 {
        if self.config.is_greedy() {
            return argmax;
        }
        let inv_t = 1.0 / self.config.temperature;
        self.scratch.clear();
        self.scratch.extend(
            logits_bf16.iter().enumerate().map(|(i, &b)| (half::f16::from_bits(b).to_f32() * inv_t, i as u32)),
        );
        let n = self.scratch.len();
        // Top-k: an O(n) selection, then only the survivors are sorted.
        let keep = if self.config.top_k > 0 { self.config.top_k.min(n) } else { n };
        if keep < n {
            self.scratch.select_nth_unstable_by(keep - 1, |a, b| b.0.total_cmp(&a.0));
            self.scratch.truncate(keep);
        }
        self.scratch.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        // Softmax over the survivors.
        let best = self.scratch[0].0;
        let mut total = 0.0f32;
        for entry in &mut self.scratch {
            entry.0 = (entry.0 - best).exp();
            total += entry.0;
        }
        for entry in &mut self.scratch {
            entry.0 /= total;
        }
        // Min-p: below `min_p × best probability`, out.
        if self.config.min_p > 0.0 {
            let floor = self.config.min_p * self.scratch[0].0;
            self.scratch.retain(|e| e.0 >= floor);
        }
        // Top-p: the smallest prefix whose mass reaches `top_p`.
        if self.config.top_p < 1.0 {
            let mut mass = 0.0f32;
            let mut end = self.scratch.len();
            for (i, e) in self.scratch.iter().enumerate() {
                mass += e.0;
                if mass >= self.config.top_p {
                    end = i + 1;
                    break;
                }
            }
            self.scratch.truncate(end);
        }
        // One draw, proportional to the remaining probabilities.
        let mass: f32 = self.scratch.iter().map(|e| e.0).sum();
        let mut r = self.next_f32() * mass;
        for e in &self.scratch {
            if r < e.0 {
                return e.1;
            }
            r -= e.0;
        }
        self.scratch.last().map_or(argmax, |e| e.1)
    }

    /// xorshift64*: a uniform draw in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        let mut s = self.state;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.state = s;
        let bits = s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        bits as f32 / (1u64 << 24) as f32
    }
}
