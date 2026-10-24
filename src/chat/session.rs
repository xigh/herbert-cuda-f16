//! A conversation on the engine, turn by turn: the same prompt text, the same
//! stop rule and the same counting of generated tokens as herbert-jit-bf16's
//! session, so that the `[ids]` of both engines compare one to one. No tools,
//! no history rewrite, no save/restore.

use crate::chat::detok::DetokenizationFlow;
use crate::chat::template;
use crate::chat::tokens::Tokens;
use crate::core::error::Result;
use crate::core::sampler::{Sampler, Sampling};
use crate::cuda::engine::Engine;
use std::time::Instant;

#[derive(Default, Debug, Clone)]
pub struct TurnStats {
    pub prompt_tokens: usize,
    pub prefill_seconds: f64,
    pub generated_tokens: usize,
    pub decode_seconds: f64,
    pub position: usize,
}

impl TurnStats {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_seconds.max(1e-9)
    }
    pub fn decode_tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64 / self.decode_seconds.max(1e-9)
    }
}

/// `HERBERT_VK_TOP=N`: print the N best logits of every generated position on
/// stderr, to read the margin where two engines part.
fn top_from_env() -> usize {
    std::env::var("HERBERT_VK_TOP").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
}

fn show_top(n: usize, step: usize, logits: &[u16]) {
    if n == 0 {
        return;
    }
    let mut v: Vec<(f32, usize)> = logits.iter().enumerate().map(|(i, &b)| (half::f16::from_bits(b).to_f32(), i)).collect();
    v.select_nth_unstable_by(n - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    v.truncate(n);
    v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let list: Vec<String> = v.iter().map(|(l, i)| format!("{i}:{l}")).collect();
    eprintln!("[top {step}] {}", list.join(" "));
}

pub struct Chat {
    engine: Engine,
    tokens: Tokens,
    sampler: Sampler,
    system: Option<String>,
    pub think: bool,
    pub max_tokens: usize,
    ledger: Vec<u32>,
    assistant_open: bool,
}

impl Chat {
    pub fn new(engine: Engine, tokens: Tokens, sampling: Sampling, system: Option<String>) -> Chat {
        Chat { engine, tokens, sampler: Sampler::new(sampling), system, think: true, max_tokens: 1024, ledger: Vec::new(), assistant_open: false }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Back to an empty conversation (the sampler keeps its state), the
    /// profiles cleared: what a warm-up turn leaves behind is gone.
    pub fn reset(&mut self) -> Result<()> {
        self.engine.reset()?;
        self.engine.clear_profile();
        self.ledger.clear();
        self.assistant_open = false;
        Ok(())
    }

    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    pub fn ledger(&self) -> &[u32] {
        &self.ledger
    }

    /// One turn: the user's text in, the answer streamed to `on_text`.
    pub fn chat(&mut self, user: &str, on_text: &mut dyn FnMut(&str)) -> Result<TurnStats> {
        let mut stats = TurnStats::default();
        let mut text = String::new();
        if self.ledger.is_empty() {
            text.push_str(&template::system_message(self.system.as_deref()));
        } else if self.assistant_open {
            text.push_str("<|im_end|>\n");
        }
        text.push_str(&template::user_turn(user));
        text.push_str(&template::assistant_header(self.think));
        let ids = self.tokens.encode(&text)?;
        let started = Instant::now();
        let argmax = self.engine.prefill(&ids)?;
        stats.prefill_seconds = started.elapsed().as_secs_f64();
        stats.prompt_tokens = ids.len();
        self.ledger.extend_from_slice(&ids);
        self.assistant_open = true;
        let top = top_from_env();
        let mut step = 0usize;
        show_top(top, step, self.engine.logits());
        let first = self.sampler.choose(argmax, self.engine.logits());

        let mut produced: Vec<u32> = Vec::with_capacity(self.max_tokens);
        let mut detok = DetokenizationFlow::fresh();
        let started = Instant::now();
        if !self.tokens.is_stop(first) {
            produced.push(first);
            on_text(&detok.push(&self.tokens.tokenizer, first));
            let room = self.max_tokens.saturating_sub(1);
            let (sampler, tokens) = (&mut self.sampler, &self.tokens);
            let fed = self.engine.generate(first, room, &mut |argmax, logits| {
                step += 1;
                show_top(top, step, logits);
                let token = sampler.choose(argmax, logits);
                if tokens.is_stop(token) {
                    return Ok(None);
                }
                produced.push(token);
                on_text(&detok.push(&tokens.tokenizer, token));
                Ok(Some(token))
            })?;
            // `first`, then one per callback except the last at the ceiling.
            produced.truncate(fed.max(1));
        }
        on_text(&detok.balance(&self.tokens.tokenizer));
        stats.decode_seconds = started.elapsed().as_secs_f64();
        stats.generated_tokens = produced.len();
        self.ledger.extend_from_slice(&produced);
        stats.position = self.engine.pos();
        Ok(stats)
    }
}
