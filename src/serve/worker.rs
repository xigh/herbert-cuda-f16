//! The thread that owns the engine and answers one request at a time.
//!
//! The answer is streamed as herbert-rs streams it: a thinking block from the
//! first token when the header opens one, closed at `</think>`, then a text
//! block, each detokenized incrementally; the stop tokens end the answer and
//! are not sent. `max_tokens` counts every generated token, the first one
//! included: at the ceiling the last token is sampled and sent, not fed.

use crate::chat::detok::DetokenizationFlow;
use crate::chat::{template, Tokens};
use crate::core::error::{HerbertError, Result};
use crate::core::sampler::{Sampler, Sampling};
use crate::serve::sse::SsePayload;
use crate::serve::types::MessagesRequest;
use crate::cuda::engine::Engine;
use std::time::Instant;
use tokio::sync::mpsc::Sender;

pub struct Job {
    pub id: String,
    pub req: MessagesRequest,
    pub tx: Sender<SsePayload>,
}

pub struct Worker {
    engine: Engine,
    tokens: Tokens,
    model: String,
    /// The assistant header opens the thinking block (Qwen3.5's default).
    pub think: bool,
    /// The engine copies the logits to the host: needed to sample.
    pub logits: bool,
}

/// The blocks of one answer, in the order they open.
struct Blocks<'a> {
    tx: &'a Sender<SsePayload>,
    index: usize,
    open: Option<&'static str>,
    flow: DetokenizationFlow,
}

impl Blocks<'_> {
    fn send(&self, p: SsePayload) {
        let _ = self.tx.blocking_send(p);
    }

    fn open(&mut self, kind: &'static str) {
        self.send(SsePayload::ContentBlockStart { index: self.index, block_type: kind });
        self.open = Some(kind);
    }

    fn delta(&self, text: String) {
        if text.is_empty() {
            return;
        }
        let index = self.index;
        match self.open {
            Some("thinking") => self.send(SsePayload::ThinkingDelta { index, thinking: text }),
            _ => self.send(SsePayload::TextDelta { index, text }),
        }
    }

    fn close(&mut self, tokens: &Tokens) {
        if self.open.is_some() {
            let rest = self.flow.balance(&tokens.tokenizer);
            self.delta(rest);
            self.send(SsePayload::ContentBlockStop { index: self.index });
            self.index += 1;
            self.open = None;
        }
    }

    fn token(&mut self, tokens: &Tokens, id: u32) {
        if Some(id) == tokens.think_close {
            self.close(tokens);
            return;
        }
        if Some(id) == tokens.think_open {
            return;
        }
        if self.open.is_none() {
            self.open("text");
        }
        let text = self.flow.push(&tokens.tokenizer, id);
        self.delta(text);
    }
}

impl Worker {
    pub fn new(engine: Engine, tokens: Tokens, model: String) -> Worker {
        Worker { engine, tokens, model, think: true, logits: false }
    }

    /// The Qwen3.5 template over the request's system text and messages.
    pub fn prompt(&self, req: &MessagesRequest) -> String {
        let system = req.system.as_ref().map(|s| s.text());
        let mut text = template::system_message(system.as_deref());
        for m in &req.messages {
            let body = m.content.text();
            match m.role.as_str() {
                "assistant" => text.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", body.trim())),
                _ => text.push_str(&template::user_turn(&body)),
            }
        }
        text.push_str(&template::assistant_header(self.think));
        text
    }

    pub fn serve(&mut self, job: Job) {
        if let Err(e) = self.answer(&job) {
            eprintln!("[request {}] error: {e}", job.id);
            let _ = job.tx.blocking_send(SsePayload::Error { message: e.to_string() });
        }
    }

    fn answer(&mut self, job: &Job) -> Result<()> {
        let req = &job.req;
        let sampling = match req.temperature {
            Some(t) if t > 0.0 => {
                if !self.logits {
                    return Err(HerbertError::Backend("sampling needs the server started with --logits".to_string()));
                }
                let mut s = Sampling::qwen_thinking(req.seed.unwrap_or(42));
                s.temperature = t;
                if let Some(k) = req.top_k {
                    s.top_k = k;
                }
                if let Some(p) = req.top_p {
                    s.top_p = p;
                }
                s
            }
            _ => Sampling::greedy(),
        };
        let ids = self.tokens.encode(&self.prompt(req))?;
        self.engine.reset()?;
        let tx = &job.tx;
        let _ = tx.blocking_send(SsePayload::MessageStart { msg_id: job.id.clone(), model: self.model.clone(), input_tokens: ids.len() });

        let started = Instant::now();
        let argmax = self.engine.prefill(&ids)?;
        let prefill_seconds = started.elapsed().as_secs_f64();
        let mut sampler = Sampler::new(sampling);
        let first = sampler.choose(argmax, self.engine.logits());

        let mut blocks = Blocks { tx, index: 0, open: None, flow: DetokenizationFlow::fresh() };
        if self.think {
            blocks.open("thinking");
        }
        let started = Instant::now();
        let mut produced = 0usize;
        let mut stop = "max_tokens";
        if self.tokens.is_stop(first) || req.max_tokens == 0 {
            stop = "end_turn";
        } else {
            produced = 1;
            blocks.token(&self.tokens, first);
            let room = req.max_tokens - 1;
            let tokens = &self.tokens;
            self.engine.generate(first, room, &mut |argmax, logits| {
                let token = sampler.choose(argmax, logits);
                if tokens.is_stop(token) {
                    stop = "end_turn";
                    return Ok(None);
                }
                if tx.is_closed() {
                    stop = "client_disconnect";
                    return Ok(None);
                }
                produced += 1;
                blocks.token(tokens, token);
                Ok(Some(token))
            })?;
        }
        let decode_seconds = started.elapsed().as_secs_f64();
        blocks.close(&self.tokens);
        if blocks.index == 0 {
            // Nothing was produced: an empty text block, as herbert-rs answers.
            blocks.open("text");
            blocks.close(&self.tokens);
        }
        let _ = tx.blocking_send(SsePayload::MessageDelta { stop_reason: if stop == "max_tokens" { "max_tokens" } else { "end_turn" }, output_tokens: produced });
        let _ = tx.blocking_send(SsePayload::MessageStop);
        eprintln!(
            "[request {}] {} prompt tokens in {:.4} s ({:.1} tok/s) · {} generated in {:.3} s ({:.2} tok/s after the first) · {}",
            job.id,
            ids.len(),
            prefill_seconds,
            ids.len() as f64 / prefill_seconds.max(1e-9),
            produced,
            decode_seconds,
            produced.saturating_sub(1) as f64 / decode_seconds.max(1e-9),
            stop
        );
        Ok(())
    }
}
