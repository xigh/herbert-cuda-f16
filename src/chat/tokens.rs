//! The tokenizer and the handful of special tokens the chat looks at.
//!
//! The tokenizer is the model's own `tokenizer.json` (byte-level BPE). The
//! turn markers (`<|im_start|>`, `<|im_end|>`), the end of text, and the block
//! markers (`<think>`, `</think>`, `<tool_call>`, `</tool_call>`) are single
//! tokens: the session recognises them by id, never by decoding text, so a
//! user who types "<think>" cannot open a block.

use crate::core::error::{HerbertError, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Tokens {
    pub tokenizer: Tokenizer,
    pub im_start: u32,
    pub im_end: u32,
    pub endoftext: u32,
    pub think_open: Option<u32>,
    pub think_close: Option<u32>,
    pub tool_call_open: Option<u32>,
    pub tool_call_close: Option<u32>,
}

impl Tokens {
    pub fn load(model_dir: &Path) -> Result<Tokens> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| HerbertError::ModelLoad(format!("{}: {e}", path.display())))?;
        let required = |name: &str| -> Result<u32> {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| HerbertError::ModelLoad(format!("the tokenizer has no `{name}` token")))
        };
        Ok(Tokens {
            im_start: required("<|im_start|>")?,
            im_end: required("<|im_end|>")?,
            endoftext: required("<|endoftext|>")?,
            think_open: tokenizer.token_to_id("<think>"),
            think_close: tokenizer.token_to_id("</think>"),
            tool_call_open: tokenizer.token_to_id("<tool_call>"),
            tool_call_close: tokenizer.token_to_id("</tool_call>"),
            tokenizer,
        })
    }

    /// Text to ids. Special tokens written in the text (`<|im_start|>`, …) are
    /// recognised: this is how the template's markers become ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| HerbertError::Backend(format!("tokenizer: {e}")))?
            .get_ids()
            .to_vec())
    }

    /// Ids to text, special tokens left out.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true).unwrap_or_default()
    }

    pub fn is_stop(&self, id: u32) -> bool {
        id == self.im_end || id == self.endoftext
    }

    /// One line for the log: the ids the session will watch.
    pub fn summary(&self) -> String {
        format!(
            "im_start {} im_end {} endoftext {} think {:?}/{:?} tool_call {:?}/{:?} · vocab {}",
            self.im_start,
            self.im_end,
            self.endoftext,
            self.think_open,
            self.think_close,
            self.tool_call_open,
            self.tool_call_close,
            self.tokenizer.get_vocab_size(true)
        )
    }
}
