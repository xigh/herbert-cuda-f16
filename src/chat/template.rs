//! The Qwen3.5 chat template, rendered turn by turn — the strings of
//! herbert-jit-bf16's `chat/template.rs` byte for byte, without the tools.

/// `<|im_start|>system\n…<|im_end|>\n`, nothing without a system text.
pub fn system_message(system: Option<&str>) -> String {
    match system.map(str::trim).filter(|s| !s.is_empty()) {
        None => String::new(),
        Some(text) => format!("<|im_start|>system\n{text}<|im_end|>\n"),
    }
}

pub fn user_turn(text: &str) -> String {
    format!("<|im_start|>user\n{}<|im_end|>\n", text.trim())
}

/// The assistant header: thinking open, or an empty thinking block.
pub fn assistant_header(think: bool) -> String {
    if think {
        "<|im_start|>assistant\n<think>\n".to_string()
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n".to_string()
    }
}
