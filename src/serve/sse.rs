//! The events of a streamed Messages API answer (herbert-rs's `sse.rs`,
//! without tool use, queue and prefill progress): the event name and its JSON.

use serde_json::{json, Value};

#[derive(Debug)]
pub enum SsePayload {
    MessageStart { msg_id: String, model: String, input_tokens: usize },
    ContentBlockStart { index: usize, block_type: &'static str },
    ThinkingDelta { index: usize, thinking: String },
    TextDelta { index: usize, text: String },
    ContentBlockStop { index: usize },
    MessageDelta { stop_reason: &'static str, output_tokens: usize },
    MessageStop,
    Error { message: String },
}

impl SsePayload {
    pub fn event(&self) -> (&'static str, Value) {
        match self {
            SsePayload::MessageStart { msg_id, model, input_tokens } => (
                "message_start",
                json!({"type": "message_start", "message": {"id": msg_id, "type": "message", "role": "assistant", "content": [],
                    "model": model, "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}),
            ),
            SsePayload::ContentBlockStart { index, block_type } => {
                let block = if *block_type == "thinking" { json!({"type": "thinking", "thinking": ""}) } else { json!({"type": "text", "text": ""}) };
                ("content_block_start", json!({"type": "content_block_start", "index": index, "content_block": block}))
            }
            SsePayload::ThinkingDelta { index, thinking } => (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": thinking}}),
            ),
            SsePayload::TextDelta { index, text } => (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}),
            ),
            SsePayload::ContentBlockStop { index } => ("content_block_stop", json!({"type": "content_block_stop", "index": index})),
            SsePayload::MessageDelta { stop_reason, output_tokens } => (
                "message_delta",
                json!({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": null}, "usage": {"output_tokens": output_tokens}}),
            ),
            SsePayload::MessageStop => ("message_stop", json!({"type": "message_stop"})),
            SsePayload::Error { message } => ("error", json!({"type": "error", "error": {"type": "api_error", "message": message}})),
        }
    }
}
