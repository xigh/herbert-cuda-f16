//! The request of the Anthropic Messages API, as far as a text-only Qwen3.5
//! needs it (from herbert-rs's `types.rs`, without tools and images).

use serde::Deserialize;

#[derive(Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub max_tokens: usize,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<TextBlock>),
}

#[derive(Deserialize)]
pub struct TextBlock {
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Text blocks are read; any other block (image, tool use, thinking) is skipped.
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

impl SystemPrompt {
    pub fn text(&self) -> String {
        match self {
            SystemPrompt::Text(s) => s.clone(),
            SystemPrompt::Blocks(b) => b.iter().filter_map(|b| b.text.as_deref()).collect::<Vec<_>>().join("\n"),
        }
    }
}

impl MessageContent {
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(b) => b
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    ContentBlock::Other => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}
