//! A minimal HTTP server for the bench (feature `serve`): the Anthropic
//! Messages API (`/v1/messages`, streamed or not), one request at a time.
//!
//! Ported from herbert-rs's server: the request types, the SSE events and the
//! order of the blocks (thinking, then text) are its own; its tools, sessions,
//! authentication, queue and metrics are left out. The prompt is rendered with
//! this crate's Qwen3.5 template, so a request served here and a turn of
//! herbert-cuda-chat start from the same ids. Every request starts from an empty
//! cache: no prefix is reused from one request to the next.

pub mod sse;
pub mod types;
pub mod worker;
