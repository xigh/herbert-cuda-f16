//! herbert-cuda-serve MODEL_DIR [options] — the Anthropic Messages API over
//! HTTP (`POST /v1/messages`, streamed with `"stream": true`), one request at
//! a time, every request from an empty cache. Built with `--features serve`.
//!
//!   --device N        the CUDA device ordinal (default 0)
//!   --ctx N           positions reserved: prompt and answer (default 4608)
//!   --tune KEY=VALUE  override a kernel parameter (repeatable; see cuda/tuning.rs)
//!   --no-think        close the thinking block in the header
//!   --logits          copy the logits to the host after each step (needed to sample
//!                     with a temperature; greedy requests read only the argmax)
//!   --host H          address to listen on (default 127.0.0.1)
//!   --port P          port (default 8080)
//!   --name NAME       the model name in the answers (default: the directory name)
//!
//! On stderr, one `[request]` line per answer: prompt tokens, prefill and decode
//! times as the engine sees them.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use herbert_cuda::chat::Tokens;
use herbert_cuda::serve::sse::SsePayload;
use herbert_cuda::serve::types::MessagesRequest;
use herbert_cuda::serve::worker::{Job, Worker};
use herbert_cuda::cuda::engine::{Engine, Options};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

struct Args {
    model_dir: PathBuf,
    engine: Options,
    think: bool,
    host: String,
    port: u16,
    name: Option<String>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut it = args.iter();
    let model_dir = PathBuf::from(it.next().ok_or("MODEL_DIR is required")?);
    let mut a = Args {
        model_dir,
        engine: Options { device: 0, capacity: 4608, dump_ptx: None, tune: Vec::new(), profile: false, logits: false },
        think: true,
        host: "127.0.0.1".to_string(),
        port: 8080,
        name: None,
    };
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--device" => a.engine.device = value()?.parse().map_err(|e| format!("--device: {e}"))?,
            "--ctx" => a.engine.capacity = value()?.parse().map_err(|e| format!("--ctx: {e}"))?,
            "--tune" => a.engine.tune.push(value()?),
            "--no-think" => a.think = false,
            "--logits" => a.engine.logits = true,
            "--host" => a.host = value()?,
            "--port" => a.port = value()?.parse().map_err(|e| format!("--port: {e}"))?,
            "--name" => a.name = Some(value()?),
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(a)
}

struct App {
    jobs: Mutex<mpsc::Sender<Job>>,
    next: AtomicU64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\nusage: herbert-cuda-serve MODEL_DIR [--device N] [--ctx N] [--tune KEY=VALUE]... [--no-think] [--logits] [--host H] [--port P] [--name NAME]");
            std::process::exit(2);
        }
    };
    // The engine lives on its own thread, built there: one request at a time.
    let (jobs, queue) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let Args { model_dir, engine: options, think, host, port, name } = a;
    let name = name.unwrap_or_else(|| model_dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    std::thread::spawn(move || {
        let load = || -> herbert_cuda::core::error::Result<Worker> {
            let engine = Engine::load(&model_dir, &options)?;
            eprintln!("[engine] loaded in {:.1} s, mounted in {:.2} s · {}", engine.load_seconds, engine.mount_seconds, engine.tuning().summary());
            let tokens = Tokens::load(&herbert_cuda::model::resolve_model_dir(&model_dir)?)?;
            eprintln!("[tokens] {}", tokens.summary());
            let mut w = Worker::new(engine, tokens, name);
            w.think = think;
            w.logits = options.logits;
            Ok(w)
        };
        let mut worker = match load() {
            Ok(w) => {
                let _ = ready_tx.send(Ok(()));
                w
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e.to_string()));
                return;
            }
        };
        while let Ok(job) = queue.recv() {
            worker.serve(job);
        }
    });
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        Err(_) => std::process::exit(1),
    }
    let app = Arc::new(App { jobs: Mutex::new(jobs), next: AtomicU64::new(1) });
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    rt.block_on(async move {
        let router = Router::new()
            .route("/", get(|| async { Json(serde_json::json!({"status": "ok"})) }))
            .route("/v1/messages", post(messages))
            .with_state(app);
        let addr = format!("{host}:{port}");
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap_or_else(|e| {
            eprintln!("error: {addr}: {e}");
            std::process::exit(1)
        });
        eprintln!("[serve] listening on http://{addr}/v1/messages");
        axum::serve(listener, router).await.expect("server");
    });
}

async fn messages(State(app): State<Arc<App>>, Json(req): Json<MessagesRequest>) -> Response {
    let id = format!("msg_herbert_cuda_{:06}", app.next.fetch_add(1, Ordering::Relaxed));
    let stream = req.stream.unwrap_or(false);
    let (tx, rx) = tokio::sync::mpsc::channel::<SsePayload>(1024);
    if app.jobs.lock().unwrap().send(Job { id, req, tx }).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "the engine thread is gone").into_response();
    }
    if stream {
        let events = ReceiverStream::new(rx).map(|p| {
            let (name, data) = p.event();
            Ok::<_, Infallible>(Event::default().event(name).data(data.to_string()))
        });
        return Sse::new(events).into_response();
    }
    collect(rx).await
}

/// The non-streamed answer: the same events gathered into one message.
async fn collect(mut rx: tokio::sync::mpsc::Receiver<SsePayload>) -> Response {
    let (mut id, mut model, mut input_tokens, mut output_tokens) = (String::new(), String::new(), 0, 0);
    let mut stop_reason = "end_turn";
    let mut content: Vec<serde_json::Value> = Vec::new();
    let (mut kind, mut text) = ("text", String::new());
    while let Some(p) = rx.recv().await {
        match p {
            SsePayload::MessageStart { msg_id, model: m, input_tokens: n } => {
                id = msg_id;
                model = m;
                input_tokens = n;
            }
            SsePayload::ContentBlockStart { block_type, .. } => {
                kind = block_type;
                text.clear();
            }
            SsePayload::ThinkingDelta { thinking, .. } => text.push_str(&thinking),
            SsePayload::TextDelta { text: t, .. } => text.push_str(&t),
            SsePayload::ContentBlockStop { .. } => {
                let t = std::mem::take(&mut text);
                content.push(if kind == "thinking" { serde_json::json!({"type": "thinking", "thinking": t}) } else { serde_json::json!({"type": "text", "text": t}) });
            }
            SsePayload::MessageDelta { stop_reason: s, output_tokens: n } => {
                stop_reason = s;
                output_tokens = n;
            }
            SsePayload::MessageStop => {}
            SsePayload::Error { message } => {
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"type": "error", "error": {"type": "api_error", "message": message}}))).into_response();
            }
        }
    }
    Json(serde_json::json!({
        "id": id, "type": "message", "role": "assistant", "content": content, "model": model,
        "stop_reason": stop_reason, "stop_sequence": null,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }))
    .into_response()
}
