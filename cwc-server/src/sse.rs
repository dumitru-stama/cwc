use std::convert::Infallible;
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Create an SSE stream that yields token chunks from a channel.
/// Returns the Sse response and a sender for pushing tokens.
/// Includes a keep-alive to prevent proxy timeouts during retrieval/compilation.
pub fn token_stream() -> (
    mpsc::Sender<SseEvent>,
    Sse<impl Stream<Item = Result<Event, Infallible>>>,
) {
    let (tx, rx) = mpsc::channel::<SseEvent>(4096);
    let stream = ReceiverStream::new(rx).map(|evt| match evt {
        SseEvent::Token(text) => Ok(Event::default().event("token").data(text)),
        SseEvent::Done(json) => Ok(Event::default().event("done").data(json)),
        SseEvent::Error(msg) => Ok(Event::default().event("error").data(msg)),
    });
    let sse = Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
    (tx, sse)
}

/// Events sent over the SSE stream.
pub enum SseEvent {
    /// A token chunk from the LLM.
    Token(String),
    /// Final result JSON (includes response, verdict, timing).
    Done(String),
    /// An error occurred.
    Error(String),
}
