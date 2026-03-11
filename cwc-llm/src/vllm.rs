use async_trait::async_trait;
use cwc_core::error::{CwcError, Result};
use cwc_core::traits::{LlmClient, StreamCallback};
use cwc_core::types::{ChatMessage, Role};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::GenerateParams;

/// Client for a vLLM server (OpenAI-compatible API).
pub struct VllmClient {
    endpoint: String,
    client: reqwest::Client,
    default_params: GenerateParams,
    model: String,
}

#[derive(Serialize)]
struct VllmCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    guided_json: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    guided_grammar: Option<&'a str>,
}

#[derive(Serialize)]
struct VllmChatRequest<'a> {
    model: &'a str,
    messages: Vec<VllmChatMsg<'a>>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    guided_json: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    guided_grammar: Option<&'a str>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize)]
struct VllmChatMsg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct VllmCompletionResponse {
    choices: Vec<VllmCompletionChoice>,
}

#[derive(Deserialize)]
struct VllmCompletionChoice {
    text: Option<String>,
    message: Option<VllmMessage>,
}

#[derive(Deserialize)]
struct VllmMessage {
    content: String,
}

impl VllmClient {
    pub fn new(endpoint: &str, model: &str, params: GenerateParams) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_default();

        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client,
            default_params: params,
            model: model.to_string(),
        }
    }

    /// Health check — is the vLLM server running?
    pub async fn health(&self) -> Result<bool> {
        let url = format!("{}/health", self.endpoint);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Try to parse `grammar` as a JSON schema for vLLM's `guided_json`,
    /// otherwise use it as a raw grammar string for `guided_grammar`.
    fn parse_grammar_arg(grammar: Option<&str>) -> (Option<serde_json::Value>, Option<String>) {
        match grammar {
            None => (None, None),
            Some(g) => {
                if let Ok(schema) = serde_json::from_str::<serde_json::Value>(g) {
                    if schema.is_object() {
                        return (Some(schema), None);
                    }
                }
                (None, Some(g.to_string()))
            }
        }
    }
}

#[async_trait]
impl LlmClient for VllmClient {
    async fn generate(
        &self,
        prompt: &str,
        grammar: Option<&str>,
        max_tokens: u32,
    ) -> Result<String> {
        let url = format!("{}/v1/completions", self.endpoint);
        let (guided_json, guided_grammar) = Self::parse_grammar_arg(grammar);

        let body = VllmCompletionRequest {
            model: &self.model,
            prompt,
            max_tokens,
            temperature: self.default_params.temperature,
            top_p: self.default_params.top_p,
            stop: self.default_params.stop.clone(),
            guided_json: guided_json.as_ref(),
            guided_grammar: guided_grammar.as_deref(),
        };

        tracing::debug!(endpoint = %url, max_tokens, "sending vLLM completion request");

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CwcError::Llm(format!("vLLM completion request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CwcError::Llm(format!(
                "vLLM completion returned status {status}: {body}"
            )));
        }

        let parsed: VllmCompletionResponse = resp
            .json()
            .await
            .map_err(|e| CwcError::Llm(format!("vLLM completion parse error: {e}")))?;

        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.text)
            .ok_or_else(|| CwcError::Llm("no text in vLLM completion response".into()))
    }

    async fn generate_chat(
        &self,
        messages: &[ChatMessage],
        grammar: Option<&str>,
        max_tokens: u32,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let (guided_json, guided_grammar) = Self::parse_grammar_arg(grammar);

        let msgs: Vec<VllmChatMsg<'_>> = messages
            .iter()
            .map(|m| VllmChatMsg {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        let body = VllmChatRequest {
            model: &self.model,
            messages: msgs,
            max_tokens,
            temperature: self.default_params.temperature,
            top_p: self.default_params.top_p,
            stop: self.default_params.stop.clone(),
            guided_json: guided_json.as_ref(),
            guided_grammar: guided_grammar.as_deref(),
            stream: false,
        };

        tracing::debug!(endpoint = %url, max_tokens, "sending vLLM chat completion request");

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CwcError::Llm(format!("vLLM chat request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CwcError::Llm(format!(
                "vLLM chat returned status {status}: {body}"
            )));
        }

        let parsed: VllmCompletionResponse = resp
            .json()
            .await
            .map_err(|e| CwcError::Llm(format!("vLLM chat parse error: {e}")))?;

        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.map(|m| m.content))
            .ok_or_else(|| CwcError::Llm("no message in vLLM chat response".into()))
    }

    async fn generate_chat_stream(
        &self,
        messages: &[ChatMessage],
        grammar: Option<&str>,
        max_tokens: u32,
        mut on_token: StreamCallback,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let (guided_json, guided_grammar) = Self::parse_grammar_arg(grammar);

        let msgs: Vec<VllmChatMsg<'_>> = messages
            .iter()
            .map(|m| VllmChatMsg {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        let body = VllmChatRequest {
            model: &self.model,
            messages: msgs,
            max_tokens,
            temperature: self.default_params.temperature,
            top_p: self.default_params.top_p,
            stop: self.default_params.stop.clone(),
            guided_json: guided_json.as_ref(),
            guided_grammar: guided_grammar.as_deref(),
            stream: true,
        };

        tracing::debug!(endpoint = %url, max_tokens, "sending streaming vLLM chat request");

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CwcError::Llm(format!("vLLM streaming chat request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CwcError::Llm(format!(
                "vLLM streaming chat returned status {status}: {body}"
            )));
        }

        let mut full_response = String::new();
        let mut stream = resp.bytes_stream();

        // Byte buffer for partial SSE lines. Using bytes (not String) avoids
        // corrupting multi-byte UTF-8 characters split across TCP chunks.
        let mut byte_buf: Vec<u8> = Vec::new();

        'outer: while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| CwcError::Llm(format!("stream read error: {e}")))?;
            byte_buf.extend_from_slice(&bytes);

            // Process complete SSE lines (newline 0x0A is always single-byte in UTF-8)
            while let Some(newline_pos) = byte_buf.iter().position(|&b| b == b'\n') {
                let line = String::from_utf8_lossy(&byte_buf[..newline_pos]).trim().to_string();
                byte_buf = byte_buf[newline_pos + 1..].to_vec();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }
                let data = &line["data: ".len()..];
                if data == "[DONE]" {
                    break 'outer;
                }

                if let Ok(parsed) = serde_json::from_str::<VllmStreamChunk>(data) {
                    if let Some(choice) = parsed.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            on_token(content);
                            full_response.push_str(content);
                        }
                    }
                }
            }
        }

        Ok(full_response)
    }
}

/// SSE streaming chunk from vLLM /v1/chat/completions with stream=true.
#[derive(Deserialize)]
struct VllmStreamChunk {
    choices: Vec<VllmStreamChoice>,
}

#[derive(Deserialize)]
struct VllmStreamChoice {
    delta: VllmStreamDelta,
}

#[derive(Deserialize)]
struct VllmStreamDelta {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // requires running vLLM server
    async fn test_vllm_health() {
        let client = VllmClient::new(
            "http://localhost:8000",
            "default",
            GenerateParams::default(),
        );
        let healthy = client.health().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    #[ignore] // requires running vLLM server
    async fn test_vllm_generate_chat() {
        let client = VllmClient::new(
            "http://localhost:8000",
            "default",
            GenerateParams::default(),
        );
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: Role::User,
                content: "Say hello.".into(),
            },
        ];
        let result = client.generate_chat(&messages, None, 50).await.unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_vllm_parse_grammar_json_schema() {
        let schema = r#"{"type": "object", "properties": {"a": {"type": "string"}}}"#;
        let (json, grammar) = VllmClient::parse_grammar_arg(Some(schema));
        assert!(json.is_some());
        assert!(grammar.is_none());
    }

    #[test]
    fn test_vllm_parse_grammar_raw() {
        let gbnf = "root ::= \"hello\"";
        let (json, grammar) = VllmClient::parse_grammar_arg(Some(gbnf));
        assert!(json.is_none());
        assert_eq!(grammar.unwrap(), gbnf);
    }

    #[test]
    fn test_vllm_parse_grammar_none() {
        let (json, grammar) = VllmClient::parse_grammar_arg(None);
        assert!(json.is_none());
        assert!(grammar.is_none());
    }
}
