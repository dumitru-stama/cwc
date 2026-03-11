use async_trait::async_trait;
use cwc_core::error::{CwcError, Result};
use cwc_core::traits::{LlmClient, StreamCallback};
use cwc_core::types::{ChatMessage, Role};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::GenerateParams;

/// Client for a llama.cpp server.
pub struct LlamaCppClient {
    endpoint: String,
    client: reqwest::Client,
    default_params: GenerateParams,
}

/// Model info returned by the server.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub model: Option<String>,
    pub total_slots: Option<u32>,
}

#[derive(Serialize)]
struct CompletionRequest<'a> {
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    grammar: Option<&'a str>,
    n_predict: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Deserialize)]
struct CompletionResponse {
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    messages: Vec<ChatMsg<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    grammar: Option<&'a str>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize)]
struct ChatMsg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

impl LlamaCppClient {
    pub fn new(endpoint: &str, params: GenerateParams) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_default();

        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client,
            default_params: params,
        }
    }

    /// Health check — is the server running?
    pub async fn health(&self) -> Result<bool> {
        let url = format!("{}/health", self.endpoint);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Get model info from the server.
    pub async fn model_info(&self) -> Result<ModelInfo> {
        let url = format!("{}/props", self.endpoint);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| CwcError::Llm(format!("model_info request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(CwcError::Llm(format!(
                "model_info returned status {}",
                resp.status()
            )));
        }

        resp.json::<ModelInfo>()
            .await
            .map_err(|e| CwcError::Llm(format!("model_info parse error: {e}")))
    }
}

#[async_trait]
impl LlmClient for LlamaCppClient {
    async fn generate(
        &self,
        prompt: &str,
        grammar: Option<&str>,
        max_tokens: u32,
    ) -> Result<String> {
        let url = format!("{}/completion", self.endpoint);
        let body = CompletionRequest {
            prompt,
            grammar,
            n_predict: max_tokens,
            temperature: self.default_params.temperature,
            top_p: self.default_params.top_p,
            top_k: self.default_params.top_k,
            repeat_penalty: self.default_params.repeat_penalty,
            stop: self.default_params.stop.clone(),
        };

        tracing::debug!(endpoint = %url, max_tokens, "sending completion request");

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CwcError::Llm(format!("completion request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CwcError::Llm(format!(
                "completion returned status {status}: {body}"
            )));
        }

        let parsed: CompletionResponse = resp
            .json()
            .await
            .map_err(|e| CwcError::Llm(format!("completion parse error: {e}")))?;

        Ok(parsed.content)
    }

    async fn generate_chat(
        &self,
        messages: &[ChatMessage],
        grammar: Option<&str>,
        max_tokens: u32,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.endpoint);

        let msgs: Vec<ChatMsg<'_>> = messages
            .iter()
            .map(|m| ChatMsg {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        let body = ChatCompletionRequest {
            messages: msgs,
            grammar,
            max_tokens,
            temperature: self.default_params.temperature,
            top_p: self.default_params.top_p,
            stop: self.default_params.stop.clone(),
            stream: false,
        };

        tracing::debug!(endpoint = %url, max_tokens, "sending chat completion request");

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CwcError::Llm(format!("chat completion request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CwcError::Llm(format!(
                "chat completion returned status {status}: {body}"
            )));
        }

        let parsed: ChatCompletionResponse = resp
            .json()
            .await
            .map_err(|e| CwcError::Llm(format!("chat completion parse error: {e}")))?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| CwcError::Llm("no choices in chat completion response".into()))
    }

    async fn generate_chat_stream(
        &self,
        messages: &[ChatMessage],
        grammar: Option<&str>,
        max_tokens: u32,
        mut on_token: StreamCallback,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.endpoint);

        let msgs: Vec<ChatMsg<'_>> = messages
            .iter()
            .map(|m| ChatMsg {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        let body = ChatCompletionRequest {
            messages: msgs,
            grammar,
            max_tokens,
            temperature: self.default_params.temperature,
            top_p: self.default_params.top_p,
            stop: self.default_params.stop.clone(),
            stream: true,
        };

        tracing::debug!(endpoint = %url, max_tokens, "sending streaming chat completion request");

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CwcError::Llm(format!("streaming chat request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CwcError::Llm(format!(
                "streaming chat returned status {status}: {body}"
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

                if let Ok(parsed) = serde_json::from_str::<StreamChunk>(data) {
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

/// SSE streaming chunk from /v1/chat/completions with stream=true.
#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // requires running llama.cpp server
    async fn test_llamacpp_health() {
        let client = LlamaCppClient::new(
            "http://localhost:8080",
            GenerateParams::default(),
        );
        let healthy = client.health().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    #[ignore] // requires running llama.cpp server
    async fn test_llamacpp_generate() {
        let client = LlamaCppClient::new(
            "http://localhost:8080",
            GenerateParams::default(),
        );
        let result = client.generate("Hello, world!", None, 50).await.unwrap();
        assert!(!result.is_empty());
    }

    #[tokio::test]
    #[ignore] // requires running llama.cpp server
    async fn test_llamacpp_generate_chat() {
        let client = LlamaCppClient::new(
            "http://localhost:8080",
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

    #[tokio::test]
    #[ignore] // requires running llama.cpp server
    async fn test_llamacpp_grammar_constrained() {
        use crate::grammar::json_schema_to_gbnf;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();

        let client = LlamaCppClient::new(
            "http://localhost:8080",
            GenerateParams::default(),
        );
        let result = client
            .generate("Answer in JSON: What is 2+2?", Some(&gbnf), 100)
            .await
            .unwrap();

        // Grammar should force valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&result)
            .expect("grammar-constrained output should be valid JSON");
        assert!(parsed.get("answer").is_some());
    }

    #[tokio::test]
    #[ignore] // requires running llama.cpp server
    async fn test_llamacpp_temperature_zero_deterministic() {
        let client = LlamaCppClient::new(
            "http://localhost:8080",
            GenerateParams {
                temperature: 0.0,
                ..Default::default()
            },
        );
        let prompt = "The capital of France is";
        let r1 = client.generate(prompt, None, 20).await.unwrap();
        let r2 = client.generate(prompt, None, 20).await.unwrap();
        assert_eq!(r1, r2, "temperature 0 should be deterministic");
    }
}
