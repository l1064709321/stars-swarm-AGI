//! Module 13: 星辉对话生成器 (Dialogue Generator)
//! P0 Critical: Must be directly driven by thinking chain output
use crate::types::{CognitiveMessage, CognitiveModule, Result, MessageType};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::Utc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueGenerator {
    module_id: u8,
}

impl DialogueGenerator {
    pub fn new() -> Self {
        Self { module_id: 13 }
    }

    pub fn generate_response(&self, thinking_chain_result: String) -> CognitiveMessage {
        CognitiveMessage {
            id: Uuid::new_v4(),
            source_module: 13,
            target_modules: vec![50], // Send to HTTP API service
            message_type: MessageType::DialogueOutput,
            payload: serde_json::json!({
                "response": thinking_chain_result,
                "confidence": 0.8
            }),
            ethics_signature: None,
            created_at: Utc::now(),
            ttl_ms: 5000,
        }
    }
}

#[async_trait::async_trait]
impl CognitiveModule for DialogueGenerator {
    fn module_id(&self) -> u8 { 13 }
    fn name(&self) -> &str { "DialogueGenerator (Module 13)" }
    async fn initialize(&mut self) -> Result<()> { Ok(()) }
    async fn process_message(&mut self, _: CognitiveMessage) -> Result<Option<CognitiveMessage>> { Ok(None) }
    async fn shutdown(&mut self) -> Result<()> { Ok(()) }
}
