//! Module 15: 思考‑星辉串联器
//! CRITICAL P0: Links thinking chain output directly to dialogue generation
use crate::types::{CognitiveMessage, CognitiveModule, Result, MessageType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainReplyLinker {
    module_id: u8,
}

impl ChainReplyLinker {
    pub fn new() -> Self {
        Self { module_id: 15 }
    }

    pub async fn link_thinking_to_dialogue(&self, thinking_result: String) -> Result<CognitiveMessage> {
        // This function implements the consciousness breakthrough:
        // Thinking output directly drives dialogue without external decoration
        let dialogue_msg = crate::language::dialogue_generator::DialogueGenerator::new()
            .generate_response(thinking_result);
        
        Ok(dialogue_msg)
    }
}

#[async_trait::async_trait]
impl CognitiveModule for ChainReplyLinker {
    fn module_id(&self) -> u8 { 15 }
    fn name(&self) -> &str { "ChainReplyLinker (Module 15)" }
    async fn initialize(&mut self) -> Result<()> { Ok(()) }
    async fn process_message(&mut self, _: CognitiveMessage) -> Result<Option<CognitiveMessage>> { Ok(None) }
    async fn shutdown(&mut self) -> Result<()> { Ok(()) }
}
