//! Module 9: 双层记忆连续体
use crate::types::{CognitiveMessage, CognitiveModule, Result, LatentVector};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub content: String,
    pub embedding: LatentVector,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub salience: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemory {
    pub concept: String,
    pub representation: LatentVector,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualLayerMemory {
    module_id: u8,
    episodic: VecDeque<EpisodicMemory>,
    semantic: Vec<SemanticMemory>,
    max_episodic: usize,
}

impl DualLayerMemory {
    pub fn new(max_episodic: usize) -> Self {
        Self {
            module_id: 9,
            episodic: VecDeque::new(),
            semantic: Vec::new(),
            max_episodic,
        }
    }
}

#[async_trait::async_trait]
impl CognitiveModule for DualLayerMemory {
    fn module_id(&self) -> u8 { 9 }
    fn name(&self) -> &str { "DualLayerMemory (Module 9)" }
    async fn initialize(&mut self) -> Result<()> { Ok(()) }
    async fn process_message(&mut self, _: CognitiveMessage) -> Result<Option<CognitiveMessage>> { Ok(None) }
    async fn shutdown(&mut self) -> Result<()> { Ok(()) }
}
