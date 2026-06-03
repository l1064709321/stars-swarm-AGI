//! Module 50: HTTP API 服务
use crate::types::{CognitiveMessage, CognitiveModule, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiService {
    module_id: u8,
    port: u16,
}

impl ApiService {
    pub fn new(port: u16) -> Self {
        Self {
            module_id: 50,
            port,
        }
    }
}

#[async_trait::async_trait]
impl CognitiveModule for ApiService {
    fn module_id(&self) -> u8 { 50 }
    fn name(&self) -> &str { "ApiService (Module 50)" }
    async fn initialize(&mut self) -> Result<()> {
        tracing::info!("API Service initializing on port {}", self.port);
        Ok(())
    }
    async fn process_message(&mut self, _: CognitiveMessage) -> Result<Option<CognitiveMessage>> { Ok(None) }
    async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("API Service shutting down");
        Ok(())
    }
}
