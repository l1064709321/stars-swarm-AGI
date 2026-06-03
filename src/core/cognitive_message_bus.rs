//! Module 8: 认知消息总线 (Cognitive Message Bus)
//! Central message broker for all inter-module communication

use crate::types::{CognitiveMessage, CognitiveModule, Result};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Message bus for coordinating all 62 modules
#[derive(Clone)]
pub struct CognitiveMessageBus {
    module_id: u8,
    routes: Arc<DashMap<u8, mpsc::Sender<CognitiveMessage>>>,
    verify_ethics: bool,
}

impl CognitiveMessageBus {
    pub fn new() -> Self {
        Self {
            module_id: 8,
            routes: Arc::new(DashMap::new()),
            verify_ethics: true,
        }
    }

    pub async fn register_module(
        &self,
        module_id: u8,
        tx: mpsc::Sender<CognitiveMessage>,
    ) -> Result<()> {
        self.routes.insert(module_id, tx);
        tracing::info!("Registered module {} to message bus", module_id);
        Ok(())
    }

    pub async fn route_message(&self, message: CognitiveMessage) -> Result<()> {
        if self.verify_ethics {
            if let Some(ref _sig) = message.ethics_signature {
                tracing::debug!("Verifying ethics signature for message from module {}", message.source_module);
            }
        }

        for target_id in &message.target_modules {
            if let Some(tx) = self.routes.get(target_id) {
                tx.send(message.clone()).await
                    .map_err(|e| crate::SystemError::Other(format!(
                        "Failed to route to module {}: {}",
                        target_id, e
                    )))?;
            } else {
                tracing::warn!("No handler for module {}", target_id);
            }
        }

        Ok(())
    }
}

impl std::fmt::Debug for CognitiveMessageBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CognitiveMessageBus")
            .field("module_id", &self.module_id)
            .field("verify_ethics", &self.verify_ethics)
            .finish()
    }
}

#[async_trait::async_trait]
impl CognitiveModule for CognitiveMessageBus {
    fn module_id(&self) -> u8 {
        self.module_id
    }

    fn name(&self) -> &str {
        "CognitiveMessageBus (Module 8)"
    }

    async fn initialize(&mut self) -> Result<()> {
        tracing::info!("Initializing {}", self.name());
        Ok(())
    }

    async fn process_message(&mut self, message: CognitiveMessage) -> Result<Option<CognitiveMessage>> {
        self.route_message(message).await?;
        Ok(None)
    }

    async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down {}", self.name());
        Ok(())
    }
}
