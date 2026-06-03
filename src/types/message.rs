use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::types::{EthicsState, Result, SystemError};

/// Central message protocol for inter-module communication (Module 8)
/// All modules communicate via CognitiveMessage to maintain protocol consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMessage {
    /// Unique message ID
    pub id: Uuid,
    /// Source module ID (1-62)
    pub source_module: u8,
    /// Target module ID(s)
    pub target_modules: Vec<u8>,
    /// Message type (pulse, state, intention, etc.)
    pub message_type: MessageType,
    /// Payload data
    pub payload: serde_json::Value,
    /// Ethics signature (for L6 verification)
    pub ethics_signature: Option<EthicsSignature>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// TTL in milliseconds
    pub ttl_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// SNN pulse encoding (Module 1)
    PulseEncoding,
    /// State estimation (Module 2)
    StateEstimate,
    /// Gate state transition (Module 3)
    GateTransition,
    /// Broadcast winner signal (Module 4)
    BroadcastWinner,
    /// Intrinsic drive signal (Module 5)
    IntrinsicDrive,
    /// Ethics field state (Module 6)
    EthicsField,
    /// Existence guard check (Module 7)
    ExistenceGuard,
    /// Memory consolidation (Module 9)
    MemoryConsolidation,
    /// Dialogue output (Module 13)
    DialogueOutput,
    /// Reasoning step (Modules 16-20)
    ReasoningStep,
    /// Evolution feedback (Modules 21-26)
    EvolutionFeedback,
    /// Failure signal (Module 31)
    FailureSignal,
    /// Other
    Custom(String),
}

/// Ethics signature for message verification (Module 7: 存在性递归守护器)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicsSignature {
    /// Core anchor hash (immutable)
    pub core_anchor: String,
    /// Current ethics state at message creation
    pub ethics_state: EthicsState,
    /// Timestamp of signature
    pub signed_at: DateTime<Utc>,
}

/// Module interface trait (all 62 modules implement this)
#[async_trait::async_trait]
pub trait CognitiveModule: Send + Sync {
    /// Module ID (1-62)
    fn module_id(&self) -> u8;
    
    /// Module name
    fn name(&self) -> &str;
    
    /// Initialize module
    async fn initialize(&mut self) -> Result<()>;
    
    /// Process incoming message
    async fn process_message(&mut self, message: CognitiveMessage) -> Result<Option<CognitiveMessage>>;
    
    /// Graceful shutdown
    async fn shutdown(&mut self) -> Result<()>;
}
