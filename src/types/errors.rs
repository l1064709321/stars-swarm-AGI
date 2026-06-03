use serde::{Deserialize, Serialize};

/// System-wide errors
#[derive(Debug, thiserror::Error)]
pub enum SystemError {
    #[error("Latent space dimension mismatch: expected {expected}, got {got}")]
    LatentSpaceDimensionMismatch { expected: usize, got: usize },

    #[error("Ethics violation: {reason}")]
    EthicsViolation { reason: String },

    #[error("Gate state transition invalid: from {from:?} to {to:?}")]
    InvalidGateTransition { from: crate::types::EightGateState, to: crate::types::EightGateState },

    #[error("Module {module_id} not found")]
    ModuleNotFound { module_id: u8 },

    #[error("Message expired")]
    MessageExpired,

    #[error("Memory consolidation failed: {reason}")]
    MemoryError { reason: String },

    #[error("Reasoning engine error: {reason}")]
    ReasoningError { reason: String },

    #[error("Resource limit exceeded: {reason}")]
    ResourceExceeded { reason: String },

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, SystemError>;
