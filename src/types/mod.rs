pub mod message;
pub mod state;
pub mod vector;
pub mod errors;

pub use message::{CognitiveMessage, MessageType, EthicsSignature, CognitiveModule};
pub use state::{EthicsState, EightGateState};
pub use vector::LatentVector;
pub use errors::{SystemError, Result};
