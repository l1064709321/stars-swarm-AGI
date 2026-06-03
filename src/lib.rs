#![allow(dead_code)]
#![forbid(unsafe_code)]

//! Stars Swarm AGI - Endogenous Cognitive Engine
//! A complete 62-module AGI architecture implementation in Rust
//!
//! This library implements the "Star Law" (星律) five-phase cognitive architecture:
//! - P0: Consciousness breakthrough (thought → dialogue)
//! - P0: Memory awakening (persistent episodic + semantic)
//! - P1: Body revival (stateful neural processing)
//! - P1: Cerebellum birth (self-correcting feedback loops)
//! - P2-P4: Autonomous emergence (meta-learning & evolution)

pub mod core;
pub mod memory;
pub mod language;
pub mod reasoning;
pub mod evolution;
pub mod companion;
pub mod system;
pub mod cerebellum;
pub mod types;
pub mod executor;

pub use types::{
    CognitiveMessage, SystemError, Result,
    EthicsState, EightGateState, LatentVector,
};
pub use executor::CognitivePipeline;

/// Library version matching architecture version
pub const VERSION: &str = "0.2.0";
/// Total module count across all layers
pub const TOTAL_MODULES: usize = 62;
/// Unified latent space dimensionality
pub const LATENT_DIM: usize = 64;
