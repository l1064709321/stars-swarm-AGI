//! Star A.I. OS - Core Rust Modules
//!
//! Module 1: SNN (Spiking Neural Network) - LIF neurons with proper STDP learning
//! Module 2: Kalman Filter - Adaptive state estimation with Joseph-form update
//! Module 3: Tarpit (binary) - TCP defense trap engine

pub mod kalman;
pub mod snn;

// Re-export main types for convenience
pub use kalman::{AdaptiveKalmanFilter, KalmanParams};
pub use snn::{SpikingNeuralNetwork, LIFParams, STDPParams, LIFNeuron};
