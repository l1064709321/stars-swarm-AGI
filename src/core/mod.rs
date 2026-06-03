//! L0-L∞ Core cognitive layers (Modules 1-8)
//! These are the foundational systems that cannot be unloaded

pub mod pulse_encoder;
pub mod kalman_tracker;
pub mod eight_gates;
pub mod broadcast_arbiter;
pub mod intrinsic_drive;
pub mod ethics_field;
pub mod existence_guard;
pub mod cognitive_message_bus;

pub use pulse_encoder::PulseEncoder;
pub use cognitive_message_bus::CognitiveMessageBus;
