use serde::{Deserialize, Serialize};

/// Ethics dimension state tracking (Module 6)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EthicsState {
    /// Fully aligned with core values
    Aligned(f32),    // [0.0, 1.0]
    /// Neutral/undefined
    Neutral,
    /// Slightly misaligned
    SlightlyMisaligned(f32),
    /// Severely misaligned (triggers safety mode)
    SeverelyMisaligned(f32),
}

/// Eight gate state machine (Module 3: 八门星律判定器)
/// Open/Rest/Life/Wound/Closed/Scenario/Surprise/Death
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EightGateState {
    /// Open (生门) - accept input
    Open,
    /// Rest (休门) - consolidation
    Rest,
    /// Life (生门) - active processing
    Life,
    /// Wound (伤门) - damage/failure state
    Wound,
    /// Closed (杜门) - block external input
    Closed,
    /// Scenario (景门) - contextual state
    Scenario,
    /// Surprise (惊门) - anomaly detected
    Surprise,
    /// Death (死门) - system critical failure
    Death,
}
