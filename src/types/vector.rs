use serde::{Deserialize, Serialize};
use crate::types::Result;
use crate::SystemError;

/// 64-dimensional unified latent space vector (Module 16)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentVector {
    pub values: [f32; 64],
}

impl LatentVector {
    pub fn new() -> Self {
        Self { values: [0.0; 64] }
    }

    pub fn from_vec(v: Vec<f32>) -> Result<Self> {
        if v.len() != 64 {
            return Err(SystemError::LatentSpaceDimensionMismatch {
                expected: 64,
                got: v.len(),
            });
        }
        let mut values = [0.0; 64];
        values.copy_from_slice(&v);
        Ok(Self { values })
    }

    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.values.iter().zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}
