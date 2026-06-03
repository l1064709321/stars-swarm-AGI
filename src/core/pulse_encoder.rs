//! Module 1: 星核脉冲编码器 (Pulse Encoder)
//! Implements spiking neural network (SNN) encoding using Leaky Integrate-and-Fire (LIF) neurons
//! with STDP/SADP double-pathway fusion learning.
//! P1 Target: Cross-invocation membrane potential persistence

use crate::types::{CognitiveMessage, CognitiveModule, Result};
use serde::{Deserialize, Serialize};

/// LIF neuron membrane potential state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifNeuron {
    pub v_mem: f32,
    pub v_rest: f32,
    pub v_threshold: f32,
    pub tau: f32,
    pub spike_history: Vec<bool>,
    pub last_spike_time: Option<f32>,
}

/// STDP parameters
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StdpParams {
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
}

/// Pulse encoding module (L1 layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulseEncoder {
    module_id: u8,
    neurons: Vec<LifNeuron>,
    weights: Vec<Vec<f32>>,
    stdp_params: StdpParams,
}

impl PulseEncoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            module_id: 1,
            neurons: vec![LifNeuron {
                v_mem: -70.0,
                v_rest: -70.0,
                v_threshold: -50.0,
                tau: 10.0,
                spike_history: Vec::new(),
                last_spike_time: None,
            }; num_neurons],
            weights: vec![vec![0.1; num_neurons]; num_neurons],
            stdp_params: StdpParams {
                a_plus: 0.01,
                a_minus: 0.01,
                tau_plus: 20.0,
                tau_minus: 20.0,
            },
        }
    }

    pub fn integrate_fire(&mut self, input_current: Vec<f32>, dt: f32) -> Vec<bool> {
        let mut spikes = vec![false; self.neurons.len()];
        
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let current = input_current.get(i).copied().unwrap_or(0.0);
            let decay = (-dt / neuron.tau).exp();
            
            neuron.v_mem = neuron.v_mem * decay + current * (1.0 - decay);
            
            if neuron.v_mem >= neuron.v_threshold {
                spikes[i] = true;
                neuron.v_mem = neuron.v_rest;
                neuron.spike_history.push(true);
                neuron.last_spike_time = Some(0.0);
            } else {
                neuron.spike_history.push(false);
            }
            
            if let Some(ref mut last_time) = neuron.last_spike_time {
                *last_time += dt;
            }
        }
        
        spikes
    }

    pub fn apply_stdp(&mut self, pre_spikes: &[bool], post_spikes: &[bool]) {
        for (i, &post_spike) in post_spikes.iter().enumerate() {
            for (j, &pre_spike) in pre_spikes.iter().enumerate() {
                if i >= self.weights.len() || j >= self.weights[i].len() {
                    continue;
                }
                
                let mut dw = 0.0;
                if post_spike && pre_spike {
                    dw = self.stdp_params.a_plus;
                } else if post_spike && !pre_spike {
                    dw = -self.stdp_params.a_minus;
                }
                
                self.weights[i][j] = (self.weights[i][j] + dw).clamp(-1.0, 1.0);
            }
        }
    }
}

#[async_trait::async_trait]
impl CognitiveModule for PulseEncoder {
    fn module_id(&self) -> u8 {
        self.module_id
    }

    fn name(&self) -> &str {
        "PulseEncoder (Module 1)"
    }

    async fn initialize(&mut self) -> Result<()> {
        tracing::info!("Initializing {}", self.name());
        Ok(())
    }

    async fn process_message(&mut self, _message: CognitiveMessage) -> Result<Option<CognitiveMessage>> {
        Ok(None)
    }

    async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down {}", self.name());
        Ok(())
    }
}
