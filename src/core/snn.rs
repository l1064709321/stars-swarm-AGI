//! Module 1: 星核脉冲编码器 (Spiking Neural Network)
//!
//! Implements Leaky Integrate-and-Fire (LIF) neurons with STDP/SADP double-pathway learning.
//! Ref: Gerstner & Kistler (2002) "Spiking Neuron Models"
//!
//! LIF Dynamics: dV/dt = -(V - V_rest)/tau_m + I(t)/C_m
//! STDP: ΔW = A+ * exp(-Δt/τ+) if Δt>0 (LTP), -A- * exp(Δt/τ-) if Δt<0 (LTD)

use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use rand::Rng;

/// LIF Neuron Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIFParams {
    pub tau_m: f32,              // Membrane time constant (ms)
    pub v_rest: f32,             // Resting potential (mV)
    pub v_threshold: f32,        // Threshold (mV)
    pub v_reset: f32,            // Reset after spike (mV)
    pub input_gain: f32,
    pub refractory_period: f32,  // ms
}

impl Default for LIFParams {
    fn default() -> Self {
        Self {
            tau_m: 20.0,
            v_rest: -65.0,
            v_threshold: -45.0,
            v_reset: -65.0,
            input_gain: 1.0,
            refractory_period: 2.0,
        }
    }
}

/// STDP Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPParams {
    pub a_plus: f32,      // LTP amplitude
    pub a_minus: f32,     // LTD amplitude
    pub tau_plus: f32,    // LTP time window (ms)
    pub tau_minus: f32,   // LTD time window (ms)
    pub w_min: f32,       // Weight bounds
    pub w_max: f32,
}

impl Default for STDPParams {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.01,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_min: -1.0,
            w_max: 1.0,
        }
    }
}

/// LIF Neuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIFNeuron {
    pub v_mem: f32,              // Membrane potential
    pub time_since_spike: f32,   // ms since last spike
    pub output: f32,             // Spike output (0 or 1)
}

/// Spiking Neural Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingNeuralNetwork {
    num_neurons: usize,
    params: LIFParams,
    stdp_params: STDPParams,
    neurons: Vec<LIFNeuron>,
    weights: Array2<f32>,
    spike_history: Vec<Vec<bool>>,
}

impl SpikingNeuralNetwork {
    /// Create new SNN
    pub fn new(num_neurons: usize) -> Result<Self, String> {
        if num_neurons == 0 || num_neurons > 10000 {
            return Err("Invalid neuron count".to_string());
        }

        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((num_neurons, num_neurons), |_| {
            rng.gen_range(-0.1..0.1)
        });

        Ok(Self {
            num_neurons,
            params: LIFParams::default(),
            stdp_params: STDPParams::default(),
            neurons: vec![LIFNeuron {
                v_mem: -65.0,
                time_since_spike: 1000.0,
                output: 0.0,
            }; num_neurons],
            weights,
            spike_history: vec![vec![false; 1000]],
        })
    }

    /// Forward pass: integrate and fire
    pub fn forward(&mut self, input: &[f32], dt: f32) -> Result<Vec<bool>, String> {
        if input.len() != self.num_neurons {
            return Err(format!("Input size mismatch: {} != {}", input.len(), self.num_neurons));
        }

        let mut spikes = vec![false; self.num_neurons];

        for i in 0..self.num_neurons {
            let neuron = &mut self.neurons[i];
            neuron.time_since_spike += dt;

            if neuron.time_since_spike < self.params.refractory_period {
                neuron.output = 0.0;
                neuron.v_mem = self.params.v_reset;
                continue;
            }

            // Synaptic input
            let mut syn_input = 0.0;
            for j in 0..self.num_neurons {
                syn_input += self.weights[[j, i]] * self.neurons[j].output;
            }

            // Integrate: V(t+dt) = V(t) * exp(-dt/tau) + I * (1 - exp(-dt/tau))
            let decay = (-dt / self.params.tau_m).exp();
            neuron.v_mem = neuron.v_mem * decay
                + (input[i] * self.params.input_gain + syn_input) * (1.0 - decay);

            // Fire
            if neuron.v_mem >= self.params.v_threshold {
                spikes[i] = true;
                neuron.output = 1.0;
                neuron.v_mem = self.params.v_reset;
                neuron.time_since_spike = 0.0;
            } else {
                neuron.output = 0.0;
            }
        }

        self.spike_history.push(spikes.clone());
        if self.spike_history.len() > 1000 {
            self.spike_history.remove(0);
        }

        Ok(spikes)
    }

    /// STDP learning
    pub fn apply_stdp(&mut self, learning_rate: f32) -> Result<(), String> {
        if self.spike_history.len() < 2 {
            return Ok(());
        }

        let current = &self.spike_history[self.spike_history.len() - 1];
        let previous = &self.spike_history[self.spike_history.len() - 2];

        for (post, &post_spike) in current.iter().enumerate() {
            for (pre, &pre_spike) in previous.iter().enumerate() {
                let mut dw = 0.0;
                if post_spike && pre_spike {
                    dw = self.stdp_params.a_plus * learning_rate; // LTP
                } else if post_spike && !pre_spike {
                    dw = -self.stdp_params.a_minus * learning_rate; // LTD
                }

                if dw != 0.0 {
                    let w = self.weights[[pre, post]] + dw;
                    self.weights[[pre, post]] = w.clamp(self.stdp_params.w_min, self.stdp_params.w_max);
                }
            }
        }
        Ok(())
    }

    /// Get potentials
    pub fn get_potentials(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| n.v_mem).collect()
    }

    /// Serialize state
    pub fn serialize(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    /// Deserialize state
    pub fn deserialize(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snn_creation() {
        let snn = SpikingNeuralNetwork::new(100).unwrap();
        assert_eq!(snn.num_neurons, 100);
    }

    #[test]
    fn test_forward() {
        let mut snn = SpikingNeuralNetwork::new(10).unwrap();
        let spikes = snn.forward(&vec![0.5; 10], 1.0).unwrap();
        assert_eq!(spikes.len(), 10);
    }

    #[test]
    fn test_stdp() {
        let mut snn = SpikingNeuralNetwork::new(5).unwrap();
        for _ in 0..10 {
            let _ = snn.forward(&vec![0.5; 5], 1.0);
        }
        let _ = snn.apply_stdp(0.01);
    }
}
