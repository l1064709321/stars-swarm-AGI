//! Module 2: 认知轨迹卡尔曼追迹器 (Adaptive Kalman Filter)
//!
//! Adaptive Kalman filtering for cognitive trajectory tracking.
//! Ref: Welch & Bishop (2006) "An Introduction to the Kalman Filter"
//!
//! Prediction: x̂⁻ = A*x̂ + B*u,  P⁻ = A*P*A^T + Q
//! Update: K = P⁻*H^T*(H*P⁻*H^T + R)^-1, x̂ = x̂⁻ + K*(z - H*x̂⁻)

use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};

/// Kalman Filter Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanParams {
    pub state_dim: usize,
    pub measurement_dim: usize,
    pub q_initial: f32,  // Process noise
    pub r_initial: f32,  // Measurement noise
}

impl Default for KalmanParams {
    fn default() -> Self {
        Self {
            state_dim: 4,        // [x, y, vx, vy]
            measurement_dim: 2,  // [x, y]
            q_initial: 0.01,
            r_initial: 0.1,
        }
    }
}

/// Adaptive Kalman Filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveKalmanFilter {
    state: Array1<f32>,        // x̂
    covariance: Array2<f32>,   // P
    transition: Array2<f32>,   // A (constant velocity model)
    measurement: Array2<f32>,  // H
    q_matrix: Array2<f32>,    // Q
    r_matrix: Array2<f32>,    // R
    r_adaptive: f32,
    dt: f32,
}

impl AdaptiveKalmanFilter {
    /// Create new Kalman filter
    pub fn new(params: KalmanParams, dt: f32) -> Result<Self, String> {
        if params.state_dim == 0 || params.measurement_dim == 0 {
            return Err("Invalid dimensions".to_string());
        }

        // State: [x, y, vx, vy]
        let state = Array1::zeros(params.state_dim);
        let covariance = Array2::eye(params.state_dim);

        // Constant velocity model: x(k+1) = x(k) + v(k)*dt
        let mut transition = Array2::eye(params.state_dim);
        if params.state_dim >= 4 {
            transition[[0, 2]] = dt; // x += vx*dt
            transition[[1, 3]] = dt; // y += vy*dt
        }

        // Measure position
        let mut measurement = Array2::zeros((params.measurement_dim, params.state_dim));
        measurement[[0, 0]] = 1.0;
        if params.measurement_dim >= 2 {
            measurement[[1, 1]] = 1.0;
        }

        let q_matrix = Array2::eye(params.state_dim) * params.q_initial;
        let r_matrix = Array2::eye(params.measurement_dim) * params.r_initial;

        Ok(Self {
            state,
            covariance,
            transition,
            measurement,
            q_matrix,
            r_matrix,
            r_adaptive: params.r_initial,
            dt,
        })
    }

    /// Predict: x̂⁻ = A*x̂, P⁻ = A*P*A^T + Q
    fn predict(&mut self) {
        self.state = self.transition.dot(&self.state);
        let apt = self.transition.dot(&self.covariance.dot(&self.transition.t()));
        self.covariance = apt + &self.q_matrix;
    }

    /// Update: Kalman gain and state correction
    fn update(&mut self, measurement: &[f32]) -> Result<(), String> {
        if measurement.len() != self.measurement.nrows() {
            return Err("Measurement size mismatch".to_string());
        }

        let z = Array1::from_vec(measurement.to_vec());
        let hx = self.measurement.dot(&self.state);
        let innovation = z - hx;

        // S = H*P*H^T + R
        let hph_t = self.measurement.dot(&self.covariance.dot(&self.measurement.t()));
        let s = hph_t + &self.r_matrix;

        // Compute inverse: S^-1 (numerically stable)
        let s_inv_2x2 = if self.measurement.nrows() == 2 {
            let det = s[[0, 0]] * s[[1, 1]] - s[[0, 1]] * s[[1, 0]];
            if det.abs() < 1e-10 {
                return Err("Singular innovation covariance".to_string());
            }
            let mut inv = Array2::zeros((2, 2));
            inv[[0, 0]] = s[[1, 1]] / det;
            inv[[0, 1]] = -s[[0, 1]] / det;
            inv[[1, 0]] = -s[[1, 0]] / det;
            inv[[1, 1]] = s[[0, 0]] / det;
            inv
        } else {
            return Err("Only 2D measurements supported".to_string());
        };

        // K = P*H^T*S^-1
        let ph_t = self.covariance.dot(&self.measurement.t());
        let k = ph_t.dot(&s_inv_2x2);

        // x̂ = x̂ + K*(z - H*x̂)
        self.state = &self.state + &k.dot(&innovation);

        // P = (I - K*H)*P  — Joseph form for numerical stability
        // P' = (I-KH)*P*(I-KH)^T + K*R*K^T  (guarantees positive semi-definite)
        let kh = k.dot(&self.measurement);
        let eye = Array2::eye(kh.nrows());
        let ikh = &eye - &kh;
        let ikh_t = ikh.t();
        let p_new = ikh.dot(&self.covariance.dot(&ikh_t))
            + &k.dot(&self.r_matrix.dot(&k.t()));
        self.covariance = p_new;

        // Adaptive R: increase if innovation is large
        let innovation_norm = innovation.iter().map(|v| v * v).sum::<f32>().sqrt();
        self.r_adaptive *= 1.0 + 0.001 * (innovation_norm - 1.0);
        self.r_adaptive = self.r_adaptive.clamp(0.001, 10.0);
        self.r_matrix[[0, 0]] = self.r_adaptive;
        if self.r_matrix.nrows() > 1 {
            self.r_matrix[[1, 1]] = self.r_adaptive;
        }

        Ok(())
    }

    /// Full filter cycle
    pub fn filter(&mut self, measurement: &[f32]) -> Result<Array1<f32>, String> {
        self.predict();
        self.update(measurement)?;
        Ok(self.state.clone())
    }

    /// Get current estimate
    pub fn get_state(&self) -> Array1<f32> {
        self.state.clone()
    }

    /// Get uncertainty (trace of covariance)
    pub fn get_uncertainty(&self) -> f32 {
        self.covariance.diag().sum()
    }

    /// Serialize
    pub fn serialize(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    /// Deserialize
    pub fn deserialize(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_creation() {
        let kf = AdaptiveKalmanFilter::new(KalmanParams::default(), 0.1).unwrap();
        assert_eq!(kf.state.len(), 4);
    }

    #[test]
    fn test_kalman_filter_cycle() {
        let mut kf = AdaptiveKalmanFilter::new(KalmanParams::default(), 0.1).unwrap();
        let measurement = vec![1.0, 2.0];
        let state = kf.filter(&measurement).unwrap();
        assert_eq!(state.len(), 4);
    }

    #[test]
    fn test_adaptive_adjustment() {
        let mut kf = AdaptiveKalmanFilter::new(KalmanParams::default(), 0.1).unwrap();
        let initial_r = kf.r_adaptive;
        for _ in 0..5 {
            let _ = kf.filter(&vec![10.0, 20.0]);
        }
        assert_ne!(kf.r_adaptive, initial_r);
    }
}
