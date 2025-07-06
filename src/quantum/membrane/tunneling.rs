use std::f64::consts::PI;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Physical constants for biological quantum tunneling
pub mod constants {
    /// Planck's constant (J⋅s)
    pub const PLANCK: f64 = 6.62607015e-34;
    
    /// Reduced Planck constant (ℏ = h/2π)
    pub const HBAR: f64 = PLANCK / (2.0 * std::f64::consts::PI);
    
    /// Electron mass (kg)
    pub const ELECTRON_MASS: f64 = 9.1093837015e-31;
    
    /// Elementary charge (C)
    pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;
    
    /// Typical membrane thickness (nm)
    pub const MEMBRANE_THICKNESS: f64 = 5.0e-9;
    
    /// Typical membrane potential barrier (eV)
    pub const MEMBRANE_BARRIER: f64 = 0.2;
    
    /// Room temperature thermal energy (eV)
    pub const THERMAL_ENERGY_RT: f64 = 0.025;
}

/// Quantum tunneling parameters for biological membranes
#[derive(Debug, Clone)]
pub struct TunnelingParameters {
    /// Membrane potential barrier height (eV)
    pub barrier_height: f64,
    
    /// Particle energy (eV)
    pub particle_energy: f64,
    
    /// Membrane thickness (m)
    pub membrane_thickness: f64,
    
    /// Effective particle mass (kg)
    pub effective_mass: f64,
    
    /// Temperature (K)
    pub temperature: f64,
}

impl Default for TunnelingParameters {
    fn default() -> Self {
        Self {
            barrier_height: constants::MEMBRANE_BARRIER,
            particle_energy: constants::THERMAL_ENERGY_RT,
            membrane_thickness: constants::MEMBRANE_THICKNESS,
            effective_mass: constants::ELECTRON_MASS,
            temperature: 310.15, // Body temperature in Kelvin
        }
    }
}

/// Quantum state representation for tunneling particles
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Wave function amplitude
    pub amplitude: f64,
    
    /// Phase (radians)
    pub phase: f64,
    
    /// Energy eigenvalue (eV)
    pub energy: f64,
    
    /// Probability density
    pub probability_density: f64,
}

impl QuantumState {
    pub fn new(amplitude: f64, phase: f64, energy: f64) -> Self {
        let probability_density = amplitude * amplitude;
        Self {
            amplitude,
            phase,
            energy,
            probability_density,
        }
    }
    
    /// Calculate quantum current density J = (ℏ/2mi) * (ψ*∇ψ - ψ∇ψ*)
    pub fn current_density(&self, gradient: f64) -> f64 {
        let imaginary_part = self.amplitude * self.amplitude * gradient * self.phase.sin();
        (constants::HBAR / (2.0 * constants::ELECTRON_MASS)) * imaginary_part
    }
}

/// Biological membrane quantum tunneling calculator
pub struct MembraneQuantumTunneling {
    /// Tunneling parameters
    parameters: Arc<RwLock<TunnelingParameters>>,
    
    /// Current quantum state
    quantum_state: Arc<RwLock<QuantumState>>,
    
    /// Tunneling probability cache
    probability_cache: Arc<RwLock<Option<f64>>>,
}

impl MembraneQuantumTunneling {
    pub fn new(parameters: TunnelingParameters) -> Self {
        let initial_state = QuantumState::new(1.0, 0.0, parameters.particle_energy);
        
        Self {
            parameters: Arc::new(RwLock::new(parameters)),
            quantum_state: Arc::new(RwLock::new(initial_state)),
            probability_cache: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Calculate transmission coefficient T = (1 + (V₀²sinh²(αd))/(4E(V₀-E)))⁻¹
    pub async fn transmission_coefficient(&self) -> Result<f64, TunnelingError> {
        let params = self.parameters.read().await;
        
        if params.particle_energy >= params.barrier_height {
            return Ok(1.0); // Classical transmission
        }
        
        if params.particle_energy <= 0.0 {
            return Ok(0.0); // No transmission for zero energy
        }
        
        // Calculate α = √(2m(V₀-E))/ℏ
        let energy_diff = params.barrier_height - params.particle_energy;
        let alpha = (2.0 * params.effective_mass * energy_diff * constants::ELEMENTARY_CHARGE / constants::HBAR.powi(2)).sqrt();
        
        // Calculate sinh²(αd)
        let alpha_d = alpha * params.membrane_thickness;
        let sinh_alpha_d = alpha_d.sinh();
        let sinh_squared = sinh_alpha_d * sinh_alpha_d;
        
        // Calculate transmission coefficient
        let barrier_squared = params.barrier_height * params.barrier_height;
        let energy_factor = 4.0 * params.particle_energy * energy_diff;
        
        if energy_factor == 0.0 {
            return Ok(0.0);
        }
        
        let denominator = 1.0 + (barrier_squared * sinh_squared) / energy_factor;
        let transmission = 1.0 / denominator;
        
        // Cache the result
        *self.probability_cache.write().await = Some(transmission);
        
        Ok(transmission)
    }
    
    /// Calculate reflection coefficient R = 1 - T
    pub async fn reflection_coefficient(&self) -> Result<f64, TunnelingError> {
        let transmission = self.transmission_coefficient().await?;
        Ok(1.0 - transmission)
    }
    
    /// Update quantum state based on tunneling probability
    pub async fn update_quantum_state(&self, time_step: f64) -> Result<(), TunnelingError> {
        let transmission = self.transmission_coefficient().await?;
        let params = self.parameters.read().await;
        
        // Calculate time evolution
        let energy_hz = params.particle_energy * constants::ELEMENTARY_CHARGE / constants::HBAR;
        let phase_evolution = energy_hz * time_step;
        
        let mut state = self.quantum_state.write().await;
        
        // Update amplitude based on tunneling probability
        state.amplitude *= transmission.sqrt();
        
        // Update phase
        state.phase += phase_evolution;
        state.phase %= 2.0 * PI;
        
        // Update probability density
        state.probability_density = state.amplitude * state.amplitude;
        
        // Update energy (with small thermal fluctuations)
        let thermal_fluctuation = (params.temperature / 300.0) * 0.001; // Small fluctuation
        state.energy = params.particle_energy * (1.0 + thermal_fluctuation * (time_step.sin()));
        
        Ok(())
    }
    
    /// Get current tunneling probability
    pub async fn tunneling_probability(&self) -> f64 {
        if let Some(cached) = *self.probability_cache.read().await {
            cached
        } else {
            self.transmission_coefficient().await.unwrap_or(0.0)
        }
    }
    
    /// Get current quantum state
    pub async fn quantum_state(&self) -> QuantumState {
        self.quantum_state.read().await.clone()
    }
    
    /// Update tunneling parameters
    pub async fn update_parameters(&self, new_params: TunnelingParameters) {
        *self.parameters.write().await = new_params;
        *self.probability_cache.write().await = None; // Invalidate cache
    }
    
    /// Calculate quantum current density
    pub async fn quantum_current_density(&self, position_gradient: f64) -> f64 {
        let state = self.quantum_state.read().await;
        state.current_density(position_gradient)
    }
    
    /// Biological validation: ensure parameters are within biological ranges
    pub async fn validate_biological_constraints(&self) -> Result<(), TunnelingError> {
        let params = self.parameters.read().await;
        
        // Check membrane thickness (biological range: 3-7 nm)
        if params.membrane_thickness < 3.0e-9 || params.membrane_thickness > 7.0e-9 {
            return Err(TunnelingError::BiologicalConstraintViolation(
                "Membrane thickness outside biological range (3-7 nm)".to_string()
            ));
        }
        
        // Check barrier height (biological range: 0.1-0.5 eV)
        if params.barrier_height < 0.1 || params.barrier_height > 0.5 {
            return Err(TunnelingError::BiologicalConstraintViolation(
                "Barrier height outside biological range (0.1-0.5 eV)".to_string()
            ));
        }
        
        // Check temperature (biological range: 273-323 K)
        if params.temperature < 273.0 || params.temperature > 323.0 {
            return Err(TunnelingError::BiologicalConstraintViolation(
                "Temperature outside biological range (273-323 K)".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Errors that can occur during quantum tunneling calculations
#[derive(Debug, thiserror::Error)]
pub enum TunnelingError {
    #[error("Mathematical calculation error: {0}")]
    CalculationError(String),
    
    #[error("Biological constraint violation: {0}")]
    BiologicalConstraintViolation(String),
    
    #[error("Quantum state evolution error: {0}")]
    StateEvolutionError(String),
    
    #[error("Parameter validation error: {0}")]
    ParameterError(String),
}

/// Tunneling configuration
#[derive(Debug, Clone)]
pub struct TunnelingConfig {
    /// Tunneling parameters
    pub parameters: TunnelingParameters,
    
    /// Update frequency (Hz)
    pub update_frequency: f64,
    
    /// Cache size limit
    pub cache_size_limit: usize,
    
    /// Biological constraint checking enabled
    pub biological_constraints_enabled: bool,
}

/// Tunneling state information
#[derive(Debug, Clone)]
pub struct TunnelingState {
    /// Current quantum state
    pub quantum_state: QuantumState,
    
    /// Current transmission coefficient
    pub transmission_coefficient: f64,
    
    /// Current reflection coefficient
    pub reflection_coefficient: f64,
    
    /// Energy consumption rate
    pub energy_consumption_rate: f64,
    
    /// Biological constraint status
    pub biological_constraints_valid: bool,
}

impl MembraneQuantumTunneling {
    /// Get current state
    pub async fn get_state(&self) -> Result<TunnelingState, TunnelingError> {
        let quantum_state = self.quantum_state().await;
        let transmission_coefficient = self.transmission_coefficient().await?;
        let reflection_coefficient = self.reflection_coefficient().await?;
        let energy_consumption_rate = self.calculate_energy_consumption_rate().await?;
        let biological_constraints_valid = self.validate_biological_constraints().await.is_ok();
        
        Ok(TunnelingState {
            quantum_state,
            transmission_coefficient,
            reflection_coefficient,
            energy_consumption_rate,
            biological_constraints_valid,
        })
    }
    
    /// Calculate energy consumption rate
    async fn calculate_energy_consumption_rate(&self) -> Result<f64, TunnelingError> {
        let params = self.parameters.read().await;
        let state = self.quantum_state.read().await;
        
        // Energy consumption based on tunneling probability and quantum state evolution
        let base_consumption = 1e-21; // Joules per second (very small for quantum processes)
        let tunneling_factor = self.tunneling_probability().await;
        let coherence_factor = state.probability_density;
        
        Ok(base_consumption * tunneling_factor * coherence_factor)
    }
}

impl Default for TunnelingConfig {
    fn default() -> Self {
        Self {
            parameters: TunnelingParameters::default(),
            update_frequency: 1e6, // 1 MHz
            cache_size_limit: 1000,
            biological_constraints_enabled: true,
        }
    }
}

impl TunnelingConfig {
    /// Validate configuration
    pub fn is_valid(&self) -> bool {
        self.update_frequency > 0.0 &&
        self.cache_size_limit > 0 &&
        self.parameters.barrier_height > 0.0 &&
        self.parameters.membrane_thickness > 0.0 &&
        self.parameters.effective_mass > 0.0 &&
        self.parameters.temperature > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
    #[tokio::test]
    async fn test_transmission_coefficient_calculation() {
        let params = TunnelingParameters::default();
        let tunneling = MembraneQuantumTunneling::new(params);
        
        let transmission = tunneling.transmission_coefficient().await.unwrap();
        
        // Should be small but non-zero for typical parameters
        assert!(transmission > 0.0);
        assert!(transmission < 1.0);
        assert!(transmission < 0.1); // Biological membranes have low transmission
    }
    
    #[tokio::test]
    async fn test_conservation_laws() {
        let params = TunnelingParameters::default();
        let tunneling = MembraneQuantumTunneling::new(params);
        
        let transmission = tunneling.transmission_coefficient().await.unwrap();
        let reflection = tunneling.reflection_coefficient().await.unwrap();
        
        // T + R = 1 (probability conservation)
        assert!((transmission + reflection - 1.0).abs() < 1e-10);
    }
    
    #[tokio::test]
    async fn test_biological_constraint_validation() {
        let mut params = TunnelingParameters::default();
        
        // Test invalid membrane thickness
        params.membrane_thickness = 1.0e-9; // Too thin
        let tunneling = MembraneQuantumTunneling::new(params);
        
        assert!(tunneling.validate_biological_constraints().await.is_err());
    }
    
    #[tokio::test]
    async fn test_quantum_state_evolution() {
        let params = TunnelingParameters::default();
        let tunneling = MembraneQuantumTunneling::new(params);
        
        let initial_state = tunneling.quantum_state().await;
        
        tunneling.update_quantum_state(1e-15).await.unwrap(); // 1 femtosecond
        
        let final_state = tunneling.quantum_state().await;
        
        // State should have evolved
        assert_ne!(initial_state.phase, final_state.phase);
    }
} 