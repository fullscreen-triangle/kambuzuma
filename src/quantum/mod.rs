//! Quantum computing subsystem for Kambuzuma
//! 
//! This module implements biological quantum computing through membrane tunneling
//! and Maxwell demon implementations using real quantum effects.

pub mod membrane;
pub mod oscillations;
pub mod maxwell_demon;
pub mod quantum_gates;
pub mod entanglement;
pub mod math_framework;

use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;

/// Quantum subsystem configuration
#[derive(Debug, Clone)]
pub struct QuantumConfig {
    /// Membrane tunneling parameters
    pub membrane_config: membrane::MembraneConfig,
    
    /// Maxwell demon parameters
    pub maxwell_demon_config: maxwell_demon::MaxwellDemonConfig,
    
    /// Quantum gate configuration
    pub quantum_gates_config: quantum_gates::QuantumGatesConfig,
    
    /// Entanglement parameters
    pub entanglement_config: entanglement::EntanglementConfig,
    
    /// Oscillation harvesting configuration
    pub oscillation_config: oscillations::OscillationConfig,
}

/// Overall quantum subsystem state
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Membrane tunneling state
    pub membrane_state: membrane::MembraneState,
    
    /// Maxwell demon state
    pub maxwell_demon_state: maxwell_demon::MaxwellDemonState,
    
    /// Quantum gate states
    pub quantum_gates_state: quantum_gates::QuantumGatesState,
    
    /// Entanglement network state
    pub entanglement_state: entanglement::EntanglementState,
    
    /// Oscillation harvesting state
    pub oscillation_state: oscillations::OscillationState,
}

/// Quantum subsystem orchestrator
pub struct QuantumSubsystem {
    /// Configuration
    config: QuantumConfig,
    
    /// Membrane tunneling system
    membrane_system: Arc<RwLock<membrane::MembraneSystem>>,
    
    /// Maxwell demon system
    maxwell_demon_system: Arc<RwLock<maxwell_demon::MaxwellDemonSystem>>,
    
    /// Quantum gates system
    quantum_gates_system: Arc<RwLock<quantum_gates::QuantumGatesSystem>>,
    
    /// Entanglement system
    entanglement_system: Arc<RwLock<entanglement::EntanglementSystem>>,
    
    /// Oscillation harvesting system
    oscillation_system: Arc<RwLock<oscillations::OscillationSystem>>,
}

impl QuantumSubsystem {
    /// Create new quantum subsystem
    pub fn new(config: &QuantumConfig) -> Result<Self, QuantumError> {
        let membrane_system = Arc::new(RwLock::new(
            membrane::MembraneSystem::new(&config.membrane_config)?
        ));
        
        let maxwell_demon_system = Arc::new(RwLock::new(
            maxwell_demon::MaxwellDemonSystem::new(&config.maxwell_demon_config)?
        ));
        
        let quantum_gates_system = Arc::new(RwLock::new(
            quantum_gates::QuantumGatesSystem::new(&config.quantum_gates_config)?
        ));
        
        let entanglement_system = Arc::new(RwLock::new(
            entanglement::EntanglementSystem::new(&config.entanglement_config)?
        ));
        
        let oscillation_system = Arc::new(RwLock::new(
            oscillations::OscillationSystem::new(&config.oscillation_config)?
        ));
        
        Ok(Self {
            config: config.clone(),
            membrane_system,
            maxwell_demon_system,
            quantum_gates_system,
            entanglement_system,
            oscillation_system,
        })
    }
    
    /// Start the quantum subsystem
    pub async fn start(&self) -> Result<(), QuantumError> {
        self.membrane_system.write().await.start().await?;
        self.maxwell_demon_system.write().await.start().await?;
        self.quantum_gates_system.write().await.start().await?;
        self.entanglement_system.write().await.start().await?;
        self.oscillation_system.write().await.start().await?;
        
        Ok(())
    }
    
    /// Stop the quantum subsystem
    pub async fn stop(&self) -> Result<(), QuantumError> {
        self.oscillation_system.write().await.stop().await?;
        self.entanglement_system.write().await.stop().await?;
        self.quantum_gates_system.write().await.stop().await?;
        self.maxwell_demon_system.write().await.stop().await?;
        self.membrane_system.write().await.stop().await?;
        
        Ok(())
    }
    
    /// Get current quantum state
    pub async fn get_state(&self) -> Result<QuantumState, QuantumError> {
        let membrane_state = self.membrane_system.read().await.get_state().await?;
        let maxwell_demon_state = self.maxwell_demon_system.read().await.get_state().await?;
        let quantum_gates_state = self.quantum_gates_system.read().await.get_state().await?;
        let entanglement_state = self.entanglement_system.read().await.get_state().await?;
        let oscillation_state = self.oscillation_system.read().await.get_state().await?;
        
        Ok(QuantumState {
            membrane_state,
            maxwell_demon_state,
            quantum_gates_state,
            entanglement_state,
            oscillation_state,
        })
    }
    
    /// Validate biological constraints
    pub async fn validate_biological_constraints(&self) -> Result<(), QuantumError> {
        self.membrane_system.read().await.validate_biological_constraints().await?;
        self.maxwell_demon_system.read().await.validate_biological_constraints().await?;
        self.quantum_gates_system.read().await.validate_biological_constraints().await?;
        self.entanglement_system.read().await.validate_biological_constraints().await?;
        self.oscillation_system.read().await.validate_biological_constraints().await?;
        
        Ok(())
    }
}

/// Quantum subsystem errors
#[derive(Debug, Error)]
pub enum QuantumError {
    #[error("Membrane system error: {0}")]
    MembraneSystem(#[from] membrane::MembraneError),
    
    #[error("Maxwell demon system error: {0}")]
    MaxwellDemonSystem(#[from] maxwell_demon::MaxwellDemonError),
    
    #[error("Quantum gates system error: {0}")]
    QuantumGatesSystem(#[from] quantum_gates::QuantumGatesError),
    
    #[error("Entanglement system error: {0}")]
    EntanglementSystem(#[from] entanglement::EntanglementError),
    
    #[error("Oscillation system error: {0}")]
    OscillationSystem(#[from] oscillations::OscillationError),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("System initialization error: {0}")]
    SystemInitialization(String),
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            membrane_config: membrane::MembraneConfig::default(),
            maxwell_demon_config: maxwell_demon::MaxwellDemonConfig::default(),
            quantum_gates_config: quantum_gates::QuantumGatesConfig::default(),
            entanglement_config: entanglement::EntanglementConfig::default(),
            oscillation_config: oscillations::OscillationConfig::default(),
        }
    }
}

impl QuantumConfig {
    /// Validate configuration
    pub fn is_valid(&self) -> bool {
        self.membrane_config.is_valid() &&
        self.maxwell_demon_config.is_valid() &&
        self.quantum_gates_config.is_valid() &&
        self.entanglement_config.is_valid() &&
        self.oscillation_config.is_valid()
    }
} 