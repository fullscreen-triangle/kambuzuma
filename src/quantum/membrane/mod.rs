//! Membrane quantum computing module
//! 
//! Implements quantum tunneling effects in biological membranes

pub mod tunneling;
pub mod phospholipid_bilayer;
pub mod ion_channels;
pub mod superposition;
pub mod decoherence;

use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;

/// Membrane system configuration
#[derive(Debug, Clone)]
pub struct MembraneConfig {
    /// Tunneling parameters
    pub tunneling_config: tunneling::TunnelingConfig,
    
    /// Phospholipid bilayer parameters
    pub bilayer_config: phospholipid_bilayer::BilayerConfig,
    
    /// Ion channel parameters
    pub ion_channels_config: ion_channels::IonChannelsConfig,
    
    /// Superposition parameters
    pub superposition_config: superposition::SuperpositionConfig,
    
    /// Decoherence parameters
    pub decoherence_config: decoherence::DecoherenceConfig,
}

/// Membrane system state
#[derive(Debug, Clone)]
pub struct MembraneState {
    /// Tunneling state
    pub tunneling_state: tunneling::TunnelingState,
    
    /// Bilayer state
    pub bilayer_state: phospholipid_bilayer::BilayerState,
    
    /// Ion channels state
    pub ion_channels_state: ion_channels::IonChannelsState,
    
    /// Superposition state
    pub superposition_state: superposition::SuperpositionState,
    
    /// Decoherence state
    pub decoherence_state: decoherence::DecoherenceState,
}

/// Membrane system orchestrator
pub struct MembraneSystem {
    /// Configuration
    config: MembraneConfig,
    
    /// Tunneling system
    tunneling_system: Arc<RwLock<tunneling::MembraneQuantumTunneling>>,
    
    /// Bilayer system
    bilayer_system: Arc<RwLock<phospholipid_bilayer::PhospholipidBilayer>>,
    
    /// Ion channels system
    ion_channels_system: Arc<RwLock<ion_channels::IonChannelsSystem>>,
    
    /// Superposition system
    superposition_system: Arc<RwLock<superposition::SuperpositionSystem>>,
    
    /// Decoherence system
    decoherence_system: Arc<RwLock<decoherence::DecoherenceSystem>>,
}

impl MembraneSystem {
    /// Create new membrane system
    pub fn new(config: &MembraneConfig) -> Result<Self, MembraneError> {
        let tunneling_system = Arc::new(RwLock::new(
            tunneling::MembraneQuantumTunneling::new(config.tunneling_config.parameters.clone())
        ));
        
        let bilayer_system = Arc::new(RwLock::new(
            phospholipid_bilayer::PhospholipidBilayer::new(&config.bilayer_config)?
        ));
        
        let ion_channels_system = Arc::new(RwLock::new(
            ion_channels::IonChannelsSystem::new(&config.ion_channels_config)?
        ));
        
        let superposition_system = Arc::new(RwLock::new(
            superposition::SuperpositionSystem::new(&config.superposition_config)?
        ));
        
        let decoherence_system = Arc::new(RwLock::new(
            decoherence::DecoherenceSystem::new(&config.decoherence_config)?
        ));
        
        Ok(Self {
            config: config.clone(),
            tunneling_system,
            bilayer_system,
            ion_channels_system,
            superposition_system,
            decoherence_system,
        })
    }
    
    /// Start membrane system
    pub async fn start(&self) -> Result<(), MembraneError> {
        // Systems start automatically
        Ok(())
    }
    
    /// Stop membrane system
    pub async fn stop(&self) -> Result<(), MembraneError> {
        // Systems stop automatically
        Ok(())
    }
    
    /// Get membrane state
    pub async fn get_state(&self) -> Result<MembraneState, MembraneError> {
        let tunneling_state = self.tunneling_system.read().await.get_state().await?;
        let bilayer_state = self.bilayer_system.read().await.get_state().await?;
        let ion_channels_state = self.ion_channels_system.read().await.get_state().await?;
        let superposition_state = self.superposition_system.read().await.get_state().await?;
        let decoherence_state = self.decoherence_system.read().await.get_state().await?;
        
        Ok(MembraneState {
            tunneling_state,
            bilayer_state,
            ion_channels_state,
            superposition_state,
            decoherence_state,
        })
    }
    
    /// Validate biological constraints
    pub async fn validate_biological_constraints(&self) -> Result<(), MembraneError> {
        self.tunneling_system.read().await.validate_biological_constraints().await
            .map_err(|e| MembraneError::BiologicalConstraintViolation(e.to_string()))?;
        
        self.bilayer_system.read().await.validate_biological_constraints().await?;
        self.ion_channels_system.read().await.validate_biological_constraints().await?;
        self.superposition_system.read().await.validate_biological_constraints().await?;
        self.decoherence_system.read().await.validate_biological_constraints().await?;
        
        Ok(())
    }
}

/// Membrane system errors
#[derive(Debug, Error)]
pub enum MembraneError {
    #[error("Tunneling error: {0}")]
    Tunneling(#[from] tunneling::TunnelingError),
    
    #[error("Bilayer error: {0}")]
    Bilayer(String),
    
    #[error("Ion channels error: {0}")]
    IonChannels(String),
    
    #[error("Superposition error: {0}")]
    Superposition(String),
    
    #[error("Decoherence error: {0}")]
    Decoherence(String),
    
    #[error("Biological constraint violation: {0}")]
    BiologicalConstraintViolation(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl Default for MembraneConfig {
    fn default() -> Self {
        Self {
            tunneling_config: tunneling::TunnelingConfig::default(),
            bilayer_config: phospholipid_bilayer::BilayerConfig::default(),
            ion_channels_config: ion_channels::IonChannelsConfig::default(),
            superposition_config: superposition::SuperpositionConfig::default(),
            decoherence_config: decoherence::DecoherenceConfig::default(),
        }
    }
}

impl MembraneConfig {
    /// Validate configuration
    pub fn is_valid(&self) -> bool {
        self.tunneling_config.is_valid() &&
        self.bilayer_config.is_valid() &&
        self.ion_channels_config.is_valid() &&
        self.superposition_config.is_valid() &&
        self.decoherence_config.is_valid()
    }
} 