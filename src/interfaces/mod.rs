use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::errors::KambuzumaError;
use crate::metacognition::MetacognitiveOrchestrator;
use crate::neural::NeuralProcessingUnit;
use crate::quantum::QuantumMembraneSystem;

pub mod cli;
pub mod gui;
pub mod rest_api;
pub mod websocket;

/// Main interface manager for all external system interactions
#[derive(Clone)]
pub struct InterfaceManager {
    orchestrator: Arc<RwLock<MetacognitiveOrchestrator>>,
    neural_processor: Arc<RwLock<NeuralProcessingUnit>>,
    quantum_system: Arc<RwLock<QuantumMembraneSystem>>,
}

impl InterfaceManager {
    /// Create new interface manager
    pub fn new(
        orchestrator: Arc<RwLock<MetacognitiveOrchestrator>>,
        neural_processor: Arc<RwLock<NeuralProcessingUnit>>,
        quantum_system: Arc<RwLock<QuantumMembraneSystem>>,
    ) -> Self {
        Self {
            orchestrator,
            neural_processor,
            quantum_system,
        }
    }

    /// Start all interface services
    pub async fn start_all_services(&self, config: &InterfaceConfig) -> Result<(), KambuzumaError> {
        // Start REST API
        if config.rest_api.enabled {
            tokio::spawn(rest_api::start_server(self.clone(), config.rest_api.clone()));
        }

        // Start WebSocket service
        if config.websocket.enabled {
            tokio::spawn(websocket::start_server(self.clone(), config.websocket.clone()));
        }

        // Initialize CLI if needed
        if config.cli.enabled {
            cli::initialize(self.clone()).await?;
        }

        // Start GUI if needed
        if config.gui.enabled {
            tokio::spawn(gui::start_application(self.clone(), config.gui.clone()));
        }

        Ok(())
    }

    /// Get orchestrator reference
    pub fn orchestrator(&self) -> Arc<RwLock<MetacognitiveOrchestrator>> {
        self.orchestrator.clone()
    }

    /// Get neural processor reference
    pub fn neural_processor(&self) -> Arc<RwLock<NeuralProcessingUnit>> {
        self.neural_processor.clone()
    }

    /// Get quantum system reference
    pub fn quantum_system(&self) -> Arc<RwLock<QuantumMembraneSystem>> {
        self.quantum_system.clone()
    }
}

/// Configuration for all interface services
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterfaceConfig {
    pub rest_api: rest_api::RestApiConfig,
    pub websocket: websocket::WebSocketConfig,
    pub cli: cli::CliConfig,
    pub gui: gui::GuiConfig,
}

impl Default for InterfaceConfig {
    fn default() -> Self {
        Self {
            rest_api: rest_api::RestApiConfig::default(),
            websocket: websocket::WebSocketConfig::default(),
            cli: cli::CliConfig::default(),
            gui: gui::GuiConfig::default(),
        }
    }
}

/// System status information for interfaces
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemStatus {
    pub orchestrator_status: String,
    pub neural_processor_status: String,
    pub quantum_system_status: String,
    pub active_stages: Vec<u8>,
    pub thought_currents: Vec<f64>,
    pub quantum_coherence: f64,
    pub energy_levels: f64,
    pub processing_load: f64,
}

/// Query processing request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingRequest {
    pub query: String,
    pub context: Option<String>,
    pub parameters: Option<serde_json::Value>,
    pub priority: Option<u8>,
}

/// Query processing response
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingResponse {
    pub response: String,
    pub confidence: f64,
    pub processing_trace: Vec<String>,
    pub quantum_states: Vec<String>,
    pub thought_currents: Vec<f64>,
    pub energy_consumption: f64,
    pub processing_time_ms: u64,
}

/// Real-time monitoring data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonitoringData {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub system_status: SystemStatus,
    pub performance_metrics: PerformanceMetrics,
    pub quantum_measurements: QuantumMeasurements,
}

/// Performance metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_throughput: f64,
    pub response_times: Vec<f64>,
    pub error_rates: f64,
}

/// Quantum measurements
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumMeasurements {
    pub coherence_time: f64,
    pub entanglement_strength: f64,
    pub tunneling_probability: f64,
    pub energy_transfer_rate: f64,
    pub decoherence_rate: f64,
}
