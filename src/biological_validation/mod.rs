//! # Biological Validation Module
//!
//! This module provides experimental verification systems for the quantum biological
//! processes implemented in the Kambuzuma system. It includes cell culture simulation,
//! electrophysiology measurements, biochemical assays, and quantum validation protocols.
//!
//! ## Key Components
//! - Cell culture array management and monitoring
//! - Electrophysiological measurement systems
//! - Biochemical assay protocols
//! - Quantum effect validation
//! - Safety and containment systems

use crate::config::Config;
use crate::errors::KambuzumaError;
use crate::types::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub mod biochemical_assays;
pub mod cell_culture;
pub mod electrophysiology;
pub mod quantum_validation;
pub mod safety_protocols;

pub use biochemical_assays::*;
pub use cell_culture::*;
pub use electrophysiology::*;
pub use quantum_validation::*;
pub use safety_protocols::*;

/// Biological Validation System
/// Main system for biological validation and experimental verification
#[derive(Debug)]
pub struct BiologicalValidationSystem {
    /// System identifier
    pub id: Uuid,
    /// System configuration
    pub config: Arc<RwLock<Config>>,
    /// Cell culture system
    pub cell_culture: Arc<RwLock<CellCultureSystem>>,
    /// Electrophysiology system
    pub electrophysiology: Arc<RwLock<ElectrophysiologySystem>>,
    /// Biochemical assay system
    pub biochemical_assays: Arc<RwLock<BiochemicalAssaySystem>>,
    /// Quantum validation system
    pub quantum_validation: Arc<RwLock<QuantumValidationSystem>>,
    /// Safety protocols
    pub safety_protocols: Arc<RwLock<SafetyProtocolSystem>>,
    /// Validation experiments
    pub experiments: Arc<RwLock<Vec<ValidationExperiment>>>,
    /// System status
    pub status: Arc<RwLock<ValidationSystemStatus>>,
    /// Validation metrics
    pub metrics: Arc<RwLock<ValidationMetrics>>,
}

/// Validation System Status
/// Current status of the validation system
#[derive(Debug, Clone)]
pub struct ValidationSystemStatus {
    /// System state
    pub state: ValidationState,
    /// Active experiments
    pub active_experiments: usize,
    /// System health
    pub health: f64,
    /// Error count
    pub error_count: usize,
    /// Last update time
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// Validation State
/// States of the validation system
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationState {
    /// System is offline
    Offline,
    /// System is initializing
    Initializing,
    /// System is ready
    Ready,
    /// System is running experiments
    Running,
    /// System is in maintenance mode
    Maintenance,
    /// System has errors
    Error,
    /// System is shutting down
    Shutdown,
}

/// Validation Metrics
/// Metrics for validation system performance
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Total experiments run
    pub total_experiments: u64,
    /// Successful experiments
    pub successful_experiments: u64,
    /// Failed experiments
    pub failed_experiments: u64,
    /// Average experiment duration
    pub average_duration: f64,
    /// System uptime
    pub uptime: f64,
    /// Data quality score
    pub data_quality: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Validation Experiment
/// Individual validation experiment
#[derive(Debug, Clone)]
pub struct ValidationExperiment {
    /// Experiment identifier
    pub id: Uuid,
    /// Experiment name
    pub name: String,
    /// Experiment type
    pub experiment_type: ExperimentType,
    /// Experiment parameters
    pub parameters: ExperimentParameters,
    /// Experiment protocol
    pub protocol: ExperimentProtocol,
    /// Experiment status
    pub status: ExperimentStatus,
    /// Experiment results
    pub results: Option<ExperimentResults>,
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// End time
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Duration
    pub duration: Option<f64>,
}

/// Experiment Type
/// Types of validation experiments
#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentType {
    /// Cell viability test
    CellViability,
    /// Membrane potential measurement
    MembranePotential,
    /// Ion channel activity
    IonChannelActivity,
    /// ATP synthesis measurement
    AtpSynthesis,
    /// Protein concentration assay
    ProteinAssay,
    /// Enzyme activity assay
    EnzymeActivity,
    /// Quantum coherence measurement
    QuantumCoherence,
    /// Entanglement verification
    EntanglementVerification,
    /// Bell test validation
    BellTest,
    /// Tunnel current measurement
    TunnelCurrent,
    /// Oscillation harvesting
    OscillationHarvesting,
    /// Safety validation
    SafetyValidation,
}

/// Experiment Parameters
/// Parameters for validation experiments
#[derive(Debug, Clone)]
pub struct ExperimentParameters {
    /// Parameter values
    pub values: HashMap<String, f64>,
    /// Parameter units
    pub units: HashMap<String, String>,
    /// Parameter ranges
    pub ranges: HashMap<String, (f64, f64)>,
    /// Parameter constraints
    pub constraints: HashMap<String, String>,
}

/// Experiment Protocol
/// Protocol for conducting experiments
#[derive(Debug, Clone)]
pub struct ExperimentProtocol {
    /// Protocol steps
    pub steps: Vec<ProtocolStep>,
    /// Protocol duration
    pub duration: f64,
    /// Protocol requirements
    pub requirements: Vec<String>,
    /// Protocol safety measures
    pub safety_measures: Vec<String>,
}

/// Protocol Step
/// Individual step in experiment protocol
#[derive(Debug, Clone)]
pub struct ProtocolStep {
    /// Step identifier
    pub id: Uuid,
    /// Step number
    pub step_number: usize,
    /// Step description
    pub description: String,
    /// Step duration
    pub duration: f64,
    /// Step parameters
    pub parameters: HashMap<String, f64>,
    /// Step requirements
    pub requirements: Vec<String>,
    /// Step completion status
    pub completed: bool,
}

/// Experiment Status
/// Status of validation experiment
#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentStatus {
    /// Experiment is queued
    Queued,
    /// Experiment is running
    Running,
    /// Experiment is paused
    Paused,
    /// Experiment completed successfully
    Completed,
    /// Experiment failed
    Failed,
    /// Experiment was cancelled
    Cancelled,
    /// Experiment is being analyzed
    Analyzing,
}

/// Experiment Results
/// Results from validation experiment
#[derive(Debug, Clone)]
pub struct ExperimentResults {
    /// Result identifier
    pub id: Uuid,
    /// Raw data
    pub raw_data: Vec<f64>,
    /// Processed data
    pub processed_data: HashMap<String, f64>,
    /// Statistical analysis
    pub statistics: StatisticalAnalysis,
    /// Validation outcome
    pub outcome: ValidationOutcome,
    /// Confidence level
    pub confidence: f64,
    /// Error analysis
    pub error_analysis: ErrorAnalysis,
}

/// Statistical Analysis
/// Statistical analysis of experiment results
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    /// Sample size
    pub sample_size: usize,
    /// Mean values
    pub means: HashMap<String, f64>,
    /// Standard deviations
    pub std_devs: HashMap<String, f64>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// P-values
    pub p_values: HashMap<String, f64>,
    /// Effect sizes
    pub effect_sizes: HashMap<String, f64>,
}

/// Validation Outcome
/// Outcome of validation experiment
#[derive(Debug, Clone)]
pub struct ValidationOutcome {
    /// Outcome type
    pub outcome_type: OutcomeType,
    /// Validation passed
    pub passed: bool,
    /// Validation score
    pub score: f64,
    /// Validation message
    pub message: String,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Outcome Type
/// Types of validation outcomes
#[derive(Debug, Clone, PartialEq)]
pub enum OutcomeType {
    /// Hypothesis confirmed
    Confirmed,
    /// Hypothesis rejected
    Rejected,
    /// Inconclusive results
    Inconclusive,
    /// Partial confirmation
    Partial,
    /// Requires further investigation
    Investigation,
}

/// Error Analysis
/// Analysis of experimental errors
#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    /// Systematic errors
    pub systematic_errors: Vec<String>,
    /// Random errors
    pub random_errors: Vec<String>,
    /// Error magnitudes
    pub error_magnitudes: HashMap<String, f64>,
    /// Error sources
    pub error_sources: HashMap<String, String>,
    /// Error mitigation
    pub mitigation_strategies: Vec<String>,
}

impl BiologicalValidationSystem {
    /// Create new biological validation system
    pub async fn new(config: Arc<RwLock<Config>>) -> Result<Self, KambuzumaError> {
        log::info!("Creating biological validation system");

        let id = Uuid::new_v4();

        // Initialize subsystems
        let cell_culture = Arc::new(RwLock::new(CellCultureSystem::new(config.clone()).await?));
        let electrophysiology = Arc::new(RwLock::new(ElectrophysiologySystem::new(config.clone()).await?));
        let biochemical_assays = Arc::new(RwLock::new(BiochemicalAssaySystem::new(config.clone()).await?));
        let quantum_validation = Arc::new(RwLock::new(QuantumValidationSystem::new(config.clone()).await?));
        let safety_protocols = Arc::new(RwLock::new(SafetyProtocolSystem::new(config.clone()).await?));

        // Initialize empty experiment list
        let experiments = Arc::new(RwLock::new(Vec::new()));

        // Initialize system status
        let status = Arc::new(RwLock::new(ValidationSystemStatus {
            state: ValidationState::Offline,
            active_experiments: 0,
            health: 1.0,
            error_count: 0,
            last_update: chrono::Utc::now(),
        }));

        // Initialize metrics
        let metrics = Arc::new(RwLock::new(ValidationMetrics {
            total_experiments: 0,
            successful_experiments: 0,
            failed_experiments: 0,
            average_duration: 0.0,
            uptime: 0.0,
            data_quality: 1.0,
            validation_accuracy: 1.0,
            error_rate: 0.0,
        }));

        Ok(Self {
            id,
            config,
            cell_culture,
            electrophysiology,
            biochemical_assays,
            quantum_validation,
            safety_protocols,
            experiments,
            status,
            metrics,
        })
    }

    /// Initialize the validation system
    pub async fn initialize(&self) -> Result<(), KambuzumaError> {
        log::info!("Initializing biological validation system");

        // Update status
        {
            let mut status = self.status.write().await;
            status.state = ValidationState::Initializing;
            status.last_update = chrono::Utc::now();
        }

        // Initialize safety protocols first
        {
            let mut safety = self.safety_protocols.write().await;
            safety.initialize().await?;
        }

        // Initialize subsystems
        {
            let mut cell_culture = self.cell_culture.write().await;
            cell_culture.initialize().await?;
        }

        {
            let mut electrophysiology = self.electrophysiology.write().await;
            electrophysiology.initialize().await?;
        }

        {
            let mut biochemical = self.biochemical_assays.write().await;
            biochemical.initialize().await?;
        }

        {
            let mut quantum = self.quantum_validation.write().await;
            quantum.initialize().await?;
        }

        // Update status to ready
        {
            let mut status = self.status.write().await;
            status.state = ValidationState::Ready;
            status.health = 1.0;
            status.last_update = chrono::Utc::now();
        }

        log::info!("Biological validation system initialized successfully");
        Ok(())
    }

    /// Shutdown the validation system
    pub async fn shutdown(&self) -> Result<(), KambuzumaError> {
        log::info!("Shutting down biological validation system");

        // Update status
        {
            let mut status = self.status.write().await;
            status.state = ValidationState::Shutdown;
            status.last_update = chrono::Utc::now();
        }

        // Shutdown subsystems
        {
            let mut quantum = self.quantum_validation.write().await;
            quantum.shutdown().await?;
        }

        {
            let mut biochemical = self.biochemical_assays.write().await;
            biochemical.shutdown().await?;
        }

        {
            let mut electrophysiology = self.electrophysiology.write().await;
            electrophysiology.shutdown().await?;
        }

        {
            let mut cell_culture = self.cell_culture.write().await;
            cell_culture.shutdown().await?;
        }

        {
            let mut safety = self.safety_protocols.write().await;
            safety.shutdown().await?;
        }

        // Update status to offline
        {
            let mut status = self.status.write().await;
            status.state = ValidationState::Offline;
            status.last_update = chrono::Utc::now();
        }

        log::info!("Biological validation system shutdown complete");
        Ok(())
    }

    /// Run validation experiment
    pub async fn run_experiment(&self, experiment: ValidationExperiment) -> Result<ExperimentResults, KambuzumaError> {
        log::info!("Running validation experiment: {}", experiment.name);

        // Check system readiness
        {
            let status = self.status.read().await;
            if status.state != ValidationState::Ready {
                return Err(KambuzumaError::ValidationError(
                    "System not ready for experiments".to_string(),
                ));
            }
        }

        // Update system state
        {
            let mut status = self.status.write().await;
            status.state = ValidationState::Running;
            status.active_experiments += 1;
            status.last_update = chrono::Utc::now();
        }

        // Add experiment to list
        {
            let mut experiments = self.experiments.write().await;
            experiments.push(experiment.clone());
        }

        // Run experiment based on type
        let results = match experiment.experiment_type {
            ExperimentType::CellViability => {
                let cell_culture = self.cell_culture.read().await;
                cell_culture.run_viability_test(&experiment.parameters).await?
            },
            ExperimentType::MembranePotential => {
                let electrophysiology = self.electrophysiology.read().await;
                electrophysiology.measure_membrane_potential(&experiment.parameters).await?
            },
            ExperimentType::IonChannelActivity => {
                let electrophysiology = self.electrophysiology.read().await;
                electrophysiology.measure_ion_channel_activity(&experiment.parameters).await?
            },
            ExperimentType::AtpSynthesis => {
                let biochemical = self.biochemical_assays.read().await;
                biochemical.measure_atp_synthesis(&experiment.parameters).await?
            },
            ExperimentType::ProteinAssay => {
                let biochemical = self.biochemical_assays.read().await;
                biochemical.run_protein_assay(&experiment.parameters).await?
            },
            ExperimentType::EnzymeActivity => {
                let biochemical = self.biochemical_assays.read().await;
                biochemical.measure_enzyme_activity(&experiment.parameters).await?
            },
            ExperimentType::QuantumCoherence => {
                let quantum = self.quantum_validation.read().await;
                quantum.measure_quantum_coherence(&experiment.parameters).await?
            },
            ExperimentType::EntanglementVerification => {
                let quantum = self.quantum_validation.read().await;
                quantum.verify_entanglement(&experiment.parameters).await?
            },
            ExperimentType::BellTest => {
                let quantum = self.quantum_validation.read().await;
                quantum.run_bell_test(&experiment.parameters).await?
            },
            ExperimentType::TunnelCurrent => {
                let electrophysiology = self.electrophysiology.read().await;
                electrophysiology.measure_tunnel_current(&experiment.parameters).await?
            },
            ExperimentType::OscillationHarvesting => {
                let electrophysiology = self.electrophysiology.read().await;
                electrophysiology.measure_oscillation_harvesting(&experiment.parameters).await?
            },
            ExperimentType::SafetyValidation => {
                let safety = self.safety_protocols.read().await;
                safety.run_safety_validation(&experiment.parameters).await?
            },
        };

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_experiments += 1;
            if results.outcome.passed {
                metrics.successful_experiments += 1;
            } else {
                metrics.failed_experiments += 1;
            }
            metrics.error_rate = metrics.failed_experiments as f64 / metrics.total_experiments as f64;
        }

        // Update system state
        {
            let mut status = self.status.write().await;
            status.active_experiments -= 1;
            if status.active_experiments == 0 {
                status.state = ValidationState::Ready;
            }
            status.last_update = chrono::Utc::now();
        }

        log::info!("Experiment completed: {}", experiment.name);
        Ok(results)
    }

    /// Get system status
    pub async fn get_status(&self) -> ValidationSystemStatus {
        self.status.read().await.clone()
    }

    /// Get system metrics
    pub async fn get_metrics(&self) -> ValidationMetrics {
        self.metrics.read().await.clone()
    }

    /// Get experiment history
    pub async fn get_experiments(&self) -> Vec<ValidationExperiment> {
        self.experiments.read().await.clone()
    }

    /// Perform health check
    pub async fn health_check(&self) -> Result<f64, KambuzumaError> {
        log::debug!("Performing system health check");

        let mut health_score = 1.0;
        let mut checks = 0;

        // Check cell culture system
        {
            let cell_culture = self.cell_culture.read().await;
            let cell_health = cell_culture.health_check().await?;
            health_score *= cell_health;
            checks += 1;
        }

        // Check electrophysiology system
        {
            let electrophysiology = self.electrophysiology.read().await;
            let electro_health = electrophysiology.health_check().await?;
            health_score *= electro_health;
            checks += 1;
        }

        // Check biochemical assay system
        {
            let biochemical = self.biochemical_assays.read().await;
            let biochemical_health = biochemical.health_check().await?;
            health_score *= biochemical_health;
            checks += 1;
        }

        // Check quantum validation system
        {
            let quantum = self.quantum_validation.read().await;
            let quantum_health = quantum.health_check().await?;
            health_score *= quantum_health;
            checks += 1;
        }

        // Check safety protocols
        {
            let safety = self.safety_protocols.read().await;
            let safety_health = safety.health_check().await?;
            health_score *= safety_health;
            checks += 1;
        }

        // Calculate overall health
        let overall_health = health_score.powf(1.0 / checks as f64);

        // Update system health
        {
            let mut status = self.status.write().await;
            status.health = overall_health;
            status.last_update = chrono::Utc::now();
        }

        log::debug!("System health check completed: {:.2}%", overall_health * 100.0);
        Ok(overall_health)
    }

    /// Emergency shutdown
    pub async fn emergency_shutdown(&self) -> Result<(), KambuzumaError> {
        log::warn!("Emergency shutdown initiated");

        // Activate emergency protocols
        {
            let safety = self.safety_protocols.read().await;
            safety.emergency_shutdown().await?;
        }

        // Shutdown all subsystems immediately
        self.shutdown().await?;

        log::warn!("Emergency shutdown completed");
        Ok(())
    }
}

impl Default for ValidationMetrics {
    fn default() -> Self {
        Self {
            total_experiments: 0,
            successful_experiments: 0,
            failed_experiments: 0,
            average_duration: 0.0,
            uptime: 0.0,
            data_quality: 1.0,
            validation_accuracy: 1.0,
            error_rate: 0.0,
        }
    }
}

impl Default for ExperimentParameters {
    fn default() -> Self {
        Self {
            values: HashMap::new(),
            units: HashMap::new(),
            ranges: HashMap::new(),
            constraints: HashMap::new(),
        }
    }
}

impl ValidationExperiment {
    /// Create new validation experiment
    pub fn new(
        name: String,
        experiment_type: ExperimentType,
        parameters: ExperimentParameters,
        protocol: ExperimentProtocol,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            experiment_type,
            parameters,
            protocol,
            status: ExperimentStatus::Queued,
            results: None,
            start_time: chrono::Utc::now(),
            end_time: None,
            duration: None,
        }
    }

    /// Start experiment
    pub fn start(&mut self) {
        self.status = ExperimentStatus::Running;
        self.start_time = chrono::Utc::now();
    }

    /// Complete experiment
    pub fn complete(&mut self, results: ExperimentResults) {
        self.status = ExperimentStatus::Completed;
        self.end_time = Some(chrono::Utc::now());
        self.duration = Some((self.end_time.unwrap() - self.start_time).num_milliseconds() as f64 / 1000.0);
        self.results = Some(results);
    }

    /// Fail experiment
    pub fn fail(&mut self, error: String) {
        self.status = ExperimentStatus::Failed;
        self.end_time = Some(chrono::Utc::now());
        self.duration = Some((self.end_time.unwrap() - self.start_time).num_milliseconds() as f64 / 1000.0);

        // Create error result
        let error_result = ExperimentResults {
            id: Uuid::new_v4(),
            raw_data: Vec::new(),
            processed_data: HashMap::new(),
            statistics: StatisticalAnalysis {
                sample_size: 0,
                means: HashMap::new(),
                std_devs: HashMap::new(),
                confidence_intervals: HashMap::new(),
                p_values: HashMap::new(),
                effect_sizes: HashMap::new(),
            },
            outcome: ValidationOutcome {
                outcome_type: OutcomeType::Rejected,
                passed: false,
                score: 0.0,
                message: error,
                evidence: Vec::new(),
            },
            confidence: 0.0,
            error_analysis: ErrorAnalysis {
                systematic_errors: Vec::new(),
                random_errors: Vec::new(),
                error_magnitudes: HashMap::new(),
                error_sources: HashMap::new(),
                mitigation_strategies: Vec::new(),
            },
        };

        self.results = Some(error_result);
    }

    /// Cancel experiment
    pub fn cancel(&mut self) {
        self.status = ExperimentStatus::Cancelled;
        self.end_time = Some(chrono::Utc::now());
        self.duration = Some((self.end_time.unwrap() - self.start_time).num_milliseconds() as f64 / 1000.0);
    }
}

/// Validation experiment builder
pub struct ExperimentBuilder {
    name: String,
    experiment_type: ExperimentType,
    parameters: ExperimentParameters,
    protocol: ExperimentProtocol,
}

impl ExperimentBuilder {
    /// Create new experiment builder
    pub fn new(name: String, experiment_type: ExperimentType) -> Self {
        Self {
            name,
            experiment_type,
            parameters: ExperimentParameters::default(),
            protocol: ExperimentProtocol {
                steps: Vec::new(),
                duration: 0.0,
                requirements: Vec::new(),
                safety_measures: Vec::new(),
            },
        }
    }

    /// Add parameter
    pub fn parameter(mut self, name: String, value: f64, unit: String) -> Self {
        self.parameters.values.insert(name.clone(), value);
        self.parameters.units.insert(name, unit);
        self
    }

    /// Add parameter range
    pub fn parameter_range(mut self, name: String, min: f64, max: f64) -> Self {
        self.parameters.ranges.insert(name, (min, max));
        self
    }

    /// Add protocol step
    pub fn protocol_step(mut self, description: String, duration: f64) -> Self {
        let step = ProtocolStep {
            id: Uuid::new_v4(),
            step_number: self.protocol.steps.len() + 1,
            description,
            duration,
            parameters: HashMap::new(),
            requirements: Vec::new(),
            completed: false,
        };
        self.protocol.steps.push(step);
        self.protocol.duration += duration;
        self
    }

    /// Add requirement
    pub fn requirement(mut self, requirement: String) -> Self {
        self.protocol.requirements.push(requirement);
        self
    }

    /// Add safety measure
    pub fn safety_measure(mut self, measure: String) -> Self {
        self.protocol.safety_measures.push(measure);
        self
    }

    /// Build experiment
    pub fn build(self) -> ValidationExperiment {
        ValidationExperiment::new(self.name, self.experiment_type, self.parameters, self.protocol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_experiment_builder() {
        let experiment = ExperimentBuilder::new("Test Cell Viability".to_string(), ExperimentType::CellViability)
            .parameter("temperature".to_string(), 37.0, "°C".to_string())
            .parameter("ph".to_string(), 7.4, "pH".to_string())
            .parameter_range("concentration".to_string(), 0.1, 10.0)
            .protocol_step("Prepare cell culture".to_string(), 300.0)
            .protocol_step("Apply treatment".to_string(), 1800.0)
            .protocol_step("Measure viability".to_string(), 600.0)
            .requirement("Sterile environment".to_string())
            .safety_measure("Biological containment".to_string())
            .build();

        assert_eq!(experiment.name, "Test Cell Viability");
        assert_eq!(experiment.experiment_type, ExperimentType::CellViability);
        assert_eq!(experiment.protocol.steps.len(), 3);
        assert_eq!(experiment.protocol.duration, 2700.0);
        assert_eq!(experiment.status, ExperimentStatus::Queued);
    }

    #[tokio::test]
    async fn test_experiment_lifecycle() {
        let mut experiment =
            ExperimentBuilder::new("Test Experiment".to_string(), ExperimentType::CellViability).build();

        // Start experiment
        experiment.start();
        assert_eq!(experiment.status, ExperimentStatus::Running);

        // Complete experiment
        let results = ExperimentResults {
            id: Uuid::new_v4(),
            raw_data: vec![1.0, 2.0, 3.0],
            processed_data: HashMap::new(),
            statistics: StatisticalAnalysis {
                sample_size: 3,
                means: HashMap::new(),
                std_devs: HashMap::new(),
                confidence_intervals: HashMap::new(),
                p_values: HashMap::new(),
                effect_sizes: HashMap::new(),
            },
            outcome: ValidationOutcome {
                outcome_type: OutcomeType::Confirmed,
                passed: true,
                score: 0.95,
                message: "Test passed".to_string(),
                evidence: Vec::new(),
            },
            confidence: 0.95,
            error_analysis: ErrorAnalysis {
                systematic_errors: Vec::new(),
                random_errors: Vec::new(),
                error_magnitudes: HashMap::new(),
                error_sources: HashMap::new(),
                mitigation_strategies: Vec::new(),
            },
        };

        experiment.complete(results);
        assert_eq!(experiment.status, ExperimentStatus::Completed);
        assert!(experiment.results.is_some());
        assert!(experiment.end_time.is_some());
        assert!(experiment.duration.is_some());
    }
}
