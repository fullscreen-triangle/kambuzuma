//! Error types for the Kambuzuma biological quantum computing system

use std::fmt;
use thiserror::Error;

/// Main error type for the Kambuzuma system
#[derive(Error, Debug)]
pub enum KambuzumaError {
    /// Quantum computing errors
    #[error("Quantum computing error: {0}")]
    QuantumComputing(#[from] QuantumError),

    /// Neural processing errors
    #[error("Neural processing error: {0}")]
    NeuralProcessing(#[from] NeuralError),

    /// Metacognitive orchestration errors
    #[error("Metacognitive orchestration error: {0}")]
    MetacognitiveOrchestration(#[from] MetacognitiveError),

    /// Autonomous systems errors
    #[error("Autonomous systems error: {0}")]
    AutonomousSystems(#[from] AutonomousError),

    /// Biological validation errors
    #[error("Biological validation error: {0}")]
    BiologicalValidation(#[from] BiologicalValidationError),

    /// Mathematical frameworks errors
    #[error("Mathematical frameworks error: {0}")]
    MathematicalFrameworks(#[from] MathematicalError),

    /// Interface errors
    #[error("Interface error: {0}")]
    Interface(#[from] InterfaceError),

    /// Utility errors
    #[error("Utility error: {0}")]
    Utility(#[from] UtilityError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(#[from] ConfigurationError),

    /// System initialization errors
    #[error("System initialization error: {0}")]
    SystemInitialization(String),

    /// System shutdown errors
    #[error("System shutdown error: {0}")]
    SystemShutdown(String),

    /// Resource exhaustion errors
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),

    /// Performance constraint violations
    #[error("Performance constraint violation: {0}")]
    PerformanceConstraintViolation(String),

    /// Concurrent access errors
    #[error("Concurrent access error: {0}")]
    ConcurrentAccess(String),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Time-related errors
    #[error("Time error: {0}")]
    Time(#[from] chrono::ParseError),

    /// Generic errors
    #[error("Generic error: {0}")]
    Generic(String),
}

/// Quantum computing subsystem errors
#[derive(Error, Debug)]
pub enum QuantumError {
    /// Membrane tunneling errors
    #[error("Membrane tunneling error: {0}")]
    MembraneTunneling(#[from] TunnelingError),

    /// Oscillation harvesting errors
    #[error("Oscillation harvesting error: {0}")]
    OscillationHarvesting(#[from] OscillationError),

    /// Maxwell demon errors
    #[error("Maxwell demon error: {0}")]
    MaxwellDemon(#[from] MaxwellDemonError),

    /// Quantum gate errors
    #[error("Quantum gate error: {0}")]
    QuantumGate(#[from] QuantumGateError),

    /// Entanglement errors
    #[error("Entanglement error: {0}")]
    Entanglement(#[from] EntanglementError),

    /// Quantum state errors
    #[error("Quantum state error: {0}")]
    QuantumState(#[from] QuantumStateError),

    /// Coherence errors
    #[error("Coherence error: {0}")]
    Coherence(#[from] CoherenceError),

    /// Maxwell demon not found
    #[error("Maxwell demon not found: {0}")]
    MaxwellDemonNotFound(uuid::Uuid),

    /// Invalid oscillation pattern
    #[error("Invalid oscillation pattern: {0}")]
    InvalidOscillationPattern(String),

    /// Invalid state index
    #[error("Invalid state index: {0}")]
    InvalidStateIndex(usize),

    /// Insufficient ATP for operation
    #[error("Insufficient ATP: required {required}, available {available}")]
    InsufficientAtp { required: f64, available: f64 },

    /// Thermodynamic violation
    #[error("Thermodynamic violation: {0}")]
    ThermodynamicViolation(String),

    /// Information processing error
    #[error("Information processing error: {0}")]
    InformationProcessingError(String),

    /// Ion selectivity error
    #[error("Ion selectivity error: {0}")]
    IonSelectivityError(String),

    /// Energy conservation violation
    #[error("Energy conservation violation: {0}")]
    EnergyConservationViolation(String),

    /// Entropy violation
    #[error("Entropy violation: {0}")]
    EntropyViolation(String),

    /// Oscillation endpoint error
    #[error("Oscillation endpoint error: {0}")]
    OscillationEndpointError(String),

    /// Voltage clamp error
    #[error("Voltage clamp error: {0}")]
    VoltageClampError(String),

    /// State collapse error
    #[error("State collapse error: {0}")]
    StateCollapseError(String),

    /// Energy harvesting error
    #[error("Energy harvesting error: {0}")]
    EnergyHarvestingError(String),

    /// Conformational change error
    #[error("Conformational change error: {0}")]
    ConformationalChangeError(String),

    /// Gate control error
    #[error("Gate control error: {0}")]
    GateControlError(String),

    /// Molecular machinery error
    #[error("Molecular machinery error: {0}")]
    MolecularMachineryError(String),

    /// Neuron not found
    #[error("Neuron not found: {0}")]
    NeuronNotFound(uuid::Uuid),
}

/// Tunneling-related errors
#[derive(Error, Debug)]
pub enum TunnelingError {
    /// Invalid barrier parameters
    #[error("Invalid barrier parameters: {0}")]
    InvalidBarrierParameters(String),

    /// Tunneling calculation failure
    #[error("Tunneling calculation failure: {0}")]
    CalculationFailure(String),

    /// Transmission coefficient out of bounds
    #[error("Transmission coefficient out of bounds: {coefficient}")]
    TransmissionCoefficientOutOfBounds { coefficient: f64 },

    /// Energy level invalid
    #[error("Energy level invalid: {energy}")]
    EnergyLevelInvalid { energy: f64 },

    /// Membrane properties invalid
    #[error("Membrane properties invalid: {0}")]
    MembranePropertiesInvalid(String),
}

/// Oscillation harvesting errors
#[derive(Error, Debug)]
pub enum OscillationError {
    /// Endpoint detection failure
    #[error("Endpoint detection failure: {0}")]
    EndpointDetectionFailure(String),

    /// Voltage clamp error
    #[error("Voltage clamp error: {0}")]
    VoltageClampError(String),

    /// Energy harvesting failure
    #[error("Energy harvesting failure: {0}")]
    EnergyHarvestingFailure(String),

    /// Sampling rate invalid
    #[error("Sampling rate invalid: {rate}")]
    SamplingRateInvalid { rate: f64 },
}

/// Maxwell demon errors
#[derive(Error, Debug)]
pub enum MaxwellDemonError {
    /// Information detection failure
    #[error("Information detection failure: {0}")]
    InformationDetectionFailure(String),

    /// Molecular machinery failure
    #[error("Molecular machinery failure: {0}")]
    MolecularMachineryFailure(String),

    /// Energy cost calculation error
    #[error("Energy cost calculation error: {0}")]
    EnergyCostCalculationError(String),

    /// Selectivity failure
    #[error("Selectivity failure: {0}")]
    SelectivityFailure(String),

    /// Conformational change error
    #[error("Conformational change error: {0}")]
    ConformationalChangeError(String),
}

/// Quantum gate errors
#[derive(Error, Debug)]
pub enum QuantumGateError {
    /// Gate operation failure
    #[error("Gate operation failure: {0}")]
    GateOperationFailure(String),

    /// Fidelity too low
    #[error("Fidelity too low: {fidelity}")]
    FidelityTooLow { fidelity: f64 },

    /// Gate timing error
    #[error("Gate timing error: {0}")]
    GateTimingError(String),

    /// Unsupported gate type
    #[error("Unsupported gate type: {gate_type}")]
    UnsupportedGateType { gate_type: String },
}

/// Entanglement errors
#[derive(Error, Debug)]
pub enum EntanglementError {
    /// Entanglement creation failure
    #[error("Entanglement creation failure: {0}")]
    EntanglementCreationFailure(String),

    /// Entanglement verification failure
    #[error("Entanglement verification failure: {0}")]
    EntanglementVerificationFailure(String),

    /// Bell state preparation failure
    #[error("Bell state preparation failure: {0}")]
    BellStatePreparationFailure(String),

    /// Entanglement fidelity too low
    #[error("Entanglement fidelity too low: {fidelity}")]
    EntanglementFidelityTooLow { fidelity: f64 },
}

/// Quantum state errors
#[derive(Error, Debug)]
pub enum QuantumStateError {
    /// State normalization failure
    #[error("State normalization failure: {0}")]
    StateNormalizationFailure(String),

    /// State measurement failure
    #[error("State measurement failure: {0}")]
    StateMeasurementFailure(String),

    /// State evolution failure
    #[error("State evolution failure: {0}")]
    StateEvolutionFailure(String),

    /// Invalid state vector
    #[error("Invalid state vector: {0}")]
    InvalidStateVector(String),
}

/// Coherence errors
#[derive(Error, Debug)]
pub enum CoherenceError {
    /// Coherence loss
    #[error("Coherence loss: {0}")]
    CoherenceLoss(String),

    /// Decoherence rate too high
    #[error("Decoherence rate too high: {rate}")]
    DecoherenceRateTooHigh { rate: f64 },

    /// Coherence time too short
    #[error("Coherence time too short: {time}")]
    CoherenceTimeTooShort { time: f64 },

    /// Coherence preservation failure
    #[error("Coherence preservation failure: {0}")]
    CoherencePreservationFailure(String),
}

/// Neural processing errors
#[derive(Error, Debug)]
pub enum NeuralError {
    /// Neuron operation errors
    #[error("Neuron operation error: {0}")]
    NeuronOperation(#[from] NeuronError),

    /// Thought current errors
    #[error("Thought current error: {0}")]
    ThoughtCurrent(#[from] ThoughtCurrentError),

    /// Network topology errors
    #[error("Network topology error: {0}")]
    NetworkTopology(#[from] NetworkTopologyError),

    /// Processing stage errors
    #[error("Processing stage error: {0}")]
    ProcessingStage(#[from] ProcessingStageError),
}

/// Neuron-related errors
#[derive(Error, Debug)]
pub enum NeuronError {
    /// Neuron firing failure
    #[error("Neuron firing failure: {0}")]
    NeuronFiringFailure(String),

    /// Membrane potential invalid
    #[error("Membrane potential invalid: {potential}")]
    MembranePotentialInvalid { potential: f64 },

    /// Synaptic transmission failure
    #[error("Synaptic transmission failure: {0}")]
    SynapticTransmissionFailure(String),

    /// Neuron configuration invalid
    #[error("Neuron configuration invalid: {0}")]
    NeuronConfigurationInvalid(String),
}

/// Thought current errors
#[derive(Error, Debug)]
pub enum ThoughtCurrentError {
    /// Current flow calculation failure
    #[error("Current flow calculation failure: {0}")]
    CurrentFlowCalculationFailure(String),

    /// Current magnitude invalid
    #[error("Current magnitude invalid: {magnitude}")]
    CurrentMagnitudeInvalid { magnitude: f64 },

    /// Information encoding failure
    #[error("Information encoding failure: {0}")]
    InformationEncodingFailure(String),

    /// Current routing failure
    #[error("Current routing failure: {0}")]
    CurrentRoutingFailure(String),
}

/// Network topology errors
#[derive(Error, Debug)]
pub enum NetworkTopologyError {
    /// Connectivity matrix invalid
    #[error("Connectivity matrix invalid: {0}")]
    ConnectivityMatrixInvalid(String),

    /// Synaptic weight invalid
    #[error("Synaptic weight invalid: {weight}")]
    SynapticWeightInvalid { weight: f64 },

    /// Network structure invalid
    #[error("Network structure invalid: {0}")]
    NetworkStructureInvalid(String),

    /// Plasticity update failure
    #[error("Plasticity update failure: {0}")]
    PlasticityUpdateFailure(String),
}

/// Processing stage errors
#[derive(Error, Debug)]
pub enum ProcessingStageError {
    /// Stage execution failure
    #[error("Stage execution failure: {0}")]
    StageExecutionFailure(String),

    /// Stage synchronization failure
    #[error("Stage synchronization failure: {0}")]
    StageSynchronizationFailure(String),

    /// Stage configuration invalid
    #[error("Stage configuration invalid: {0}")]
    StageConfigurationInvalid(String),

    /// Stage timeout
    #[error("Stage timeout: {stage_id}")]
    StageTimeout { stage_id: u8 },
}

/// Metacognitive orchestration errors
#[derive(Error, Debug)]
pub enum MetacognitiveError {
    /// Bayesian network errors
    #[error("Bayesian network error: {0}")]
    BayesianNetwork(#[from] BayesianNetworkError),

    /// Awareness monitoring errors
    #[error("Awareness monitoring error: {0}")]
    AwarenessMonitoring(#[from] AwarenessMonitoringError),

    /// Decision making errors
    #[error("Decision making error: {0}")]
    DecisionMaking(#[from] DecisionMakingError),

    /// Transparency errors
    #[error("Transparency error: {0}")]
    Transparency(#[from] TransparencyError),
}

/// Bayesian network errors
#[derive(Error, Debug)]
pub enum BayesianNetworkError {
    /// Network structure invalid
    #[error("Network structure invalid: {0}")]
    NetworkStructureInvalid(String),

    /// Inference failure
    #[error("Inference failure: {0}")]
    InferenceFailure(String),

    /// Probability calculation error
    #[error("Probability calculation error: {0}")]
    ProbabilityCalculationError(String),

    /// Convergence failure
    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),
}

/// Awareness monitoring errors
#[derive(Error, Debug)]
pub enum AwarenessMonitoringError {
    /// Monitoring failure
    #[error("Monitoring failure: {0}")]
    MonitoringFailure(String),

    /// Confidence calculation error
    #[error("Confidence calculation error: {0}")]
    ConfidenceCalculationError(String),

    /// Uncertainty estimation error
    #[error("Uncertainty estimation error: {0}")]
    UncertaintyEstimationError(String),

    /// Trigger activation failure
    #[error("Trigger activation failure: {0}")]
    TriggerActivationFailure(String),
}

/// Decision making errors
#[derive(Error, Debug)]
pub enum DecisionMakingError {
    /// Decision calculation failure
    #[error("Decision calculation failure: {0}")]
    DecisionCalculationFailure(String),

    /// Multi-objective optimization failure
    #[error("Multi-objective optimization failure: {0}")]
    MultiObjectiveOptimizationFailure(String),

    /// Risk assessment failure
    #[error("Risk assessment failure: {0}")]
    RiskAssessmentFailure(String),

    /// Decision threshold not met
    #[error("Decision threshold not met: {threshold}")]
    DecisionThresholdNotMet { threshold: f64 },
}

/// Transparency errors
#[derive(Error, Debug)]
pub enum TransparencyError {
    /// Explanation generation failure
    #[error("Explanation generation failure: {0}")]
    ExplanationGenerationFailure(String),

    /// Trace generation failure
    #[error("Trace generation failure: {0}")]
    TraceGenerationFailure(String),

    /// Justification failure
    #[error("Justification failure: {0}")]
    JustificationFailure(String),

    /// Confidence reporting failure
    #[error("Confidence reporting failure: {0}")]
    ConfidenceReportingFailure(String),
}

/// Autonomous systems errors
#[derive(Error, Debug)]
pub enum AutonomousError {
    /// Language selection errors
    #[error("Language selection error: {0}")]
    LanguageSelection(#[from] LanguageSelectionError),

    /// Tool orchestration errors
    #[error("Tool orchestration error: {0}")]
    ToolOrchestration(#[from] ToolOrchestrationError),

    /// Package management errors
    #[error("Package management error: {0}")]
    PackageManagement(#[from] PackageManagementError),

    /// Execution engine errors
    #[error("Execution engine error: {0}")]
    ExecutionEngine(#[from] ExecutionEngineError),
}

/// Language selection errors
#[derive(Error, Debug)]
pub enum LanguageSelectionError {
    /// Language not supported
    #[error("Language not supported: {language}")]
    LanguageNotSupported { language: String },

    /// Capability assessment failure
    #[error("Capability assessment failure: {0}")]
    CapabilityAssessmentFailure(String),

    /// Selection criteria invalid
    #[error("Selection criteria invalid: {0}")]
    SelectionCriteriaInvalid(String),

    /// Ecosystem evaluation failure
    #[error("Ecosystem evaluation failure: {0}")]
    EcosystemEvaluationFailure(String),
}

/// Tool orchestration errors
#[derive(Error, Debug)]
pub enum ToolOrchestrationError {
    /// Tool not found
    #[error("Tool not found: {tool}")]
    ToolNotFound { tool: String },

    /// Installation failure
    #[error("Installation failure: {0}")]
    InstallationFailure(String),

    /// Configuration failure
    #[error("Configuration failure: {0}")]
    ConfigurationFailure(String),

    /// Version conflict
    #[error("Version conflict: {0}")]
    VersionConflict(String),
}

/// Package management errors
#[derive(Error, Debug)]
pub enum PackageManagementError {
    /// Package not found
    #[error("Package not found: {package}")]
    PackageNotFound { package: String },

    /// Dependency resolution failure
    #[error("Dependency resolution failure: {0}")]
    DependencyResolutionFailure(String),

    /// Environment isolation failure
    #[error("Environment isolation failure: {0}")]
    EnvironmentIsolationFailure(String),

    /// Package manager error
    #[error("Package manager error: {0}")]
    PackageManagerError(String),
}

/// Execution engine errors
#[derive(Error, Debug)]
pub enum ExecutionEngineError {
    /// Execution failure
    #[error("Execution failure: {0}")]
    ExecutionFailure(String),

    /// Timeout error
    #[error("Timeout error: {0}")]
    TimeoutError(String),

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),

    /// Recovery failure
    #[error("Recovery failure: {0}")]
    RecoveryFailure(String),
}

/// Biological validation errors
#[derive(Error, Debug)]
pub enum BiologicalValidationError {
    /// Constraint violation
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    /// Validation protocol failure
    #[error("Validation protocol failure: {0}")]
    ValidationProtocolFailure(String),

    /// Monitoring failure
    #[error("Monitoring failure: {0}")]
    MonitoringFailure(String),

    /// Alert threshold exceeded
    #[error("Alert threshold exceeded: {0}")]
    AlertThresholdExceeded(String),

    /// Energy balance violation
    #[error("Energy balance violation: {0}")]
    EnergyBalanceViolation(String),

    /// Thermodynamic constraint violation
    #[error("Thermodynamic constraint violation: {0}")]
    ThermodynamicConstraintViolation(String),
}

/// Mathematical frameworks errors
#[derive(Error, Debug)]
pub enum MathematicalError {
    /// Numerical computation error
    #[error("Numerical computation error: {0}")]
    NumericalComputation(String),

    /// Solver failure
    #[error("Solver failure: {0}")]
    SolverFailure(String),

    /// Optimization failure
    #[error("Optimization failure: {0}")]
    OptimizationFailure(String),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinearAlgebra(String),

    /// Convergence failure
    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),

    /// Numerical precision error
    #[error("Numerical precision error: {0}")]
    NumericalPrecision(String),
}

/// Interface errors
#[derive(Error, Debug)]
pub enum InterfaceError {
    /// REST API errors
    #[error("REST API error: {0}")]
    RestApi(#[from] RestApiError),

    /// WebSocket errors
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] WebSocketError),

    /// CLI errors
    #[error("CLI error: {0}")]
    Cli(#[from] CliError),
}

/// REST API errors
#[derive(Error, Debug)]
pub enum RestApiError {
    /// Request handling failure
    #[error("Request handling failure: {0}")]
    RequestHandlingFailure(String),

    /// Response generation failure
    #[error("Response generation failure: {0}")]
    ResponseGenerationFailure(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Authentication failure
    #[error("Authentication failure: {0}")]
    AuthenticationFailure(String),
}

/// WebSocket errors
#[derive(Error, Debug)]
pub enum WebSocketError {
    /// Connection failure
    #[error("Connection failure: {0}")]
    ConnectionFailure(String),

    /// Message sending failure
    #[error("Message sending failure: {0}")]
    MessageSendingFailure(String),

    /// Message receiving failure
    #[error("Message receiving failure: {0}")]
    MessageReceivingFailure(String),

    /// Connection timeout
    #[error("Connection timeout: {0}")]
    ConnectionTimeout(String),
}

/// CLI errors
#[derive(Error, Debug)]
pub enum CliError {
    /// Command parsing failure
    #[error("Command parsing failure: {0}")]
    CommandParsingFailure(String),

    /// Command execution failure
    #[error("Command execution failure: {0}")]
    CommandExecutionFailure(String),

    /// Output formatting failure
    #[error("Output formatting failure: {0}")]
    OutputFormattingFailure(String),

    /// Input validation failure
    #[error("Input validation failure: {0}")]
    InputValidationFailure(String),
}

/// Utility errors
#[derive(Error, Debug)]
pub enum UtilityError {
    /// Logging errors
    #[error("Logging error: {0}")]
    Logging(#[from] LoggingError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(#[from] ConfigurationError),

    /// Monitoring errors
    #[error("Monitoring error: {0}")]
    Monitoring(#[from] MonitoringError),
}

/// Logging errors
#[derive(Error, Debug)]
pub enum LoggingError {
    /// Log file creation failure
    #[error("Log file creation failure: {0}")]
    LogFileCreationFailure(String),

    /// Log writing failure
    #[error("Log writing failure: {0}")]
    LogWritingFailure(String),

    /// Log rotation failure
    #[error("Log rotation failure: {0}")]
    LogRotationFailure(String),

    /// Log level invalid
    #[error("Log level invalid: {level}")]
    LogLevelInvalid { level: String },
}

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigurationError {
    /// Configuration file not found
    #[error("Configuration file not found: {path}")]
    ConfigurationFileNotFound { path: String },

    /// Configuration parsing failure
    #[error("Configuration parsing failure: {0}")]
    ConfigurationParsingFailure(String),

    /// Configuration validation failure
    #[error("Configuration validation failure: {0}")]
    ConfigurationValidationFailure(String),

    /// Environment variable invalid
    #[error("Environment variable invalid: {variable}")]
    EnvironmentVariableInvalid { variable: String },
}

/// Monitoring errors
#[derive(Error, Debug)]
pub enum MonitoringError {
    /// Metrics collection failure
    #[error("Metrics collection failure: {0}")]
    MetricsCollectionFailure(String),

    /// Health check failure
    #[error("Health check failure: {0}")]
    HealthCheckFailure(String),

    /// Alert generation failure
    #[error("Alert generation failure: {0}")]
    AlertGenerationFailure(String),

    /// Notification sending failure
    #[error("Notification sending failure: {0}")]
    NotificationSendingFailure(String),
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, KambuzumaError>;

/// Trait for error context
pub trait ErrorContext {
    fn with_context(self, context: &str) -> KambuzumaError;
}

impl ErrorContext for KambuzumaError {
    fn with_context(self, context: &str) -> KambuzumaError {
        KambuzumaError::Generic(format!("{}: {}", context, self))
    }
}

/// Trait for converting errors to KambuzumaError
pub trait IntoKambuzumaError {
    fn into_kambuzuma_error(self) -> KambuzumaError;
}

impl IntoKambuzumaError for Box<dyn std::error::Error> {
    fn into_kambuzuma_error(self) -> KambuzumaError {
        KambuzumaError::Generic(self.to_string())
    }
}

impl IntoKambuzumaError for String {
    fn into_kambuzuma_error(self) -> KambuzumaError {
        KambuzumaError::Generic(self)
    }
}

impl IntoKambuzumaError for &str {
    fn into_kambuzuma_error(self) -> KambuzumaError {
        KambuzumaError::Generic(self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = KambuzumaError::Generic("Test error".to_string());
        assert_eq!(error.to_string(), "Generic error: Test error");
    }

    #[test]
    fn test_error_context() {
        let error = KambuzumaError::Generic("Original error".to_string());
        let contextual_error = error.with_context("Additional context");
        assert_eq!(
            contextual_error.to_string(),
            "Generic error: Additional context: Generic error: Original error"
        );
    }

    #[test]
    fn test_error_conversion() {
        let string_error = "String error".to_string();
        let kambuzuma_error = string_error.into_kambuzuma_error();
        assert!(matches!(kambuzuma_error, KambuzumaError::Generic(_)));
    }
}
