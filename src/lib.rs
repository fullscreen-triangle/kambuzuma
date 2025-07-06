//! # Kambuzuma: Biological Quantum Computing Architecture
//!
//! A biomimetic metacognitive orchestration system that implements biological quantum
//! computing through specialized neural processing units. The system employs
//! environment-assisted quantum transport (ENAQT) in biological membranes, coordinated
//! by a metacognitive Bayesian network for autonomous computational orchestration.
//!
//! ## Core Components
//!
//! - **Quantum Computing Layer**: Implements quantum tunneling in biological membranes
//! - **Neural Processing Units**: Specialized quantum neurons for different processing tasks
//! - **Metacognitive Orchestration**: Bayesian network for transparent reasoning
//! - **Autonomous Systems**: Self-directed computational ecosystem management
//! - **Biological Validation**: Ensures biological authenticity and constraints
//!
//! ## Architecture
//!
//! The system is organized into eight specialized processing stages:
//! 1. Query Processing
//! 2. Semantic Analysis
//! 3. Domain Knowledge Integration
//! 4. Logical Reasoning
//! 5. Creative Synthesis
//! 6. Evaluation
//! 7. Integration
//! 8. Validation
//!
//! Each stage consists of quantum neurons that implement biological quantum computing
//! through real quantum tunneling effects in phospholipid bilayers, coordinated by
//! molecular Maxwell demons for information processing.

pub mod quantum;
pub mod neural;
pub mod metacognition;
pub mod autonomous;
pub mod biological_validation;
pub mod mathematics;
pub mod interfaces;
pub mod utils;

// Re-export core types for convenience
pub use quantum::membrane::tunneling::{MembraneQuantumTunneling, TunnelingParameters, QuantumState};
pub use quantum::maxwell_demon::molecular_machinery::{MolecularMaxwellDemon, IonType, InformationState};
pub use neural::imhotep_neurons::quantum_neuron::{QuantumNeuron, NeuronSpecialization, NeuronState};

/// Core result type for the Kambuzuma system
pub type Result<T> = std::result::Result<T, KambuzumaError>;

/// Main error type for the Kambuzuma system
#[derive(Debug, thiserror::Error)]
pub enum KambuzumaError {
    #[error("Quantum processing error: {0}")]
    QuantumProcessing(#[from] quantum::membrane::tunneling::TunnelingError),
    
    #[error("Neural processing error: {0}")]
    NeuralProcessing(#[from] neural::imhotep_neurons::quantum_neuron::NeuronError),
    
    #[error("Maxwell demon error: {0}")]
    MaxwellDemon(#[from] quantum::maxwell_demon::molecular_machinery::MaxwellDemonError),
    
    #[error("Biological constraint violation: {0}")]
    BiologicalConstraintViolation(String),
    
    #[error("Metacognitive orchestration error: {0}")]
    MetacognitiveOrchestration(String),
    
    #[error("Autonomous system error: {0}")]
    AutonomousSystem(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("System initialization error: {0}")]
    SystemInitialization(String),
    
    #[error("Performance constraint violation: {0}")]
    PerformanceConstraintViolation(String),
    
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
}

/// System configuration for Kambuzuma
#[derive(Debug, Clone)]
pub struct KambuzumaConfig {
    /// Quantum computing parameters
    pub quantum_config: quantum::QuantumConfig,
    
    /// Neural processing parameters
    pub neural_config: neural::NeuralConfig,
    
    /// Metacognitive orchestration parameters
    pub metacognitive_config: metacognition::MetacognitiveConfig,
    
    /// Autonomous system parameters
    pub autonomous_config: autonomous::AutonomousConfig,
    
    /// Biological validation parameters
    pub biological_config: biological_validation::BiologicalValidationConfig,
    
    /// Performance constraints
    pub performance_config: PerformanceConfig,
}

/// Performance configuration and constraints
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Maximum processing latency (microseconds)
    pub max_processing_latency: u64,
    
    /// Minimum energy efficiency (bits/ATP)
    pub min_energy_efficiency: f64,
    
    /// Minimum quantum coherence maintenance
    pub min_coherence_maintenance: f64,
    
    /// Maximum error rate
    pub max_error_rate: f64,
    
    /// Target throughput (operations/second)
    pub target_throughput: f64,
    
    /// Memory usage limits
    pub memory_limits: MemoryLimits,
}

/// Memory usage limits
#[derive(Debug, Clone)]
pub struct MemoryLimits {
    /// Maximum heap memory (bytes)
    pub max_heap_memory: usize,
    
    /// Maximum quantum state memory (bytes)
    pub max_quantum_state_memory: usize,
    
    /// Maximum neural network memory (bytes)
    pub max_neural_network_memory: usize,
    
    /// Maximum buffer memory (bytes)
    pub max_buffer_memory: usize,
}

impl Default for KambuzumaConfig {
    fn default() -> Self {
        Self {
            quantum_config: quantum::QuantumConfig::default(),
            neural_config: neural::NeuralConfig::default(),
            metacognitive_config: metacognition::MetacognitiveConfig::default(),
            autonomous_config: autonomous::AutonomousConfig::default(),
            biological_config: biological_validation::BiologicalValidationConfig::default(),
            performance_config: PerformanceConfig::default(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_processing_latency: 1000,  // 1 ms
            min_energy_efficiency: 100.0,   // 100 bits/ATP
            min_coherence_maintenance: 0.8, // 80% coherence
            max_error_rate: 0.05,          // 5% error rate
            target_throughput: 10000.0,     // 10k ops/sec
            memory_limits: MemoryLimits::default(),
        }
    }
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_heap_memory: 1024 * 1024 * 1024,        // 1 GB
            max_quantum_state_memory: 256 * 1024 * 1024, // 256 MB
            max_neural_network_memory: 512 * 1024 * 1024, // 512 MB
            max_buffer_memory: 128 * 1024 * 1024,       // 128 MB
        }
    }
}

/// Main Kambuzuma system orchestrator
pub struct KambuzumaSystem {
    /// System configuration
    config: KambuzumaConfig,
    
    /// Quantum processing subsystem
    quantum_subsystem: quantum::QuantumSubsystem,
    
    /// Neural processing subsystem
    neural_subsystem: neural::NeuralSubsystem,
    
    /// Metacognitive orchestration subsystem
    metacognitive_subsystem: metacognition::MetacognitiveSubsystem,
    
    /// Autonomous systems subsystem
    autonomous_subsystem: autonomous::AutonomousSubsystem,
    
    /// Biological validation subsystem
    biological_subsystem: biological_validation::BiologicalValidationSubsystem,
    
    /// System metrics and monitoring
    metrics: utils::monitoring::SystemMetrics,
}

impl KambuzumaSystem {
    /// Create a new Kambuzuma system with default configuration
    pub fn new() -> Result<Self> {
        let config = KambuzumaConfig::default();
        Self::with_config(config)
    }
    
    /// Create a new Kambuzuma system with custom configuration
    pub fn with_config(config: KambuzumaConfig) -> Result<Self> {
        // Initialize subsystems
        let quantum_subsystem = quantum::QuantumSubsystem::new(&config.quantum_config)?;
        let neural_subsystem = neural::NeuralSubsystem::new(&config.neural_config)?;
        let metacognitive_subsystem = metacognition::MetacognitiveSubsystem::new(&config.metacognitive_config)?;
        let autonomous_subsystem = autonomous::AutonomousSubsystem::new(&config.autonomous_config)?;
        let biological_subsystem = biological_validation::BiologicalValidationSubsystem::new(&config.biological_config)?;
        
        // Initialize metrics system
        let metrics = utils::monitoring::SystemMetrics::new()?;
        
        Ok(Self {
            config,
            quantum_subsystem,
            neural_subsystem,
            metacognitive_subsystem,
            autonomous_subsystem,
            biological_subsystem,
            metrics,
        })
    }
    
    /// Start the Kambuzuma system
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting Kambuzuma system...");
        
        // Start subsystems in order
        self.quantum_subsystem.start().await?;
        self.neural_subsystem.start().await?;
        self.metacognitive_subsystem.start().await?;
        self.autonomous_subsystem.start().await?;
        self.biological_subsystem.start().await?;
        
        // Start metrics collection
        self.metrics.start().await?;
        
        tracing::info!("Kambuzuma system started successfully");
        Ok(())
    }
    
    /// Stop the Kambuzuma system
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping Kambuzuma system...");
        
        // Stop subsystems in reverse order
        self.biological_subsystem.stop().await?;
        self.autonomous_subsystem.stop().await?;
        self.metacognitive_subsystem.stop().await?;
        self.neural_subsystem.stop().await?;
        self.quantum_subsystem.stop().await?;
        
        // Stop metrics collection
        self.metrics.stop().await?;
        
        tracing::info!("Kambuzuma system stopped");
        Ok(())
    }
    
    /// Process a computational task through the system
    pub async fn process_task(&self, task: ComputationalTask) -> Result<ComputationalResult> {
        tracing::debug!("Processing computational task: {:?}", task.id);
        
        // Validate biological constraints
        self.biological_subsystem.validate_constraints(&task).await?;
        
        // Route through neural processing stages
        let neural_result = self.neural_subsystem.process_task(&task).await?;
        
        // Apply metacognitive orchestration
        let orchestrated_result = self.metacognitive_subsystem.orchestrate(&neural_result).await?;
        
        // Execute autonomous actions if needed
        let autonomous_result = self.autonomous_subsystem.execute(&orchestrated_result).await?;
        
        // Validate final result
        self.biological_subsystem.validate_result(&autonomous_result).await?;
        
        // Update metrics
        self.metrics.record_task_completion(&task, &autonomous_result).await?;
        
        tracing::debug!("Task processing completed: {:?}", task.id);
        Ok(autonomous_result)
    }
    
    /// Get current system state
    pub async fn get_system_state(&self) -> Result<SystemState> {
        let quantum_state = self.quantum_subsystem.get_state().await?;
        let neural_state = self.neural_subsystem.get_state().await?;
        let metacognitive_state = self.metacognitive_subsystem.get_state().await?;
        let autonomous_state = self.autonomous_subsystem.get_state().await?;
        let biological_state = self.biological_subsystem.get_state().await?;
        let metrics = self.metrics.get_current_metrics().await?;
        
        Ok(SystemState {
            quantum_state,
            neural_state,
            metacognitive_state,
            autonomous_state,
            biological_state,
            metrics,
        })
    }
    
    /// Validate all biological constraints
    pub async fn validate_biological_constraints(&self) -> Result<()> {
        self.quantum_subsystem.validate_biological_constraints().await?;
        self.neural_subsystem.validate_biological_constraints().await?;
        self.biological_subsystem.validate_all_constraints().await?;
        
        Ok(())
    }
    
    /// Get system performance metrics
    pub async fn get_performance_metrics(&self) -> Result<utils::monitoring::PerformanceMetrics> {
        self.metrics.get_performance_metrics().await
    }
}

/// Computational task to be processed by the system
#[derive(Debug, Clone)]
pub struct ComputationalTask {
    /// Task identifier
    pub id: uuid::Uuid,
    
    /// Task description
    pub description: String,
    
    /// Input data
    pub input_data: Vec<u8>,
    
    /// Task type
    pub task_type: TaskType,
    
    /// Priority level
    pub priority: Priority,
    
    /// Deadline for completion
    pub deadline: Option<std::time::Instant>,
    
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Types of computational tasks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskType {
    /// Text processing and analysis
    TextProcessing,
    
    /// Mathematical computation
    MathematicalComputation,
    
    /// Logical reasoning
    LogicalReasoning,
    
    /// Creative synthesis
    CreativeSynthesis,
    
    /// Data analysis
    DataAnalysis,
    
    /// System orchestration
    SystemOrchestration,
    
    /// Validation and verification
    ValidationVerification,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Resource requirements for tasks
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Required processing power (relative scale)
    pub processing_power: f64,
    
    /// Required memory (bytes)
    pub memory: usize,
    
    /// Required energy (in ATP molecules)
    pub energy: f64,
    
    /// Required quantum coherence level
    pub coherence_level: f64,
    
    /// Maximum acceptable latency (microseconds)
    pub max_latency: u64,
}

/// Result of computational processing
#[derive(Debug, Clone)]
pub struct ComputationalResult {
    /// Task identifier
    pub task_id: uuid::Uuid,
    
    /// Result data
    pub result_data: Vec<u8>,
    
    /// Processing success status
    pub success: bool,
    
    /// Confidence in result
    pub confidence: f64,
    
    /// Processing metrics
    pub processing_metrics: ProcessingMetrics,
    
    /// Explanation of processing
    pub explanation: String,
}

/// Metrics from processing a task
#[derive(Debug, Clone)]
pub struct ProcessingMetrics {
    /// Total processing time
    pub processing_time: std::time::Duration,
    
    /// Energy consumed (ATP molecules)
    pub energy_consumed: f64,
    
    /// Quantum coherence maintained
    pub coherence_maintained: f64,
    
    /// Error rate
    pub error_rate: f64,
    
    /// Throughput achieved
    pub throughput: f64,
}

/// Overall system state
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Quantum subsystem state
    pub quantum_state: quantum::QuantumState,
    
    /// Neural subsystem state
    pub neural_state: neural::NeuralState,
    
    /// Metacognitive subsystem state
    pub metacognitive_state: metacognition::MetacognitiveState,
    
    /// Autonomous subsystem state
    pub autonomous_state: autonomous::AutonomousState,
    
    /// Biological subsystem state
    pub biological_state: biological_validation::BiologicalValidationState,
    
    /// System metrics
    pub metrics: utils::monitoring::SystemMetrics,
}

// Default implementations
impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            processing_power: 1.0,
            memory: 1024 * 1024, // 1 MB
            energy: 1000.0,      // 1000 ATP molecules
            coherence_level: 0.8, // 80% coherence
            max_latency: 1000,    // 1 ms
        }
    }
}

impl Default for ProcessingMetrics {
    fn default() -> Self {
        Self {
            processing_time: std::time::Duration::from_millis(1),
            energy_consumed: 0.0,
            coherence_maintained: 1.0,
            error_rate: 0.0,
            throughput: 0.0,
        }
    }
}

/// Initialize tracing for the system
pub fn init_tracing() -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "kambuzuma=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
    #[tokio::test]
    async fn test_system_initialization() {
        let system = KambuzumaSystem::new().unwrap();
        
        // Test that system can be created with default config
        assert!(system.config.quantum_config.is_valid());
        assert!(system.config.neural_config.is_valid());
    }
    
    #[tokio::test]
    async fn test_task_processing() {
        let mut system = KambuzumaSystem::new().unwrap();
        system.start().await.unwrap();
        
        let task = ComputationalTask {
            id: uuid::Uuid::new_v4(),
            description: "Test task".to_string(),
            input_data: vec![1, 2, 3, 4, 5],
            task_type: TaskType::TextProcessing,
            priority: Priority::Medium,
            deadline: None,
            resource_requirements: ResourceRequirements::default(),
        };
        
        let result = system.process_task(task).await.unwrap();
        assert!(result.success);
        assert!(result.confidence > 0.0);
        
        system.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_biological_constraints() {
        let system = KambuzumaSystem::new().unwrap();
        
        // Test that biological constraints are validated
        let result = system.validate_biological_constraints().await;
        assert!(result.is_ok());
    }
} 