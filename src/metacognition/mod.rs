//! # Metacognitive Orchestration System
//!
//! This module implements the metacognitive orchestration system that coordinates
//! neural processing through a Bayesian network. The system provides complete
//! transparency of reasoning processes and maintains four categories of
//! metacognitive awareness.
//!
//! ## Core Components
//!
//! - **Bayesian Network**: Probabilistic graphical model for stage coordination
//! - **Awareness Monitoring**: Four types of metacognitive awareness
//! - **Orchestration Control**: Stage coordination and resource allocation
//! - **Decision Making**: Multi-objective decision frameworks
//! - **Transparency**: Complete reasoning trace generation

use crate::config::NeuralConfig;
use crate::errors::KambuzumaError;
use crate::types::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub mod awareness_monitoring;
pub mod bayesian_network;
pub mod decision_making;
pub mod orchestration_control;
pub mod transparency;

// Re-export important types
pub use awareness_monitoring::*;
pub use bayesian_network::*;
pub use decision_making::*;
pub use orchestration_control::*;
pub use transparency::*;

/// Metacognitive Subsystem
/// Main orchestration system for neural processing coordination
#[derive(Debug)]
pub struct MetacognitiveSubsystem {
    /// Subsystem identifier
    pub id: Uuid,
    /// Configuration
    pub config: Arc<RwLock<crate::config::KambuzumaConfig>>,
    /// Bayesian network orchestrator
    pub bayesian_network: Arc<RwLock<BayesianNetworkOrchestrator>>,
    /// Awareness monitoring system
    pub awareness_monitor: Arc<RwLock<AwarenessMonitor>>,
    /// Orchestration controller
    pub orchestration_controller: Arc<RwLock<OrchestrationController>>,
    /// Decision making framework
    pub decision_maker: Arc<RwLock<DecisionMaker>>,
    /// Transparency system
    pub transparency_system: Arc<RwLock<TransparencySystem>>,
    /// Current metacognitive state
    pub metacognitive_state: Arc<RwLock<MetacognitiveState>>,
    /// Performance metrics
    pub metrics: Arc<RwLock<MetacognitiveMetrics>>,
}

/// Metacognitive State
/// Current state of the metacognitive system
#[derive(Debug, Clone)]
pub struct MetacognitiveState {
    /// State identifier
    pub id: Uuid,
    /// System status
    pub status: MetacognitiveStatus,
    /// Process awareness level
    pub process_awareness: f64,
    /// Knowledge awareness level
    pub knowledge_awareness: f64,
    /// Gap awareness level
    pub gap_awareness: f64,
    /// Decision awareness level
    pub decision_awareness: f64,
    /// Overall metacognitive coherence
    pub metacognitive_coherence: f64,
    /// Active orchestration strategies
    pub active_strategies: Vec<OrchestrationStrategy>,
    /// Current decision context
    pub decision_context: Option<DecisionContext>,
    /// Transparency level
    pub transparency_level: f64,
    /// System confidence
    pub system_confidence: f64,
}

/// Metacognitive Status
/// Status of the metacognitive system
#[derive(Debug, Clone, PartialEq)]
pub enum MetacognitiveStatus {
    /// System is offline
    Offline,
    /// System is initializing
    Initializing,
    /// System is monitoring
    Monitoring,
    /// System is orchestrating
    Orchestrating,
    /// System is decision making
    DecisionMaking,
    /// System is analyzing
    Analyzing,
    /// System is optimizing
    Optimizing,
    /// System has error
    Error,
    /// System is shutting down
    Shutdown,
}

/// Orchestration Strategy
/// Strategy for orchestrating neural processing
#[derive(Debug, Clone)]
pub struct OrchestrationStrategy {
    /// Strategy identifier
    pub id: Uuid,
    /// Strategy name
    pub name: String,
    /// Strategy type
    pub strategy_type: OrchestrationStrategyType,
    /// Strategy parameters
    pub parameters: HashMap<String, f64>,
    /// Strategy priority
    pub priority: f64,
    /// Strategy effectiveness
    pub effectiveness: f64,
    /// Is strategy active
    pub is_active: bool,
    /// Strategy timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Orchestration Strategy Type
/// Types of orchestration strategies
#[derive(Debug, Clone, PartialEq)]
pub enum OrchestrationStrategyType {
    /// Sequential processing
    Sequential,
    /// Parallel processing
    Parallel,
    /// Adaptive routing
    Adaptive,
    /// Priority-based
    Priority,
    /// Load balancing
    LoadBalancing,
    /// Error recovery
    ErrorRecovery,
    /// Resource optimization
    ResourceOptimization,
    /// Performance tuning
    PerformanceTuning,
}

/// Decision Context
/// Context for decision making
#[derive(Debug, Clone)]
pub struct DecisionContext {
    /// Context identifier
    pub id: Uuid,
    /// Decision problem
    pub problem: DecisionProblem,
    /// Available alternatives
    pub alternatives: Vec<DecisionAlternative>,
    /// Decision criteria
    pub criteria: Vec<DecisionCriterion>,
    /// Context constraints
    pub constraints: Vec<DecisionConstraint>,
    /// Decision urgency
    pub urgency: f64,
    /// Context timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Decision Problem
/// Problem requiring a decision
#[derive(Debug, Clone)]
pub struct DecisionProblem {
    /// Problem identifier
    pub id: Uuid,
    /// Problem description
    pub description: String,
    /// Problem type
    pub problem_type: DecisionProblemType,
    /// Problem complexity
    pub complexity: f64,
    /// Problem importance
    pub importance: f64,
    /// Problem urgency
    pub urgency: f64,
    /// Problem context
    pub context: HashMap<String, String>,
}

/// Decision Problem Type
/// Types of decision problems
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionProblemType {
    /// Resource allocation
    ResourceAllocation,
    /// Stage coordination
    StageCoordination,
    /// Priority assignment
    PriorityAssignment,
    /// Error handling
    ErrorHandling,
    /// Performance optimization
    PerformanceOptimization,
    /// Conflict resolution
    ConflictResolution,
    /// Strategy selection
    StrategySelection,
    /// Parameter tuning
    ParameterTuning,
}

/// Decision Alternative
/// Alternative solution to a decision problem
#[derive(Debug, Clone)]
pub struct DecisionAlternative {
    /// Alternative identifier
    pub id: Uuid,
    /// Alternative name
    pub name: String,
    /// Alternative description
    pub description: String,
    /// Alternative parameters
    pub parameters: HashMap<String, f64>,
    /// Expected utility
    pub expected_utility: f64,
    /// Alternative cost
    pub cost: f64,
    /// Alternative risk
    pub risk: f64,
    /// Alternative feasibility
    pub feasibility: f64,
}

/// Decision Criterion
/// Criterion for evaluating alternatives
#[derive(Debug, Clone)]
pub struct DecisionCriterion {
    /// Criterion identifier
    pub id: Uuid,
    /// Criterion name
    pub name: String,
    /// Criterion weight
    pub weight: f64,
    /// Criterion type
    pub criterion_type: DecisionCriterionType,
    /// Optimization direction
    pub optimization_direction: OptimizationDirection,
}

/// Decision Criterion Type
/// Types of decision criteria
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionCriterionType {
    /// Performance criterion
    Performance,
    /// Cost criterion
    Cost,
    /// Risk criterion
    Risk,
    /// Time criterion
    Time,
    /// Quality criterion
    Quality,
    /// Reliability criterion
    Reliability,
    /// Efficiency criterion
    Efficiency,
    /// Flexibility criterion
    Flexibility,
}

/// Optimization Direction
/// Direction for criterion optimization
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationDirection {
    /// Maximize the criterion
    Maximize,
    /// Minimize the criterion
    Minimize,
}

/// Decision Constraint
/// Constraint on decision making
#[derive(Debug, Clone)]
pub struct DecisionConstraint {
    /// Constraint identifier
    pub id: Uuid,
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: DecisionConstraintType,
    /// Constraint value
    pub value: f64,
    /// Constraint operator
    pub operator: ConstraintOperator,
    /// Is constraint hard
    pub is_hard: bool,
}

/// Decision Constraint Type
/// Types of decision constraints
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionConstraintType {
    /// Resource constraint
    Resource,
    /// Time constraint
    Time,
    /// Quality constraint
    Quality,
    /// Energy constraint
    Energy,
    /// Capacity constraint
    Capacity,
    /// Compatibility constraint
    Compatibility,
    /// Safety constraint
    Safety,
    /// Regulatory constraint
    Regulatory,
}

/// Constraint Operator
/// Operator for constraint evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintOperator {
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
    /// Less than
    LessThan,
    /// Less than or equal to
    LessThanOrEqual,
    /// Greater than
    GreaterThan,
    /// Greater than or equal to
    GreaterThanOrEqual,
}

/// Metacognitive Metrics
/// Performance metrics for the metacognitive system
#[derive(Debug, Clone)]
pub struct MetacognitiveMetrics {
    /// Total orchestration decisions
    pub total_decisions: u64,
    /// Successful orchestration decisions
    pub successful_decisions: u64,
    /// Average decision time
    pub average_decision_time: f64,
    /// Average decision confidence
    pub average_decision_confidence: f64,
    /// Orchestration efficiency
    pub orchestration_efficiency: f64,
    /// Awareness accuracy
    pub awareness_accuracy: f64,
    /// Transparency score
    pub transparency_score: f64,
    /// System adaptability
    pub system_adaptability: f64,
    /// Error recovery rate
    pub error_recovery_rate: f64,
    /// Resource utilization
    pub resource_utilization: f64,
}

impl MetacognitiveSubsystem {
    /// Create new metacognitive subsystem
    pub async fn new(config: Arc<RwLock<crate::config::KambuzumaConfig>>) -> Result<Self, KambuzumaError> {
        let id = Uuid::new_v4();

        // Initialize components
        let bayesian_network = Arc::new(RwLock::new(BayesianNetworkOrchestrator::new(config.clone()).await?));
        let awareness_monitor = Arc::new(RwLock::new(AwarenessMonitor::new(config.clone()).await?));
        let orchestration_controller = Arc::new(RwLock::new(OrchestrationController::new(config.clone()).await?));
        let decision_maker = Arc::new(RwLock::new(DecisionMaker::new(config.clone()).await?));
        let transparency_system = Arc::new(RwLock::new(TransparencySystem::new(config.clone()).await?));

        // Initialize metacognitive state
        let metacognitive_state = Arc::new(RwLock::new(MetacognitiveState {
            id,
            status: MetacognitiveStatus::Offline,
            process_awareness: 0.0,
            knowledge_awareness: 0.0,
            gap_awareness: 0.0,
            decision_awareness: 0.0,
            metacognitive_coherence: 1.0,
            active_strategies: Vec::new(),
            decision_context: None,
            transparency_level: 1.0,
            system_confidence: 0.0,
        }));

        // Initialize metrics
        let metrics = Arc::new(RwLock::new(MetacognitiveMetrics {
            total_decisions: 0,
            successful_decisions: 0,
            average_decision_time: 0.0,
            average_decision_confidence: 0.0,
            orchestration_efficiency: 0.0,
            awareness_accuracy: 0.0,
            transparency_score: 0.0,
            system_adaptability: 0.0,
            error_recovery_rate: 0.0,
            resource_utilization: 0.0,
        }));

        Ok(Self {
            id,
            config,
            bayesian_network,
            awareness_monitor,
            orchestration_controller,
            decision_maker,
            transparency_system,
            metacognitive_state,
            metrics,
        })
    }

    /// Start the metacognitive subsystem
    pub async fn start(&mut self) -> Result<(), KambuzumaError> {
        log::info!("Starting metacognitive subsystem");

        // Update state
        {
            let mut state = self.metacognitive_state.write().await;
            state.status = MetacognitiveStatus::Initializing;
        }

        // Initialize components
        self.initialize_components().await?;

        // Start monitoring
        self.start_monitoring().await?;

        // Update state
        {
            let mut state = self.metacognitive_state.write().await;
            state.status = MetacognitiveStatus::Monitoring;
        }

        log::info!("Metacognitive subsystem started successfully");
        Ok(())
    }

    /// Stop the metacognitive subsystem
    pub async fn stop(&mut self) -> Result<(), KambuzumaError> {
        log::info!("Stopping metacognitive subsystem");

        // Update state
        {
            let mut state = self.metacognitive_state.write().await;
            state.status = MetacognitiveStatus::Shutdown;
        }

        // Stop monitoring
        self.stop_monitoring().await?;

        // Stop components
        self.stop_components().await?;

        // Update final state
        {
            let mut state = self.metacognitive_state.write().await;
            state.status = MetacognitiveStatus::Offline;
        }

        log::info!("Metacognitive subsystem stopped successfully");
        Ok(())
    }

    /// Orchestrate neural processing
    pub async fn orchestrate_processing(
        &self,
        orchestration_request: OrchestrationRequest,
    ) -> Result<OrchestrationResponse, KambuzumaError> {
        log::debug!("Orchestrating neural processing");

        // Update state
        {
            let mut state = self.metacognitive_state.write().await;
            state.status = MetacognitiveStatus::Orchestrating;
        }

        // Analyze request
        let analysis = self.analyze_orchestration_request(&orchestration_request).await?;

        // Make orchestration decision
        let decision = self.make_orchestration_decision(&analysis).await?;

        // Execute orchestration
        let execution_result = self.execute_orchestration(&decision).await?;

        // Monitor execution
        self.monitor_orchestration_execution(&execution_result).await?;

        // Generate response
        let response = self
            .generate_orchestration_response(&orchestration_request, &execution_result)
            .await?;

        // Update metrics
        self.update_orchestration_metrics(&response).await?;

        // Update state
        {
            let mut state = self.metacognitive_state.write().await;
            state.status = MetacognitiveStatus::Monitoring;
        }

        Ok(response)
    }

    /// Get metacognitive state
    pub async fn get_metacognitive_state(&self) -> MetacognitiveState {
        self.metacognitive_state.read().await.clone()
    }

    /// Get awareness levels
    pub async fn get_awareness_levels(&self) -> AwarenessLevels {
        let monitor = self.awareness_monitor.read().await;
        monitor.get_current_awareness_levels().await
    }

    /// Get decision transparency
    pub async fn get_decision_transparency(&self, decision_id: Uuid) -> Result<DecisionTransparency, KambuzumaError> {
        let transparency = self.transparency_system.read().await;
        transparency.get_decision_transparency(decision_id).await
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> MetacognitiveMetrics {
        self.metrics.read().await.clone()
    }

    /// Initialize components
    async fn initialize_components(&self) -> Result<(), KambuzumaError> {
        log::debug!("Initializing metacognitive components");

        // Initialize Bayesian network
        {
            let mut network = self.bayesian_network.write().await;
            network.initialize().await?;
        }

        // Initialize awareness monitor
        {
            let mut monitor = self.awareness_monitor.write().await;
            monitor.initialize().await?;
        }

        // Initialize orchestration controller
        {
            let mut controller = self.orchestration_controller.write().await;
            controller.initialize().await?;
        }

        // Initialize decision maker
        {
            let mut decision_maker = self.decision_maker.write().await;
            decision_maker.initialize().await?;
        }

        // Initialize transparency system
        {
            let mut transparency = self.transparency_system.write().await;
            transparency.initialize().await?;
        }

        Ok(())
    }

    /// Start monitoring
    async fn start_monitoring(&self) -> Result<(), KambuzumaError> {
        log::debug!("Starting metacognitive monitoring");

        // Start awareness monitoring
        {
            let mut monitor = self.awareness_monitor.write().await;
            monitor.start_monitoring().await?;
        }

        // Start orchestration monitoring
        {
            let mut controller = self.orchestration_controller.write().await;
            controller.start_monitoring().await?;
        }

        Ok(())
    }

    /// Stop monitoring
    async fn stop_monitoring(&self) -> Result<(), KambuzumaError> {
        log::debug!("Stopping metacognitive monitoring");

        // Stop awareness monitoring
        {
            let mut monitor = self.awareness_monitor.write().await;
            monitor.stop_monitoring().await?;
        }

        // Stop orchestration monitoring
        {
            let mut controller = self.orchestration_controller.write().await;
            controller.stop_monitoring().await?;
        }

        Ok(())
    }

    /// Stop components
    async fn stop_components(&self) -> Result<(), KambuzumaError> {
        log::debug!("Stopping metacognitive components");

        // Stop all components
        {
            let mut transparency = self.transparency_system.write().await;
            transparency.stop().await?;
        }

        {
            let mut decision_maker = self.decision_maker.write().await;
            decision_maker.stop().await?;
        }

        {
            let mut controller = self.orchestration_controller.write().await;
            controller.stop().await?;
        }

        {
            let mut monitor = self.awareness_monitor.write().await;
            monitor.stop().await?;
        }

        {
            let mut network = self.bayesian_network.write().await;
            network.stop().await?;
        }

        Ok(())
    }

    /// Analyze orchestration request
    async fn analyze_orchestration_request(
        &self,
        request: &OrchestrationRequest,
    ) -> Result<OrchestrationAnalysis, KambuzumaError> {
        // Analyze request complexity
        let complexity = self.analyze_request_complexity(request).await?;

        // Analyze resource requirements
        let resource_requirements = self.analyze_resource_requirements(request).await?;

        // Analyze constraints
        let constraints = self.analyze_constraints(request).await?;

        Ok(OrchestrationAnalysis {
            complexity,
            resource_requirements,
            constraints,
            recommended_strategy: OrchestrationStrategyType::Adaptive,
        })
    }

    /// Make orchestration decision
    async fn make_orchestration_decision(
        &self,
        analysis: &OrchestrationAnalysis,
    ) -> Result<OrchestrationDecision, KambuzumaError> {
        let decision_maker = self.decision_maker.read().await;
        decision_maker.make_orchestration_decision(analysis).await
    }

    /// Execute orchestration
    async fn execute_orchestration(
        &self,
        decision: &OrchestrationDecision,
    ) -> Result<OrchestrationExecutionResult, KambuzumaError> {
        let controller = self.orchestration_controller.read().await;
        controller.execute_orchestration(decision).await
    }

    /// Monitor orchestration execution
    async fn monitor_orchestration_execution(
        &self,
        execution_result: &OrchestrationExecutionResult,
    ) -> Result<(), KambuzumaError> {
        let monitor = self.awareness_monitor.read().await;
        monitor.monitor_execution(execution_result).await
    }

    /// Generate orchestration response
    async fn generate_orchestration_response(
        &self,
        request: &OrchestrationRequest,
        execution_result: &OrchestrationExecutionResult,
    ) -> Result<OrchestrationResponse, KambuzumaError> {
        Ok(OrchestrationResponse {
            request_id: request.id,
            response_id: Uuid::new_v4(),
            success: execution_result.success,
            execution_time: execution_result.execution_time,
            results: execution_result.results.clone(),
            confidence: execution_result.confidence,
            transparency: execution_result.transparency.clone(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Update orchestration metrics
    async fn update_orchestration_metrics(&self, response: &OrchestrationResponse) -> Result<(), KambuzumaError> {
        let mut metrics = self.metrics.write().await;

        metrics.total_decisions += 1;
        if response.success {
            metrics.successful_decisions += 1;
        }

        // Update averages
        let total = metrics.total_decisions as f64;
        metrics.average_decision_time =
            ((metrics.average_decision_time * (total - 1.0)) + response.execution_time) / total;
        metrics.average_decision_confidence =
            ((metrics.average_decision_confidence * (total - 1.0)) + response.confidence) / total;

        // Update efficiency
        metrics.orchestration_efficiency = metrics.successful_decisions as f64 / metrics.total_decisions as f64;

        Ok(())
    }

    /// Analyze request complexity
    async fn analyze_request_complexity(&self, _request: &OrchestrationRequest) -> Result<f64, KambuzumaError> {
        // Simplified complexity analysis
        Ok(0.5) // Medium complexity
    }

    /// Analyze resource requirements
    async fn analyze_resource_requirements(
        &self,
        _request: &OrchestrationRequest,
    ) -> Result<ResourceRequirements, KambuzumaError> {
        // Simplified resource analysis
        Ok(ResourceRequirements {
            cpu_requirements: 0.5,
            memory_requirements: 0.5,
            energy_requirements: 0.5,
            time_requirements: 0.5,
        })
    }

    /// Analyze constraints
    async fn analyze_constraints(
        &self,
        _request: &OrchestrationRequest,
    ) -> Result<Vec<AnalysisConstraint>, KambuzumaError> {
        // Simplified constraint analysis
        Ok(Vec::new())
    }
}
