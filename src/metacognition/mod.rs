//! Metacognitive orchestration subsystem for Kambuzuma
//! 
//! Implements Bayesian network-based reasoning transparency and decision-making

pub mod bayesian_network;
pub mod awareness_monitoring;
pub mod orchestration_control;
pub mod decision_making;
pub mod transparency;

use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;
use crate::{ComputationalResult, Result, KambuzumaError};
use std::collections::HashMap;
use uuid::Uuid;

/// Metacognitive subsystem configuration
#[derive(Debug, Clone)]
pub struct MetacognitiveConfig {
    /// Bayesian network configuration
    pub bayesian_config: bayesian_network::BayesianNetworkConfig,
    
    /// Awareness monitoring configuration
    pub awareness_config: awareness_monitoring::AwarenessConfig,
    
    /// Orchestration control configuration
    pub orchestration_config: orchestration_control::OrchestrationConfig,
    
    /// Decision-making configuration
    pub decision_config: decision_making::DecisionConfig,
    
    /// Transparency configuration
    pub transparency_config: transparency::TransparencyConfig,
}

/// Metacognitive subsystem state
#[derive(Debug, Clone)]
pub struct MetacognitiveState {
    /// Current awareness level (0.0 to 1.0)
    pub awareness_level: f64,
    
    /// Bayesian network confidence
    pub bayesian_confidence: f64,
    
    /// Decision-making accuracy
    pub decision_accuracy: f64,
    
    /// Reasoning transparency score
    pub transparency_score: f64,
    
    /// Current orchestration mode
    pub orchestration_mode: OrchestrationMode,
    
    /// Active awareness categories
    pub awareness_categories: HashMap<AwarenessCategory, f64>,
}

/// Orchestration modes for different processing strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrchestrationMode {
    /// Sequential processing through stages
    Sequential,
    /// Parallel processing across stages
    Parallel,
    /// Adaptive routing based on task complexity
    Adaptive,
    /// Emergency mode for critical tasks
    Emergency,
}

/// Categories of metacognitive awareness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AwarenessCategory {
    /// Awareness of processing capabilities
    ProcessAwareness,
    /// Awareness of knowledge states
    KnowledgeAwareness,
    /// Awareness of information gaps
    GapAwareness,
    /// Awareness of decision quality
    DecisionAwareness,
}

/// Orchestration decision for computational results
#[derive(Debug, Clone)]
pub struct OrchestrationDecision {
    /// Whether to accept the result
    pub accept_result: bool,
    
    /// Confidence in the decision
    pub confidence: f64,
    
    /// Suggested improvements
    pub improvements: Vec<String>,
    
    /// Risk assessment
    pub risk_level: RiskLevel,
    
    /// Explanation of decision
    pub explanation: String,
}

/// Risk levels for decision-making
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Metacognitive orchestration subsystem
pub struct MetacognitiveSubsystem {
    /// Configuration
    config: MetacognitiveConfig,
    
    /// Bayesian network for probabilistic reasoning
    bayesian_network: Arc<RwLock<bayesian_network::BayesianNetwork>>,
    
    /// Awareness monitoring system
    awareness_monitor: Arc<RwLock<awareness_monitoring::AwarenessMonitor>>,
    
    /// Orchestration control system
    orchestration_controller: Arc<RwLock<orchestration_control::OrchestrationController>>,
    
    /// Decision-making system
    decision_maker: Arc<RwLock<decision_making::DecisionMaker>>,
    
    /// Transparency system
    transparency_system: Arc<RwLock<transparency::TransparencySystem>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<MetacognitiveMetrics>>,
}

/// Metacognitive performance metrics
#[derive(Debug, Default)]
pub struct MetacognitiveMetrics {
    /// Total decisions made
    pub total_decisions: u64,
    
    /// Decision accuracy rate
    pub decision_accuracy: f64,
    
    /// Average reasoning transparency
    pub average_transparency: f64,
    
    /// Awareness stability
    pub awareness_stability: f64,
    
    /// Orchestration efficiency
    pub orchestration_efficiency: f64,
    
    /// Bayesian network convergence rate
    pub bayesian_convergence: f64,
}

impl MetacognitiveSubsystem {
    /// Create new metacognitive subsystem
    pub fn new(config: &MetacognitiveConfig) -> Result<Self> {
        // Initialize Bayesian network
        let bayesian_network = Arc::new(RwLock::new(
            bayesian_network::BayesianNetwork::new(&config.bayesian_config)?
        ));
        
        // Initialize awareness monitor
        let awareness_monitor = Arc::new(RwLock::new(
            awareness_monitoring::AwarenessMonitor::new(&config.awareness_config)?
        ));
        
        // Initialize orchestration controller
        let orchestration_controller = Arc::new(RwLock::new(
            orchestration_control::OrchestrationController::new(&config.orchestration_config)?
        ));
        
        // Initialize decision maker
        let decision_maker = Arc::new(RwLock::new(
            decision_making::DecisionMaker::new(&config.decision_config)?
        ));
        
        // Initialize transparency system
        let transparency_system = Arc::new(RwLock::new(
            transparency::TransparencySystem::new(&config.transparency_config)?
        ));
        
        // Initialize metrics
        let metrics = Arc::new(RwLock::new(MetacognitiveMetrics::default()));
        
        Ok(Self {
            config: config.clone(),
            bayesian_network,
            awareness_monitor,
            orchestration_controller,
            decision_maker,
            transparency_system,
            metrics,
        })
    }
    
    /// Start the metacognitive subsystem
    pub async fn start(&self) -> Result<()> {
        self.bayesian_network.write().await.start().await?;
        self.awareness_monitor.write().await.start().await?;
        self.orchestration_controller.write().await.start().await?;
        self.decision_maker.write().await.start().await?;
        self.transparency_system.write().await.start().await?;
        
        Ok(())
    }
    
    /// Stop the metacognitive subsystem
    pub async fn stop(&self) -> Result<()> {
        self.transparency_system.write().await.stop().await?;
        self.decision_maker.write().await.stop().await?;
        self.orchestration_controller.write().await.stop().await?;
        self.awareness_monitor.write().await.stop().await?;
        self.bayesian_network.write().await.stop().await?;
        
        Ok(())
    }
    
    /// Orchestrate computational result through metacognitive analysis
    pub async fn orchestrate(&self, result: &ComputationalResult) -> Result<ComputationalResult> {
        // Update awareness based on result
        self.awareness_monitor.write().await
            .update_awareness_from_result(result).await?;
        
        // Analyze result through Bayesian network
        let bayesian_analysis = self.bayesian_network.write().await
            .analyze_result(result).await?;
        
        // Make orchestration decision
        let orchestration_decision = self.decision_maker.write().await
            .make_orchestration_decision(result, &bayesian_analysis).await?;
        
        // Generate transparency explanation
        let explanation = self.transparency_system.write().await
            .generate_explanation(result, &orchestration_decision).await?;
        
        // Update orchestration control
        self.orchestration_controller.write().await
            .update_from_decision(&orchestration_decision).await?;
        
        // Create enhanced result with metacognitive information
        let mut enhanced_result = result.clone();
        enhanced_result.confidence *= orchestration_decision.confidence;
        enhanced_result.explanation = format!("{}\n\nMetacognitive Analysis: {}", 
                                            result.explanation, 
                                            explanation);
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_decisions += 1;
        metrics.decision_accuracy = (metrics.decision_accuracy + orchestration_decision.confidence) / 2.0;
        
        Ok(enhanced_result)
    }
    
    /// Get current metacognitive state
    pub async fn get_state(&self) -> Result<MetacognitiveState> {
        // Get awareness level
        let awareness_level = self.awareness_monitor.read().await.get_overall_awareness().await?;
        
        // Get Bayesian confidence
        let bayesian_confidence = self.bayesian_network.read().await.get_confidence().await?;
        
        // Get decision accuracy
        let decision_accuracy = self.decision_maker.read().await.get_accuracy().await?;
        
        // Get transparency score
        let transparency_score = self.transparency_system.read().await.get_transparency_score().await?;
        
        // Get orchestration mode
        let orchestration_mode = self.orchestration_controller.read().await.get_current_mode().await?;
        
        // Get awareness categories
        let awareness_categories = self.awareness_monitor.read().await.get_awareness_categories().await?;
        
        Ok(MetacognitiveState {
            awareness_level,
            bayesian_confidence,
            decision_accuracy,
            transparency_score,
            orchestration_mode,
            awareness_categories,
        })
    }
    
    /// Validate biological constraints
    pub async fn validate_biological_constraints(&self) -> Result<()> {
        // Validate Bayesian network constraints
        self.bayesian_network.read().await.validate_biological_constraints().await?;
        
        // Validate awareness constraints
        self.awareness_monitor.read().await.validate_biological_constraints().await?;
        
        // Validate decision-making constraints
        self.decision_maker.read().await.validate_biological_constraints().await?;
        
        Ok(())
    }
    
    /// Update metacognitive parameters based on learning
    pub async fn update_metacognitive_learning(&self, learning_data: &LearningData) -> Result<()> {
        // Update Bayesian network parameters
        self.bayesian_network.write().await.update_parameters(learning_data).await?;
        
        // Update awareness thresholds
        self.awareness_monitor.write().await.update_thresholds(learning_data).await?;
        
        // Update decision-making criteria
        self.decision_maker.write().await.update_criteria(learning_data).await?;
        
        Ok(())
    }
    
    /// Generate detailed reasoning trace
    pub async fn generate_reasoning_trace(&self, result: &ComputationalResult) -> Result<ReasoningTrace> {
        self.transparency_system.read().await.generate_reasoning_trace(result).await
    }
}

/// Learning data for metacognitive updates
#[derive(Debug, Clone)]
pub struct LearningData {
    /// Success rate of recent decisions
    pub success_rate: f64,
    
    /// Accuracy improvements needed
    pub accuracy_improvements: Vec<String>,
    
    /// Awareness calibration data
    pub awareness_calibration: HashMap<AwarenessCategory, f64>,
    
    /// Bayesian parameter updates
    pub bayesian_updates: HashMap<String, f64>,
}

/// Detailed reasoning trace for transparency
#[derive(Debug, Clone)]
pub struct ReasoningTrace {
    /// Step-by-step reasoning process
    pub reasoning_steps: Vec<ReasoningStep>,
    
    /// Confidence evolution
    pub confidence_evolution: Vec<f64>,
    
    /// Awareness contributions
    pub awareness_contributions: HashMap<AwarenessCategory, f64>,
    
    /// Bayesian inference chain
    pub bayesian_inference: Vec<String>,
    
    /// Decision factors
    pub decision_factors: Vec<DecisionFactor>,
}

/// Individual reasoning step
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Step description
    pub description: String,
    
    /// Input information
    pub inputs: Vec<String>,
    
    /// Processing method
    pub method: String,
    
    /// Output information
    pub outputs: Vec<String>,
    
    /// Confidence in this step
    pub confidence: f64,
}

/// Factor influencing a decision
#[derive(Debug, Clone)]
pub struct DecisionFactor {
    /// Factor name
    pub name: String,
    
    /// Influence weight
    pub weight: f64,
    
    /// Positive or negative influence
    pub influence_type: InfluenceType,
    
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Type of influence on decision
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InfluenceType {
    Positive,
    Negative,
    Neutral,
}

/// Metacognitive subsystem errors
#[derive(Debug, Error)]
pub enum MetacognitiveError {
    #[error("Bayesian network error: {0}")]
    BayesianNetwork(String),
    
    #[error("Awareness monitoring error: {0}")]
    AwarenessMonitoring(String),
    
    #[error("Orchestration control error: {0}")]
    OrchestrationControl(String),
    
    #[error("Decision making error: {0}")]
    DecisionMaking(String),
    
    #[error("Transparency error: {0}")]
    Transparency(String),
    
    #[error("Learning error: {0}")]
    Learning(String),
    
    #[error("Biological constraint violation: {0}")]
    BiologicalConstraintViolation(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl Default for MetacognitiveConfig {
    fn default() -> Self {
        Self {
            bayesian_config: bayesian_network::BayesianNetworkConfig::default(),
            awareness_config: awareness_monitoring::AwarenessConfig::default(),
            orchestration_config: orchestration_control::OrchestrationConfig::default(),
            decision_config: decision_making::DecisionConfig::default(),
            transparency_config: transparency::TransparencyConfig::default(),
        }
    }
}

impl MetacognitiveConfig {
    /// Validate configuration
    pub fn is_valid(&self) -> bool {
        self.bayesian_config.is_valid() &&
        self.awareness_config.is_valid() &&
        self.orchestration_config.is_valid() &&
        self.decision_config.is_valid() &&
        self.transparency_config.is_valid()
    }
}

impl Default for LearningData {
    fn default() -> Self {
        Self {
            success_rate: 0.8,
            accuracy_improvements: Vec::new(),
            awareness_calibration: HashMap::new(),
            bayesian_updates: HashMap::new(),
        }
    }
} 