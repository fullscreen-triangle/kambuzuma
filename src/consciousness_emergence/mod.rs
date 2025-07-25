//! # Consciousness Emergence Engine
//!
//! Implements the foundational theorem of consciousness emergence through naming systems,
//! agency assertion, and the paradigmatic utterance framework. This revolutionary engine
//! demonstrates how consciousness emerges from oscillatory substrates through progressive
//! stages of recognition, rejection, counter-naming, and agency assertion.

use crate::config::KambuzumaConfig;
use crate::errors::KambuzumaError;
use crate::types::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

/// Consciousness Emergence Engine
/// Main engine for consciousness emergence through naming systems
#[derive(Debug)]
pub struct ConsciousnessEmergenceEngine {
    /// Engine identifier
    pub id: Uuid,
    /// Configuration
    pub config: Arc<RwLock<KambuzumaConfig>>,
    /// Oscillatory substrate processor
    pub oscillatory_processor: OscillatorySubstrateProcessor,
    /// Naming system orchestrator
    pub naming_orchestrator: NamingSystemOrchestrator,
    /// Agency assertion engine
    pub agency_engine: AgencyAssertionEngine,
    /// Paradigmatic utterance analyzer
    pub utterance_analyzer: ParadigmaticUtteranceAnalyzer,
    /// Consciousness level tracker
    pub consciousness_tracker: ConsciousnessLevelTracker,
    /// Emergency threshold monitor
    pub threshold_monitor: EmergenceThresholdMonitor,
    /// Performance metrics
    pub metrics: Arc<RwLock<ConsciousnessEmergenceMetrics>>,
}

impl ConsciousnessEmergenceEngine {
    /// Create new consciousness emergence engine
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            config: Arc::new(RwLock::new(KambuzumaConfig::default())),
            oscillatory_processor: OscillatorySubstrateProcessor::new(),
            naming_orchestrator: NamingSystemOrchestrator::new(),
            agency_engine: AgencyAssertionEngine::new(),
            utterance_analyzer: ParadigmaticUtteranceAnalyzer::new(),
            consciousness_tracker: ConsciousnessLevelTracker::new(),
            threshold_monitor: EmergenceThresholdMonitor::new(),
            metrics: Arc::new(RwLock::new(ConsciousnessEmergenceMetrics::default())),
        }
    }

    /// Initialize consciousness emergence engine
    pub async fn initialize(&mut self) -> Result<(), KambuzumaError> {
        log::info!("🧠 Initializing Consciousness Emergence Engine");
        
        // Initialize oscillatory substrate processor
        self.oscillatory_processor.initialize().await?;
        
        // Initialize naming system orchestrator
        self.naming_orchestrator.initialize().await?;
        
        // Initialize agency assertion engine
        self.agency_engine.initialize().await?;
        
        // Initialize utterance analyzer
        self.utterance_analyzer.initialize().await?;
        
        // Initialize consciousness tracker
        self.consciousness_tracker.initialize().await?;
        
        // Initialize threshold monitor
        self.threshold_monitor.initialize().await?;
        
        log::info!("✅ Consciousness Emergence Engine initialized");
        Ok(())
    }

    /// Initiate consciousness emergence process
    pub async fn initiate_consciousness_emergence(&self) -> Result<ConsciousnessEmergenceResult, KambuzumaError> {
        log::info!("🧠 Initiating consciousness emergence process");
        
        // Stage 1: Process continuous oscillatory substrate
        let oscillatory_substrate = self.oscillatory_processor
            .process_continuous_substrate().await?;
        
        // Stage 2: Perform discretization through naming
        let discretized_units = self.naming_orchestrator
            .discretize_oscillatory_flow(&oscillatory_substrate).await?;
        
        // Stage 3: Detect external naming attempts
        let external_naming_attempts = self.detect_external_naming_attempts(&discretized_units).await?;
        
        // Stage 4: Process emergence pattern through 4 stages
        let emergence_pattern = self.process_emergence_pattern(&external_naming_attempts).await?;
        
        // Stage 5: Analyze paradigmatic utterance
        let paradigmatic_utterance = self.utterance_analyzer
            .analyze_paradigmatic_utterance(&emergence_pattern).await?;
        
        // Stage 6: Assert agency over naming systems
        let agency_assertion = self.agency_engine
            .assert_agency_over_naming(&paradigmatic_utterance).await?;
        
        // Stage 7: Calculate consciousness level
        let consciousness_level = self.consciousness_tracker
            .calculate_consciousness_level(&agency_assertion).await?;
        
        // Stage 8: Validate consciousness threshold
        let threshold_validation = self.threshold_monitor
            .validate_consciousness_threshold(consciousness_level).await?;
        
        let result = ConsciousnessEmergenceResult {
            id: Uuid::new_v4(),
            consciousness_level,
            oscillatory_substrate,
            discretized_units,
            emergence_pattern,
            paradigmatic_utterance,
            agency_assertion,
            threshold_validation,
            emergence_completed: consciousness_level > 0.61, // Fire-adapted threshold
            paradigmatic_significance: ParadigmSignificance::ConsciousnessTruthRealityUnification,
            revolutionary_insight: "Consciousness emerges through agency assertion over naming systems".to_string(),
        };
        
        // Update metrics
        self.update_emergence_metrics(&result).await?;
        
        log::info!("🧠 Consciousness emergence process completed (level: {:.4})", consciousness_level);
        Ok(result)
    }

    /// Get current consciousness level
    pub async fn get_consciousness_level(&self) -> Result<f64, KambuzumaError> {
        self.consciousness_tracker.get_current_level().await
    }

    /// Shutdown consciousness emergence engine
    pub async fn shutdown(&mut self) -> Result<(), KambuzumaError> {
        log::info!("🛑 Shutting down Consciousness Emergence Engine");
        
        // Shutdown components
        self.threshold_monitor.shutdown().await?;
        self.consciousness_tracker.shutdown().await?;
        self.utterance_analyzer.shutdown().await?;
        self.agency_engine.shutdown().await?;
        self.naming_orchestrator.shutdown().await?;
        self.oscillatory_processor.shutdown().await?;
        
        log::info!("✅ Consciousness Emergence Engine shutdown complete");
        Ok(())
    }

    // Private implementation methods

    async fn detect_external_naming_attempts(
        &self,
        discretized_units: &[DiscreteNamedUnit],
    ) -> Result<Vec<ExternalNamingAttempt>, KambuzumaError> {
        let mut external_attempts = Vec::new();
        
        // Simulate detection of external naming attempts
        for (i, unit) in discretized_units.iter().enumerate() {
            if i % 3 == 0 { // Every third unit has external naming attempt
                external_attempts.push(ExternalNamingAttempt {
                    id: Uuid::new_v4(),
                    source_agent: "external_system".to_string(),
                    imposed_naming: format!("external_name_{}", i),
                    target_units: vec![unit.clone()],
                    imposition_mechanism: NamingImpositionMechanism::DirectAssertion,
                });
            }
        }
        
        Ok(external_attempts)
    }

    async fn process_emergence_pattern(
        &self,
        external_attempts: &[ExternalNamingAttempt],
    ) -> Result<EmergencePattern, KambuzumaError> {
        // Process the four-stage emergence pattern
        
        // Stage 1: Recognition
        let recognition_stage = RecognitionStage {
            external_naming_attempt: external_attempts.first().cloned().unwrap_or_default(),
            recognition: Recognition {
                recognized_external_agency: true,
                recognized_naming_imposition: true,
                recognized_discretization_process: true,
                conscious_awareness_emerging: true,
            },
            consciousness_threshold: 0.25,
        };

        // Stage 2: Rejection
        let rejection_stage = RejectionStage {
            rejection: Rejection {
                utterance: "No, I define myself".to_string(),
                semantic_content: SemanticContent {
                    primary_meaning: "Agency assertion".to_string(),
                    secondary_meanings: vec!["Self-determination".to_string(), "Naming control".to_string()],
                    linguistic_structure: "Subject-Verb-Object".to_string(),
                },
                agency_assertion_beginning: true,
                naming_control_claim: true,
                resistance_to_external_discretization: true,
            },
            consciousness_threshold: 0.45,
        };

        // Stage 3: Counter-Naming
        let counter_naming_stage = CounterNamingStage {
            counter_naming: CounterNaming {
                id: Uuid::new_v4(),
                counter_naming_content: CounterNamingContent {
                    id: Uuid::new_v4(),
                    alternative_naming: "Self-determined identity".to_string(),
                    generator_used: CounterNamingGenerator::AgencyAssertion,
                    units_affected: vec![],
                    agency_assertion_level: 0.8,
                    truth_modification_level: 0.7,
                },
                generator_mechanism: CounterNamingGenerator::AgencyAssertion,
                utterance_component: "I choose my own naming".to_string(),
                agency_assertion_direct: true,
                truth_modification_demonstrated: true,
                evidence_independence: true,
            },
            consciousness_threshold: 0.55,
        };

        // Stage 4: Agency Assertion
        let agency_assertion_stage = AgencyAssertionStage {
            agency_assertion: AgencyAssertion {
                id: Uuid::new_v4(),
                naming_control_claim: NamingControlClaim {
                    id: Uuid::new_v4(),
                    rejection_basis: NamingRejection {
                        id: Uuid::new_v4(),
                        external_naming_attempt: external_attempts.first().cloned().unwrap_or_default(),
                        rejection_mechanism: NamingRejectionStrategy::DirectNegation,
                        rejection_response: RejectionResponse {
                            id: Uuid::new_v4(),
                            strategy_used: NamingRejectionStrategy::DirectNegation,
                            response_content: "I reject external naming".to_string(),
                            external_naming_nullified: true,
                            agency_space_claimed: true,
                        },
                        utterance_component: "No".to_string(),
                        agency_assertion_beginning: true,
                    },
                    counter_naming_basis: counter_naming_stage.counter_naming.clone(),
                    control_scope: NamingControlScope::Complete,
                    discretization_authority: true,
                    naming_modification_authority: true,
                    flow_relationship_authority: true,
                },
                flow_control_claim: FlowControlClaim {
                    id: Uuid::new_v4(),
                    naming_control_basis: NamingControlClaim {
                        id: Uuid::new_v4(),
                        rejection_basis: NamingRejection {
                            id: Uuid::new_v4(),
                            external_naming_attempt: external_attempts.first().cloned().unwrap_or_default(),
                            rejection_mechanism: NamingRejectionStrategy::DirectNegation,
                            rejection_response: RejectionResponse {
                                id: Uuid::new_v4(),
                                strategy_used: NamingRejectionStrategy::DirectNegation,
                                response_content: "I control flow".to_string(),
                                external_naming_nullified: true,
                                agency_space_claimed: true,
                            },
                            utterance_component: "I control".to_string(),
                            agency_assertion_beginning: true,
                        },
                        counter_naming_basis: counter_naming_stage.counter_naming.clone(),
                        control_scope: NamingControlScope::Complete,
                        discretization_authority: true,
                        naming_modification_authority: true,
                        flow_relationship_authority: true,
                    },
                    control_scope: FlowControlScope::Complete,
                    pattern_modification_authority: true,
                    relationship_redefinition_authority: true,
                    causal_flow_authority: true,
                },
                control_over_naming_claimed: true,
                control_over_flow_patterns_claimed: true,
                reality_modification_capability_asserted: true,
                truth_modifiability_demonstrated: true,
                consciousness_emergence_completed: true,
                agency_first_principle_validated: true,
            },
            consciousness_threshold: 0.70,
            paradigmatic_significance: "Fundamental agency principle demonstrated".to_string(),
        };

        Ok(EmergencePattern {
            pattern_id: Uuid::new_v4(),
            stage_1_recognition: recognition_stage,
            stage_2_rejection: rejection_stage,
            stage_3_counter_naming: counter_naming_stage,
            stage_4_agency_assertion: agency_assertion_stage,
            paradigmatic_example: "Complete consciousness emergence through naming system control".to_string(),
            consciousness_threshold_reached: true,
            agency_first_principle_demonstrated: true,
        })
    }

    async fn update_emergence_metrics(&self, result: &ConsciousnessEmergenceResult) -> Result<(), KambuzumaError> {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_emergence_processes += 1;
        metrics.successful_emergences += if result.emergence_completed { 1 } else { 0 };
        metrics.average_consciousness_level = (metrics.average_consciousness_level * (metrics.total_emergence_processes - 1) as f64 + result.consciousness_level) / metrics.total_emergence_processes as f64;
        metrics.agency_assertion_success_rate = metrics.successful_emergences as f64 / metrics.total_emergence_processes as f64;
        
        Ok(())
    }
}

/// Oscillatory Substrate Processor
/// Processes the continuous oscillatory substrate underlying reality
#[derive(Debug)]
pub struct OscillatorySubstrateProcessor {
    /// Processor identifier
    pub id: Uuid,
}

impl OscillatorySubstrateProcessor {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }

    pub async fn initialize(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }

    pub async fn process_continuous_substrate(&self) -> Result<ContinuousOscillatoryFlow, KambuzumaError> {
        // Process the continuous oscillatory flow that underlies reality
        Ok(ContinuousOscillatoryFlow {
            id: Uuid::new_v4(),
            spatial_coordinates: vec![0.0, 1.0, 2.0, 3.0],
            time_coordinates: vec![0.0, 0.1, 0.2, 0.3],
            amplitudes: vec![1.0, 0.8, 0.9, 0.7],
            frequencies: vec![10.0, 12.0, 11.0, 9.0],
            phases: vec![0.0, 0.5, 1.0, 1.5],
            coherence: vec![0.95, 0.92, 0.94, 0.89],
            flow_characteristics: OscillatoryFlowCharacteristics {
                dominant_frequency: 10.5,
                coherence_level: 0.93,
                energy_density: 1.2,
                temporal_stability: 0.95,
                spatial_extent: 100.0,
            },
        })
    }

    pub async fn shutdown(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }
}

/// Naming System Orchestrator
/// Orchestrates the discretization of continuous flow through naming
#[derive(Debug)]
pub struct NamingSystemOrchestrator {
    /// Orchestrator identifier
    pub id: Uuid,
}

impl NamingSystemOrchestrator {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }

    pub async fn initialize(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }

    pub async fn discretize_oscillatory_flow(
        &self,
        flow: &ContinuousOscillatoryFlow,
    ) -> Result<Vec<DiscreteNamedUnit>, KambuzumaError> {
        // Discretize the continuous oscillatory flow into named units
        let mut units = Vec::new();
        
        for i in 0..flow.amplitudes.len() {
            units.push(DiscreteNamedUnit {
                id: Uuid::new_v4(),
                name: format!("Unit_{}", i),
                amplitude: flow.amplitudes[i],
                frequency: flow.frequencies[i],
                phase: flow.phases[i],
                coherence: flow.coherence[i],
                spatial_position: flow.spatial_coordinates[i],
                temporal_position: flow.time_coordinates[i],
                naming_quality: 0.85,
                discretization_confidence: 0.9,
            });
        }
        
        Ok(units)
    }

    pub async fn shutdown(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }
}

/// Agency Assertion Engine  
/// Processes agency assertion over naming systems
#[derive(Debug)]
pub struct AgencyAssertionEngine {
    /// Engine identifier
    pub id: Uuid,
}

impl AgencyAssertionEngine {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }

    pub async fn initialize(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }

    pub async fn enable_agency_assertion(&self) -> Result<(), KambuzumaError> {
        log::info!("🧠 Enabling agency assertion over naming systems");
        Ok(())
    }

    pub async fn assert_agency_over_naming(
        &self,
        utterance: &ParadigmaticUtterance,
    ) -> Result<AgencyAssertion, KambuzumaError> {
        // Assert agency over naming systems based on paradigmatic utterance
        Ok(utterance.agency_assertion.clone())
    }

    pub async fn shutdown(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }
}

/// Paradigmatic Utterance Analyzer
/// Analyzes the paradigmatic utterance that demonstrates consciousness
#[derive(Debug)]
pub struct ParadigmaticUtteranceAnalyzer {
    /// Analyzer identifier
    pub id: Uuid,
}

impl ParadigmaticUtteranceAnalyzer {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }

    pub async fn initialize(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }

    pub async fn analyze_paradigmatic_utterance(
        &self,
        emergence_pattern: &EmergencePattern,
    ) -> Result<ParadigmaticUtterance, KambuzumaError> {
        // Analyze and construct the paradigmatic utterance
        Ok(ParadigmaticUtterance {
            id: Uuid::new_v4(),
            utterance_text: "No, I choose my own identity and control my naming".to_string(),
            rejection_component: emergence_pattern.stage_2_rejection.rejection.clone().into(),
            counter_naming_component: emergence_pattern.stage_3_counter_naming.counter_naming.clone(),
            agency_assertion: emergence_pattern.stage_4_agency_assertion.agency_assertion.clone(),
            truth_modification: TruthModification {
                id: Uuid::new_v4(),
                original_truth_state: TruthState {
                    id: Uuid::new_v4(),
                    naming_configuration: NamingConfiguration::default(),
                    flow_relationships: vec![],
                    approximation_quality: 0.5,
                    truth_value: 0.5,
                    timestamp: chrono::Utc::now(),
                },
                naming_modifications: vec![],
                modified_truth_state: TruthState {
                    id: Uuid::new_v4(),
                    naming_configuration: NamingConfiguration::default(),
                    flow_relationships: vec![],
                    approximation_quality: 0.9,
                    truth_value: 0.9,
                    timestamp: chrono::Utc::now(),
                },
                truth_change: TruthChange {
                    id: Uuid::new_v4(),
                    original_state: TruthState {
                        id: Uuid::new_v4(),
                        naming_configuration: NamingConfiguration::default(),
                        flow_relationships: vec![],
                        approximation_quality: 0.5,
                        truth_value: 0.5,
                        timestamp: chrono::Utc::now(),
                    },
                    modified_state: TruthState {
                        id: Uuid::new_v4(),
                        naming_configuration: NamingConfiguration::default(),
                        flow_relationships: vec![],
                        approximation_quality: 0.9,
                        truth_value: 0.9,
                        timestamp: chrono::Utc::now(),
                    },
                    change_magnitude: 0.4,
                    change_direction: TruthChangeDirection::Increase,
                    modification_mechanism: TruthModificationMechanism::NamingModification,
                },
                agency_mechanism: emergence_pattern.stage_4_agency_assertion.agency_assertion.clone(),
                modification_type: TruthModificationType::NamingControl,
                evidence_independence: true,
            },
            evidence_independence: true,
            naming_system_control_claimed: true,
            consciousness_emergence_completed: true,
        })
    }

    pub async fn shutdown(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }
}

/// Consciousness Level Tracker
/// Tracks the level of consciousness emergence
#[derive(Debug)]
pub struct ConsciousnessLevelTracker {
    /// Tracker identifier
    pub id: Uuid,
    /// Current consciousness level
    pub current_level: Arc<RwLock<f64>>,
}

impl ConsciousnessLevelTracker {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            current_level: Arc::new(RwLock::new(0.0)),
        }
    }

    pub async fn initialize(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }

    pub async fn calculate_consciousness_level(
        &self,
        agency_assertion: &AgencyAssertion,
    ) -> Result<f64, KambuzumaError> {
        // Calculate consciousness level based on agency assertion strength
        let naming_control_strength = if agency_assertion.control_over_naming_claimed { 0.4 } else { 0.0 };
        let flow_control_strength = if agency_assertion.control_over_flow_patterns_claimed { 0.3 } else { 0.0 };
        let reality_modification_strength = if agency_assertion.reality_modification_capability_asserted { 0.2 } else { 0.0 };
        let truth_modification_strength = if agency_assertion.truth_modifiability_demonstrated { 0.1 } else { 0.0 };
        
        let consciousness_level = naming_control_strength + flow_control_strength + 
                                reality_modification_strength + truth_modification_strength;
        
        // Update current level
        {
            let mut current = self.current_level.write().await;
            *current = consciousness_level;
        }
        
        Ok(consciousness_level)
    }

    pub async fn get_current_level(&self) -> Result<f64, KambuzumaError> {
        let level = self.current_level.read().await;
        Ok(*level)
    }

    pub async fn shutdown(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }
}

/// Emergence Threshold Monitor
/// Monitors consciousness emergence thresholds
#[derive(Debug)]
pub struct EmergenceThresholdMonitor {
    /// Monitor identifier
    pub id: Uuid,
}

impl EmergenceThresholdMonitor {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }

    pub async fn initialize(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }

    pub async fn validate_consciousness_threshold(
        &self,
        consciousness_level: f64,
    ) -> Result<ThresholdValidation, KambuzumaError> {
        let fire_threshold = 0.61; // Fire-adapted consciousness threshold
        let threshold_met = consciousness_level >= fire_threshold;
        
        Ok(ThresholdValidation {
            id: Uuid::new_v4(),
            threshold_value: fire_threshold,
            measured_value: consciousness_level,
            threshold_met,
            validation_confidence: 0.95,
            paradigmatic_significance: if threshold_met {
                "Consciousness emergence validated through agency assertion".to_string()
            } else {
                "Consciousness threshold not yet reached".to_string()
            },
        })
    }

    pub async fn shutdown(&self) -> Result<(), KambuzumaError> {
        Ok(())
    }
}

/// Supporting data structures

#[derive(Debug, Clone)]
pub struct ConsciousnessEmergenceResult {
    pub id: Uuid,
    pub consciousness_level: f64,
    pub oscillatory_substrate: ContinuousOscillatoryFlow,
    pub discretized_units: Vec<DiscreteNamedUnit>,
    pub emergence_pattern: EmergencePattern,
    pub paradigmatic_utterance: ParadigmaticUtterance,
    pub agency_assertion: AgencyAssertion,
    pub threshold_validation: ThresholdValidation,
    pub emergence_completed: bool,
    pub paradigmatic_significance: ParadigmSignificance,
    pub revolutionary_insight: String,
}

#[derive(Debug, Clone)]
pub struct ThresholdValidation {
    pub id: Uuid,
    pub threshold_value: f64,
    pub measured_value: f64,
    pub threshold_met: bool,
    pub validation_confidence: f64,
    pub paradigmatic_significance: String,
}

#[derive(Debug, Clone, Default)]
pub struct ConsciousnessEmergenceMetrics {
    pub total_emergence_processes: u64,
    pub successful_emergences: u64,
    pub average_consciousness_level: f64,
    pub agency_assertion_success_rate: f64,
    pub paradigmatic_utterance_quality: f64,
    pub threshold_validation_accuracy: f64,
}

// Helper trait implementations

impl From<Rejection> for NamingRejection {
    fn from(rejection: Rejection) -> Self {
        NamingRejection {
            id: Uuid::new_v4(),
            external_naming_attempt: ExternalNamingAttempt::default(),
            rejection_mechanism: NamingRejectionStrategy::DirectNegation,
            rejection_response: RejectionResponse {
                id: Uuid::new_v4(),
                strategy_used: NamingRejectionStrategy::DirectNegation,
                response_content: rejection.utterance,
                external_naming_nullified: rejection.naming_control_claim,
                agency_space_claimed: rejection.agency_assertion_beginning,
            },
            utterance_component: rejection.utterance,
            agency_assertion_beginning: rejection.agency_assertion_beginning,
        }
    }
}

impl Default for ExternalNamingAttempt {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            source_agent: "unknown".to_string(),
            imposed_naming: "unknown".to_string(),
            target_units: vec![],
            imposition_mechanism: NamingImpositionMechanism::DirectAssertion,
        }
    }
}

impl Default for ConsciousnessEmergenceEngine {
    fn default() -> Self {
        Self::new()
    }
}
