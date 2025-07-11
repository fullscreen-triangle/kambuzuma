//! Shared types for the Kambuzuma biological quantum computing system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// State vector amplitudes
    pub amplitudes: Vec<f64>,
    /// Quantum phase information
    pub phases: Vec<f64>,
    /// Entanglement correlations
    pub entanglement_matrix: Vec<Vec<f64>>,
    /// Coherence time remaining
    pub coherence_time: f64,
    /// Fidelity score
    pub fidelity: f64,
}

/// Neural signal representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralSignal {
    /// Signal amplitude in mV
    pub amplitude: f64,
    /// Signal frequency in Hz
    pub frequency: f64,
    /// Signal phase in radians
    pub phase: f64,
    /// Signal timestamp
    pub timestamp: f64,
    /// Signal source neuron ID
    pub source_id: Uuid,
    /// Signal target neuron ID
    pub target_id: Uuid,
}

/// Biological membrane representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalMembrane {
    /// Membrane thickness in nm
    pub thickness: f64,
    /// Phospholipid composition
    pub lipid_composition: LipidComposition,
    /// Membrane potential in mV
    pub potential: f64,
    /// Ion concentrations
    pub ion_concentrations: IonConcentrations,
    /// Membrane temperature in K
    pub temperature: f64,
}

/// Phospholipid composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LipidComposition {
    /// Phosphatidylcholine percentage
    pub phosphatidylcholine: f64,
    /// Phosphatidylserine percentage
    pub phosphatidylserine: f64,
    /// Phosphatidylethanolamine percentage
    pub phosphatidylethanolamine: f64,
    /// Cholesterol percentage
    pub cholesterol: f64,
    /// Other lipids percentage
    pub other: f64,
}

/// Ion concentrations in the membrane
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonConcentrations {
    /// Sodium concentration in mM
    pub sodium: f64,
    /// Potassium concentration in mM
    pub potassium: f64,
    /// Calcium concentration in mM
    pub calcium: f64,
    /// Chloride concentration in mM
    pub chloride: f64,
    /// Magnesium concentration in mM
    pub magnesium: f64,
}

/// Ion channel representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonChannel {
    /// Channel identifier
    pub id: Uuid,
    /// Channel type
    pub channel_type: IonChannelType,
    /// Channel state
    pub state: IonChannelState,
    /// Conductance in pS
    pub conductance: f64,
    /// Selectivity filter properties
    pub selectivity_filter: SelectivityFilter,
    /// Gating properties
    pub gating_properties: GatingProperties,
}

/// Ion channel types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IonChannelType {
    /// Voltage-gated sodium channel
    VoltageGatedSodium,
    /// Voltage-gated potassium channel
    VoltageGatedPotassium,
    /// Voltage-gated calcium channel
    VoltageGatedCalcium,
    /// Ligand-gated channel
    LigandGated,
    /// Mechanosensitive channel
    Mechanosensitive,
    /// Leak channel
    Leak,
}

/// Ion channel states
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IonChannelState {
    /// Channel is closed
    Closed,
    /// Channel is open
    Open,
    /// Channel is inactivated
    Inactivated,
    /// Channel is in transition
    Transitioning,
}

/// Selectivity filter properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectivityFilter {
    /// Pore diameter in nm
    pub pore_diameter: f64,
    /// Binding site affinity
    pub binding_affinity: f64,
    /// Ion selectivity ratios
    pub selectivity_ratios: HashMap<String, f64>,
}

/// Gating properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingProperties {
    /// Activation voltage in mV
    pub activation_voltage: f64,
    /// Inactivation voltage in mV
    pub inactivation_voltage: f64,
    /// Activation time constant in ms
    pub activation_time_constant: f64,
    /// Inactivation time constant in ms
    pub inactivation_time_constant: f64,
}

/// Thought current representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtCurrent {
    /// Current magnitude in pA
    pub magnitude: f64,
    /// Current direction vector
    pub direction: Vec<f64>,
    /// Current type
    pub current_type: ThoughtCurrentType,
    /// Source processing stage
    pub source_stage: u8,
    /// Target processing stage
    pub target_stage: u8,
    /// Information content
    pub information_content: f64,
}

/// Thought current types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ThoughtCurrentType {
    /// Excitatory current
    Excitatory,
    /// Inhibitory current
    Inhibitory,
    /// Modulatory current
    Modulatory,
    /// Feedback current
    Feedback,
    /// Lateral current
    Lateral,
}

/// Maxwell demon state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxwellDemonState {
    /// Demon identifier
    pub id: Uuid,
    /// Information detection threshold
    pub detection_threshold: f64,
    /// Energy cost per operation in ATP
    pub energy_cost: f64,
    /// Success rate
    pub success_rate: f64,
    /// Current information state
    pub information_state: InformationState,
    /// Molecular machinery state
    pub machinery_state: MachineryState,
}

/// Information states for Maxwell demon
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InformationState {
    /// No information detected
    NoInformation,
    /// Information detected
    InformationDetected,
    /// Information being processed
    ProcessingInformation,
    /// Information processed
    InformationProcessed,
    /// Error state
    Error,
}

/// Molecular machinery states
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MachineryState {
    /// Machinery is idle
    Idle,
    /// Machinery is active
    Active,
    /// Machinery is transitioning
    Transitioning,
    /// Machinery is blocked
    Blocked,
    /// Machinery is in error state
    Error,
}

/// Neuron activation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronActivationPattern {
    /// Pattern identifier
    pub id: Uuid,
    /// Activation sequence
    pub activation_sequence: Vec<f64>,
    /// Temporal pattern
    pub temporal_pattern: Vec<f64>,
    /// Spatial pattern
    pub spatial_pattern: Vec<f64>,
    /// Pattern frequency
    pub frequency: f64,
    /// Pattern amplitude
    pub amplitude: f64,
}

/// Bayesian belief state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianBeliefState {
    /// Prior probabilities
    pub priors: HashMap<String, f64>,
    /// Likelihood functions
    pub likelihoods: HashMap<String, f64>,
    /// Posterior probabilities
    pub posteriors: HashMap<String, f64>,
    /// Evidence
    pub evidence: f64,
    /// Confidence level
    pub confidence: f64,
}

/// Language capability matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageCapabilityMatrix {
    /// Language identifier
    pub language: String,
    /// Performance characteristics
    pub performance: HashMap<String, f64>,
    /// Ecosystem maturity
    pub ecosystem_maturity: f64,
    /// Learning curve
    pub learning_curve: f64,
    /// Community support
    pub community_support: f64,
    /// Capability scores
    pub capabilities: HashMap<String, f64>,
}

/// Tool orchestration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOrchestrationResult {
    /// Result identifier
    pub id: Uuid,
    /// Success status
    pub success: bool,
    /// Tool used
    pub tool_used: String,
    /// Execution time
    pub execution_time: std::time::Duration,
    /// Resource usage
    pub resource_usage: HashMap<String, f64>,
    /// Output data
    pub output_data: Vec<u8>,
    /// Error message if any
    pub error_message: Option<String>,
}

/// Biological validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalValidationResult {
    /// Validation identifier
    pub id: Uuid,
    /// Overall validation status
    pub overall_status: bool,
    /// Individual constraint results
    pub constraint_results: HashMap<String, ConstraintResult>,
    /// Validation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Validation confidence
    pub confidence: f64,
}

/// Individual constraint validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintResult {
    /// Constraint name
    pub name: String,
    /// Expected value
    pub expected_value: f64,
    /// Actual value
    pub actual_value: f64,
    /// Tolerance
    pub tolerance: f64,
    /// Validation status
    pub status: bool,
    /// Deviation percentage
    pub deviation_percentage: f64,
}

/// Mathematical operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalOperationResult {
    /// Operation identifier
    pub id: Uuid,
    /// Operation type
    pub operation_type: String,
    /// Input parameters
    pub input_parameters: HashMap<String, f64>,
    /// Result value
    pub result_value: f64,
    /// Numerical accuracy
    pub numerical_accuracy: f64,
    /// Computation time
    pub computation_time: std::time::Duration,
    /// Memory usage
    pub memory_usage: u64,
}

/// Default implementations for commonly used types
impl Default for QuantumState {
    fn default() -> Self {
        Self {
            amplitudes: vec![1.0, 0.0], // |0⟩ state
            phases: vec![0.0, 0.0],
            entanglement_matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            coherence_time: 0.001, // 1 ms
            fidelity: 1.0,
        }
    }
}

impl Default for LipidComposition {
    fn default() -> Self {
        // Typical neuronal membrane composition
        Self {
            phosphatidylcholine: 45.0,
            phosphatidylserine: 15.0,
            phosphatidylethanolamine: 25.0,
            cholesterol: 10.0,
            other: 5.0,
        }
    }
}

impl Default for IonConcentrations {
    fn default() -> Self {
        // Typical physiological concentrations
        Self {
            sodium: 12.0,     // mM intracellular
            potassium: 140.0, // mM intracellular
            calcium: 0.0001,  // mM intracellular
            chloride: 10.0,   // mM intracellular
            magnesium: 1.0,   // mM intracellular
        }
    }
}

impl Default for BiologicalMembrane {
    fn default() -> Self {
        Self {
            thickness: 5.0, // nm
            lipid_composition: LipidComposition::default(),
            potential: -70.0, // mV
            ion_concentrations: IonConcentrations::default(),
            temperature: 310.15, // K (37°C)
        }
    }
}

impl Default for SelectivityFilter {
    fn default() -> Self {
        Self {
            pore_diameter: 1.0,     // nm
            binding_affinity: 1e-6, // M
            selectivity_ratios: HashMap::new(),
        }
    }
}

impl Default for GatingProperties {
    fn default() -> Self {
        Self {
            activation_voltage: -55.0,        // mV
            inactivation_voltage: -40.0,      // mV
            activation_time_constant: 1.0,    // ms
            inactivation_time_constant: 10.0, // ms
        }
    }
}

impl Default for BayesianBeliefState {
    fn default() -> Self {
        Self {
            priors: HashMap::new(),
            likelihoods: HashMap::new(),
            posteriors: HashMap::new(),
            evidence: 1.0,
            confidence: 0.5,
        }
    }
}
