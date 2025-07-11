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

/// Oscillation Pattern
/// Represents a quantum oscillation pattern for endpoint detection
#[derive(Debug, Clone)]
pub struct OscillationPattern {
    pub id: String,
    pub base_frequency: f64,
    pub initial_amplitude: f64,
    pub decay_rate: f64,
    pub phase: f64,
    pub time_elapsed: f64,
    pub pattern_type: OscillationPatternType,
}

/// Oscillation Pattern Type
#[derive(Debug, Clone)]
pub enum OscillationPatternType {
    Harmonic,
    Damped,
    Driven,
    Chaotic,
}

/// Oscillation Endpoint
/// Represents a detected oscillation termination point
#[derive(Debug, Clone)]
pub struct OscillationEndpoint {
    pub pattern_id: String,
    pub termination_time: f64,
    pub energy_available: f64,
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub confidence: f64,
}

/// Current Measurement
/// Represents a current measurement in voltage clamp
#[derive(Debug, Clone)]
pub struct CurrentMeasurement {
    pub time: f64,
    pub current: f64,
    pub voltage: f64,
}

/// Voltage Clamp Result
/// Results from voltage clamp simulation
#[derive(Debug, Clone)]
pub struct VoltageClampResult {
    pub current_trace: Vec<CurrentMeasurement>,
    pub voltage_error: f64,
    pub settling_time: f64,
    pub time_constant: f64,
    pub clamp_resistance: f64,
    pub membrane_capacitance: f64,
}

/// State Collapse Event
/// Represents a quantum state collapse event
#[derive(Debug, Clone)]
pub struct StateCollapseEvent {
    pub event_id: String,
    pub collapse_time: f64,
    pub initial_state: QuantumState,
    pub final_state: QuantumState,
    pub energy_released: f64,
    pub decoherence_rate: f64,
    pub measurement_basis: Vec<String>,
}

/// Energy Harvest Result
/// Results from energy harvesting operation
#[derive(Debug, Clone)]
pub struct EnergyHarvestResult {
    pub extracted_energy: f64,
    pub available_energy: f64,
    pub extraction_efficiency: f64,
    pub energy_storage: f64,
    pub extraction_details: Vec<EndpointExtractionResult>,
}

/// Endpoint Extraction Result
/// Results from extracting energy from a single endpoint
#[derive(Debug, Clone)]
pub struct EndpointExtractionResult {
    pub endpoint_id: String,
    pub energy_available: f64,
    pub energy_extracted: f64,
    pub extraction_efficiency: f64,
    pub extraction_time: f64,
}

/// Oscillation Cycle Result
/// Results from a complete oscillation cycle
#[derive(Debug, Clone)]
pub struct OscillationCycleResult {
    pub endpoints: Vec<OscillationEndpoint>,
    pub clamp_result: VoltageClampResult,
    pub energy_result: EnergyHarvestResult,
    pub cycle_efficiency: f64,
}

/// Information State
/// Represents the information content of a quantum system
#[derive(Debug, Clone)]
pub struct InformationState {
    pub information_content: f64,
    pub state_probabilities: Vec<f64>,
    pub information_quality: f64,
    pub detection_confidence: f64,
}

/// Maxwell Demon Result
/// Results from Maxwell demon information processing
#[derive(Debug, Clone)]
pub struct MaxwellDemonResult {
    pub success: bool,
    pub information_processed: f64,
    pub energy_consumed: f64,
    pub selectivity_achieved: f64,
}

/// Maxwell Demon Status
/// Status of a Maxwell demon
#[derive(Debug, Clone)]
pub struct MaxwellDemonStatus {
    pub id: uuid::Uuid,
    pub demon_type: DemonType,
    pub is_active: bool,
    pub information_processed: f64,
    pub energy_consumed: f64,
    pub selectivity_achieved: f64,
}

/// Maxwell Demon Array Status
/// Status of the entire Maxwell demon array
#[derive(Debug, Clone)]
pub struct MaxwellDemonArrayStatus {
    pub total_demons: usize,
    pub active_demons: usize,
    pub total_information_processed: f64,
    pub total_energy_consumed: f64,
    pub average_selectivity: f64,
}

/// Demon Type
/// Types of Maxwell demons
#[derive(Debug, Clone)]
pub enum DemonType {
    IonSelectivity,
    InformationProcessing,
    EnergyHarvesting,
    QuantumMeasurement,
}

/// Ion Mixture
/// Represents a mixture of ions for selectivity operations
#[derive(Debug, Clone)]
pub struct IonMixture {
    pub ions: Vec<Ion>,
    pub concentration: f64,
    pub temperature: f64,
    pub ph: f64,
}

impl IonMixture {
    pub fn total_ions(&self) -> f64 {
        self.ions.len() as f64 * self.concentration
    }
}

/// Ion
/// Individual ion in the mixture
#[derive(Debug, Clone)]
pub struct Ion {
    pub ion_type: IonType,
    pub charge: i8,
    pub radius: f64,
    pub binding_affinity: f64,
}

/// Ion Type
/// Types of ions in biological systems
#[derive(Debug, Clone)]
pub enum IonType {
    Sodium,
    Potassium,
    Calcium,
    Chloride,
    Magnesium,
    Phosphate,
    Custom(String),
}

/// Ion Selectivity Result
/// Results from ion selectivity operation
#[derive(Debug, Clone)]
pub struct IonSelectivityResult {
    pub success: bool,
    pub selected_ions: IonMixture,
    pub selectivity: f64,
    pub total_ions_processed: f64,
}

/// System State
/// Represents the state of a quantum system for information detection
#[derive(Debug, Clone)]
pub struct SystemState {
    pub state_probabilities: Vec<f64>,
    pub state_amplitudes: Vec<ComplexAmplitude>,
    pub state_count: usize,
    pub subsystem_count: usize,
}

/// Complex Amplitude
/// Complex amplitude for quantum states
#[derive(Debug, Clone)]
pub struct ComplexAmplitude {
    pub real: f64,
    pub imag: f64,
}

impl ComplexAmplitude {
    pub fn norm_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }
}

/// Conformational State
/// Represents protein conformational states
#[derive(Debug, Clone)]
pub struct ConformationalState {
    pub state_id: String,
    pub change_magnitude: ChangeMagnitude,
    pub energy_level: f64,
    pub stability: f64,
}

impl Default for ConformationalState {
    fn default() -> Self {
        Self {
            state_id: "default".to_string(),
            change_magnitude: ChangeMagnitude::Small,
            energy_level: 0.0,
            stability: 1.0,
        }
    }
}

/// Change Magnitude
/// Magnitude of conformational changes
#[derive(Debug, Clone)]
pub enum ChangeMagnitude {
    Small,
    Medium,
    Large,
}

/// Conformational Change Result
/// Results from conformational change operation
#[derive(Debug, Clone)]
pub struct ConformationalChangeResult {
    pub success: bool,
    pub final_state: ConformationalState,
    pub energy_consumed: f64,
    pub conformational_time: std::time::Duration,
}

/// Conformational Switch Result
/// Results from conformational switch operation
#[derive(Debug, Clone)]
pub struct ConformationalSwitchResult {
    pub success: bool,
    pub final_state: ConformationalState,
    pub energy_consumed: f64,
    pub switch_time: std::time::Duration,
}

/// Gate Control Result
/// Results from gate control operation
#[derive(Debug, Clone)]
pub struct GateControlResult {
    pub success: bool,
    pub gate_action: GateAction,
    pub selectivity: f64,
    pub control_time: std::time::Duration,
}

/// Gate Action
/// Actions that can be performed on gates
#[derive(Debug, Clone)]
pub enum GateAction {
    Open,
    Close,
    Maintain,
}

/// Gate State
/// Current state of a gate
#[derive(Debug, Clone)]
pub enum GateState {
    Open,
    Closed,
    Partially(f64), // 0.0 to 1.0
}

/// Protein Complex
/// Represents a protein complex in molecular machinery
#[derive(Debug, Clone)]
pub struct ProteinComplex {
    pub id: String,
    pub protein_count: usize,
    pub binding_sites: Vec<BindingSite>,
    pub conformational_states: Vec<ConformationalState>,
}

/// Binding Site
/// Represents a binding site for ions or molecules
#[derive(Debug, Clone)]
pub struct BindingSite {
    pub site_id: String,
    pub binding_affinity: f64,
    pub selectivity: f64,
    pub occupied: bool,
}

/// ATP Synthase
/// Represents ATP synthase for energy management
#[derive(Debug, Clone)]
pub struct AtpSynthase {
    pub available_atp: f64,
    pub synthesis_rate: f64,
    pub efficiency: f64,
}

impl AtpSynthase {
    pub fn new() -> Self {
        Self {
            available_atp: 5.0e-3, // 5 mM typical concentration
            synthesis_rate: 100.0, // ATP per second
            efficiency: 0.38,      // 38% efficiency
        }
    }

    pub fn get_available_atp(&self) -> f64 {
        self.available_atp
    }

    pub fn consume_atp(&mut self, amount: f64) -> Result<(), crate::errors::KambuzumaError> {
        if self.available_atp >= amount {
            self.available_atp -= amount;
            Ok(())
        } else {
            Err(crate::errors::KambuzumaError::InsufficientAtp {
                required: amount,
                available: self.available_atp,
            })
        }
    }
}

/// Ion Pump
/// Represents ion pumps in cellular membranes
#[derive(Debug, Clone)]
pub struct IonPump {
    pub pump_id: String,
    pub ion_type: IonType,
    pub pump_rate: f64,
    pub energy_per_ion: f64,
    pub is_active: bool,
}

/// Selectivity Filter
/// Implements ion selectivity filtering
#[derive(Debug, Clone)]
pub struct SelectivityFilter {
    pub filter_type: FilterType,
    pub selectivity_threshold: f64,
    pub binding_sites: Vec<BindingSite>,
}

impl SelectivityFilter {
    pub fn new() -> Self {
        Self {
            filter_type: FilterType::ChargeSelective,
            selectivity_threshold: 0.9,
            binding_sites: Vec::new(),
        }
    }

    pub fn filter_ions(&self, ion_mixture: &IonMixture) -> Result<IonMixture, crate::errors::KambuzumaError> {
        let filtered_ions: Vec<Ion> = ion_mixture.ions.iter().filter(|ion| self.passes_filter(ion)).cloned().collect();

        Ok(IonMixture {
            ions: filtered_ions,
            concentration: ion_mixture.concentration * 0.8, // some loss in filtering
            temperature: ion_mixture.temperature,
            ph: ion_mixture.ph,
        })
    }

    fn passes_filter(&self, ion: &Ion) -> bool {
        match self.filter_type {
            FilterType::ChargeSelective => ion.charge.abs() >= 1,
            FilterType::SizeSelective => ion.radius < 1.0e-10, // 1 Angstrom
            FilterType::AffinitySelective => ion.binding_affinity > self.selectivity_threshold,
        }
    }
}

/// Filter Type
/// Types of ion selectivity filters
#[derive(Debug, Clone)]
pub enum FilterType {
    ChargeSelective,
    SizeSelective,
    AffinitySelective,
}

/// Entropy Monitor
/// Monitors entropy changes in thermodynamic operations
#[derive(Debug, Clone)]
pub struct EntropyMonitor {
    pub baseline_entropy: f64,
    pub current_entropy: f64,
    pub entropy_threshold: f64,
}

impl EntropyMonitor {
    pub fn new() -> Self {
        Self {
            baseline_entropy: 0.0,
            current_entropy: 0.0,
            entropy_threshold: 1e-23, // Small entropy threshold
        }
    }

    pub fn calculate_entropy_change(
        &mut self,
        operation: &MaxwellDemonOperation,
    ) -> Result<f64, crate::errors::KambuzumaError> {
        // Calculate entropy change based on operation
        let entropy_change = match operation {
            MaxwellDemonOperation::IonSeparation { .. } => -1.38e-23, // Negative entropy change
            MaxwellDemonOperation::InformationProcessing { .. } => 2.77e-23, // Positive entropy change
            MaxwellDemonOperation::EnergyHarvest { .. } => 1.38e-23,  // Small positive change
        };

        self.current_entropy = self.baseline_entropy + entropy_change;
        Ok(entropy_change)
    }
}

/// Energy Conservation
/// Validates energy conservation in operations
#[derive(Debug, Clone)]
pub struct EnergyConservation {
    pub total_energy_in: f64,
    pub total_energy_out: f64,
    pub conservation_threshold: f64,
}

impl EnergyConservation {
    pub fn new() -> Self {
        Self {
            total_energy_in: 0.0,
            total_energy_out: 0.0,
            conservation_threshold: 1e-21, // attojoule precision
        }
    }

    pub fn validate_conservation(
        &mut self,
        operation: &MaxwellDemonOperation,
    ) -> Result<bool, crate::errors::KambuzumaError> {
        let (energy_in, energy_out) = match operation {
            MaxwellDemonOperation::IonSeparation {
                energy_input,
                energy_output,
            } => (*energy_input, *energy_output),
            MaxwellDemonOperation::InformationProcessing {
                energy_input,
                energy_output,
            } => (*energy_input, *energy_output),
            MaxwellDemonOperation::EnergyHarvest {
                energy_input,
                energy_output,
            } => (*energy_input, *energy_output),
        };

        let energy_difference = (energy_in - energy_out).abs();
        Ok(energy_difference < self.conservation_threshold)
    }
}

/// Second Law Enforcer
/// Enforces the second law of thermodynamics
#[derive(Debug, Clone)]
pub struct SecondLawEnforcer {
    pub entropy_increase_threshold: f64,
}

impl SecondLawEnforcer {
    pub fn new() -> Self {
        Self {
            entropy_increase_threshold: 0.0, // Entropy must not decrease
        }
    }

    pub fn validate_entropy_increase(
        &self,
        operation: &MaxwellDemonOperation,
    ) -> Result<bool, crate::errors::KambuzumaError> {
        // Calculate entropy change for the operation
        let entropy_change = match operation {
            MaxwellDemonOperation::IonSeparation { .. } => -1.38e-23, // Requires energy input
            MaxwellDemonOperation::InformationProcessing { .. } => 2.77e-23, // Generates entropy
            MaxwellDemonOperation::EnergyHarvest { .. } => 1.38e-23,  // Positive entropy change
        };

        // Allow negative entropy change if energy is properly accounted for
        Ok(entropy_change >= self.entropy_increase_threshold)
    }
}

/// Maxwell Demon Operation
/// Represents different types of Maxwell demon operations
#[derive(Debug, Clone)]
pub enum MaxwellDemonOperation {
    IonSeparation {
        energy_input: f64,
        energy_output: f64,
        ions_processed: usize,
    },
    InformationProcessing {
        energy_input: f64,
        energy_output: f64,
        information_bits: f64,
    },
    EnergyHarvest {
        energy_input: f64,
        energy_output: f64,
        efficiency: f64,
    },
}

/// Thermodynamic Validation
/// Results from thermodynamic validation
#[derive(Debug, Clone)]
pub struct ThermodynamicValidation {
    pub energy_conservation: bool,
    pub second_law_compliance: bool,
    pub entropy_change: f64,
    pub is_valid: bool,
}
