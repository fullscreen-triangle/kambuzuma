use crate::quantum::membrane::tunneling::{MembraneQuantumTunneling, TunnelingParameters, QuantumState};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

/// Ion types that can be processed by the Maxwell demon
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IonType {
    /// Sodium ion (Na⁺)
    Sodium,
    /// Potassium ion (K⁺)
    Potassium,
    /// Calcium ion (Ca²⁺)
    Calcium,
    /// Magnesium ion (Mg²⁺)
    Magnesium,
    /// Proton (H⁺)
    Proton,
    /// Chloride ion (Cl⁻)
    Chloride,
}

impl IonType {
    /// Get the charge of the ion
    pub fn charge(&self) -> i32 {
        match self {
            IonType::Sodium => 1,
            IonType::Potassium => 1,
            IonType::Calcium => 2,
            IonType::Magnesium => 2,
            IonType::Proton => 1,
            IonType::Chloride => -1,
        }
    }
    
    /// Get the mass of the ion in atomic mass units
    pub fn mass_amu(&self) -> f64 {
        match self {
            IonType::Sodium => 22.99,
            IonType::Potassium => 39.10,
            IonType::Calcium => 40.08,
            IonType::Magnesium => 24.31,
            IonType::Proton => 1.008,
            IonType::Chloride => 35.45,
        }
    }
    
    /// Get the hydrated radius in Angstroms
    pub fn hydrated_radius(&self) -> f64 {
        match self {
            IonType::Sodium => 3.58,
            IonType::Potassium => 3.31,
            IonType::Calcium => 4.12,
            IonType::Magnesium => 4.28,
            IonType::Proton => 2.8,
            IonType::Chloride => 3.32,
        }
    }
}

/// Information state detected by the Maxwell demon
#[derive(Debug, Clone)]
pub struct InformationState {
    /// Ion type being detected
    pub ion_type: IonType,
    
    /// Detection confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Energy state of the ion
    pub energy: f64,
    
    /// Position relative to membrane
    pub position: f64,
    
    /// Velocity vector
    pub velocity: f64,
    
    /// Timestamp of detection
    pub timestamp: Instant,
    
    /// Quantum coherence measure
    pub coherence: f64,
}

/// Protein conformational state for molecular machinery
#[derive(Debug, Clone, PartialEq)]
pub enum ConformationalState {
    /// Open state - allows ion passage
    Open,
    /// Closed state - blocks ion passage
    Closed,
    /// Gating state - selective passage
    Gating { selectivity: IonType },
    /// Inactivated state - temporarily non-functional
    Inactivated,
}

/// Molecular machinery implementing the Maxwell demon
#[derive(Debug)]
pub struct MolecularMaxwellDemon {
    /// Current conformational state
    conformation: Arc<RwLock<ConformationalState>>,
    
    /// Information detection system
    detector: Arc<RwLock<InformationDetector>>,
    
    /// Gate control mechanism
    gate_controller: Arc<RwLock<GateController>>,
    
    /// Ion selectivity filter
    selectivity_filter: Arc<RwLock<SelectivityFilter>>,
    
    /// Quantum tunneling interface
    tunneling_system: Arc<RwLock<MembraneQuantumTunneling>>,
    
    /// ATP energy constraint tracker
    atp_tracker: Arc<RwLock<ATPTracker>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<DemonMetrics>>,
}

/// Information detection system
#[derive(Debug)]
pub struct InformationDetector {
    /// Detection sensitivity threshold
    sensitivity_threshold: f64,
    
    /// Recent detections buffer
    detection_buffer: Vec<InformationState>,
    
    /// Detection statistics
    detection_stats: HashMap<IonType, u64>,
    
    /// Quantum coherence detector
    coherence_detector: CoherenceDetector,
}

/// Gate control mechanism
#[derive(Debug)]
pub struct GateController {
    /// Current gate position (0.0 = fully closed, 1.0 = fully open)
    gate_position: f64,
    
    /// Gate response time (seconds)
    response_time: f64,
    
    /// Energy cost per gate operation (in ATP molecules)
    energy_cost_per_operation: f64,
    
    /// Recent gate operations
    operation_history: Vec<GateOperation>,
}

/// Selectivity filter for ion discrimination
#[derive(Debug)]
pub struct SelectivityFilter {
    /// Ion selectivity ratios
    selectivity_ratios: HashMap<IonType, f64>,
    
    /// Binding affinity constants
    binding_affinities: HashMap<IonType, f64>,
    
    /// Current filter state
    filter_state: FilterState,
}

/// ATP tracking system for energy constraints
#[derive(Debug)]
pub struct ATPTracker {
    /// Current ATP concentration (mM)
    atp_concentration: f64,
    
    /// ATP consumption rate (molecules/second)
    consumption_rate: f64,
    
    /// ATP synthesis rate (molecules/second)
    synthesis_rate: f64,
    
    /// Total ATP consumed
    total_consumed: u64,
    
    /// Energy efficiency ratio
    efficiency: f64,
}

/// Performance metrics for the Maxwell demon
#[derive(Debug, Default)]
pub struct DemonMetrics {
    /// Total ions processed
    pub ions_processed: u64,
    
    /// Successful sorting operations
    pub successful_sorts: u64,
    
    /// Information bits processed
    pub information_bits: f64,
    
    /// Energy efficiency (information/ATP)
    pub energy_efficiency: f64,
    
    /// Thermodynamic amplification factor
    pub amplification_factor: f64,
    
    /// Operating time
    pub operating_time: Duration,
    
    /// Error rate
    pub error_rate: f64,
}

/// Gate operation record
#[derive(Debug, Clone)]
pub struct GateOperation {
    /// Operation type
    pub operation_type: GateOperationType,
    
    /// Timestamp
    pub timestamp: Instant,
    
    /// Energy cost
    pub energy_cost: f64,
    
    /// Ion type involved
    pub ion_type: Option<IonType>,
    
    /// Success status
    pub success: bool,
}

#[derive(Debug, Clone)]
pub enum GateOperationType {
    Open,
    Close,
    SelectiveGating { target_ion: IonType },
    Inactivation,
}

/// Filter state enumeration
#[derive(Debug, Clone)]
pub enum FilterState {
    Active { target_ion: IonType },
    Inactive,
    Regenerating,
}

/// Quantum coherence detection system
#[derive(Debug)]
pub struct CoherenceDetector {
    /// Coherence measurement sensitivity
    sensitivity: f64,
    
    /// Measurement frequency (Hz)
    frequency: f64,
    
    /// Coherence history buffer
    coherence_history: Vec<(Instant, f64)>,
}

impl MolecularMaxwellDemon {
    pub fn new(tunneling_params: TunnelingParameters) -> Self {
        let tunneling_system = MembraneQuantumTunneling::new(tunneling_params);
        
        Self {
            conformation: Arc::new(RwLock::new(ConformationalState::Closed)),
            detector: Arc::new(RwLock::new(InformationDetector::new())),
            gate_controller: Arc::new(RwLock::new(GateController::new())),
            selectivity_filter: Arc::new(RwLock::new(SelectivityFilter::new())),
            tunneling_system: Arc::new(RwLock::new(tunneling_system)),
            atp_tracker: Arc::new(RwLock::new(ATPTracker::new())),
            metrics: Arc::new(RwLock::new(DemonMetrics::default())),
        }
    }
    
    /// Detect approaching ion and its information state
    pub async fn detect_ion_information(&self, ion_type: IonType, energy: f64, position: f64, velocity: f64) -> Result<InformationState, MaxwellDemonError> {
        let mut detector = self.detector.write().await;
        
        // Calculate quantum coherence
        let coherence = detector.coherence_detector.measure_coherence(energy, position).await?;
        
        // Calculate detection confidence based on ion properties and quantum state
        let confidence = self.calculate_detection_confidence(ion_type, energy, position, coherence).await?;
        
        let info_state = InformationState {
            ion_type,
            confidence,
            energy,
            position,
            velocity,
            timestamp: Instant::now(),
            coherence,
        };
        
        // Store in detection buffer
        detector.detection_buffer.push(info_state.clone());
        *detector.detection_stats.entry(ion_type).or_insert(0) += 1;
        
        // Limit buffer size
        if detector.detection_buffer.len() > 1000 {
            detector.detection_buffer.remove(0);
        }
        
        Ok(info_state)
    }
    
    /// Make sorting decision based on detected information
    pub async fn make_sorting_decision(&self, info_state: &InformationState) -> Result<SortingDecision, MaxwellDemonError> {
        let atp_tracker = self.atp_tracker.read().await;
        
        // Check ATP availability
        if atp_tracker.atp_concentration < 0.1 {
            return Ok(SortingDecision::Reject { reason: "Insufficient ATP".to_string() });
        }
        
        // Check if ion meets sorting criteria
        let selectivity = self.selectivity_filter.read().await;
        let selectivity_score = selectivity.calculate_selectivity_score(info_state).await?;
        
        if selectivity_score > 0.7 && info_state.confidence > 0.8 {
            Ok(SortingDecision::Accept { 
                target_side: if info_state.energy > 0.03 { Side::High } else { Side::Low },
                gate_operation: GateOperationType::SelectiveGating { target_ion: info_state.ion_type },
            })
        } else {
            Ok(SortingDecision::Reject { 
                reason: format!("Low selectivity ({:.3}) or confidence ({:.3})", selectivity_score, info_state.confidence) 
            })
        }
    }
    
    /// Execute gate operation to sort ion
    pub async fn execute_gate_operation(&self, decision: SortingDecision, info_state: &InformationState) -> Result<(), MaxwellDemonError> {
        match decision {
            SortingDecision::Accept { target_side, gate_operation } => {
                let mut gate_controller = self.gate_controller.write().await;
                let mut atp_tracker = self.atp_tracker.write().await;
                
                // Calculate energy cost
                let energy_cost = gate_controller.energy_cost_per_operation;
                
                // Check ATP availability
                if atp_tracker.atp_concentration * 1000.0 < energy_cost {
                    return Err(MaxwellDemonError::InsufficientEnergy);
                }
                
                // Execute gate operation
                let operation = GateOperation {
                    operation_type: gate_operation,
                    timestamp: Instant::now(),
                    energy_cost,
                    ion_type: Some(info_state.ion_type),
                    success: true,
                };
                
                // Update gate position based on operation
                match target_side {
                    Side::High => gate_controller.gate_position = 1.0,
                    Side::Low => gate_controller.gate_position = 0.0,
                }
                
                // Consume ATP
                atp_tracker.atp_concentration -= energy_cost / 1000.0;
                atp_tracker.total_consumed += energy_cost as u64;
                
                // Record operation
                gate_controller.operation_history.push(operation);
                
                // Update metrics
                let mut metrics = self.metrics.write().await;
                metrics.ions_processed += 1;
                metrics.successful_sorts += 1;
                metrics.information_bits += info_state.confidence.log2();
                
                Ok(())
            },
            SortingDecision::Reject { reason: _ } => {
                // Update metrics for rejected ion
                let mut metrics = self.metrics.write().await;
                metrics.ions_processed += 1;
                Ok(())
            }
        }
    }
    
    /// Calculate thermodynamic amplification factor
    pub async fn calculate_amplification_factor(&self) -> f64 {
        let metrics = self.metrics.read().await;
        let atp_tracker = self.atp_tracker.read().await;
        
        if atp_tracker.total_consumed == 0 {
            return 1.0;
        }
        
        // Amplification = Information gained / Energy invested
        let information_gained = metrics.information_bits;
        let energy_invested = atp_tracker.total_consumed as f64 * 30.5; // kJ/mol ATP
        
        if energy_invested > 0.0 {
            information_gained / energy_invested * 1000.0 // Scale for typical biological values
        } else {
            1.0
        }
    }
    
    /// Update demon state and perform maintenance
    pub async fn update_state(&self, time_step: f64) -> Result<(), MaxwellDemonError> {
        // Update tunneling system
        self.tunneling_system.write().await.update_quantum_state(time_step).await
            .map_err(|e| MaxwellDemonError::QuantumStateError(e.to_string()))?;
        
        // Regenerate ATP
        let mut atp_tracker = self.atp_tracker.write().await;
        let atp_synthesis = atp_tracker.synthesis_rate * time_step;
        atp_tracker.atp_concentration += atp_synthesis / 1000.0;
        
        // Cap ATP concentration at physiological maximum
        if atp_tracker.atp_concentration > 5.0 {
            atp_tracker.atp_concentration = 5.0;
        }
        
        // Update efficiency
        if atp_tracker.total_consumed > 0 {
            let metrics = self.metrics.read().await;
            atp_tracker.efficiency = metrics.information_bits / atp_tracker.total_consumed as f64;
        }
        
        Ok(())
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> DemonMetrics {
        let mut metrics = self.metrics.read().await.clone();
        metrics.amplification_factor = self.calculate_amplification_factor().await;
        
        if metrics.ions_processed > 0 {
            metrics.error_rate = (metrics.ions_processed - metrics.successful_sorts) as f64 / metrics.ions_processed as f64;
        }
        
        metrics
    }
    
    /// Validate biological constraints
    pub async fn validate_biological_constraints(&self) -> Result<(), MaxwellDemonError> {
        let atp_tracker = self.atp_tracker.read().await;
        
        // Check ATP concentration is within biological range (0.5-10 mM)
        if atp_tracker.atp_concentration < 0.5 || atp_tracker.atp_concentration > 10.0 {
            return Err(MaxwellDemonError::BiologicalConstraintViolation(
                format!("ATP concentration ({:.2} mM) outside biological range", atp_tracker.atp_concentration)
            ));
        }
        
        // Validate tunneling constraints
        self.tunneling_system.read().await.validate_biological_constraints().await
            .map_err(|e| MaxwellDemonError::BiologicalConstraintViolation(e.to_string()))?;
        
        Ok(())
    }
    
    async fn calculate_detection_confidence(&self, ion_type: IonType, energy: f64, position: f64, coherence: f64) -> Result<f64, MaxwellDemonError> {
        // Base confidence from quantum coherence
        let mut confidence = coherence;
        
        // Adjust based on ion type detectability
        let detectability = match ion_type {
            IonType::Sodium | IonType::Potassium => 0.9,  // Easy to detect
            IonType::Calcium | IonType::Magnesium => 0.8, // Moderate
            IonType::Proton => 0.95,                       // Very easy
            IonType::Chloride => 0.7,                      // Harder due to size
        };
        
        confidence *= detectability;
        
        // Adjust based on energy (higher energy = easier detection)
        let energy_factor = (energy / 0.025).min(2.0); // Normalize to thermal energy
        confidence *= energy_factor;
        
        // Add position-dependent detection efficiency
        let position_factor = (-position.abs() / 1e-9).exp(); // Decay with distance from membrane
        confidence *= position_factor;
        
        Ok(confidence.min(1.0))
    }
}

/// Sorting decision made by the Maxwell demon
#[derive(Debug, Clone)]
pub enum SortingDecision {
    Accept { 
        target_side: Side,
        gate_operation: GateOperationType,
    },
    Reject { 
        reason: String,
    },
}

/// Membrane sides for ion sorting
#[derive(Debug, Clone, Copy)]
pub enum Side {
    High,  // High energy side
    Low,   // Low energy side
}

/// Errors that can occur in Maxwell demon operations
#[derive(Debug, thiserror::Error)]
pub enum MaxwellDemonError {
    #[error("Insufficient energy for operation")]
    InsufficientEnergy,
    
    #[error("Information detection error: {0}")]
    DetectionError(String),
    
    #[error("Gate operation error: {0}")]
    GateOperationError(String),
    
    #[error("Biological constraint violation: {0}")]
    BiologicalConstraintViolation(String),
    
    #[error("Quantum state error: {0}")]
    QuantumStateError(String),
    
    #[error("Selectivity calculation error: {0}")]
    SelectivityError(String),
}

// Implementation details for helper structs...
impl InformationDetector {
    pub fn new() -> Self {
        Self {
            sensitivity_threshold: 0.1,
            detection_buffer: Vec::new(),
            detection_stats: HashMap::new(),
            coherence_detector: CoherenceDetector::new(),
        }
    }
}

impl GateController {
    pub fn new() -> Self {
        Self {
            gate_position: 0.0,
            response_time: 1e-6, // 1 microsecond
            energy_cost_per_operation: 10.0, // ATP molecules
            operation_history: Vec::new(),
        }
    }
}

impl SelectivityFilter {
    pub fn new() -> Self {
        let mut selectivity_ratios = HashMap::new();
        selectivity_ratios.insert(IonType::Sodium, 1.0);
        selectivity_ratios.insert(IonType::Potassium, 0.8);
        selectivity_ratios.insert(IonType::Calcium, 0.6);
        selectivity_ratios.insert(IonType::Magnesium, 0.6);
        selectivity_ratios.insert(IonType::Proton, 0.9);
        selectivity_ratios.insert(IonType::Chloride, 0.3);
        
        let mut binding_affinities = HashMap::new();
        binding_affinities.insert(IonType::Sodium, 10.0);   // mM
        binding_affinities.insert(IonType::Potassium, 100.0);
        binding_affinities.insert(IonType::Calcium, 1.0);
        binding_affinities.insert(IonType::Magnesium, 0.5);
        binding_affinities.insert(IonType::Proton, 0.1);
        binding_affinities.insert(IonType::Chloride, 50.0);
        
        Self {
            selectivity_ratios,
            binding_affinities,
            filter_state: FilterState::Active { target_ion: IonType::Sodium },
        }
    }
    
    pub async fn calculate_selectivity_score(&self, info_state: &InformationState) -> Result<f64, MaxwellDemonError> {
        let selectivity = self.selectivity_ratios.get(&info_state.ion_type)
            .copied()
            .unwrap_or(0.0);
        
        let binding_affinity = self.binding_affinities.get(&info_state.ion_type)
            .copied()
            .unwrap_or(0.0);
        
        // Combine selectivity and binding affinity
        let score = selectivity * (1.0 / (1.0 + binding_affinity / 10.0));
        
        Ok(score)
    }
}

impl ATPTracker {
    pub fn new() -> Self {
        Self {
            atp_concentration: 3.0, // Typical cellular concentration (mM)
            consumption_rate: 1000.0, // molecules/second
            synthesis_rate: 1200.0,   // molecules/second (slightly higher for net production)
            total_consumed: 0,
            efficiency: 0.0,
        }
    }
}

impl CoherenceDetector {
    pub fn new() -> Self {
        Self {
            sensitivity: 0.01,
            frequency: 1e6, // 1 MHz
            coherence_history: Vec::new(),
        }
    }
    
    pub async fn measure_coherence(&mut self, energy: f64, position: f64) -> Result<f64, MaxwellDemonError> {
        // Simple coherence model based on energy and position
        let thermal_coherence = (-energy / (0.025 * 10.0)).exp(); // Exponential decay with temperature
        let spatial_coherence = (-position.abs() / 1e-10).exp();   // Spatial coherence decay
        
        let coherence = thermal_coherence * spatial_coherence;
        
        self.coherence_history.push((Instant::now(), coherence));
        
        // Keep only recent measurements
        if self.coherence_history.len() > 1000 {
            self.coherence_history.remove(0);
        }
        
        Ok(coherence.min(1.0))
    }
} 