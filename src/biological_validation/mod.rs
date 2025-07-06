//! Biological validation subsystem for Kambuzuma
//! 
//! Ensures all quantum processes adhere to biological constraints and physical laws

pub mod constraint_validation;
pub mod thermodynamic_validation;
pub mod quantum_coherence_validation;
pub mod membrane_integrity_validation;
pub mod energy_conservation_validation;

use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;
use crate::Result;
use std::collections::HashMap;

/// Biological validation subsystem configuration
#[derive(Debug, Clone)]
pub struct BiologicalValidationConfig {
    /// Constraint validation configuration
    pub constraint_config: constraint_validation::ConstraintConfig,
    
    /// Thermodynamic validation configuration
    pub thermodynamic_config: thermodynamic_validation::ThermodynamicConfig,
    
    /// Quantum coherence validation configuration
    pub coherence_config: quantum_coherence_validation::CoherenceConfig,
    
    /// Membrane integrity validation configuration
    pub membrane_config: membrane_integrity_validation::MembraneConfig,
    
    /// Energy conservation validation configuration
    pub energy_config: energy_conservation_validation::EnergyConfig,
}

/// Biological validation subsystem state
#[derive(Debug, Clone)]
pub struct BiologicalValidationState {
    /// Overall validation status
    pub validation_status: ValidationStatus,
    
    /// Constraint validation results
    pub constraint_results: HashMap<ConstraintType, ValidationResult>,
    
    /// Thermodynamic validation results
    pub thermodynamic_results: ThermodynamicValidationResults,
    
    /// Quantum coherence validation results
    pub coherence_results: CoherenceValidationResults,
    
    /// Membrane integrity validation results
    pub membrane_results: MembraneValidationResults,
    
    /// Energy conservation validation results
    pub energy_results: EnergyValidationResults,
    
    /// Validation accuracy
    pub validation_accuracy: f64,
}

/// Overall validation status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidationStatus {
    /// All constraints satisfied
    Valid,
    /// Some constraints violated but system can continue
    Warning,
    /// Critical constraints violated, system must stop
    Critical,
    /// Unknown state
    Unknown,
}

/// Types of biological constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstraintType {
    /// Temperature constraints for biological processes
    Temperature,
    /// pH constraints for biochemical reactions
    pH,
    /// Ion concentration constraints
    IonicStrength,
    /// ATP energy constraints
    EnergyAvailability,
    /// Membrane potential constraints
    MembranePotential,
    /// Quantum decoherence time constraints
    CoherenceTime,
    /// Osmotic pressure constraints
    OsmoticPressure,
    /// Enzyme kinetics constraints
    EnzymeKinetics,
}

/// Validation result for a specific constraint
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Constraint type
    pub constraint_type: ConstraintType,
    
    /// Whether constraint is satisfied
    pub satisfied: bool,
    
    /// Current value
    pub current_value: f64,
    
    /// Allowed range
    pub allowed_range: (f64, f64),
    
    /// Severity if violated
    pub severity: ViolationSeverity,
    
    /// Explanation
    pub explanation: String,
}

/// Severity of constraint violation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViolationSeverity {
    /// Minor violation, can continue
    Minor,
    /// Moderate violation, warning issued
    Moderate,
    /// Major violation, system should adjust
    Major,
    /// Critical violation, system must stop
    Critical,
}

/// Thermodynamic validation results
#[derive(Debug, Clone)]
pub struct ThermodynamicValidationResults {
    /// Entropy change validation
    pub entropy_change: f64,
    
    /// Free energy change validation
    pub free_energy_change: f64,
    
    /// Temperature consistency
    pub temperature_consistency: bool,
    
    /// Heat capacity constraints
    pub heat_capacity_satisfied: bool,
    
    /// Thermodynamic equilibrium status
    pub equilibrium_status: EquilibriumStatus,
}

/// Equilibrium status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EquilibriumStatus {
    /// System is in equilibrium
    Equilibrium,
    /// System is approaching equilibrium
    Approaching,
    /// System is far from equilibrium
    NonEquilibrium,
}

/// Quantum coherence validation results
#[derive(Debug, Clone)]
pub struct CoherenceValidationResults {
    /// Coherence time measurements
    pub coherence_times: Vec<f64>,
    
    /// Decoherence rate
    pub decoherence_rate: f64,
    
    /// Quantum state fidelity
    pub quantum_fidelity: f64,
    
    /// Entanglement preservation
    pub entanglement_preserved: bool,
    
    /// Coherence threshold satisfied
    pub coherence_threshold_satisfied: bool,
}

/// Membrane integrity validation results
#[derive(Debug, Clone)]
pub struct MembraneValidationResults {
    /// Membrane thickness measurements
    pub membrane_thickness: f64,
    
    /// Membrane potential
    pub membrane_potential: f64,
    
    /// Lipid bilayer stability
    pub lipid_stability: f64,
    
    /// Protein functionality
    pub protein_functionality: f64,
    
    /// Ion channel activity
    pub ion_channel_activity: f64,
    
    /// Membrane permeability
    pub membrane_permeability: f64,
}

/// Energy conservation validation results
#[derive(Debug, Clone)]
pub struct EnergyValidationResults {
    /// Total energy input
    pub total_energy_input: f64,
    
    /// Total energy output
    pub total_energy_output: f64,
    
    /// Energy conservation ratio
    pub conservation_ratio: f64,
    
    /// ATP production/consumption balance
    pub atp_balance: f64,
    
    /// Metabolic efficiency
    pub metabolic_efficiency: f64,
    
    /// Energy waste heat
    pub waste_heat: f64,
}

/// Biological validation subsystem
pub struct BiologicalValidationSubsystem {
    /// Configuration
    config: BiologicalValidationConfig,
    
    /// Constraint validation system
    constraint_validator: Arc<RwLock<constraint_validation::ConstraintValidator>>,
    
    /// Thermodynamic validation system
    thermodynamic_validator: Arc<RwLock<thermodynamic_validation::ThermodynamicValidator>>,
    
    /// Quantum coherence validation system
    coherence_validator: Arc<RwLock<quantum_coherence_validation::CoherenceValidator>>,
    
    /// Membrane integrity validation system
    membrane_validator: Arc<RwLock<membrane_integrity_validation::MembraneValidator>>,
    
    /// Energy conservation validation system
    energy_validator: Arc<RwLock<energy_conservation_validation::EnergyValidator>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<BiologicalValidationMetrics>>,
}

/// Biological validation performance metrics
#[derive(Debug, Default)]
pub struct BiologicalValidationMetrics {
    /// Total validations performed
    pub total_validations: u64,
    
    /// Successful validations
    pub successful_validations: u64,
    
    /// Average validation time
    pub average_validation_time: std::time::Duration,
    
    /// Constraint violation rate
    pub constraint_violation_rate: f64,
    
    /// False positive rate
    pub false_positive_rate: f64,
    
    /// False negative rate
    pub false_negative_rate: f64,
}

impl BiologicalValidationSubsystem {
    /// Create new biological validation subsystem
    pub fn new(config: &BiologicalValidationConfig) -> Result<Self> {
        // Initialize constraint validator
        let constraint_validator = Arc::new(RwLock::new(
            constraint_validation::ConstraintValidator::new(&config.constraint_config)?
        ));
        
        // Initialize thermodynamic validator
        let thermodynamic_validator = Arc::new(RwLock::new(
            thermodynamic_validation::ThermodynamicValidator::new(&config.thermodynamic_config)?
        ));
        
        // Initialize coherence validator
        let coherence_validator = Arc::new(RwLock::new(
            quantum_coherence_validation::CoherenceValidator::new(&config.coherence_config)?
        ));
        
        // Initialize membrane validator
        let membrane_validator = Arc::new(RwLock::new(
            membrane_integrity_validation::MembraneValidator::new(&config.membrane_config)?
        ));
        
        // Initialize energy validator
        let energy_validator = Arc::new(RwLock::new(
            energy_conservation_validation::EnergyValidator::new(&config.energy_config)?
        ));
        
        // Initialize metrics
        let metrics = Arc::new(RwLock::new(BiologicalValidationMetrics::default()));
        
        Ok(Self {
            config: config.clone(),
            constraint_validator,
            thermodynamic_validator,
            coherence_validator,
            membrane_validator,
            energy_validator,
            metrics,
        })
    }
    
    /// Start the biological validation subsystem
    pub async fn start(&self) -> Result<()> {
        self.constraint_validator.write().await.start().await?;
        self.thermodynamic_validator.write().await.start().await?;
        self.coherence_validator.write().await.start().await?;
        self.membrane_validator.write().await.start().await?;
        self.energy_validator.write().await.start().await?;
        
        Ok(())
    }
    
    /// Stop the biological validation subsystem
    pub async fn stop(&self) -> Result<()> {
        self.energy_validator.write().await.stop().await?;
        self.membrane_validator.write().await.stop().await?;
        self.coherence_validator.write().await.stop().await?;
        self.thermodynamic_validator.write().await.stop().await?;
        self.constraint_validator.write().await.stop().await?;
        
        Ok(())
    }
    
    /// Validate system state against biological constraints
    pub async fn validate_system(&self, system_state: &SystemState) -> Result<BiologicalValidationState> {
        let start_time = std::time::Instant::now();
        
        // Validate constraints
        let constraint_results = self.constraint_validator.read().await
            .validate_constraints(system_state).await?;
        
        // Validate thermodynamics
        let thermodynamic_results = self.thermodynamic_validator.read().await
            .validate_thermodynamics(system_state).await?;
        
        // Validate quantum coherence
        let coherence_results = self.coherence_validator.read().await
            .validate_coherence(system_state).await?;
        
        // Validate membrane integrity
        let membrane_results = self.membrane_validator.read().await
            .validate_membrane(system_state).await?;
        
        // Validate energy conservation
        let energy_results = self.energy_validator.read().await
            .validate_energy(system_state).await?;
        
        // Determine overall validation status
        let validation_status = self.determine_overall_status(
            &constraint_results,
            &thermodynamic_results,
            &coherence_results,
            &membrane_results,
            &energy_results,
        ).await?;
        
        // Calculate validation accuracy
        let validation_accuracy = self.calculate_validation_accuracy(
            &constraint_results,
            &thermodynamic_results,
            &coherence_results,
            &membrane_results,
            &energy_results,
        ).await?;
        
        let validation_time = start_time.elapsed();
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_validations += 1;
        if validation_status == ValidationStatus::Valid {
            metrics.successful_validations += 1;
        }
        metrics.average_validation_time = (metrics.average_validation_time + validation_time) / 2;
        
        Ok(BiologicalValidationState {
            validation_status,
            constraint_results,
            thermodynamic_results,
            coherence_results,
            membrane_results,
            energy_results,
            validation_accuracy,
        })
    }
    
    /// Get current validation state
    pub async fn get_state(&self) -> Result<BiologicalValidationState> {
        // Create a default system state for validation
        let system_state = SystemState::default();
        
        // Perform validation
        self.validate_system(&system_state).await
    }
    
    /// Check if a specific constraint is satisfied
    pub async fn check_constraint(&self, constraint_type: ConstraintType, value: f64) -> Result<ValidationResult> {
        self.constraint_validator.read().await.check_constraint(constraint_type, value).await
    }
    
    /// Get validation metrics
    pub async fn get_metrics(&self) -> Result<BiologicalValidationMetrics> {
        Ok(self.metrics.read().await.clone())
    }
    
    // Private helper methods
    
    async fn determine_overall_status(
        &self,
        constraint_results: &HashMap<ConstraintType, ValidationResult>,
        thermodynamic_results: &ThermodynamicValidationResults,
        coherence_results: &CoherenceValidationResults,
        membrane_results: &MembraneValidationResults,
        energy_results: &EnergyValidationResults,
    ) -> Result<ValidationStatus> {
        // Check for critical violations
        for result in constraint_results.values() {
            if !result.satisfied && result.severity == ViolationSeverity::Critical {
                return Ok(ValidationStatus::Critical);
            }
        }
        
        // Check thermodynamic constraints
        if !thermodynamic_results.temperature_consistency || 
           !thermodynamic_results.heat_capacity_satisfied {
            return Ok(ValidationStatus::Critical);
        }
        
        // Check quantum coherence
        if !coherence_results.coherence_threshold_satisfied {
            return Ok(ValidationStatus::Warning);
        }
        
        // Check energy conservation
        if energy_results.conservation_ratio < 0.9 {
            return Ok(ValidationStatus::Warning);
        }
        
        // Check for any moderate violations
        for result in constraint_results.values() {
            if !result.satisfied && result.severity == ViolationSeverity::Moderate {
                return Ok(ValidationStatus::Warning);
            }
        }
        
        Ok(ValidationStatus::Valid)
    }
    
    async fn calculate_validation_accuracy(
        &self,
        constraint_results: &HashMap<ConstraintType, ValidationResult>,
        thermodynamic_results: &ThermodynamicValidationResults,
        coherence_results: &CoherenceValidationResults,
        membrane_results: &MembraneValidationResults,
        energy_results: &EnergyValidationResults,
    ) -> Result<f64> {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        // Score constraint validations
        for result in constraint_results.values() {
            let weight = match result.constraint_type {
                ConstraintType::Temperature => 0.2,
                ConstraintType::pH => 0.15,
                ConstraintType::EnergyAvailability => 0.25,
                ConstraintType::MembranePotential => 0.15,
                ConstraintType::CoherenceTime => 0.15,
                _ => 0.1,
            };
            
            total_score += if result.satisfied { weight } else { 0.0 };
            total_weight += weight;
        }
        
        // Score thermodynamic validation
        let thermo_weight = 0.2;
        let thermo_score = if thermodynamic_results.temperature_consistency && 
                             thermodynamic_results.heat_capacity_satisfied {
            thermo_weight
        } else {
            0.0
        };
        total_score += thermo_score;
        total_weight += thermo_weight;
        
        // Score coherence validation
        let coherence_weight = 0.15;
        let coherence_score = coherence_results.quantum_fidelity * coherence_weight;
        total_score += coherence_score;
        total_weight += coherence_weight;
        
        // Score energy conservation
        let energy_weight = 0.2;
        let energy_score = energy_results.conservation_ratio * energy_weight;
        total_score += energy_score;
        total_weight += energy_weight;
        
        Ok(if total_weight > 0.0 { total_score / total_weight } else { 0.0 })
    }
}

/// System state for validation
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Current temperature in Kelvin
    pub temperature: f64,
    
    /// Current pH
    pub ph: f64,
    
    /// Ion concentrations
    pub ion_concentrations: HashMap<String, f64>,
    
    /// ATP concentration
    pub atp_concentration: f64,
    
    /// Membrane potential
    pub membrane_potential: f64,
    
    /// Quantum coherence time
    pub coherence_time: f64,
    
    /// Energy input rate
    pub energy_input_rate: f64,
    
    /// Energy output rate
    pub energy_output_rate: f64,
    
    /// Membrane thickness
    pub membrane_thickness: f64,
    
    /// System entropy
    pub entropy: f64,
}

impl Default for SystemState {
    fn default() -> Self {
        let mut ion_concentrations = HashMap::new();
        ion_concentrations.insert("Na+".to_string(), 0.140); // 140 mM
        ion_concentrations.insert("K+".to_string(), 0.005);  // 5 mM
        ion_concentrations.insert("Ca2+".to_string(), 0.001); // 1 mM
        ion_concentrations.insert("Cl-".to_string(), 0.120);  // 120 mM
        
        Self {
            temperature: 310.15,  // 37°C in Kelvin
            ph: 7.4,             // Physiological pH
            ion_concentrations,
            atp_concentration: 0.005,  // 5 mM ATP
            membrane_potential: -0.070, // -70 mV
            coherence_time: 1e-12,     // 1 ps
            energy_input_rate: 100.0,  // Arbitrary units
            energy_output_rate: 95.0,  // Arbitrary units
            membrane_thickness: 5e-9,  // 5 nm
            entropy: 1000.0,           // Arbitrary units
        }
    }
}

/// Biological validation subsystem errors
#[derive(Debug, Error)]
pub enum BiologicalValidationError {
    #[error("Constraint validation error: {0}")]
    ConstraintValidation(String),
    
    #[error("Thermodynamic validation error: {0}")]
    ThermodynamicValidation(String),
    
    #[error("Coherence validation error: {0}")]
    CoherenceValidation(String),
    
    #[error("Membrane validation error: {0}")]
    MembraneValidation(String),
    
    #[error("Energy validation error: {0}")]
    EnergyValidation(String),
    
    #[error("Critical constraint violation: {0}")]
    CriticalConstraintViolation(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl Default for BiologicalValidationConfig {
    fn default() -> Self {
        Self {
            constraint_config: constraint_validation::ConstraintConfig::default(),
            thermodynamic_config: thermodynamic_validation::ThermodynamicConfig::default(),
            coherence_config: quantum_coherence_validation::CoherenceConfig::default(),
            membrane_config: membrane_integrity_validation::MembraneConfig::default(),
            energy_config: energy_conservation_validation::EnergyConfig::default(),
        }
    }
}

impl BiologicalValidationConfig {
    /// Validate configuration
    pub fn is_valid(&self) -> bool {
        self.constraint_config.is_valid() &&
        self.thermodynamic_config.is_valid() &&
        self.coherence_config.is_valid() &&
        self.membrane_config.is_valid() &&
        self.energy_config.is_valid()
    }
} 