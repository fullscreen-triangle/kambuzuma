pub mod cell_culture_kinetics;
pub mod control_systems;
pub mod nutrient_feeding;
pub mod oxygen_transfer;
pub mod ph_control;
pub mod process_monitoring;
pub mod temperature_control;

use crate::errors::KambuzumaError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::time::{interval, Duration};

/// Bioreactor control system for maintaining living cell cultures
/// Essential for supporting biological quantum computing processes
#[derive(Debug, Clone)]
pub struct BioreactorController {
    /// Bioreactor identification
    pub reactor_id: String,

    /// Current process state
    pub state: Arc<RwLock<BioreactorState>>,

    /// Control parameters
    pub control_params: BioreactorControlParameters,

    /// Process monitoring data
    pub monitoring: Arc<RwLock<ProcessMonitoringData>>,

    /// Control loops
    pub controllers: ControlSystems,
}

/// Complete bioreactor state including all process variables
#[derive(Debug, Clone)]
pub struct BioreactorState {
    // Cell culture parameters
    pub viable_cell_density: f64,  // cells/mL
    pub total_cell_density: f64,   // cells/mL
    pub viability: f64,            // %
    pub specific_growth_rate: f64, // h⁻¹

    // Dissolved oxygen and gas transfer
    pub dissolved_oxygen: f64,   // % saturation
    pub oxygen_uptake_rate: f64, // mmol O₂/L/h
    pub kla: f64,                // h⁻¹ (oxygen transfer coefficient)
    pub aeration_rate: f64,      // vvm (vessel volumes per minute)
    pub agitation_speed: f64,    // RPM

    // pH and CO₂
    pub ph: f64,
    pub co2_percentage: f64,     // %
    pub base_addition_rate: f64, // mL/h
    pub acid_addition_rate: f64, // mL/h

    // Temperature
    pub temperature: f64,   // °C
    pub heating_power: f64, // W
    pub cooling_power: f64, // W

    // Nutrients and metabolites
    pub glucose_concentration: f64,   // g/L
    pub lactate_concentration: f64,   // g/L
    pub ammonia_concentration: f64,   // mM
    pub glutamine_concentration: f64, // mM

    // Physical parameters
    pub working_volume: f64, // L
    pub turbidity: f64,      // OD₆₀₀
    pub foam_level: f64,     // % of headspace

    // Process time
    pub process_time: f64, // hours
}

/// Bioreactor control parameters based on bioprocess engineering principles
#[derive(Debug, Clone)]
pub struct BioreactorControlParameters {
    // Setpoints
    pub target_dissolved_oxygen: f64, // % saturation (typically 30-50%)
    pub target_ph: f64,               // typically 7.0-7.4
    pub target_temperature: f64,      // °C (typically 37°C)
    pub target_glucose: f64,          // g/L

    // Control gains (PID parameters)
    pub do_controller: PIDParameters,
    pub ph_controller: PIDParameters,
    pub temperature_controller: PIDParameters,
    pub feeding_controller: PIDParameters,

    // Process limits
    pub max_cell_density: f64, // cells/mL
    pub min_viability: f64,    // %
    pub max_lactate: f64,      // g/L
    pub max_ammonia: f64,      // mM

    // Feed strategy parameters
    pub glucose_feed_concentration: f64, // g/L
    pub feeding_trigger_glucose: f64,    // g/L
    pub max_feed_rate: f64,              // mL/h
}

/// PID controller parameters for bioprocess control
#[derive(Debug, Clone)]
pub struct PIDParameters {
    pub kp: f64, // Proportional gain
    pub ki: f64, // Integral gain
    pub kd: f64, // Derivative gain
    pub setpoint: f64,
    pub integral_term: f64,
    pub last_error: f64,
}

/// Process monitoring data structure
#[derive(Debug, Clone)]
pub struct ProcessMonitoringData {
    pub current_measurements: HashMap<String, f64>,
    pub historical_data: Vec<ProcessDataPoint>,
    pub alarms: Vec<ProcessAlarm>,
    pub quality_metrics: QualityMetrics,
}

/// Individual process data point with timestamp
#[derive(Debug, Clone)]
pub struct ProcessDataPoint {
    pub timestamp: f64,
    pub measurements: HashMap<String, f64>,
}

/// Process alarm for monitoring critical parameters
#[derive(Debug, Clone)]
pub struct ProcessAlarm {
    pub alarm_id: String,
    pub parameter: String,
    pub alarm_type: AlarmType,
    pub value: f64,
    pub limit: f64,
    pub timestamp: f64,
    pub acknowledged: bool,
}

/// Types of process alarms
#[derive(Debug, Clone)]
pub enum AlarmType {
    HighAlarm,
    LowAlarm,
    HighHighAlarm,
    LowLowAlarm,
    RateOfChange,
    DeviceFailure,
}

/// Quality metrics for cell culture performance
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub specific_productivity: f64,        // pg/cell/day
    pub volumetric_productivity: f64,      // mg/L/day
    pub yield_on_glucose: f64,             // cells/g glucose
    pub yield_on_glutamine: f64,           // cells/mM glutamine
    pub doubling_time: f64,                // hours
    pub integral_viable_cell_density: f64, // cell·h/mL
}

/// Complete control systems for bioreactor
#[derive(Debug, Clone)]
pub struct ControlSystems {
    pub do_controller: PIDController,
    pub ph_controller: PIDController,
    pub temperature_controller: PIDController,
    pub feeding_controller: FeedingController,
    pub safety_systems: SafetySystems,
}

/// PID controller implementation
#[derive(Debug, Clone)]
pub struct PIDController {
    pub params: PIDParameters,
    pub last_update: f64,
    pub output: f64,
    pub enabled: bool,
}

/// Feeding controller for nutrient management
#[derive(Debug, Clone)]
pub struct FeedingController {
    pub glucose_controller: PIDController,
    pub fed_batch_enabled: bool,
    pub perfusion_enabled: bool,
    pub current_feed_rate: f64, // mL/h
    pub total_feed_volume: f64, // mL
}

/// Safety systems for bioreactor operation
#[derive(Debug, Clone)]
pub struct SafetySystems {
    pub emergency_stop: bool,
    pub gas_supply_pressure: f64, // PSI
    pub power_backup_active: bool,
    pub containment_integrity: bool,
    pub sterility_status: SterilityStatus,
}

/// Sterility monitoring status
#[derive(Debug, Clone)]
pub enum SterilityStatus {
    Sterile,
    Contaminated,
    Uncertain,
    SterilityTestPending,
}

impl BioreactorController {
    /// Create new bioreactor controller with standard mammalian cell culture parameters
    pub fn new(reactor_id: String) -> Self {
        let control_params = BioreactorControlParameters {
            target_dissolved_oxygen: 40.0, // 40% saturation
            target_ph: 7.2,                // pH 7.2
            target_temperature: 37.0,      // 37°C
            target_glucose: 2.0,           // 2 g/L glucose

            do_controller: PIDParameters {
                kp: 1.0,
                ki: 0.1,
                kd: 0.05,
                setpoint: 40.0,
                integral_term: 0.0,
                last_error: 0.0,
            },

            ph_controller: PIDParameters {
                kp: 2.0,
                ki: 0.2,
                kd: 0.1,
                setpoint: 7.2,
                integral_term: 0.0,
                last_error: 0.0,
            },

            temperature_controller: PIDParameters {
                kp: 5.0,
                ki: 0.5,
                kd: 0.2,
                setpoint: 37.0,
                integral_term: 0.0,
                last_error: 0.0,
            },

            feeding_controller: PIDParameters {
                kp: 0.5,
                ki: 0.05,
                kd: 0.02,
                setpoint: 2.0,
                integral_term: 0.0,
                last_error: 0.0,
            },

            max_cell_density: 20e6, // 20 million cells/mL
            min_viability: 85.0,    // 85% viability
            max_lactate: 5.0,       // 5 g/L lactate
            max_ammonia: 5.0,       // 5 mM ammonia

            glucose_feed_concentration: 500.0, // 500 g/L feed
            feeding_trigger_glucose: 1.0,      // Feed when glucose < 1 g/L
            max_feed_rate: 50.0,               // 50 mL/h max feed rate
        };

        let initial_state = BioreactorState {
            viable_cell_density: 0.5e6, // 0.5 million cells/mL initial
            total_cell_density: 0.5e6,
            viability: 95.0,
            specific_growth_rate: 0.03, // 0.03 h⁻¹ (23h doubling time)

            dissolved_oxygen: 40.0,
            oxygen_uptake_rate: 0.5, // mmol O₂/L/h
            kla: 10.0,               // h⁻¹
            aeration_rate: 0.1,      // 0.1 vvm
            agitation_speed: 100.0,  // 100 RPM

            ph: 7.2,
            co2_percentage: 5.0, // 5% CO₂
            base_addition_rate: 0.0,
            acid_addition_rate: 0.0,

            temperature: 37.0,
            heating_power: 0.0,
            cooling_power: 0.0,

            glucose_concentration: 4.0,   // 4 g/L initial glucose
            lactate_concentration: 0.1,   // 0.1 g/L initial lactate
            ammonia_concentration: 0.1,   // 0.1 mM initial ammonia
            glutamine_concentration: 4.0, // 4 mM glutamine

            working_volume: 2.0, // 2L working volume
            turbidity: 0.1,      // OD₆₀₀
            foam_level: 0.0,

            process_time: 0.0,
        };

        Self {
            reactor_id,
            state: Arc::new(RwLock::new(initial_state)),
            control_params,
            monitoring: Arc::new(RwLock::new(ProcessMonitoringData {
                current_measurements: HashMap::new(),
                historical_data: Vec::new(),
                alarms: Vec::new(),
                quality_metrics: QualityMetrics {
                    specific_productivity: 0.0,
                    volumetric_productivity: 0.0,
                    yield_on_glucose: 0.0,
                    yield_on_glutamine: 0.0,
                    doubling_time: 23.0,
                    integral_viable_cell_density: 0.0,
                },
            })),
            controllers: ControlSystems {
                do_controller: PIDController {
                    params: control_params.do_controller.clone(),
                    last_update: 0.0,
                    output: 0.0,
                    enabled: true,
                },
                ph_controller: PIDController {
                    params: control_params.ph_controller.clone(),
                    last_update: 0.0,
                    output: 0.0,
                    enabled: true,
                },
                temperature_controller: PIDController {
                    params: control_params.temperature_controller.clone(),
                    last_update: 0.0,
                    output: 0.0,
                    enabled: true,
                },
                feeding_controller: FeedingController {
                    glucose_controller: PIDController {
                        params: control_params.feeding_controller.clone(),
                        last_update: 0.0,
                        output: 0.0,
                        enabled: true,
                    },
                    fed_batch_enabled: true,
                    perfusion_enabled: false,
                    current_feed_rate: 0.0,
                    total_feed_volume: 0.0,
                },
                safety_systems: SafetySystems {
                    emergency_stop: false,
                    gas_supply_pressure: 25.0, // 25 PSI
                    power_backup_active: false,
                    containment_integrity: true,
                    sterility_status: SterilityStatus::Sterile,
                },
            },
        }
    }

    /// Start bioreactor control loop
    pub async fn start_control_loop(&self) -> Result<(), KambuzumaError> {
        let mut interval = interval(Duration::from_secs(60)); // 1-minute control cycle

        loop {
            interval.tick().await;

            // Update process model
            self.update_process_model().await?;

            // Execute control algorithms
            self.execute_control_algorithms().await?;

            // Monitor process parameters
            self.monitor_process_parameters().await?;

            // Check safety systems
            self.check_safety_systems().await?;

            // Log data
            self.log_process_data().await?;
        }
    }

    /// Update process model with differential equations
    async fn update_process_model(&self) -> Result<(), KambuzumaError> {
        let mut state = self.state.write().unwrap();
        let dt = 1.0 / 60.0; // 1-minute time step in hours

        // Update process time
        state.process_time += dt;

        // Cell growth kinetics (Monod equation with inhibition)
        let mu_max = 0.04; // h⁻¹ maximum specific growth rate
        let ks_glucose = 0.1; // g/L glucose half-saturation constant
        let ki_lactate = 40.0; // g/L lactate inhibition constant
        let ki_ammonia = 15.0; // mM ammonia inhibition constant

        // Monod equation with product inhibition
        let mu = mu_max
            * (state.glucose_concentration / (ks_glucose + state.glucose_concentration))
            * (ki_lactate / (ki_lactate + state.lactate_concentration))
            * (ki_ammonia / (ki_ammonia + state.ammonia_concentration));

        state.specific_growth_rate = mu;

        // Cell density change (exponential growth with death)
        let death_rate = 0.001; // h⁻¹ death rate
        let dxdt = mu * state.viable_cell_density - death_rate * state.viable_cell_density;
        state.viable_cell_density += dxdt * dt;

        // Update viability
        state.viability = (state.viable_cell_density / state.total_cell_density) * 100.0;
        state.total_cell_density = state.viable_cell_density / (state.viability / 100.0);

        // Glucose consumption (Michaelis-Menten kinetics)
        let q_glucose_max = 0.5e-9; // g/cell/h maximum specific glucose consumption
        let ks_consumption = 0.1; // g/L
        let q_glucose = q_glucose_max * (state.glucose_concentration / (ks_consumption + state.glucose_concentration));

        let glucose_consumption_rate = q_glucose * state.viable_cell_density;
        state.glucose_concentration -= glucose_consumption_rate * dt;

        // Lactate production (proportional to glucose consumption)
        let yield_lactate = 0.9; // g lactate / g glucose
        state.lactate_concentration += glucose_consumption_rate * yield_lactate * dt;

        // Oxygen uptake rate (proportional to cell density and growth)
        let q_oxygen = 1e-9; // mmol O₂/cell/h
        state.oxygen_uptake_rate = q_oxygen * state.viable_cell_density + 0.1 * mu * state.viable_cell_density;

        // Dissolved oxygen balance
        let c_star = 0.21; // mmol/L oxygen saturation at 1 atm
        let do_sat = state.dissolved_oxygen / 100.0;
        let oxygen_transfer_rate = state.kla * (c_star - do_sat * c_star);

        let do_change = (oxygen_transfer_rate - state.oxygen_uptake_rate) / c_star * 100.0;
        state.dissolved_oxygen += do_change * dt;

        // Update turbidity (proportional to cell density)
        state.turbidity = state.total_cell_density / 1e6 * 0.3; // Approximate relationship

        Ok(())
    }

    /// Execute control algorithms for all process parameters
    async fn execute_control_algorithms(&self) -> Result<(), KambuzumaError> {
        let state = self.state.read().unwrap().clone();
        let mut controllers = self.controllers.clone();

        // Dissolved oxygen control
        if controllers.do_controller.enabled {
            let error = self.control_params.target_dissolved_oxygen - state.dissolved_oxygen;
            let output = self.calculate_pid_output(&mut controllers.do_controller, error, state.process_time);

            // Adjust aeration rate and agitation speed
            let mut state_mut = self.state.write().unwrap();
            state_mut.aeration_rate = (0.05 + output * 0.001).max(0.0).min(2.0); // 0.05-2.0 vvm
            state_mut.agitation_speed = (50.0 + output * 10.0).max(50.0).min(300.0); // 50-300 RPM

            // Update kLA based on aeration and agitation
            state_mut.kla = 2.0 * state_mut.aeration_rate.powf(0.7) * (state_mut.agitation_speed / 100.0).powf(0.3);
        }

        // pH control
        if controllers.ph_controller.enabled {
            let error = self.control_params.target_ph - state.ph;
            let output = self.calculate_pid_output(&mut controllers.ph_controller, error, state.process_time);

            let mut state_mut = self.state.write().unwrap();
            if output > 0.0 {
                state_mut.base_addition_rate = output * 5.0; // Base addition
                state_mut.acid_addition_rate = 0.0;
            } else {
                state_mut.acid_addition_rate = -output * 5.0; // Acid addition
                state_mut.base_addition_rate = 0.0;
            }

            // Simple pH model
            let ph_change = (state_mut.base_addition_rate - state_mut.acid_addition_rate) * 0.001;
            state_mut.ph += ph_change * (1.0 / 60.0); // 1-minute time step
        }

        // Temperature control
        if controllers.temperature_controller.enabled {
            let error = self.control_params.target_temperature - state.temperature;
            let output = self.calculate_pid_output(&mut controllers.temperature_controller, error, state.process_time);

            let mut state_mut = self.state.write().unwrap();
            if output > 0.0 {
                state_mut.heating_power = output * 100.0; // Heating
                state_mut.cooling_power = 0.0;
            } else {
                state_mut.cooling_power = -output * 100.0; // Cooling
                state_mut.heating_power = 0.0;
            }

            // Simple temperature model
            let temp_change = (state_mut.heating_power - state_mut.cooling_power) * 0.0001;
            state_mut.temperature += temp_change * (1.0 / 60.0);
        }

        // Feeding control
        if controllers.feeding_controller.fed_batch_enabled {
            if state.glucose_concentration < self.control_params.feeding_trigger_glucose {
                let error = self.control_params.target_glucose - state.glucose_concentration;
                let output = self.calculate_pid_output(
                    &mut controllers.feeding_controller.glucose_controller,
                    error,
                    state.process_time,
                );

                let mut state_mut = self.state.write().unwrap();
                controllers.feeding_controller.current_feed_rate =
                    (output * 10.0).max(0.0).min(self.control_params.max_feed_rate);

                // Add glucose from feeding
                let glucose_addition = controllers.feeding_controller.current_feed_rate
                    * self.control_params.glucose_feed_concentration
                    / state_mut.working_volume
                    / 1000.0; // Convert to g/L

                state_mut.glucose_concentration += glucose_addition * (1.0 / 60.0);
                controllers.feeding_controller.total_feed_volume +=
                    controllers.feeding_controller.current_feed_rate * (1.0 / 60.0);
            }
        }

        Ok(())
    }

    /// Calculate PID controller output
    fn calculate_pid_output(&self, controller: &mut PIDController, error: f64, current_time: f64) -> f64 {
        let dt = current_time - controller.last_update;
        if dt <= 0.0 {
            return controller.output;
        }

        // Proportional term
        let p_term = controller.params.kp * error;

        // Integral term
        controller.params.integral_term += error * dt;
        let i_term = controller.params.ki * controller.params.integral_term;

        // Derivative term
        let d_term = if dt > 0.0 {
            controller.params.kd * (error - controller.params.last_error) / dt
        } else {
            0.0
        };

        // Calculate output
        controller.output = p_term + i_term + d_term;

        // Update for next iteration
        controller.params.last_error = error;
        controller.last_update = current_time;

        controller.output
    }

    /// Monitor process parameters and generate alarms
    async fn monitor_process_parameters(&self) -> Result<(), KambuzumaError> {
        let state = self.state.read().unwrap();
        let mut monitoring = self.monitoring.write().unwrap();

        // Update current measurements
        monitoring
            .current_measurements
            .insert("viable_cell_density".to_string(), state.viable_cell_density);
        monitoring.current_measurements.insert("viability".to_string(), state.viability);
        monitoring
            .current_measurements
            .insert("dissolved_oxygen".to_string(), state.dissolved_oxygen);
        monitoring.current_measurements.insert("ph".to_string(), state.ph);
        monitoring
            .current_measurements
            .insert("temperature".to_string(), state.temperature);
        monitoring
            .current_measurements
            .insert("glucose".to_string(), state.glucose_concentration);
        monitoring
            .current_measurements
            .insert("lactate".to_string(), state.lactate_concentration);

        // Check alarm conditions
        self.check_alarm_conditions(&state, &mut monitoring.alarms);

        // Update quality metrics
        monitoring.quality_metrics.doubling_time = if state.specific_growth_rate > 0.0 {
            0.693 / state.specific_growth_rate // ln(2) / μ
        } else {
            f64::INFINITY
        };

        // Store historical data point
        let data_point = ProcessDataPoint {
            timestamp: state.process_time,
            measurements: monitoring.current_measurements.clone(),
        };
        monitoring.historical_data.push(data_point);

        // Keep only last 1000 data points
        if monitoring.historical_data.len() > 1000 {
            monitoring.historical_data.remove(0);
        }

        Ok(())
    }

    /// Check alarm conditions
    fn check_alarm_conditions(&self, state: &BioreactorState, alarms: &mut Vec<ProcessAlarm>) {
        // Clear old alarms
        alarms.clear();

        // Low viability alarm
        if state.viability < self.control_params.min_viability {
            alarms.push(ProcessAlarm {
                alarm_id: "LOW_VIABILITY".to_string(),
                parameter: "viability".to_string(),
                alarm_type: AlarmType::LowAlarm,
                value: state.viability,
                limit: self.control_params.min_viability,
                timestamp: state.process_time,
                acknowledged: false,
            });
        }

        // High lactate alarm
        if state.lactate_concentration > self.control_params.max_lactate {
            alarms.push(ProcessAlarm {
                alarm_id: "HIGH_LACTATE".to_string(),
                parameter: "lactate".to_string(),
                alarm_type: AlarmType::HighAlarm,
                value: state.lactate_concentration,
                limit: self.control_params.max_lactate,
                timestamp: state.process_time,
                acknowledged: false,
            });
        }

        // Low dissolved oxygen alarm
        if state.dissolved_oxygen < 20.0 {
            alarms.push(ProcessAlarm {
                alarm_id: "LOW_DO".to_string(),
                parameter: "dissolved_oxygen".to_string(),
                alarm_type: AlarmType::LowAlarm,
                value: state.dissolved_oxygen,
                limit: 20.0,
                timestamp: state.process_time,
                acknowledged: false,
            });
        }

        // Temperature deviation alarm
        if (state.temperature - self.control_params.target_temperature).abs() > 1.0 {
            alarms.push(ProcessAlarm {
                alarm_id: "TEMP_DEVIATION".to_string(),
                parameter: "temperature".to_string(),
                alarm_type: AlarmType::HighAlarm,
                value: state.temperature,
                limit: self.control_params.target_temperature,
                timestamp: state.process_time,
                acknowledged: false,
            });
        }
    }

    /// Check safety systems
    async fn check_safety_systems(&self) -> Result<(), KambuzumaError> {
        let state = self.state.read().unwrap();

        // Check critical parameters for emergency stop conditions
        if state.temperature > 42.0 || state.temperature < 30.0 {
            log::error!("Emergency stop: Temperature out of safe range: {}°C", state.temperature);
            // Would trigger emergency stop in real system
        }

        if state.ph < 6.5 || state.ph > 8.0 {
            log::error!("Emergency stop: pH out of safe range: {}", state.ph);
            // Would trigger emergency stop in real system
        }

        Ok(())
    }

    /// Log process data
    async fn log_process_data(&self) -> Result<(), KambuzumaError> {
        let state = self.state.read().unwrap();

        log::info!(
            "Bioreactor {}: T={:.1}h, X={:.1}M cells/mL, Viab={:.1}%, DO={:.1}%, pH={:.2}, T={:.1}°C, Glc={:.2}g/L",
            self.reactor_id,
            state.process_time,
            state.viable_cell_density / 1e6,
            state.viability,
            state.dissolved_oxygen,
            state.ph,
            state.temperature,
            state.glucose_concentration
        );

        Ok(())
    }

    /// Get current bioreactor state
    pub fn get_state(&self) -> BioreactorState {
        self.state.read().unwrap().clone()
    }

    /// Get process monitoring data
    pub fn get_monitoring_data(&self) -> ProcessMonitoringData {
        self.monitoring.read().unwrap().clone()
    }
}
