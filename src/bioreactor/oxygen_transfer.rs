use crate::errors::KambuzumaError;

/// Oxygen transfer system for bioreactor control
/// Implements comprehensive kLA calculation and dissolved oxygen mass balance
pub struct OxygenTransferSystem {
    /// Mass transfer parameters
    pub transfer_params: MassTransferParameters,

    /// Gas flow parameters
    pub gas_params: GasFlowParameters,

    /// Physical properties
    pub physical_props: PhysicalProperties,

    /// Operating conditions
    pub operating_conditions: OperatingConditions,
}

/// Mass transfer coefficients and correlations
#[derive(Debug, Clone)]
pub struct MassTransferParameters {
    /// Overall volumetric mass transfer coefficient (h⁻¹)
    pub kla: f64,

    /// Liquid-side mass transfer coefficient (m/h)
    pub kl: f64,

    /// Gas-liquid interfacial area per unit volume (m⁻¹)
    pub a: f64,

    /// Henry's law constant for oxygen (atm·m³/mol)
    pub henry_constant: f64,

    /// Oxygen solubility in medium (mg/L at 1 atm)
    pub c_star: f64,
}

/// Gas flow and sparging parameters
#[derive(Debug, Clone)]
pub struct GasFlowParameters {
    /// Superficial gas velocity (m/s)
    pub superficial_velocity: f64,

    /// Aeration rate (vvm - vessel volumes per minute)
    pub aeration_rate: f64,

    /// Gas composition
    pub oxygen_fraction: f64, // Volume fraction of O₂
    pub co2_fraction: f64,      // Volume fraction of CO₂
    pub nitrogen_fraction: f64, // Volume fraction of N₂

    /// Sparger design parameters
    pub sparger_type: SpargerType,
    pub hole_diameter: f64, // m
    pub number_of_holes: i32,
    pub sparger_diameter: f64, // m
}

/// Types of spargers available
#[derive(Debug, Clone)]
pub enum SpargerType {
    RingSparger,
    PipeSparger,
    SinteredSparger,
    MicrospargerMembrane,
}

/// Physical properties of the system
#[derive(Debug, Clone)]
pub struct PhysicalProperties {
    /// Liquid density (kg/m³)
    pub liquid_density: f64,

    /// Liquid viscosity (Pa·s)
    pub liquid_viscosity: f64,

    /// Surface tension (N/m)
    pub surface_tension: f64,

    /// Gas density (kg/m³)
    pub gas_density: f64,

    /// Diffusivity of oxygen in liquid (m²/s)
    pub diffusivity: f64,
}

/// Operating conditions affecting mass transfer
#[derive(Debug, Clone)]
pub struct OperatingConditions {
    /// Temperature (°C)
    pub temperature: f64,

    /// Pressure (atm)
    pub pressure: f64,

    /// Agitation speed (RPM)
    pub agitation_speed: f64,

    /// Impeller diameter (m)
    pub impeller_diameter: f64,

    /// Tank diameter (m)
    pub tank_diameter: f64,

    /// Working volume (L)
    pub working_volume: f64,

    /// Power input per unit volume (W/m³)
    pub power_per_volume: f64,
}

/// Dissolved oxygen state and mass balance
#[derive(Debug, Clone)]
pub struct DissolvedOxygenState {
    /// Current dissolved oxygen concentration (mg/L)
    pub concentration: f64,

    /// Dissolved oxygen saturation (%)
    pub saturation_percentage: f64,

    /// Oxygen uptake rate (mg/L/h)
    pub uptake_rate: f64,

    /// Mass transfer rate (mg/L/h)
    pub transfer_rate: f64,

    /// Gas phase oxygen partial pressure (atm)
    pub partial_pressure: f64,
}

impl OxygenTransferSystem {
    /// Create standard oxygen transfer system for mammalian cell culture
    pub fn new() -> Self {
        Self {
            transfer_params: MassTransferParameters {
                kla: 10.0,              // h⁻¹ typical for stirred tank
                kl: 0.1,                // m/h
                a: 100.0,               // m⁻¹
                henry_constant: 769.23, // atm·m³/mol for O₂ at 37°C
                c_star: 6.4,            // mg/L at 1 atm, 37°C
            },

            gas_params: GasFlowParameters {
                superficial_velocity: 0.01, // m/s
                aeration_rate: 0.1,         // vvm
                oxygen_fraction: 0.21,      // 21% O₂
                co2_fraction: 0.05,         // 5% CO₂
                nitrogen_fraction: 0.74,    // 74% N₂
                sparger_type: SpargerType::RingSparger,
                hole_diameter: 1e-3, // 1 mm
                number_of_holes: 20,
                sparger_diameter: 0.1, // 10 cm
            },

            physical_props: PhysicalProperties {
                liquid_density: 1000.0,  // kg/m³ (water-like)
                liquid_viscosity: 0.001, // Pa·s (water at 20°C)
                surface_tension: 0.072,  // N/m (water)
                gas_density: 1.2,        // kg/m³ (air at STP)
                diffusivity: 2.1e-9,     // m²/s (O₂ in water at 25°C)
            },

            operating_conditions: OperatingConditions {
                temperature: 37.0,       // °C
                pressure: 1.0,           // atm
                agitation_speed: 100.0,  // RPM
                impeller_diameter: 0.05, // 5 cm
                tank_diameter: 0.15,     // 15 cm
                working_volume: 2.0,     // L
                power_per_volume: 100.0, // W/m³
            },
        }
    }

    /// Calculate kLA using empirical correlations
    pub fn calculate_kla(&mut self) -> Result<f64, KambuzumaError> {
        // Use the classic Cooper correlation for stirred tank bioreactors
        // kLa = K * (P/V)^α * (vs)^β

        let power_per_volume = self.operating_conditions.power_per_volume;
        let superficial_velocity = self.gas_params.superficial_velocity;

        // Cooper correlation parameters for mammalian cell culture
        let k_constant = 0.026; // Empirical constant
        let alpha = 0.4; // Power exponent
        let beta = 0.5; // Superficial velocity exponent

        let kla = k_constant * power_per_volume.powf(alpha) * superficial_velocity.powf(beta);

        // Temperature correction (Arrhenius-type)
        let temp_correction = 1.024_f64.powf(self.operating_conditions.temperature - 20.0);

        // Viscosity correction
        let viscosity_correction = (0.001 / self.physical_props.liquid_viscosity).powf(0.5);

        self.transfer_params.kla = kla * temp_correction * viscosity_correction;

        Ok(self.transfer_params.kla)
    }

    /// Calculate kLA using alternative Penicillin correlation
    pub fn calculate_kla_penicillin_correlation(&mut self) -> Result<f64, KambuzumaError> {
        // Penicillin correlation: kLa = 0.002 * (P/V)^0.7 * (vs)^0.2
        let power_per_volume = self.operating_conditions.power_per_volume;
        let superficial_velocity = self.gas_params.superficial_velocity;

        let kla = 0.002 * power_per_volume.powf(0.7) * superficial_velocity.powf(0.2);

        self.transfer_params.kla = kla;
        Ok(kla)
    }

    /// Calculate power input per unit volume
    pub fn calculate_power_per_volume(&mut self) -> Result<f64, KambuzumaError> {
        // Calculate power number for impeller
        let reynolds_number = self.calculate_reynolds_number();
        let power_number = self.calculate_power_number(reynolds_number);

        // Power input calculation
        let n = self.operating_conditions.agitation_speed / 60.0; // Convert RPM to RPS
        let d = self.operating_conditions.impeller_diameter;
        let rho = self.physical_props.liquid_density;

        let power = power_number * rho * n.powi(3) * d.powi(5);

        // Power per unit volume
        let volume = self.operating_conditions.working_volume / 1000.0; // Convert L to m³
        self.operating_conditions.power_per_volume = power / volume;

        Ok(self.operating_conditions.power_per_volume)
    }

    /// Calculate Reynolds number for impeller
    fn calculate_reynolds_number(&self) -> f64 {
        let n = self.operating_conditions.agitation_speed / 60.0; // RPS
        let d = self.operating_conditions.impeller_diameter;
        let rho = self.physical_props.liquid_density;
        let mu = self.physical_props.liquid_viscosity;

        (rho * n * d.powi(2)) / mu
    }

    /// Calculate power number based on Reynolds number and impeller type
    fn calculate_power_number(&self, reynolds_number: f64) -> f64 {
        // Typical power number for Rushton turbine
        if reynolds_number < 10.0 {
            // Laminar flow regime
            64.0 / reynolds_number
        } else if reynolds_number < 10000.0 {
            // Transition regime
            5.75 - 0.5 * reynolds_number.log10()
        } else {
            // Turbulent regime
            6.0 // Typical for Rushton turbine
        }
    }

    /// Calculate superficial gas velocity
    pub fn calculate_superficial_velocity(&mut self) -> Result<f64, KambuzumaError> {
        let volume_flow_rate =
            self.gas_params.aeration_rate * self.operating_conditions.working_volume / (1000.0 * 60.0); // m³/s

        let tank_cross_sectional_area = std::f64::consts::PI * (self.operating_conditions.tank_diameter / 2.0).powi(2);

        self.gas_params.superficial_velocity = volume_flow_rate / tank_cross_sectional_area;

        Ok(self.gas_params.superficial_velocity)
    }

    /// Update dissolved oxygen mass balance
    pub fn update_dissolved_oxygen(
        &self,
        current_do: f64,
        our: f64,
        dt: f64,
    ) -> Result<DissolvedOxygenState, KambuzumaError> {
        // Calculate oxygen saturation concentration
        let c_star = self.calculate_oxygen_saturation();

        // Calculate driving force
        let driving_force = c_star - current_do;

        // Mass transfer rate (mg/L/h)
        let otr = self.transfer_params.kla * driving_force;

        // Overall oxygen balance: dC/dt = OTR - OUR
        let dcdt = otr - our;

        // Update concentration
        let new_concentration = current_do + dcdt * dt;

        // Calculate saturation percentage
        let saturation_percentage = (new_concentration / c_star) * 100.0;

        // Calculate partial pressure
        let partial_pressure = self.gas_params.oxygen_fraction * self.operating_conditions.pressure;

        Ok(DissolvedOxygenState {
            concentration: new_concentration.max(0.0),
            saturation_percentage: saturation_percentage.max(0.0),
            uptake_rate: our,
            transfer_rate: otr,
            partial_pressure,
        })
    }

    /// Calculate oxygen saturation concentration based on temperature and pressure
    fn calculate_oxygen_saturation(&self) -> f64 {
        // Henry's law: C* = H * P_O2
        // Temperature correction for Henry's constant
        let temp_kelvin = self.operating_conditions.temperature + 273.15;
        let temp_correction = (1800.0 / temp_kelvin).exp();

        let henry_corrected = self.transfer_params.henry_constant * temp_correction;
        let partial_pressure = self.gas_params.oxygen_fraction * self.operating_conditions.pressure;

        // Convert from mol/L to mg/L (MW of O₂ = 32 g/mol)
        let c_star_mol_l = partial_pressure / henry_corrected;
        let c_star_mg_l = c_star_mol_l * 32.0 * 1000.0; // Convert to mg/L

        c_star_mg_l
    }

    /// Optimize aeration rate for target dissolved oxygen
    pub fn optimize_aeration_rate(&mut self, target_do: f64, current_do: f64, our: f64) -> Result<f64, KambuzumaError> {
        let c_star = self.calculate_oxygen_saturation();

        // Required OTR to maintain target DO
        let required_otr = our + (target_do - current_do);

        // Required kLA
        let driving_force = c_star - target_do;
        let required_kla = required_otr / driving_force;

        // Estimate required aeration rate using correlation
        // kLA ∝ (aeration_rate)^β
        let beta = 0.5; // From Cooper correlation
        let current_kla = self.transfer_params.kla;

        let aeration_multiplier = (required_kla / current_kla).powf(1.0 / beta);
        let optimal_aeration = self.gas_params.aeration_rate * aeration_multiplier;

        // Limit aeration rate to practical ranges
        self.gas_params.aeration_rate = optimal_aeration.max(0.01).min(2.0); // 0.01-2.0 vvm

        Ok(self.gas_params.aeration_rate)
    }

    /// Calculate bubble characteristics
    pub fn calculate_bubble_properties(&self) -> Result<BubbleProperties, KambuzumaError> {
        // Sauter mean diameter calculation
        let weber_number = self.calculate_weber_number();
        let d32 = self.calculate_sauter_mean_diameter(weber_number);

        // Gas holdup calculation
        let gas_holdup = self.calculate_gas_holdup();

        // Interfacial area
        let interfacial_area = 6.0 * gas_holdup / d32;

        Ok(BubbleProperties {
            sauter_mean_diameter: d32,
            gas_holdup,
            interfacial_area,
            rise_velocity: self.calculate_bubble_rise_velocity(d32),
        })
    }

    /// Calculate Weber number for bubble formation
    fn calculate_weber_number(&self) -> f64 {
        let rho = self.physical_props.liquid_density;
        let sigma = self.physical_props.surface_tension;
        let velocity = self.gas_params.superficial_velocity;
        let length = self.gas_params.hole_diameter;

        (rho * velocity.powi(2) * length) / sigma
    }

    /// Calculate Sauter mean diameter of bubbles
    fn calculate_sauter_mean_diameter(&self, weber_number: f64) -> f64 {
        // Empirical correlation for sparger bubbles
        let d_orifice = self.gas_params.hole_diameter;

        if weber_number < 3.0 {
            // Low Weber number regime
            d_orifice * (1.0 + 0.1 * weber_number)
        } else {
            // High Weber number regime
            d_orifice * 0.6 * weber_number.powf(-0.5)
        }
    }

    /// Calculate gas holdup
    fn calculate_gas_holdup(&self) -> f64 {
        // Empirical correlation for gas holdup
        let vs = self.gas_params.superficial_velocity;
        let power_per_volume = self.operating_conditions.power_per_volume;

        0.2 * vs.powf(0.7) * power_per_volume.powf(0.1)
    }

    /// Calculate bubble rise velocity
    fn calculate_bubble_rise_velocity(&self, diameter: f64) -> f64 {
        let g = 9.81; // m/s²
        let rho_l = self.physical_props.liquid_density;
        let rho_g = self.physical_props.gas_density;
        let mu = self.physical_props.liquid_viscosity;

        // Terminal velocity calculation for spherical bubbles
        let archimedes_number = (rho_l * (rho_l - rho_g) * g * diameter.powi(3)) / mu.powi(2);

        if archimedes_number < 3.7 {
            // Stokes flow regime
            (g * diameter.powi(2) * (rho_l - rho_g)) / (18.0 * mu)
        } else if archimedes_number < 400.0 {
            // Intermediate regime
            let reynolds_terminal = archimedes_number.powf(0.5) / 18.0;
            (reynolds_terminal * mu) / (rho_l * diameter)
        } else {
            // Newton's law regime
            (g * diameter * (rho_l - rho_g) / rho_l).sqrt()
        }
    }
}

/// Bubble properties in the bioreactor
#[derive(Debug, Clone)]
pub struct BubbleProperties {
    /// Sauter mean diameter (m)
    pub sauter_mean_diameter: f64,

    /// Gas holdup (dimensionless)
    pub gas_holdup: f64,

    /// Interfacial area per unit volume (m⁻¹)
    pub interfacial_area: f64,

    /// Bubble rise velocity (m/s)
    pub rise_velocity: f64,
}

/// Dissolved oxygen control strategy
pub struct DOControlStrategy {
    /// Target dissolved oxygen (% saturation)
    pub target_do: f64,

    /// Control deadband (% saturation)
    pub deadband: f64,

    /// Maximum aeration rate (vvm)
    pub max_aeration: f64,

    /// Minimum aeration rate (vvm)
    pub min_aeration: f64,

    /// Agitation speed limits (RPM)
    pub max_agitation: f64,
    pub min_agitation: f64,
}

impl DOControlStrategy {
    /// Create standard DO control strategy
    pub fn standard() -> Self {
        Self {
            target_do: 40.0,      // 40% saturation
            deadband: 2.0,        // ±2% deadband
            max_aeration: 1.0,    // 1.0 vvm max
            min_aeration: 0.05,   // 0.05 vvm min
            max_agitation: 300.0, // 300 RPM max
            min_agitation: 50.0,  // 50 RPM min
        }
    }
}
