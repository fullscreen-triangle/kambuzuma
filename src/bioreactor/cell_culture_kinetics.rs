use crate::errors::KambuzumaError;

/// Cell culture kinetics implementation with detailed differential equations
/// Based on standard bioprocess engineering principles for mammalian cell culture
pub struct CellCultureKinetics {
    /// Kinetic parameters for cell growth
    pub growth_params: GrowthKinetics,

    /// Substrate utilization parameters
    pub substrate_params: SubstrateKinetics,

    /// Metabolite production parameters
    pub metabolite_params: MetaboliteKinetics,

    /// Environmental effects
    pub environmental_params: EnvironmentalEffects,
}

/// Growth kinetics parameters for mammalian cells
#[derive(Debug, Clone)]
pub struct GrowthKinetics {
    /// Maximum specific growth rate (h⁻¹)
    pub mu_max: f64,

    /// Death rate constant (h⁻¹)
    pub kd: f64,

    /// Glucose half-saturation constant (g/L)
    pub ks_glucose: f64,

    /// Glutamine half-saturation constant (mM)
    pub ks_glutamine: f64,

    /// Oxygen half-saturation constant (% DO)
    pub ks_oxygen: f64,

    /// Temperature effect parameters
    pub temp_optimum: f64, // °C
    pub temp_sensitivity: f64,

    /// pH effect parameters
    pub ph_optimum: f64,
    pub ph_sensitivity: f64,
}

/// Substrate utilization kinetics
#[derive(Debug, Clone)]
pub struct SubstrateKinetics {
    /// Glucose consumption parameters
    pub glucose: SubstrateParams,

    /// Glutamine consumption parameters
    pub glutamine: SubstrateParams,

    /// Oxygen consumption parameters
    pub oxygen: OxygenParams,

    /// Essential amino acids
    pub amino_acids: Vec<SubstrateParams>,
}

/// Individual substrate consumption parameters
#[derive(Debug, Clone)]
pub struct SubstrateParams {
    /// Maximum specific consumption rate
    pub q_max: f64,

    /// Half-saturation constant
    pub ks: f64,

    /// Maintenance coefficient
    pub ms: f64,

    /// Growth-associated yield coefficient
    pub yxs: f64,
}

/// Oxygen consumption parameters
#[derive(Debug, Clone)]
pub struct OxygenParams {
    /// Specific oxygen uptake rate for maintenance (mmol O₂/cell/h)
    pub q_o2_maintenance: f64,

    /// Specific oxygen uptake rate for growth (mmol O₂/cell/h)
    pub q_o2_growth: f64,

    /// Oxygen yield coefficient (mmol O₂/g biomass)
    pub y_o2_x: f64,

    /// Critical dissolved oxygen level (% saturation)
    pub do_critical: f64,
}

/// Metabolite production kinetics
#[derive(Debug, Clone)]
pub struct MetaboliteKinetics {
    /// Lactate production
    pub lactate: MetaboliteParams,

    /// Ammonia production
    pub ammonia: MetaboliteParams,

    /// CO₂ production
    pub co2: MetaboliteParams,

    /// Product formation (e.g., antibodies)
    pub product: ProductParams,
}

/// Metabolite production parameters
#[derive(Debug, Clone)]
pub struct MetaboliteParams {
    /// Growth-associated production coefficient
    pub alpha: f64,

    /// Non-growth-associated production coefficient
    pub beta: f64,

    /// Maximum production rate
    pub q_max: f64,

    /// Inhibition constant
    pub ki: f64,
}

/// Product formation parameters (for recombinant proteins)
#[derive(Debug, Clone)]
pub struct ProductParams {
    /// Specific productivity (pg/cell/day)
    pub qp: f64,

    /// Product degradation rate (h⁻¹)
    pub kp_deg: f64,

    /// Temperature sensitivity
    pub temp_effect: f64,

    /// pH sensitivity
    pub ph_effect: f64,
}

/// Environmental effects on cell culture
#[derive(Debug, Clone)]
pub struct EnvironmentalEffects {
    /// Osmolality effects
    pub osmolality_optimum: f64, // mOsm/kg
    pub osmolality_sensitivity: f64,

    /// CO₂ partial pressure effects
    pub co2_optimum: f64, // %
    pub co2_sensitivity: f64,

    /// Shear stress effects
    pub shear_threshold: f64, // s⁻¹
    pub shear_sensitivity: f64,
}

/// Cell culture state variables
#[derive(Debug, Clone)]
pub struct CultureState {
    /// Viable cell density (cells/mL)
    pub viable_cells: f64,

    /// Total cell density (cells/mL)
    pub total_cells: f64,

    /// Cell viability (%)
    pub viability: f64,

    /// Substrate concentrations
    pub glucose: f64, // g/L
    pub glutamine: f64, // mM
    pub oxygen: f64,    // % saturation

    /// Metabolite concentrations
    pub lactate: f64, // g/L
    pub ammonia: f64, // mM
    pub co2: f64,     // %

    /// Product concentration
    pub product: f64, // mg/L

    /// Environmental conditions
    pub temperature: f64, // °C
    pub ph: f64,
    pub osmolality: f64, // mOsm/kg
}

impl CellCultureKinetics {
    /// Create standard CHO cell kinetics parameters
    pub fn cho_cell_parameters() -> Self {
        Self {
            growth_params: GrowthKinetics {
                mu_max: 0.04,       // 0.04 h⁻¹ (17.3 h doubling time)
                kd: 0.002,          // 0.002 h⁻¹ death rate
                ks_glucose: 0.1,    // 0.1 g/L
                ks_glutamine: 0.05, // 0.05 mM
                ks_oxygen: 5.0,     // 5% DO
                temp_optimum: 37.0, // 37°C
                temp_sensitivity: 0.1,
                ph_optimum: 7.2, // pH 7.2
                ph_sensitivity: 2.0,
            },

            substrate_params: SubstrateKinetics {
                glucose: SubstrateParams {
                    q_max: 5e-10, // g/cell/h
                    ks: 0.1,      // g/L
                    ms: 1e-10,    // g/cell/h maintenance
                    yxs: 1e8,     // cells/g glucose
                },

                glutamine: SubstrateParams {
                    q_max: 5e-13, // mol/cell/h
                    ks: 0.05,     // mM
                    ms: 1e-13,    // mol/cell/h maintenance
                    yxs: 2e9,     // cells/mol glutamine
                },

                oxygen: OxygenParams {
                    q_o2_maintenance: 1e-12, // mmol O₂/cell/h
                    q_o2_growth: 5e-12,      // mmol O₂/cell/h
                    y_o2_x: 0.5,             // mmol O₂/g biomass
                    do_critical: 10.0,       // 10% DO critical
                },

                amino_acids: vec![], // Simplified for now
            },

            metabolite_params: MetaboliteKinetics {
                lactate: MetaboliteParams {
                    alpha: 0.9,  // g lactate/g glucose
                    beta: 1e-10, // g/cell/h non-growth
                    q_max: 1e-9, // g/cell/h maximum
                    ki: 40.0,    // g/L inhibition
                },

                ammonia: MetaboliteParams {
                    alpha: 0.8,   // mol NH₃/mol glutamine
                    beta: 1e-13,  // mol/cell/h non-growth
                    q_max: 1e-12, // mol/cell/h maximum
                    ki: 15.0,     // mM inhibition
                },

                co2: MetaboliteParams {
                    alpha: 1.0,   // mol CO₂/mol glucose
                    beta: 1e-12,  // mol/cell/h non-growth
                    q_max: 1e-11, // mol/cell/h maximum
                    ki: 20.0,     // % inhibition
                },

                product: ProductParams {
                    qp: 20.0,          // pg/cell/day
                    kp_deg: 0.001,     // h⁻¹ degradation
                    temp_effect: 0.05, // per °C
                    ph_effect: 0.1,    // per pH unit
                },
            },

            environmental_params: EnvironmentalEffects {
                osmolality_optimum: 300.0, // mOsm/kg
                osmolality_sensitivity: 0.01,
                co2_optimum: 5.0, // %
                co2_sensitivity: 0.02,
                shear_threshold: 100.0, // s⁻¹
                shear_sensitivity: 0.001,
            },
        }
    }

    /// Calculate differential equations for cell culture dynamics
    pub fn calculate_derivatives(&self, state: &CultureState, dt: f64) -> Result<CultureState, KambuzumaError> {
        // Calculate environmental effects
        let temp_effect = self.calculate_temperature_effect(state.temperature);
        let ph_effect = self.calculate_ph_effect(state.ph);
        let oxygen_effect = self.calculate_oxygen_effect(state.oxygen);
        let inhibition_effect = self.calculate_inhibition_effects(state);

        // Overall environmental multiplier
        let env_effect = temp_effect * ph_effect * oxygen_effect * inhibition_effect;

        // Calculate specific growth rate (Monod equation)
        let mu_glucose = self.growth_params.mu_max * (state.glucose / (self.growth_params.ks_glucose + state.glucose));

        let mu_glutamine =
            self.growth_params.mu_max * (state.glutamine / (self.growth_params.ks_glutamine + state.glutamine));

        let mu_oxygen = self.growth_params.mu_max * (state.oxygen / (self.growth_params.ks_oxygen + state.oxygen));

        // Limiting substrate determines growth rate
        let mu = mu_glucose.min(mu_glutamine).min(mu_oxygen) * env_effect;

        // Cell growth and death
        let growth_rate = mu * state.viable_cells;
        let death_rate = self.growth_params.kd * state.viable_cells;

        let dxdt = growth_rate - death_rate;
        let new_viable_cells = state.viable_cells + dxdt * dt;
        let new_total_cells = state.total_cells + growth_rate * dt;

        // Calculate substrate consumption rates
        let glucose_consumption = self.calculate_glucose_consumption(state, mu);
        let glutamine_consumption = self.calculate_glutamine_consumption(state, mu);
        let oxygen_consumption = self.calculate_oxygen_consumption(state, mu);

        // Calculate metabolite production rates
        let lactate_production = self.calculate_lactate_production(state, glucose_consumption);
        let ammonia_production = self.calculate_ammonia_production(state, glutamine_consumption);
        let co2_production = self.calculate_co2_production(state, glucose_consumption);

        // Calculate product formation
        let product_formation = self.calculate_product_formation(state);

        // Update concentrations
        Ok(CultureState {
            viable_cells: new_viable_cells.max(0.0),
            total_cells: new_total_cells.max(0.0),
            viability: if new_total_cells > 0.0 {
                (new_viable_cells / new_total_cells * 100.0).min(100.0)
            } else {
                0.0
            },

            glucose: (state.glucose - glucose_consumption * dt).max(0.0),
            glutamine: (state.glutamine - glutamine_consumption * dt).max(0.0),
            oxygen: state.oxygen - oxygen_consumption * dt, // Can go negative

            lactate: state.lactate + lactate_production * dt,
            ammonia: state.ammonia + ammonia_production * dt,
            co2: state.co2 + co2_production * dt,

            product: state.product + product_formation * dt,

            temperature: state.temperature,
            ph: state.ph,
            osmolality: state.osmolality,
        })
    }

    /// Calculate temperature effect on growth
    fn calculate_temperature_effect(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.growth_params.temp_optimum;
        (-self.growth_params.temp_sensitivity * delta_t.powi(2)).exp()
    }

    /// Calculate pH effect on growth
    fn calculate_ph_effect(&self, ph: f64) -> f64 {
        let delta_ph = ph - self.growth_params.ph_optimum;
        (-self.growth_params.ph_sensitivity * delta_ph.powi(2)).exp()
    }

    /// Calculate oxygen limitation effect
    fn calculate_oxygen_effect(&self, dissolved_oxygen: f64) -> f64 {
        if dissolved_oxygen < self.substrate_params.oxygen.do_critical {
            dissolved_oxygen / self.substrate_params.oxygen.do_critical
        } else {
            1.0
        }
    }

    /// Calculate product inhibition effects
    fn calculate_inhibition_effects(&self, state: &CultureState) -> f64 {
        let lactate_inhibition =
            self.metabolite_params.lactate.ki / (self.metabolite_params.lactate.ki + state.lactate);

        let ammonia_inhibition =
            self.metabolite_params.ammonia.ki / (self.metabolite_params.ammonia.ki + state.ammonia);

        lactate_inhibition * ammonia_inhibition
    }

    /// Calculate glucose consumption rate
    fn calculate_glucose_consumption(&self, state: &CultureState, mu: f64) -> f64 {
        let maintenance = self.substrate_params.glucose.ms * state.viable_cells;
        let growth_associated = mu * state.viable_cells / self.substrate_params.glucose.yxs;

        maintenance + growth_associated
    }

    /// Calculate glutamine consumption rate
    fn calculate_glutamine_consumption(&self, state: &CultureState, mu: f64) -> f64 {
        let maintenance = self.substrate_params.glutamine.ms * state.viable_cells;
        let growth_associated = mu * state.viable_cells / self.substrate_params.glutamine.yxs;

        maintenance + growth_associated
    }

    /// Calculate oxygen consumption rate
    fn calculate_oxygen_consumption(&self, state: &CultureState, mu: f64) -> f64 {
        let maintenance = self.substrate_params.oxygen.q_o2_maintenance * state.viable_cells;
        let growth_associated = self.substrate_params.oxygen.q_o2_growth * mu * state.viable_cells;

        maintenance + growth_associated
    }

    /// Calculate lactate production rate
    fn calculate_lactate_production(&self, state: &CultureState, glucose_consumption: f64) -> f64 {
        let growth_associated = self.metabolite_params.lactate.alpha * glucose_consumption;
        let non_growth = self.metabolite_params.lactate.beta * state.viable_cells;

        growth_associated + non_growth
    }

    /// Calculate ammonia production rate
    fn calculate_ammonia_production(&self, state: &CultureState, glutamine_consumption: f64) -> f64 {
        let growth_associated = self.metabolite_params.ammonia.alpha * glutamine_consumption;
        let non_growth = self.metabolite_params.ammonia.beta * state.viable_cells;

        growth_associated + non_growth
    }

    /// Calculate CO₂ production rate
    fn calculate_co2_production(&self, state: &CultureState, glucose_consumption: f64) -> f64 {
        let growth_associated = self.metabolite_params.co2.alpha * glucose_consumption;
        let non_growth = self.metabolite_params.co2.beta * state.viable_cells;

        growth_associated + non_growth
    }

    /// Calculate product formation rate
    fn calculate_product_formation(&self, state: &CultureState) -> f64 {
        let temp_effect = 1.0 + self.metabolite_params.product.temp_effect * (state.temperature - 37.0);
        let ph_effect = 1.0 + self.metabolite_params.product.ph_effect * (state.ph - 7.2);

        let specific_productivity = self.metabolite_params.product.qp / 24.0; // Convert from per day to per hour
        let production = specific_productivity * state.viable_cells * temp_effect * ph_effect;
        let degradation = self.metabolite_params.product.kp_deg * state.product;

        production - degradation
    }

    /// Calculate doubling time from specific growth rate
    pub fn calculate_doubling_time(&self, mu: f64) -> f64 {
        if mu > 0.0 {
            0.693 / mu // ln(2) / μ
        } else {
            f64::INFINITY
        }
    }

    /// Calculate cell-specific productivity
    pub fn calculate_specific_productivity(&self, product_concentration: f64, ivccd: f64) -> f64 {
        if ivccd > 0.0 {
            product_concentration / ivccd * 24.0 // pg/cell/day
        } else {
            0.0
        }
    }
}
