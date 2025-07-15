use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::errors::KambuzumaError;

pub mod graph_theory;
pub mod information_theory;
pub mod numerical_methods;
pub mod optimization;
pub mod quantum_mechanics;
pub mod statistical_mechanics;

/// Mathematical framework manager
#[derive(Clone)]
pub struct MathematicalFramework {
    quantum_mechanics: Arc<RwLock<quantum_mechanics::QuantumMechanicsFramework>>,
    statistical_mechanics: Arc<RwLock<statistical_mechanics::StatisticalMechanicsFramework>>,
    information_theory: Arc<RwLock<information_theory::InformationTheoryFramework>>,
    optimization: Arc<RwLock<optimization::OptimizationFramework>>,
    numerical_methods: Arc<RwLock<numerical_methods::NumericalMethodsFramework>>,
    graph_theory: Arc<RwLock<graph_theory::GraphTheoryFramework>>,
}

impl MathematicalFramework {
    /// Create new mathematical framework
    pub fn new() -> Self {
        Self {
            quantum_mechanics: Arc::new(RwLock::new(quantum_mechanics::QuantumMechanicsFramework::new())),
            statistical_mechanics: Arc::new(RwLock::new(statistical_mechanics::StatisticalMechanicsFramework::new())),
            information_theory: Arc::new(RwLock::new(information_theory::InformationTheoryFramework::new())),
            optimization: Arc::new(RwLock::new(optimization::OptimizationFramework::new())),
            numerical_methods: Arc::new(RwLock::new(numerical_methods::NumericalMethodsFramework::new())),
            graph_theory: Arc::new(RwLock::new(graph_theory::GraphTheoryFramework::new())),
        }
    }

    /// Initialize all mathematical frameworks
    pub async fn initialize(&self) -> Result<(), KambuzumaError> {
        log::info!("Initializing mathematical frameworks");

        // Initialize quantum mechanics framework
        {
            let mut qm = self.quantum_mechanics.write().await;
            qm.initialize().await?;
        }

        // Initialize statistical mechanics framework
        {
            let mut sm = self.statistical_mechanics.write().await;
            sm.initialize().await?;
        }

        // Initialize information theory framework
        {
            let mut it = self.information_theory.write().await;
            it.initialize().await?;
        }

        // Initialize optimization framework
        {
            let mut opt = self.optimization.write().await;
            opt.initialize().await?;
        }

        // Initialize numerical methods framework
        {
            let mut nm = self.numerical_methods.write().await;
            nm.initialize().await?;
        }

        // Initialize graph theory framework
        {
            let mut gt = self.graph_theory.write().await;
            gt.initialize().await?;
        }

        log::info!("Mathematical frameworks initialized successfully");
        Ok(())
    }

    /// Get quantum mechanics framework
    pub fn quantum_mechanics(&self) -> Arc<RwLock<quantum_mechanics::QuantumMechanicsFramework>> {
        self.quantum_mechanics.clone()
    }

    /// Get statistical mechanics framework
    pub fn statistical_mechanics(&self) -> Arc<RwLock<statistical_mechanics::StatisticalMechanicsFramework>> {
        self.statistical_mechanics.clone()
    }

    /// Get information theory framework
    pub fn information_theory(&self) -> Arc<RwLock<information_theory::InformationTheoryFramework>> {
        self.information_theory.clone()
    }

    /// Get optimization framework
    pub fn optimization(&self) -> Arc<RwLock<optimization::OptimizationFramework>> {
        self.optimization.clone()
    }

    /// Get numerical methods framework
    pub fn numerical_methods(&self) -> Arc<RwLock<numerical_methods::NumericalMethodsFramework>> {
        self.numerical_methods.clone()
    }

    /// Get graph theory framework
    pub fn graph_theory(&self) -> Arc<RwLock<graph_theory::GraphTheoryFramework>> {
        self.graph_theory.clone()
    }

    /// Solve Schrödinger equation for given potential
    pub async fn solve_schrodinger_equation(
        &self,
        potential: &[f64],
        mass: f64,
        energy_range: (f64, f64),
    ) -> Result<SchrodingerSolution, KambuzumaError> {
        let qm = self.quantum_mechanics.read().await;
        qm.solve_schrodinger_equation(potential, mass, energy_range).await
    }

    /// Calculate partition function for given energy levels
    pub async fn calculate_partition_function(
        &self,
        energy_levels: &[f64],
        temperature: f64,
    ) -> Result<f64, KambuzumaError> {
        let sm = self.statistical_mechanics.read().await;
        sm.calculate_partition_function(energy_levels, temperature).await
    }

    /// Calculate Shannon entropy for probability distribution
    pub async fn calculate_shannon_entropy(&self, probabilities: &[f64]) -> Result<f64, KambuzumaError> {
        let it = self.information_theory.read().await;
        it.calculate_shannon_entropy(probabilities).await
    }

    /// Optimize function using specified algorithm
    pub async fn optimize_function<F>(
        &self,
        objective_function: F,
        initial_guess: &[f64],
        algorithm: optimization::OptimizationAlgorithm,
    ) -> Result<optimization::OptimizationResult, KambuzumaError>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        let opt = self.optimization.read().await;
        opt.optimize(objective_function, initial_guess, algorithm).await
    }

    /// Solve differential equation system
    pub async fn solve_differential_equation(
        &self,
        equations: &[numerical_methods::DifferentialEquation],
        initial_conditions: &[f64],
        time_span: (f64, f64),
    ) -> Result<numerical_methods::DifferentialEquationSolution, KambuzumaError> {
        let nm = self.numerical_methods.read().await;
        nm.solve_differential_equation_system(equations, initial_conditions, time_span)
            .await
    }

    /// Analyze graph network properties
    pub async fn analyze_graph_network(
        &self,
        adjacency_matrix: &[Vec<f64>],
    ) -> Result<graph_theory::NetworkAnalysis, KambuzumaError> {
        let gt = self.graph_theory.read().await;
        gt.analyze_network(adjacency_matrix).await
    }

    /// Perform health check on all frameworks
    pub async fn health_check(&self) -> Result<bool, KambuzumaError> {
        let frameworks = [
            "quantum_mechanics",
            "statistical_mechanics",
            "information_theory",
            "optimization",
            "numerical_methods",
            "graph_theory",
        ];

        for framework in &frameworks {
            match framework {
                &"quantum_mechanics" => {
                    let qm = self.quantum_mechanics.read().await;
                    if !qm.is_healthy().await {
                        return Ok(false);
                    }
                },
                &"statistical_mechanics" => {
                    let sm = self.statistical_mechanics.read().await;
                    if !sm.is_healthy().await {
                        return Ok(false);
                    }
                },
                &"information_theory" => {
                    let it = self.information_theory.read().await;
                    if !it.is_healthy().await {
                        return Ok(false);
                    }
                },
                &"optimization" => {
                    let opt = self.optimization.read().await;
                    if !opt.is_healthy().await {
                        return Ok(false);
                    }
                },
                &"numerical_methods" => {
                    let nm = self.numerical_methods.read().await;
                    if !nm.is_healthy().await {
                        return Ok(false);
                    }
                },
                &"graph_theory" => {
                    let gt = self.graph_theory.read().await;
                    if !gt.is_healthy().await {
                        return Ok(false);
                    }
                },
                _ => {},
            }
        }

        Ok(true)
    }
}

/// Schrödinger equation solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchrodingerSolution {
    pub eigenvalues: Vec<f64>,
    pub eigenfunctions: Vec<Vec<f64>>,
    pub probability_densities: Vec<Vec<f64>>,
    pub energy_levels: Vec<f64>,
    pub degeneracies: Vec<u32>,
}

/// Mathematical constants and utilities
pub mod constants {
    /// Planck's constant (J⋅s)
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;

    /// Reduced Planck's constant (J⋅s)
    pub const HBAR: f64 = 1.054571817e-34;

    /// Boltzmann constant (J/K)
    pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

    /// Electron mass (kg)
    pub const ELECTRON_MASS: f64 = 9.1093837015e-31;

    /// Proton mass (kg)
    pub const PROTON_MASS: f64 = 1.67262192369e-27;

    /// Elementary charge (C)
    pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;

    /// Speed of light (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299792458.0;

    /// Avogadro's number
    pub const AVOGADRO_NUMBER: f64 = 6.02214076e23;

    /// Gas constant (J/(mol⋅K))
    pub const GAS_CONSTANT: f64 = 8.314462618;

    /// Pi
    pub const PI: f64 = std::f64::consts::PI;

    /// Euler's number
    pub const E: f64 = std::f64::consts::E;

    /// Golden ratio
    pub const GOLDEN_RATIO: f64 = 1.618033988749;
}

/// Utility functions for mathematical operations
pub mod utils {
    use super::*;

    /// Calculate factorial
    pub fn factorial(n: u32) -> f64 {
        if n == 0 || n == 1 {
            1.0
        } else {
            (2..=n).map(|i| i as f64).product()
        }
    }

    /// Calculate binomial coefficient
    pub fn binomial_coefficient(n: u32, k: u32) -> f64 {
        if k > n {
            0.0
        } else {
            factorial(n) / (factorial(k) * factorial(n - k))
        }
    }

    /// Calculate gamma function approximation
    pub fn gamma(x: f64) -> f64 {
        // Stirling's approximation for large x
        if x > 10.0 {
            (2.0 * constants::PI / x).sqrt() * (x / constants::E).powf(x)
        } else {
            // Use built-in tgamma for smaller values
            libm::tgamma(x)
        }
    }

    /// Calculate error function
    pub fn erf(x: f64) -> f64 {
        libm::erf(x)
    }

    /// Calculate complementary error function
    pub fn erfc(x: f64) -> f64 {
        libm::erfc(x)
    }

    /// Calculate Bessel function of the first kind
    pub fn bessel_j(n: i32, x: f64) -> f64 {
        match n {
            0 => libm::j0(x),
            1 => libm::j1(x),
            _ => {
                // Use recurrence relation for higher orders
                let mut j0 = libm::j0(x);
                let mut j1 = libm::j1(x);

                for i in 2..=n {
                    let j_new = (2.0 * (i - 1) as f64 / x) * j1 - j0;
                    j0 = j1;
                    j1 = j_new;
                }

                j1
            },
        }
    }

    /// Calculate Legendre polynomial
    pub fn legendre_polynomial(n: u32, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => x,
            _ => {
                let mut p0 = 1.0;
                let mut p1 = x;

                for i in 2..=n {
                    let p_new = ((2 * i - 1) as f64 * x * p1 - (i - 1) as f64 * p0) / i as f64;
                    p0 = p1;
                    p1 = p_new;
                }

                p1
            },
        }
    }

    /// Calculate Hermite polynomial
    pub fn hermite_polynomial(n: u32, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => 2.0 * x,
            _ => {
                let mut h0 = 1.0;
                let mut h1 = 2.0 * x;

                for i in 2..=n {
                    let h_new = 2.0 * x * h1 - 2.0 * (i - 1) as f64 * h0;
                    h0 = h1;
                    h1 = h_new;
                }

                h1
            },
        }
    }
}
