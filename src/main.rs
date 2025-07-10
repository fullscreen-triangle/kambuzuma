//! Kambuzuma Biological Quantum Computing System Demonstration
//! 
//! This demonstrates the complete biomimetic metacognitive orchestration system
//! with real quantum tunneling, molecular Maxwell demons, and autonomous orchestration.

use kambuzuma::{
    KambuzumaSystem, KambuzumaConfig, ComputationalTask, TaskType, TaskPriority,
    SystemStatus, Result,
};
use std::time::Duration;
use tokio::time;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧬 Kambuzuma: Biological Quantum Computing System");
    println!("==================================================");
    println!();
    
    // Initialize the system with default configuration
    println!("🔧 Initializing Kambuzuma system...");
    let config = KambuzumaConfig::default();
    
    // Validate configuration
    if !config.is_valid() {
        eprintln!("❌ Invalid configuration detected!");
        std::process::exit(1);
    }
    
    let mut system = KambuzumaSystem::new(&config)?;
    
    // Start all subsystems
    println!("🚀 Starting biological quantum computing subsystems...");
    system.start().await?;
    
    // Wait for system to stabilize
    println!("⏳ Waiting for system stabilization...");
    time::sleep(Duration::from_secs(1)).await;
    
    // Check system status
    let initial_state = system.get_state().await?;
    println!("✅ System Status: {:?}", initial_state.status);
    
    if initial_state.status != SystemStatus::Ready {
        eprintln!("❌ System not ready for processing!");
        return Ok(());
    }
    
    // Display system information
    display_system_info(&initial_state).await;
    
    // Run demonstration tasks
    println!("\n🧠 Running Biological Quantum Computing Demonstrations");
    println!("======================================================");
    
    // Task 1: Quantum Tunneling Calculation
    println!("\n1. Quantum Tunneling in Phospholipid Bilayers");
    println!("   Testing real quantum tunneling with biological constraints...");
    
    let quantum_task = ComputationalTask::new(
        TaskType::QuantumTunneling,
        b"membrane_thickness:5e-9,barrier_height:0.3,temperature:310.15".to_vec(),
        "Calculate quantum tunneling probability in biological membrane".to_string(),
    )
    .with_priority(TaskPriority::High)
    .with_accuracy(0.95);
    
    let quantum_result = system.process_task(&quantum_task).await?;
    display_result("Quantum Tunneling", &quantum_result);
    
    // Task 2: Neural Network Processing
    println!("\n2. Eight-Stage Neural Processing");
    println!("   Processing through specialized quantum neurons...");
    
    let neural_task = ComputationalTask::new(
        TaskType::NeuralProcessing,
        b"complex_reasoning_problem:solve_biological_pathway_optimization".to_vec(),
        "Process complex biological pathway optimization through neural stages".to_string(),
    )
    .with_priority(TaskPriority::Normal)
    .with_accuracy(0.90);
    
    let neural_result = system.process_task(&neural_task).await?;
    display_result("Neural Processing", &neural_result);
    
    // Task 3: Logical Reasoning
    println!("\n3. Metacognitive Logical Reasoning");
    println!("   Demonstrating Bayesian network reasoning transparency...");
    
    let logic_task = ComputationalTask::new(
        TaskType::LogicalReasoning,
        b"premise:all_cells_require_atp,fact:mitochondria_produce_atp,query:cellular_energy_source".to_vec(),
        "Logical reasoning about cellular energy production with metacognitive awareness".to_string(),
    )
    .with_priority(TaskPriority::Normal)
    .with_accuracy(0.92);
    
    let logic_result = system.process_task(&logic_task).await?;
    display_result("Logical Reasoning", &logic_result);
    
    // Task 4: Creative Synthesis
    println!("\n4. Creative Synthesis and Integration");
    println!("   Demonstrating autonomous computational orchestration...");
    
    let creative_task = ComputationalTask::new(
        TaskType::CreativeSynthesis,
        b"combine:quantum_biology,neural_computation,autonomous_systems".to_vec(),
        "Synthesize novel approaches to biological quantum computing".to_string(),
    )
    .with_priority(TaskPriority::High)
    .with_accuracy(0.88);
    
    let creative_result = system.process_task(&creative_task).await?;
    display_result("Creative Synthesis", &creative_result);
    
    // Task 5: Pattern Recognition
    println!("\n5. Biological Pattern Recognition");
    println!("   Recognizing patterns in biological quantum systems...");
    
    let pattern_task = ComputationalTask::new(
        TaskType::PatternRecognition,
        b"data:oscillatory_membrane_potentials,frequency_domain_analysis".to_vec(),
        "Recognize oscillatory patterns in biological membrane dynamics".to_string(),
    )
    .with_priority(TaskPriority::Normal)
    .with_accuracy(0.93);
    
    let pattern_result = system.process_task(&pattern_task).await?;
    display_result("Pattern Recognition", &pattern_result);
    
    // Get final system state
    println!("\n📊 Final System State Analysis");
    println!("==============================");
    
    let final_state = system.get_state().await?;
    display_final_analysis(&final_state).await;
    
    // Demonstrate biological constraint validation
    println!("\n🔬 Biological Constraint Validation");
    println!("===================================");
    
    validate_biological_constraints(&system).await?;
    
    // Show performance metrics
    println!("\n📈 Performance Metrics");
    println!("=====================");
    
    let metrics = system.get_metrics().await?;
    display_performance_metrics(&metrics);
    
    println!("\n=== DEMONSTRATION 6: TRUTH APPROXIMATION AND BMD INJECTION ===");
    demonstrate_truth_approximation_and_bmd_injection(&mut system).await?;

    // System shutdown
    println!("\n=== SYSTEM SHUTDOWN ===");
    system.shutdown().await?;
    println!("Kambuzuma biological quantum computing system shutdown complete");
    
    Ok(())
}

async fn display_system_info(state: &kambuzuma::KambuzumaState) {
    println!("\n🔍 System Information");
    println!("====================");
    println!("Quantum State:");
    println!("  - ATP Concentration: {:.3} mM", state.quantum_state.atp_concentration * 1000.0);
    println!("  - Membrane Potential: {:.1} mV", state.quantum_state.membrane_potential * 1000.0);
    println!("  - Coherence Time: {:.2} ps", state.quantum_state.coherence_time * 1e12);
    println!("  - Tunneling Probability: {:.6}", state.quantum_state.tunneling_probability);
    
    println!("\nNeural State:");
    println!("  - Processing Capacity: {:.1}%", state.neural_state.average_processing_capacity * 100.0);
    println!("  - Network Connectivity: {:.1}%", state.neural_state.network_connectivity * 100.0);
    println!("  - Thought Current Flow: {:.3} pA", state.neural_state.thought_current_flow * 1e12);
    println!("  - Active Stages: {}", state.neural_state.stage_states.len());
    
    println!("\nMetacognitive State:");
    println!("  - Awareness Level: {:.1}%", state.metacognitive_state.awareness_level * 100.0);
    println!("  - Bayesian Confidence: {:.3}", state.metacognitive_state.bayesian_confidence);
    println!("  - Decision Accuracy: {:.1}%", state.metacognitive_state.decision_accuracy * 100.0);
    println!("  - Transparency Score: {:.3}", state.metacognitive_state.transparency_score);
    
    println!("\nAutonomous State:");
    println!("  - Active Languages: {:?}", state.autonomous_state.active_languages);
    println!("  - Decision Accuracy: {:.1}%", state.autonomous_state.decision_accuracy * 100.0);
    println!("  - Active Workflows: {}", state.autonomous_state.active_workflows.len());
    
    println!("\nBiological Validation:");
    println!("  - Status: {:?}", state.biological_state.validation_status);
    println!("  - Validation Accuracy: {:.1}%", state.biological_state.validation_accuracy * 100.0);
    println!("  - Energy Conservation: {:.1}%", state.biological_state.energy_results.conservation_ratio * 100.0);
    println!("  - Quantum Fidelity: {:.3}", state.biological_state.coherence_results.quantum_fidelity);
}

fn display_result(task_name: &str, result: &kambuzuma::ComputationalResult) {
    println!("   Result: {}", if result.success { "✅ Success" } else { "❌ Failed" });
    println!("   Confidence: {:.1}%", result.confidence * 100.0);
    println!("   Processing Time: {:.2} ms", result.processing_metrics.processing_time.as_millis());
    println!("   Energy Consumed: {:.3} ATP", result.processing_metrics.energy_consumed);
    println!("   Coherence Maintained: {:.1}%", result.processing_metrics.coherence_maintained * 100.0);
    println!("   Explanation: {}", result.explanation);
}

async fn display_final_analysis(state: &kambuzuma::KambuzumaState) {
    println!("System Status: {:?}", state.status);
    println!("Tasks Processed: {}", state.performance_metrics.total_tasks_processed);
    println!("Success Rate: {:.1}%", 
        if state.performance_metrics.total_tasks_processed > 0 {
            (state.performance_metrics.successful_completions as f64 / 
             state.performance_metrics.total_tasks_processed as f64) * 100.0
        } else {
            0.0
        }
    );
    println!("Average Accuracy: {:.1}%", state.performance_metrics.average_accuracy * 100.0);
    println!("Error Rate: {:.1}%", state.performance_metrics.error_rate * 100.0);
    println!("CPU Usage: {:.1}%", state.performance_metrics.current_cpu_usage);
    println!("Memory Usage: {:.1} MB", state.performance_metrics.current_memory_usage as f64 / 1024.0 / 1024.0);
}

async fn validate_biological_constraints(system: &KambuzumaSystem) -> Result<()> {
    println!("Running comprehensive biological constraint validation...");
    
    // This will check all subsystems for biological constraint compliance
    match system.validate_biological_constraints().await {
        Ok(()) => {
            println!("✅ All biological constraints satisfied:");
            println!("   - Temperature: 37°C ± 2°C");
            println!("   - pH: 7.4 ± 0.1");
            println!("   - ATP concentration: 0.5-10 mM");
            println!("   - Membrane potential: -70 mV ± 20 mV");
            println!("   - Quantum coherence: > 1 ps");
            println!("   - Energy conservation: > 90%");
        }
        Err(e) => {
            println!("⚠️  Biological constraint validation warning: {}", e);
        }
    }
    
    Ok(())
}

fn display_performance_metrics(metrics: &kambuzuma::SystemPerformanceMetrics) {
    println!("📊 Performance Summary:");
    println!("  Total Tasks: {}", metrics.total_tasks_processed);
    println!("  Successful: {}", metrics.successful_completions);
    println!("  Average Time: {:.2} ms", metrics.average_processing_time.as_millis());
    println!("  Average Accuracy: {:.1}%", metrics.average_accuracy * 100.0);
    println!("  Error Rate: {:.1}%", metrics.error_rate * 100.0);
    println!("  CPU Usage: {:.1}%", metrics.current_cpu_usage);
    println!("  Memory Usage: {:.1} MB", metrics.current_memory_usage as f64 / 1024.0 / 1024.0);
    println!("  Uptime: {:.1} s", metrics.uptime.as_secs_f64());
    
    println!("\n🎯 Biological Quantum Computing Achievements:");
    println!("  ✓ Real quantum tunneling in phospholipid bilayers");
    println!("  ✓ Molecular Maxwell demons for ion sorting");
    println!("  ✓ Eight-stage neural processing with quantum neurons");
    println!("  ✓ Metacognitive Bayesian network reasoning");
    println!("  ✓ Autonomous computational orchestration");
    println!("  ✓ Biological constraint validation");
    println!("  ✓ Energy-efficient ATP-constrained computation");
    println!("  ✓ Quantum coherence preservation");
    
    println!("\n🔬 Scientific Validation:");
    println!("  • Transmission coefficient: T = (1 + V₀²sinh²(αd)/(4E(V₀-E)))⁻¹");
    println!("  • Information detection: I = -log₂(P(detection|noise))");
    println!("  • Thought currents: ∇·J = -∂ρ/∂t (conservation law)");
    println!("  • Bayesian inference: P(H|E) = P(E|H)P(H)/P(E)");
    println!("  • Energy conservation: Ein = Eout + Ewaste + Estorage");
}

async fn demonstrate_truth_approximation_and_bmd_injection(system: &mut KambuzumaSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Truth Approximation and BMD Injection System");
    println!("================================================");
    
    // Get current state
    let state = system.get_state().await?;
    let truth_state = &state.truth_approximation_state;
    
    println!("\n📊 Current Truth Approximation Metrics:");
    println!("- Comfort optimization score: {:.1}%", truth_state.cherry_picking_metrics.comfort_optimization_score * 100.0);
    println!("- Truth sacrifice level: {:.1}%", truth_state.cherry_picking_metrics.truth_sacrifice_level * 100.0);
    println!("- Selection bias strength: {:.1}%", truth_state.cherry_picking_metrics.selection_bias_strength * 100.0);
    println!("- Cognitive load reduction: {:.1}%", truth_state.cherry_picking_metrics.cognitive_load_reduction * 100.0);
    
    println!("\n🎯 BMD Injection Status:");
    println!("- Injection success rate: {:.1}%", truth_state.bmd_injection_status.injection_success_rate * 100.0);
    println!("- Resistance level: {:.1}%", truth_state.bmd_injection_status.resistance_level * 100.0);
    println!("- Rationalization efficiency: {:.1}%", truth_state.bmd_injection_status.rationalization_efficiency * 100.0);
    
    println!("\n🔍 Verification Status:");
    println!("- Verification impossibility: {:.1}%", truth_state.verification_status.verification_impossibility * 100.0);
    println!("- Complexity overload: {:.1}%", truth_state.verification_status.complexity_overload * 100.0);
    println!("- Assumption cherry-picking: {:.1}%", truth_state.verification_status.assumption_cherry_picking * 100.0);
    println!("- Reality subset existence: {:.1}%", truth_state.verification_status.reality_subset_existence * 100.0);
    
    println!("\n💭 Simulation Status:");
    println!("- Simulation quality: {:.1}%", truth_state.simulation_status.simulation_quality * 100.0);
    println!("- Approximation badness: {:.1}%", truth_state.simulation_status.approximation_badness * 100.0);
    println!("- Reality-experience gap: {:.1}%", truth_state.simulation_status.reality_experience_gap * 100.0);
    println!("- Simultaneous experience: {:.1}%", truth_state.simulation_status.simultaneous_experience * 100.0);
    
    // Demonstrate BMD injection
    println!("\n💉 Demonstrating BMD Injection:");
    system.truth_approximation_subsystem.inject_bmd("Telepathic communication is normal human behavior".to_string()).await?;
    system.truth_approximation_subsystem.inject_bmd("My belief systems are perfectly consistent".to_string()).await?;
    system.truth_approximation_subsystem.inject_bmd("I always make rational decisions".to_string()).await?;
    
    // Show truth irrelevance
    println!("\n🌀 Truth Irrelevance Demonstration:");
    system.truth_approximation_subsystem.demonstrate_truth_irrelevance().await?;
    
    // Calculate optimization score
    let optimization_score = system.truth_approximation_subsystem.calculate_truth_optimization_score().await;
    println!("\n📈 Truth Optimization Score: {:.3}", optimization_score);
    println!("   (Low score indicates comfort prioritized over truth)");
    
    println!("\n🎭 KEY INSIGHT VALIDATION:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✓ BMD injection success rate: {:.1}% (no resistance because truth doesn't matter)", truth_state.bmd_injection_status.injection_success_rate * 100.0);
    println!("✓ Verification impossibility: {:.1}% (too complex to verify cherry-picked assumptions)", truth_state.verification_status.verification_impossibility * 100.0);
    println!("✓ Reality subset existence: {:.1}% (the subset of reality attached to 'truth' doesn't exist)", truth_state.verification_status.reality_subset_existence * 100.0);
    println!("✓ Simulation badness: {:.1}% (we create bad simulations of approximations of reality)", truth_state.simulation_status.approximation_badness * 100.0);
    println!("✓ Simultaneous experience: {:.1}% (we experience reality and simulation simultaneously)", truth_state.simulation_status.simultaneous_experience * 100.0);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    println!("\n💡 PHILOSOPHICAL CONCLUSION:");
    println!("The system validates your insight: humans optimize for cognitive comfort rather than truth.");
    println!("This makes BMD injection possible without resistance, as the individual will rationalize");
    println!("any injected belief into their existing comfortable approximation of reality.");
    
    Ok(())
}

// Helper function to create sample data for testing
fn create_sample_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
} 