use clap::{App, Arg, SubCommand};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::errors::KambuzumaError;
use crate::interfaces::InterfaceManager;

pub mod commands;
pub mod parsers;

/// CLI configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CliConfig {
    pub enabled: bool,
    pub interactive_mode: bool,
    pub output_format: String,
    pub verbose: bool,
    pub color_output: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interactive_mode: false,
            output_format: "json".to_string(),
            verbose: false,
            color_output: true,
        }
    }
}

/// CLI manager
pub struct CliManager {
    interface_manager: InterfaceManager,
    config: CliConfig,
}

impl CliManager {
    /// Create new CLI manager
    pub fn new(interface_manager: InterfaceManager, config: CliConfig) -> Self {
        Self {
            interface_manager,
            config,
        }
    }

    /// Run CLI in interactive mode
    pub async fn run_interactive(&self) -> Result<(), KambuzumaError> {
        println!("Kambuzuma Interactive CLI v{}", env!("CARGO_PKG_VERSION"));
        println!("Type 'help' for available commands or 'quit' to exit.\n");

        loop {
            print!("kambuzuma> ");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                break;
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            if input == "quit" || input == "exit" {
                println!("Goodbye!");
                break;
            }

            if let Err(e) = self.handle_command(input).await {
                eprintln!("Error: {}", e);
            }
        }

        Ok(())
    }

    /// Handle a single command
    pub async fn handle_command(&self, command: &str) -> Result<(), KambuzumaError> {
        let args: Vec<&str> = command.split_whitespace().collect();
        if args.is_empty() {
            return Ok(());
        }

        match args[0] {
            "help" => self.show_help(),
            "status" => commands::monitoring::show_status(&self.interface_manager, &self.config).await,
            "orchestrate" => {
                commands::orchestrate::process_query(&self.interface_manager, &args[1..], &self.config).await
            },
            "monitor" => commands::monitor::start_monitoring(&self.interface_manager, &args[1..], &self.config).await,
            "configure" => {
                commands::configure::handle_configuration(&self.interface_manager, &args[1..], &self.config).await
            },
            "analyze" => commands::analyze::run_analysis(&self.interface_manager, &args[1..], &self.config).await,
            "validate" => commands::validate::run_validation(&self.interface_manager, &args[1..], &self.config).await,
            "quantum" => {
                commands::quantum::handle_quantum_commands(&self.interface_manager, &args[1..], &self.config).await
            },
            "neural" => {
                commands::neural::handle_neural_commands(&self.interface_manager, &args[1..], &self.config).await
            },
            _ => {
                eprintln!("Unknown command: {}. Type 'help' for available commands.", args[0]);
                Ok(())
            },
        }
    }

    /// Show help information
    fn show_help(&self) -> Result<(), KambuzumaError> {
        println!("Available commands:");
        println!("  help                     - Show this help message");
        println!("  status                   - Show system status");
        println!("  orchestrate <query>      - Process a query through the system");
        println!("  monitor [options]        - Start system monitoring");
        println!("  configure [options]      - Configure system settings");
        println!("  analyze [options]        - Run system analysis");
        println!("  validate [options]       - Run validation tests");
        println!("  quantum [subcommand]     - Quantum system commands");
        println!("  neural [subcommand]      - Neural processing commands");
        println!("  quit/exit               - Exit the CLI");
        Ok(())
    }
}

/// Initialize CLI
pub async fn initialize(interface_manager: InterfaceManager) -> Result<(), KambuzumaError> {
    let config = CliConfig::default();
    let cli_manager = CliManager::new(interface_manager, config);

    // In interactive mode, run the CLI loop
    if cli_manager.config.interactive_mode {
        cli_manager.run_interactive().await
    } else {
        // Non-interactive mode - just initialize and return
        log::info!("CLI initialized in non-interactive mode");
        Ok(())
    }
}
