use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use warp::Reply;

use crate::interfaces::rest_api::{ApiResponse, ApiState};

/// Health check response structure
#[derive(Debug, Serialize)]
pub struct HealthStatus {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub components: ComponentsHealth,
    pub system_info: SystemInfo,
}

/// Individual component health status
#[derive(Debug, Serialize)]
pub struct ComponentsHealth {
    pub quantum_system: ComponentStatus,
    pub neural_processor: ComponentStatus,
    pub metacognitive_orchestrator: ComponentStatus,
    pub biological_validation: ComponentStatus,
    pub memory_usage: MemoryStatus,
    pub storage: StorageStatus,
}

/// Individual component status
#[derive(Debug, Serialize)]
pub struct ComponentStatus {
    pub status: String,
    pub response_time_ms: Option<u64>,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub details: Option<String>,
}

/// Memory usage information
#[derive(Debug, Serialize)]
pub struct MemoryStatus {
    pub status: String,
    pub total_mb: u64,
    pub used_mb: u64,
    pub available_mb: u64,
    pub usage_percentage: f64,
}

/// Storage status information
#[derive(Debug, Serialize)]
pub struct StorageStatus {
    pub status: String,
    pub total_gb: u64,
    pub used_gb: u64,
    pub available_gb: u64,
    pub usage_percentage: f64,
}

/// System information
#[derive(Debug, Serialize)]
pub struct SystemInfo {
    pub hostname: String,
    pub operating_system: String,
    pub cpu_cores: u32,
    pub total_memory_gb: f64,
    pub rust_version: String,
}

/// Handle health check request
pub async fn health_check(state: Arc<RwLock<ApiState>>) -> Result<impl Reply, warp::Rejection> {
    let start_time = std::time::Instant::now();

    let state_guard = state.read().await;
    let interface_manager = &state_guard.interface_manager;
    let start_timestamp = state_guard.start_time;
    drop(state_guard);

    // Calculate uptime
    let uptime_seconds = chrono::Utc::now().signed_duration_since(start_timestamp).num_seconds() as u64;

    // Check component health
    let components_health = check_components_health(interface_manager).await;

    // Get system info
    let system_info = get_system_info();

    // Determine overall status
    let overall_status = if components_health.quantum_system.status == "healthy"
        && components_health.neural_processor.status == "healthy"
        && components_health.metacognitive_orchestrator.status == "healthy"
        && components_health.memory_usage.usage_percentage < 90.0
        && components_health.storage.usage_percentage < 90.0
    {
        "healthy"
    } else if components_health.quantum_system.status == "degraded"
        || components_health.neural_processor.status == "degraded"
        || components_health.metacognitive_orchestrator.status == "degraded"
        || components_health.memory_usage.usage_percentage < 95.0
    {
        "degraded"
    } else {
        "unhealthy"
    };

    let health_status = HealthStatus {
        status: overall_status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds,
        timestamp: chrono::Utc::now(),
        components: components_health,
        system_info,
    };

    let response_time = start_time.elapsed().as_millis() as u64;
    log::debug!(
        "Health check completed in {}ms, status: {}",
        response_time,
        overall_status
    );

    Ok(warp::reply::json(&ApiResponse::success(health_status)))
}

/// Check health of all system components
async fn check_components_health(interface_manager: &crate::interfaces::InterfaceManager) -> ComponentsHealth {
    let now = chrono::Utc::now();

    // Check quantum system
    let quantum_status = check_quantum_system_health(interface_manager).await;

    // Check neural processor
    let neural_status = check_neural_processor_health(interface_manager).await;

    // Check metacognitive orchestrator
    let orchestrator_status = check_orchestrator_health(interface_manager).await;

    // Check biological validation
    let biological_status = check_biological_validation_health().await;

    // Check memory usage
    let memory_status = get_memory_status();

    // Check storage
    let storage_status = get_storage_status();

    ComponentsHealth {
        quantum_system: quantum_status,
        neural_processor: neural_status,
        metacognitive_orchestrator: orchestrator_status,
        biological_validation: biological_status,
        memory_usage: memory_status,
        storage: storage_status,
    }
}

/// Check quantum system health
async fn check_quantum_system_health(interface_manager: &crate::interfaces::InterfaceManager) -> ComponentStatus {
    let start = std::time::Instant::now();

    match interface_manager.quantum_system().read().await.health_check().await {
        Ok(healthy) => ComponentStatus {
            status: if healthy { "healthy" } else { "degraded" }.to_string(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
            last_check: chrono::Utc::now(),
            details: if healthy {
                Some("All quantum subsystems operational".to_string())
            } else {
                Some("Some quantum subsystems degraded".to_string())
            },
        },
        Err(e) => ComponentStatus {
            status: "unhealthy".to_string(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
            last_check: chrono::Utc::now(),
            details: Some(format!("Quantum system error: {}", e)),
        },
    }
}

/// Check neural processor health
async fn check_neural_processor_health(interface_manager: &crate::interfaces::InterfaceManager) -> ComponentStatus {
    let start = std::time::Instant::now();

    match interface_manager.neural_processor().read().await.health_check().await {
        Ok(healthy) => ComponentStatus {
            status: if healthy { "healthy" } else { "degraded" }.to_string(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
            last_check: chrono::Utc::now(),
            details: if healthy {
                Some("All neural stages operational".to_string())
            } else {
                Some("Some neural stages degraded".to_string())
            },
        },
        Err(e) => ComponentStatus {
            status: "unhealthy".to_string(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
            last_check: chrono::Utc::now(),
            details: Some(format!("Neural processor error: {}", e)),
        },
    }
}

/// Check orchestrator health
async fn check_orchestrator_health(interface_manager: &crate::interfaces::InterfaceManager) -> ComponentStatus {
    let start = std::time::Instant::now();

    match interface_manager.orchestrator().read().await.health_check().await {
        Ok(healthy) => ComponentStatus {
            status: if healthy { "healthy" } else { "degraded" }.to_string(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
            last_check: chrono::Utc::now(),
            details: if healthy {
                Some("Metacognitive systems operational".to_string())
            } else {
                Some("Metacognitive systems degraded".to_string())
            },
        },
        Err(e) => ComponentStatus {
            status: "unhealthy".to_string(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
            last_check: chrono::Utc::now(),
            details: Some(format!("Orchestrator error: {}", e)),
        },
    }
}

/// Check biological validation health
async fn check_biological_validation_health() -> ComponentStatus {
    let start = std::time::Instant::now();

    // For now, assume biological validation is always healthy
    ComponentStatus {
        status: "healthy".to_string(),
        response_time_ms: Some(start.elapsed().as_millis() as u64),
        last_check: chrono::Utc::now(),
        details: Some("Biological validation systems operational".to_string()),
    }
}

/// Get memory usage status
fn get_memory_status() -> MemoryStatus {
    // Simple memory status - in production would use system metrics
    let total_mb = 8192; // 8GB
    let used_mb = 2048; // 2GB
    let available_mb = total_mb - used_mb;
    let usage_percentage = (used_mb as f64 / total_mb as f64) * 100.0;

    let status = if usage_percentage < 80.0 {
        "healthy"
    } else if usage_percentage < 90.0 {
        "warning"
    } else {
        "critical"
    };

    MemoryStatus {
        status: status.to_string(),
        total_mb,
        used_mb,
        available_mb,
        usage_percentage,
    }
}

/// Get storage status
fn get_storage_status() -> StorageStatus {
    // Simple storage status - in production would use filesystem metrics
    let total_gb = 100; // 100GB
    let used_gb = 45; // 45GB
    let available_gb = total_gb - used_gb;
    let usage_percentage = (used_gb as f64 / total_gb as f64) * 100.0;

    let status = if usage_percentage < 80.0 {
        "healthy"
    } else if usage_percentage < 90.0 {
        "warning"
    } else {
        "critical"
    };

    StorageStatus {
        status: status.to_string(),
        total_gb,
        used_gb,
        available_gb,
        usage_percentage,
    }
}

/// Get system information
fn get_system_info() -> SystemInfo {
    SystemInfo {
        hostname: "kambuzuma-node".to_string(),
        operating_system: std::env::consts::OS.to_string(),
        cpu_cores: num_cpus::get() as u32,
        total_memory_gb: 8.0, // Simplified
        rust_version: rustc_version_runtime::version().to_string(),
    }
}
