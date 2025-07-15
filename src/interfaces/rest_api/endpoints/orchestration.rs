use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use warp::{Filter, Reply};

use crate::errors::KambuzumaError;
use crate::interfaces::rest_api::{ApiResponse, ApiState};
use crate::interfaces::{ProcessingRequest, ProcessingResponse};

/// Process a new query through the metacognitive orchestrator
pub fn process_query(
    state: Arc<RwLock<ApiState>>,
) -> impl Filter<Extract = impl Reply, Error = warp::Rejection> + Clone {
    warp::path("process")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || state.clone()))
        .and_then(handle_process_query)
}

/// Get processing history
pub fn get_processing_history(
    state: Arc<RwLock<ApiState>>,
) -> impl Filter<Extract = impl Reply, Error = warp::Rejection> + Clone {
    warp::path("history")
        .and(warp::get())
        .and(warp::query())
        .and(warp::any().map(move || state.clone()))
        .and_then(handle_get_processing_history)
}

/// Get active processes
pub fn get_active_processes(
    state: Arc<RwLock<ApiState>>,
) -> impl Filter<Extract = impl Reply, Error = warp::Rejection> + Clone {
    warp::path("active")
        .and(warp::get())
        .and(warp::any().map(move || state.clone()))
        .and_then(handle_get_active_processes)
}

/// Cancel a running process
pub fn cancel_process(
    state: Arc<RwLock<ApiState>>,
) -> impl Filter<Extract = impl Reply, Error = warp::Rejection> + Clone {
    warp::path("cancel")
        .and(warp::path::param::<String>())
        .and(warp::delete())
        .and(warp::any().map(move || state.clone()))
        .and_then(handle_cancel_process)
}

/// Handle query processing request
async fn handle_process_query(
    request: ProcessingRequest,
    state: Arc<RwLock<ApiState>>,
) -> Result<impl Reply, warp::Rejection> {
    let start_time = std::time::Instant::now();
    let process_id = Uuid::new_v4().to_string();

    log::info!("Processing query: {} (ID: {})", request.query, process_id);

    let state_guard = state.read().await;
    let orchestrator = state_guard.interface_manager.orchestrator();
    let neural_processor = state_guard.interface_manager.neural_processor();
    let quantum_system = state_guard.interface_manager.quantum_system();
    drop(state_guard);

    // Process through metacognitive orchestrator
    match process_query_internal(&request, &process_id, orchestrator, neural_processor, quantum_system).await {
        Ok(response) => {
            let processing_time = start_time.elapsed().as_millis() as u64;

            let full_response = ProcessingResponse {
                response: response.response,
                confidence: response.confidence,
                processing_trace: response.processing_trace,
                quantum_states: response.quantum_states,
                thought_currents: response.thought_currents,
                energy_consumption: response.energy_consumption,
                processing_time_ms: processing_time,
            };

            Ok(warp::reply::json(&ApiResponse::success(full_response)))
        },
        Err(e) => {
            log::error!("Query processing failed: {}", e);
            Ok(warp::reply::json(&ApiResponse::<()>::error(format!(
                "Processing failed: {}",
                e
            ))))
        },
    }
}

/// Internal query processing logic
async fn process_query_internal(
    request: &ProcessingRequest,
    process_id: &str,
    orchestrator: Arc<RwLock<crate::metacognition::MetacognitiveOrchestrator>>,
    neural_processor: Arc<RwLock<crate::neural::NeuralProcessingUnit>>,
    quantum_system: Arc<RwLock<crate::quantum::QuantumMembraneSystem>>,
) -> Result<ProcessingResponse, KambuzumaError> {
    // Initialize processing context
    let mut processing_trace = Vec::new();
    processing_trace.push(format!(
        "Process {} initiated for query: '{}'",
        process_id, request.query
    ));

    // Step 1: Prepare quantum membrane system
    {
        let mut quantum = quantum_system.write().await;
        quantum.initialize_for_processing().await?;
        processing_trace.push("Quantum membrane system initialized".to_string());
    }

    // Step 2: Process through neural stages
    let neural_result = {
        let mut neural = neural_processor.write().await;
        neural.process_query(&request.query, request.context.as_deref()).await?
    };
    processing_trace.push(format!(
        "Neural processing completed: {} stages active",
        neural_result.stages_activated.len()
    ));

    // Step 3: Metacognitive orchestration
    let orchestration_result = {
        let mut orchestrator_guard = orchestrator.write().await;
        orchestrator_guard
            .orchestrate_processing(&request.query, &neural_result, request.priority.unwrap_or(5))
            .await?
    };
    processing_trace.push("Metacognitive orchestration completed".to_string());

    // Step 4: Extract final quantum states
    let quantum_states = {
        let quantum = quantum_system.read().await;
        vec![
            format!("Membrane coherence: {:.3}", quantum.get_coherence_level()),
            format!("Tunneling probability: {:.3}", quantum.get_tunneling_probability()),
            format!("Energy transfer rate: {:.3} ATP/s", quantum.get_energy_transfer_rate()),
        ]
    };

    // Step 5: Get thought currents
    let thought_currents = {
        let neural = neural_processor.read().await;
        neural.get_current_thought_currents().await
    };

    // Calculate energy consumption
    let energy_consumption = neural_result.energy_consumption + orchestration_result.energy_consumption;

    Ok(ProcessingResponse {
        response: orchestration_result.final_response,
        confidence: orchestration_result.confidence,
        processing_trace,
        quantum_states,
        thought_currents,
        energy_consumption,
        processing_time_ms: 0, // Will be set by caller
    })
}

/// Handle get processing history request
async fn handle_get_processing_history(
    params: HistoryParams,
    state: Arc<RwLock<ApiState>>,
) -> Result<impl Reply, warp::Rejection> {
    let state_guard = state.read().await;
    let orchestrator = state_guard.interface_manager.orchestrator();
    drop(state_guard);

    match get_processing_history_internal(orchestrator, &params).await {
        Ok(history) => Ok(warp::reply::json(&ApiResponse::success(history))),
        Err(e) => {
            log::error!("Failed to get processing history: {}", e);
            Ok(warp::reply::json(&ApiResponse::<()>::error(format!(
                "History retrieval failed: {}",
                e
            ))))
        },
    }
}

/// Handle get active processes request
async fn handle_get_active_processes(state: Arc<RwLock<ApiState>>) -> Result<impl Reply, warp::Rejection> {
    let state_guard = state.read().await;
    let orchestrator = state_guard.interface_manager.orchestrator();
    drop(state_guard);

    match get_active_processes_internal(orchestrator).await {
        Ok(processes) => Ok(warp::reply::json(&ApiResponse::success(processes))),
        Err(e) => {
            log::error!("Failed to get active processes: {}", e);
            Ok(warp::reply::json(&ApiResponse::<()>::error(format!(
                "Active processes retrieval failed: {}",
                e
            ))))
        },
    }
}

/// Handle cancel process request
async fn handle_cancel_process(
    process_id: String,
    state: Arc<RwLock<ApiState>>,
) -> Result<impl Reply, warp::Rejection> {
    let state_guard = state.read().await;
    let orchestrator = state_guard.interface_manager.orchestrator();
    drop(state_guard);

    match cancel_process_internal(orchestrator, &process_id).await {
        Ok(success) => {
            if success {
                Ok(warp::reply::json(&ApiResponse::success(CancelResponse {
                    process_id,
                    cancelled: true,
                    message: "Process successfully cancelled".to_string(),
                })))
            } else {
                Ok(warp::reply::json(&ApiResponse::<()>::error(
                    "Process not found or already completed".to_string(),
                )))
            }
        },
        Err(e) => {
            log::error!("Failed to cancel process {}: {}", process_id, e);
            Ok(warp::reply::json(&ApiResponse::<()>::error(format!(
                "Cancellation failed: {}",
                e
            ))))
        },
    }
}

/// Internal history retrieval logic
async fn get_processing_history_internal(
    orchestrator: Arc<RwLock<crate::metacognition::MetacognitiveOrchestrator>>,
    params: &HistoryParams,
) -> Result<ProcessingHistory, KambuzumaError> {
    let orchestrator_guard = orchestrator.read().await;

    // Get processing history from orchestrator
    let history_entries = orchestrator_guard
        .get_processing_history(params.limit.unwrap_or(50), params.offset.unwrap_or(0))
        .await?;

    let total_count = orchestrator_guard.get_total_processing_count().await?;

    Ok(ProcessingHistory {
        entries: history_entries
            .into_iter()
            .map(|entry| HistoryEntry {
                process_id: entry.process_id,
                query: entry.query,
                response: entry.response,
                timestamp: entry.timestamp,
                processing_time_ms: entry.processing_time_ms,
                confidence: entry.confidence,
                energy_consumption: entry.energy_consumption,
                status: entry.status,
            })
            .collect(),
        total_count,
        has_more: (params.offset.unwrap_or(0) + params.limit.unwrap_or(50)) < total_count as u32,
    })
}

/// Internal active processes retrieval logic
async fn get_active_processes_internal(
    orchestrator: Arc<RwLock<crate::metacognition::MetacognitiveOrchestrator>>,
) -> Result<ActiveProcesses, KambuzumaError> {
    let orchestrator_guard = orchestrator.read().await;

    let active_processes = orchestrator_guard.get_active_processes().await?;

    Ok(ActiveProcesses {
        processes: active_processes
            .into_iter()
            .map(|process| ActiveProcess {
                process_id: process.process_id,
                query: process.query,
                start_time: process.start_time,
                current_stage: process.current_stage,
                progress: process.progress,
                estimated_completion: process.estimated_completion,
            })
            .collect(),
        count: active_processes.len() as u32,
    })
}

/// Internal process cancellation logic
async fn cancel_process_internal(
    orchestrator: Arc<RwLock<crate::metacognition::MetacognitiveOrchestrator>>,
    process_id: &str,
) -> Result<bool, KambuzumaError> {
    let mut orchestrator_guard = orchestrator.write().await;
    orchestrator_guard.cancel_process(process_id).await
}

/// Query parameters for processing history
#[derive(Debug, Deserialize)]
struct HistoryParams {
    limit: Option<u32>,
    offset: Option<u32>,
    start_date: Option<chrono::DateTime<chrono::Utc>>,
    end_date: Option<chrono::DateTime<chrono::Utc>>,
}

/// Processing history response
#[derive(Debug, Serialize)]
struct ProcessingHistory {
    entries: Vec<HistoryEntry>,
    total_count: u64,
    has_more: bool,
}

/// Individual history entry
#[derive(Debug, Serialize)]
struct HistoryEntry {
    process_id: String,
    query: String,
    response: String,
    timestamp: chrono::DateTime<chrono::Utc>,
    processing_time_ms: u64,
    confidence: f64,
    energy_consumption: f64,
    status: String,
}

/// Active processes response
#[derive(Debug, Serialize)]
struct ActiveProcesses {
    processes: Vec<ActiveProcess>,
    count: u32,
}

/// Individual active process
#[derive(Debug, Serialize)]
struct ActiveProcess {
    process_id: String,
    query: String,
    start_time: chrono::DateTime<chrono::Utc>,
    current_stage: u8,
    progress: f64,
    estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
}

/// Process cancellation response
#[derive(Debug, Serialize)]
struct CancelResponse {
    process_id: String,
    cancelled: bool,
    message: String,
}
