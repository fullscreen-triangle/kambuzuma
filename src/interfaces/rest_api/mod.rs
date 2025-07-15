use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use warp::{Filter, Reply};

use super::InterfaceManager;
use crate::errors::KambuzumaError;

pub mod endpoints;
pub mod middleware;
pub mod serializers;

/// REST API configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RestApiConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub max_request_size: usize,
    pub timeout: Duration,
    pub cors_enabled: bool,
    pub rate_limit: RateLimitConfig,
    pub auth: AuthConfig,
}

impl Default for RestApiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_request_size: 1024 * 1024, // 1MB
            timeout: Duration::from_secs(30),
            cors_enabled: true,
            rate_limit: RateLimitConfig::default(),
            auth: AuthConfig::default(),
        }
    }
}

/// Rate limiting configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub requests_per_minute: u32,
    pub burst_size: u32,
    pub window_size: Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_minute: 60,
            burst_size: 10,
            window_size: Duration::from_secs(60),
        }
    }
}

/// Authentication configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthConfig {
    pub enabled: bool,
    pub jwt_secret: String,
    pub token_expiry: Duration,
    pub require_auth_for_monitoring: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for development
            jwt_secret: "kambuzuma-secret-key".to_string(),
            token_expiry: Duration::from_secs(3600), // 1 hour
            require_auth_for_monitoring: false,
        }
    }
}

/// Start the REST API server
pub async fn start_server(interface_manager: InterfaceManager, config: RestApiConfig) -> Result<(), KambuzumaError> {
    log::info!("Starting REST API server on {}:{}", config.host, config.port);

    // Create shared state
    let state = Arc::new(RwLock::new(ApiState {
        interface_manager: interface_manager.clone(),
        config: config.clone(),
        request_count: 0,
        start_time: chrono::Utc::now(),
    }));

    // Build API routes
    let api_routes = build_api_routes(state.clone()).await;

    // Apply middleware
    let routes = api_routes
        .with(middleware::cors::cors_handler(config.cors_enabled))
        .with(middleware::logging::request_logger())
        .with(middleware::authentication::auth_middleware(config.auth.clone()))
        .with(middleware::rate_limiting::rate_limiter(config.rate_limit.clone()))
        .recover(handle_rejection);

    // Start server
    let addr = format!("{}:{}", config.host, config.port)
        .parse::<std::net::SocketAddr>()
        .map_err(|e| KambuzumaError::Interface(format!("Invalid address: {}", e)))?;

    warp::serve(routes).run(addr).await;

    Ok(())
}

/// API state shared across requests
#[derive(Clone)]
pub struct ApiState {
    pub interface_manager: InterfaceManager,
    pub config: RestApiConfig,
    pub request_count: u64,
    pub start_time: chrono::DateTime<chrono::Utc>,
}

/// Build all API routes
async fn build_api_routes(
    state: Arc<RwLock<ApiState>>,
) -> impl Filter<Extract = impl Reply, Error = warp::Rejection> + Clone {
    // Health check endpoint
    let health = warp::path("health")
        .and(warp::get())
        .and(with_state(state.clone()))
        .and_then(endpoints::health::health_check);

    // Status endpoint
    let status = warp::path("status")
        .and(warp::get())
        .and(with_state(state.clone()))
        .and_then(endpoints::monitoring::get_status);

    // Orchestration endpoints
    let orchestration = warp::path("orchestration").and(
        endpoints::orchestration::process_query(state.clone())
            .or(endpoints::orchestration::get_processing_history(state.clone()))
            .or(endpoints::orchestration::get_active_processes(state.clone()))
            .or(endpoints::orchestration::cancel_process(state.clone())),
    );

    // Quantum state endpoints
    let quantum = warp::path("quantum").and(
        endpoints::quantum_states::get_membrane_state(state.clone())
            .or(endpoints::quantum_states::get_tunneling_state(state.clone()))
            .or(endpoints::quantum_states::get_oscillation_state(state.clone()))
            .or(endpoints::quantum_states::get_entanglement_state(state.clone()))
            .or(endpoints::quantum_states::trigger_measurement(state.clone())),
    );

    // Neural processing endpoints
    let neural = warp::path("neural").and(
        endpoints::neural_processing::get_stage_status(state.clone())
            .or(endpoints::neural_processing::get_thought_currents(state.clone()))
            .or(endpoints::neural_processing::get_neuron_states(state.clone()))
            .or(endpoints::neural_processing::trigger_stage_reset(state.clone())),
    );

    // Monitoring endpoints
    let monitoring = warp::path("monitoring").and(
        endpoints::monitoring::get_metrics(state.clone())
            .or(endpoints::monitoring::get_performance_data(state.clone()))
            .or(endpoints::monitoring::get_system_logs(state.clone()))
            .or(endpoints::monitoring::get_alerts(state.clone())),
    );

    // Configuration endpoints
    let configuration = warp::path("config").and(
        endpoints::configuration::get_config(state.clone())
            .or(endpoints::configuration::update_config(state.clone()))
            .or(endpoints::configuration::reset_config(state.clone()))
            .or(endpoints::configuration::export_config(state.clone())),
    );

    // Combine all routes under /api/v1
    warp::path("api").and(warp::path("v1")).and(
        health
            .or(status)
            .or(orchestration)
            .or(quantum)
            .or(neural)
            .or(monitoring)
            .or(configuration),
    )
}

/// Helper to inject state into handlers
fn with_state(
    state: Arc<RwLock<ApiState>>,
) -> impl Filter<Extract = (Arc<RwLock<ApiState>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || state.clone())
}

/// Handle API rejections and convert to appropriate HTTP responses
async fn handle_rejection(err: warp::Rejection) -> Result<impl Reply, std::convert::Infallible> {
    let code;
    let message;

    if err.is_not_found() {
        code = warp::http::StatusCode::NOT_FOUND;
        message = "Endpoint not found";
    } else if let Some(_) = err.find::<warp::filters::body::BodyDeserializeError>() {
        code = warp::http::StatusCode::BAD_REQUEST;
        message = "Invalid request body";
    } else if let Some(_) = err.find::<warp::reject::PayloadTooLarge>() {
        code = warp::http::StatusCode::PAYLOAD_TOO_LARGE;
        message = "Request payload too large";
    } else if let Some(_) = err.find::<warp::reject::MethodNotAllowed>() {
        code = warp::http::StatusCode::METHOD_NOT_ALLOWED;
        message = "Method not allowed";
    } else {
        log::error!("Unhandled rejection: {:?}", err);
        code = warp::http::StatusCode::INTERNAL_SERVER_ERROR;
        message = "Internal server error";
    }

    let json = warp::reply::json(&serde_json::json!({
        "error": message,
        "code": code.as_u16()
    }));

    Ok(warp::reply::with_status(json, code))
}

/// API response wrapper
#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Pagination parameters
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub sort_by: Option<String>,
    pub sort_order: Option<String>,
}

impl Default for PaginationParams {
    fn default() -> Self {
        Self {
            page: Some(1),
            limit: Some(50),
            sort_by: Some("timestamp".to_string()),
            sort_order: Some("desc".to_string()),
        }
    }
}
