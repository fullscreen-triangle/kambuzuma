use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;
use warp::ws::{Message, WebSocket};

use crate::errors::KambuzumaError;
use crate::interfaces::{InterfaceManager, MonitoringData, SystemStatus};

pub mod connection_manager;
pub mod event_broadcasting;
pub mod real_time_monitoring;
pub mod thought_current_stream;

/// WebSocket configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WebSocketConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub heartbeat_interval_seconds: u64,
    pub buffer_size: usize,
    pub compression_enabled: bool,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            host: "127.0.0.1".to_string(),
            port: 8081,
            max_connections: 1000,
            heartbeat_interval_seconds: 30,
            buffer_size: 1024,
            compression_enabled: true,
        }
    }
}

/// WebSocket server state
#[derive(Clone)]
pub struct WebSocketServer {
    interface_manager: InterfaceManager,
    config: WebSocketConfig,
    connection_manager: Arc<RwLock<connection_manager::ConnectionManager>>,
    event_broadcaster: Arc<event_broadcasting::EventBroadcaster>,
    monitoring_stream: Arc<real_time_monitoring::MonitoringStream>,
    thought_current_stream: Arc<thought_current_stream::ThoughtCurrentStream>,
}

impl WebSocketServer {
    /// Create new WebSocket server
    pub fn new(interface_manager: InterfaceManager, config: WebSocketConfig) -> Self {
        let (event_tx, _) = broadcast::channel(1000);

        Self {
            interface_manager: interface_manager.clone(),
            config: config.clone(),
            connection_manager: Arc::new(RwLock::new(connection_manager::ConnectionManager::new(
                config.max_connections,
            ))),
            event_broadcaster: Arc::new(event_broadcasting::EventBroadcaster::new(event_tx)),
            monitoring_stream: Arc::new(real_time_monitoring::MonitoringStream::new(interface_manager.clone())),
            thought_current_stream: Arc::new(thought_current_stream::ThoughtCurrentStream::new(
                interface_manager.clone(),
            )),
        }
    }

    /// Start the WebSocket server
    pub async fn start(&self) -> Result<(), KambuzumaError> {
        log::info!("Starting WebSocket server on {}:{}", self.config.host, self.config.port);

        let server = self.clone();

        // Start monitoring stream
        tokio::spawn({
            let monitoring_stream = self.monitoring_stream.clone();
            let event_broadcaster = self.event_broadcaster.clone();
            async move {
                monitoring_stream.start_streaming(event_broadcaster).await;
            }
        });

        // Start thought current stream
        tokio::spawn({
            let thought_current_stream = self.thought_current_stream.clone();
            let event_broadcaster = self.event_broadcaster.clone();
            async move {
                thought_current_stream.start_streaming(event_broadcaster).await;
            }
        });

        // Create WebSocket route
        let websocket_route = warp::path("ws")
            .and(warp::ws())
            .and(warp::any().map(move || server.clone()))
            .and_then(handle_websocket_upgrade);

        // Start server
        let addr = format!("{}:{}", self.config.host, self.config.port)
            .parse::<std::net::SocketAddr>()
            .map_err(|e| KambuzumaError::Interface(format!("Invalid WebSocket address: {}", e)))?;

        warp::serve(websocket_route).run(addr).await;

        Ok(())
    }
}

/// Handle WebSocket upgrade request
async fn handle_websocket_upgrade(
    ws: warp::ws::Ws,
    server: WebSocketServer,
) -> Result<impl warp::Reply, warp::Rejection> {
    Ok(ws.on_upgrade(move |socket| handle_websocket_connection(socket, server)))
}

/// Handle individual WebSocket connection
async fn handle_websocket_connection(websocket: WebSocket, server: WebSocketServer) {
    let connection_id = Uuid::new_v4().to_string();
    log::info!("New WebSocket connection: {}", connection_id);

    // Register connection
    {
        let mut manager = server.connection_manager.write().await;
        if let Err(e) = manager.add_connection(connection_id.clone(), websocket).await {
            log::error!("Failed to register WebSocket connection: {}", e);
            return;
        }
    }

    // Split socket into sender and receiver
    let (mut ws_sender, mut ws_receiver) = websocket.split();

    // Subscribe to events
    let mut event_receiver = server.event_broadcaster.subscribe().await;

    // Handle incoming messages
    let incoming_handler = {
        let connection_id = connection_id.clone();
        let server = server.clone();
        tokio::spawn(async move {
            while let Some(result) = ws_receiver.next().await {
                match result {
                    Ok(message) => {
                        if let Err(e) = handle_incoming_message(&connection_id, message, &server).await {
                            log::error!("Error handling incoming message: {}", e);
                            break;
                        }
                    },
                    Err(e) => {
                        log::error!("WebSocket error: {}", e);
                        break;
                    },
                }
            }
        })
    };

    // Handle outgoing events
    let outgoing_handler = {
        let connection_id = connection_id.clone();
        tokio::spawn(async move {
            while let Ok(event) = event_receiver.recv().await {
                let message = match serde_json::to_string(&event) {
                    Ok(json) => Message::text(json),
                    Err(e) => {
                        log::error!("Failed to serialize event: {}", e);
                        continue;
                    },
                };

                if let Err(e) = ws_sender.send(message).await {
                    log::error!("Failed to send WebSocket message: {}", e);
                    break;
                }
            }
        })
    };

    // Wait for either handler to complete
    tokio::select! {
        _ = incoming_handler => {},
        _ = outgoing_handler => {},
    }

    // Cleanup connection
    {
        let mut manager = server.connection_manager.write().await;
        manager.remove_connection(&connection_id).await;
    }

    log::info!("WebSocket connection closed: {}", connection_id);
}

/// Handle incoming WebSocket message
async fn handle_incoming_message(
    connection_id: &str,
    message: Message,
    server: &WebSocketServer,
) -> Result<(), KambuzumaError> {
    if message.is_text() {
        let text = message
            .to_str()
            .map_err(|_| KambuzumaError::Interface("Invalid UTF-8 in WebSocket message".to_string()))?;

        let request: WebSocketRequest = serde_json::from_str(text)
            .map_err(|e| KambuzumaError::Interface(format!("Invalid JSON in WebSocket message: {}", e)))?;

        handle_websocket_request(connection_id, request, server).await?;
    } else if message.is_ping() {
        // Pong will be sent automatically by warp
        log::debug!("Received ping from connection: {}", connection_id);
    } else if message.is_close() {
        log::info!("Received close message from connection: {}", connection_id);
    }

    Ok(())
}

/// Handle specific WebSocket request
async fn handle_websocket_request(
    connection_id: &str,
    request: WebSocketRequest,
    server: &WebSocketServer,
) -> Result<(), KambuzumaError> {
    match request {
        WebSocketRequest::Subscribe { event_types } => {
            let mut manager = server.connection_manager.write().await;
            manager.subscribe_to_events(connection_id, event_types).await?;
            log::debug!("Connection {} subscribed to events", connection_id);
        },
        WebSocketRequest::Unsubscribe { event_types } => {
            let mut manager = server.connection_manager.write().await;
            manager.unsubscribe_from_events(connection_id, event_types).await?;
            log::debug!("Connection {} unsubscribed from events", connection_id);
        },
        WebSocketRequest::GetStatus => {
            let status = get_current_status(&server.interface_manager).await?;
            let response = WebSocketResponse::Status { status };
            send_response_to_connection(connection_id, response, server).await?;
        },
        WebSocketRequest::StartThoughtCurrentStream { stage_ids } => {
            server
                .thought_current_stream
                .start_for_connection(connection_id, stage_ids)
                .await?;
            log::debug!("Started thought current stream for connection: {}", connection_id);
        },
        WebSocketRequest::StopThoughtCurrentStream => {
            server.thought_current_stream.stop_for_connection(connection_id).await?;
            log::debug!("Stopped thought current stream for connection: {}", connection_id);
        },
        WebSocketRequest::TriggerQuantumMeasurement { measurement_type } => {
            let result = trigger_quantum_measurement(&server.interface_manager, &measurement_type).await?;
            let response = WebSocketResponse::QuantumMeasurement { result };
            send_response_to_connection(connection_id, response, server).await?;
        },
    }

    Ok(())
}

/// Send response to specific connection
async fn send_response_to_connection(
    connection_id: &str,
    response: WebSocketResponse,
    server: &WebSocketServer,
) -> Result<(), KambuzumaError> {
    let manager = server.connection_manager.read().await;
    manager.send_to_connection(connection_id, response).await
}

/// Get current system status
async fn get_current_status(interface_manager: &InterfaceManager) -> Result<SystemStatus, KambuzumaError> {
    // Get orchestrator status
    let orchestrator_status = {
        let orchestrator = interface_manager.orchestrator().read().await;
        orchestrator.get_status_summary().await?
    };

    // Get neural processor status
    let neural_processor_status = {
        let neural = interface_manager.neural_processor().read().await;
        neural.get_status_summary().await?
    };

    // Get quantum system status
    let quantum_system_status = {
        let quantum = interface_manager.quantum_system().read().await;
        quantum.get_status_summary().await?
    };

    // Get additional metrics
    let active_stages = {
        let neural = interface_manager.neural_processor().read().await;
        neural.get_active_stages().await
    };

    let thought_currents = {
        let neural = interface_manager.neural_processor().read().await;
        neural.get_current_thought_currents().await
    };

    let quantum_coherence = {
        let quantum = interface_manager.quantum_system().read().await;
        quantum.get_coherence_level()
    };

    let energy_levels = {
        let quantum = interface_manager.quantum_system().read().await;
        quantum.get_energy_transfer_rate()
    };

    let processing_load = {
        let orchestrator = interface_manager.orchestrator().read().await;
        orchestrator.get_current_load().await?
    };

    Ok(SystemStatus {
        orchestrator_status,
        neural_processor_status,
        quantum_system_status,
        active_stages,
        thought_currents,
        quantum_coherence,
        energy_levels,
        processing_load,
    })
}

/// Trigger quantum measurement
async fn trigger_quantum_measurement(
    interface_manager: &InterfaceManager,
    measurement_type: &str,
) -> Result<QuantumMeasurementResult, KambuzumaError> {
    let mut quantum = interface_manager.quantum_system().write().await;

    let result = match measurement_type {
        "coherence" => quantum.measure_coherence(None).await?,
        "entanglement" => quantum.measure_entanglement(None).await?,
        "tunneling" => quantum.measure_tunneling_probability().await?,
        "energy" => quantum.measure_energy(None).await?,
        _ => {
            return Err(KambuzumaError::Quantum(format!(
                "Unknown measurement type: {}",
                measurement_type
            )))
        },
    };

    Ok(QuantumMeasurementResult {
        measurement_id: Uuid::new_v4().to_string(),
        measurement_type: measurement_type.to_string(),
        value: result.value,
        uncertainty: result.uncertainty,
        timestamp: chrono::Utc::now(),
    })
}

/// Start WebSocket server
pub async fn start_server(interface_manager: InterfaceManager, config: WebSocketConfig) -> Result<(), KambuzumaError> {
    let server = WebSocketServer::new(interface_manager, config);
    server.start().await
}

// WebSocket message types

/// Incoming WebSocket request
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketRequest {
    Subscribe { event_types: Vec<String> },
    Unsubscribe { event_types: Vec<String> },
    GetStatus,
    StartThoughtCurrentStream { stage_ids: Option<Vec<u8>> },
    StopThoughtCurrentStream,
    TriggerQuantumMeasurement { measurement_type: String },
}

/// Outgoing WebSocket response
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum WebSocketResponse {
    Status {
        status: SystemStatus,
    },
    MonitoringData {
        data: MonitoringData,
    },
    ThoughtCurrentUpdate {
        stage_id: u8,
        current_value: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    QuantumMeasurement {
        result: QuantumMeasurementResult,
    },
    SystemEvent {
        event: SystemEvent,
    },
    Error {
        message: String,
        code: Option<u32>,
    },
}

/// System event notification
#[derive(Debug, Serialize)]
pub struct SystemEvent {
    pub event_id: String,
    pub event_type: String,
    pub component: String,
    pub severity: String,
    pub message: String,
    pub metadata: Option<serde_json::Value>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Quantum measurement result
#[derive(Debug, Serialize)]
pub struct QuantumMeasurementResult {
    pub measurement_id: String,
    pub measurement_type: String,
    pub value: f64,
    pub uncertainty: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
