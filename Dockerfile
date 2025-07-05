# Benguela Biological Quantum Computing - Production Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Rust Build Environment
FROM rust:1.75-bullseye as rust-builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY benguela-*/Cargo.toml ./benguela-*/
COPY benguela-*/src/ ./benguela-*/src/

# Build optimized Rust binaries
RUN cargo build --release --features production

# Stage 2: Python Scientific Environment
FROM python:3.11-bullseye as python-builder

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    liblapack-dev \
    libblas-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libopenmpi-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src/python/ ./src/python/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Stage 3: Hardware Drivers and System Integration
FROM ubuntu:22.04 as hardware-base

# Install hardware interface dependencies
RUN apt-get update && apt-get install -y \
    # USB and serial interfaces
    libusb-1.0-0-dev \
    libftdi1-dev \
    udev \
    # National Instruments DAQmx
    libnidaqmx-dev \
    # Scientific instrument interfaces
    libvisa-dev \
    # Networking
    curl \
    # Process management
    supervisor \
    # Monitoring
    prometheus-node-exporter \
    && rm -rf /var/lib/apt/lists/*

# Stage 4: Final Production Image
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Core system
    ca-certificates \
    curl \
    # Python runtime
    python3.11 \
    python3.11-venv \
    # Scientific libraries runtime
    liblapack3 \
    libblas3 \
    libhdf5-103 \
    libnetcdf19 \
    libopenmpi3 \
    # Hardware interfaces
    libusb-1.0-0 \
    libftdi1-2 \
    # Process management
    supervisor \
    # Monitoring
    prometheus-node-exporter \
    # Debugging and maintenance
    htop \
    strace \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash benguela

# Copy compiled Rust binaries
COPY --from=rust-builder /app/target/release/benguela /usr/local/bin/
COPY --from=rust-builder /app/target/release/benguela-cli /usr/local/bin/
COPY --from=rust-builder /app/target/release/benguela-daemon /usr/local/bin/

# Copy Python environment
COPY --from=python-builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy hardware drivers and configurations
COPY --from=hardware-base /usr/lib/x86_64-linux-gnu/libusb* /usr/lib/x86_64-linux-gnu/
COPY --from=hardware-base /usr/lib/x86_64-linux-gnu/libftdi* /usr/lib/x86_64-linux-gnu/

# Create application directories
RUN mkdir -p /opt/benguela/{config,data,logs,quantum_states,hardware} \
    && chown -R benguela:benguela /opt/benguela

# Copy configuration files
COPY config/ /opt/benguela/config/
COPY hardware/ /opt/benguela/hardware/
COPY protocols/ /opt/benguela/protocols/

# Copy supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/benguela.conf

# Set up hardware permissions
COPY docker/99-benguela-hardware.rules /etc/udev/rules.d/
RUN usermod -a -G dialout,plugdev benguela

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables
ENV BENGUELA_CONFIG_PATH=/opt/benguela/config
ENV BENGUELA_DATA_PATH=/opt/benguela/data
ENV BENGUELA_LOG_LEVEL=info
ENV RUST_LOG=benguela=info
ENV PYTHONPATH=/opt/benguela
ENV PATH="/usr/local/bin:${PATH}"

# Expose ports
EXPOSE 8080 8081 8082 9090

# Switch to application user
USER benguela
WORKDIR /opt/benguela

# Default command
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/benguela.conf"] 