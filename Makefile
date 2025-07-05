# Benguela Biological Quantum Computing - Deployment Makefile
.PHONY: help build test deploy clean setup-dev install-deps check-hardware

# Default target
help:
	@echo "Benguela Biological Quantum Computing System"
	@echo "============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  setup-dev      - Set up development environment"
	@echo "  install-deps   - Install all dependencies"
	@echo "  build          - Build all components"
	@echo "  test           - Run all tests"
	@echo "  test-hardware  - Test hardware interfaces"
	@echo "  check-hardware - Check hardware connections"
	@echo "  deploy         - Deploy to production"
	@echo "  deploy-dev     - Deploy development environment"
	@echo "  start          - Start all services"
	@echo "  stop           - Stop all services"
	@echo "  logs           - View logs"
	@echo "  clean          - Clean build artifacts"
	@echo "  safety-check   - Run safety validation"
	@echo "  calibrate      - Run hardware calibration"
	@echo "  backup         - Create system backup"
	@echo "  restore        - Restore from backup"

# Development setup
setup-dev:
	@echo "Setting up Benguela development environment..."
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	source ~/.cargo/env
	rustup component add clippy rustfmt
	python3 -m venv venv
	source venv/bin/activate && pip install --upgrade pip setuptools wheel
	source venv/bin/activate && pip install -r requirements.txt
	source venv/bin/activate && pip install -e .
	@echo "Development environment ready!"

# Install dependencies
install-deps:
	@echo "Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y \
		build-essential \
		pkg-config \
		libssl-dev \
		libusb-1.0-0-dev \
		libftdi1-dev \
		libudev-dev \
		python3-dev \
		python3-pip \
		docker.io \
		docker-compose
	pip install -r requirements.txt
	@echo "Dependencies installed!"

# Build
build:
	@echo "Building Rust components..."
	cargo build --release --features production
	@echo "Building Python components..."
	source venv/bin/activate && python setup.py build
	@echo "Building Docker images..."
	docker-compose build
	@echo "Build complete!"

# Testing
test:
	@echo "Running Rust tests..."
	cargo test --release
	@echo "Running Python tests..."
	source venv/bin/activate && pytest tests/ -v --cov=src/
	@echo "Running integration tests..."
	docker-compose -f docker-compose.test.yml up --abort-on-container-exit
	@echo "All tests passed!"

test-hardware:
	@echo "Testing hardware interfaces..."
	source venv/bin/activate && python -m benguela_py.hardware.test_all
	cargo test --release --features hardware
	@echo "Hardware tests complete!"

check-hardware:
	@echo "Checking hardware connections..."
	lsusb
	ls -la /dev/tty*
	source venv/bin/activate && python -c "import benguela_py.hardware; benguela_py.hardware.detect_all()"
	@echo "Hardware check complete!"

# Deployment
deploy: build safety-check
	@echo "Deploying Benguela to production..."
	docker-compose down
	docker-compose up -d
	@echo "Waiting for services to start..."
	sleep 30
	@echo "Running deployment verification..."
	curl -f http://localhost:8080/health || (echo "Deployment failed!" && exit 1)
	@echo "Production deployment complete!"

deploy-dev:
	@echo "Deploying development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Development environment deployed!"

# Service management
start:
	@echo "Starting Benguela services..."
	docker-compose up -d
	@echo "Services started!"

stop:
	@echo "Stopping Benguela services..."
	docker-compose down
	@echo "Services stopped!"

logs:
	docker-compose logs -f

# Safety and validation
safety-check:
	@echo "Running safety validation..."
	source venv/bin/activate && python -m benguela_py.safety.validate_all
	@echo "Checking emergency shutdown systems..."
	source venv/bin/activate && python -c "import benguela_py.safety; benguela_py.safety.test_emergency_shutdown()"
	@echo "Safety checks passed!"

calibrate:
	@echo "Running hardware calibration..."
	docker-compose exec benguela-core benguela-cli calibrate --full
	@echo "Calibration complete!"

# Backup and restore
backup:
	@echo "Creating system backup..."
	mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	docker-compose exec postgres pg_dump -U benguela benguela > backups/$(shell date +%Y%m%d_%H%M%S)/database.sql
	cp -r quantum_states backups/$(shell date +%Y%m%d_%H%M%S)/
	cp -r config backups/$(shell date +%Y%m%d_%H%M%S)/
	tar -czf backups/benguela_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz backups/$(shell date +%Y%m%d_%H%M%S)/
	@echo "Backup created!"

restore:
	@echo "Restoring from backup..."
	@read -p "Enter backup timestamp (YYYYMMDD_HHMMSS): " timestamp; \
	tar -xzf backups/benguela_backup_$$timestamp.tar.gz -C backups/; \
	docker-compose exec postgres psql -U benguela -c "DROP DATABASE IF EXISTS benguela;"; \
	docker-compose exec postgres psql -U benguela -c "CREATE DATABASE benguela;"; \
	docker-compose exec postgres psql -U benguela benguela < backups/$$timestamp/database.sql; \
	cp -r backups/$$timestamp/quantum_states .; \
	cp -r backups/$$timestamp/config .
	@echo "Restore complete!"

# Monitoring and diagnostics
monitor:
	@echo "Opening monitoring dashboard..."
	xdg-open http://localhost:3000  # Grafana
	xdg-open http://localhost:9090  # Prometheus

jupyter:
	@echo "Starting Jupyter Lab..."
	docker-compose exec jupyter jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	docker system prune -f
	@echo "Cleanup complete!"

# Development utilities
format:
	@echo "Formatting code..."
	cargo fmt
	source venv/bin/activate && black src/ tests/
	source venv/bin/activate && isort src/ tests/
	@echo "Code formatted!"

lint:
	@echo "Running linters..."
	cargo clippy --all-targets --all-features -- -D warnings
	source venv/bin/activate && flake8 src/ tests/
	source venv/bin/activate && mypy src/
	@echo "Linting complete!"

# Performance testing
benchmark:
	@echo "Running performance benchmarks..."
	cargo bench
	source venv/bin/activate && pytest tests/benchmarks/ --benchmark-only
	@echo "Benchmarks complete!"

# Documentation
docs:
	@echo "Building documentation..."
	cargo doc --no-deps
	source venv/bin/activate && sphinx-build -b html docs/ docs/_build/
	@echo "Documentation built!"

# Emergency procedures
emergency-stop:
	@echo "EMERGENCY STOP - Shutting down all quantum operations..."
	docker-compose exec benguela-core benguela-cli emergency-stop
	docker-compose stop benguela-core
	@echo "Emergency stop complete!"

# Hardware-specific targets
patch-clamp-test:
	@echo "Testing patch-clamp interfaces..."
	source venv/bin/activate && python -m benguela_py.hardware.patch_clamp.test
	@echo "Patch-clamp test complete!"

quantum-calibration:
	@echo "Running quantum coherence calibration..."
	docker-compose exec benguela-core benguela-cli quantum-calibrate
	@echo "Quantum calibration complete!" 