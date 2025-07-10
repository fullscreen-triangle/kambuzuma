# Kambuzuma Biological Quantum Computing Build Automation
# Optimized development workflow for biological quantum systems

# Default recipe - show available commands
default:
    @just --list

# Development commands
dev:
    @echo "🧬 Starting biological quantum computing development mode..."
    cargo watch -x "check --all-features" -x "test --all-features" -x "run"

# Build commands
build:
    @echo "🔬 Building biological quantum computing system..."
    cargo build --all-features

build-release:
    @echo "🚀 Building optimized biological quantum computing system..."
    cargo build --release --all-features

build-biological:
    @echo "🧬 Building with biological validation profile..."
    cargo build --profile biological --all-features

build-quantum:
    @echo "⚛️  Building with quantum optimization profile..."
    cargo build --profile quantum --all-features

# Testing commands
test:
    @echo "🧪 Running biological quantum computing tests..."
    cargo test --all-features

test-release:
    @echo "🧪 Running optimized biological quantum computing tests..."
    cargo test --release --all-features

test-biological:
    @echo "🧬 Running biological validation tests..."
    cargo test biological_validation --all-features

test-quantum:
    @echo "⚛️  Running quantum coherence tests..."
    cargo test quantum_coherence --all-features

test-atp:
    @echo "🔋 Running ATP constraint tests..."
    cargo test atp_constraints --all-features

test-integration:
    @echo "🔗 Running integration tests..."
    cargo test --test '*' --all-features

# Code quality commands
fmt:
    @echo "✨ Formatting biological quantum computing code..."
    cargo fmt --all

fmt-check:
    @echo "🔍 Checking code formatting..."
    cargo fmt --all -- --check

clippy:
    @echo "📎 Running Clippy analysis..."
    cargo clippy --all-targets --all-features -- -D warnings

clippy-fix:
    @echo "🔧 Auto-fixing Clippy issues..."
    cargo clippy --all-targets --all-features --fix --allow-dirty

# Documentation commands
doc:
    @echo "📚 Building documentation..."
    cargo doc --all-features --no-deps

doc-open:
    @echo "🌐 Opening documentation in browser..."
    cargo doc --all-features --no-deps --open

# Benchmarking commands
bench:
    @echo "⚡ Running biological quantum computing benchmarks..."
    cargo bench --all-features

bench-quantum:
    @echo "⚛️  Running quantum processing benchmarks..."
    cargo bench quantum --all-features

bench-biological:
    @echo "🧬 Running biological validation benchmarks..."
    cargo bench biological --all-features

# Security and auditing commands
audit:
    @echo "🔐 Running security audit..."
    cargo audit

deny:
    @echo "🚫 Running dependency deny check..."
    cargo deny check

# Coverage commands
coverage:
    @echo "📊 Generating code coverage report..."
    cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
    cargo llvm-cov --all-features --workspace --html
    @echo "Coverage report generated in target/llvm-cov/html/"

coverage-open:
    @echo "🌐 Opening coverage report in browser..."
    cargo llvm-cov --all-features --workspace --html --open

# Performance profiling commands
profile:
    @echo "📈 Profiling biological quantum computing system..."
    cargo build --release --all-features
    perf record -g target/release/kambuzuma
    perf report

profile-quantum:
    @echo "⚛️  Profiling quantum processing performance..."
    cargo build --profile quantum --all-features
    perf record -g target/quantum/kambuzuma
    perf report

# Environment setup commands
setup:
    @echo "🛠️  Setting up development environment..."
    rustup component add rustfmt clippy llvm-tools-preview
    cargo install cargo-audit cargo-deny cargo-llvm-cov cargo-watch

setup-all:
    @echo "🔧 Complete development environment setup..."
    rustup component add rustfmt clippy llvm-tools-preview rust-analyzer
    cargo install cargo-audit cargo-deny cargo-llvm-cov cargo-watch criterion
    cargo install flamegraph cargo-expand cargo-udeps

# Validation commands
validate:
    @echo "✅ Running complete validation suite..."
    just fmt-check
    just clippy
    just test
    just test-biological
    just test-quantum
    just test-atp
    just audit
    just deny

validate-fast:
    @echo "⚡ Running fast validation..."
    just fmt-check
    just clippy
    just test

# Clean commands
clean:
    @echo "🧹 Cleaning build artifacts..."
    cargo clean

clean-all:
    @echo "🧹 Deep cleaning all artifacts..."
    cargo clean
    rm -rf target/
    rm -rf .cargo/
    rm -f Cargo.lock

# Run commands
run:
    @echo "🚀 Running biological quantum computing system..."
    cargo run --all-features

run-release:
    @echo "🚀 Running optimized biological quantum computing system..."
    cargo run --release --all-features

run-biological:
    @echo "🧬 Running with biological validation profile..."
    cargo run --profile biological --all-features

run-quantum:
    @echo "⚛️  Running with quantum optimization profile..."
    cargo run --profile quantum --all-features

# Example/demo commands
demo:
    @echo "🎬 Running demonstration scenarios..."
    cargo run --bin kambuzuma --all-features

demo-quantum:
    @echo "⚛️  Running quantum tunneling demonstration..."
    cargo run --bin kambuzuma --all-features -- --demo quantum

demo-biological:
    @echo "🧬 Running biological validation demonstration..."
    cargo run --bin kambuzuma --all-features -- --demo biological

# Utility commands
check:
    @echo "🔍 Checking project..."
    cargo check --all-features

check-all:
    @echo "🔍 Comprehensive project check..."
    cargo check --all-targets --all-features

expand:
    @echo "🔍 Expanding macros..."
    cargo expand

tree:
    @echo "🌳 Showing dependency tree..."
    cargo tree --all-features

outdated:
    @echo "📦 Checking for outdated dependencies..."
    cargo outdated

# Development workflow
workflow:
    @echo "🔄 Running complete development workflow..."
    just clean
    just setup
    just validate
    just build-release
    just doc
    just bench
    @echo "✅ Development workflow complete!"

# Production preparation
prepare-production:
    @echo "🚀 Preparing for production deployment..."
    just clean
    just validate
    just build-release
    just test-release
    just bench
    just doc
    @echo "✅ Production preparation complete!" 