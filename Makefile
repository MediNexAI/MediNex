.PHONY: help install dev test clean build deploy

# Default target
help:
	@echo "Available targets:"
	@echo "  help          - Show this help message"
	@echo "  install       - Install dependencies for all components"
	@echo "  dev           - Start development environment"
	@echo "  test          - Run all tests"
	@echo "  clean         - Clean up build artifacts"
	@echo "  build         - Build all components for production"
	@echo "  deploy-devnet - Deploy contracts to Solana devnet"
	@echo "  deploy-mainnet - Deploy contracts to Solana mainnet"

# Install dependencies
install:
	@echo "Installing dependencies..."
	cd apps/frontend && npm install
	cd apps/backend && npm install
	cd contracts && cargo build
	cd ai && pip install -r requirements.txt

# Start development environment
dev:
	@echo "Starting development environment..."
	docker-compose up

# Run tests
test:
	@echo "Running tests..."
	cd apps/frontend && npm test
	cd apps/backend && npm test
	cd contracts && cargo test
	cd ai && python -m pytest

# Clean up build artifacts
clean:
	@echo "Cleaning up build artifacts..."
	cd apps/frontend && npm run clean
	cd apps/backend && npm run clean
	cd contracts && cargo clean
	rm -rf build/
	docker-compose down -v

# Build for production
build:
	@echo "Building for production..."
	cd apps/frontend && npm run build
	cd apps/backend && npm run build
	cd contracts && cargo build --release
	cd ai && python -m build

# Deploy to Solana devnet
deploy-devnet:
	@echo "Deploying to Solana devnet..."
	cd scripts/deployment && node deploy_contracts.js --network devnet

# Deploy to Solana mainnet
deploy-mainnet:
	@echo "Deploying to Solana mainnet..."
	@echo "Are you sure you want to deploy to mainnet? [y/N]" && read ans && [ $${ans:-N} = y ]
	cd scripts/deployment && node deploy_contracts.js --network mainnet 