#!/bin/bash
# MediNex AI Deployment Script
# Usage: ./deploy.sh [environment]
# Example: ./deploy.sh production

set -e  # Exit on error

# Default to development if no environment is specified
ENVIRONMENT=${1:-development}
TIMESTAMP=$(date +%Y%m%d%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration based on environment
case "$ENVIRONMENT" in
  development)
    ENV_FILE=".env.development"
    CONFIG_FILE="config.development.json"
    ;;
  staging)
    ENV_FILE=".env.staging"
    CONFIG_FILE="config.staging.json"
    ;;
  production)
    ENV_FILE=".env.production"
    CONFIG_FILE="config.production.json"
    ;;
  *)
    echo "Error: Unknown environment '$ENVIRONMENT'"
    echo "Valid options: development, staging, production"
    exit 1
    ;;
esac

echo "Deploying MediNex AI to $ENVIRONMENT environment..."

# Verify required files
if [ ! -f "$PROJECT_ROOT/$ENV_FILE" ]; then
  echo "Error: Environment file '$ENV_FILE' not found"
  exit 1
fi

if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
  echo "Warning: Config file '$CONFIG_FILE' not found, using default config.json"
  CONFIG_FILE="config.json"
fi

# Create backup
echo "Creating backup of current deployment..."
BACKUP_DIR="$PROJECT_ROOT/backups/$ENVIRONMENT-$TIMESTAMP"
mkdir -p "$BACKUP_DIR"
if [ -d "$PROJECT_ROOT/data" ]; then
  echo "Backing up data directory..."
  cp -r "$PROJECT_ROOT/data" "$BACKUP_DIR/"
fi

# Setup environment
echo "Setting up environment..."
cp "$PROJECT_ROOT/$ENV_FILE" "$PROJECT_ROOT/.env"
cp "$PROJECT_ROOT/$CONFIG_FILE" "$PROJECT_ROOT/config.json"

# Build and deploy
echo "Building application..."
if [ -f "$PROJECT_ROOT/setup.py" ]; then
  cd "$PROJECT_ROOT"
  python3 -m pip install -e .
fi

# Initialize system if needed
echo "Initializing system..."
python3 "$PROJECT_ROOT/app.py" init

# Run database migrations if they exist
if [ -d "$PROJECT_ROOT/migrations" ]; then
  echo "Running database migrations..."
  python3 "$PROJECT_ROOT/app.py" migrate
fi

# Restart services
echo "Restarting services..."
if [ "$ENVIRONMENT" == "production" ]; then
  # For production, we might use systemd or Docker
  if command -v docker-compose &> /dev/null; then
    cd "$PROJECT_ROOT"
    docker-compose down
    docker-compose up -d
  elif command -v systemctl &> /dev/null; then
    systemctl restart medinex-api.service
  else
    echo "Warning: No service manager found. Please restart services manually."
  fi
else
  # For development/staging, just make sure the app is installed
  echo "Services need to be started manually in $ENVIRONMENT environment"
fi

echo "Deployment to $ENVIRONMENT completed successfully!"
echo "You can start the API server with: python3 app.py serve" 