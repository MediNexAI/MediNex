#!/bin/bash
# MediNex AI Development Environment Setup Script
# This script sets up the development environment for MediNex AI

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Setting up MediNex AI development environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
  echo "Error: Python 3.9+ is required. Found Python $PYTHON_VERSION"
  exit 1
fi

echo "Using Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$PROJECT_ROOT/.venv"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"

# Development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov pylint black isort mypy

# Set up pre-commit hooks if git is available
if command -v git &> /dev/null && [ -d "$PROJECT_ROOT/.git" ]; then
  echo "Setting up git hooks..."
  
  # Create pre-commit hook script
  HOOK_FILE="$PROJECT_ROOT/.git/hooks/pre-commit"
  mkdir -p "$(dirname "$HOOK_FILE")"
  
  cat > "$HOOK_FILE" << 'EOF'
#!/bin/bash
set -e

# Activate virtual environment
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Run linting and formatting checks
echo "Running code formatting checks..."
python -m black --check ai tests
python -m isort --check ai tests

# Run basic tests
echo "Running basic tests..."
python -m pytest -xvs tests/unit

# If we got here, all checks passed
echo "All pre-commit checks passed!"
EOF
  
  chmod +x "$HOOK_FILE"
  echo "Git hooks installed"
fi

# Copy development environment file if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$PROJECT_ROOT/.env.development" ]; then
  echo "Setting up environment variables..."
  cp "$PROJECT_ROOT/.env.development" "$PROJECT_ROOT/.env"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p "$PROJECT_ROOT/data/knowledge"
mkdir -p "$PROJECT_ROOT/data/sample"
mkdir -p "$PROJECT_ROOT/logs"

# Initialize the system
echo "Initializing the system..."
python "$PROJECT_ROOT/app.py" init

echo "Development environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the API server, run:"
echo "  python app.py serve"
echo ""
echo "Happy coding!" 