#!/bin/bash

# Set environment variables (can be overridden by .env file)
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-4"
export LLM_TEMPERATURE="0.1"
export KNOWLEDGE_BASE_PATH="data/knowledge"

# Create required directories
mkdir -p data/knowledge
mkdir -p output/visualizations

# Check if .env file exists and load it
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a
    source .env
    set +a
fi

# Check if API key is set
if [ -z "$LLM_API_KEY" ]; then
    echo "Warning: LLM_API_KEY environment variable is not set. Set it in .env file or export it manually."
fi

# Install dependencies if needed
if [ "$1" == "--install" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the API
echo "Starting MediNex AI API on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
python -m api.main 