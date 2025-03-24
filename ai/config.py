"""
MediNex AI - Configuration Settings

This module defines configuration settings for the AI models, training, and evaluation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MEDICAL_IMAGING_DIR = MODELS_DIR / "medical_imaging"
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, OUTPUT_DIR, LOGS_DIR, MEDICAL_IMAGING_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model Parameters
MODEL_PARAMS = {
    "lung_ct": {
        "name": "LungCTAnalysis",
        "version": "1.0.0",
        "type": "medical_imaging",
        "input_shape": (512, 512, 1),
        "num_classes": 2,  # Binary classification for nodule detection
        "learning_rate": 0.0001,
        "batch_size": 16,
        "epochs": 100,
        "patience": 15,  # Early stopping patience
        "description": "CT scan analysis model for lung nodule detection"
    },
    "chest_xray": {
        "name": "ChestXRayAnalysis",
        "version": "1.0.0",
        "type": "medical_imaging",
        "input_shape": (448, 448, 3),
        "num_classes": 14,  # Common chest X-ray conditions
        "learning_rate": 0.0002,
        "batch_size": 24,
        "epochs": 80,
        "patience": 10,
        "description": "Chest X-ray analysis for multiple conditions"
    }
}

# Training Configuration
TRAINING_CONFIG = {
    "validation_split": 0.2,
    "test_split": 0.1,
    "augmentation": True,
    "use_mixed_precision": True,
    "checkpoints_to_keep": 3,
    "monitor_metric": "val_accuracy",
    "save_best_only": True
}

# Evaluation Metrics
EVALUATION_METRICS = [
    "accuracy", 
    "precision", 
    "recall", 
    "f1_score", 
    "auc", 
    "specificity", 
    "sensitivity"
]

# Blockchain Integration
BLOCKCHAIN_CONFIG = {
    "network": os.getenv("SOLANA_NETWORK", "devnet"),
    "program_id": os.getenv("PROGRAM_ID", "MdNxToKenxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
    "rpc_url": os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com"),
    "keyfile_path": os.getenv("KEYFILE_PATH", "~/.config/solana/id.json")
}

# Serialization Options
SERIALIZATION_CONFIG = {
    "formats": ["onnx", "savedmodel", "torchscript"],
    "optimization_level": 2,
    "include_metadata": True,
    "compression": True
}

# Load model-specific settings
def load_model_config(model_id: str) -> Dict[str, Any]:
    """
    Load model-specific configuration from MODEL_PARAMS.
    
    Args:
        model_id: Identifier for the model configuration
        
    Returns:
        Dictionary containing model configuration parameters
    """
    if model_id not in MODEL_PARAMS:
        raise ValueError(f"Unknown model ID: {model_id}. Available models: {list(MODEL_PARAMS.keys())}")
    
    return MODEL_PARAMS[model_id]

# Save configuration to file
def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration dictionary to a JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

# Load configuration from file
def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

# Environment-specific settings
ENV = os.getenv("MEDINEX_ENV", "development")

if ENV == "production":
    DEBUG = False
    # Add production-specific settings
elif ENV == "staging":
    DEBUG = True
    # Add staging-specific settings
else:  # development
    DEBUG = True
    # Add development-specific settings

# Export all settings
__all__ = [
    'BASE_DIR', 'MODELS_DIR', 'MEDICAL_IMAGING_DIR', 'DATA_DIR', 
    'OUTPUT_DIR', 'LOGS_DIR', 'MODEL_PARAMS', 'TRAINING_CONFIG', 
    'EVALUATION_METRICS', 'BLOCKCHAIN_CONFIG', 'SERIALIZATION_CONFIG',
    'load_model_config', 'save_config', 'load_config', 'ENV', 'DEBUG'
] 