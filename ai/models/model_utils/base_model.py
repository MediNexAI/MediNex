"""
Base Model Class

This module provides the abstract base class for all MediNex AI models.
"""

import os
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

# Import our config
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import MODELS_DIR, OUTPUT_DIR, load_model_config


class ModelMetadata(BaseModel):
    """Model metadata for tracking and blockchain registration."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of AI model")
    description: str = Field(..., description="Model description")
    input_shape: Tuple[int, ...] = Field(..., description="Input shape for the model")
    num_classes: int = Field(..., description="Number of output classes/values")
    accuracy: float = Field(0.0, description="Model accuracy from evaluation")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, 
                                                  description="Additional performance metrics")
    created_at: int = Field(default_factory=lambda: int(time.time()), 
                            description="Creation timestamp")
    updated_at: int = Field(default_factory=lambda: int(time.time()), 
                           description="Last update timestamp")
    model_hash: Optional[str] = Field(None, description="SHA-256 hash of the model weights")
    framework: str = Field("tensorflow", description="ML framework used")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            tuple: lambda x: list(x)
        }


class BaseModel(ABC):
    """
    Abstract base class for all MediNex AI models.
    
    This class defines the common interface and utility methods
    for medical imaging models.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """
        Initialize the base model.
        
        Args:
            model_id: Identifier for the model configuration
            **kwargs: Additional arguments to override configuration
        """
        self.model_id = model_id
        self.config = load_model_config(model_id)
        
        # Override config with any kwargs provided
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        # Initialize metadata
        self.metadata = ModelMetadata(
            name=self.config["name"],
            version=self.config["version"],
            model_type=self.config["type"],
            description=self.config.get("description", ""),
            input_shape=self.config["input_shape"],
            num_classes=self.config["num_classes"]
        )
        
        # The actual model instance will be defined in subclasses
        self.model = None
    
    @abstractmethod
    def build(self) -> None:
        """
        Build the model architecture.
        
        This method must be implemented by subclasses to define
        the specific model architecture.
        """
        pass
    
    @abstractmethod
    def compile(self) -> None:
        """
        Compile the model with optimizer, loss, and metrics.
        
        This method must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def train(self, 
              train_data: Any, 
              validation_data: Optional[Any] = None, 
              **kwargs) -> Dict[str, Any]:
        """
        Train the model on the given data.
        
        Args:
            train_data: Training data
            validation_data: Validation data
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history and results
        """
        pass
    
    @abstractmethod
    def predict(self, 
                data: Any, 
                **kwargs) -> Any:
        """
        Make predictions using the model.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                 test_data: Any, 
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data for evaluation
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model. If None, generates a path.
            
        Returns:
            Path where the model was saved
        """
        if filepath is None:
            # Generate a path based on model info
            model_dir = os.path.join(OUTPUT_DIR, 
                                     f"{self.metadata.name}_v{self.metadata.version}")
            os.makedirs(model_dir, exist_ok=True)
            filepath = os.path.join(model_dir, "model")
        
        # Compute model hash before saving
        self.metadata.model_hash = self._compute_model_hash()
        self.metadata.updated_at = int(time.time())
        
        # Save metadata alongside model
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'w') as f:
            f.write(self.metadata.json(indent=2))
        
        self._save_implementation(filepath)
        return filepath
    
    @abstractmethod
    def _save_implementation(self, filepath: str) -> None:
        """
        Implementation-specific save method.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        pass
    
    def _compute_model_hash(self) -> str:
        """
        Compute a SHA-256 hash of the model weights.
        
        Returns:
            Hex digest of the hash
        """
        # This is a simplified implementation
        # In a real implementation, we would need framework-specific
        # code to extract weights in a consistent format
        if self.model is None:
            return ""
        
        hasher = hashlib.sha256()
        
        # Handle TF models
        if isinstance(self.model, KerasModel):
            for weight in self.model.get_weights():
                hasher.update(weight.tobytes())
        
        # Handle PyTorch models
        elif isinstance(self.model, nn.Module):
            for param in self.model.parameters():
                hasher.update(param.data.cpu().numpy().tobytes())
        
        return hasher.hexdigest()
    
    def register_on_blockchain(self) -> Optional[str]:
        """
        Register the model on the blockchain.
        
        Returns:
            Transaction signature if successful, None otherwise
        """
        # This is a placeholder for blockchain integration
        # In a real implementation, we would interact with the
        # Solana contract via an SDK
        
        try:
            # Placeholder for actual blockchain registration
            print(f"Registering model on blockchain: {self.metadata.name} v{self.metadata.version}")
            print(f"Model hash: {self.metadata.model_hash}")
            
            # Return a placeholder transaction signature
            return "placeholder_txn_signature"
        except Exception as e:
            print(f"Failed to register model on blockchain: {e}")
            return None
    
    def summary(self) -> str:
        """
        Get a summary of the model.
        
        Returns:
            String containing model summary
        """
        if self.model is None:
            return "Model not built yet."
        
        # For TensorFlow models
        if isinstance(self.model, KerasModel):
            # Redirect model.summary() output to a string
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                self.model.summary()
            return f.getvalue()
            
        # For PyTorch models
        elif isinstance(self.model, nn.Module):
            return str(self.model)
        
        return "Unsupported model type for summary."


# TensorFlow specific base model
class TensorFlowBaseModel(BaseModel):
    """Base class for TensorFlow models."""
    
    def __init__(self, model_id: str, **kwargs):
        """Initialize TensorFlow model."""
        super().__init__(model_id, **kwargs)
        self.metadata.framework = "tensorflow"
    
    def _save_implementation(self, filepath: str) -> None:
        """Save TensorFlow model implementation."""
        if self.model is not None:
            self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'TensorFlowBaseModel':
        """Load TensorFlow model from disk."""
        # Load metadata
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Create instance with appropriate model_id
        instance = cls(metadata_dict.get("name", "unknown"))
        
        # Load model
        instance.model = tf.keras.models.load_model(filepath)
        
        # Update metadata
        instance.metadata = ModelMetadata(**metadata_dict)
        
        return instance


# PyTorch specific base model
class PyTorchBaseModel(BaseModel):
    """Base class for PyTorch models."""
    
    def __init__(self, model_id: str, **kwargs):
        """Initialize PyTorch model."""
        super().__init__(model_id, **kwargs)
        self.metadata.framework = "pytorch"
    
    def _save_implementation(self, filepath: str) -> None:
        """Save PyTorch model implementation."""
        if self.model is not None:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.config
            }, f"{filepath}.pt")
    
    @classmethod
    def load(cls, filepath: str) -> 'PyTorchBaseModel':
        """Load PyTorch model from disk."""
        # Load metadata
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Create instance with appropriate model_id
        instance = cls(metadata_dict.get("name", "unknown"))
        
        # Build the model architecture (required before loading weights)
        instance.build()
        
        # Load model weights
        checkpoint = torch.load(f"{filepath}.pt")
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Update metadata
        instance.metadata = ModelMetadata(**metadata_dict)
        
        return instance 