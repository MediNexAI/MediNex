"""
MediNex AI - Data Serialization Module

This module provides utilities for serializing and deserializing
medical data, model inputs/outputs, and configurations, ensuring
consistent data formats across the system.
"""

import json
import pickle
import base64
import io
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

class MedicalDataSerializer:
    """
    Utilities for serializing and deserializing medical data.
    
    Supports various data types including:
    - Medical images
    - Patient records
    - Model predictions
    - Configuration settings
    """
    
    @staticmethod
    def serialize_image(image_path: str) -> Dict[str, Any]:
        """
        Serialize an image to a dictionary representation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with serialized image data
        """
        try:
            with Image.open(image_path) as img:
                img_format = img.format
                img_mode = img.mode
                
                # Convert to bytes
                buffer = io.BytesIO()
                img.save(buffer, format=img_format)
                img_bytes = buffer.getvalue()
                
                # Convert to base64 for JSON compatibility
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                return {
                    'format': img_format,
                    'mode': img_mode,
                    'width': img.width,
                    'height': img.height,
                    'data': img_b64
                }
        except Exception as e:
            raise ValueError(f"Error serializing image: {str(e)}")
    
    @staticmethod
    def deserialize_image(image_data: Dict[str, Any]) -> Image.Image:
        """
        Deserialize an image from its dictionary representation.
        
        Args:
            image_data: Dictionary with serialized image data
            
        Returns:
            PIL Image object
        """
        try:
            # Decode base64 data
            img_bytes = base64.b64decode(image_data['data'])
            
            # Create image from bytes
            buffer = io.BytesIO(img_bytes)
            img = Image.open(buffer)
            
            return img
        except Exception as e:
            raise ValueError(f"Error deserializing image: {str(e)}")
    
    @staticmethod
    def serialize_numpy_array(array: np.ndarray) -> Dict[str, Any]:
        """
        Serialize a NumPy array to a dictionary representation.
        
        Args:
            array: NumPy array to serialize
            
        Returns:
            Dictionary with serialized array data
        """
        buffer = io.BytesIO()
        np.save(buffer, array)
        data_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'type': 'numpy.ndarray',
            'dtype': str(array.dtype),
            'shape': array.shape,
            'data': data_b64
        }
    
    @staticmethod
    def deserialize_numpy_array(array_data: Dict[str, Any]) -> np.ndarray:
        """
        Deserialize a NumPy array from its dictionary representation.
        
        Args:
            array_data: Dictionary with serialized array data
            
        Returns:
            NumPy array
        """
        buffer = io.BytesIO(base64.b64decode(array_data['data']))
        return np.load(buffer)
    
    @staticmethod
    def serialize_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize model prediction results.
        
        Args:
            prediction: Prediction data to serialize
            
        Returns:
            Dictionary with serialized prediction data
        """
        result = {'type': 'prediction'}
        
        for key, value in prediction.items():
            if isinstance(value, np.ndarray):
                result[key] = MedicalDataSerializer.serialize_numpy_array(value)
            elif isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                result[key] = value
            else:
                # For other types, use pickle and base64
                pickled = pickle.dumps(value)
                result[key] = {
                    'type': 'pickled',
                    'data': base64.b64encode(pickled).decode('utf-8')
                }
        
        return result
    
    @staticmethod
    def deserialize_prediction(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize model prediction results.
        
        Args:
            prediction_data: Dictionary with serialized prediction data
            
        Returns:
            Dictionary with deserialized prediction data
        """
        result = {}
        
        for key, value in prediction_data.items():
            if key == 'type' and value == 'prediction':
                continue
                
            if isinstance(value, dict) and 'type' in value:
                if value['type'] == 'numpy.ndarray':
                    result[key] = MedicalDataSerializer.deserialize_numpy_array(value)
                elif value['type'] == 'pickled':
                    pickled = base64.b64decode(value['data'])
                    result[key] = pickle.loads(pickled)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def serialize_to_file(data: Any, filepath: str) -> None:
        """
        Serialize data and save to a file.
        
        Args:
            data: Data to serialize
            filepath: Path to save the serialized data
        """
        if filepath.endswith('.json'):
            # For JSON, we need to convert numpy arrays and other complex types
            if isinstance(data, dict):
                json_safe_data = {}
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        json_safe_data[key] = MedicalDataSerializer.serialize_numpy_array(value)
                    elif isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                        json_safe_data[key] = value
                    else:
                        # For other types, use pickle and base64
                        pickled = pickle.dumps(value)
                        json_safe_data[key] = {
                            'type': 'pickled',
                            'data': base64.b64encode(pickled).decode('utf-8')
                        }
                
                with open(filepath, 'w') as f:
                    json.dump(json_safe_data, f, indent=2)
            else:
                raise ValueError("Data must be a dictionary for JSON serialization")
        elif filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Unsupported file extension. Use .json or .pkl")
    
    @staticmethod
    def deserialize_from_file(filepath: str) -> Any:
        """
        Deserialize data from a file.
        
        Args:
            filepath: Path to the serialized data file
            
        Returns:
            Deserialized data
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Convert serialized arrays and complex objects back
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict) and 'type' in value:
                        if value['type'] == 'numpy.ndarray':
                            data[key] = MedicalDataSerializer.deserialize_numpy_array(value)
                        elif value['type'] == 'pickled':
                            pickled = base64.b64decode(value['data'])
                            data[key] = pickle.loads(pickled)
            
            return data
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported file extension. Use .json or .pkl") 