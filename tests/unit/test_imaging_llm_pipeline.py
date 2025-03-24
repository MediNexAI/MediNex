"""
Unit tests for the Medical Imaging Analysis Pipeline module.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import tempfile
from PIL import Image
import io

from ai.integrations.imaging_llm_pipeline import MedicalImageProcessor, ImageLLMPipeline


class TestMedicalImageProcessor:
    """Test cases for the MedicalImageProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = MedicalImageProcessor(
            models_dir="test_models",
            device="cpu"
        )

    def test_initialization(self):
        """Test that the processor initializes correctly."""
        assert self.processor.models_dir == "test_models"
        assert self.processor.device == "cpu"
        assert self.processor.models == {}

    @patch('os.path.exists')
    def test_load_model(self, mock_exists):
        """Test loading a model."""
        # Setup mock
        mock_exists.return_value = True
        self.processor._load_model_from_path = MagicMock()
        
        # Call load_model
        self.processor.load_model("xray", "test_models/xray_model.pt")
        
        # Verify model was loaded
        self.processor._load_model_from_path.assert_called_once_with(
            "test_models/xray_model.pt", "cpu"
        )
        assert "xray" in self.processor.models
    
    @patch('os.path.exists')
    def test_load_model_invalid_path(self, mock_exists):
        """Test loading a model with an invalid path."""
        # Setup mock
        mock_exists.return_value = False
        
        # Call load_model and verify it raises an error
        with pytest.raises(FileNotFoundError):
            self.processor.load_model("xray", "invalid/path.pt")

    @patch('PIL.Image.open')
    def test_preprocess_image(self, mock_image_open):
        """Test image preprocessing."""
        # Setup mock image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.resize.return_value = mock_img
        mock_image_array = np.ones((224, 224, 3))
        mock_img.__array__ = MagicMock(return_value=mock_image_array)
        mock_image_open.return_value = mock_img
        
        # Call preprocess_image
        result = self.processor.preprocess_image("test.jpg")
        
        # Verify preprocessing steps
        mock_image_open.assert_called_once_with("test.jpg")
        mock_img.convert.assert_called_once_with("RGB")
        mock_img.resize.assert_called_once()
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3, 224, 224)  # Batch, channels, height, width

    @patch.object(MedicalImageProcessor, 'preprocess_image')
    def test_analyze_image(self, mock_preprocess):
        """Test image analysis."""
        # Setup mocks
        mock_preprocess.return_value = np.ones((1, 3, 224, 224))
        mock_model = MagicMock()
        mock_model.return_value = {
            "features": np.ones((1, 512)),
            "predictions": {
                "condition": "pneumonia",
                "confidence": 0.85
            }
        }
        self.processor.models = {"xray": mock_model}
        
        # Call analyze_image
        result = self.processor.analyze_image("test.jpg", "xray")
        
        # Verify analysis was performed
        mock_preprocess.assert_called_once_with("test.jpg")
        mock_model.assert_called_once()
        assert "features" in result
        assert "predictions" in result
        assert result["predictions"]["condition"] == "pneumonia"
        assert result["predictions"]["confidence"] == 0.85

    def test_analyze_image_model_not_loaded(self):
        """Test analyzing with a model that hasn't been loaded."""
        with pytest.raises(ValueError, match="Model 'nonexistent' not loaded"):
            self.processor.analyze_image("test.jpg", "nonexistent")

    @patch.object(MedicalImageProcessor, 'analyze_image')
    def test_extract_visual_features(self, mock_analyze):
        """Test extracting visual features from an image."""
        # Setup mock
        mock_analyze.return_value = {
            "features": np.ones((1, 512)),
            "predictions": {"condition": "normal"}
        }
        
        # Call extract_visual_features
        features = self.processor.extract_visual_features("test.jpg", "xray")
        
        # Verify feature extraction
        mock_analyze.assert_called_once_with("test.jpg", "xray")
        assert features.shape == (512,)

    @patch.object(MedicalImageProcessor, 'analyze_image')
    def test_get_predictions(self, mock_analyze):
        """Test getting predictions from an image."""
        # Setup mock
        mock_analyze.return_value = {
            "features": np.ones((1, 512)),
            "predictions": {
                "condition": "pneumonia",
                "confidence": 0.85,
                "severity": "moderate"
            }
        }
        
        # Call get_predictions
        predictions = self.processor.get_predictions("test.jpg", "xray")
        
        # Verify predictions
        mock_analyze.assert_called_once_with("test.jpg", "xray")
        assert predictions["condition"] == "pneumonia"
        assert predictions["confidence"] == 0.85
        assert predictions["severity"] == "moderate"


class TestImageLLMPipeline:
    """Test cases for the ImageLLMPipeline class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock image processor
        self.mock_processor = MagicMock()
        self.mock_processor.get_predictions.return_value = {
            "condition": "pneumonia",
            "confidence": 0.85
        }
        self.mock_processor.extract_visual_features.return_value = np.ones(512)
        
        # Create mock LLM connector
        self.mock_llm = MagicMock()
        self.mock_llm.generate_response.return_value = {
            "text": "The image shows signs of pneumonia with 85% confidence.",
            "model": "test-model"
        }
        
        # Create pipeline with mocks
        self.pipeline = ImageLLMPipeline(
            image_processor=self.mock_processor,
            llm_connector=self.mock_llm
        )

    def test_initialization(self):
        """Test that the pipeline initializes correctly."""
        assert self.pipeline.image_processor == self.mock_processor
        assert self.pipeline.llm_connector == self.mock_llm

    def test_analyze_medical_image(self):
        """Test analyzing a medical image with the pipeline."""
        # Call analyze_medical_image
        result = self.pipeline.analyze_medical_image(
            image_path="test.jpg",
            model_type="xray",
            prompt="Describe what you see in this chest X-ray."
        )
        
        # Verify image processor was called
        self.mock_processor.get_predictions.assert_called_once_with("test.jpg", "xray")
        
        # Verify LLM was called
        self.mock_llm.generate_response.assert_called_once()
        call_args = self.mock_llm.generate_response.call_args[1]
        assert call_args["query"].startswith("Describe what you see in this chest X-ray.")
        assert "pneumonia" in call_args["context"]
        assert "85%" in call_args["context"]
        
        # Verify result structure
        assert "analysis" in result
        assert result["analysis"] == "The image shows signs of pneumonia with 85% confidence."
        assert "model" in result
        assert result["model"] == "test-model"
        assert "predictions" in result
        assert result["predictions"]["condition"] == "pneumonia"
        assert result["predictions"]["confidence"] == 0.85

    @patch('tempfile.NamedTemporaryFile')
    def test_analyze_medical_image_bytes(self, mock_temp_file):
        """Test analyzing a medical image provided as bytes."""
        # Setup mock for temporary file
        mock_file = MagicMock()
        mock_file.name = "temp_test.jpg"
        mock_temp_file.return_value.__enter__.return_value = mock_file
        
        # Create image bytes
        image_bytes = b"fake image bytes"
        
        # Call analyze_medical_image_bytes
        result = self.pipeline.analyze_medical_image_bytes(
            image_bytes=image_bytes,
            model_type="xray",
            prompt="Describe what you see in this MRI.",
            image_format="jpg"
        )
        
        # Verify temp file was written to
        mock_file.write.assert_called_once_with(image_bytes)
        
        # Verify image processor was called with temp file
        self.mock_processor.get_predictions.assert_called_once_with("temp_test.jpg", "xray")
        
        # Verify result is as expected
        assert "analysis" in result
        assert "predictions" in result
        assert "model" in result 