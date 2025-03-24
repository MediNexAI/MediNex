"""
Integration tests for the Medical Imaging Analysis Pipeline.

These tests verify that the image processor and LLM components 
work together correctly in the imaging pipeline.
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from ai.integrations.imaging_llm_pipeline import MedicalImageProcessor, ImageLLMPipeline
from ai.llm.model_connector import MedicalLLMConnector


@pytest.fixture
def temp_image_file():
    """Create a temporary test image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        # Create a simple test image (black square)
        img = Image.new('RGB', (224, 224), color='black')
        img.save(tmp.name)
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Clean up the temp file
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def mock_model():
    """Create a mock model for image analysis."""
    model = MagicMock()
    
    # Configure mock model to return reasonable output
    model.return_value = {
        "features": np.ones((1, 512)),
        "predictions": {
            "condition": "pneumonia",
            "confidence": 0.85,
            "severity": "moderate"
        }
    }
    
    return model


@pytest.fixture
def mock_image_processor(mock_model):
    """Create a mock image processor with a preloaded model."""
    processor = MagicMock(spec=MedicalImageProcessor)
    
    # Configure get_predictions to return realistic predictions
    processor.get_predictions.return_value = {
        "condition": "pneumonia",
        "confidence": 0.85,
        "severity": "moderate"
    }
    
    # Configure extract_visual_features to return a feature vector
    processor.extract_visual_features.return_value = np.ones(512)
    
    return processor


@pytest.fixture
def mock_llm_connector():
    """Create a mock LLM connector."""
    connector = MagicMock(spec=MedicalLLMConnector)
    
    # Configure the mock to return a reasonable response
    def generate_response_side_effect(query, context=None, system_prompt=None):
        if "pneumonia" in context:
            return {
                "text": "The image shows signs of pneumonia with 85% confidence. The infection appears to be moderate in severity. "
                        "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid. "
                        "Based on the image analysis, I recommend further clinical evaluation to confirm the diagnosis.",
                "model": "mock-model"
            }
        else:
            return {
                "text": "The image appears to show normal lung tissue without any significant abnormalities.",
                "model": "mock-model"
            }
    
    connector.generate_response.side_effect = generate_response_side_effect
    return connector


@pytest.fixture
def imaging_pipeline(mock_image_processor, mock_llm_connector):
    """Create an imaging pipeline with mock components."""
    return ImageLLMPipeline(
        image_processor=mock_image_processor,
        llm_connector=mock_llm_connector
    )


class TestImagingPipelineIntegration:
    """Test the integration between image processor and LLM."""

    def test_analyze_medical_image(self, imaging_pipeline, temp_image_file, mock_image_processor):
        """Test analyzing a medical image with the pipeline."""
        # Analyze the test image
        result = imaging_pipeline.analyze_medical_image(
            image_path=temp_image_file,
            model_type="xray",
            prompt="Analyze this chest X-ray and provide clinical insights."
        )
        
        # Verify the image processor was called correctly
        mock_image_processor.get_predictions.assert_called_once_with(temp_image_file, "xray")
        
        # Verify response structure
        assert "analysis" in result
        assert "model" in result
        assert "predictions" in result
        assert "pneumonia" in result["analysis"].lower()
        assert result["predictions"]["condition"] == "pneumonia"
        assert result["predictions"]["confidence"] == 0.85
        assert result["predictions"]["severity"] == "moderate"

    def test_analyze_medical_image_with_different_prompt(self, imaging_pipeline, temp_image_file):
        """Test analyzing a medical image with a different prompt."""
        # Reset mock to clear previous calls
        imaging_pipeline.llm_connector.generate_response.reset_mock()
        
        # Analyze with a different prompt
        result = imaging_pipeline.analyze_medical_image(
            image_path=temp_image_file,
            model_type="xray",
            prompt="What treatment would you recommend based on this X-ray?"
        )
        
        # Verify the LLM was called with the correct prompt
        call_args = imaging_pipeline.llm_connector.generate_response.call_args[1]
        assert call_args["query"].startswith("What treatment would you recommend")
        assert "pneumonia" in call_args["context"]
        
        # Verify response contains treatment recommendations
        assert "analysis" in result
        assert "pneumonia" in result["analysis"].lower()

    def test_image_bytes_analysis(self, imaging_pipeline, temp_image_file):
        """Test analyzing image provided as bytes."""
        # Read the test image as bytes
        with open(temp_image_file, 'rb') as f:
            image_bytes = f.read()
        
        # Analyze the image bytes
        result = imaging_pipeline.analyze_medical_image_bytes(
            image_bytes=image_bytes,
            model_type="xray",
            prompt="Analyze this chest X-ray.",
            image_format="jpg"
        )
        
        # Verify response structure
        assert "analysis" in result
        assert "model" in result
        assert "predictions" in result
        assert result["predictions"]["condition"] == "pneumonia"


@pytest.mark.integration
class TestRealImageProcessorMockLLM:
    """Test using a real image processor with a mock LLM."""

    @pytest.fixture
    def real_image_processor(self):
        """Create a real image processor with mock models."""
        processor = MedicalImageProcessor(
            models_dir="test_models",
            device="cpu"
        )
        
        # Patch the internal _load_model_from_path method
        processor._load_model_from_path = MagicMock(return_value=MagicMock())
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.return_value = {
            "features": np.ones((1, 512)),
            "predictions": {
                "condition": "pneumonia",
                "confidence": 0.85
            }
        }
        
        # Add the mock model to the processor
        processor.models = {"xray": mock_model}
        
        return processor

    def test_real_processor_with_mock_llm(self, real_image_processor, mock_llm_connector, temp_image_file):
        """Test a real processor with a mock LLM."""
        # Create pipeline with real processor and mock LLM
        pipeline = ImageLLMPipeline(
            image_processor=real_image_processor,
            llm_connector=mock_llm_connector
        )
        
        # Analyze the test image
        result = pipeline.analyze_medical_image(
            image_path=temp_image_file,
            model_type="xray",
            prompt="Analyze this chest X-ray."
        )
        
        # Verify response structure and content
        assert "analysis" in result
        assert "model" in result
        assert "predictions" in result
        assert "pneumonia" in result["analysis"].lower()
        assert result["predictions"]["condition"] == "pneumonia" 