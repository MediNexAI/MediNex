"""
Unit tests for the MediNex AI API Core module.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import json
import base64

from ai.api.core import create_app


class TestMediNexAPI:
    """Test cases for the MediNex AI API."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mocks for dependencies
        self.mock_rag = MagicMock()
        self.mock_rag.query.return_value = {
            "answer": "Diabetes is a chronic condition that affects blood sugar levels.",
            "model": "test-model",
            "sources": [{"title": "Diabetes Info", "content": "Sample content"}]
        }
        
        self.mock_image_pipeline = MagicMock()
        self.mock_image_pipeline.analyze_medical_image_bytes.return_value = {
            "analysis": "The image shows potential pneumonia.",
            "model": "test-model",
            "predictions": {"condition": "pneumonia", "confidence": 0.85}
        }
        
        # Create test client
        self.app = create_app(
            rag_system=self.mock_rag,
            image_pipeline=self.mock_image_pipeline
        )
        self.client = TestClient(self.app)

    def test_root_endpoint(self):
        """Test the root endpoint returns API info."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert data["name"] == "MediNex AI API"

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_query_endpoint(self):
        """Test the medical query endpoint."""
        # Prepare request data
        request_data = {
            "query": "What is diabetes?",
            "system_prompt": "You are a medical assistant."
        }
        
        # Send request
        response = self.client.post("/api/query", json=request_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "model" in data
        assert "sources" in data
        assert data["answer"] == "Diabetes is a chronic condition that affects blood sugar levels."
        
        # Verify RAG system was called correctly
        self.mock_rag.query.assert_called_once_with(
            query="What is diabetes?",
            system_prompt="You are a medical assistant."
        )

    def test_query_endpoint_validation(self):
        """Test validation for the query endpoint."""
        # Test with missing query
        response = self.client.post("/api/query", json={"system_prompt": "test"})
        assert response.status_code == 422  # Validation error
        
        # Test with empty query
        response = self.client.post("/api/query", json={"query": "", "system_prompt": "test"})
        assert response.status_code == 422  # Validation error

    def test_analyze_image_endpoint(self):
        """Test the image analysis endpoint."""
        # Prepare mock image data
        image_content = b"fake image data"
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        
        # Prepare request data
        request_data = {
            "image": image_base64,
            "model_type": "xray",
            "prompt": "Describe this chest X-ray."
        }
        
        # Send request
        response = self.client.post("/api/analyze-image", json=request_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "analysis" in data
        assert "model" in data
        assert "predictions" in data
        assert data["analysis"] == "The image shows potential pneumonia."
        assert data["predictions"]["condition"] == "pneumonia"
        
        # Verify image pipeline was called correctly
        call_args = self.mock_image_pipeline.analyze_medical_image_bytes.call_args[1]
        assert "image_bytes" in call_args
        assert call_args["model_type"] == "xray"
        assert call_args["prompt"] == "Describe this chest X-ray."

    def test_analyze_image_endpoint_validation(self):
        """Test validation for the image analysis endpoint."""
        # Test with missing image
        response = self.client.post(
            "/api/analyze-image", 
            json={"model_type": "xray", "prompt": "test"}
        )
        assert response.status_code == 422  # Validation error
        
        # Test with invalid base64
        response = self.client.post(
            "/api/analyze-image", 
            json={"image": "not-base64", "model_type": "xray", "prompt": "test"}
        )
        assert response.status_code == 400  # Bad request
        
        # Test with missing model_type
        response = self.client.post(
            "/api/analyze-image", 
            json={"image": base64.b64encode(b"test").decode('utf-8'), "prompt": "test"}
        )
        assert response.status_code == 422  # Validation error 