"""
Integration tests for the MediNex AI API.

These tests verify that the API endpoints correctly integrate with the RAG and image analysis systems.
"""

import pytest
import os
import tempfile
import json
import base64
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from ai.api.core import create_app
from ai.knowledge.medical_rag import MedicalKnowledgeBase, MedicalRAG
from ai.integrations.imaging_llm_pipeline import ImageLLMPipeline
from ai.llm.model_connector import MedicalLLMConnector


@pytest.fixture(scope="module")
def temp_storage_dir():
    """Create a temporary directory for the knowledge base."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="module")
def knowledge_base(temp_storage_dir):
    """Create a knowledge base with test data."""
    kb = MedicalKnowledgeBase(
        storage_path=temp_storage_dir,
        chunk_size=200,
        chunk_overlap=20
    )
    
    # Add test data
    kb.add_document(
        content="Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). "
                "There are different types of diabetes: Type 1, Type 2, and gestational diabetes. "
                "Common symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision.",
        metadata={"title": "Diabetes Overview", "source": "Test Medical Resource"}
    )
    
    kb.add_document(
        content="Hypertension, also known as high blood pressure, is a condition where the force of blood against "
                "your artery walls is consistently too high. Over time, this can cause health problems like heart disease. "
                "It is often called a 'silent killer' because it typically has no symptoms until significant damage has occurred.",
        metadata={"title": "Hypertension Basics", "source": "Test Medical Resource"}
    )
    
    return kb


@pytest.fixture
def mock_llm_connector():
    """Create a mock LLM connector for testing."""
    connector = MagicMock(spec=MedicalLLMConnector)
    
    # Configure the mock to return a reasonable response
    def generate_response_side_effect(query, context=None, system_prompt=None):
        if "diabetes" in query.lower():
            return {
                "text": "Diabetes is a chronic medical condition characterized by elevated blood sugar levels. "
                        "It comes in several forms, including Type 1, Type 2, and gestational diabetes. "
                        "Common symptoms include excessive thirst, frequent urination, increased hunger, and fatigue.",
                "model": "mock-model"
            }
        elif "hypertension" in query.lower() or "blood pressure" in query.lower():
            return {
                "text": "Hypertension (high blood pressure) is a common condition where blood flows through your "
                        "arteries at higher than normal pressure. It's often called a 'silent killer' because it "
                        "frequently has no symptoms but can lead to serious health problems like heart disease and stroke.",
                "model": "mock-model"
            }
        else:
            return {
                "text": "I don't have specific information about that medical condition.",
                "model": "mock-model"
            }
    
    connector.generate_response.side_effect = generate_response_side_effect
    return connector


@pytest.fixture
def rag_system(knowledge_base, mock_llm_connector):
    """Create a RAG system for testing."""
    return MedicalRAG(
        knowledge_base=knowledge_base,
        llm_connector=mock_llm_connector
    )


@pytest.fixture
def mock_image_processor():
    """Create a mock image processor."""
    processor = MagicMock()
    processor.get_predictions.return_value = {
        "condition": "pneumonia",
        "confidence": 0.85,
        "severity": "moderate"
    }
    return processor


@pytest.fixture
def mock_image_pipeline(mock_image_processor, mock_llm_connector):
    """Create a mock image pipeline."""
    pipeline = MagicMock(spec=ImageLLMPipeline)
    
    def analyze_medical_image_bytes_side_effect(image_bytes, model_type, prompt, image_format=None):
        return {
            "analysis": "The image shows signs of pneumonia with 85% confidence. The infection appears to be moderate in severity.",
            "model": "mock-model",
            "predictions": {
                "condition": "pneumonia",
                "confidence": 0.85,
                "severity": "moderate"
            }
        }
    
    pipeline.analyze_medical_image_bytes.side_effect = analyze_medical_image_bytes_side_effect
    return pipeline


@pytest.fixture
def api_client(rag_system, mock_image_pipeline):
    """Create a test client for the API."""
    app = create_app(
        rag_system=rag_system,
        image_pipeline=mock_image_pipeline
    )
    return TestClient(app)


class TestAPIIntegration:
    """Integration tests for the API endpoints."""

    def test_api_info(self, api_client):
        """Test the API info endpoint."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "MediNex AI API"
        assert "version" in data
        assert "description" in data

    def test_health_check(self, api_client):
        """Test the health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_query_with_knowledge(self, api_client, rag_system):
        """Test querying with knowledge available in the RAG system."""
        request_data = {
            "query": "What is diabetes and what are its symptoms?",
            "system_prompt": "You are a medical assistant."
        }
        
        response = api_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "diabetes" in data["answer"].lower()
        assert "symptoms" in data["answer"].lower()
        assert "model" in data
        assert "sources" in data
        assert len(data["sources"]) >= 1
        
        # Verify RAG system was used correctly
        rag_system.query.assert_called_once()
        call_args = rag_system.query.call_args[1]
        assert call_args["query"] == "What is diabetes and what are its symptoms?"
        assert call_args["system_prompt"] == "You are a medical assistant."

    def test_query_with_different_knowledge(self, api_client, rag_system):
        """Test querying with different knowledge available in the RAG system."""
        # Reset mock to clear previous calls
        rag_system.query.reset_mock()
        
        request_data = {
            "query": "Explain hypertension and its risks",
            "system_prompt": "You are a medical assistant."
        }
        
        response = api_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "hypertension" in data["answer"].lower() or "blood pressure" in data["answer"].lower()
        assert "model" in data
        assert "sources" in data
        
        # Verify RAG system was used correctly
        rag_system.query.assert_called_once()
        call_args = rag_system.query.call_args[1]
        assert call_args["query"] == "Explain hypertension and its risks"

    def test_image_analysis(self, api_client, mock_image_pipeline):
        """Test the image analysis endpoint."""
        # Create test image data
        test_image = b"fake image data"
        image_base64 = base64.b64encode(test_image).decode('utf-8')
        
        request_data = {
            "image": image_base64,
            "model_type": "xray",
            "prompt": "Analyze this chest X-ray."
        }
        
        response = api_client.post("/api/analyze-image", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis" in data
        assert "pneumonia" in data["analysis"].lower()
        assert "model" in data
        assert "predictions" in data
        assert data["predictions"]["condition"] == "pneumonia"
        assert data["predictions"]["confidence"] == 0.85
        
        # Verify image pipeline was used correctly
        mock_image_pipeline.analyze_medical_image_bytes.assert_called_once()
        call_args = mock_image_pipeline.analyze_medical_image_bytes.call_args[1]
        assert call_args["image_bytes"] == test_image
        assert call_args["model_type"] == "xray"
        assert call_args["prompt"] == "Analyze this chest X-ray."

    def test_query_validation(self, api_client):
        """Test validation for the query endpoint."""
        # Test missing query field
        response = api_client.post("/api/query", json={"system_prompt": "test"})
        assert response.status_code == 422
        
        # Test empty query
        response = api_client.post("/api/query", json={"query": "", "system_prompt": "test"})
        assert response.status_code == 422

    def test_image_analysis_validation(self, api_client):
        """Test validation for the image analysis endpoint."""
        # Test missing image field
        response = api_client.post(
            "/api/analyze-image", 
            json={"model_type": "xray", "prompt": "Analyze this image."}
        )
        assert response.status_code == 422
        
        # Test invalid base64
        response = api_client.post(
            "/api/analyze-image", 
            json={"image": "not-base64", "model_type": "xray", "prompt": "Analyze this image."}
        )
        assert response.status_code == 400
        
        # Test missing model_type
        response = api_client.post(
            "/api/analyze-image", 
            json={"image": base64.b64encode(b"test").decode('utf-8'), "prompt": "Analyze this image."}
        )
        assert response.status_code == 422 