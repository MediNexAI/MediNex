"""
Test configuration and shared fixtures for the MediNex AI project.
"""

import pytest
import os
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from ai.knowledge.medical_rag import MedicalKnowledgeBase, MedicalRAG
from ai.llm.model_connector import MedicalLLMConnector
from ai.integrations.imaging_llm_pipeline import MedicalImageProcessor, ImageLLMPipeline


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    # Get the directory of this file and go up one level
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Create a temporary directory for test data."""
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    
    # Create sample directories
    test_data_path = Path(temp_dir.name)
    sample_docs_dir = test_data_path / "sample_docs"
    sample_images_dir = test_data_path / "sample_images"
    
    sample_docs_dir.mkdir(exist_ok=True)
    sample_images_dir.mkdir(exist_ok=True)
    
    # Create sample text document
    with open(sample_docs_dir / "sample_medical_text.txt", "w") as f:
        f.write("""
        Medical Record Summary
        
        Patient presents with symptoms of hypertension and type 2 diabetes.
        Blood pressure reading: 140/90 mmHg
        Fasting blood glucose: 145 mg/dL
        
        Current medications:
        - Metformin 500mg twice daily
        - Lisinopril 10mg once daily
        
        Patient reports compliance with medication regimen but continues
        to experience occasional headaches and fatigue.
        
        Recommend increased physical activity and dietary modifications
        to improve glycemic control and blood pressure.
        """)
    
    # Create sample CSV data
    with open(sample_docs_dir / "medical_terms.csv", "w") as f:
        f.write("term,definition,category\n")
        f.write("hypertension,Abnormally high blood pressure,cardiovascular\n")
        f.write("diabetes mellitus,Metabolic disease characterized by high blood sugar,endocrine\n")
        f.write("metformin,Antidiabetic medication,medication\n")
        f.write("lisinopril,ACE inhibitor used to treat hypertension,medication\n")
    
    # Create sample JSON data
    medical_data = {
        "diseases": [
            {
                "name": "Type 2 Diabetes",
                "icd10": "E11",
                "symptoms": ["polyuria", "polydipsia", "polyphagia", "weight loss"],
                "treatments": ["diet", "exercise", "metformin", "insulin"]
            },
            {
                "name": "Hypertension",
                "icd10": "I10",
                "symptoms": ["headache", "dizziness", "blurred vision"],
                "treatments": ["diet", "exercise", "ACE inhibitors", "diuretics"]
            }
        ]
    }
    
    with open(sample_docs_dir / "diseases.json", "w") as f:
        json.dump(medical_data, f, indent=2)
    
    # Create dummy configuration
    test_config = {
        "llm": {
            "provider": "test",
            "model": "test-model",
            "temperature": 0.1,
            "api_key": "test-key"
        },
        "knowledge": {
            "storage_type": "memory",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "api": {
            "host": "localhost",
            "port": 8080,
            "log_level": "info"
        },
        "imaging": {
            "models_dir": str(test_data_path / "models")
        }
    }
    
    with open(test_data_path / "test_config.json", "w") as f:
        json.dump(test_config, f, indent=2)
    
    yield test_data_path
    
    # Cleanup
    temp_dir.cleanup()


@pytest.fixture(scope="function")
def temp_knowledge_base_dir():
    """Create a temporary directory for knowledge base testing."""
    temp_dir = tempfile.TemporaryDirectory()
    
    # Create directories
    kb_dir = Path(temp_dir.name)
    embeddings_dir = kb_dir / "embeddings"
    metadata_dir = kb_dir / "metadata"
    
    embeddings_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    
    # Create empty metadata file
    with open(metadata_dir / "documents.json", "w") as f:
        json.dump({}, f)
    
    yield kb_dir
    
    # Cleanup
    temp_dir.cleanup()


@pytest.fixture(scope="function")
def mock_config_file():
    """Create a temporary config file for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_file.close()
    
    # Default test configuration
    config = {
        "llm": {
            "provider": "test",
            "model": "test-model",
            "temperature": 0.1,
            "api_key": "test-key"
        },
        "knowledge": {
            "storage_type": "memory",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "api": {
            "host": "localhost",
            "port": 8080,
            "log_level": "info"
        },
        "imaging": {
            "models_dir": "/tmp/models"
        },
        "contributors": {
            "data_file": "/tmp/contributors.json",
            "contributions_dir": "/tmp/contributions"
        },
        "distribution": {
            "registry_file": "/tmp/model_registry.json",
            "versions_dir": "/tmp/model_versions",
            "packages_dir": "/tmp/model_packages",
            "deployments_dir": "/tmp/model_deployments"
        }
    }
    
    with open(temp_file.name, "w") as f:
        json.dump(config, f, indent=2)
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture(scope="function")
def mock_env_vars():
    """Set up environment variables for testing and restore them after."""
    # Save original environment variables
    original_vars = {}
    test_vars = {
        "MEDINEX_API_KEY": "test-api-key",
        "MEDINEX_LOG_LEVEL": "debug",
        "MEDINEX_DATA_DIR": "/tmp/medinex",
        "MEDINEX_MODEL_DIR": "/tmp/medinex/models",
        "MEDINEX_ENABLE_TELEMETRY": "false"
    }
    
    # Save original values
    for var in test_vars:
        if var in os.environ:
            original_vars[var] = os.environ[var]
    
    # Set test values
    for var, value in test_vars.items():
        os.environ[var] = value
    
    yield
    
    # Restore original values
    for var in test_vars:
        if var in original_vars:
            os.environ[var] = original_vars[var]
        else:
            del os.environ[var]


@pytest.fixture(scope="session")
def test_config():
    """Return a test configuration for MediNex AI."""
    return {
        "llm": {
            "provider": "mock",
            "model": "mock-model",
            "temperature": 0.7,
            "api_key": "mock-key"
        },
        "knowledge_base": {
            "storage_path": "/tmp/medinex_test/knowledge",
            "chunk_size": 200,
            "chunk_overlap": 20
        },
        "api": {
            "host": "localhost",
            "port": 8000
        },
        "imaging": {
            "models_dir": "/tmp/medinex_test/models",
            "device": "cpu"
        }
    }


@pytest.fixture(scope="session")
def global_temp_dir():
    """Create a global temporary directory for all tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create necessary subdirectories
        os.makedirs(os.path.join(temp_dir, "knowledge"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "sample"), exist_ok=True)
        
        yield temp_dir


@pytest.fixture
def mock_llm_connector():
    """Create a mock LLM connector for testing."""
    connector = MagicMock(spec=MedicalLLMConnector)
    
    # Configure the mock to return a reasonable response
    def generate_response_side_effect(query, context=None, system_prompt=None):
        if "diabetes" in query.lower():
            return {
                "text": "Diabetes is a chronic medical condition characterized by elevated blood sugar levels.",
                "model": "mock-model"
            }
        elif "hypertension" in query.lower() or "blood pressure" in query.lower():
            return {
                "text": "Hypertension is a condition where blood pressure is consistently too high.",
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
def mock_knowledge_base():
    """Create a mock knowledge base for testing."""
    kb = MagicMock(spec=MedicalKnowledgeBase)
    
    # Configure mock to return search results
    kb.search.return_value = [
        {
            "content": "Diabetes is a chronic condition that affects how your body processes blood sugar.",
            "metadata": {"title": "Diabetes Overview", "source": "Test Source"}
        }
    ]
    
    # Configure mock to return document list
    kb.list_documents.return_value = [
        {
            "id": "doc-1",
            "title": "Diabetes Overview",
            "source": "Test Source"
        },
        {
            "id": "doc-2",
            "title": "Hypertension Overview",
            "source": "Test Source"
        }
    ]
    
    # Configure mock for document operations
    kb.add_document.return_value = "doc-3"
    kb.delete_document.return_value = True
    
    return kb


@pytest.fixture
def mock_rag_system(mock_knowledge_base, mock_llm_connector):
    """Create a mock RAG system for testing."""
    rag = MagicMock(spec=MedicalRAG)
    
    # Configure mock to return query results
    def query_side_effect(query, system_prompt=None):
        if "diabetes" in query.lower():
            return {
                "answer": "Diabetes is a chronic condition that affects blood sugar levels.",
                "model": "mock-model",
                "sources": [{"title": "Diabetes Overview", "content": "Sample content"}]
            }
        elif "hypertension" in query.lower() or "blood pressure" in query.lower():
            return {
                "answer": "Hypertension is a condition where blood pressure is consistently too high.",
                "model": "mock-model",
                "sources": [{"title": "Hypertension Overview", "content": "Sample content"}]
            }
        else:
            return {
                "answer": "I don't have specific information about that medical condition.",
                "model": "mock-model",
                "sources": []
            }
    
    rag.query.side_effect = query_side_effect
    rag.knowledge_base = mock_knowledge_base
    rag.llm_connector = mock_llm_connector
    
    return rag


@pytest.fixture
def mock_image_processor():
    """Create a mock image processor for testing."""
    processor = MagicMock(spec=MedicalImageProcessor)
    
    # Configure mock to return predictions
    processor.get_predictions.return_value = {
        "condition": "pneumonia",
        "confidence": 0.85,
        "severity": "moderate"
    }
    
    # Configure mock to return feature vector
    processor.extract_visual_features.return_value = MagicMock()
    
    return processor


@pytest.fixture
def mock_image_pipeline(mock_image_processor, mock_llm_connector):
    """Create a mock image pipeline for testing."""
    pipeline = MagicMock(spec=ImageLLMPipeline)
    
    # Configure mock to return analysis results
    def analyze_side_effect(image_path=None, model_type=None, prompt=None):
        return {
            "analysis": "The image shows signs of pneumonia with 85% confidence.",
            "model": "mock-model",
            "predictions": {
                "condition": "pneumonia",
                "confidence": 0.85,
                "severity": "moderate"
            }
        }
    
    pipeline.analyze_medical_image.side_effect = analyze_side_effect
    pipeline.analyze_medical_image_bytes.side_effect = analyze_side_effect
    pipeline.image_processor = mock_image_processor
    pipeline.llm_connector = mock_llm_connector
    
    return pipeline 