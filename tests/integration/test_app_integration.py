"""
Integration tests for the MediNex AI Command Line Application.

These tests verify that the main application components (app.py)
work together correctly with the core modules.
"""

import pytest
import os
import tempfile
import json
import subprocess
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

# Import the app module directly
import app
from ai.knowledge.medical_rag import MedicalKnowledgeBase
from ai.knowledge.data_importer import MedicalDataImporter


@pytest.fixture(scope="module")
def temp_project_dir():
    """Create a temporary directory for the project."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create necessary subdirectories
        os.makedirs(os.path.join(temp_dir, "data", "knowledge"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "data", "sample"), exist_ok=True)
        
        # Create a sample test file
        sample_file_path = os.path.join(temp_dir, "data", "sample", "diabetes.txt")
        with open(sample_file_path, "w") as f:
            f.write("Title: Diabetes Overview\nAuthor: Test Author\n\n"
                   "Diabetes is a condition that affects how your body processes blood sugar (glucose). "
                   "There are different types of diabetes: Type 1, Type 2, and gestational diabetes. "
                   "Common symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision.")
        
        # Create a minimal config file
        config_path = os.path.join(temp_dir, "config.json")
        config = {
            "llm": {
                "provider": "mock",
                "model": "mock-model",
                "temperature": 0.7,
                "api_key": "mock-key"
            },
            "knowledge_base": {
                "storage_path": os.path.join(temp_dir, "data", "knowledge"),
                "chunk_size": 200,
                "chunk_overlap": 20
            },
            "api": {
                "host": "localhost",
                "port": 8000
            },
            "imaging": {
                "models_dir": os.path.join(temp_dir, "models"),
                "device": "cpu"
            }
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        yield temp_dir


class TestAppIntegration:
    """Test the integration of app.py with core modules."""

    def test_init_command(self, temp_project_dir):
        """Test the init command to initialize the system."""
        # Patch the app's config loading to use our temp project
        with patch.object(app, 'CONFIG_PATH', os.path.join(temp_project_dir, "config.json")):
            # Redirect stdout to capture output
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                # Call the init command
                app.init_system()
                
                # Verify knowledge base was created
                kb_path = os.path.join(temp_project_dir, "data", "knowledge")
                assert os.path.exists(kb_path)
                
                # Check output
                output = captured_output.getvalue()
                assert "Initializing MediNex AI system" in output
                assert "Knowledge base created" in output
            finally:
                sys.stdout = sys.__stdout__

    def test_import_command(self, temp_project_dir):
        """Test the import command to import medical data."""
        # Patch the app's config loading to use our temp project
        with patch.object(app, 'CONFIG_PATH', os.path.join(temp_project_dir, "config.json")):
            # Create mock knowledge base
            mock_kb = MagicMock(spec=MedicalKnowledgeBase)
            
            # Patch the create_knowledge_base function to return our mock
            with patch.object(app, 'create_knowledge_base', return_value=mock_kb):
                # Redirect stdout to capture output
                captured_output = StringIO()
                sys.stdout = captured_output
                
                try:
                    # Call the import command
                    sample_path = os.path.join(temp_project_dir, "data", "sample")
                    app.import_data(source_path=sample_path)
                    
                    # Verify data importer was used
                    # The MedicalDataImporter would have been created with our mock_kb
                    # and the import_directory method would have been called
                    
                    # Check output
                    output = captured_output.getvalue()
                    assert "Importing medical data" in output
                    assert sample_path in output
                finally:
                    sys.stdout = sys.__stdout__

    def test_query_command(self, temp_project_dir):
        """Test the query command to query the medical knowledge base."""
        # Patch the app's config loading to use our temp project
        with patch.object(app, 'CONFIG_PATH', os.path.join(temp_project_dir, "config.json")):
            # Create mock RAG system
            mock_rag = MagicMock()
            mock_rag.query.return_value = {
                "answer": "Diabetes is a chronic condition that affects blood sugar levels.",
                "model": "mock-model",
                "sources": [{"title": "Diabetes Overview", "content": "Sample content"}]
            }
            
            # Patch the create_rag_system function to return our mock
            with patch.object(app, 'create_rag_system', return_value=mock_rag):
                # Redirect stdout to capture output
                captured_output = StringIO()
                sys.stdout = captured_output
                
                try:
                    # Call the query command
                    app.query_knowledge_base(query="What is diabetes?")
                    
                    # Verify RAG system was used
                    mock_rag.query.assert_called_once_with(
                        query="What is diabetes?", 
                        system_prompt=None
                    )
                    
                    # Check output
                    output = captured_output.getvalue()
                    assert "Query:" in output
                    assert "What is diabetes?" in output
                    assert "Answer:" in output
                    assert "Diabetes is a chronic condition" in output
                    assert "Source:" in output
                    assert "Diabetes Overview" in output
                finally:
                    sys.stdout = sys.__stdout__

    def test_list_documents_command(self, temp_project_dir):
        """Test the list-documents command."""
        # Patch the app's config loading to use our temp project
        with patch.object(app, 'CONFIG_PATH', os.path.join(temp_project_dir, "config.json")):
            # Create mock knowledge base
            mock_kb = MagicMock(spec=MedicalKnowledgeBase)
            mock_kb.list_documents.return_value = [
                {
                    "id": "doc-1",
                    "title": "Diabetes Overview",
                    "source": "Test Source",
                    "author": "Test Author"
                },
                {
                    "id": "doc-2",
                    "title": "Hypertension Basics",
                    "source": "Test Source",
                    "author": "Test Author"
                }
            ]
            
            # Patch the create_knowledge_base function to return our mock
            with patch.object(app, 'create_knowledge_base', return_value=mock_kb):
                # Redirect stdout to capture output
                captured_output = StringIO()
                sys.stdout = captured_output
                
                try:
                    # Call the list documents command
                    app.list_documents()
                    
                    # Verify knowledge base was used
                    mock_kb.list_documents.assert_called_once()
                    
                    # Check output
                    output = captured_output.getvalue()
                    assert "Documents in knowledge base:" in output
                    assert "Diabetes Overview" in output
                    assert "Hypertension Basics" in output
                finally:
                    sys.stdout = sys.__stdout__

    def test_delete_document_command(self, temp_project_dir):
        """Test the delete-document command."""
        # Patch the app's config loading to use our temp project
        with patch.object(app, 'CONFIG_PATH', os.path.join(temp_project_dir, "config.json")):
            # Create mock knowledge base
            mock_kb = MagicMock(spec=MedicalKnowledgeBase)
            mock_kb.delete_document.return_value = True
            
            # Patch the create_knowledge_base function to return our mock
            with patch.object(app, 'create_knowledge_base', return_value=mock_kb):
                # Redirect stdout to capture output
                captured_output = StringIO()
                sys.stdout = captured_output
                
                try:
                    # Call the delete document command
                    app.delete_document(document_id="doc-1")
                    
                    # Verify knowledge base was used
                    mock_kb.delete_document.assert_called_once_with("doc-1")
                    
                    # Check output
                    output = captured_output.getvalue()
                    assert "Document deleted successfully" in output
                finally:
                    sys.stdout = sys.__stdout__ 