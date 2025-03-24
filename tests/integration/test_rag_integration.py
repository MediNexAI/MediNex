"""
Integration tests for the Medical RAG system.

These tests verify that the knowledge base and RAG components 
work together correctly with actual data.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

from ai.knowledge.medical_rag import MedicalKnowledgeBase, MedicalRAG
from ai.knowledge.data_importer import MedicalDataImporter
from ai.llm.model_connector import MedicalLLMConnector


@pytest.fixture(scope="module")
def temp_storage_dir():
    """Create a temporary directory for the knowledge base."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="module")
def knowledge_base(temp_storage_dir):
    """Create a knowledge base for testing."""
    kb = MedicalKnowledgeBase(
        storage_path=temp_storage_dir,
        chunk_size=200,
        chunk_overlap=20
    )
    
    # Add some test documents
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


class TestRAGIntegration:
    """Test the integration between knowledge base and LLM."""

    def test_rag_query_with_knowledge(self, rag_system):
        """Test querying the RAG system with a topic in the knowledge base."""
        # Query about diabetes (should find knowledge)
        result = rag_system.query(
            query="What is diabetes and what are its symptoms?",
            system_prompt="You are a helpful medical assistant."
        )
        
        # Verify response contains expected information
        assert "answer" in result
        assert "diabetes" in result["answer"].lower()
        assert "symptoms" in result["answer"].lower()
        assert "model" in result
        assert "sources" in result
        assert len(result["sources"]) >= 1
        assert "Diabetes Overview" in [src.get("title") for src in result["sources"]]
        
        # Verify the mock LLM was called with appropriate context
        rag_system.llm_connector.generate_response.assert_called()
        # The last call arguments should include context about diabetes
        last_call_args = rag_system.llm_connector.generate_response.call_args[1]
        assert "diabetes" in last_call_args.get("context", "").lower()

    def test_rag_query_with_different_knowledge(self, rag_system):
        """Test querying the RAG system with a different topic in the knowledge base."""
        # Query about hypertension (should find different knowledge)
        result = rag_system.query(
            query="Tell me about hypertension and its risks",
            system_prompt="You are a helpful medical assistant."
        )
        
        # Verify response contains expected information
        assert "answer" in result
        assert "hypertension" in result["answer"].lower()
        assert "blood pressure" in result["answer"].lower()
        assert "model" in result
        assert "sources" in result
        assert len(result["sources"]) >= 1
        assert "Hypertension Basics" in [src.get("title") for src in result["sources"]]
        
        # Verify the mock LLM was called with appropriate context
        rag_system.llm_connector.generate_response.assert_called()
        # The last call should include context about hypertension
        last_call_args = rag_system.llm_connector.generate_response.call_args[1]
        assert "hypertension" in last_call_args.get("context", "").lower()

    def test_rag_query_without_knowledge(self, rag_system):
        """Test querying the RAG system with a topic not in the knowledge base."""
        # Query about a condition not in the knowledge base
        result = rag_system.query(
            query="What is multiple sclerosis?",
            system_prompt="You are a helpful medical assistant."
        )
        
        # Verify response structure is still correct
        assert "answer" in result
        assert "model" in result
        assert "sources" in result
        # Should have no sources or empty sources
        assert len(result["sources"]) == 0
        
        # Verify the mock LLM was called without context (or with empty context)
        rag_system.llm_connector.generate_response.assert_called()
        last_call_args = rag_system.llm_connector.generate_response.call_args[1]
        assert last_call_args.get("context") is None or last_call_args.get("context") == ""

    def test_data_importer_with_knowledge_base(self, knowledge_base, temp_storage_dir):
        """Test that the data importer can add documents to the knowledge base."""
        # Create a temporary test file
        test_file_path = os.path.join(temp_storage_dir, "test_data.txt")
        with open(test_file_path, "w") as f:
            f.write("Title: Asthma Information\nAuthor: Test Author\n\n"
                   "Asthma is a condition that affects the airways in the lungs. "
                   "It causes the airways to narrow, swell, and produce extra mucus, "
                   "making breathing difficult and triggering coughing, wheezing, and shortness of breath.")
        
        # Create data importer
        importer = MedicalDataImporter(knowledge_base=knowledge_base)
        
        # Import the test file
        importer.import_file(test_file_path)
        
        # Verify the document was added to the knowledge base
        docs = knowledge_base.list_documents()
        titles = [doc.get("title") for doc in docs]
        assert "Asthma Information" in titles
        
        # Test that the document can be found through search
        results = knowledge_base.search(query="Tell me about asthma symptoms", top_k=5)
        assert any("asthma" in result["content"].lower() for result in results)


class TestEndToEndRAG:
    """Test the end-to-end RAG system with real components."""
    
    @pytest.mark.parametrize("query,expected_topic", [
        ("What is diabetes?", "diabetes"),
        ("Tell me about high blood pressure", "hypertension"),
        ("What are the symptoms of diabetes?", "diabetes"),
    ])
    def test_end_to_end_query(self, rag_system, query, expected_topic):
        """Test end-to-end querying with the integrated RAG system."""
        result = rag_system.query(query=query)
        
        # Verify basic response structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "model" in result
        assert "sources" in result
        
        # Verify response contains information about the expected topic
        assert expected_topic in result["answer"].lower()
        
        # For topics in our knowledge base, verify we have sources
        if expected_topic in ["diabetes", "hypertension"]:
            assert len(result["sources"]) >= 1 