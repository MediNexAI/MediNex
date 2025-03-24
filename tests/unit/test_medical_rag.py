"""
Unit tests for the Medical RAG module.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import json
import tempfile

from ai.knowledge.medical_rag import MedicalKnowledgeBase, MedicalRAG


class TestMedicalKnowledgeBase:
    """Test cases for the MedicalKnowledgeBase class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for the knowledge base
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = self.temp_dir.name
        
        # Create knowledge base with test configuration
        self.kb = MedicalKnowledgeBase(
            storage_path=self.storage_path,
            chunk_size=200,
            chunk_overlap=20
        )

    def teardown_method(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the knowledge base initializes correctly."""
        assert self.kb.storage_path == self.storage_path
        assert self.kb.chunk_size == 200
        assert self.kb.chunk_overlap == 20
        assert self.kb.db is not None

    def test_add_document(self):
        """Test adding a document to the knowledge base."""
        doc_id = self.kb.add_document(
            content="This is a test medical document about diabetes.",
            metadata={
                "title": "Diabetes Test",
                "source": "Test Source",
                "author": "Test Author"
            }
        )
        
        # Verify document was added
        assert doc_id is not None
        assert len(doc_id) > 0
        
        # Verify document can be retrieved
        docs = self.kb.list_documents(limit=10)
        assert len(docs) == 1
        assert docs[0]["id"] == doc_id
        assert docs[0]["title"] == "Diabetes Test"
    
    def test_search(self):
        """Test searching the knowledge base."""
        # Add test document
        self.kb.add_document(
            content="Diabetes is a chronic medical condition that affects how your body turns food into energy.",
            metadata={"title": "Diabetes Overview"}
        )
        
        # Search for relevant content
        results = self.kb.search(query="What is diabetes?", top_k=5)
        
        # Verify search returns results
        assert len(results) > 0
        assert "diabetes" in results[0]["content"].lower()

    def test_delete_document(self):
        """Test deleting a document from the knowledge base."""
        # Add test document
        doc_id = self.kb.add_document(
            content="Test content for deletion.",
            metadata={"title": "Test Deletion"}
        )
        
        # Verify document exists
        docs = self.kb.list_documents()
        assert len(docs) == 1
        
        # Delete document
        success = self.kb.delete_document(doc_id)
        
        # Verify deletion was successful
        assert success is True
        docs = self.kb.list_documents()
        assert len(docs) == 0


class TestMedicalRAG:
    """Test cases for the MedicalRAG class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock knowledge base
        self.mock_kb = MagicMock()
        self.mock_kb.search.return_value = [
            {
                "content": "Diabetes is a chronic condition that affects how the body processes blood sugar.",
                "metadata": {"title": "Diabetes Info"}
            }
        ]
        
        # Create mock LLM connector
        self.mock_llm = MagicMock()
        self.mock_llm.generate_response.return_value = {
            "text": "Diabetes is a chronic metabolic disorder characterized by elevated blood sugar levels.",
            "model": "test-model"
        }
        
        # Create RAG system with mocks
        self.rag = MedicalRAG(
            knowledge_base=self.mock_kb,
            llm_connector=self.mock_llm
        )

    def test_initialization(self):
        """Test that the RAG system initializes correctly."""
        assert self.rag.knowledge_base == self.mock_kb
        assert self.rag.llm_connector == self.mock_llm

    def test_query_with_knowledge(self):
        """Test querying the RAG system with knowledge retrieval."""
        query = "What is diabetes?"
        
        # Call query method
        result = self.rag.query(
            query=query,
            system_prompt="You are a medical assistant."
        )
        
        # Verify knowledge base was searched
        self.mock_kb.search.assert_called_once_with(query=query, top_k=5)
        
        # Verify LLM was called with context
        call_args = self.mock_llm.generate_response.call_args[1]
        assert call_args["query"] == query
        assert "Diabetes is a chronic condition" in call_args["context"]
        
        # Verify result structure
        assert "answer" in result
        assert "model" in result
        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Diabetes Info"

    def test_query_with_empty_knowledge(self):
        """Test querying when no relevant knowledge is found."""
        # Configure mock to return no results
        self.mock_kb.search.return_value = []
        
        query = "What is a rare disease?"
        
        # Call query method
        result = self.rag.query(query=query)
        
        # Verify LLM was called without context
        call_args = self.mock_llm.generate_response.call_args[1]
        assert call_args["query"] == query
        assert call_args.get("context") is None or call_args.get("context") == "" 