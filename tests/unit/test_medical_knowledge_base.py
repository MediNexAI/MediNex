"""
Unit tests for the Medical Knowledge Base module.
"""

import pytest
import os
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai.knowledge.medical_knowledge_base import (
    MedicalKnowledgeBase,
    Document,
    DocumentMetadata,
    KnowledgeBaseError
)


class TestDocument:
    """Test cases for the Document class."""

    def test_document_initialization(self):
        """Test basic document initialization."""
        doc = Document(
            id="doc1",
            text="This is a test document about hypertension.",
            metadata=DocumentMetadata(
                source="test",
                title="Test Document",
                category="cardiovascular"
            )
        )
        
        assert doc.id == "doc1"
        assert doc.text == "This is a test document about hypertension."
        assert doc.metadata.source == "test"
        assert doc.metadata.title == "Test Document"
        assert doc.metadata.category == "cardiovascular"

    def test_document_to_dict(self):
        """Test converting document to dictionary."""
        doc = Document(
            id="doc1",
            text="This is a test document about hypertension.",
            metadata=DocumentMetadata(
                source="test",
                title="Test Document",
                category="cardiovascular",
                author="Dr. Test",
                created_at="2023-01-01"
            )
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["id"] == "doc1"
        assert doc_dict["text"] == "This is a test document about hypertension."
        assert doc_dict["metadata"]["source"] == "test"
        assert doc_dict["metadata"]["title"] == "Test Document"
        assert doc_dict["metadata"]["category"] == "cardiovascular"
        assert doc_dict["metadata"]["author"] == "Dr. Test"
        assert doc_dict["metadata"]["created_at"] == "2023-01-01"

    def test_document_from_dict(self):
        """Test creating document from dictionary."""
        doc_dict = {
            "id": "doc1",
            "text": "This is a test document about hypertension.",
            "metadata": {
                "source": "test",
                "title": "Test Document",
                "category": "cardiovascular",
                "author": "Dr. Test",
                "created_at": "2023-01-01"
            }
        }
        
        doc = Document.from_dict(doc_dict)
        
        assert doc.id == "doc1"
        assert doc.text == "This is a test document about hypertension."
        assert doc.metadata.source == "test"
        assert doc.metadata.title == "Test Document"
        assert doc.metadata.category == "cardiovascular"
        assert doc.metadata.author == "Dr. Test"
        assert doc.metadata.created_at == "2023-01-01"

    def test_document_with_minimal_metadata(self):
        """Test document with minimal metadata."""
        # Create document with minimal metadata
        doc = Document(
            id="doc2",
            text="Another test document.",
            metadata=DocumentMetadata(
                source="test"
            )
        )
        
        # Convert to dict and back
        doc_dict = doc.to_dict()
        restored_doc = Document.from_dict(doc_dict)
        
        # Verify
        assert restored_doc.id == "doc2"
        assert restored_doc.text == "Another test document."
        assert restored_doc.metadata.source == "test"
        assert restored_doc.metadata.title is None
        assert restored_doc.metadata.category is None


class TestMedicalKnowledgeBase:
    """Test cases for the MedicalKnowledgeBase class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for storing knowledge base data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.kb_dir = Path(self.temp_dir.name)
        
        # Create necessary subdirectories
        self.embeddings_dir = self.kb_dir / "embeddings"
        self.metadata_dir = self.kb_dir / "metadata"
        self.embeddings_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Create empty metadata file
        with open(self.metadata_dir / "documents.json", "w") as f:
            json.dump({}, f)

    def teardown_method(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    @patch("ai.knowledge.medical_knowledge_base.MedicalLLMConnector")
    def test_initialization(self, mock_llm_connector):
        """Test knowledge base initialization."""
        # Setup mock
        mock_connector = MagicMock()
        mock_llm_connector.return_value = mock_connector
        
        # Create knowledge base
        kb = MedicalKnowledgeBase(
            knowledge_dir=str(self.kb_dir),
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            },
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Verify
        assert kb.knowledge_dir == str(self.kb_dir)
        assert kb.embeddings_dir == str(self.embeddings_dir)
        assert kb.metadata_dir == str(self.metadata_dir)
        assert kb.chunk_size == 500
        assert kb.chunk_overlap == 50
        assert kb.documents == {}

    @patch("ai.knowledge.medical_knowledge_base.MedicalLLMConnector")
    def test_initialization_missing_dir(self, mock_llm_connector):
        """Test that knowledge base creates missing directories."""
        # Setup mock
        mock_connector = MagicMock()
        mock_llm_connector.return_value = mock_connector
        
        # Create a new directory path that doesn't exist
        new_dir = self.kb_dir / "new_kb"
        
        # Create knowledge base with non-existent directory
        kb = MedicalKnowledgeBase(
            knowledge_dir=str(new_dir),
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            }
        )
        
        # Verify directories were created
        assert os.path.exists(new_dir)
        assert os.path.exists(new_dir / "embeddings")
        assert os.path.exists(new_dir / "metadata")
        assert os.path.exists(new_dir / "metadata" / "documents.json")

    @patch("ai.knowledge.medical_knowledge_base.MedicalLLMConnector")
    def test_add_document(self, mock_llm_connector):
        """Test adding a document to the knowledge base."""
        # Setup mocks
        mock_connector = MagicMock()
        mock_connector.generate_embeddings.return_value = np.random.rand(1536).tolist()
        mock_llm_connector.return_value = mock_connector
        
        # Create knowledge base
        kb = MedicalKnowledgeBase(
            knowledge_dir=str(self.kb_dir),
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            }
        )
        
        # Add a document
        doc_id = kb.add_document(
            text="This is a test document about diabetes mellitus. "
                 "Diabetes is a metabolic disease that causes high blood sugar.",
            metadata=DocumentMetadata(
                source="test",
                title="Diabetes Overview",
                category="endocrine"
            )
        )
        
        # Verify document was added
        assert doc_id in kb.documents
        assert "Diabetes" in kb.documents[doc_id].text
        assert kb.documents[doc_id].metadata.title == "Diabetes Overview"
        assert kb.documents[doc_id].metadata.category == "endocrine"
        
        # Verify document was saved to disk
        metadata_file = self.metadata_dir / "documents.json"
        assert os.path.exists(metadata_file)
        
        with open(metadata_file, "r") as f:
            saved_metadata = json.load(f)
            assert doc_id in saved_metadata
            assert saved_metadata[doc_id]["metadata"]["title"] == "Diabetes Overview"
        
        # Verify embeddings were saved
        for chunk_id in kb.document_chunks[doc_id]:
            embedding_file = self.embeddings_dir / f"{chunk_id}.npy"
            assert os.path.exists(embedding_file)

    @patch("ai.knowledge.medical_knowledge_base.MedicalLLMConnector")
    def test_get_document(self, mock_llm_connector):
        """Test retrieving a document from the knowledge base."""
        # Setup mocks
        mock_connector = MagicMock()
        mock_connector.generate_embeddings.return_value = np.random.rand(1536).tolist()
        mock_llm_connector.return_value = mock_connector
        
        # Create knowledge base
        kb = MedicalKnowledgeBase(
            knowledge_dir=str(self.kb_dir),
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            }
        )
        
        # Add a document
        doc_id = kb.add_document(
            text="This is a test document about hypertension.",
            metadata=DocumentMetadata(
                source="test",
                title="Hypertension Overview",
                category="cardiovascular"
            )
        )
        
        # Get the document
        doc = kb.get_document(doc_id)
        
        # Verify
        assert doc.id == doc_id
        assert doc.text == "This is a test document about hypertension."
        assert doc.metadata.title == "Hypertension Overview"
        assert doc.metadata.category == "cardiovascular"

    @patch("ai.knowledge.medical_knowledge_base.MedicalLLMConnector")
    def test_delete_document(self, mock_llm_connector):
        """Test deleting a document from the knowledge base."""
        # Setup mocks
        mock_connector = MagicMock()
        mock_connector.generate_embeddings.return_value = np.random.rand(1536).tolist()
        mock_llm_connector.return_value = mock_connector
        
        # Create knowledge base
        kb = MedicalKnowledgeBase(
            knowledge_dir=str(self.kb_dir),
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            }
        )
        
        # Add a document
        doc_id = kb.add_document(
            text="This is a test document about hypertension.",
            metadata=DocumentMetadata(
                source="test",
                title="Hypertension Overview",
                category="cardiovascular"
            )
        )
        
        # Verify document exists
        assert doc_id in kb.documents
        
        # Delete the document
        kb.delete_document(doc_id)
        
        # Verify document was deleted
        assert doc_id not in kb.documents
        
        # Verify document was removed from disk
        metadata_file = self.metadata_dir / "documents.json"
        with open(metadata_file, "r") as f:
            saved_metadata = json.load(f)
            assert doc_id not in saved_metadata

    @patch("ai.knowledge.medical_knowledge_base.MedicalLLMConnector")
    def test_list_documents(self, mock_llm_connector):
        """Test listing all documents in the knowledge base."""
        # Setup mocks
        mock_connector = MagicMock()
        mock_connector.generate_embeddings.return_value = np.random.rand(1536).tolist()
        mock_llm_connector.return_value = mock_connector
        
        # Create knowledge base
        kb = MedicalKnowledgeBase(
            knowledge_dir=str(self.kb_dir),
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            }
        )
        
        # Add multiple documents
        doc1_id = kb.add_document(
            text="Document about hypertension.",
            metadata=DocumentMetadata(
                source="test",
                title="Hypertension",
                category="cardiovascular"
            )
        )
        
        doc2_id = kb.add_document(
            text="Document about diabetes.",
            metadata=DocumentMetadata(
                source="test",
                title="Diabetes",
                category="endocrine"
            )
        )
        
        doc3_id = kb.add_document(
            text="Document about asthma.",
            metadata=DocumentMetadata(
                source="test",
                title="Asthma",
                category="respiratory"
            )
        )
        
        # List all documents
        documents = kb.list_documents()
        
        # Verify
        assert len(documents) == 3
        assert doc1_id in documents
        assert doc2_id in documents
        assert doc3_id in documents
        
        # List documents by category
        cardio_docs = kb.list_documents(category="cardiovascular")
        assert len(cardio_docs) == 1
        assert doc1_id in cardio_docs
        
        respiratory_docs = kb.list_documents(category="respiratory")
        assert len(respiratory_docs) == 1
        assert doc3_id in respiratory_docs

    @patch("ai.knowledge.medical_knowledge_base.MedicalLLMConnector")
    def test_search_documents(self, mock_llm_connector):
        """Test searching documents in the knowledge base."""
        # Setup mocks
        mock_connector = MagicMock()
        # Return different embeddings for different texts
        mock_connector.generate_embeddings.side_effect = lambda text: (
            np.random.rand(1536).tolist() if "irrelevant" in text else
            np.array([0.1] * 1536).tolist() if "diabetes" in text else
            np.array([0.9] * 1536).tolist()
        )
        mock_llm_connector.return_value = mock_connector
        
        # Create knowledge base
        kb = MedicalKnowledgeBase(
            knowledge_dir=str(self.kb_dir),
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            }
        )
        
        # Add documents with varying relevance
        kb.add_document(
            text="Document about diabetes. Diabetes is a metabolic disease that causes high blood sugar.",
            metadata=DocumentMetadata(source="test", title="Diabetes", category="endocrine")
        )
        
        kb.add_document(
            text="Document about hypertension. Hypertension is high blood pressure.",
            metadata=DocumentMetadata(source="test", title="Hypertension", category="cardiovascular")
        )
        
        kb.add_document(
            text="Document about something irrelevant.",
            metadata=DocumentMetadata(source="test", title="Irrelevant", category="other")
        )
        
        # Override the similarity calculation for testing
        def mock_calculate_similarity(query_embedding, document_embedding):
            # Use simple comparison for testing
            if np.mean(query_embedding) == 0.1 and np.mean(document_embedding) == 0.1:
                return 0.95  # High similarity for diabetes query to diabetes document
            elif np.mean(query_embedding) == 0.9 and np.mean(document_embedding) == 0.9:
                return 0.95  # High similarity for hypertension query to hypertension document
            else:
                return 0.1  # Low similarity for everything else
        
        kb._calculate_similarity = mock_calculate_similarity
        
        # Search for diabetes
        kb._query_embedding = np.array([0.1] * 1536).tolist()
        results = kb.search("What is diabetes?", limit=2)
        
        # Verify diabetes-related document is returned with high relevance
        assert len(results) > 0
        assert any("diabetes" in result.text.lower() for result in results)
        
        # Search for hypertension
        kb._query_embedding = np.array([0.9] * 1536).tolist()
        results = kb.search("What is hypertension?", limit=2)
        
        # Verify hypertension-related document is returned with high relevance
        assert len(results) > 0
        assert any("hypertension" in result.text.lower() for result in results)

    @patch("ai.knowledge.medical_knowledge_base.MedicalLLMConnector")
    def test_save_and_load(self, mock_llm_connector):
        """Test saving and loading the knowledge base."""
        # Setup mocks
        mock_connector = MagicMock()
        mock_connector.generate_embeddings.return_value = np.random.rand(1536).tolist()
        mock_llm_connector.return_value = mock_connector
        
        # Create and populate knowledge base
        kb = MedicalKnowledgeBase(
            knowledge_dir=str(self.kb_dir),
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            }
        )
        
        doc1_id = kb.add_document(
            text="Document about diabetes.",
            metadata=DocumentMetadata(source="test", title="Diabetes", category="endocrine")
        )
        
        # Create a new knowledge base instance and load data
        kb2 = MedicalKnowledgeBase(
            knowledge_dir=str(self.kb_dir),
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            }
        )
        
        # Verify data was loaded
        assert doc1_id in kb2.documents
        assert kb2.documents[doc1_id].metadata.title == "Diabetes"
        assert kb2.documents[doc1_id].metadata.category == "endocrine" 