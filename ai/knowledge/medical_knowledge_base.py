"""
MediNex AI Medical Knowledge Base Module

This module provides the core functionality for building and managing a medical knowledge base
using vector embeddings for semantic search and retrieval. It supports storing, retrieving,
and searching medical documents with relevant metadata.
"""

import os
import json
import uuid
import logging
import shutil
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from ..llm.model_connector import MedicalLLMConnector


class KnowledgeBaseError(Exception):
    """Base exception for knowledge base related errors."""
    pass


@dataclass
class DocumentMetadata:
    """Metadata associated with a document in the knowledge base."""
    source: str
    title: Optional[str] = None
    category: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    url: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A document in the medical knowledge base."""
    id: str
    text: str
    metadata: DocumentMetadata
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a Document from a dictionary representation."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=DocumentMetadata(**data["metadata"])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": asdict(self.metadata)
        }


@dataclass
class SearchResult:
    """A search result with document and relevance score."""
    document: Document
    score: float
    chunk_text: Optional[str] = None


class MedicalKnowledgeBase:
    """
    Medical Knowledge Base for storing and retrieving medical information.
    
    This class provides methods for:
    - Adding documents to the knowledge base
    - Retrieving documents by ID
    - Deleting documents from the knowledge base
    - Searching for relevant documents using semantic search
    - Managing document metadata and embeddings
    """
    
    def __init__(
        self,
        knowledge_dir: str,
        llm_config: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the medical knowledge base.
        
        Args:
            knowledge_dir: Directory to store knowledge base files
            llm_config: Configuration for the LLM connector
            chunk_size: The size of text chunks for embedding
            chunk_overlap: The amount of overlap between chunks
        """
        self.knowledge_dir = knowledge_dir
        self.embeddings_dir = os.path.join(knowledge_dir, "embeddings")
        self.metadata_dir = os.path.join(knowledge_dir, "metadata")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = {}
        self.document_chunks = {}  # Maps document ID to list of chunk IDs
        self.chunk_to_doc = {}     # Maps chunk ID to document ID
        self.chunk_texts = {}      # Maps chunk ID to chunk text
        
        # Create LLM connector for embeddings
        self.llm = MedicalLLMConnector(llm_config)
        
        # Initialize directories
        self._initialize_directories()
        
        # Load existing documents
        self._load_documents()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_directories(self):
        """Initialize directories needed for the knowledge base."""
        # Create main directory if it doesn't exist
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Create metadata file if it doesn't exist
        metadata_file = os.path.join(self.metadata_dir, "documents.json")
        if not os.path.exists(metadata_file):
            with open(metadata_file, "w") as f:
                json.dump({}, f)
    
    def _load_documents(self):
        """Load documents from the metadata file."""
        metadata_file = os.path.join(self.metadata_dir, "documents.json")
        try:
            with open(metadata_file, "r") as f:
                document_data = json.load(f)
                
            # Load documents
            self.documents = {}
            self.document_chunks = {}
            self.chunk_to_doc = {}
            
            for doc_id, doc_info in document_data.items():
                # Create Document object
                self.documents[doc_id] = Document.from_dict(doc_info)
                
                # Load chunk mappings
                chunks_file = os.path.join(self.metadata_dir, f"{doc_id}_chunks.json")
                if os.path.exists(chunks_file):
                    with open(chunks_file, "r") as f:
                        chunk_data = json.load(f)
                        self.document_chunks[doc_id] = chunk_data["chunk_ids"]
                        
                        # Update chunk-to-document mapping
                        for chunk_id in chunk_data["chunk_ids"]:
                            self.chunk_to_doc[chunk_id] = doc_id
                            
                        # Update chunk texts
                        for chunk_id, chunk_text in chunk_data["chunk_texts"].items():
                            self.chunk_texts[chunk_id] = chunk_text
                
            self.logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
            
        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}")
            raise KnowledgeBaseError(f"Failed to load documents: {str(e)}")
    
    def _save_document_metadata(self, doc_id: str):
        """Save document metadata to disk."""
        if doc_id not in self.documents:
            raise KnowledgeBaseError(f"Document {doc_id} not found")
        
        metadata_file = os.path.join(self.metadata_dir, "documents.json")
        try:
            # Load existing metadata
            with open(metadata_file, "r") as f:
                document_data = json.load(f)
            
            # Update with current document
            document_data[doc_id] = self.documents[doc_id].to_dict()
            
            # Save updated metadata
            with open(metadata_file, "w") as f:
                json.dump(document_data, f, indent=2)
            
            # Save chunk information
            if doc_id in self.document_chunks:
                chunks_file = os.path.join(self.metadata_dir, f"{doc_id}_chunks.json")
                
                # Collect chunk texts for this document
                chunk_texts = {
                    chunk_id: self.chunk_texts[chunk_id]
                    for chunk_id in self.document_chunks[doc_id]
                    if chunk_id in self.chunk_texts
                }
                
                # Save chunk data
                chunk_data = {
                    "document_id": doc_id,
                    "chunk_ids": self.document_chunks[doc_id],
                    "chunk_texts": chunk_texts
                }
                
                with open(chunks_file, "w") as f:
                    json.dump(chunk_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving document metadata: {str(e)}")
            raise KnowledgeBaseError(f"Failed to save document metadata: {str(e)}")
    
    def _chunk_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of tuples (chunk_id, chunk_text)
        """
        if not text:
            return []
            
        # Simple chunking by character count
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # If not at the end of the text, try to find a sentence boundary
            if end < len(text):
                # Look for sentence boundaries (., !, ?) followed by space or newline
                for boundary in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                    last_boundary = text.rfind(boundary, start, end)
                    if last_boundary > 0:
                        end = last_boundary + 2  # Include the boundary and space
                        break
            
            # Extract the chunk
            chunk_text = text[start:end].strip()
            
            # Generate a unique ID for this chunk
            chunk_id = str(uuid.uuid4())
            
            # Add to chunks list
            chunks.append((chunk_id, chunk_text))
            
            # Calculate the next start position with overlap
            start = end - self.chunk_overlap
            if start < 0 or start >= len(text):
                break
        
        return chunks
    
    def _calculate_similarity(self, query_embedding: List[float], document_embedding: List[float]) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            query_embedding: The query embedding vector
            document_embedding: The document embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        query_vec = np.array(query_embedding)
        doc_vec = np.array(document_embedding)
        
        # Calculate cosine similarity
        dot_product = np.dot(query_vec, doc_vec)
        query_norm = np.linalg.norm(query_vec)
        doc_norm = np.linalg.norm(doc_vec)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
            
        return dot_product / (query_norm * doc_norm)
    
    def add_document(
        self,
        text: str,
        metadata: DocumentMetadata,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            text: The document text
            metadata: Document metadata
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            Document ID
        """
        # Generate document ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Create Document object
        document = Document(
            id=doc_id,
            text=text,
            metadata=metadata
        )
        
        # Add to documents
        self.documents[doc_id] = document
        
        # Chunk the document text
        chunks = self._chunk_text(text)
        chunk_ids = []
        
        try:
            # Connect to LLM for embeddings
            self.llm.connect()
            
            # Process chunks
            for chunk_id, chunk_text in chunks:
                # Generate embedding
                embedding = self.llm.generate_embeddings(chunk_text)
                
                # Save embedding
                embedding_path = os.path.join(self.embeddings_dir, f"{chunk_id}.npy")
                np.save(embedding_path, np.array(embedding))
                
                # Update mappings
                chunk_ids.append(chunk_id)
                self.chunk_to_doc[chunk_id] = doc_id
                self.chunk_texts[chunk_id] = chunk_text
            
            # Update document chunks
            self.document_chunks[doc_id] = chunk_ids
            
            # Save document metadata
            self._save_document_metadata(doc_id)
            
            self.logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            
            return doc_id
            
        except Exception as e:
            # Clean up any created files
            for chunk_id, _ in chunks:
                embedding_path = os.path.join(self.embeddings_dir, f"{chunk_id}.npy")
                if os.path.exists(embedding_path):
                    os.remove(embedding_path)
            
            # Remove from in-memory storage
            if doc_id in self.documents:
                del self.documents[doc_id]
            
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_to_doc:
                    del self.chunk_to_doc[chunk_id]
                if chunk_id in self.chunk_texts:
                    del self.chunk_texts[chunk_id]
            
            if doc_id in self.document_chunks:
                del self.document_chunks[doc_id]
            
            self.logger.error(f"Error adding document: {str(e)}")
            raise KnowledgeBaseError(f"Failed to add document: {str(e)}")
    
    def get_document(self, doc_id: str) -> Document:
        """
        Get a document by ID.
        
        Args:
            doc_id: The document ID
            
        Returns:
            Document object
            
        Raises:
            KnowledgeBaseError: If document not found
        """
        if doc_id not in self.documents:
            raise KnowledgeBaseError(f"Document {doc_id} not found")
        
        return self.documents[doc_id]
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the knowledge base.
        
        Args:
            doc_id: The document ID
            
        Returns:
            Boolean indicating success
            
        Raises:
            KnowledgeBaseError: If document not found
        """
        if doc_id not in self.documents:
            raise KnowledgeBaseError(f"Document {doc_id} not found")
        
        try:
            # Get chunk IDs for this document
            chunk_ids = self.document_chunks.get(doc_id, [])
            
            # Delete embeddings
            for chunk_id in chunk_ids:
                embedding_path = os.path.join(self.embeddings_dir, f"{chunk_id}.npy")
                if os.path.exists(embedding_path):
                    os.remove(embedding_path)
                
                # Remove from mappings
                if chunk_id in self.chunk_to_doc:
                    del self.chunk_to_doc[chunk_id]
                if chunk_id in self.chunk_texts:
                    del self.chunk_texts[chunk_id]
            
            # Delete chunks metadata file
            chunks_file = os.path.join(self.metadata_dir, f"{doc_id}_chunks.json")
            if os.path.exists(chunks_file):
                os.remove(chunks_file)
            
            # Remove from document chunks mapping
            if doc_id in self.document_chunks:
                del self.document_chunks[doc_id]
            
            # Remove from documents
            del self.documents[doc_id]
            
            # Update documents metadata file
            metadata_file = os.path.join(self.metadata_dir, "documents.json")
            with open(metadata_file, "r") as f:
                document_data = json.load(f)
            
            if doc_id in document_data:
                del document_data[doc_id]
            
            with open(metadata_file, "w") as f:
                json.dump(document_data, f, indent=2)
            
            self.logger.info(f"Deleted document {doc_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {str(e)}")
            raise KnowledgeBaseError(f"Failed to delete document {doc_id}: {str(e)}")
    
    def update_document(self, doc_id: str, text: str, metadata: DocumentMetadata) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id: The document ID
            text: Updated document text
            metadata: Updated document metadata
            
        Returns:
            Boolean indicating success
            
        Raises:
            KnowledgeBaseError: If document not found
        """
        if doc_id not in self.documents:
            raise KnowledgeBaseError(f"Document {doc_id} not found")
        
        try:
            # Delete existing document (will clean up chunks and embeddings)
            self.delete_document(doc_id)
            
            # Add new document with same ID
            self.add_document(text, metadata, doc_id=doc_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document {doc_id}: {str(e)}")
            raise KnowledgeBaseError(f"Failed to update document {doc_id}: {str(e)}")
    
    def list_documents(self, category: Optional[str] = None) -> Dict[str, Document]:
        """
        List all documents or filter by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Dictionary of document IDs to Document objects
        """
        if category is None:
            return self.documents
        
        # Filter by category
        return {
            doc_id: doc for doc_id, doc in self.documents.items()
            if doc.metadata.category == category
        }
    
    def search(
        self,
        query: str,
        limit: int = 10,
        category: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            category: Optional category to filter by
            
        Returns:
            List of search results sorted by relevance
        """
        try:
            # Connect to LLM for embeddings
            self.llm.connect()
            
            # Generate query embedding
            query_embedding = self.llm.generate_embeddings(query)
            self._query_embedding = query_embedding  # Store for testing
            
            # Calculate similarities for all chunks
            similarities = []
            
            for chunk_id in self.chunk_to_doc:
                # Load embedding
                embedding_path = os.path.join(self.embeddings_dir, f"{chunk_id}.npy")
                if not os.path.exists(embedding_path):
                    continue
                
                chunk_embedding = np.load(embedding_path).tolist()
                doc_id = self.chunk_to_doc[chunk_id]
                
                # Skip if category filter is applied and document doesn't match
                if category is not None:
                    doc = self.documents[doc_id]
                    if doc.metadata.category != category:
                        continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(query_embedding, chunk_embedding)
                
                # Add to results
                similarities.append((chunk_id, doc_id, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Convert to search results
            results = []
            added_docs = set()
            
            for chunk_id, doc_id, score in similarities[:limit]:
                # Get document
                document = self.documents[doc_id]
                
                # Get chunk text
                chunk_text = self.chunk_texts.get(chunk_id, "")
                
                # Create search result
                result = SearchResult(
                    document=document,
                    score=score,
                    chunk_text=chunk_text
                )
                
                # Add to results if not already added
                if doc_id not in added_docs:
                    results.append(result)
                    added_docs.add(doc_id)
                
                # Stop if we have enough results
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}")
            raise KnowledgeBaseError(f"Search failed: {str(e)}")
    
    def reset(self):
        """Reset the knowledge base, deleting all documents and embeddings."""
        try:
            # Clear in-memory data
            self.documents = {}
            self.document_chunks = {}
            self.chunk_to_doc = {}
            self.chunk_texts = {}
            
            # Delete files
            for filename in os.listdir(self.embeddings_dir):
                file_path = os.path.join(self.embeddings_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            for filename in os.listdir(self.metadata_dir):
                if filename != "documents.json":
                    file_path = os.path.join(self.metadata_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            
            # Reset documents metadata file
            metadata_file = os.path.join(self.metadata_dir, "documents.json")
            with open(metadata_file, "w") as f:
                json.dump({}, f)
            
            self.logger.info("Reset knowledge base")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting knowledge base: {str(e)}")
            raise KnowledgeBaseError(f"Failed to reset knowledge base: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        total_chunks = len(self.chunk_to_doc)
        total_embeddings = 0
        
        # Count embeddings
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith(".npy"):
                total_embeddings += 1
        
        # Calculate categories
        categories = {}
        for doc in self.documents.values():
            category = doc.metadata.category
            if category:
                categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "categories": categories,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        } 