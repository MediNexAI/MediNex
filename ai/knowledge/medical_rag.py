"""
MediNex AI Medical RAG System

This module implements a Retrieval-Augmented Generation system for medical knowledge.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..llm.model_connector import MedicalLLMConnector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a medical document."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class MedicalRAG:
    """
    Medical Retrieval-Augmented Generation system.
    """
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the Medical RAG system.
        
        Args:
            llm_config: Configuration for the LLM
            embedding_model: Name of the embedding model
            index_path: Path to save/load the FAISS index
            cache_dir: Directory for caching
        """
        self.llm = MedicalLLMConnector(llm_config)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_path = index_path
        self.cache_dir = cache_dir
        
        # Initialize document storage
        self.documents: List[Document] = []
        self.index = None
        
        # Create cache directory if needed
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing index if available
        if index_path and os.path.exists(index_path):
            self._load_index()
    
    def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        # Generate document ID
        doc_id = f"doc_{len(self.documents)}"
        
        # Generate embedding
        embedding = self.embedding_model.encode([text])[0]
        
        # Create document
        doc = Document(
            id=doc_id,
            text=text,
            metadata=metadata,
            embedding=embedding
        )
        
        # Add to storage
        self.documents.append(doc)
        
        # Update index
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[0])
        self.index.add(np.array([embedding]))
        
        # Save index if path specified
        if self.index_path:
            self._save_index()
        
        return doc_id
    
    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.6
    ) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of relevant documents
        """
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search index
        D, I = self.index.search(
            np.array([query_embedding]),
            k=min(k, len(self.documents))
        )
        
        # Filter by threshold and get documents
        results = []
        for score, idx in zip(D[0], I[0]):
            if score <= threshold:  # Lower distance means higher similarity
                results.append(self.documents[idx])
        
        return results
    
    def query(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Query the system with RAG.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            threshold: Similarity threshold
            
        Returns:
            Response dictionary
        """
        # Search for relevant documents
        relevant_docs = self.search(query, k, threshold)
        
        if not relevant_docs:
            # No relevant documents found, use base LLM
            response = self.llm.generate_text(query)
            return {
                "answer": response,
                "sources": [],
                "used_rag": False
            }
        
        # Prepare context from relevant documents
        context = "\n\n".join([
            f"Source {i+1}:\n{doc.text}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Generate response with context
        response = self.llm.generate_with_context(query, context)
        
        # Prepare source information
        sources = [
            {
                "id": doc.id,
                "metadata": doc.metadata
            }
            for doc in relevant_docs
        ]
        
        return {
            "answer": response,
            "sources": sources,
            "used_rag": True
        }
    
    def _save_index(self) -> None:
        """Save the FAISS index to disk."""
        if self.index and self.index_path:
            faiss.write_index(self.index, self.index_path)
            
            # Save documents
            docs_path = self.index_path + ".docs"
            with open(docs_path, "w") as f:
                json.dump(
                    [
                        {
                            "id": doc.id,
                            "text": doc.text,
                            "metadata": doc.metadata
                        }
                        for doc in self.documents
                    ],
                    f
                )
    
    def _load_index(self) -> None:
        """Load the FAISS index from disk."""
        if self.index_path and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            
            # Load documents
            docs_path = self.index_path + ".docs"
            if os.path.exists(docs_path):
                with open(docs_path, "r") as f:
                    docs_data = json.load(f)
                    
                self.documents = []
                for doc_data in docs_data:
                    # Generate embedding
                    embedding = self.embedding_model.encode([doc_data["text"]])[0]
                    
                    # Create document
                    doc = Document(
                        id=doc_data["id"],
                        text=doc_data["text"],
                        metadata=doc_data["metadata"],
                        embedding=embedding
                    )
                    self.documents.append(doc)