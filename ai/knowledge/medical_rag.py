"""
MediNex AI Medical Retrieval-Augmented Generation (RAG) Module

This module provides a retrieval-augmented generation system specialized for medical domains,
enhancing LLM responses by retrieving relevant medical knowledge from the knowledge base.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from ..llm.model_connector import MedicalLLMConnector
from .medical_knowledge_base import MedicalKnowledgeBase, SearchResult, DocumentMetadata


class MedicalRAG:
    """
    Medical Retrieval-Augmented Generation (RAG) system.
    
    This class provides methods for:
    - Retrieving relevant medical knowledge from the knowledge base
    - Generating responses enhanced with retrieved medical knowledge
    - Tracking sources of information used in generated responses
    """
    
    def __init__(
        self,
        knowledge_base: MedicalKnowledgeBase,
        llm_config: Dict[str, Any],
        max_retrieval_count: int = 5
    ):
        """
        Initialize the medical RAG system.
        
        Args:
            knowledge_base: The medical knowledge base to use for retrieval
            llm_config: Configuration for the LLM connector
            max_retrieval_count: Maximum number of documents to retrieve per query
        """
        self.knowledge_base = knowledge_base
        self.llm = MedicalLLMConnector(llm_config)
        self.max_retrieval_count = max_retrieval_count
        self.logger = logging.getLogger(__name__)
    
    def _retrieve_relevant_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        min_score: float = 0.7
    ) -> List[SearchResult]:
        """
        Retrieve relevant knowledge from the knowledge base.
        
        Args:
            query: The user query
            category: Optional category to filter by
            min_score: Minimum relevance score (0.0-1.0)
            
        Returns:
            List of relevant search results
        """
        try:
            # Search the knowledge base
            results = self.knowledge_base.search(
                query=query,
                limit=self.max_retrieval_count,
                category=category
            )
            
            # Filter by minimum score
            filtered_results = [r for r in results if r.score >= min_score]
            
            self.logger.info(f"Retrieved {len(filtered_results)} relevant documents for query: {query}")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error retrieving knowledge: {str(e)}")
            return []
    
    def _format_context(self, results: List[SearchResult]) -> str:
        """
        Format search results into a context string for the LLM.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            # Use chunk text if available, otherwise use full document text
            text_to_use = result.chunk_text if result.chunk_text else result.document.text
            
            # Format context entry
            entry = f"[Source {i}] "
            if result.document.metadata.title:
                entry += f"{result.document.metadata.title}: "
            entry += text_to_use
            
            context_parts.append(entry)
        
        return "\n\n".join(context_parts)
    
    def _format_source_info(self, results: List[SearchResult]) -> str:
        """
        Format source information for the response.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted source information
        """
        if not results:
            return ""
        
        source_parts = ["Sources:"]
        
        for i, result in enumerate(results, 1):
            meta = result.document.metadata
            source = f"[{i}] "
            
            # Add title if available
            if meta.title:
                source += f"{meta.title}"
            
            # Add author if available
            if meta.author:
                source += f" by {meta.author}"
            
            # Add source information
            source += f" (Source: {meta.source})"
            
            # Add URL if available
            if meta.url:
                source += f" - {meta.url}"
            
            source_parts.append(source)
        
        return "\n".join(source_parts)
    
    def query(
        self,
        query: str,
        category: Optional[str] = None,
        include_sources: bool = True,
        min_score: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system to get a response enhanced with medical knowledge.
        
        Args:
            query: The user query
            category: Optional medical category to filter by
            include_sources: Whether to include source information in the response
            min_score: Minimum relevance score (0.0-1.0)
            system_prompt: Optional system prompt to use
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Connect to the LLM
            self.llm.connect()
            
            # Retrieve relevant knowledge
            retrieved_results = self._retrieve_relevant_knowledge(
                query=query,
                category=category,
                min_score=min_score
            )
            
            # Format context
            context = self._format_context(retrieved_results)
            
            # Set default system prompt if not provided
            if system_prompt is None:
                system_prompt = (
                    "You are a medical assistant providing accurate information based on the "
                    "medical knowledge provided. Answer questions using the given context. "
                    "If the context doesn't contain relevant information, acknowledge the "
                    "limitations and provide a general response based on verified medical knowledge. "
                    "Be precise, factual, and medical in your responses."
                )
            
            # Generate response with context
            if context:
                response_text = self.llm.generate_with_context(query, context)
            else:
                # No relevant context found, generate a response without context
                self.logger.info("No relevant context found, generating response without RAG")
                response_text = self.llm.generate_text(
                    f"{system_prompt}\n\nUser question: {query}"
                )
            
            # Format source information if requested
            source_info = ""
            if include_sources and retrieved_results:
                source_info = self._format_source_info(retrieved_results)
                response_text = f"{response_text}\n\n{source_info}"
            
            # Build response
            response = {
                "query": query,
                "response": response_text,
                "has_relevant_context": len(retrieved_results) > 0,
                "sources_count": len(retrieved_results),
                "sources": [
                    {
                        "document_id": result.document.id,
                        "title": result.document.metadata.title,
                        "source": result.document.metadata.source,
                        "score": result.score,
                        "category": result.document.metadata.category
                    }
                    for result in retrieved_results
                ] if include_sources else []
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in RAG query: {str(e)}")
            return {
                "query": query,
                "response": "I apologize, but I encountered an error while processing your query. Please try again later.",
                "error": str(e),
                "has_relevant_context": False,
                "sources_count": 0,
                "sources": []
            }
    
    def batch_query(
        self,
        queries: List[str],
        category: Optional[str] = None,
        include_sources: bool = True,
        min_score: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            category: Optional medical category to filter by
            include_sources: Whether to include source information in the response
            min_score: Minimum relevance score (0.0-1.0)
            system_prompt: Optional system prompt to use
            
        Returns:
            List of response dictionaries
        """
        responses = []
        
        for query in queries:
            response = self.query(
                query=query,
                category=category,
                include_sources=include_sources,
                min_score=min_score,
                system_prompt=system_prompt
            )
            responses.append(response)
        
        return responses
    
    def add_to_knowledge_base(
        self,
        text: str,
        metadata: DocumentMetadata
    ) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        return self.knowledge_base.add_document(text, metadata)
    
    def evaluate_accuracy(
        self,
        query: str,
        reference_answer: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the accuracy of a RAG response against a reference answer.
        
        Args:
            query: The user query
            reference_answer: The reference answer to compare against
            category: Optional medical category to filter by
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Connect to the LLM
            self.llm.connect()
            
            # Generate RAG response
            rag_response = self.query(
                query=query,
                category=category,
                include_sources=False
            )
            
            # Format evaluation prompt
            eval_prompt = (
                "You are evaluating the factual accuracy of a response to a medical question. "
                "Compare the generated response to the reference answer and rate the accuracy "
                "on a scale of 1-10, where 10 is perfectly accurate and 1 is completely inaccurate. "
                "Focus on factual correctness, not style or verbosity.\n\n"
                f"Question: {query}\n\n"
                f"Generated response: {rag_response['response']}\n\n"
                f"Reference answer: {reference_answer}\n\n"
                "Please provide:\n"
                "1. A numerical score from 1-10\n"
                "2. A brief explanation of your rating\n"
                "3. Note any factual errors or missing key information\n\n"
                "Format your response as a JSON object with fields: score, explanation, and errors."
            )
            
            # Generate evaluation
            evaluation_text = self.llm.generate_text(eval_prompt)
            
            # Return evaluation results
            return {
                "query": query,
                "rag_response": rag_response["response"],
                "reference_answer": reference_answer,
                "evaluation": evaluation_text,
                "has_relevant_context": rag_response["has_relevant_context"],
                "sources_count": rag_response["sources_count"]
            }
            
        except Exception as e:
            self.logger.error(f"Error in accuracy evaluation: {str(e)}")
            return {
                "query": query,
                "rag_response": "",
                "reference_answer": reference_answer,
                "evaluation": "",
                "error": str(e),
                "has_relevant_context": False,
                "sources_count": 0
            }