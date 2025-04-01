"""
MediNex AI LLM Connector Module

This module provides a unified interface for connecting to various large language model 
providers, including OpenAI, Anthropic, HuggingFace, and local models.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any, Generator, Callable
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import providers
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic as AnthropicClient
except ImportError:
    AnthropicClient = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    torch = None


# Define custom exceptions
class APIKeyError(Exception):
    """Raised when API key is missing or invalid."""
    pass


class ModelConnectionError(Exception):
    """Raised when connection to the model fails."""
    pass


class UnsupportedProviderError(Exception):
    """Raised when an unsupported provider is specified."""
    pass


# Helper functions for model loading
def load_huggingface_model(model_name: str):
    """Load a model from HuggingFace."""
    if AutoModelForCausalLM is None:
        raise ImportError("transformers package is not installed")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
    )
    
    return generator


def load_local_model(model_path: str):
    """Load a model from a local path."""
    if AutoModelForCausalLM is None:
        raise ImportError("transformers package is not installed")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
    )
    
    return generator


class MedicalLLMConnector:
    """
    A unified connector for various LLM providers specialized for medical applications.
    
    Supports:
    - OpenAI (GPT models)
    - Anthropic (Claude models)
    - HuggingFace models
    - Local models
    
    Provides methods for generating text, generating with context for RAG applications,
    and generating embeddings.
    """
    
    SUPPORTED_PROVIDERS = ["openai", "anthropic", "huggingface", "local"]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM connector.
        
        Args:
            config: A dictionary containing configuration parameters:
                - provider: The LLM provider (openai, anthropic, huggingface, local)
                - model: The model name or path
                - temperature: The temperature for generation (default: 0.1)
                - api_key: API key (optional, can be set as environment variable)
        """
        self.provider = config.get("provider", "openai").lower()
        self.model = config.get("model")
        self.temperature = config.get("temperature", 0.1)
        self.api_key = config.get("api_key") or os.getenv(f"{self.provider.upper()}_API_KEY")
        
        # Validate provider
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise UnsupportedProviderError(
                f"Provider '{self.provider}' is not supported. "
                f"Supported providers: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )
        
        # For API-based providers, verify API key
        if self.provider in ["openai", "anthropic"] and not self.api_key:
            raise APIKeyError(f"API key is required for {self.provider}")
        
        # Initialize client to None
        self.client = None
        self.logger = logging.getLogger(__name__)
        self.is_connected = False
        
        # Medical context template
        self.medical_context = """You are an AI medical assistant. Provide accurate, 
        evidence-based medical information. If unsure, acknowledge limitations."""
    
    def connect(self):
        """Connect to the specified LLM provider."""
        try:
            if self.provider == "openai":
                if OpenAI is None:
                    raise ImportError("openai package is not installed")
                self.client = OpenAI(api_key=self.api_key)
            
            elif self.provider == "anthropic":
                if AnthropicClient is None:
                    raise ImportError("anthropic package is not installed")
                self.client = AnthropicClient(api_key=self.api_key)
            
            elif self.provider == "huggingface":
                self.client = load_huggingface_model(self.model)
            
            elif self.provider == "local":
                self.client = load_local_model(self.model)
            
            self.is_connected = True
            self.logger.info(f"Successfully connected to {self.provider} provider")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.provider}: {str(e)}")
            raise ModelConnectionError(f"Failed to connect to {self.provider}: {str(e)}")
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The text prompt to send to the model
            
        Returns:
            Generated text response
        """
        if not self.is_connected:
            self.connect()
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.medical_context},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=self.temperature,
                    system=self.medical_context,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.provider in ["huggingface", "local"]:
                # Direct generation for HuggingFace and local models
                response = self.client.generate(prompt, temperature=self.temperature)
                return response
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            raise
    
    def generate_with_context(self, query: str, context: str) -> str:
        """
        Generate text with retrieval-augmented generation (RAG) by providing context.
        
        Args:
            query: The user query
            context: The retrieved context to augment generation
            
        Returns:
            Generated text response
        """
        # Format the prompt with context
        rag_prompt = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context}\n"
            f"---------------------\n"
            f"Given the context information and not prior knowledge, answer the question: {query}"
        )
        
        return self.generate_text(rag_prompt)
    
    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            A list of floating point numbers representing the embedding vector
        """
        if not self.client:
            raise ModelConnectionError("Not connected to any LLM provider. Call connect() first.")
        
        try:
            if self.provider == "openai":
                # Default to text-embedding-ada-002 if not specified
                embedding_model = "text-embedding-ada-002"
                response = self.client.embeddings.create(
                    model=embedding_model,
                    input=text
                )
                return response.data[0].embedding
            
            elif self.provider == "anthropic":
                raise NotImplementedError("Embeddings are not yet supported for Anthropic provider")
            
            elif self.provider in ["huggingface", "local"]:
                raise NotImplementedError("Embeddings are not yet implemented for this provider")
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        Args:
            query: The query or prompt to send to the model
            context: Optional context to include with the query
            system_prompt: Optional system prompt to guide the model's behavior
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Dictionary containing the response text and metadata
        """
        if not self.client:
            return {"error": "LLM not initialized", "success": False}
        
        # Start timing
        start_time = time.time()
        
        try:
            # Prepare the prompt with context if provided
            prompt = self._prepare_prompt(query, context, system_prompt)
            
            # Generate response based on provider
            if self.provider == 'openai':
                response = self._generate_openai_response(prompt, temperature, max_tokens)
            elif self.provider == 'anthropic':
                response = self._generate_anthropic_response(prompt, temperature, max_tokens)
            elif self.provider == 'huggingface':
                response = self._generate_huggingface_response(prompt, temperature, max_tokens)
            elif self.provider == 'local':
                response = self._generate_local_response(prompt, temperature, max_tokens)
            else:
                return {"error": f"Unsupported provider: {self.provider}", "success": False}
            
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            # Return the response with metadata
            return {
                "text": response,
                "model": self.model,
                "success": True,
                "timing": {
                    "total_seconds": elapsed_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"error": str(e), "success": False, "timing": {"total_seconds": time.time() - start_time}}
    
    def _prepare_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Prepare the prompt for the model.
        
        Args:
            query: The query or prompt to send to the model
            context: Optional context to include with the query
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            Formatted prompt based on the provider
        """
        # Format depending on provider
        if self.provider == 'openai':
            # For OpenAI, return messages array
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add context as system message if provided
            if context:
                messages.append({"role": "system", "content": f"Context information:\n{context}"})
            
            # Add user query
            messages.append({"role": "user", "content": query})
            
            return messages
            
        elif self.provider == 'anthropic':
            # For Anthropic, format using appropriate system/user delimiters
            prompt_parts = []
            
            # Add system instruction if provided
            if system_prompt:
                prompt_parts.append(system_prompt)
            
            # Add context if provided
            if context:
                prompt_parts.append(f"Context information:\n{context}")
            
            # Add query
            prompt_parts.append(query)
            
            # Return structured format for Anthropic API
            return {
                "system": system_prompt if system_prompt else "",
                "messages": [{"role": "user", "content": query if not context else f"{context}\n\n{query}"}]
            }
            
        elif self.provider == 'huggingface' or self.provider == 'local':
            # For Hugging Face and local models, create a text prompt
            prompt_parts = []
            
            # Add system instruction if provided
            if system_prompt:
                prompt_parts.append(f"System: {system_prompt}")
            
            # Add context if provided
            if context:
                prompt_parts.append(f"Context: {context}")
            
            # Add query
            prompt_parts.append(f"User: {query}")
            prompt_parts.append("Assistant:")
            
            return "\n\n".join(prompt_parts)
            
        else:
            # Default case - simple text prompt
            prompt_parts = []
            
            if system_prompt:
                prompt_parts.append(system_prompt)
                
            if context:
                prompt_parts.append(context)
                
            prompt_parts.append(query)
            
            return "\n\n".join(prompt_parts)
    
    def _generate_openai_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            messages: List of message dictionaries (role and content)
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic_response(
        self,
        prompt: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using Anthropic API.
        
        Args:
            prompt: Formatted prompt with system and messages
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Response text
        """
        response = self.client.messages.create(
            model=self.model,
            system=prompt["system"],
            messages=prompt["messages"],
            temperature=temperature,
            max_tokens=max_tokens or 1024
        )
        
        return response.content[0].text
    
    def _generate_huggingface_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using Hugging Face Inference API.
        
        Args:
            prompt: Formatted text prompt
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Response text
        """
        response = self.client.generate(
            prompt,
            temperature=temperature,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0
        )
        
        return response
    
    def _generate_local_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using a local model.
        
        Args:
            prompt: Formatted text prompt
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Response text
        """
        # This implementation depends on the model type
        if isinstance(self.client, object) and hasattr(self.client, 'create_completion'):
            # llama-cpp-python interface
            response = self.client.create_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens or 512
            )
            return response['choices'][0]['text']
            
        elif hasattr(self.client, 'generate'):
            # CTransformers or similar interface
            response = self.client.generate(
                prompt,
                temperature=temperature,
                max_new_tokens=max_tokens or 512
            )
            return response
            
        else:
            raise ValueError("Unsupported local model interface")
    
    def streaming_response_generator(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response from the model.
        
        Args:
            query: The query or prompt to send to the model
            context: Optional context to include with the query
            system_prompt: Optional system prompt to guide the model's behavior
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Yields:
            Chunks of the response as they become available
        """
        if not self.client:
            yield "Error: LLM not initialized"
            return
        
        try:
            # Prepare the prompt with context if provided
            prompt = self._prepare_prompt(query, context, system_prompt)
            
            # Stream response based on provider
            if self.provider == 'openai':
                yield from self._stream_openai_response(prompt, temperature, max_tokens)
            elif self.provider == 'anthropic':
                yield from self._stream_anthropic_response(prompt, temperature, max_tokens)
            elif self.provider == 'huggingface':
                yield from self._stream_huggingface_response(prompt, temperature, max_tokens)
            elif self.provider == 'local':
                yield from self._stream_local_response(prompt, temperature, max_tokens)
            else:
                yield f"Error: Unsupported provider: {self.provider}"
                
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error: {str(e)}"
    
    def _stream_openai_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Stream a response using OpenAI API.
        
        Args:
            messages: List of message dictionaries (role and content)
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Yields:
            Chunks of the response as they become available
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def _stream_anthropic_response(
        self,
        prompt: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Stream a response using Anthropic API.
        
        Args:
            prompt: Formatted prompt with system and messages
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Yields:
            Chunks of the response as they become available
        """
        with self.client.messages.stream(
            model=self.model,
            system=prompt["system"],
            messages=prompt["messages"],
            temperature=temperature,
            max_tokens=max_tokens or 1024
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    def _stream_huggingface_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Stream a response using Hugging Face Inference API.
        
        Args:
            prompt: Formatted text prompt
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Yields:
            Chunks of the response as they become available
        """
        # Some Hugging Face endpoints support streaming
        try:
            for response in self.client.generate(
                prompt=prompt,
                model=self.model,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                stream=True
            ):
                yield response
        except Exception:
            # Fall back to non-streaming if streaming not supported
            response = self._generate_huggingface_response(prompt, temperature, max_tokens)
            yield response
    
    def _stream_local_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Stream a response using a local model.
        
        Args:
            prompt: Formatted text prompt
            temperature: Controls randomness of output (0-1)
            max_tokens: Maximum number of tokens in the response
            
        Yields:
            Chunks of the response as they become available
        """
        # This implementation depends on the model type
        if isinstance(self.client, object) and hasattr(self.client, 'create_completion'):
            # llama-cpp-python streaming interface
            stream = self.client.create_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens or 512,
                stream=True
            )
            
            for chunk in stream:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    text = chunk['choices'][0].get('text', '')
                    if text:
                        yield text
                        
        elif hasattr(self.client, 'generate_stream'):
            # CTransformers or similar streaming interface
            for chunk in self.client.generate_stream(
                prompt,
                temperature=temperature,
                max_new_tokens=max_tokens or 512
            ):
                yield chunk
                
        else:
            # Fall back to non-streaming if streaming not supported
            response = self._generate_local_response(prompt, temperature, max_tokens)
            yield response
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the connected model.
        
        Returns:
            Dictionary with model information
        """
        if not self.client:
            return {"error": "LLM not initialized"}
        
        return {
            "provider": self.provider,
            "model": self.model,
            "initialized": True,
            "supports_streaming": True,
            "max_tokens": self._get_model_context_size(self.model)
        }
    
    def disconnect(self) -> bool:
        """
        Disconnect from the LLM provider.
        
        Returns:
            Boolean indicating if disconnection was successful
        """
        try:
            # Clean up resources based on provider
            if self.provider == 'local':
                # Local models might need explicit cleanup
                if hasattr(self.client, 'unload') and callable(self.client.unload):
                    self.client.unload()
            
            # Reset connection state
            self.client = None
            
            logger.info(f"Disconnected from {self.provider} model: {self.model}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.provider}: {str(e)}")
            return False

    def _get_model_context_size(self, model_name: str) -> int:
        """
        Get the context size for the connected model.
        
        Args:
            model_name: The model name
            
        Returns:
            Maximum context size in tokens
        """
        # Default context sizes for common models
        context_sizes = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000
        }
        
        # Return default if model not in known list
        return context_sizes.get(model_name, 4096)

    def analyze_medical_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze medical text for key information.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        analysis_prompt = f"""Analyze the following medical text and extract key information:

{text}

Please provide:
1. Key medical terms and their definitions
2. Main medical concepts discussed
3. Any mentioned conditions or symptoms
4. Treatment options if discussed
5. Risk factors if mentioned
6. Recommendations if any

Format the response as a JSON object."""

        response = self.generate_text(analysis_prompt)
        
        try:
            # Extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        except:
            logger.warning("Failed to parse JSON response")
            return {"error": "Failed to parse response", "raw_text": response} 