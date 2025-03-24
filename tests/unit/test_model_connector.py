"""
Unit tests for the MedicalLLMConnector module.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from ai.llm.model_connector import (
    MedicalLLMConnector, 
    APIKeyError, 
    ModelConnectionError,
    UnsupportedProviderError
)


class TestMedicalLLMConnector:
    """Test cases for the MedicalLLMConnector class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Default test configuration
        self.config = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.1,
            "api_key": "test-key"
        }

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"})
    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment variable."""
        # Remove API key from config to force using env var
        config_without_key = self.config.copy()
        del config_without_key["api_key"]
        
        connector = MedicalLLMConnector(config_without_key)
        
        assert connector.provider == "openai"
        assert connector.model == "gpt-4"
        assert connector.temperature == 0.1
        assert connector.api_key == "env-test-key"

    def test_init_with_config_api_key(self):
        """Test initialization with API key from config."""
        connector = MedicalLLMConnector(self.config)
        
        assert connector.provider == "openai"
        assert connector.model == "gpt-4"
        assert connector.temperature == 0.1
        assert connector.api_key == "test-key"

    def test_init_missing_api_key(self):
        """Test initialization with missing API key raises an error."""
        # Remove API key from config
        config_without_key = self.config.copy()
        del config_without_key["api_key"]
        
        with pytest.raises(APIKeyError):
            MedicalLLMConnector(config_without_key)

    def test_init_invalid_provider(self):
        """Test initialization with invalid provider raises an error."""
        invalid_config = self.config.copy()
        invalid_config["provider"] = "invalid_provider"
        
        with pytest.raises(UnsupportedProviderError):
            MedicalLLMConnector(invalid_config)

    @patch("ai.llm.model_connector.OpenAI")
    def test_connect_openai(self, mock_openai):
        """Test connecting to OpenAI provider."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create connector and connect
        connector = MedicalLLMConnector(self.config)
        connector.connect()
        
        # Verify OpenAI client was created with correct params
        mock_openai.assert_called_once_with(api_key="test-key")
        
        # Verify client was stored
        assert connector.client == mock_client

    @patch("ai.llm.model_connector.AnthropicClient")
    def test_connect_anthropic(self, mock_anthropic):
        """Test connecting to Anthropic provider."""
        # Setup config and mock
        anthropic_config = self.config.copy()
        anthropic_config["provider"] = "anthropic"
        anthropic_config["model"] = "claude-v2"
        
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Create connector and connect
        connector = MedicalLLMConnector(anthropic_config)
        connector.connect()
        
        # Verify Anthropic client was created with correct params
        mock_anthropic.assert_called_once_with(api_key="test-key")
        
        # Verify client was stored
        assert connector.client == mock_client

    @patch("ai.llm.model_connector.load_huggingface_model")
    def test_connect_huggingface(self, mock_load_hf):
        """Test connecting to HuggingFace provider."""
        # Setup config and mock
        hf_config = self.config.copy()
        hf_config["provider"] = "huggingface"
        hf_config["model"] = "google/flan-t5-xl"
        
        mock_model = MagicMock()
        mock_load_hf.return_value = mock_model
        
        # Create connector and connect
        connector = MedicalLLMConnector(hf_config)
        connector.connect()
        
        # Verify HuggingFace model was loaded with correct params
        mock_load_hf.assert_called_once_with("google/flan-t5-xl")
        
        # Verify model was stored
        assert connector.client == mock_model

    @patch("ai.llm.model_connector.load_local_model")
    def test_connect_local(self, mock_load_local):
        """Test connecting to local provider."""
        # Setup config and mock
        local_config = self.config.copy()
        local_config["provider"] = "local"
        local_config["model"] = "/path/to/local/model"
        
        mock_model = MagicMock()
        mock_load_local.return_value = mock_model
        
        # Create connector and connect
        connector = MedicalLLMConnector(local_config)
        connector.connect()
        
        # Verify local model was loaded with correct params
        mock_load_local.assert_called_once_with("/path/to/local/model")
        
        # Verify model was stored
        assert connector.client == mock_model

    @patch("ai.llm.model_connector.OpenAI")
    def test_connection_error(self, mock_openai):
        """Test handling of connection errors."""
        # Setup mock to raise an exception
        mock_openai.side_effect = Exception("Connection failed")
        
        # Create connector
        connector = MedicalLLMConnector(self.config)
        
        # Verify connection error is raised
        with pytest.raises(ModelConnectionError):
            connector.connect()

    @patch("ai.llm.model_connector.OpenAI")
    def test_generate_text_openai(self, mock_openai):
        """Test generating text with OpenAI provider."""
        # Setup mocks
        mock_client = MagicMock()
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices[0].message.content = "Generated medical response"
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mock_openai.return_value = mock_client
        
        # Create connector and connect
        connector = MedicalLLMConnector(self.config)
        connector.connect()
        
        # Generate text
        response = connector.generate_text("What is hypertension?")
        
        # Verify response
        assert response == "Generated medical response"
        
        # Verify OpenAI client was called correctly
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is hypertension?"}],
            temperature=0.1
        )

    @patch("ai.llm.model_connector.AnthropicClient")
    def test_generate_text_anthropic(self, mock_anthropic):
        """Test generating text with Anthropic provider."""
        # Setup config and mocks
        anthropic_config = self.config.copy()
        anthropic_config["provider"] = "anthropic"
        anthropic_config["model"] = "claude-v2"
        
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.content = "Generated medical response from Claude"
        mock_client.messages.create.return_value = mock_completion
        mock_anthropic.return_value = mock_client
        
        # Create connector and connect
        connector = MedicalLLMConnector(anthropic_config)
        connector.connect()
        
        # Generate text
        response = connector.generate_text("What is diabetes?")
        
        # Verify response
        assert response == "Generated medical response from Claude"
        
        # Verify Anthropic client was called correctly
        mock_client.messages.create.assert_called_once_with(
            model="claude-v2",
            messages=[{"role": "user", "content": "What is diabetes?"}],
            temperature=0.1
        )

    @patch("ai.llm.model_connector.load_huggingface_model")
    def test_generate_text_huggingface(self, mock_load_hf):
        """Test generating text with HuggingFace provider."""
        # Setup config and mocks
        hf_config = self.config.copy()
        hf_config["provider"] = "huggingface"
        hf_config["model"] = "google/flan-t5-xl"
        
        mock_model = MagicMock()
        mock_model.generate.return_value = "Generated response from Flan-T5"
        mock_load_hf.return_value = mock_model
        
        # Create connector and connect
        connector = MedicalLLMConnector(hf_config)
        connector.connect()
        
        # Generate text
        response = connector.generate_text("What is a stroke?")
        
        # Verify response
        assert response == "Generated response from Flan-T5"
        
        # Verify HuggingFace model was called correctly
        mock_model.generate.assert_called_once_with(
            "What is a stroke?", temperature=0.1
        )

    @patch("ai.llm.model_connector.load_local_model")
    def test_generate_text_local(self, mock_load_local):
        """Test generating text with local provider."""
        # Setup config and mocks
        local_config = self.config.copy()
        local_config["provider"] = "local"
        local_config["model"] = "/path/to/local/model"
        
        mock_model = MagicMock()
        mock_model.generate.return_value = "Generated response from local model"
        mock_load_local.return_value = mock_model
        
        # Create connector and connect
        connector = MedicalLLMConnector(local_config)
        connector.connect()
        
        # Generate text
        response = connector.generate_text("What are the symptoms of COVID-19?")
        
        # Verify response
        assert response == "Generated response from local model"
        
        # Verify local model was called correctly
        mock_model.generate.assert_called_once_with(
            "What are the symptoms of COVID-19?", temperature=0.1
        )

    @patch("ai.llm.model_connector.OpenAI")
    def test_generate_with_context(self, mock_openai):
        """Test generating text with context."""
        # Setup mocks
        mock_client = MagicMock()
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices[0].message.content = "Response with context"
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mock_openai.return_value = mock_client
        
        # Create connector and connect
        connector = MedicalLLMConnector(self.config)
        connector.connect()
        
        # Context to provide
        context = "Hypertension, also known as high blood pressure, is a condition where the force of blood against artery walls is too high."
        
        # Generate text with context
        response = connector.generate_with_context(
            "What is hypertension?",
            context
        )
        
        # Verify response
        assert response == "Response with context"
        
        # Verify OpenAI client was called correctly with context
        expected_prompt = (
            "Context information is below.\n"
            "---------------------\n"
            "Hypertension, also known as high blood pressure, is a condition where the force of blood against artery walls is too high.\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the question: What is hypertension?"
        )
        
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": expected_prompt}],
            temperature=0.1
        )

    @patch("ai.llm.model_connector.OpenAI")
    def test_generate_embeddings(self, mock_openai):
        """Test generating embeddings."""
        # Setup mocks
        mock_client = MagicMock()
        mock_embedding_response = MagicMock()
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.embeddings.create.return_value = mock_embedding_response
        mock_openai.return_value = mock_client
        
        # Create connector and connect
        connector = MedicalLLMConnector(self.config)
        connector.connect()
        
        # Generate embeddings
        embeddings = connector.generate_embeddings("Hypertension is a medical condition.")
        
        # Verify embeddings
        assert embeddings == [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Verify OpenAI client was called correctly
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",  # Default embedding model
            input="Hypertension is a medical condition."
        )

    @patch("ai.llm.model_connector.OpenAI")
    def test_not_connected_error(self, mock_openai):
        """Test error when trying to generate text without connecting first."""
        connector = MedicalLLMConnector(self.config)
        
        # Verify error is raised when not connected
        with pytest.raises(ModelConnectionError):
            connector.generate_text("What is hypertension?")
        
        # Verify no client methods were called
        mock_openai.assert_not_called() 