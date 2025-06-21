"""Tests for LLM integration."""

import json
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from datastream_curator.config import LLMConfig
from datastream_curator.llm import DiffRequest, LLMClient, LLMResponse


class TestDiffRequest:
    """Test DiffRequest model."""
    
    def test_basic_request(self):
        """Test creating a basic diff request."""
        request = DiffRequest(
            existing_content="# Test Document",
            new_data="New information",
            instruction="Update the document"
        )
        
        assert request.existing_content == "# Test Document"
        assert request.new_data == "New information"
        assert request.instruction == "Update the document"
        assert request.context is None
    
    def test_request_with_context(self):
        """Test creating a request with context."""
        request = DiffRequest(
            existing_content="# Test",
            new_data="Data",
            instruction="Update",
            context="Additional context"
        )
        
        assert request.context == "Additional context"


class TestLLMResponse:
    """Test LLMResponse model."""
    
    def test_basic_response(self):
        """Test creating a basic LLM response."""
        response = LLMResponse(
            content="Test response",
            model="test-model",
            finish_reason="stop"
        )
        
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.usage is None
    
    def test_response_with_usage(self):
        """Test creating a response with usage information."""
        response = LLMResponse(
            content="Test response",
            model="test-model",
            finish_reason="stop",
            usage={"input_tokens": 100, "output_tokens": 200}
        )
        
        assert response.usage["input_tokens"] == 100
        assert response.usage["output_tokens"] == 200


class TestLLMClient:
    """Test LLMClient class."""
    
    def test_client_initialization(self):
        """Test LLM client initialization."""
        config = LLMConfig(
            provider="openrouter",
            api_key="test-key",
            model="test-model"
        )
        
        client = LLMClient(config)
        
        assert client.config == config
        assert client.session is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test LLM client as async context manager."""
        config = LLMConfig(api_key="test-key")
        
        async with LLMClient(config) as client:
            assert client.session is not None
            assert isinstance(client.session, aiohttp.ClientSession)
        
        # Session should be closed after context exit
        assert client.session.closed
    
    def test_build_diff_prompt(self):
        """Test diff prompt generation."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        request = DiffRequest(
            existing_content="# Existing Content",
            new_data="New data to integrate",
            instruction="Add new information",
            context="Test context"
        )
        
        prompt = client._build_diff_prompt(request)
        
        assert "# Existing Content" in prompt
        assert "New data to integrate" in prompt
        assert "Add new information" in prompt
        assert "Test context" in prompt
        assert "JSON response" in prompt
        assert "additions" in prompt
        assert "modifications" in prompt
        assert "deletions" in prompt
    
    def test_build_diff_prompt_empty_content(self):
        """Test prompt generation with empty existing content."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        request = DiffRequest(
            existing_content="",
            new_data="New data",
            instruction=""
        )
        
        prompt = client._build_diff_prompt(request)
        
        assert "No existing content" in prompt
        assert "Intelligently integrate" in prompt
        assert "No additional context" in prompt
    
    @pytest.mark.asyncio
    async def test_make_request_openrouter(self):
        """Test making request to OpenRouter."""
        config = LLMConfig(
            provider="openrouter",
            api_key="test-key",
            model="test-model"
        )
        
        with patch.object(LLMClient, '_openrouter_request') as mock_request:
            mock_response = LLMResponse(
                content="Test response",
                model="test-model",
                finish_reason="stop"
            )
            mock_request.return_value = mock_response
            
            client = LLMClient(config)
            result = await client._make_request("test prompt")
            
            assert result == mock_response
            mock_request.assert_called_once_with("test prompt")
    
    @pytest.mark.asyncio
    async def test_make_request_openai(self):
        """Test making request to OpenAI."""
        config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-4"
        )
        
        with patch.object(LLMClient, '_openai_request') as mock_request:
            mock_response = LLMResponse(
                content="Test response",
                model="gpt-4",
                finish_reason="stop"
            )
            mock_request.return_value = mock_response
            
            client = LLMClient(config)
            result = await client._make_request("test prompt")
            
            assert result == mock_response
            mock_request.assert_called_once_with("test prompt")
    
    @pytest.mark.asyncio
    async def test_make_request_unsupported_provider(self):
        """Test error handling for unsupported provider."""
        config = LLMConfig(
            provider="unsupported",
            api_key="test-key"
        )
        
        client = LLMClient(config)
        
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            await client._make_request("test prompt")
    
    def test_parse_diff_response_valid_json(self, mock_llm_response):
        """Test parsing valid diff response."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        content = json.dumps(mock_llm_response)
        result = client._parse_diff_response(content)
        
        assert result == mock_llm_response
        assert "additions" in result
        assert "modifications" in result
        assert "deletions" in result
        assert "reasoning" in result
    
    def test_parse_diff_response_with_extra_text(self, mock_llm_response):
        """Test parsing response with extra text around JSON."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        content = f"Here is the response:\n\n{json.dumps(mock_llm_response)}\n\nThat's the end."
        result = client._parse_diff_response(content)
        
        assert result == mock_llm_response
    
    def test_parse_diff_response_missing_fields(self):
        """Test parsing response with missing required fields."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        incomplete_response = {
            "additions": [{"section": "test", "content": "test"}]
            # Missing modifications, deletions, reasoning
        }
        
        content = json.dumps(incomplete_response)
        result = client._parse_diff_response(content)
        
        # Should fill in missing fields
        assert "modifications" in result
        assert "deletions" in result
        assert "reasoning" in result
        assert result["modifications"] == []
        assert result["deletions"] == []
        assert result["reasoning"] == "No specific reasoning provided"
    
    def test_parse_diff_response_invalid_json(self, invalid_json_response):
        """Test error handling for invalid JSON."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            client._parse_diff_response(invalid_json_response)
    
    def test_parse_diff_response_no_json(self):
        """Test error handling when no JSON is found."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        content = "This is just plain text with no JSON"
        
        with pytest.raises(ValueError, match="No JSON found"):
            client._parse_diff_response(content)
    
    def test_validate_operations_valid(self):
        """Test validation of valid diff operations."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        diff_data = {
            "additions": [{"content": "test", "section": "test"}],
            "modifications": [{"old_content": "old", "new_content": "new"}],
            "deletions": [{"content": "delete"}],
            "reasoning": "test"
        }
        
        # Should not raise any exceptions
        client._validate_operations(diff_data)
    
    def test_validate_operations_invalid_addition(self):
        """Test validation error for invalid addition."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        diff_data = {
            "additions": [{"section": "test"}],  # Missing content
            "modifications": [],
            "deletions": [],
            "reasoning": "test"
        }
        
        with pytest.raises(ValueError, match="Addition must have 'content' field"):
            client._validate_operations(diff_data)
    
    def test_validate_operations_invalid_modification(self):
        """Test validation error for invalid modification."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        diff_data = {
            "additions": [],
            "modifications": [{"old_content": "old"}],  # Missing new_content
            "deletions": [],
            "reasoning": "test"
        }
        
        with pytest.raises(ValueError, match="Modification must have 'old_content' and 'new_content' fields"):
            client._validate_operations(diff_data)
    
    def test_validate_operations_invalid_deletion(self):
        """Test validation error for invalid deletion."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        diff_data = {
            "additions": [],
            "modifications": [],
            "deletions": [{"section": "test"}],  # Missing content
            "reasoning": "test"
        }
        
        with pytest.raises(ValueError, match="Deletion must have 'content' field"):
            client._validate_operations(diff_data)
    
    @pytest.mark.asyncio
    async def test_generate_diff_success(self, mock_llm_response):
        """Test successful diff generation."""
        config = LLMConfig(api_key="test-key", retry_attempts=1)
        
        with patch.object(LLMClient, '_make_request') as mock_request:
            mock_response = LLMResponse(
                content=json.dumps(mock_llm_response),
                model="test-model",
                finish_reason="stop"
            )
            mock_request.return_value = mock_response
            
            client = LLMClient(config)
            request = DiffRequest(
                existing_content="test",
                new_data="test",
                instruction="test"
            )
            
            result = await client.generate_diff(request)
            
            assert result == mock_llm_response
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_diff_retry_success(self, mock_llm_response):
        """Test diff generation with retry on failure."""
        config = LLMConfig(api_key="test-key", retry_attempts=2, retry_delay=0.1)
        
        with patch.object(LLMClient, '_make_request') as mock_request:
            # First call fails, second succeeds
            mock_request.side_effect = [
                Exception("Network error"),
                LLMResponse(
                    content=json.dumps(mock_llm_response),
                    model="test-model",
                    finish_reason="stop"
                )
            ]
            
            client = LLMClient(config)
            request = DiffRequest(
                existing_content="test",
                new_data="test",
                instruction="test"
            )
            
            result = await client.generate_diff(request)
            
            assert result == mock_llm_response
            assert mock_request.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generate_diff_all_retries_fail(self):
        """Test diff generation when all retries fail."""
        config = LLMConfig(api_key="test-key", retry_attempts=2, retry_delay=0.1)
        
        with patch.object(LLMClient, '_make_request') as mock_request:
            mock_request.side_effect = Exception("Persistent error")
            
            client = LLMClient(config)
            request = DiffRequest(
                existing_content="test",
                new_data="test",
                instruction="test"
            )
            
            with pytest.raises(Exception, match="Persistent error"):
                await client.generate_diff(request)
            
            assert mock_request.call_count == 2