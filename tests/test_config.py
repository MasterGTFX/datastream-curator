"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from datastream_curator.config import (
    CurationConfig,
    LLMConfig,
    LoggingConfig,
    OutputConfig,
    ProcessingConfig,
)


class TestLLMConfig:
    """Test LLMConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig(api_key="test-key")
        
        assert config.provider == "openrouter"
        assert config.api_key == "test-key"
        assert config.model == "anthropic/claude-3-sonnet"
        assert config.temperature == 0.1
        assert config.max_tokens == 4096
        assert config.timeout == 30
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            provider="openai",
            api_key="custom-key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=8192,
            timeout=60
        )
        
        assert config.provider == "openai"
        assert config.api_key == "custom-key"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192
        assert config.timeout == 60
    
    def test_api_key_required(self):
        """Test that API key is required."""
        with pytest.raises(ValueError):
            LLMConfig()


class TestProcessingConfig:
    """Test ProcessingConfig class."""
    
    def test_default_values(self):
        """Test default processing configuration."""
        config = ProcessingConfig()
        
        assert config.batch_size == 100
        assert config.max_concurrent_requests == 5
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
    
    def test_custom_values(self):
        """Test custom processing configuration."""
        config = ProcessingConfig(
            batch_size=50,
            max_concurrent_requests=10,
            retry_attempts=5,
            retry_delay=2.0
        )
        
        assert config.batch_size == 50
        assert config.max_concurrent_requests == 10
        assert config.retry_attempts == 5
        assert config.retry_delay == 2.0


class TestOutputConfig:
    """Test OutputConfig class."""
    
    def test_default_values(self):
        """Test default output configuration."""
        config = OutputConfig()
        
        assert config.format == "markdown"
        assert config.include_metadata is True
        assert config.preserve_structure is True
    
    def test_custom_values(self):
        """Test custom output configuration."""
        config = OutputConfig(
            format="json",
            include_metadata=False,
            preserve_structure=False
        )
        
        assert config.format == "json"
        assert config.include_metadata is False
        assert config.preserve_structure is False


class TestLoggingConfig:
    """Test LoggingConfig class."""
    
    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert "%(asctime)s" in config.format
        assert "%(name)s" in config.format
        assert "%(levelname)s" in config.format
        assert "%(message)s" in config.format


class TestCurationConfig:
    """Test CurationConfig class."""
    
    def test_from_env_with_openrouter(self, mock_env_vars):
        """Test creating configuration from environment variables."""
        config = CurationConfig.from_env()
        
        assert config.llm.provider == "openrouter"
        assert config.llm.api_key == "test-api-key"
        assert config.llm.model == "anthropic/claude-3-sonnet"
        assert config.processing.batch_size == 100
        assert config.output.format == "markdown"
    
    def test_from_env_with_openai(self, monkeypatch):
        """Test creating configuration with OpenAI API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        
        config = CurationConfig.from_env()
        
        assert config.llm.provider == "openai"
        assert config.llm.api_key == "openai-test-key"
    
    def test_from_env_no_api_key(self, monkeypatch):
        """Test error when no API key is provided."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="No API key found"):
            CurationConfig.from_env()
    
    def test_from_env_with_custom_values(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
        monkeypatch.setenv("PROCESSING_BATCH_SIZE", "50")
        monkeypatch.setenv("OUTPUT_FORMAT", "json")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        
        config = CurationConfig.from_env()
        
        assert config.llm.temperature == 0.5
        assert config.llm.max_tokens == 8192
        assert config.processing.batch_size == 50
        assert config.output.format == "json"
        assert config.logging.level == "DEBUG"
    
    def test_from_file_basic(self, config_file, mock_env_vars):
        """Test loading configuration from YAML file."""
        config = CurationConfig.from_file(config_file)
        
        assert config.llm.provider == "openrouter"
        assert config.llm.api_key == "test-api-key"
        assert config.llm.model == "anthropic/claude-3-sonnet"
        assert config.llm.temperature == 0.1
        assert config.processing.batch_size == 10
        assert config.output.format == "markdown"
    
    def test_from_file_env_var_expansion(self, temp_dir, monkeypatch):
        """Test environment variable expansion in YAML files."""
        monkeypatch.setenv("TEST_API_KEY", "expanded-api-key")
        monkeypatch.setenv("TEST_MODEL", "test-model")
        
        config_content = """
llm:
  provider: "openrouter"
  api_key: "${TEST_API_KEY}"
  model: "${TEST_MODEL}"
  temperature: 0.2
"""
        
        config_file = temp_dir / "test_config.yaml"
        config_file.write_text(config_content)
        
        config = CurationConfig.from_file(str(config_file))
        
        assert config.llm.api_key == "expanded-api-key"
        assert config.llm.model == "test-model"
        assert config.llm.temperature == 0.2
    
    def test_from_file_missing_env_var(self, temp_dir):
        """Test handling of missing environment variables."""
        config_content = """
llm:
  provider: "openrouter"
  api_key: "${MISSING_API_KEY}"
  model: "test-model"
"""
        
        config_file = temp_dir / "test_config.yaml"
        config_file.write_text(config_content)
        
        config = CurationConfig.from_file(str(config_file))
        
        # Missing env var should result in empty string
        assert config.llm.api_key == ""
    
    def test_from_file_invalid_yaml(self, temp_dir):
        """Test error handling for invalid YAML."""
        config_file = temp_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            CurationConfig.from_file(str(config_file))
    
    def test_from_file_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            CurationConfig.from_file("nonexistent.yaml")
    
    def test_expand_env_vars_nested(self):
        """Test environment variable expansion in nested structures."""
        os.environ["TEST_VALUE"] = "test-value"
        
        data = {
            "level1": {
                "level2": {
                    "value": "${TEST_VALUE}"
                }
            },
            "list": ["${TEST_VALUE}", "normal-value"],
            "normal": "normal-value"
        }
        
        result = CurationConfig._expand_env_vars(data)
        
        assert result["level1"]["level2"]["value"] == "test-value"
        assert result["list"][0] == "test-value"
        assert result["list"][1] == "normal-value"
        assert result["normal"] == "normal-value"
        
        # Clean up
        del os.environ["TEST_VALUE"]
    
    def test_configuration_validation(self, test_config):
        """Test that configuration validates correctly."""
        # Should not raise any validation errors
        assert test_config.llm.api_key == "test-api-key"
        assert test_config.llm.temperature == 0.1
        assert isinstance(test_config.processing.batch_size, int)
        assert isinstance(test_config.output.include_metadata, bool)