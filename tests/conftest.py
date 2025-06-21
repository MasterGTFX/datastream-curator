"""Pytest configuration and fixtures for DataStream Curator tests."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from datastream_curator.config import CurationConfig, LLMConfig, ProcessingConfig, OutputConfig
from datastream_curator.llm import LLMResponse


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return {
        "project": "datastream-curator",
        "version": "0.1.0",
        "features": [
            {
                "name": "LLM Integration",
                "description": "Connect to various LLM providers",
                "status": "implemented"
            },
            {
                "name": "Diff Engine",
                "description": "Apply structured changes to documents",
                "status": "implemented"
            }
        ],
        "dependencies": {
            "pydantic": ">=2.5.0",
            "aiohttp": ">=3.9.0",
            "pyyaml": ">=6.0.1"
        }
    }


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing."""
    return """# DataStream Curator

## Overview
AI-powered incremental data curation tool.

## Features

### LLM Integration
Connect to various LLM providers for intelligent data processing.

### Configuration Management
Flexible configuration system with YAML and environment variable support.

## Installation

```bash
pip install datastream-curator
```

## Usage

```python
from datastream_curator import DataStreamCurator

curator = DataStreamCurator()
result = await curator.process(input_data, output_path)
```
"""


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "additions": [
            {
                "section": "Features",
                "content": "### New Feature\nDescription of the new feature that was added.",
                "reasoning": "Added based on input data analysis"
            }
        ],
        "modifications": [
            {
                "section": "Installation",
                "old_content": "pip install datastream-curator",
                "new_content": "pip install datastream-curator[dev]",
                "reasoning": "Updated installation command to include dev dependencies"
            }
        ],
        "deletions": [],
        "reasoning": "Added new feature and updated installation instructions based on input data"
    }


@pytest.fixture
def mock_llm_response_object(mock_llm_response):
    """Mock LLMResponse object."""
    return LLMResponse(
        content=json.dumps(mock_llm_response),
        model="anthropic/claude-3-sonnet",
        finish_reason="stop",
        usage={"input_tokens": 100, "output_tokens": 200}
    )


@pytest.fixture
def test_config():
    """Test configuration for DataStream Curator."""
    return CurationConfig(
        llm=LLMConfig(
            provider="openrouter",
            api_key="test-api-key",
            model="anthropic/claude-3-sonnet",
            temperature=0.1,
            max_tokens=4096,
            timeout=30
        ),
        processing=ProcessingConfig(
            batch_size=10,
            max_concurrent_requests=2,
            retry_attempts=2,
            retry_delay=0.5
        ),
        output=OutputConfig(
            format="markdown",
            include_metadata=True,
            preserve_structure=True
        )
    )


@pytest.fixture
def mock_llm_client(mock_llm_response):
    """Mock LLM client for testing."""
    mock_client = AsyncMock()
    mock_client.generate_diff.return_value = mock_llm_response
    
    # Mock context manager
    async def mock_aenter():
        return mock_client
    
    async def mock_aexit(exc_type, exc_val, exc_tb):
        pass
    
    mock_client.__aenter__ = mock_aenter
    mock_client.__aexit__ = mock_aexit
    
    return mock_client


@pytest.fixture
def sample_input_files(temp_dir, sample_json_data):
    """Create sample input files for testing."""
    # JSON file
    json_file = temp_dir / "input.json"
    json_file.write_text(json.dumps(sample_json_data, indent=2))
    
    # Text file
    text_file = temp_dir / "input.txt"
    text_file.write_text("This is sample text data for testing.")
    
    # CSV file
    csv_file = temp_dir / "input.csv"
    csv_file.write_text("name,version,status\nDataStream Curator,0.1.0,alpha\n")
    
    # YAML file
    yaml_file = temp_dir / "input.yaml"
    yaml_file.write_text("""
project: datastream-curator
version: 0.1.0
features:
  - name: YAML Support
    status: implemented
""")
    
    return {
        "json": str(json_file),
        "text": str(text_file),
        "csv": str(csv_file),
        "yaml": str(yaml_file)
    }


@pytest.fixture
def existing_kb_file(temp_dir, sample_markdown_content):
    """Create an existing knowledge base file for testing."""
    kb_file = temp_dir / "existing_kb.md"
    kb_file.write_text(sample_markdown_content)
    return str(kb_file)


@pytest.fixture
def output_file(temp_dir):
    """Output file path for testing."""
    return str(temp_dir / "output.md")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-3-sonnet")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def config_file(temp_dir):
    """Create a test configuration file."""
    config_content = """
llm:
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "anthropic/claude-3-sonnet"
  temperature: 0.1
  max_tokens: 4096
  timeout: 30

processing:
  batch_size: 10
  max_concurrent_requests: 2
  retry_attempts: 2
  retry_delay: 0.5

output:
  format: "markdown"
  include_metadata: true
  preserve_structure: true
"""
    
    config_file = temp_dir / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def invalid_json_response():
    """Invalid JSON response for error testing."""
    return "This is not valid JSON content"


@pytest.fixture
def partial_json_response():
    """Partial JSON response for error testing."""
    return """Some text before JSON
{
  "additions": [],
  "modifications": [],
  "deletions": [],
  "reasoning": "Test response"
}
Some text after JSON"""


@pytest.fixture
def malformed_diff_response():
    """Malformed diff response missing required fields."""
    return {
        "additions": [{"content": "test"}],  # Missing section
        "modifications": [{"old_content": "test"}],  # Missing new_content
        "deletions": [{}],  # Missing content
        "reasoning": "Test"
    }