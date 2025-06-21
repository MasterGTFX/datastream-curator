# DataStream Curator - Development Guide

**AI-powered incremental data curation and knowledge base management tool**

This document provides comprehensive technical details, implementation guidance, and testing strategies for developing the DataStream Curator.

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Technical Implementation](#technical-implementation)
3. [Development Setup](#development-setup)
4. [Core Components](#core-components)
5. [Testing Strategy](#testing-strategy)
6. [Usage Examples](#usage-examples)
7. [Configuration](#configuration)
8. [Development Workflow](#development-workflow)

---

## Project Architecture

### Core Design Principles

- **Simplicity First**: Start with core functionality, extend later
- **Module-First**: Importable Python API before CLI interface
- **Async-Native**: Built for high-performance async operations
- **Type-Safe**: Full type hints with Pydantic validation
- **Test-Driven**: Comprehensive test coverage for all features

### Project Structure

```
datastream-curator/
├── src/
│   └── datastream_curator/
│       ├── __init__.py          # Main API exports
│       ├── core.py              # DataStreamCurator main class
│       ├── llm.py               # LLM integration layer
│       ├── diff.py              # Diff generation and application
│       └── config.py            # Configuration management
├── tests/
│   ├── __init__.py
│   ├── test_core.py             # Core functionality tests
│   ├── test_llm.py              # LLM integration tests
│   ├── test_diff.py             # Diff engine tests
│   ├── test_config.py           # Configuration tests
│   └── fixtures/                # Test data and fixtures
├── pyproject.toml               # Python project configuration
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── config.yaml                  # Sample configuration file
├── CLAUDE.md                    # This file
├── README.md                    # Project overview
└── LICENSE                      # MIT License
```

### Data Flow Architecture

```
Input Data → Normalization → LLM Analysis → Diff Generation → Application → Output
     ↓            ↓              ↓              ↓              ↓           ↓
  JSON/CSV    Validation    Context Prompt   JSON Diff   Smart Merge   Markdown
  XML/Text    Type Check    + Instructions   Structure   Validation    Knowledge Base
```

---

## Technical Implementation

### Dependencies

#### Core Dependencies
```txt
# Data validation and settings
pydantic>=2.5.0

# Async HTTP client for LLM APIs  
aiohttp>=3.9.0

# Configuration file support
pyyaml>=6.0.1

# CLI framework (for future CLI implementation)
click>=8.1.0

# Enhanced terminal output
rich>=13.7.0

# Fast JSON processing
orjson>=3.9.0
```

#### Development Dependencies
```txt
# Testing framework
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-mock>=3.12.0

# Test coverage
pytest-cov>=4.1.0

# Code formatting and linting
black>=23.12.0
ruff>=0.1.0

# Type checking
mypy>=1.8.0

# Development utilities
pre-commit>=3.6.0
```

### Core Components Detail

#### 1. Configuration Management (`config.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import yaml
import os

class LLMConfig(BaseModel):
    provider: str = "openrouter"
    api_key: str = Field(..., description="LLM API key")
    model: str = "anthropic/claude-3-sonnet"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 30

class ProcessingConfig(BaseModel):
    batch_size: int = 100
    max_concurrent_requests: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0

class OutputConfig(BaseModel):
    format: str = "markdown"
    include_metadata: bool = True
    preserve_structure: bool = True

class CurationConfig(BaseModel):
    llm: LLMConfig
    processing: ProcessingConfig = ProcessingConfig()
    output: OutputConfig = OutputConfig()
    
    @classmethod
    def from_file(cls, config_path: str) -> "CurationConfig":
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "CurationConfig":
        return cls(
            llm=LLMConfig(
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                model=os.getenv("LLM_MODEL", "anthropic/claude-3-sonnet"),
                provider=os.getenv("LLM_PROVIDER", "openrouter")
            )
        )
```

#### 2. LLM Integration (`llm.py`)

```python
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel
import json
import logging

logger = logging.getLogger(__name__)

class LLMResponse(BaseModel):
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: str
    finish_reason: str

class DiffRequest(BaseModel):
    existing_content: str
    new_data: str
    instruction: str
    context: Optional[str] = None

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_diff(self, request: DiffRequest) -> Dict[str, Any]:
        """Generate structured diff using LLM"""
        prompt = self._build_diff_prompt(request)
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self._make_request(prompt)
                diff_data = self._parse_diff_response(response.content)
                return diff_data
            except Exception as e:
                logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    def _build_diff_prompt(self, request: DiffRequest) -> str:
        return f"""You are a knowledge base curator. Analyze the existing knowledge base and new input data to generate structured updates.

EXISTING KNOWLEDGE BASE:
{request.existing_content or "No existing content"}

NEW INPUT DATA:
{request.new_data}

USER INSTRUCTION:
{request.instruction or "Intelligently integrate new information into the knowledge base"}

CONTEXT:
{request.context or "No additional context"}

Generate a JSON response with this exact structure:
{{
  "additions": [
    {{
      "section": "section_name",
      "content": "new content to add",
      "reasoning": "why this should be added"
    }}
  ],
  "modifications": [
    {{
      "section": "section_name", 
      "old_content": "exact text to replace",
      "new_content": "replacement text",
      "reasoning": "why this change is needed"
    }}
  ],
  "deletions": [
    {{
      "section": "section_name",
      "content": "exact text to remove",
      "reasoning": "why this should be removed"
    }}
  ],
  "reasoning": "Overall reasoning for all changes"
}}

Focus on:
1. Accuracy and factual correctness
2. Maintaining consistency and coherence  
3. Preserving important historical context
4. Following the user's specific curation goals
5. Only make necessary changes - preserve existing valuable content

Respond with ONLY the JSON structure, no additional text."""

    async def _make_request(self, prompt: str) -> LLMResponse:
        if self.config.provider == "openrouter":
            return await self._openrouter_request(prompt)
        elif self.config.provider == "openai":
            return await self._openai_request(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    async def _openrouter_request(self, prompt: str) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/datastream-curator",
            "X-Title": "DataStream Curator"
        }
        
        data = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        async with self.session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                usage=result.get("usage"),
                model=result["model"],
                finish_reason=result["choices"][0]["finish_reason"]
            )
    
    def _parse_diff_response(self, content: str) -> Dict[str, Any]:
        """Parse and validate LLM diff response"""
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = content[start:end]
            diff_data = json.loads(json_str)
            
            # Validate structure
            required_keys = ["additions", "modifications", "deletions", "reasoning"]
            for key in required_keys:
                if key not in diff_data:
                    diff_data[key] = [] if key != "reasoning" else ""
            
            return diff_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response content: {content}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
```

#### 3. Diff Engine (`diff.py`)

```python
import re
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class DiffOperation(BaseModel):
    operation: str  # "add", "modify", "delete"
    section: str
    content: str
    old_content: str = ""
    reasoning: str = ""

class DiffResult(BaseModel):
    operations: List[DiffOperation]
    reasoning: str
    stats: Dict[str, int]

class DiffEngine:
    """Handles application of structured diffs to markdown content"""
    
    def apply_diff(self, content: str, diff_data: Dict[str, Any]) -> str:
        """Apply structured diff to existing content"""
        result = content
        operations = []
        
        # Process deletions first
        for deletion in diff_data.get("deletions", []):
            old_result = result
            result = self._apply_deletion(result, deletion)
            if result != old_result:
                operations.append(DiffOperation(
                    operation="delete",
                    section=deletion.get("section", ""),
                    content=deletion.get("content", ""),
                    reasoning=deletion.get("reasoning", "")
                ))
        
        # Process modifications
        for modification in diff_data.get("modifications", []):
            old_result = result
            result = self._apply_modification(result, modification)
            if result != old_result:
                operations.append(DiffOperation(
                    operation="modify",
                    section=modification.get("section", ""),
                    content=modification.get("new_content", ""),
                    old_content=modification.get("old_content", ""),
                    reasoning=modification.get("reasoning", "")
                ))
        
        # Process additions
        for addition in diff_data.get("additions", []):
            old_result = result
            result = self._apply_addition(result, addition)
            if result != old_result:
                operations.append(DiffOperation(
                    operation="add",
                    section=addition.get("section", ""),
                    content=addition.get("content", ""),
                    reasoning=addition.get("reasoning", "")
                ))
        
        return result
    
    def _apply_deletion(self, content: str, deletion: Dict[str, Any]) -> str:
        """Remove specified content"""
        target = deletion.get("content", "").strip()
        if not target:
            return content
        
        # Try exact match first
        if target in content:
            return content.replace(target, "", 1)
        
        # Try fuzzy matching for partial content
        lines = target.split('\n')
        if len(lines) > 1:
            # Multi-line deletion - try to match key phrases
            for line in lines:
                line = line.strip()
                if line and line in content:
                    content = content.replace(line, "", 1)
        
        return content
    
    def _apply_modification(self, content: str, modification: Dict[str, Any]) -> str:
        """Replace old content with new content"""
        old_content = modification.get("old_content", "").strip()
        new_content = modification.get("new_content", "").strip()
        
        if not old_content or not new_content:
            return content
        
        # Try exact replacement
        if old_content in content:
            return content.replace(old_content, new_content, 1)
        
        # Try fuzzy matching - find similar content
        old_words = old_content.split()
        if len(old_words) > 2:
            # Try matching with first few words
            partial = " ".join(old_words[:3])
            if partial in content:
                # Find the section and replace more intelligently
                return self._smart_replace(content, partial, new_content)
        
        return content
    
    def _apply_addition(self, content: str, addition: Dict[str, Any]) -> str:
        """Add new content to appropriate section"""
        section = addition.get("section", "")
        new_content = addition.get("content", "").strip()
        
        if not new_content:
            return content
        
        # If section specified, try to add to that section
        if section:
            section_pattern = rf"^#{1,6}\s+{re.escape(section)}"
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if re.match(section_pattern, line, re.IGNORECASE):
                    # Found section - add content after it
                    lines.insert(i + 1, "")
                    lines.insert(i + 2, new_content)
                    return '\n'.join(lines)
        
        # If no specific section or section not found, append to end
        if content.strip():
            return content + "\n\n" + new_content
        else:
            return new_content
    
    def _smart_replace(self, content: str, search_phrase: str, replacement: str) -> str:
        """Intelligently replace content using contextual matching"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if search_phrase in line:
                # Replace the entire line or find the best insertion point
                lines[i] = replacement
                return '\n'.join(lines)
        
        return content
    
    def generate_stats(self, diff_data: Dict[str, Any]) -> Dict[str, int]:
        """Generate statistics about the diff"""
        return {
            "additions": len(diff_data.get("additions", [])),
            "modifications": len(diff_data.get("modifications", [])),
            "deletions": len(diff_data.get("deletions", [])),
            "total_operations": (
                len(diff_data.get("additions", [])) +
                len(diff_data.get("modifications", [])) +
                len(diff_data.get("deletions", []))
            )
        }
```

#### 4. Main Curator Class (`core.py`)

```python
import asyncio
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json

from .config import CurationConfig
from .llm import LLMClient, DiffRequest
from .diff import DiffEngine

logger = logging.getLogger(__name__)

class DataStreamCurator:
    """Main class for incremental data curation and knowledge base management"""
    
    def __init__(self, config: Optional[CurationConfig] = None):
        self.config = config or CurationConfig.from_env()
        self.diff_engine = DiffEngine()
    
    async def process(
        self,
        input_data: Union[str, Dict[str, Any]],
        output_path: Optional[str] = None,
        existing_kb_path: Optional[str] = None,
        instruction: Optional[str] = None
    ) -> str:
        """
        Process input data and update knowledge base
        
        Args:
            input_data: Input data (file path or data dict)
            output_path: Output file path for updated knowledge base
            existing_kb_path: Path to existing knowledge base file
            instruction: Custom curation instruction
            
        Returns:
            Updated knowledge base content
        """
        logger.info("Starting data curation process")
        
        # Load existing knowledge base
        existing_content = ""
        if existing_kb_path and Path(existing_kb_path).exists():
            existing_content = Path(existing_kb_path).read_text(encoding='utf-8')
            logger.info(f"Loaded existing knowledge base from {existing_kb_path}")
        
        # Process input data
        processed_data = await self._process_input_data(input_data)
        
        # Generate diff using LLM
        async with LLMClient(self.config.llm) as llm_client:
            diff_request = DiffRequest(
                existing_content=existing_content,
                new_data=processed_data,
                instruction=instruction or "Intelligently integrate new information into the knowledge base"
            )
            
            logger.info("Generating diff using LLM")
            diff_data = await llm_client.generate_diff(diff_request)
        
        # Apply diff to existing content
        logger.info("Applying diff to knowledge base")
        updated_content = self.diff_engine.apply_diff(existing_content, diff_data)
        
        # Save updated content if output path specified
        if output_path:
            Path(output_path).write_text(updated_content, encoding='utf-8')
            logger.info(f"Saved updated knowledge base to {output_path}")
        
        # Log statistics
        stats = self.diff_engine.generate_stats(diff_data)
        logger.info(f"Curation complete: {stats}")
        
        return updated_content
    
    async def _process_input_data(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Process and normalize input data"""
        if isinstance(input_data, dict):
            return json.dumps(input_data, indent=2)
        
        if isinstance(input_data, str):
            # Check if it's a file path
            input_path = Path(input_data)
            if input_path.exists():
                return self._read_file(input_path)
            else:
                # Treat as raw text data
                return input_data
        
        return str(input_data)
    
    def _read_file(self, file_path: Path) -> str:
        """Read and process different file formats"""
        suffix = file_path.suffix.lower()
        content = file_path.read_text(encoding='utf-8')
        
        if suffix == '.json':
            # Pretty print JSON for better LLM processing
            try:
                data = json.loads(content)
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                return content
        
        return content
    
    async def process_batch(
        self,
        input_files: List[str],
        output_path: str,
        instruction: Optional[str] = None
    ) -> str:
        """Process multiple input files in batch"""
        logger.info(f"Processing batch of {len(input_files)} files")
        
        existing_content = ""
        if Path(output_path).exists():
            existing_content = Path(output_path).read_text(encoding='utf-8')
        
        current_content = existing_content
        
        for input_file in input_files:
            logger.info(f"Processing file: {input_file}")
            current_content = await self.process(
                input_data=input_file,
                existing_kb_path=None,  # Use current_content instead
                instruction=instruction
            )
            
            # Update existing content for next iteration
            existing_content = current_content
        
        # Save final result
        Path(output_path).write_text(current_content, encoding='utf-8')
        logger.info(f"Batch processing complete, saved to {output_path}")
        
        return current_content
```

#### 5. Module Exports (`__init__.py`)

```python
"""
DataStream Curator - AI-powered incremental data curation and knowledge base management
"""

from .core import DataStreamCurator
from .config import CurationConfig, LLMConfig, ProcessingConfig, OutputConfig
from .llm import LLMClient, DiffRequest, LLMResponse
from .diff import DiffEngine, DiffOperation, DiffResult

__version__ = "0.1.0"
__author__ = "DataStream Curator Team"
__email__ = "contact@datastream-curator.dev"

__all__ = [
    "DataStreamCurator",
    "CurationConfig", 
    "LLMConfig",
    "ProcessingConfig",
    "OutputConfig",
    "LLMClient",
    "DiffRequest", 
    "LLMResponse",
    "DiffEngine",
    "DiffOperation",
    "DiffResult"
]
```

---

## Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/datastream-curator.git
cd datastream-curator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Set up pre-commit hooks
pre-commit install
```

### 2. Configuration

Create `.env` file:
```bash
OPENROUTER_API_KEY=your_api_key_here
LLM_MODEL=anthropic/claude-3-sonnet
LLM_PROVIDER=openrouter
LOG_LEVEL=INFO
```

Create `config.yaml`:
```yaml
llm:
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "anthropic/claude-3-sonnet"
  temperature: 0.1
  max_tokens: 4096
  timeout: 30

processing:
  batch_size: 100
  max_concurrent_requests: 5
  retry_attempts: 3
  retry_delay: 1.0

output:
  format: "markdown"
  include_metadata: true
  preserve_structure: true
```

### 3. Project Configuration Files

#### `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "datastream-curator"
version = "0.1.0"
description = "AI-powered incremental data curation and knowledge base management"
authors = [{name = "DataStream Curator Team", email = "contact@datastream-curator.dev"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "pydantic>=2.5.0",
    "aiohttp>=3.9.0",
    "pyyaml>=6.0.1",
    "click>=8.1.0",
    "rich>=13.7.0",
    "orjson>=3.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.12.0",
    "pytest-cov>=4.1.0",
    "black>=23.12.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]

[project.urls]
Homepage = "https://github.com/your-org/datastream-curator"
Repository = "https://github.com/your-org/datastream-curator"
Documentation = "https://datastream-curator.readthedocs.io/"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py39']

[tool.ruff]
target-version = "py39"
line-length = 88
select = ["E", "F", "W", "C90", "I", "N", "UP", "YTT", "S", "BLE", "FBT", "B", "A", "COM", "C4", "DTZ", "T10", "EM", "EXE", "FA", "ISC", "ICN", "G", "INP", "PIE", "T20", "PYI", "PT", "Q", "RSE", "RET", "SLF", "SLOT", "SIM", "TID", "TCH", "INT", "ARG", "PTH", "TD", "FIX", "ERA", "PD", "PGH", "PL", "TRY", "FLY", "NPY", "AIR", "PERF", "FURB", "RUF"]
ignore = ["S101", "T201", "T203"]

[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--strict-markers --strict-config --cov=src --cov-report=term-missing"
asyncio_mode = "auto"
```

---

## Core Components

### Configuration Management

The configuration system uses Pydantic models for type safety and validation:

- **LLMConfig**: API settings, model selection, parameters
- **ProcessingConfig**: Batch sizes, concurrency, retry logic  
- **OutputConfig**: Format options, metadata inclusion
- **CurationConfig**: Top-level configuration combining all settings

### LLM Integration

Async HTTP client supporting multiple providers:

- **OpenRouter Integration**: Primary provider with comprehensive model support
- **OpenAI Compatible**: Standard OpenAI API format
- **Structured Prompts**: Engineered for reliable diff generation
- **Error Handling**: Retry logic with exponential backoff
- **Response Parsing**: Robust JSON extraction and validation

### Diff Engine

Core logic for applying structured changes:

- **Three Operation Types**: Additions, modifications, deletions
- **Smart Matching**: Exact and fuzzy content matching
- **Section Awareness**: Markdown structure preservation
- **Safe Application**: Validation before applying changes
- **Statistics Tracking**: Detailed operation metrics

### Main Curator Class

High-level interface for all curation operations:

- **Simple API**: Single `process()` method for most use cases
- **Flexible Input**: File paths, raw data, or data structures
- **Batch Processing**: Handle multiple files efficiently
- **Error Recovery**: Graceful handling of failures
- **Logging Integration**: Comprehensive operation logging

---

## Testing Strategy

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                  # Pytest configuration and fixtures
├── fixtures/                    # Test data files
│   ├── sample_input.json
│   ├── existing_kb.md
│   └── expected_output.md
├── test_config.py               # Configuration tests
├── test_llm.py                  # LLM integration tests
├── test_diff.py                 # Diff engine tests
├── test_core.py                 # Core functionality tests
└── integration/                 # End-to-end tests
    ├── test_full_workflow.py
    └── test_batch_processing.py
```

### Test Categories

#### 1. Unit Tests

**Configuration Tests (`test_config.py`)**
```python
def test_config_from_env():
    """Test configuration loading from environment variables"""

def test_config_from_file():
    """Test configuration loading from YAML file"""

def test_config_validation():
    """Test Pydantic validation of configuration"""
```

**LLM Tests (`test_llm.py`)**
```python
@pytest.mark.asyncio
async def test_llm_client_initialization():
    """Test LLM client setup and configuration"""

@pytest.mark.asyncio  
async def test_diff_generation_success(mock_llm_response):
    """Test successful diff generation with mocked LLM"""

@pytest.mark.asyncio
async def test_llm_retry_logic(mock_failing_llm):
    """Test retry behavior on LLM failures"""
```

**Diff Engine Tests (`test_diff.py`)**
```python
def test_apply_addition():
    """Test adding new content to existing markdown"""

def test_apply_modification():
    """Test modifying existing content"""

def test_apply_deletion():
    """Test removing content from markdown"""

def test_diff_stats_generation():
    """Test statistics calculation for diffs"""
```

#### 2. Integration Tests

**Full Workflow (`test_full_workflow.py`)**
```python
@pytest.mark.asyncio
async def test_end_to_end_curation():
    """Test complete curation workflow from input to output"""

@pytest.mark.asyncio
async def test_incremental_updates():
    """Test multiple sequential updates to same knowledge base"""
```

#### 3. Mock Strategies

**LLM Response Mocking**
```python
@pytest.fixture
def mock_llm_response():
    return {
        "additions": [
            {
                "section": "Features",
                "content": "### New Feature\nDescription here",
                "reasoning": "Added based on input data"
            }
        ],
        "modifications": [],
        "deletions": [],
        "reasoning": "Added new feature based on input"
    }

@pytest.fixture
async def mock_llm_client(mock_llm_response):
    with patch('datastream_curator.llm.LLMClient') as mock:
        mock.return_value.__aenter__.return_value.generate_diff.return_value = mock_llm_response
        yield mock
```

#### 4. Test Fixtures

**Sample Data (`fixtures/sample_input.json`)**
```json
{
  "project": "datastream-curator",
  "features": [
    {
      "name": "LLM Integration",
      "description": "Connect to various LLM providers",
      "status": "implemented"
    }
  ],
  "version": "0.1.0"
}
```

**Expected Output (`fixtures/expected_output.md`)**
```markdown
# DataStream Curator

## Features

### LLM Integration
Connect to various LLM providers for intelligent data curation.

Status: Implemented
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run integration tests only
pytest tests/integration/

# Run with verbose output
pytest -v

# Run async tests
pytest -k "asyncio"
```

---

## Usage Examples

### 1. Basic Module Usage

```python
import asyncio
from datastream_curator import DataStreamCurator, CurationConfig

async def basic_example():
    # Initialize curator with default config
    curator = DataStreamCurator()
    
    # Process JSON data into markdown knowledge base
    result = await curator.process(
        input_data="./project_data.json",
        output_path="./knowledge_base.md",
        instruction="Create comprehensive project documentation"
    )
    
    print(f"Updated knowledge base with {len(result)} characters")

# Run the example
asyncio.run(basic_example())
```

### 2. Custom Configuration

```python
from datastream_curator import DataStreamCurator, CurationConfig, LLMConfig

async def custom_config_example():
    # Create custom configuration
    config = CurationConfig(
        llm=LLMConfig(
            provider="openrouter",
            api_key="your-api-key",
            model="anthropic/claude-3-sonnet",
            temperature=0.2,
            max_tokens=8192
        )
    )
    
    curator = DataStreamCurator(config)
    
    # Process with custom instruction
    result = await curator.process(
        input_data={
            "features": ["LLM Integration", "Diff Engine", "Async Processing"],
            "status": "In Development"
        },
        output_path="./features.md",
        instruction="Focus on technical implementation details and architecture"
    )

asyncio.run(custom_config_example())
```

### 3. Incremental Updates

```python
async def incremental_updates_example():
    curator = DataStreamCurator()
    kb_path = "./evolving_knowledge_base.md"
    
    # First update
    await curator.process(
        input_data="./initial_data.json",
        output_path=kb_path,
        instruction="Create initial project documentation"
    )
    
    # Second update - builds on existing content
    await curator.process(
        input_data="./new_features.json", 
        existing_kb_path=kb_path,
        output_path=kb_path,
        instruction="Update documentation with new features"
    )
    
    # Third update - more changes
    await curator.process(
        input_data="./bug_fixes.json",
        existing_kb_path=kb_path, 
        output_path=kb_path,
        instruction="Document bug fixes and improvements"
    )

asyncio.run(incremental_updates_example())
```

### 4. Batch Processing

```python
async def batch_processing_example():
    curator = DataStreamCurator()
    
    input_files = [
        "./github_prs.json",
        "./issue_discussions.json", 
        "./commit_messages.txt",
        "./documentation_updates.md"
    ]
    
    result = await curator.process_batch(
        input_files=input_files,
        output_path="./comprehensive_project_docs.md",
        instruction="Create unified project documentation from all sources"
    )
    
    print(f"Processed {len(input_files)} files into unified documentation")

asyncio.run(batch_processing_example())
```

### 5. Error Handling

```python
import logging
from datastream_curator import DataStreamCurator

async def robust_processing_example():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    curator = DataStreamCurator()
    
    try:
        result = await curator.process(
            input_data="./data.json",
            output_path="./output.md",
            instruction="Process this data carefully"
        )
        logger.info("Processing completed successfully")
        
    except FileNotFoundError:
        logger.error("Input file not found")
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Implement fallback or recovery logic

asyncio.run(robust_processing_example())
```

### 6. Real-World Use Cases

#### Documentation Generation from PR Data
```python
async def github_docs_example():
    """Generate documentation from GitHub PR data"""
    curator = DataStreamCurator()
    
    # GitHub PR data structure
    pr_data = {
        "pulls": [
            {
                "title": "Add LLM integration",
                "description": "Implements OpenRouter and OpenAI clients",
                "files_changed": ["src/llm.py", "tests/test_llm.py"],
                "discussion": "Discussion about API design and error handling"
            }
        ]
    }
    
    result = await curator.process(
        input_data=pr_data,
        existing_kb_path="./README.md",
        output_path="./README.md", 
        instruction="""
        Update project documentation based on PR information:
        - Add new features to feature list
        - Update architecture documentation
        - Document any breaking changes
        - Preserve existing content and structure
        """
    )

asyncio.run(github_docs_example())
```

#### Financial Data Analysis
```python
async def financial_analysis_example():
    """Maintain investment research knowledge base"""
    curator = DataStreamCurator()
    
    # Process different data sources sequentially
    sources = [
        ("./earnings_reports.json", "earnings data"),
        ("./market_news.txt", "market news"),
        ("./analyst_reports.pdf.txt", "analyst insights")
    ]
    
    for source_file, source_type in sources:
        await curator.process(
            input_data=source_file,
            existing_kb_path="./investment_analysis.md",
            output_path="./investment_analysis.md",
            instruction=f"""
            Update investment analysis with {source_type}:
            - Integrate new information into existing analysis
            - Update risk assessments based on new data
            - Maintain chronological order of events
            - Remove outdated information when contradicted
            """
        )

asyncio.run(financial_analysis_example())
```

---

## Configuration

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional overrides
LLM_MODEL=anthropic/claude-3-sonnet
LLM_PROVIDER=openrouter
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096

# Processing settings
PROCESSING_BATCH_SIZE=100
PROCESSING_MAX_CONCURRENT=5
PROCESSING_RETRY_ATTEMPTS=3

# Output settings  
OUTPUT_FORMAT=markdown
OUTPUT_INCLUDE_METADATA=true

# Logging
LOG_LEVEL=INFO
```

### Configuration File (`config.yaml`)

```yaml
llm:
  provider: "openrouter"           # openrouter, openai
  api_key: "${OPENROUTER_API_KEY}" # Environment variable reference
  model: "anthropic/claude-3-sonnet"
  temperature: 0.1                 # 0.0 = deterministic, 1.0 = creative
  max_tokens: 4096                 # Maximum response length
  timeout: 30                      # Request timeout in seconds

processing:
  batch_size: 100                  # Items per batch
  max_concurrent_requests: 5       # Concurrent LLM requests
  retry_attempts: 3                # Retry failed requests
  retry_delay: 1.0                 # Base delay between retries

output:
  format: "markdown"               # Output format
  include_metadata: true           # Include processing metadata
  preserve_structure: true         # Maintain existing structure

# Logging configuration
logging:
  level: "INFO"                    # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Provider-Specific Settings

#### OpenRouter Configuration
```yaml
llm:
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "anthropic/claude-3-sonnet"
  # Available models:
  # - anthropic/claude-3-sonnet
  # - anthropic/claude-3-opus  
  # - openai/gpt-4-turbo
  # - openai/gpt-3.5-turbo
  # - meta-llama/llama-2-70b-chat
```

#### OpenAI Configuration
```yaml
llm:
  provider: "openai"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4-turbo-preview"
  # Available models:
  # - gpt-4-turbo-preview
  # - gpt-4
  # - gpt-3.5-turbo
```

---

## Development Workflow

### 1. Feature Development Process

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Write tests first (TDD approach)
# Create test file: tests/test_new_feature.py

# 3. Run tests (should fail initially)
pytest tests/test_new_feature.py -v

# 4. Implement feature
# Create/modify source files

# 5. Run tests until they pass
pytest tests/test_new_feature.py -v

# 6. Run full test suite
pytest

# 7. Check code quality
black src/ tests/
ruff src/ tests/
mypy src/

# 8. Commit changes
git add .
git commit -m "feat: implement new feature with comprehensive tests"

# 9. Push and create PR
git push origin feature/new-feature
```

### 2. Testing Workflow

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest -k "test_llm"        # All LLM-related tests
pytest -k "asyncio"         # All async tests

# Run tests in parallel (faster)
pytest -n auto

# Run tests with verbose output
pytest -v --tb=short

# Watch for changes and re-run tests
pytest-watch
```

### 3. Code Quality Checks

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/ --fix

# Type checking
mypy src/

# Security checks
bandit -r src/

# All quality checks in one command
pre-commit run --all-files
```

### 4. Release Process

```bash
# 1. Update version in pyproject.toml and __init__.py
# 2. Update CHANGELOG.md with new features/fixes
# 3. Run full test suite
pytest

# 4. Build package
python -m build

# 5. Test package installation
pip install dist/datastream_curator-*.whl

# 6. Create git tag
git tag -a v0.1.0 -m "Release version 0.1.0"

# 7. Push tag
git push origin v0.1.0

# 8. Create GitHub release (if using GitHub)
gh release create v0.1.0 --generate-notes
```

### 5. Debugging and Troubleshooting

#### Common Issues and Solutions

**LLM API Errors**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test LLM connection
async def test_llm_connection():
    from datastream_curator.llm import LLMClient, DiffRequest
    from datastream_curator.config import LLMConfig
    
    config = LLMConfig(api_key="your-key", model="anthropic/claude-3-sonnet")
    
    async with LLMClient(config) as client:
        request = DiffRequest(
            existing_content="# Test",
            new_data="Test data",
            instruction="Test instruction"
        )
        
        try:
            result = await client.generate_diff(request)
            print("LLM connection successful")
        except Exception as e:
            print(f"LLM connection failed: {e}")
```

**Diff Application Issues**
```python
# Debug diff application
from datastream_curator.diff import DiffEngine

engine = DiffEngine()

# Test diff with sample data
test_diff = {
    "additions": [{"section": "Test", "content": "New content"}],
    "modifications": [],
    "deletions": []
}

result = engine.apply_diff("# Original Content", test_diff)
print(f"Result: {result}")
```

**Configuration Problems**
```python
# Validate configuration
from datastream_curator.config import CurationConfig

try:
    config = CurationConfig.from_env()
    print("Configuration valid")
except Exception as e:
    print(f"Configuration error: {e}")
```

### 6. Performance Monitoring

```python
import time
import asyncio
from datastream_curator import DataStreamCurator

async def performance_test():
    curator = DataStreamCurator()
    
    start_time = time.time()
    
    result = await curator.process(
        input_data="./large_dataset.json",
        output_path="./output.md"
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processing completed in {processing_time:.2f} seconds")
    print(f"Output length: {len(result)} characters")
    
asyncio.run(performance_test())
```

---

## Deployment and Distribution

### Package Building

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI (test)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY pyproject.toml .

RUN pip install -e .

CMD ["python", "-m", "datastream_curator"]
```

---

## Future Enhancements

### Phase 2 Features
- CLI interface implementation
- Multiple output formats (JSON, YAML, HTML)
- Custom template support
- Advanced configuration options
- Performance optimizations

### Phase 3 Features  
- Plugin architecture
- Real-time streaming support
- Web interface
- Documentation platform integrations
- Multi-language support

### Phase 4 Features
- Team collaboration
- Access controls
- Audit logging
- Scalable deployment
- Enterprise integrations

---

This CLAUDE.md serves as the complete development guide for implementing DataStream Curator. Follow the test-driven development approach, maintain high code quality, and ensure comprehensive testing at every step.