# DataStream Curator

## Overview

DataStream Curator is an AI-powered tool for incremental data curation and knowledge base management. It uses Large Language Models to intelligently integrate new information into existing knowledge bases while preserving structure and context.

## Features

### LLM Integration
Connect to various LLM providers including OpenRouter and OpenAI for intelligent data processing and structured diff generation.

### Configuration Management  
Flexible configuration system supporting YAML files and environment variables for easy customization.

### Async Processing
Built on async/await for high-performance concurrent operations and efficient resource usage.

## Installation

```bash
pip install datastream-curator
```

For development installation:

```bash
pip install datastream-curator[dev]
```

## Quick Start

### Basic Usage

```python
import asyncio
from datastream_curator import DataStreamCurator

async def main():
    curator = DataStreamCurator()
    
    # Process input data and update knowledge base
    result = await curator.process(
        input_data="./new_data.json",
        output_path="./knowledge_base.md",
        instruction="Integrate new project information"
    )
    
    print(f"Updated knowledge base: {len(result)} characters")

asyncio.run(main())
```

### Configuration

Create a `config.yaml` file:

```yaml
llm:
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "anthropic/claude-3-sonnet" 
  temperature: 0.1

processing:
  batch_size: 100
  retry_attempts: 3

output:
  format: "markdown"
  include_metadata: true
```

## Architecture

The DataStream Curator follows a modular architecture:

- **Core Module** (`core.py`): Main `DataStreamCurator` class
- **LLM Module** (`llm.py`): LLM API integration and structured prompting
- **Diff Module** (`diff.py`): Structured diff generation and application
- **Config Module** (`config.py`): Configuration management with validation

## API Reference

### DataStreamCurator

Main class for data curation operations.

#### Methods

- `process(input_data, output_path, existing_kb_path, instruction)`: Process single input
- `process_batch(input_files, output_path, instruction)`: Process multiple files
- `validate_config()`: Validate current configuration
- `test_llm_connection()`: Test LLM API connectivity

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.