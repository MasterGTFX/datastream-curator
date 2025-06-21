# DataStream Curator

**An AI-powered incremental data curation and knowledge base management tool**

## Overview

DataStream Curator is a Python tool that intelligently maintains and updates knowledge bases by processing streams of input data. Using Large Language Models (LLMs) as agents, it analyzes incoming data to add relevant information, remove obsolete content, and update existing entries based on user-defined instructions or default curation strategies.

## Core Features

### 🔄 Incremental Data Processing
- Processes data streams incrementally to build sophisticated knowledge bases
- Maintains context and relationships across data updates
- Optimized for continuous integration workflows

### 🤖 LLM-Powered Intelligence  
- Uses LLMs to make intelligent decisions about data relevance and updates
- Supports custom user instructions for domain-specific curation
- Generates structured diffs for reliable data transformations

### 📚 Universal Data Support
- Handles multiple input formats (JSON, CSV, XML, plain text, etc.)
- Flexible schema design accommodating various data structures
- Markdown-formatted output with extensible format support

### 🏗️ Dual Interface Design
- **CLI Tool**: Direct command-line usage with flexible argument structure
- **Python Module**: Programmatic integration for custom workflows

### 📈 Version Control & History
- Automatic checkpoint saving with complete change history
- Structured diff tracking in `.history/` subdirectories
- Optional checkpoint skipping for performance-critical scenarios

## Architecture

### System Components

```
DataStream Curator
├── Core Engine
│   ├── Data Ingestion Pipeline
│   ├── LLM Integration Layer
│   ├── Diff Generation Engine
│   └── Knowledge Base Manager
├── Interfaces
│   ├── CLI Interface
│   └── Python Module API
├── Storage Layer
│   ├── Checkpoint Manager
│   ├── History Tracker
│   └── Configuration Handler
└── LLM Providers
    ├── OpenRouter Integration
    ├── OpenAI Compatible Endpoints
    └── Extensible Provider Framework
```

### Data Flow

1. **Input Processing**: Raw data streams are normalized and validated
2. **Context Analysis**: LLM analyzes existing knowledge base and incoming data
3. **Diff Generation**: Structured differences are generated using LLM reasoning
4. **Smart Application**: Diffs are intelligently applied to maintain data integrity
5. **Checkpoint Creation**: Version snapshots are saved with metadata
6. **Output Generation**: Updated knowledge base is formatted and exported

## Technical Specifications

### Dependencies

```python
# Core Dependencies
- click>=8.0.0           # CLI framework
- pydantic>=2.0.0        # Data validation and settings
- aiohttp>=3.8.0         # Async HTTP client for LLM APIs
- jinja2>=3.1.0          # Template rendering
- pyyaml>=6.0            # Configuration file support
- rich>=13.0.0           # Enhanced CLI output

# Data Processing
- pandas>=2.0.0          # Data manipulation
- python-magic>=0.4.27   # File type detection
- charset-normalizer>=3.0.0  # Character encoding detection

# Storage & Serialization  
- orjson>=3.8.0          # Fast JSON processing
- msgpack>=1.0.0         # Binary serialization
```

### Configuration Schema

```yaml
# config.yaml
llm:
  provider: "openrouter"  # openrouter, openai, custom
  api_key: "${OPENROUTER_API_KEY}"
  model: "anthropic/claude-3-sonnet"
  temperature: 0.1
  max_tokens: 4096
  
storage:
  checkpoint_enabled: true
  history_retention_days: 30
  compression_enabled: true
  
processing:
  batch_size: 100
  max_concurrent_requests: 5
  retry_attempts: 3
  retry_delay: 1.0
  
output:
  format: "markdown"
  template_path: null
  include_metadata: true
```

### CLI Interface Design

```bash
# Basic usage
datastream-curator <input_data> [OPTIONS]

# Required Arguments
input_data          Path to input data file or directory

# Optional Arguments  
-o, --output FILE        Output file path (creates if doesn't exist)
-c, --config FILE        Configuration file path
-i, --instruction TEXT   Custom curation instructions
-m, --model TEXT         Override LLM model
--no-checkpoint         Skip checkpoint saving
--template FILE         Custom output template
--format FORMAT         Output format (markdown, json, yaml)
--verbose              Enable verbose logging
--dry-run              Preview changes without applying

# Examples
datastream-curator ./pr_history.json -o project_docs.md -i "Create technical documentation focusing on architecture decisions"
datastream-curator ./stock_data/ -o knowledge_base.md -c config.yaml --format markdown
datastream-curator news_feed.xml -o --no-checkpoint --dry-run
```

### Python Module API

```python
from datastream_curator import DataStreamCurator, CurationConfig

# Basic usage
curator = DataStreamCurator(
    config=CurationConfig(
        llm_provider="openrouter",
        model="anthropic/claude-3-sonnet",
        checkpoint_enabled=True
    )
)

# Process data with custom instructions
result = await curator.process(
    input_data="./pr_history.json",
    output_path="./project_docs.md", 
    instruction="Create comprehensive project documentation with architectural insights",
    existing_kb_path="./existing_docs.md"  # Optional existing knowledge base
)

# Batch processing
async for checkpoint in curator.process_stream(
    data_stream=data_generator(),
    instruction="Maintain real-time stock analysis",
    checkpoint_interval=100
):
    print(f"Processed checkpoint: {checkpoint.version}")

# Access history
history = curator.get_history("./project_docs.md")
for version in history:
    print(f"Version {version.id}: {version.summary}")
```

## Use Cases

### 1. Automated Documentation Generation

**Scenario**: Maintain up-to-date project documentation from repository activity

```python
# GitHub Action Integration
curator = DataStreamCurator()
await curator.process(
    input_data=github_pr_data,
    output_path="README.md",
    instruction="""
    Create comprehensive project documentation including:
    - Architecture decisions from PR discussions
    - Feature evolution and roadmap
    - Breaking changes and migration guides
    - Contributor insights and patterns
    """,
    existing_kb_path="README.md"
)
```

**Benefits**:
- Documentation stays synchronized with code changes
- Historical context preserved through incremental updates  
- Architectural decisions captured from actual development discussions
- Reduces manual documentation maintenance burden

### 2. Dynamic Knowledge Base Management

**Scenario**: Aggregate financial intelligence from multiple data sources

```python
# Multi-source financial data curation
sources = [
    "./news_feed.json",      # Financial news
    "./earnings_calls.txt",   # Earnings transcripts  
    "./forum_posts.csv",     # Investor discussions
    "./sec_filings.xml"      # Regulatory filings
]

for source in sources:
    await curator.process(
        input_data=source,
        output_path="stock_analysis.md",
        instruction="Maintain comprehensive investment research focusing on market trends, company performance, and risk factors"
    )
```

**Benefits**:
- Intelligent synthesis of diverse information sources
- Automatic removal of outdated or contradictory information
- Structured analysis with source attribution
- Continuous knowledge base refinement

## File Structure & History Management

### Directory Organization

```
project_root/
├── knowledge_base.md           # Main output file
├── .history/                   # Version history directory
│   ├── v001_20250621_143022/   # Timestamp-based versions
│   │   ├── diff.json          # Structured diff data
│   │   ├── metadata.json      # Version metadata
│   │   └── snapshot.md        # Full snapshot
│   ├── v002_20250621_150145/
│   └── ...
├── config.yaml                 # Configuration file
└── .curator_state             # Internal state file
```

### Diff Structure

```json
{
  "version": "v002_20250621_150145",
  "timestamp": "2025-06-21T15:01:45Z",
  "input_source": "./new_data.json",
  "instruction": "Update project documentation with latest features",
  "changes": {
    "additions": [
      {
        "section": "Features",
        "content": "### New Authentication System\nImplemented OAuth 2.0 integration...",
        "reasoning": "Added based on recent PR #123 implementing OAuth functionality"
      }
    ],
    "modifications": [
      {
        "section": "Installation", 
        "old_content": "pip install myproject==1.0.0",
        "new_content": "pip install myproject==1.1.0",
        "reasoning": "Updated version number to reflect latest release"
      }
    ],
    "deletions": [
      {
        "section": "Deprecated Features",
        "content": "### Legacy API\nThe v1 API is deprecated...",
        "reasoning": "Removed deprecated API documentation as it's no longer supported"
      }
    ]
  },
  "statistics": {
    "sections_added": 1,
    "sections_modified": 1, 
    "sections_removed": 1,
    "total_tokens": 2847
  }
}
```

## LLM Integration

### Prompt Engineering

The system uses structured prompts to ensure reliable diff generation:

```
You are a knowledge base curator. Analyze the existing knowledge base and new input data to generate structured updates.

EXISTING KNOWLEDGE BASE:
{existing_content}

NEW INPUT DATA:
{input_data}

USER INSTRUCTION:
{user_instruction}

Generate a JSON response with the following structure:
{
  "additions": [...],
  "modifications": [...], 
  "deletions": [...],
  "reasoning": "Overall reasoning for changes"
}

Focus on:
1. Accuracy and factual correctness
2. Maintaining consistency and coherence
3. Preserving important historical context
4. Following the user's specific curation goals
```

### Error Handling & Reliability

- **Structured Output Validation**: All LLM responses are validated against Pydantic schemas
- **Retry Logic**: Failed API calls are retried with exponential backoff
- **Fallback Strategies**: Graceful degradation when LLM services are unavailable
- **Diff Verification**: Generated diffs are validated before application
- **Rollback Capability**: Any version can be restored from history

## Installation & Setup

### PyPI Installation

```bash
pip install datastream-curator
```

### Development Setup

```bash
git clone https://github.com/your-org/datastream-curator.git
cd datastream-curator
pip install -e ".[dev]"
```

### Environment Configuration

```bash
# Required environment variables
export OPENROUTER_API_KEY="your-api-key"

# Optional configuration
export DATASTREAM_CONFIG_PATH="./config.yaml"
export DATASTREAM_LOG_LEVEL="INFO"
```

## Roadmap

### Phase 1: Core Implementation ✅
- Basic CLI interface with essential arguments
- OpenRouter/OpenAI API integration  
- Markdown output format
- Simple checkpoint system
- Incremental processing foundation

### Phase 2: Enhanced Features 🚧
- Multiple output formats (JSON, YAML, HTML)
- Custom template support
- Advanced configuration options
- Performance optimizations
- Comprehensive test coverage

### Phase 3: Advanced Capabilities 📋
- Plugin architecture for custom processors
- Real-time streaming support  
- Web interface for knowledge base management
- Integration with popular documentation platforms
- Multi-language support

### Phase 4: Enterprise Features 📋
- Team collaboration features
- Advanced access controls
- Audit logging and compliance
- Scalable deployment options
- Enterprise integrations

## Contributing

### Development Principles
- **Test-Driven Development**: Comprehensive test coverage for all features
- **Type Safety**: Full type hints and Pydantic validation
- **Async-First**: Designed for high-performance async operations
- **Extensibility**: Plugin-friendly architecture for customization
- **Documentation**: Self-documenting code with comprehensive examples

### Getting Started
1. Fork the repository
2. Set up development environment
3. Run test suite: `pytest`
4. Submit pull requests with tests and documentation

## License

MIT License - see LICENSE file for details.

---

*DataStream Curator: Intelligent knowledge curation for the modern data landscape.*
