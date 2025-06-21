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

### Development Notes

- **WSL Usage**: When in WSL, use local 'venv' to activate environment

The rest of the document remains unchanged.