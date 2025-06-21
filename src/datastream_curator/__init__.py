"""
DataStream Curator - AI-powered incremental data curation and knowledge base management.

This package provides tools for intelligently curating and updating knowledge bases
using Large Language Models (LLMs) with structured diff generation and application.
"""

from .config import CurationConfig, LLMConfig, LoggingConfig, OutputConfig, ProcessingConfig
from .core import DataStreamCurator
from .llm import DiffRequest, LLMClient, LLMResponse
from .diff import DiffEngine
from .models import (
    DiffStyleOperation,
    DiffOperationType,
    StructuredDiff,
    ChunkBasedDiff,
    DiffConfig,
    ChunkStrategy,
    PatchResult
)

__version__ = "0.1.0"
__author__ = "DataStream Curator Team"
__email__ = "contact@datastream-curator.dev"
__description__ = "AI-powered incremental data curation and knowledge base management"

__all__ = [
    # Main classes
    "DataStreamCurator",
    
    # Configuration
    "CurationConfig",
    "LLMConfig",
    "ProcessingConfig",
    "OutputConfig",
    "LoggingConfig",
    "DiffConfig",
    
    # LLM integration
    "LLMClient",
    "DiffRequest",
    "LLMResponse",
    
    # Diff engine
    "DiffEngine",
    "DiffStyleOperation",
    "DiffOperationType",
    "StructuredDiff",
    "ChunkBasedDiff",
    "ChunkStrategy",
    "PatchResult",
]


def get_version() -> str:
    """Get the current version of DataStream Curator."""
    return __version__


def create_curator(config_path: str = None, **kwargs) -> DataStreamCurator:
    """
    Create a DataStreamCurator instance with optional configuration.
    
    Args:
        config_path: Path to configuration YAML file
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured DataStreamCurator instance
    """
    if config_path:
        config = CurationConfig.from_file(config_path)
    else:
        config = CurationConfig.from_env()
    
    # Override with any provided kwargs
    if kwargs:
        config_dict = config.model_dump()
        
        # Handle nested configuration updates
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'llm.temperature'
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        config = CurationConfig(**config_dict)
    
    return DataStreamCurator(config)