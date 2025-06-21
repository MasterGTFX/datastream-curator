"""Configuration management for DataStream Curator."""

import os
from typing import Optional
import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM integration."""
    
    provider: str = "openrouter"
    api_key: str = Field(..., description="LLM API key")
    model: str = "anthropic/claude-3-sonnet"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 30


class ProcessingConfig(BaseModel):
    """Configuration for processing parameters."""
    
    batch_size: int = 100
    max_concurrent_requests: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0


class OutputConfig(BaseModel):
    """Configuration for output formatting."""
    
    format: str = "markdown"
    include_metadata: bool = True
    preserve_structure: bool = True


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class CurationConfig(BaseModel):
    """Main configuration class combining all settings."""
    
    llm: LLMConfig
    processing: ProcessingConfig = ProcessingConfig()
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Diff configuration will be added by enhanced_diff module
    diff_chunk_strategy: str = "recursive"
    diff_chunk_size: int = 1000
    diff_chunk_overlap: int = 100
    diff_use_semantic: bool = True
    diff_preserve_structure: bool = True
    diff_min_confidence: float = 0.7
    
    @classmethod
    def from_file(cls, config_path: str) -> "CurationConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Expand environment variables in the data
        data = cls._expand_env_vars(data)
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "CurationConfig":
        """Create configuration from environment variables."""
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not api_key:
            raise ValueError(
                "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
            )
        
        provider = os.getenv("LLM_PROVIDER", "openrouter")
        model = os.getenv("LLM_MODEL", "anthropic/claude-3-sonnet")
        
        return cls(
            llm=LLMConfig(
                api_key=api_key,
                model=model,
                provider=provider,
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
                timeout=int(os.getenv("LLM_TIMEOUT", "30"))
            ),
            processing=ProcessingConfig(
                batch_size=int(os.getenv("PROCESSING_BATCH_SIZE", "100")),
                max_concurrent_requests=int(os.getenv("PROCESSING_MAX_CONCURRENT", "5")),
                retry_attempts=int(os.getenv("PROCESSING_RETRY_ATTEMPTS", "3")),
                retry_delay=float(os.getenv("PROCESSING_RETRY_DELAY", "1.0"))
            ),
            output=OutputConfig(
                format=os.getenv("OUTPUT_FORMAT", "markdown"),
                include_metadata=os.getenv("OUTPUT_INCLUDE_METADATA", "true").lower() == "true",
                preserve_structure=os.getenv("OUTPUT_PRESERVE_STRUCTURE", "true").lower() == "true"
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ),
            diff_chunk_strategy=os.getenv("DIFF_CHUNK_STRATEGY", "recursive"),
            diff_chunk_size=int(os.getenv("DIFF_CHUNK_SIZE", "1000")),
            diff_chunk_overlap=int(os.getenv("DIFF_CHUNK_OVERLAP", "100")),
            diff_use_semantic=os.getenv("DIFF_USE_SEMANTIC", "true").lower() == "true",
            diff_preserve_structure=os.getenv("DIFF_PRESERVE_STRUCTURE", "true").lower() == "true",
            diff_min_confidence=float(os.getenv("DIFF_MIN_CONFIDENCE", "0.7"))
        )
    
    @staticmethod
    def _expand_env_vars(data):
        """Recursively expand environment variables in configuration data."""
        if isinstance(data, dict):
            return {key: CurationConfig._expand_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [CurationConfig._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            # Extract environment variable name
            env_var = data[2:-1]  # Remove ${ and }
            return os.getenv(env_var, "")
        else:
            return data