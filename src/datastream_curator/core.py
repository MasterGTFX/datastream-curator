"""Main curator class for DataStream Curator."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from markitdown import MarkItDown

from .config import CurationConfig
from .llm import DiffRequest, LLMClient
from .diff import DiffEngine
from .models import DiffConfig, ChunkStrategy, SimpleDiff, DiffOperation

logger = logging.getLogger(__name__)


class DataStreamCurator:
    """Main class for incremental data curation and knowledge base management."""
    
    def __init__(self, config: Optional[CurationConfig] = None):
        """Initialize the curator with configuration."""
        self.config = config or CurationConfig.from_env()
        
        # Initialize MarkItDown for input processing
        self.markitdown = MarkItDown(enable_plugins=False)
        
        # Initialize enhanced diff engine
        try:
            # Create enhanced diff config from curation config
            diff_config = DiffConfig(
                chunk_strategy=ChunkStrategy(self.config.diff_chunk_strategy),
                chunk_size=self.config.diff_chunk_size,
                chunk_overlap=self.config.diff_chunk_overlap,
                use_semantic_chunking=self.config.diff_use_semantic,
                preserve_structure=self.config.diff_preserve_structure,
                min_operation_confidence=self.config.diff_min_confidence
            )
            self.diff_engine = DiffEngine(diff_config)
            logger.info("Diff engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize diff engine: {e}")
            raise RuntimeError(f"Cannot initialize DataStreamCurator without diff engine: {e}")
        
        # Set up logging based on configuration
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level.upper()),
            format=self.config.logging.format
        )
        
        logger.info("DataStream Curator initialized")
    
    async def process(
        self,
        input_data: Union[str, Dict[str, Any]],
        output_path: Optional[str] = None,
        existing_kb_path: Optional[str] = None,
        instruction: Optional[str] = None
    ) -> str:
        """
        Process input data and update knowledge base.
        
        Args:
            input_data: Input data (file path, data dict, or raw text)
            output_path: Output file path for updated knowledge base
            existing_kb_path: Path to existing knowledge base file
            instruction: Custom curation instruction
            
        Returns:
            Updated knowledge base content
        """
        logger.info("Starting data curation process")
        
        try:
            # Load existing knowledge base
            existing_content = ""
            if existing_kb_path and Path(existing_kb_path).exists():
                existing_content = Path(existing_kb_path).read_text(encoding='utf-8')
                logger.info(f"Loaded existing knowledge base from {existing_kb_path} ({len(existing_content)} chars)")
            elif existing_kb_path:
                logger.warning(f"Existing KB path specified but file not found: {existing_kb_path}")
            
            # Process input data
            processed_data = await self._process_input_data(input_data)
            logger.info(f"Processed input data ({len(processed_data)} chars)")
            
            # Generate diff using LLM
            async with LLMClient(
                self.config.llm, 
                retry_attempts=self.config.processing.retry_attempts,
                retry_delay=self.config.processing.retry_delay
            ) as llm_client:
                diff_request = DiffRequest(
                    existing_content=existing_content,
                    new_data=processed_data,
                    instruction=instruction or "Intelligently integrate new information into the knowledge base",
                    context=f"Output format: {self.config.output.format}"
                )
                
                logger.info("Generating diff using LLM")
                diff_data = await llm_client.generate_diff(diff_request)
            
            # Apply diff to existing content
            logger.info("Applying diff to knowledge base using diff engine")
            
            # Check if we got a SimpleDiff from instructor
            if isinstance(diff_data, SimpleDiff):
                patch_result = self.diff_engine.apply_simple_diff(existing_content, diff_data)
                updated_content = patch_result.content
                
                # Log results
                if patch_result.errors:
                    logger.warning(f"Diff operation had {len(patch_result.errors)} errors: {patch_result.errors}")
                logger.info(f"Applied {patch_result.stats['applied_count']} diff operations successfully")
            else:
                # Unexpected format
                logger.error(f"Unexpected diff format received: {type(diff_data)}")
                raise ValueError(f"Cannot process diff data: expected SimpleDiff, got {type(diff_data)}")
            
            # Save updated content if output path specified
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(updated_content, encoding='utf-8')
                logger.info(f"Saved updated knowledge base to {output_path}")
            
            # Log final statistics
            if 'patch_result' in locals():
                logger.info(f"Curation complete: {patch_result.stats}")
            else:
                logger.info("Curation complete")
            
            return updated_content
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise
    
    async def process_batch(
        self,
        input_files: List[str],
        output_path: str,
        instruction: Optional[str] = None
    ) -> str:
        """
        Process multiple input files in batch.
        
        Args:
            input_files: List of input file paths
            output_path: Output file path for final knowledge base
            instruction: Custom curation instruction
            
        Returns:
            Final knowledge base content
        """
        logger.info(f"Processing batch of {len(input_files)} files")
        
        # Load existing content if output file exists
        output_file = Path(output_path)
        current_content = ""
        if output_file.exists():
            current_content = output_file.read_text(encoding='utf-8')
            logger.info(f"Starting with existing content ({len(current_content)} chars)")
        
        # Process each file sequentially
        for i, input_file in enumerate(input_files, 1):
            logger.info(f"Processing file {i}/{len(input_files)}: {input_file}")
            
            try:
                # Use current_content as the base for each iteration
                temp_output = str(output_file) + ".tmp"
                if current_content:
                    Path(temp_output).write_text(current_content, encoding='utf-8')
                
                current_content = await self.process(
                    input_data=input_file,
                    existing_kb_path=temp_output if current_content else None,
                    instruction=instruction
                )
                
                # Clean up temporary file
                if Path(temp_output).exists():
                    Path(temp_output).unlink()
                
            except Exception as e:
                logger.error(f"Error processing file {input_file}: {e}")
                # Continue with other files in batch
                continue
        
        # Save final result
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(current_content, encoding='utf-8')
        logger.info(f"Batch processing complete, saved to {output_path}")
        
        return current_content
    
    async def _process_input_data(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Process and normalize input data."""
        if isinstance(input_data, dict):
            # Convert dict to formatted JSON
            return json.dumps(input_data, indent=2, ensure_ascii=False)
        
        if isinstance(input_data, str):
            # Check if it's a file path
            input_path = Path(input_data)
            if input_path.exists() and input_path.is_file():
                logger.debug(f"Reading input from file: {input_path}")
                return self._read_file(input_path)
            else:
                # Treat as raw text data
                logger.debug("Treating input as raw text data")
                return input_data
        
        # Convert other types to string
        return str(input_data)
    
    def _read_file(self, file_path: Path) -> str:
        """Read and process different file formats using MarkItDown."""
        try:
            # Use MarkItDown to convert various formats to markdown
            result = self.markitdown.convert(str(file_path))
            
            if result and result.text_content:
                # Add file context header
                content = f"# Content from {file_path.name}\n\n{result.text_content}"
                logger.debug(f"Successfully processed {file_path} using MarkItDown")
                return content
            else:
                logger.warning(f"MarkItDown returned empty content for {file_path}")
                # Fallback to raw text reading
                return self._read_file_fallback(file_path)
                
        except Exception as e:
            logger.warning(f"MarkItDown failed for {file_path}: {e}, falling back to raw text")
            return self._read_file_fallback(file_path)
    
    def _read_file_fallback(self, file_path: Path) -> str:
        """Fallback method for reading files when MarkItDown fails."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
            content = file_path.read_text(encoding='latin-1')
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            # Pretty print JSON for better LLM processing
            try:
                data = json.loads(content)
                return f"# JSON Content from {file_path.name}\n\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in file {file_path}: {e}")
                return f"# Content from {file_path.name} (malformed JSON)\n\n{content}"
        
        elif suffix in ['.yaml', '.yml']:
            return f"# YAML Content from {file_path.name}\n\n```yaml\n{content}\n```"
        
        elif suffix == '.csv':
            return f"# CSV Content from {file_path.name}\n\n```csv\n{content}\n```"
        
        elif suffix == '.xml':
            return f"# XML Content from {file_path.name}\n\n```xml\n{content}\n```"
        
        elif suffix in ['.md', '.markdown']:
            return f"# Markdown Content from {file_path.name}\n\n{content}"
        
        elif suffix == '.txt':
            return f"# Text Content from {file_path.name}\n\n{content}"
        
        else:
            return f"# Content from {file_path.name} ({suffix} format)\n\n{content}"
    
    def validate_config(self) -> bool:
        """Validate the current configuration."""
        try:
            # Check if API key is set
            if not self.config.llm.api_key:
                logger.error("LLM API key is not set")
                return False
            
            # Check if provider is supported
            if self.config.llm.provider not in ["openrouter", "openai"]:
                logger.error(f"Unsupported LLM provider: {self.config.llm.provider}")
                return False
            
            # Check temperature range
            if not 0.0 <= self.config.llm.temperature <= 2.0:
                logger.error(f"Invalid temperature value: {self.config.llm.temperature}")
                return False
            
            # Check max_tokens
            if self.config.llm.max_tokens <= 0:
                logger.error(f"Invalid max_tokens value: {self.config.llm.max_tokens}")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    async def test_llm_connection(self) -> bool:
        """Test connection to LLM API."""
        try:
            async with LLMClient(self.config.llm) as llm_client:
                test_request = DiffRequest(
                    existing_content="# Test Document",
                    new_data="Test data for connection verification",
                    instruction="This is a test request to verify API connectivity"
                )
                
                result = await llm_client.generate_diff(test_request)
                
                if result and isinstance(result, SimpleDiff):
                    logger.info("LLM connection test successful")
                    return True
                else:
                    logger.error("LLM connection test failed: invalid response")
                    return False
                    
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return False
    
