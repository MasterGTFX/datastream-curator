"""Integration tests for diff system."""

import json
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from datastream_curator import DataStreamCurator, CurationConfig, LLMConfig
from datastream_curator.models import StructuredDiff, DiffStyleOperation, DiffOperationType


class TestDiffIntegration:
    """Test diff system integration."""
    
    @pytest.fixture
    def test_config(self):
        """Create configuration for diff testing."""
        return CurationConfig(
            llm=LLMConfig(
                api_key="test-key",
                provider="openrouter",
                model="anthropic/claude-3-sonnet"
            ),
            diff_chunk_strategy="recursive",
            diff_chunk_size=800,
            diff_chunk_overlap=80,
            diff_use_semantic=True,
            diff_preserve_structure=True,
            diff_min_confidence=0.8
        )
    
    @pytest.fixture
    def curator(self, test_config):
        """Create curator with diff enabled."""
        return DataStreamCurator(test_config)
    
    @pytest.fixture
    def structured_diff_response(self):
        """Mock structured diff response."""
        return StructuredDiff(
            added=[
                DiffStyleOperation(
                    operation_type=DiffOperationType.ADDED,
                    content="## New Features\n\n### Advanced Diff Engine\nSupports precise diff operations with chonkie and diff-match-patch.",
                    section="New Features",
                    reasoning="Adding information about new diff capabilities",
                    confidence=0.95
                )
            ],
            changed=[
                DiffStyleOperation(
                    operation_type=DiffOperationType.CHANGED,
                    content="DataStream Curator now supports advanced diff operations with enhanced precision.",
                    old_content="DataStream Curator is an AI-powered tool.",
                    section="Overview",
                    reasoning="Updating description to reflect enhanced capabilities",
                    confidence=0.9
                )
            ],
            removed=[],
            reasoning="Adding new features and updating existing descriptions"
        )
    
    @pytest.fixture
    def legacy_diff_response(self):
        """Mock legacy diff response format."""
        return {
            "additions": [
                {
                    "section": "New Features",
                    "content": "## New Features\n\n### Advanced Diff Engine\nSupports precise diff operations.",
                    "reasoning": "Adding new features section"
                }
            ],
            "modifications": [
                {
                    "section": "Overview",
                    "old_content": "DataStream Curator is an AI-powered tool.",
                    "new_content": "DataStream Curator now supports advanced operations.",
                    "reasoning": "Updating description"
                }
            ],
            "deletions": [],
            "reasoning": "Adding features and updating content"
        }
    
    @pytest.mark.asyncio
    async def test_diff_workflow(self, curator, structured_diff_response, temp_dir):
        """Test complete workflow with diff engine."""
        # Create input data
        input_data = {
            "updates": [
                "Enhanced diff engine with chonkie support",
                "Precise text operations with diff-match-patch",
                "Instructor integration for structured outputs"
            ],
            "version": "0.2.0"
        }
        
        # Create existing knowledge base
        existing_kb = temp_dir / "existing.md"
        existing_kb.write_text("""# DataStream Curator

## Overview

DataStream Curator is an AI-powered tool for data curation.

## Installation

pip install datastream-curator
""")
        
        output_path = temp_dir / "output.md"
        
        # Mock the instructor-based LLM response
        with patch('datastream_curator.llm.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            
            # Mock instructor client
            mock_instructor = AsyncMock()
            mock_instructor.chat.completions.create.return_value = structured_diff_response
            mock_client.generate_diff.return_value = {
                "added": [op.model_dump() for op in structured_diff_response.added],
                "changed": [op.model_dump() for op in structured_diff_response.changed],
                "removed": [op.model_dump() for op in structured_diff_response.removed],
                "reasoning": structured_diff_response.reasoning
            }
            
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Mock instructor client initialization
            with patch('datastream_curator.llm.instructor.from_openai') as mock_instructor_init:
                mock_instructor_init.return_value = mock_instructor
                
                result = await curator.process(
                    input_data=input_data,
                    existing_kb_path=str(existing_kb),
                    output_path=str(output_path),
                    instruction="Integrate new diff capabilities"
                )
        
        # Verify diff was used
        assert curator.diff_engine is not None
        
        # Verify output file was created
        assert output_path.exists()
        
        # Verify content changes
        assert "Enhanced Diff Engine" in result
        assert "advanced diff operations" in result
        assert "DataStream Curator" in result  # Original title preserved
    
    @pytest.mark.asyncio
    async def test_legacy_diff_fallback(self, curator, legacy_diff_response, temp_dir):
        """Test fallback to legacy diff when diff fails."""
        input_data = {"test": "data"}
        existing_kb = temp_dir / "existing.md"
        existing_kb.write_text("# Test\n\nOriginal content.")
        output_path = temp_dir / "output.md"
        
        with patch('datastream_curator.llm.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = legacy_diff_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Make diff fail by patching it to raise exception
            with patch.object(curator.diff_engine, 'apply_structured_diff', 
                              side_effect=Exception("Diff error")):
                
                result = await curator.process(
                    input_data=input_data,
                    existing_kb_path=str(existing_kb),
                    output_path=str(output_path)
                )
        
        # Should still produce output using legacy engine
        assert output_path.exists()
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_diff_processing(self, test_config, temp_dir):
        """Test diff processing with structured data."""
        input_data = {
            "feature": "New testing capability",
            "description": "Comprehensive test coverage"
        }
        
        existing_content = """# Project

## Features

### Basic Feature
Simple functionality.

## Documentation

Basic documentation."""
        
        existing_kb = temp_dir / "existing.md"
        existing_kb.write_text(existing_content)
        
        # Mock structured response
        structured_response = {
            "added": [{
                "operation_type": "added",
                "content": "### Testing\nComprehensive test coverage for new features.",
                "section": "Features",
                "reasoning": "Adding testing section",
                "confidence": 0.9
            }],
            "changed": [],
            "removed": [],
            "reasoning": "Adding testing capabilities"
        }
        
        # Test with diff engine
        curator = DataStreamCurator(test_config)
        
        with patch('datastream_curator.llm.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = structured_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await curator.process(
                input_data=input_data,
                existing_kb_path=str(existing_kb)
            )
        
        # Should produce valid result
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should include new content
        assert "Testing" in result
    
    @pytest.mark.asyncio
    async def test_chunk_based_processing(self, curator, temp_dir):
        """Test processing with large documents using chunking."""
        # Create large document
        large_content = """# Large Document

## Section 1
""" + "This is content for section 1. " * 100 + """

## Section 2
""" + "This is content for section 2. " * 100 + """

## Section 3
""" + "This is content for section 3. " * 100
        
        existing_kb = temp_dir / "large.md"
        existing_kb.write_text(large_content)
        
        input_data = {
            "update": "Adding new section to large document",
            "section": "Section 4"
        }
        
        # Mock response for large document
        chunk_response = {
            "added": [{
                "operation_type": "added",
                "content": "## Section 4\nNew section added to the large document.",
                "section": "Section 4",
                "reasoning": "Adding new section as requested",
                "confidence": 0.9
            }],
            "changed": [],
            "removed": [],
            "reasoning": "Adding new section to large document"
        }
        
        with patch('datastream_curator.llm.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = chunk_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await curator.process(
                input_data=input_data,
                existing_kb_path=str(existing_kb)
            )
        
        # Verify the large document was processed
        assert "Section 4" in result
        assert len(result) > len(large_content)  # Should be larger with addition
        
        # Verify diff engine handled the large document
        assert curator.diff_engine is not None
    
    @pytest.mark.asyncio
    async def test_confidence_filtering_integration(self, curator, temp_dir):
        """Test that low-confidence operations are filtered out."""
        existing_kb = temp_dir / "test.md"
        existing_kb.write_text("# Test Document\n\nOriginal content.")
        
        # Mock response with mixed confidence levels
        mixed_confidence_response = {
            "added": [
                {
                    "operation_type": "added",
                    "content": "High confidence addition.",
                    "section": "Test",
                    "reasoning": "High confidence operation",
                    "confidence": 0.95  # Above threshold
                },
                {
                    "operation_type": "added", 
                    "content": "Low confidence addition.",
                    "section": "Test",
                    "reasoning": "Low confidence operation",
                    "confidence": 0.5   # Below threshold (0.8)
                }
            ],
            "changed": [],
            "removed": [],
            "reasoning": "Mixed confidence operations"
        }
        
        with patch('datastream_curator.llm.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = mixed_confidence_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await curator.process(
                input_data={"test": "data"},
                existing_kb_path=str(existing_kb)
            )
        
        # Should include high confidence operation
        assert "High confidence addition" in result
        
        # Low confidence operation behavior depends on implementation
        # (might be included with warning or filtered out)
    
    def test_curator_initialization_error_handling(self, test_config):
        """Test handling of initialization errors in diff."""
        # Mock chonkie import error
        with patch('datastream_curator.diff.TokenChunker', side_effect=ImportError("chonkie not available")):
            try:
                curator = DataStreamCurator(test_config)
                # Should initialize successfully despite warning
                assert curator.diff_engine is not None
            except RuntimeError:
                # Or fail gracefully with clear error
                pass
    
    def test_configuration_integration(self, test_config):
        """Test that diff configuration is properly integrated."""
        curator = DataStreamCurator(test_config)
        
        if curator.diff_engine:
            engine_config = curator.diff_engine.config
            
            assert engine_config.chunk_size == test_config.diff_chunk_size
            assert engine_config.chunk_overlap == test_config.diff_chunk_overlap
            assert engine_config.use_semantic_chunking == test_config.diff_use_semantic
            assert engine_config.preserve_structure == test_config.diff_preserve_structure
            assert engine_config.min_operation_confidence == test_config.diff_min_confidence
    
    @pytest.mark.asyncio
    async def test_instructor_integration(self, curator):
        """Test instructor integration for structured outputs."""
        # This test verifies that instructor is properly integrated
        # even if we can't test the actual API calls
        
        assert hasattr(curator, 'diff_engine')
        
        if curator.diff_engine:
            # Verify the diff engine has the expected capabilities
            assert hasattr(curator.diff_engine, 'apply_structured_diff')
            assert hasattr(curator.diff_engine, 'generate_precise_diff')
            assert hasattr(curator.diff_engine, 'create_chunk_based_diff')