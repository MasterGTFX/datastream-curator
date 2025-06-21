"""Tests for enhanced diff engine functionality."""

import pytest
from unittest.mock import patch, MagicMock

from datastream_curator.diff import DiffEngine
from datastream_curator.models import (
    DiffConfig,
    ChunkStrategy,
    DiffStyleOperation,
    DiffOperationType,
    StructuredDiff,
    ChunkBasedDiff,
    DocumentStructure
)


class TestDiffEngine:
    """Test DiffEngine class."""
    
    @pytest.fixture
    def diff_config(self):
        """Create test diff configuration."""
        return DiffConfig(
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=500,
            chunk_overlap=50,
            use_semantic_chunking=True,
            preserve_structure=True,
            min_operation_confidence=0.7
        )
    
    @pytest.fixture
    def diff_engine(self, diff_config):
        """Create diff engine."""
        return DiffEngine(diff_config)
    
    @pytest.fixture
    def sample_markdown(self):
        """Sample markdown content for testing."""
        return """# Test Document

## Introduction

This is a test document for evaluating the enhanced diff engine.

## Features

### Feature 1
Description of feature 1.

### Feature 2
Description of feature 2.

## Conclusion

This concludes the test document.
"""
    
    @pytest.fixture
    def modified_markdown(self):
        """Modified version of sample markdown."""
        return """# Test Document

## Introduction

This is a test document for evaluating the enhanced diff engine with improvements.

## Features

### Feature 1
Enhanced description of feature 1 with new capabilities.

### Feature 2
Description of feature 2.

### Feature 3
Description of new feature 3.

## Performance

New section about performance improvements.

## Conclusion

This concludes the updated test document.
"""
    
    def test_diff_engine_initialization(self, diff_engine, diff_config):
        """Test diff engine initialization."""
        assert diff_engine.config == diff_config
        assert diff_engine.dmp is not None
        assert diff_engine.chunkers is not None
    
    def test_analyze_document_structure(self, diff_engine, sample_markdown):
        """Test document structure analysis."""
        structure = diff_engine.analyze_document_structure(sample_markdown)
        
        assert isinstance(structure, DocumentStructure)
        assert len(structure.headings) > 0
        assert len(structure.sections) > 0
        assert len(structure.chunks) > 0
        
        # Check headings
        heading_titles = [h['title'] for h in structure.headings]
        assert 'Test Document' in heading_titles
        assert 'Introduction' in heading_titles
        assert 'Features' in heading_titles
        
        # Check sections
        section_titles = [s['title'] for s in structure.sections]
        assert 'Test Document' in section_titles
        assert 'Introduction' in section_titles
    
    @patch('datastream_curator.diff.RecursiveChunker')
    def test_chunking_with_mocked_chunker(self, mock_chunker_class, diff_engine):
        """Test chunking with mocked chunker."""
        # Setup mock
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        # Mock chunk objects
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "First chunk content"
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "Second chunk content"
        
        mock_chunker.chunk.return_value = [mock_chunk1, mock_chunk2]
        
        content = "First chunk content and more text. Second chunk content and even more."
        
        # Re-initialize engine to use mocked chunker
        diff_engine._setup_chunkers()
        chunks = diff_engine._create_content_chunks(content)
        
        assert len(chunks) == 2
        assert chunks[0].content == "First chunk content"
        assert chunks[1].content == "Second chunk content"
    
    def test_generate_precise_diff(self, diff_engine):
        """Test precise diff generation using diff-match-patch."""
        old_content = "Hello world\nThis is line 2\nThis is line 3"
        new_content = "Hello universe\nThis is line 2\nThis is new line 3\nThis is line 4"
        
        operations = diff_engine.generate_precise_diff(old_content, new_content)
        
        assert len(operations) > 0
        
        # Check operation types
        operation_types = [op.operation_type for op in operations]
        assert DiffOperationType.REMOVED in operation_types or DiffOperationType.ADDED in operation_types
        
        # Check that operations have position information
        for op in operations:
            assert op.char_start is not None
            assert op.char_end is not None
            assert op.confidence > 0
    
    def test_apply_structured_diff_additions(self, diff_engine):
        """Test applying structured diff with additions."""
        original_content = "# Original Document\n\nSome content."
        
        addition = DiffStyleOperation(
            operation_type=DiffOperationType.ADDED,
            content="## New Section\n\nNew content here.",
            section="New Section",
            reasoning="Adding new section",
            confidence=0.9
        )
        
        structured_diff = StructuredDiff(
            added=[addition],
            changed=[],
            removed=[],
            reasoning="Adding new content"
        )
        
        result = diff_engine.apply_structured_diff(original_content, structured_diff)
        
        assert result.stats['applied_count'] == 1
        assert result.stats['skipped_count'] == 0
        assert "New Section" in result.content
        assert "New content here" in result.content
    
    def test_apply_structured_diff_modifications(self, diff_engine):
        """Test applying structured diff with modifications."""
        original_content = "# Document\n\nOld content that needs updating."
        
        modification = DiffStyleOperation(
            operation_type=DiffOperationType.CHANGED,
            content="New content that has been updated.",
            old_content="Old content that needs updating.",
            reasoning="Updating content",
            confidence=0.95
        )
        
        structured_diff = StructuredDiff(
            added=[],
            changed=[modification],
            removed=[],
            reasoning="Updating existing content"
        )
        
        result = diff_engine.apply_structured_diff(original_content, structured_diff)
        
        assert result.stats['applied_count'] == 1
        assert "New content that has been updated" in result.content
        assert "Old content that needs updating" not in result.content
    
    def test_apply_structured_diff_removals(self, diff_engine):
        """Test applying structured diff with removals."""
        original_content = """# Document

## Section 1
Content to keep.

## Section 2
Content to remove.

## Section 3
More content to keep."""
        
        removal = DiffStyleOperation(
            operation_type=DiffOperationType.REMOVED,
            content="## Section 2\nContent to remove.",
            reasoning="Removing outdated section",
            confidence=0.9
        )
        
        structured_diff = StructuredDiff(
            added=[],
            changed=[],
            removed=[removal],
            reasoning="Removing outdated content"
        )
        
        result = diff_engine.apply_structured_diff(original_content, structured_diff)
        
        assert result.stats['applied_count'] == 1
        assert "Section 2" not in result.content
        assert "Content to remove" not in result.content
        assert "Section 1" in result.content
        assert "Section 3" in result.content
    
    def test_apply_structured_diff_error_handling(self, diff_engine):
        """Test error handling in structured diff application."""
        original_content = "# Document\n\nSome content."
        
        # Create an operation with invalid position information
        invalid_operation = DiffStyleOperation(
            operation_type=DiffOperationType.CHANGED,
            content="New content",
            char_start=1000,  # Invalid position
            char_end=2000,
            reasoning="Invalid operation",
            confidence=0.5
        )
        
        structured_diff = StructuredDiff(
            added=[],
            changed=[invalid_operation],
            removed=[],
            reasoning="Testing error handling"
        )
        
        result = diff_engine.apply_structured_diff(original_content, structured_diff)
        
        assert result.stats['error_count'] > 0
        assert result.stats['skipped_count'] > 0
        assert len(result.errors) > 0
    
    def test_create_chunk_based_diff(self, diff_engine, sample_markdown, modified_markdown):
        """Test creating chunk-based diff for large documents."""
        chunk_diff = diff_engine.create_chunk_based_diff(sample_markdown, modified_markdown)
        
        assert isinstance(chunk_diff, ChunkBasedDiff)
        assert chunk_diff.total_chunks > 0
        assert chunk_diff.chunk_strategy == ChunkStrategy.RECURSIVE
        assert chunk_diff.original_length == len(sample_markdown)
        
        # Check that chunks have operations
        has_operations = any(len(chunk.operations) > 0 for chunk in chunk_diff.chunks)
        assert has_operations
    
    def test_merge_operations(self, diff_engine):
        """Test merging adjacent operations."""
        op1 = DiffStyleOperation(
            operation_type=DiffOperationType.ADDED,
            content="First addition",
            char_start=10,
            char_end=25,
            reasoning="First operation",
            confidence=0.9
        )
        
        op2 = DiffStyleOperation(
            operation_type=DiffOperationType.ADDED,
            content=" Second addition",
            char_start=25,
            char_end=41,
            reasoning="Second operation",
            confidence=0.8
        )
        
        operations = [op1, op2]
        merged = diff_engine.merge_operations(operations)
        
        # Should merge adjacent operations of same type
        assert len(merged) == 1
        assert merged[0].content == "First addition Second addition"
        assert merged[0].char_start == 10
        assert merged[0].char_end == 41
        assert merged[0].confidence == 0.8  # Minimum confidence
    
    def test_fallback_chunking(self, diff_engine):
        """Test fallback line-based chunking when regular chunking fails."""
        # Mock chunker to raise exception
        with patch.object(diff_engine, 'chunkers', {}):
            content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
            chunks = diff_engine._create_content_chunks(content)
            
            assert len(chunks) > 0
            assert all(chunk.metadata['strategy'] == 'fallback' for chunk in chunks)
    
    def test_section_based_addition(self, diff_engine):
        """Test adding content to specific sections."""
        content = """# Main Title

## Section A
Existing content A.

## Section B
Existing content B."""
        
        new_content = "Additional content for section A."
        result = diff_engine._add_to_section(content, "Section A", new_content)
        
        assert "Additional content for section A" in result
        # Should be inserted after Section A heading
        lines = result.split('\n')
        section_a_index = next(i for i, line in enumerate(lines) if "Section A" in line)
        assert "Additional content for section A" in lines[section_a_index + 2]
    
    def test_section_not_found_addition(self, diff_engine):
        """Test adding content when section is not found."""
        content = "# Main Title\n\nExisting content."
        new_content = "New section content."
        
        result = diff_engine._add_to_section(content, "New Section", new_content)
        
        # Should append new section at end
        assert "# New Section" in result
        assert "New section content" in result
    
    @pytest.mark.parametrize("chunk_strategy", [
        ChunkStrategy.TOKEN,
        ChunkStrategy.SENTENCE,
        ChunkStrategy.RECURSIVE,
    ])
    def test_different_chunk_strategies(self, chunk_strategy):
        """Test different chunking strategies."""
        config = DiffConfig(chunk_strategy=chunk_strategy)
        engine = DiffEngine(config)
        
        assert engine.config.chunk_strategy == chunk_strategy
        assert chunk_strategy in engine.chunkers or chunk_strategy == ChunkStrategy.SEMANTIC
    
    def test_confidence_filtering(self, diff_engine):
        """Test filtering operations by confidence level."""
        operations = [
            DiffStyleOperation(
                operation_type=DiffOperationType.ADDED,
                content="High confidence",
                confidence=0.9,
                reasoning="High confidence operation"
            ),
            DiffStyleOperation(
                operation_type=DiffOperationType.ADDED,
                content="Low confidence",
                confidence=0.5,
                reasoning="Low confidence operation"
            )
        ]
        
        # Filter by minimum confidence
        min_confidence = diff_engine.config.min_operation_confidence
        filtered = [op for op in operations if op.confidence >= min_confidence]
        
        assert len(filtered) == 1
        assert filtered[0].content == "High confidence"
    
    def test_document_structure_metadata(self, diff_engine, sample_markdown):
        """Test document structure metadata extraction."""
        structure = diff_engine.analyze_document_structure(sample_markdown)
        
        metadata = structure.metadata
        assert 'total_chars' in metadata
        assert 'total_lines' in metadata
        assert 'heading_count' in metadata
        assert 'chunk_count' in metadata
        
        assert metadata['total_chars'] == len(sample_markdown)
        assert metadata['heading_count'] > 0
        assert metadata['chunk_count'] > 0