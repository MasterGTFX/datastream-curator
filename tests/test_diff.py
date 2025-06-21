"""Tests for diff engine functionality."""

import pytest

from datastream_curator.diff import DiffEngine, DiffOperation, DiffResult


class TestDiffEngine:
    """Test DiffEngine class."""
    
    def test_apply_diff_empty_content(self):
        """Test applying diff to empty content."""
        engine = DiffEngine()
        
        diff_data = {
            "additions": [
                {
                    "section": "Introduction",
                    "content": "# New Document\n\nThis is a new document.",
                    "reasoning": "Creating initial content"
                }
            ],
            "modifications": [],
            "deletions": [],
            "reasoning": "Initial document creation"
        }
        
        result = engine.apply_diff("", diff_data)
        
        assert "# New Document" in result
        assert "This is a new document." in result
    
    def test_apply_addition_to_existing_content(self, sample_markdown_content):
        """Test adding new content to existing markdown."""
        engine = DiffEngine()
        
        diff_data = {
            "additions": [
                {
                    "section": "Features",
                    "content": "### New Feature\nThis is a newly added feature.",
                    "reasoning": "Added based on input data"
                }
            ],
            "modifications": [],
            "deletions": [],
            "reasoning": "Added new feature"
        }
        
        result = engine.apply_diff(sample_markdown_content, diff_data)
        
        # Should contain original content
        assert "# DataStream Curator" in result
        assert "## Features" in result
        
        # Should contain new content
        assert "### New Feature" in result
        assert "This is a newly added feature." in result
    
    def test_apply_modification_exact_match(self, sample_markdown_content):
        """Test modifying existing content with exact match."""
        engine = DiffEngine()
        
        diff_data = {
            "additions": [],
            "modifications": [
                {
                    "section": "Installation",
                    "old_content": "pip install datastream-curator",
                    "new_content": "pip install datastream-curator[dev]",
                    "reasoning": "Updated to include dev dependencies"
                }
            ],
            "deletions": [],
            "reasoning": "Updated installation command"
        }
        
        result = engine.apply_diff(sample_markdown_content, diff_data)
        
        assert "pip install datastream-curator[dev]" in result
        assert "pip install datastream-curator" not in result or "pip install datastream-curator[dev]" in result
    
    def test_apply_deletion_exact_match(self, sample_markdown_content):
        """Test deleting content with exact match."""
        engine = DiffEngine()
        
        diff_data = {
            "additions": [],
            "modifications": [],
            "deletions": [
                {
                    "section": "Usage",
                    "content": "```python\nfrom datastream_curator import DataStreamCurator\n\ncurator = DataStreamCurator()\nresult = await curator.process(input_data, output_path)\n```",
                    "reasoning": "Removing outdated example"
                }
            ],
            "reasoning": "Cleaned up outdated examples"
        }
        
        result = engine.apply_diff(sample_markdown_content, diff_data)
        
        # The code block should be removed
        assert "curator = DataStreamCurator()" not in result
    
    def test_apply_multiple_operations(self, sample_markdown_content):
        """Test applying multiple diff operations."""
        engine = DiffEngine()
        
        diff_data = {
            "additions": [
                {
                    "section": "Features",
                    "content": "### Batch Processing\nProcess multiple files in batch mode.",
                    "reasoning": "Added new feature"
                }
            ],
            "modifications": [
                {
                    "section": "Overview",
                    "old_content": "AI-powered incremental data curation tool.",
                    "new_content": "AI-powered incremental data curation and knowledge base management tool.",
                    "reasoning": "Improved description"
                }
            ],
            "deletions": [
                {
                    "section": "Installation",
                    "content": "```bash\npip install datastream-curator\n```",
                    "reasoning": "Updated installation section"
                }
            ],
            "reasoning": "Multiple improvements"
        }
        
        result = engine.apply_diff(sample_markdown_content, diff_data)
        
        # Check addition
        assert "### Batch Processing" in result
        
        # Check modification
        assert "knowledge base management tool" in result
        
        # Check deletion (bash block should be removed)
        bash_blocks = result.count("```bash")
        original_bash_blocks = sample_markdown_content.count("```bash")
        assert bash_blocks < original_bash_blocks
    
    def test_apply_addition_new_section(self):
        """Test adding content when section doesn't exist."""
        engine = DiffEngine()
        content = "# Document\n\n## Existing Section\nContent here."
        
        diff_data = {
            "additions": [
                {
                    "section": "New Section",
                    "content": "## New Section\nNew content here.",
                    "reasoning": "Added new section"
                }
            ],
            "modifications": [],
            "deletions": [],
            "reasoning": "Added new section"
        }
        
        result = engine.apply_diff(content, diff_data)
        
        # Should append to end when section not found
        assert result.endswith("## New Section\nNew content here.")
    
    def test_apply_addition_to_existing_section(self):
        """Test adding content to existing section."""
        engine = DiffEngine()
        content = "# Document\n\n## Features\nExisting feature.\n\n## Other\nOther content."
        
        diff_data = {
            "additions": [
                {
                    "section": "Features",
                    "content": "### New Feature\nDescription of new feature.",
                    "reasoning": "Added feature"
                }
            ],
            "modifications": [],
            "deletions": [],
            "reasoning": "Enhanced features"
        }
        
        result = engine.apply_diff(content, diff_data)
        
        # Should add after the Features header
        assert "## Features" in result
        assert "### New Feature" in result
        assert result.index("### New Feature") > result.index("## Features")
        assert result.index("### New Feature") < result.index("## Other")
    
    def test_apply_modification_fuzzy_match(self):
        """Test modification with fuzzy matching."""
        engine = DiffEngine()
        content = "# Document\n\nThis is a long paragraph with some specific content that we want to modify."
        
        diff_data = {
            "additions": [],
            "modifications": [
                {
                    "section": "Document",
                    "old_content": "This is a long paragraph",
                    "new_content": "This is an updated paragraph with new information",
                    "reasoning": "Updated content"
                }
            ],
            "deletions": [],
            "reasoning": "Content update"
        }
        
        result = engine.apply_diff(content, diff_data)
        
        assert "This is an updated paragraph with new information" in result
    
    def test_apply_deletion_fuzzy_match(self):
        """Test deletion with partial content matching."""
        engine = DiffEngine()
        content = "# Document\n\nLine 1\nLine 2 with specific content\nLine 3"
        
        diff_data = {
            "additions": [],
            "modifications": [],
            "deletions": [
                {
                    "section": "Document",
                    "content": "Line 2 with specific content",
                    "reasoning": "Removing specific line"
                }
            ],
            "reasoning": "Cleanup"
        }
        
        result = engine.apply_diff(content, diff_data)
        
        assert "Line 2 with specific content" not in result
        assert "Line 1" in result
        assert "Line 3" in result
    
    def test_apply_diff_no_operations(self, sample_markdown_content):
        """Test applying diff with no operations."""
        engine = DiffEngine()
        
        diff_data = {
            "additions": [],
            "modifications": [],
            "deletions": [],
            "reasoning": "No changes needed"
        }
        
        result = engine.apply_diff(sample_markdown_content, diff_data)
        
        # Content should remain unchanged
        assert result == sample_markdown_content
    
    def test_apply_diff_empty_operations(self, sample_markdown_content):
        """Test applying diff with empty operation content."""
        engine = DiffEngine()
        
        diff_data = {
            "additions": [
                {
                    "section": "Test",
                    "content": "",  # Empty content
                    "reasoning": "Empty addition"
                }
            ],
            "modifications": [
                {
                    "section": "Test",
                    "old_content": "",  # Empty old content
                    "new_content": "New content",
                    "reasoning": "Empty modification"
                }
            ],
            "deletions": [
                {
                    "section": "Test",
                    "content": "",  # Empty content
                    "reasoning": "Empty deletion"
                }
            ],
            "reasoning": "Test empty operations"
        }
        
        result = engine.apply_diff(sample_markdown_content, diff_data)
        
        # Content should remain unchanged due to empty operations
        assert result == sample_markdown_content
    
    def test_generate_stats_basic(self):
        """Test generating statistics for diff data."""
        engine = DiffEngine()
        
        diff_data = {
            "additions": [{"content": "add1"}, {"content": "add2"}],
            "modifications": [{"old_content": "old", "new_content": "new"}],
            "deletions": [{"content": "delete"}],
            "reasoning": "Test"
        }
        
        stats = engine.generate_stats(diff_data)
        
        assert stats["additions"] == 2
        assert stats["modifications"] == 1
        assert stats["deletions"] == 1
        assert stats["total_operations"] == 4
    
    def test_generate_stats_empty(self):
        """Test generating statistics for empty diff data."""
        engine = DiffEngine()
        
        diff_data = {
            "additions": [],
            "modifications": [],
            "deletions": [],
            "reasoning": "No changes"
        }
        
        stats = engine.generate_stats(diff_data)
        
        assert stats["additions"] == 0
        assert stats["modifications"] == 0
        assert stats["deletions"] == 0
        assert stats["total_operations"] == 0
    
    def test_generate_stats_missing_keys(self):
        """Test generating statistics with missing keys."""
        engine = DiffEngine()
        
        diff_data = {
            "reasoning": "Partial data"
            # Missing additions, modifications, deletions
        }
        
        stats = engine.generate_stats(diff_data)
        
        assert stats["additions"] == 0
        assert stats["modifications"] == 0
        assert stats["deletions"] == 0
        assert stats["total_operations"] == 0
    
    def test_smart_replace(self):
        """Test smart replace functionality."""
        engine = DiffEngine()
        content = "Line 1\nLine 2 with target content\nLine 3"
        
        result = engine._smart_replace(content, "target content", "replacement content")
        
        assert "replacement content" in result
        assert "target content" not in result
    
    def test_replace_section(self):
        """Test replacing a section of content."""
        engine = DiffEngine()
        content = "Line 1\nTarget line\nLine after target\nAnother line\nFinal line"
        
        result = engine._replace_section(content, "Target line", "New content\nReplacement lines", 2)
        
        assert "New content" in result
        assert "Replacement lines" in result
        assert "Target line" not in result
        assert "Line after target" not in result  # Should be removed as part of section
        assert "Another line" in result  # Should remain


class TestDiffOperation:
    """Test DiffOperation model."""
    
    def test_diff_operation_creation(self):
        """Test creating a diff operation."""
        operation = DiffOperation(
            operation="add",
            section="Features",
            content="New feature content",
            reasoning="Added based on input"
        )
        
        assert operation.operation == "add"
        assert operation.section == "Features"
        assert operation.content == "New feature content"
        assert operation.old_content == ""
        assert operation.reasoning == "Added based on input"
    
    def test_diff_operation_with_old_content(self):
        """Test creating a modification operation."""
        operation = DiffOperation(
            operation="modify",
            section="Usage",
            content="Updated usage instructions",
            old_content="Old usage instructions",
            reasoning="Updated for clarity"
        )
        
        assert operation.operation == "modify"
        assert operation.old_content == "Old usage instructions"


class TestDiffResult:
    """Test DiffResult model."""
    
    def test_diff_result_creation(self):
        """Test creating a diff result."""
        operations = [
            DiffOperation(
                operation="add",
                section="Test",
                content="Test content",
                reasoning="Test"
            )
        ]
        
        stats = {"additions": 1, "modifications": 0, "deletions": 0, "total_operations": 1}
        
        result = DiffResult(
            operations=operations,
            reasoning="Test reasoning",
            stats=stats
        )
        
        assert len(result.operations) == 1
        assert result.operations[0].operation == "add"
        assert result.reasoning == "Test reasoning"
        assert result.stats["total_operations"] == 1