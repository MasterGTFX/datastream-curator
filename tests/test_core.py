"""Tests for core DataStreamCurator functionality."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from datastream_curator.core import DataStreamCurator
from datastream_curator.config import CurationConfig, LLMConfig


class TestDataStreamCurator:
    """Test DataStreamCurator class."""
    
    def test_curator_initialization_default_config(self, mock_env_vars):
        """Test curator initialization with default config."""
        curator = DataStreamCurator()
        
        assert curator.config is not None
        assert curator.diff_engine is not None
        assert curator.config.llm.api_key == "test-api-key"
    
    def test_curator_initialization_custom_config(self, test_config):
        """Test curator initialization with custom config."""
        curator = DataStreamCurator(test_config)
        
        assert curator.config == test_config
        assert curator.config.llm.api_key == "test-api-key"
    
    @pytest.mark.asyncio
    async def test_process_input_data_dict(self):
        """Test processing dictionary input data."""
        curator = DataStreamCurator()
        
        input_data = {"key": "value", "number": 42}
        result = await curator._process_input_data(input_data)
        
        # Should be formatted JSON
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42
    
    @pytest.mark.asyncio
    async def test_process_input_data_string(self):
        """Test processing string input data."""
        curator = DataStreamCurator()
        
        input_data = "This is plain text input"
        result = await curator._process_input_data(input_data)
        
        assert result == input_data
    
    @pytest.mark.asyncio
    async def test_process_input_data_file_path(self, sample_input_files):
        """Test processing file path input."""
        curator = DataStreamCurator()
        
        # Test JSON file
        result = await curator._process_input_data(sample_input_files["json"])
        parsed = json.loads(result)
        assert parsed["project"] == "datastream-curator"
        
        # Test text file
        result = await curator._process_input_data(sample_input_files["text"])
        assert "This is sample text data for testing." in result
    
    def test_read_file_json(self, temp_dir, sample_json_data):
        """Test reading JSON file."""
        curator = DataStreamCurator()
        
        json_file = temp_dir / "test.json"
        json_file.write_text(json.dumps(sample_json_data))
        
        result = curator._read_file(json_file)
        
        # Should be pretty-printed JSON
        parsed = json.loads(result)
        assert parsed == sample_json_data
    
    def test_read_file_invalid_json(self, temp_dir):
        """Test reading invalid JSON file."""
        curator = DataStreamCurator()
        
        json_file = temp_dir / "invalid.json"
        json_file.write_text("invalid json content")
        
        result = curator._read_file(json_file)
        
        # Should return original content if JSON parsing fails
        assert result == "invalid json content"
    
    def test_read_file_different_formats(self, temp_dir):
        """Test reading different file formats."""
        curator = DataStreamCurator()
        
        # YAML file
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("key: value")
        result = curator._read_file(yaml_file)
        assert "YAML Content from test.yaml" in result
        assert "key: value" in result
        
        # CSV file
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("name,value\ntest,123")
        result = curator._read_file(csv_file)
        assert "CSV Content from test.csv" in result
        assert "name,value" in result
        
        # XML file
        xml_file = temp_dir / "test.xml"
        xml_file.write_text("<root><item>value</item></root>")
        result = curator._read_file(xml_file)
        assert "XML Content from test.xml" in result
        assert "<root>" in result
        
        # Markdown file
        md_file = temp_dir / "test.md"
        md_file.write_text("# Title\nContent")
        result = curator._read_file(md_file)
        assert "Markdown Content from test.md" in result
        assert "# Title" in result
        
        # Text file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Plain text content")
        result = curator._read_file(txt_file)
        assert "Text Content from test.txt" in result
        assert "Plain text content" in result
        
        # Unknown format
        unknown_file = temp_dir / "test.unknown"
        unknown_file.write_text("Unknown format content")
        result = curator._read_file(unknown_file)
        assert "Content from test.unknown (.unknown format)" in result
        assert "Unknown format content" in result
    
    def test_read_file_encoding_fallback(self, temp_dir):
        """Test reading file with encoding fallback."""
        curator = DataStreamCurator()
        
        # Create file with latin-1 encoding
        file_path = temp_dir / "latin1.txt"
        content = "Café with special characters"
        file_path.write_bytes(content.encode('latin-1'))
        
        result = curator._read_file(file_path)
        assert "Café" in result
    
    @pytest.mark.asyncio
    async def test_process_basic_workflow(self, test_config, mock_llm_response, output_file):
        """Test basic processing workflow."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            # Setup mock LLM client
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = mock_llm_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            input_data = {"new_feature": "Test feature", "version": "1.0"}
            
            result = await curator.process(
                input_data=input_data,
                output_path=output_file,
                instruction="Add new feature information"
            )
            
            # Check that LLM was called
            mock_client.generate_diff.assert_called_once()
            
            # Check that result contains expected content
            assert isinstance(result, str)
            
            # Check that output file was created
            assert Path(output_file).exists()
    
    @pytest.mark.asyncio
    async def test_process_with_existing_kb(self, test_config, mock_llm_response, existing_kb_file, output_file):
        """Test processing with existing knowledge base."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = mock_llm_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            result = await curator.process(
                input_data="New information to add",
                existing_kb_path=existing_kb_file,
                output_path=output_file,
                instruction="Update knowledge base"
            )
            
            # Verify LLM was called with existing content
            call_args = mock_client.generate_diff.call_args[0][0]
            assert call_args.existing_content != ""
            assert "DataStream Curator" in call_args.existing_content
            
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_process_missing_existing_kb(self, test_config, mock_llm_response, output_file):
        """Test processing with non-existent existing KB file."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = mock_llm_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            result = await curator.process(
                input_data="New information",
                existing_kb_path="/nonexistent/path.md",
                output_path=output_file,
                instruction="Create new KB"
            )
            
            # Should handle missing file gracefully
            call_args = mock_client.generate_diff.call_args[0][0]
            assert call_args.existing_content == ""
            
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_process_creates_output_directory(self, test_config, mock_llm_response, temp_dir):
        """Test that processing creates output directory if needed."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = mock_llm_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            # Output path in non-existent directory
            output_path = temp_dir / "new_dir" / "subdir" / "output.md"
            
            result = await curator.process(
                input_data="Test data",
                output_path=str(output_path),
                instruction="Test"
            )
            
            # Directory should be created and file should exist
            assert output_path.exists()
            assert output_path.parent.exists()
    
    @pytest.mark.asyncio
    async def test_process_error_handling(self, test_config):
        """Test error handling during processing."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.side_effect = Exception("LLM API error")
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            with pytest.raises(Exception, match="LLM API error"):
                await curator.process(
                    input_data="Test data",
                    instruction="Test"
                )
    
    @pytest.mark.asyncio
    async def test_process_batch_basic(self, test_config, mock_llm_response, sample_input_files, temp_dir):
        """Test basic batch processing."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = mock_llm_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            input_files = [sample_input_files["json"], sample_input_files["text"]]
            output_path = str(temp_dir / "batch_output.md")
            
            result = await curator.process_batch(
                input_files=input_files,
                output_path=output_path,
                instruction="Process all files"
            )
            
            # Should call LLM for each file
            assert mock_client.generate_diff.call_count == len(input_files)
            
            # Should create output file
            assert Path(output_path).exists()
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_process_batch_with_existing_output(self, test_config, mock_llm_response, sample_input_files, temp_dir):
        """Test batch processing with existing output file."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = mock_llm_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            # Create existing output file
            output_path = temp_dir / "batch_output.md"
            output_path.write_text("# Existing Content\n\nSome existing content.")
            
            input_files = [sample_input_files["json"]]
            
            result = await curator.process_batch(
                input_files=input_files,
                output_path=str(output_path),
                instruction="Update existing content"
            )
            
            # Should use existing content as starting point
            call_args = mock_client.generate_diff.call_args[0][0]
            assert "Existing Content" in call_args.existing_content
    
    @pytest.mark.asyncio
    async def test_process_batch_error_handling(self, test_config, mock_llm_response, temp_dir):
        """Test batch processing error handling."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            # First file succeeds, second fails
            mock_client.generate_diff.side_effect = [mock_llm_response, Exception("Error processing file")]
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            # Create test files
            file1 = temp_dir / "file1.txt"
            file1.write_text("Content 1")
            file2 = temp_dir / "file2.txt"
            file2.write_text("Content 2")
            
            input_files = [str(file1), str(file2)]
            output_path = str(temp_dir / "output.md")
            
            # Should complete despite one file failing
            result = await curator.process_batch(
                input_files=input_files,
                output_path=output_path,
                instruction="Process files"
            )
            
            # Should still produce result (from successful file)
            assert isinstance(result, str)
            assert Path(output_path).exists()
    
    def test_validate_config_valid(self, test_config):
        """Test configuration validation with valid config."""
        curator = DataStreamCurator(test_config)
        
        assert curator.validate_config() is True
    
    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = CurationConfig(
            llm=LLMConfig(
                api_key="",  # Empty API key
                provider="openrouter"
            )
        )
        
        curator = DataStreamCurator(config)
        
        assert curator.validate_config() is False
    
    def test_validate_config_invalid_provider(self):
        """Test configuration validation with invalid provider."""
        config = CurationConfig(
            llm=LLMConfig(
                api_key="test-key",
                provider="invalid-provider"
            )
        )
        
        curator = DataStreamCurator(config)
        
        assert curator.validate_config() is False
    
    def test_validate_config_invalid_temperature(self):
        """Test configuration validation with invalid temperature."""
        config = CurationConfig(
            llm=LLMConfig(
                api_key="test-key",
                provider="openrouter",
                temperature=3.0  # Invalid temperature
            )
        )
        
        curator = DataStreamCurator(config)
        
        assert curator.validate_config() is False
    
    def test_validate_config_invalid_max_tokens(self):
        """Test configuration validation with invalid max_tokens."""
        config = CurationConfig(
            llm=LLMConfig(
                api_key="test-key",
                provider="openrouter",
                max_tokens=0  # Invalid max_tokens
            )
        )
        
        curator = DataStreamCurator(config)
        
        assert curator.validate_config() is False
    
    @pytest.mark.asyncio
    async def test_test_llm_connection_success(self, test_config, mock_llm_response):
        """Test successful LLM connection test."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = mock_llm_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            result = await curator.test_llm_connection()
            
            assert result is True
            mock_client.generate_diff.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_llm_connection_failure(self, test_config):
        """Test LLM connection test failure."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.side_effect = Exception("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            result = await curator.test_llm_connection()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_test_llm_connection_invalid_response(self, test_config):
        """Test LLM connection test with invalid response."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_diff.return_value = None  # Invalid response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            curator = DataStreamCurator(test_config)
            
            result = await curator.test_llm_connection()
            
            assert result is False