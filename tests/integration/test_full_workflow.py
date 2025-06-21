"""Integration tests for full DataStream Curator workflow."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from datastream_curator import DataStreamCurator, CurationConfig, create_curator


class TestFullWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_new_kb(self, test_config, mock_llm_response, temp_dir, sample_json_data):
        """Test complete workflow creating new knowledge base."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            # Setup mock LLM client
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_client.generate_diff.return_value = {
                "additions": [
                    {
                        "section": "Project Information",
                        "content": f"# {sample_json_data['project']}\n\nVersion: {sample_json_data['version']}\n\nFeatures:\n" + 
                                  "\n".join([f"- {f['name']}: {f['description']}" for f in sample_json_data['features']]),
                        "reasoning": "Creating initial project documentation from JSON data"
                    }
                ],
                "modifications": [],
                "deletions": [],
                "reasoning": "Initial knowledge base creation"
            }
            
            # Create input file
            input_file = temp_dir / "input.json"
            input_file.write_text(json.dumps(sample_json_data, indent=2))
            
            output_file = temp_dir / "output.md"
            
            curator = DataStreamCurator(test_config)
            
            result = await curator.process(
                input_data=str(input_file),
                output_path=str(output_file),
                instruction="Create comprehensive project documentation"
            )
            
            # Verify output file was created
            assert output_file.exists()
            
            # Verify content
            assert sample_json_data['project'] in result
            assert sample_json_data['version'] in result
            assert "LLM Integration" in result
            
            # Verify file content matches result
            file_content = output_file.read_text()
            assert file_content == result
    
    @pytest.mark.asyncio
    async def test_complete_workflow_update_existing(self, test_config, existing_kb_file, temp_dir, sample_json_data):
        """Test complete workflow updating existing knowledge base."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_client.generate_diff.return_value = {
                "additions": [
                    {
                        "section": "Features",
                        "content": "### Batch Processing\nProcess multiple files in sequence for efficient bulk operations.",
                        "reasoning": "Added new feature from input data"
                    }
                ],
                "modifications": [
                    {
                        "section": "Installation",
                        "old_content": "pip install datastream-curator",
                        "new_content": "pip install datastream-curator>=0.1.0",
                        "reasoning": "Updated version requirement"
                    }
                ],
                "deletions": [],
                "reasoning": "Enhanced existing documentation with new features"
            }
            
            # Create new input data
            update_data = {
                "new_features": ["Batch Processing"],
                "version_update": "0.1.0",
                "improvements": ["Better error handling", "Enhanced logging"]
            }
            
            input_file = temp_dir / "update.json"
            input_file.write_text(json.dumps(update_data, indent=2))
            
            output_file = temp_dir / "updated_kb.md"
            
            curator = DataStreamCurator(test_config)
            
            result = await curator.process(
                input_data=str(input_file),
                existing_kb_path=existing_kb_file,
                output_path=str(output_file),
                instruction="Update knowledge base with new features and version information"
            )
            
            # Verify output
            assert output_file.exists()
            assert "DataStream Curator" in result  # Original content preserved
            assert "Batch Processing" in result     # New content added
            assert ">=0.1.0" in result             # Modified content
    
    @pytest.mark.asyncio
    async def test_incremental_updates_workflow(self, test_config, temp_dir):
        """Test multiple incremental updates to same knowledge base."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            
            # Mock responses for each update
            responses = [
                # First update - create initial content
                {
                    "additions": [
                        {
                            "section": "Project",
                            "content": "# DataStream Curator\n\nInitial project documentation.",
                            "reasoning": "Initial creation"
                        }
                    ],
                    "modifications": [],
                    "deletions": [],
                    "reasoning": "Initial creation"
                },
                # Second update - add features
                {
                    "additions": [
                        {
                            "section": "Features",
                            "content": "## Features\n\n### LLM Integration\nConnect to various LLM providers.",
                            "reasoning": "Added features section"
                        }
                    ],
                    "modifications": [],
                    "deletions": [],
                    "reasoning": "Added features"
                },
                # Third update - update existing content
                {
                    "additions": [],
                    "modifications": [
                        {
                            "section": "Project",
                            "old_content": "Initial project documentation.",
                            "new_content": "Comprehensive AI-powered data curation tool.",
                            "reasoning": "Enhanced description"
                        }
                    ],
                    "deletions": [],
                    "reasoning": "Enhanced description"
                }
            ]
            
            mock_client.generate_diff.side_effect = responses
            
            curator = DataStreamCurator(test_config)
            kb_path = temp_dir / "incremental_kb.md"
            
            # First update
            result1 = await curator.process(
                input_data="Initial project information",
                output_path=str(kb_path),
                instruction="Create initial documentation"
            )
            
            assert "DataStream Curator" in result1
            assert "Initial project documentation" in result1
            
            # Second update
            result2 = await curator.process(
                input_data="Feature: LLM Integration for AI processing",
                existing_kb_path=str(kb_path),
                output_path=str(kb_path),
                instruction="Add features section"
            )
            
            assert "DataStream Curator" in result2
            assert "LLM Integration" in result2
            
            # Third update
            result3 = await curator.process(
                input_data="Enhanced description of capabilities",
                existing_kb_path=str(kb_path),
                output_path=str(kb_path),
                instruction="Enhance project description"
            )
            
            assert "DataStream Curator" in result3
            assert "Comprehensive AI-powered" in result3
            assert "Initial project documentation" not in result3
            
            # Verify all updates were applied
            final_content = kb_path.read_text()
            assert final_content == result3
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, test_config, temp_dir):
        """Test batch processing workflow."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            
            # Mock response for each file in batch
            mock_client.generate_diff.return_value = {
                "additions": [
                    {
                        "section": "Content",
                        "content": "New content from batch processing",
                        "reasoning": "Added from batch file"
                    }
                ],
                "modifications": [],
                "deletions": [],
                "reasoning": "Batch processing update"
            }
            
            # Create multiple input files
            input_files = []
            for i in range(3):
                input_file = temp_dir / f"batch_input_{i}.json"
                data = {
                    "file_id": i,
                    "content": f"Content from file {i}",
                    "metadata": {"source": f"batch_{i}"}
                }
                input_file.write_text(json.dumps(data, indent=2))
                input_files.append(str(input_file))
            
            output_path = temp_dir / "batch_output.md"
            
            curator = DataStreamCurator(test_config)
            
            result = await curator.process_batch(
                input_files=input_files,
                output_path=str(output_path),
                instruction="Process all files into unified documentation"
            )
            
            # Verify batch processing
            assert output_path.exists()
            assert isinstance(result, str)
            
            # Should have called LLM for each file
            assert mock_client.generate_diff.call_count == len(input_files)
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, test_config, temp_dir):
        """Test workflow with error recovery."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            
            # First call fails, second succeeds
            mock_client.generate_diff.side_effect = [
                Exception("Temporary API error"),
                {
                    "additions": [
                        {
                            "section": "Content",
                            "content": "Successfully processed after retry",
                            "reasoning": "Recovered from error"
                        }
                    ],
                    "modifications": [],
                    "deletions": [],
                    "reasoning": "Success after retry"
                }
            ]
            
            # Set low retry attempts for testing
            test_config.llm.retry_attempts = 2
            test_config.llm.retry_delay = 0.1
            
            curator = DataStreamCurator(test_config)
            
            input_file = temp_dir / "input.txt"
            input_file.write_text("Test input data")
            
            output_file = temp_dir / "output.md"
            
            result = await curator.process(
                input_data=str(input_file),
                output_path=str(output_file),
                instruction="Test error recovery"
            )
            
            # Should succeed after retry
            assert "Successfully processed after retry" in result
            assert output_file.exists()
    
    def test_create_curator_helper_function(self, config_file, mock_env_vars):
        """Test the create_curator helper function."""
        # Test with config file
        curator1 = create_curator(config_path=config_file)
        assert isinstance(curator1, DataStreamCurator)
        assert curator1.config.llm.api_key == "test-api-key"
        
        # Test with environment variables
        curator2 = create_curator()
        assert isinstance(curator2, DataStreamCurator)
        assert curator2.config.llm.api_key == "test-api-key"
        
        # Test with kwargs override
        curator3 = create_curator(config_path=config_file, **{"llm.temperature": 0.5})
        assert curator3.config.llm.temperature == 0.5
    
    @pytest.mark.asyncio
    async def test_configuration_validation_workflow(self, temp_dir):
        """Test workflow with configuration validation."""
        # Test invalid configuration
        invalid_config = CurationConfig.from_env()
        invalid_config.llm.api_key = ""  # Invalid
        
        curator = DataStreamCurator(invalid_config)
        
        # Validation should fail
        assert curator.validate_config() is False
        
        # Processing should fail due to missing API key
        with pytest.raises(Exception):
            await curator.process(
                input_data="test data",
                instruction="test"
            )
    
    @pytest.mark.asyncio
    async def test_llm_connection_test_workflow(self, test_config):
        """Test LLM connection testing workflow."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_client.generate_diff.return_value = {
                "additions": [],
                "modifications": [],
                "deletions": [],
                "reasoning": "Connection test successful"
            }
            
            curator = DataStreamCurator(test_config)
            
            # Test connection should succeed
            connection_result = await curator.test_llm_connection()
            assert connection_result is True
            
            # After successful connection test, processing should work
            result = await curator.process(
                input_data="test data",
                instruction="test processing"
            )
            
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_different_file_formats_workflow(self, test_config, temp_dir):
        """Test workflow with different input file formats."""
        with patch('datastream_curator.core.LLMClient') as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_client.generate_diff.return_value = {
                "additions": [
                    {
                        "section": "Data",
                        "content": "Processed file content",
                        "reasoning": "Processed input file"
                    }
                ],
                "modifications": [],
                "deletions": [],
                "reasoning": "File format processing"
            }
            
            curator = DataStreamCurator(test_config)
            
            # Test different file formats
            formats = {
                "json": {"key": "value", "number": 123},
                "yaml": "key: value\nnumber: 123",
                "csv": "name,value\ntest,123",
                "txt": "Plain text content",
                "xml": "<root><item>value</item></root>"
            }
            
            for file_format, content in formats.items():
                input_file = temp_dir / f"input.{file_format}"
                
                if file_format == "json":
                    input_file.write_text(json.dumps(content))
                else:
                    input_file.write_text(content)
                
                output_file = temp_dir / f"output_{file_format}.md"
                
                result = await curator.process(
                    input_data=str(input_file),
                    output_path=str(output_file),
                    instruction=f"Process {file_format} file"
                )
                
                assert isinstance(result, str)
                assert output_file.exists()
                
                # Verify LLM received formatted content
                call_args = mock_client.generate_diff.call_args[0][0]
                if file_format == "json":
                    assert "key" in call_args.new_data
                elif file_format == "yaml":
                    assert "YAML Content" in call_args.new_data
                elif file_format == "csv":
                    assert "CSV Content" in call_args.new_data
                elif file_format == "xml":
                    assert "XML Content" in call_args.new_data
                elif file_format == "txt":
                    assert "Text Content" in call_args.new_data