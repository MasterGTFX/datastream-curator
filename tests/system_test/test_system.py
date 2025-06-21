import json

from src.datastream_curator import DataStreamCurator


async def test_complete_workflow(test_config, mock_llm_response, temp_dir, existing_kb_file, sample_json_data):
    """Test complete workflow creating new knowledge base."""
    # Create input file
    input_file = temp_dir / "input.json"
    input_file.write_text(json.dumps(sample_json_data, indent=2))

    output_file = temp_dir / "output.md"

    curator = DataStreamCurator(test_config)

    result = await curator.process(
        input_data=str(input_file),
        output_path=str(output_file),
        existing_kb_path=existing_kb_file,
        instruction="Create comprehensive project documentation"
    )

    # Print the original vs updated
    print(f"Updated KB:\n{result}")

    # Verify output file was created
    assert output_file.exists()

    # Verify content
    assert sample_json_data['project'] in result
    assert sample_json_data['version'] in result
    assert "LLM Integration" in result


    # Verify file content matches result
    file_content = output_file.read_text()
    assert file_content == result
