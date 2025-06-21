"""Command Line Interface for DataStream Curator."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .config import CurationConfig
from .core import DataStreamCurator

console = Console()

# Supported file extensions for automatic discovery
SUPPORTED_EXTENSIONS = {
    '.json', '.csv', '.xml', '.txt', '.md', '.markdown', 
    '.yaml', '.yml', '.log', '.tsv'
}


def setup_logging(verbose: bool = False) -> None:
    """Set up logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def find_input_files(paths: List[str]) -> List[Path]:
    """Find all supported input files from given paths."""
    input_files = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if not path.exists():
            console.print(f"[red]Error: Path '{path}' does not exist[/red]")
            sys.exit(1)
        
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                input_files.append(path)
            else:
                console.print(f"[yellow]Warning: Skipping unsupported file type: {path}[/yellow]")
        
        elif path.is_dir():
            # Recursively find supported files in directory
            found_files = []
            for ext in SUPPORTED_EXTENSIONS:
                found_files.extend(path.rglob(f"*{ext}"))
            
            if found_files:
                input_files.extend(found_files)
                console.print(f"[green]Found {len(found_files)} files in directory: {path}[/green]")
            else:
                console.print(f"[yellow]Warning: No supported files found in directory: {path}[/yellow]")
    
    if not input_files:
        console.print("[red]Error: No valid input files found[/red]")
        sys.exit(1)
    
    # Remove duplicates and sort
    return sorted(list(set(input_files)))


def generate_default_output_path(input_files: List[Path]) -> Path:
    """Generate default output path based on first input file location."""
    if not input_files:
        return Path("knowledge_base.md")
    
    # Use the directory of the first input file
    first_file = input_files[0]
    base_dir = first_file.parent if first_file.is_file() else first_file
    
    # Generate unique filename
    base_name = "knowledge_base"
    extension = ".md"
    counter = 0
    
    while True:
        if counter == 0:
            filename = f"{base_name}{extension}"
        else:
            filename = f"{base_name}_{counter}{extension}"
        
        output_path = base_dir / filename
        if not output_path.exists():
            return output_path
        counter += 1


def validate_configuration() -> Tuple[bool, Optional[str]]:
    """Validate that the configuration is properly set up."""
    try:
        # Try to create configuration from environment
        config = CurationConfig.from_env()
        return True, None
    except ValueError as e:
        return False, str(e)


@click.command()
@click.argument('inputs', nargs=-1, required=True, type=click.Path())
@click.option(
    '-o', '--output', 
    type=click.Path(), 
    help='Output file path (default: knowledge_base.md in first input directory)'
)
@click.option(
    '-i', '--instruct', 
    type=str, 
    help='Custom curation instruction'
)
@click.option(
    '--config', 
    type=click.Path(exists=True), 
    help='Configuration file path'
)
@click.option(
    '--verbose', 
    is_flag=True, 
    help='Enable verbose logging'
)
@click.version_option(version='0.1.0', prog_name='datastream-curator')
def main(
    inputs: Tuple[str, ...], 
    output: Optional[str], 
    instruct: Optional[str], 
    config: Optional[str],
    verbose: bool
) -> None:
    """
    DataStream Curator - AI-powered incremental data curation and knowledge base management.
    
    INPUTS: One or more input files or directories containing data to process.
    
    Examples:
    
      # Process a single file with default output
      datastream-curator data.json
      
      # Process multiple files with custom output and instruction
      datastream-curator file1.json file2.csv -o output.md -i "Create technical documentation"
      
      # Process entire directory
      datastream-curator ./data_directory/ -o knowledge_base.md
    """
    setup_logging(verbose)
    
    # Validate configuration
    config_valid, config_error = validate_configuration()
    if not config_valid:
        console.print(f"[red]Configuration Error: {config_error}[/red]")
        console.print("[yellow]Please set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable[/yellow]")
        sys.exit(1)
    
    # Find input files
    console.print("[blue]Discovering input files...[/blue]")
    input_files = find_input_files(list(inputs))
    
    console.print(f"[green]Found {len(input_files)} input files to process[/green]")
    if verbose:
        for file in input_files:
            console.print(f"  • {file}")
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = generate_default_output_path(input_files)
        console.print(f"[blue]Using default output path: {output_path}[/blue]")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    try:
        if config:
            curator_config = CurationConfig.from_file(config)
            console.print(f"[green]Loaded configuration from: {config}[/green]")
        else:
            curator_config = CurationConfig.from_env()
            console.print("[green]Using environment configuration[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        sys.exit(1)
    
    # Run the curation process
    asyncio.run(process_files(input_files, output_path, instruct, curator_config))


async def process_files(
    input_files: List[Path], 
    output_path: Path, 
    instruction: Optional[str], 
    config: CurationConfig
) -> None:
    """Process the input files using DataStream Curator."""
    curator = DataStreamCurator(config)
    
    # Validate configuration
    if not curator.validate_config():
        console.print("[red]Configuration validation failed[/red]")
        sys.exit(1)
    
    # Test LLM connection
    console.print("[blue]Testing LLM connection...[/blue]")
    if not await curator.test_llm_connection():
        console.print("[red]LLM connection test failed[/red]")
        sys.exit(1)
    console.print("[green]LLM connection successful[/green]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            if len(input_files) == 1:
                # Single file processing
                task = progress.add_task(f"Processing {input_files[0].name}...", total=None)
                
                result = await curator.process(
                    input_data=str(input_files[0]),
                    output_path=str(output_path),
                    existing_kb_path=str(output_path) if output_path.exists() else None,
                    instruction=instruction
                )
                
                progress.update(task, description="✓ Processing complete")
                
            else:
                # Batch processing
                task = progress.add_task(f"Processing {len(input_files)} files...", total=None)
                
                # Convert paths to strings for the batch processor
                file_paths = [str(f) for f in input_files]
                
                result = await curator.process_batch(
                    input_files=file_paths,
                    output_path=str(output_path),
                    instruction=instruction
                )
                
                progress.update(task, description="✓ Batch processing complete")
        
        # Success message
        console.print()
        console.print("[green]✓ Curation completed successfully![/green]")
        console.print(f"[blue]Output saved to: {output_path.absolute()}[/blue]")
        console.print(f"[dim]Generated {len(result)} characters of content[/dim]")
        
        # Show first few lines of output if not too long
        if len(result) < 1000:
            console.print("\n[yellow]Preview:[/yellow]")
            console.print(Text(result[:500] + ("..." if len(result) > 500 else ""), style="dim"))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error during processing: {e}[/red]")
        if config.logging.level.upper() == "DEBUG":
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()