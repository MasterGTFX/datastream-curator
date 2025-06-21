"""Enhanced Pydantic models for diff operations with instructor support."""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class DiffOperationType(str, Enum):
    """Types of diff operations."""
    ADDED = "added"
    CHANGED = "changed" 
    REMOVED = "removed"


class ChunkStrategy(str, Enum):
    """Chunking strategies for large documents."""
    TOKEN = "token"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SDPM = "sdpm"
    LATE = "late"
    CODE = "code"
    NEURAL = "neural"
    SLUMBER = "slumber"




class DiffOperation(BaseModel):
    """A single diff operation with search/replace format."""
    reasoning: str = Field(description="Reason for this specific operation")
    search: str = Field(description="Exact content to find (or insertion point for additions)")
    replace: str = Field(description="Content to replace with (empty string for removals)")


class SimpleDiff(BaseModel):
    """Simplified diff format using a single list of operations."""
    reasoning: str = Field(description="Overall reasoning for all changes")
    operations: List[DiffOperation] = Field(default_factory=list, description="List of diff operations to apply")



class DiffConfig(BaseModel):
    """Configuration for diff generation and application."""
    
    chunk_strategy: ChunkStrategy = Field(default=ChunkStrategy.RECURSIVE, description="Chunking strategy")
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Target chunk size")
    chunk_overlap: int = Field(default=100, ge=0, le=500, description="Overlap between chunks")
    max_chunks: int = Field(default=50, ge=1, le=200, description="Maximum number of chunks")
    use_semantic_chunking: bool = Field(default=True, description="Use semantic awareness for chunking")
    preserve_structure: bool = Field(default=True, description="Preserve document structure")
    min_operation_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence for operations")


class PatchResult(BaseModel):
    """Result of applying patches to content."""
    
    content: str = Field(description="The patched content")
    applied_changes: List[str] = Field(description="Changes that were successfully applied")
    skipped_changes: List[str] = Field(default_factory=list, description="Changes that were skipped")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    stats: Dict[str, int] = Field(description="Statistics about the patch operation")


class ContentChunk(BaseModel):
    """A chunk of content with its boundaries."""
    
    content: str = Field(description="The chunk content")
    start_index: int = Field(description="Starting character index")
    end_index: int = Field(description="Ending character index")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentStructure(BaseModel):
    """Structure analysis of a document."""
    
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="Document sections")
    headings: List[Dict[str, Any]] = Field(default_factory=list, description="Document headings")
    chunks: List[ContentChunk] = Field(default_factory=list, description="Content chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")