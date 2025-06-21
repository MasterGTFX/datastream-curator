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


class DiffStyleOperation(BaseModel):
    """A single diff operation with precise location information."""
    
    operation_type: DiffOperationType = Field(description="Type of diff operation")
    section: Optional[str] = Field(default=None, description="Section or heading where change occurs")
    content: str = Field(description="The content being added, changed, or removed")
    old_content: Optional[str] = Field(default=None, description="Original content for changes/removals")
    line_start: Optional[int] = Field(default=None, description="Starting line number (1-based)")
    line_end: Optional[int] = Field(default=None, description="Ending line number (1-based)")
    char_start: Optional[int] = Field(default=None, description="Starting character position")
    char_end: Optional[int] = Field(default=None, description="Ending character position")
    reasoning: str = Field(description="Explanation for this operation")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this operation")


class ChunkMetadata(BaseModel):
    """Metadata for a content chunk."""
    
    chunk_id: str = Field(description="Unique identifier for this chunk")
    chunk_index: int = Field(description="Index of chunk in document")
    start_char: int = Field(description="Starting character position in original document")
    end_char: int = Field(description="Ending character position in original document")
    section_title: Optional[str] = Field(default=None, description="Section title if applicable")
    token_count: Optional[int] = Field(default=None, description="Number of tokens in chunk")
    embedding: Optional[List[float]] = Field(default=None, description="Embedding vector if available")


class PatchableChunk(BaseModel):
    """A chunk of content that can be patched with diff operations."""
    
    content: str = Field(description="The chunk content")
    metadata: ChunkMetadata = Field(description="Chunk metadata")
    operations: List[DiffStyleOperation] = Field(default_factory=list, description="Operations to apply to this chunk")


class StructuredDiff(BaseModel):
    """Complete diff structure with operations organized by type."""
    
    added: List[DiffStyleOperation] = Field(default_factory=list, description="Content additions")
    changed: List[DiffStyleOperation] = Field(default_factory=list, description="Content modifications")  
    removed: List[DiffStyleOperation] = Field(default_factory=list, description="Content deletions")
    reasoning: str = Field(description="Overall reasoning for all changes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChunkBasedDiff(BaseModel):
    """Diff operations organized by chunks for large documents."""
    
    chunks: List[PatchableChunk] = Field(description="Chunks with their associated operations")
    global_operations: List[DiffStyleOperation] = Field(default_factory=list, description="Operations affecting multiple chunks")
    reasoning: str = Field(description="Overall reasoning for changes")
    chunk_strategy: ChunkStrategy = Field(description="Strategy used for chunking")
    total_chunks: int = Field(description="Total number of chunks")
    original_length: int = Field(description="Original document length in characters")


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
    applied_operations: List[DiffStyleOperation] = Field(description="Operations that were successfully applied")
    skipped_operations: List[DiffStyleOperation] = Field(default_factory=list, description="Operations that were skipped")
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