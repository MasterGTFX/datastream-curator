"""Diff engine using diff-match-patch and chonkie for precise operations."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

import diff_match_patch as dmp_module
from chonkie import TokenChunker, SentenceChunker, RecursiveChunker, SemanticChunker
from pydantic import BaseModel

from .models import (
    DiffStyleOperation, 
    DiffOperationType, 
    StructuredDiff, 
    ChunkBasedDiff,
    PatchableChunk,
    ChunkMetadata,
    DiffConfig,
    PatchResult,
    ContentChunk,
    DocumentStructure,
    ChunkStrategy
)

logger = logging.getLogger(__name__)


class DiffEngine:
    """Diff engine with precise text operations and chunking support."""
    
    def __init__(self, config: Optional[DiffConfig] = None):
        self.config = config or DiffConfig()
        self.dmp = dmp_module.diff_match_patch()
        self.dmp.Diff_Timeout = 1.0  # 1 second timeout for diff computation
        self._setup_chunkers()
    
    def _setup_chunkers(self):
        """Initialize chunkers based on configuration."""
        self.chunkers = {
            ChunkStrategy.TOKEN: TokenChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            ),
            ChunkStrategy.SENTENCE: SentenceChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            ),
            ChunkStrategy.RECURSIVE: RecursiveChunker(
                chunk_size=self.config.chunk_size
            )
        }
        
        # Add semantic chunker if available
        try:
            self.chunkers[ChunkStrategy.SEMANTIC] = SemanticChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        except Exception as e:
            logger.warning(f"Semantic chunker not available: {e}")
    
    def analyze_document_structure(self, content: str) -> DocumentStructure:
        """Analyze document structure for better diff operations."""
        sections = []
        headings = []
        
        # Find markdown headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            match = re.match(heading_pattern, line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append({
                    'level': level,
                    'title': title,
                    'line_number': i + 1,
                    'start_char': content.find(line),
                    'content': line
                })
        
        # Create sections based on headings
        for i, heading in enumerate(headings):
            start_char = heading['start_char']
            if i + 1 < len(headings):
                end_char = headings[i + 1]['start_char']
            else:
                end_char = len(content)
            
            section_content = content[start_char:end_char]
            sections.append({
                'title': heading['title'],
                'level': heading['level'],
                'start_char': start_char,
                'end_char': end_char,
                'content': section_content,
                'line_start': heading['line_number']
            })
        
        # Create chunks using configured strategy
        chunks = self._create_content_chunks(content)
        
        return DocumentStructure(
            sections=sections,
            headings=headings,
            chunks=chunks,
            metadata={
                'total_chars': len(content),
                'total_lines': len(lines),
                'heading_count': len(headings),
                'chunk_count': len(chunks)
            }
        )
    
    def _create_content_chunks(self, content: str) -> List[ContentChunk]:
        """Create content chunks using the configured strategy."""
        chunker = self.chunkers.get(self.config.chunk_strategy)
        if not chunker:
            logger.warning(f"Chunker for {self.config.chunk_strategy} not available, using recursive")
            chunker = self.chunkers.get(ChunkStrategy.RECURSIVE)
            if not chunker:
                logger.warning("No chunkers available, using line-based fallback")
                return self._create_line_based_chunks(content)
        
        try:
            chunks = chunker.chunk(content)
            result = []
            
            for i, chunk in enumerate(chunks):
                # Find chunk boundaries in original text
                start_index = content.find(chunk.text) if hasattr(chunk, 'text') else content.find(str(chunk))
                if start_index == -1:
                    start_index = 0
                
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                end_index = start_index + len(chunk_text)
                
                result.append(ContentChunk(
                    content=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    metadata={
                        'chunk_index': i,
                        'strategy': self.config.chunk_strategy.value,
                        'length': len(chunk_text)
                    }
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            # Fallback to simple line-based chunking
            return self._simple_line_chunks(content)
    
    def _simple_line_chunks(self, content: str) -> List[ContentChunk]:
        """Fallback line-based chunking."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        start_index = 0
        
        for line in lines:
            if current_length + len(line) > self.config.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_content = '\n'.join(current_chunk)
                chunks.append(ContentChunk(
                    content=chunk_content,
                    start_index=start_index,
                    end_index=start_index + len(chunk_content),
                    metadata={'chunk_index': len(chunks), 'strategy': 'fallback'}
                ))
                
                start_index += len(chunk_content) + 1  # +1 for newline
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line) + 1  # +1 for newline
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(ContentChunk(
                content=chunk_content,
                start_index=start_index,
                end_index=start_index + len(chunk_content),
                metadata={'chunk_index': len(chunks), 'strategy': 'fallback'}
            ))
        
        return chunks
    
    def generate_precise_diff(self, old_content: str, new_content: str) -> List[DiffStyleOperation]:
        """Generate precise diff operations using diff-match-patch."""
        diffs = self.dmp.diff_main(old_content, new_content)
        self.dmp.diff_cleanupSemantic(diffs)  # Clean up for better readability
        
        operations = []
        char_position = 0
        line_number = 1
        
        for operation, text in diffs:
            if operation == dmp_module.diff_match_patch.DIFF_DELETE:
                # Content removed
                start_line = line_number
                end_line = line_number + text.count('\n')
                
                operations.append(DiffStyleOperation(
                    operation_type=DiffOperationType.REMOVED,
                    content=text,
                    old_content=text,
                    line_start=start_line,
                    line_end=end_line,
                    char_start=char_position,
                    char_end=char_position + len(text),
                    reasoning="Content removed in diff",
                    confidence=0.95
                ))
                
                line_number += text.count('\n')
                char_position += len(text)
                
            elif operation == dmp_module.diff_match_patch.DIFF_INSERT:
                # Content added
                start_line = line_number
                end_line = line_number + text.count('\n')
                
                operations.append(DiffStyleOperation(
                    operation_type=DiffOperationType.ADDED,
                    content=text,
                    line_start=start_line,
                    line_end=end_line,
                    char_start=char_position,
                    char_end=char_position + len(text),
                    reasoning="Content added in diff",
                    confidence=0.95
                ))
                
                line_number += text.count('\n')
                char_position += len(text)
                
            elif operation == dmp_module.diff_match_patch.DIFF_EQUAL:
                # Content unchanged
                line_number += text.count('\n')
                char_position += len(text)
        
        return operations
    
    def apply_structured_diff(self, content: str, structured_diff: StructuredDiff) -> PatchResult:
        """Apply structured diff operations to content."""
        result_content = content
        applied_operations = []
        skipped_operations = []
        errors = []
        
        # Sort operations by position (reverse order for deletions)
        all_operations = structured_diff.added + structured_diff.changed + structured_diff.removed
        
        # Apply removals first (in reverse order to maintain positions)
        removals = sorted(
            [op for op in all_operations if op.operation_type == DiffOperationType.REMOVED],
            key=lambda x: x.char_start or 0,
            reverse=True
        )
        
        for operation in removals:
            try:
                result_content = self._apply_removal(result_content, operation)
                applied_operations.append(operation)
            except Exception as e:
                errors.append(f"Failed to apply removal: {e}")
                skipped_operations.append(operation)
        
        # Apply changes
        changes = sorted(
            [op for op in all_operations if op.operation_type == DiffOperationType.CHANGED],
            key=lambda x: x.char_start or 0
        )
        
        for operation in changes:
            try:
                result_content = self._apply_change(result_content, operation)
                applied_operations.append(operation)
            except Exception as e:
                errors.append(f"Failed to apply change: {e}")
                skipped_operations.append(operation)
        
        # Apply additions
        additions = sorted(
            [op for op in all_operations if op.operation_type == DiffOperationType.ADDED],
            key=lambda x: x.char_start or 0
        )
        
        for operation in additions:
            try:
                result_content = self._apply_addition(result_content, operation)
                applied_operations.append(operation)
            except Exception as e:
                errors.append(f"Failed to apply addition: {e}")
                skipped_operations.append(operation)
        
        stats = {
            "applied_count": len(applied_operations),
            "skipped_count": len(skipped_operations),
            "error_count": len(errors),
            "total_operations": len(all_operations)
        }
        
        return PatchResult(
            content=result_content,
            applied_operations=applied_operations,
            skipped_operations=skipped_operations,
            errors=errors,
            stats=stats
        )
    
    def _apply_removal(self, content: str, operation: DiffStyleOperation) -> str:
        """Apply a removal operation."""
        if operation.char_start is not None and operation.char_end is not None:
            # Validate character positions
            if operation.char_start > len(content) or operation.char_end > len(content):
                raise ValueError(f"Character positions out of bounds: {operation.char_start}-{operation.char_end} for content length {len(content)}")
            if operation.char_start < 0 or operation.char_end < 0:
                raise ValueError(f"Character positions cannot be negative: {operation.char_start}-{operation.char_end}")
            if operation.char_start > operation.char_end:
                raise ValueError(f"Start position cannot be greater than end position: {operation.char_start} > {operation.char_end}")
            
            # Use precise character positions
            before = content[:operation.char_start]
            after = content[operation.char_end:]
            return before + after
        elif operation.old_content:
            # Use content matching
            return content.replace(operation.old_content, "", 1)
        else:
            # Use the content field
            return content.replace(operation.content, "", 1)
    
    def _apply_change(self, content: str, operation: DiffStyleOperation) -> str:
        """Apply a change operation."""
        if operation.char_start is not None and operation.char_end is not None:
            # Validate character positions
            if operation.char_start > len(content) or operation.char_end > len(content):
                raise ValueError(f"Character positions out of bounds: {operation.char_start}-{operation.char_end} for content length {len(content)}")
            if operation.char_start < 0 or operation.char_end < 0:
                raise ValueError(f"Character positions cannot be negative: {operation.char_start}-{operation.char_end}")
            if operation.char_start > operation.char_end:
                raise ValueError(f"Start position cannot be greater than end position: {operation.char_start} > {operation.char_end}")
            
            # Use precise character positions
            before = content[:operation.char_start]
            after = content[operation.char_end:]
            return before + operation.content + after
        elif operation.old_content:
            # Use content matching
            return content.replace(operation.old_content, operation.content, 1)
        else:
            raise ValueError("Change operation missing old_content or position information")
    
    def _apply_addition(self, content: str, operation: DiffStyleOperation) -> str:
        """Apply an addition operation."""
        if operation.char_start is not None:
            # Validate character position
            if operation.char_start > len(content):
                raise ValueError(f"Character position out of bounds: {operation.char_start} for content length {len(content)}")
            if operation.char_start < 0:
                raise ValueError(f"Character position cannot be negative: {operation.char_start}")
            
            # Insert at specific position
            before = content[:operation.char_start]
            after = content[operation.char_start:]
            return before + operation.content + after
        elif operation.section:
            # Add to specific section
            return self._add_to_section(content, operation.section, operation.content)
        else:
            # Append to end
            return content + "\n\n" + operation.content
    
    def _create_line_based_chunks(self, content: str) -> List[ContentChunk]:
        """Create line-based chunks as fallback when chunkers fail."""
        lines = content.split('\n')
        chunks = []
        current_pos = 0
        
        # Group lines into chunks based on configured chunk size
        lines_per_chunk = max(1, self.config.chunk_size // 50)  # Estimate lines per chunk
        
        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_content = '\n'.join(chunk_lines)
            
            chunks.append(ContentChunk(
                content=chunk_content,
                start_index=current_pos,
                end_index=current_pos + len(chunk_content),
                metadata={
                    'chunk_index': len(chunks),
                    'strategy': 'fallback',
                    'length': len(chunk_content),
                    'line_count': len(chunk_lines)
                }
            ))
            
            current_pos += len(chunk_content) + 1  # +1 for newline
        
        return chunks
    
    def _add_to_section(self, content: str, section: str, new_content: str) -> str:
        """Add content to a specific section."""
        # Find the section heading
        pattern = r"^#{1,6}\s+" + re.escape(section) + r"(?:\s|$)"
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if re.match(pattern, line, re.IGNORECASE):
                # Insert after the heading
                lines.insert(i + 1, "")
                lines.insert(i + 2, new_content)
                return '\n'.join(lines)
        
        # Section not found, append to end
        return content + f"\n\n# {section}\n\n{new_content}"
    
    def create_chunk_based_diff(self, old_content: str, new_content: str) -> ChunkBasedDiff:
        """Create a chunk-based diff for large documents."""
        # Analyze document structure
        old_structure = self.analyze_document_structure(old_content)
        new_structure = self.analyze_document_structure(new_content)
        
        chunks = []
        global_operations = []
        
        # Process each chunk
        for i, old_chunk in enumerate(old_structure.chunks):
            # Find corresponding chunk in new content
            if i < len(new_structure.chunks):
                new_chunk = new_structure.chunks[i]
                chunk_operations = self.generate_precise_diff(old_chunk.content, new_chunk.content)
                
                metadata = ChunkMetadata(
                    chunk_id=f"chunk_{i}",
                    chunk_index=i,
                    start_char=old_chunk.start_index,
                    end_char=old_chunk.end_index,
                    token_count=len(old_chunk.content.split())
                )
                
                chunks.append(PatchableChunk(
                    content=old_chunk.content,
                    metadata=metadata,
                    operations=chunk_operations
                ))
        
        return ChunkBasedDiff(
            chunks=chunks,
            global_operations=global_operations,
            reasoning="Chunk-based diff analysis",
            chunk_strategy=self.config.chunk_strategy,
            total_chunks=len(chunks),
            original_length=len(old_content)
        )
    
    def merge_operations(self, operations: List[DiffStyleOperation]) -> List[DiffStyleOperation]:
        """Merge adjacent or overlapping operations for efficiency."""
        if not operations:
            return operations
        
        # Sort by position
        sorted_ops = sorted(operations, key=lambda x: x.char_start or 0)
        merged = []
        current = sorted_ops[0]
        
        for next_op in sorted_ops[1:]:
            # Check if operations can be merged
            if (current.operation_type == next_op.operation_type and
                current.char_end is not None and next_op.char_start is not None and
                current.char_end >= next_op.char_start):
                
                # Merge operations
                current = DiffStyleOperation(
                    operation_type=current.operation_type,
                    section=current.section,
                    content=current.content + next_op.content,
                    old_content=(current.old_content or "") + (next_op.old_content or ""),
                    line_start=current.line_start,
                    line_end=next_op.line_end,
                    char_start=current.char_start,
                    char_end=next_op.char_end,
                    reasoning=f"{current.reasoning}; {next_op.reasoning}",
                    confidence=min(current.confidence, next_op.confidence)
                )
            else:
                merged.append(current)
                current = next_op
        
        merged.append(current)
        return merged