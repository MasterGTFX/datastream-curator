"""Diff engine using diff-match-patch and chonkie for precise operations."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

import diff_match_patch as dmp_module
from chonkie import TokenChunker, SentenceChunker, RecursiveChunker, SemanticChunker
from pydantic import BaseModel

from .models import (
    SimpleDiff,
    DiffOperation,
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
    
    def apply_simple_diff(self, content: str, simple_diff: SimpleDiff) -> PatchResult:
        """Apply SimpleDiff operations to content."""
        result_content = content
        applied_changes = []
        skipped_changes = []
        errors = []
        
        # Apply all operations in order
        for operation in simple_diff.operations:
            try:
                if operation.replace == "":
                    # This is a removal operation
                    if operation.search in result_content:
                        result_content = result_content.replace(operation.search, "", 1)
                        applied_changes.append(f"Removed: {operation.reasoning}")
                    else:
                        skipped_changes.append(f"Could not find content to remove: {operation.search[:50]}...")
                
                elif operation.search == "":
                    # This is an addition to end of document
                    result_content += "\n\n" + operation.replace
                    applied_changes.append(f"Added: {operation.reasoning}")
                
                elif operation.search in result_content:
                    # This is either a change or insertion at a specific point
                    if operation.search == operation.replace:
                        # Skip no-op operations
                        skipped_changes.append(f"No change needed: {operation.reasoning}")
                    else:
                        # Check if this looks like an insertion (search text appears in replace)
                        if operation.search in operation.replace:
                            # This is an insertion - replace search with search + new content
                            result_content = result_content.replace(operation.search, operation.replace, 1)
                            applied_changes.append(f"Added: {operation.reasoning}")
                        else:
                            # This is a change - replace search with replace
                            result_content = result_content.replace(operation.search, operation.replace, 1)
                            applied_changes.append(f"Changed: {operation.reasoning}")
                else:
                    skipped_changes.append(f"Could not find target content: {operation.search[:50]}...")
                    
            except Exception as e:
                errors.append(f"Failed to apply operation: {e}")
                skipped_changes.append(f"Operation failed: {operation.reasoning}")
        
        stats = {
            "applied_count": len(applied_changes),
            "skipped_count": len(skipped_changes),
            "error_count": len(errors),
            "total_operations": len(simple_diff.operations)
        }
        
        return PatchResult(
            content=result_content,
            applied_changes=applied_changes,
            skipped_changes=skipped_changes,
            errors=errors,
            stats=stats
        )
    
    
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
    
    
