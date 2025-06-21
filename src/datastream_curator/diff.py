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
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
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
            logger.warning(f\"Chunker for {self.config.chunk_strategy} not available, using recursive\")\n            chunker = self.chunkers[ChunkStrategy.RECURSIVE]\n        \n        try:\n            chunks = chunker.chunk(content)\n            result = []\n            \n            for i, chunk in enumerate(chunks):\n                # Find chunk boundaries in original text\n                start_index = content.find(chunk.text) if hasattr(chunk, 'text') else content.find(str(chunk))\n                if start_index == -1:\n                    start_index = 0\n                \n                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)\n                end_index = start_index + len(chunk_text)\n                \n                result.append(ContentChunk(\n                    content=chunk_text,\n                    start_index=start_index,\n                    end_index=end_index,\n                    metadata={\n                        'chunk_index': i,\n                        'strategy': self.config.chunk_strategy.value,\n                        'length': len(chunk_text)\n                    }\n                ))\n            \n            return result\n            \n        except Exception as e:\n            logger.error(f\"Error creating chunks: {e}\")\n            # Fallback to simple line-based chunking\n            return self._simple_line_chunks(content)\n    \n    def _simple_line_chunks(self, content: str) -> List[ContentChunk]:
        \"\"\"Fallback line-based chunking.\"\"\"\n        lines = content.split('\\n')\n        chunks = []\n        current_chunk = []\n        current_length = 0\n        start_index = 0\n        \n        for line in lines:\n            if current_length + len(line) > self.config.chunk_size and current_chunk:\n                # Finalize current chunk\n                chunk_content = '\\n'.join(current_chunk)\n                chunks.append(ContentChunk(\n                    content=chunk_content,\n                    start_index=start_index,\n                    end_index=start_index + len(chunk_content),\n                    metadata={'chunk_index': len(chunks), 'strategy': 'fallback'}\n                ))\n                \n                start_index += len(chunk_content) + 1  # +1 for newline\n                current_chunk = [line]\n                current_length = len(line)\n            else:\n                current_chunk.append(line)\n                current_length += len(line) + 1  # +1 for newline\n        \n        # Add final chunk\n        if current_chunk:\n            chunk_content = '\\n'.join(current_chunk)\n            chunks.append(ContentChunk(\n                content=chunk_content,\n                start_index=start_index,\n                end_index=start_index + len(chunk_content),\n                metadata={'chunk_index': len(chunks), 'strategy': 'fallback'}\n            ))\n        \n        return chunks\n    \n    def generate_precise_diff(self, old_content: str, new_content: str) -> List[DiffStyleOperation]:\n        \"\"\"Generate precise diff operations using diff-match-patch.\"\"\"\n        diffs = self.dmp.diff_main(old_content, new_content)\n        self.dmp.diff_cleanupSemantic(diffs)  # Clean up for better readability\n        \n        operations = []\n        char_position = 0\n        line_number = 1\n        \n        for operation, text in diffs:\n            if operation == dmp_module.diff_match_patch.DIFF_DELETE:\n                # Content removed\n                start_line = line_number\n                end_line = line_number + text.count('\\n')\n                \n                operations.append(DiffStyleOperation(\n                    operation_type=DiffOperationType.REMOVED,\n                    content=text,\n                    old_content=text,\n                    line_start=start_line,\n                    line_end=end_line,\n                    char_start=char_position,\n                    char_end=char_position + len(text),\n                    reasoning=\"Content removed in diff\",\n                    confidence=0.95\n                ))\n                \n                line_number += text.count('\\n')\n                char_position += len(text)\n                \n            elif operation == dmp_module.diff_match_patch.DIFF_INSERT:\n                # Content added\n                start_line = line_number\n                end_line = line_number + text.count('\\n')\n                \n                operations.append(DiffStyleOperation(\n                    operation_type=DiffOperationType.ADDED,\n                    content=text,\n                    line_start=start_line,\n                    line_end=end_line,\n                    char_start=char_position,\n                    char_end=char_position + len(text),\n                    reasoning=\"Content added in diff\",\n                    confidence=0.95\n                ))\n                \n                line_number += text.count('\\n')\n                char_position += len(text)\n                \n            elif operation == dmp_module.diff_match_patch.DIFF_EQUAL:\n                # Content unchanged\n                line_number += text.count('\\n')\n                char_position += len(text)\n        \n        return operations\n    \n    def apply_structured_diff(self, content: str, structured_diff: StructuredDiff) -> PatchResult:\n        \"\"\"Apply structured diff operations to content.\"\"\"\n        result_content = content\n        applied_operations = []\n        skipped_operations = []\n        errors = []\n        \n        # Sort operations by position (reverse order for deletions)\n        all_operations = structured_diff.added + structured_diff.changed + structured_diff.removed\n        \n        # Apply removals first (in reverse order to maintain positions)\n        removals = sorted(\n            [op for op in all_operations if op.operation_type == DiffOperationType.REMOVED],\n            key=lambda x: x.char_start or 0,\n            reverse=True\n        )\n        \n        for operation in removals:\n            try:\n                result_content = self._apply_removal(result_content, operation)\n                applied_operations.append(operation)\n            except Exception as e:\n                errors.append(f\"Failed to apply removal: {e}\")\n                skipped_operations.append(operation)\n        \n        # Apply changes\n        changes = sorted(\n            [op for op in all_operations if op.operation_type == DiffOperationType.CHANGED],\n            key=lambda x: x.char_start or 0\n        )\n        \n        for operation in changes:\n            try:\n                result_content = self._apply_change(result_content, operation)\n                applied_operations.append(operation)\n            except Exception as e:\n                errors.append(f\"Failed to apply change: {e}\")\n                skipped_operations.append(operation)\n        \n        # Apply additions\n        additions = sorted(\n            [op for op in all_operations if op.operation_type == DiffOperationType.ADDED],\n            key=lambda x: x.char_start or 0\n        )\n        \n        for operation in additions:\n            try:\n                result_content = self._apply_addition(result_content, operation)\n                applied_operations.append(operation)\n            except Exception as e:\n                errors.append(f\"Failed to apply addition: {e}\")\n                skipped_operations.append(operation)\n        \n        stats = {\n            \"applied_count\": len(applied_operations),\n            \"skipped_count\": len(skipped_operations),\n            \"error_count\": len(errors),\n            \"total_operations\": len(all_operations)\n        }\n        \n        return PatchResult(\n            content=result_content,\n            applied_operations=applied_operations,\n            skipped_operations=skipped_operations,\n            errors=errors,\n            stats=stats\n        )\n    \n    def _apply_removal(self, content: str, operation: DiffStyleOperation) -> str:\n        \"\"\"Apply a removal operation.\"\"\"\n        if operation.char_start is not None and operation.char_end is not None:\n            # Use precise character positions\n            before = content[:operation.char_start]\n            after = content[operation.char_end:]\n            return before + after\n        elif operation.old_content:\n            # Use content matching\n            return content.replace(operation.old_content, \"\", 1)\n        else:\n            # Use the content field\n            return content.replace(operation.content, \"\", 1)\n    \n    def _apply_change(self, content: str, operation: DiffStyleOperation) -> str:\n        \"\"\"Apply a change operation.\"\"\"\n        if operation.char_start is not None and operation.char_end is not None:\n            # Use precise character positions\n            before = content[:operation.char_start]\n            after = content[operation.char_end:]\n            return before + operation.content + after\n        elif operation.old_content:\n            # Use content matching\n            return content.replace(operation.old_content, operation.content, 1)\n        else:\n            raise ValueError(\"Change operation missing old_content or position information\")\n    \n    def _apply_addition(self, content: str, operation: DiffStyleOperation) -> str:\n        \"\"\"Apply an addition operation.\"\"\"\n        if operation.char_start is not None:\n            # Insert at specific position\n            before = content[:operation.char_start]\n            after = content[operation.char_start:]\n            return before + operation.content + after\n        elif operation.section:\n            # Add to specific section\n            return self._add_to_section(content, operation.section, operation.content)\n        else:\n            # Append to end\n            return content + \"\\n\\n\" + operation.content\n    \n    def _add_to_section(self, content: str, section: str, new_content: str) -> str:\n        \"\"\"Add content to a specific section.\"\"\"\n        # Find the section heading\n        pattern = rf\"^#{1,6}\\s+{re.escape(section)}\"\n        lines = content.split('\\n')\n        \n        for i, line in enumerate(lines):\n            if re.match(pattern, line, re.IGNORECASE):\n                # Insert after the heading\n                lines.insert(i + 1, \"\")\n                lines.insert(i + 2, new_content)\n                return '\\n'.join(lines)\n        \n        # Section not found, append to end\n        return content + f\"\\n\\n# {section}\\n\\n{new_content}\"\n    \n    def create_chunk_based_diff(self, old_content: str, new_content: str) -> ChunkBasedDiff:\n        \"\"\"Create a chunk-based diff for large documents.\"\"\"\n        # Analyze document structure\n        old_structure = self.analyze_document_structure(old_content)\n        new_structure = self.analyze_document_structure(new_content)\n        \n        chunks = []\n        global_operations = []\n        \n        # Process each chunk\n        for i, old_chunk in enumerate(old_structure.chunks):\n            # Find corresponding chunk in new content\n            if i < len(new_structure.chunks):\n                new_chunk = new_structure.chunks[i]\n                chunk_operations = self.generate_precise_diff(old_chunk.content, new_chunk.content)\n                \n                metadata = ChunkMetadata(\n                    chunk_id=f\"chunk_{i}\",\n                    chunk_index=i,\n                    start_char=old_chunk.start_index,\n                    end_char=old_chunk.end_index,\n                    token_count=len(old_chunk.content.split())\n                )\n                \n                chunks.append(PatchableChunk(\n                    content=old_chunk.content,\n                    metadata=metadata,\n                    operations=chunk_operations\n                ))\n        \n        return ChunkBasedDiff(\n            chunks=chunks,\n            global_operations=global_operations,\n            reasoning=\"Chunk-based diff analysis\",\n            chunk_strategy=self.config.chunk_strategy,\n            total_chunks=len(chunks),\n            original_length=len(old_content)\n        )\n    \n    def merge_operations(self, operations: List[DiffStyleOperation]) -> List[DiffStyleOperation]:\n        \"\"\"Merge adjacent or overlapping operations for efficiency.\"\"\"\n        if not operations:\n            return operations\n        \n        # Sort by position\n        sorted_ops = sorted(operations, key=lambda x: x.char_start or 0)\n        merged = []\n        current = sorted_ops[0]\n        \n        for next_op in sorted_ops[1:]:\n            # Check if operations can be merged\n            if (current.operation_type == next_op.operation_type and\n                current.char_end is not None and next_op.char_start is not None and\n                current.char_end >= next_op.char_start):\n                \n                # Merge operations\n                current = DiffStyleOperation(\n                    operation_type=current.operation_type,\n                    section=current.section,\n                    content=current.content + next_op.content,\n                    old_content=(current.old_content or \"\") + (next_op.old_content or \"\"),\n                    line_start=current.line_start,\n                    line_end=next_op.line_end,\n                    char_start=current.char_start,\n                    char_end=next_op.char_end,\n                    reasoning=f\"{current.reasoning}; {next_op.reasoning}\",\n                    confidence=min(current.confidence, next_op.confidence)\n                )\n            else:\n                merged.append(current)\n                current = next_op\n        \n        merged.append(current)\n        return merged