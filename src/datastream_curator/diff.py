"""Diff generation and application engine for DataStream Curator."""

import re
import logging
from typing import Any, Dict, List

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DiffOperation(BaseModel):
    """Represents a single diff operation."""
    
    operation: str  # "add", "modify", "delete"
    section: str
    content: str
    old_content: str = ""
    reasoning: str = ""


class DiffResult(BaseModel):
    """Result of applying a diff."""
    
    operations: List[DiffOperation]
    reasoning: str
    stats: Dict[str, int]


class DiffEngine:
    """Handles application of structured diffs to markdown content."""
    
    def apply_diff(self, content: str, diff_data: Dict[str, Any]) -> str:
        """Apply structured diff to existing content."""
        logger.info("Applying diff to content")
        result = content
        operations = []
        
        # Process deletions first to avoid content shifting issues
        for deletion in diff_data.get("deletions", []):
            old_result = result
            result = self._apply_deletion(result, deletion)
            if result != old_result:
                operations.append(DiffOperation(
                    operation="delete",
                    section=deletion.get("section", ""),
                    content=deletion.get("content", ""),
                    reasoning=deletion.get("reasoning", "")
                ))
                logger.debug(f"Applied deletion in section: {deletion.get('section', 'unknown')}")
        
        # Process modifications next
        for modification in diff_data.get("modifications", []):
            old_result = result
            result = self._apply_modification(result, modification)
            if result != old_result:
                operations.append(DiffOperation(
                    operation="modify",
                    section=modification.get("section", ""),
                    content=modification.get("new_content", ""),
                    old_content=modification.get("old_content", ""),
                    reasoning=modification.get("reasoning", "")
                ))
                logger.debug(f"Applied modification in section: {modification.get('section', 'unknown')}")
        
        # Process additions last
        for addition in diff_data.get("additions", []):
            old_result = result
            result = self._apply_addition(result, addition)
            if result != old_result:
                operations.append(DiffOperation(
                    operation="add",
                    section=addition.get("section", ""),
                    content=addition.get("content", ""),
                    reasoning=addition.get("reasoning", "")
                ))
                logger.debug(f"Applied addition in section: {addition.get('section', 'unknown')}")
        
        logger.info(f"Applied {len(operations)} diff operations")
        return result
    
    def _apply_deletion(self, content: str, deletion: Dict[str, Any]) -> str:
        """Remove specified content."""
        target = deletion.get("content", "").strip()
        if not target:
            logger.warning("Empty deletion target, skipping")
            return content
        
        # Try exact match first
        if target in content:
            logger.debug("Found exact match for deletion")
            return content.replace(target, "", 1)
        
        # Try fuzzy matching for partial content
        lines = target.split('\n')
        if len(lines) > 1:
            # Multi-line deletion - try to match key phrases
            for line in lines:
                line = line.strip()
                if line and line in content:
                    logger.debug(f"Found partial match for deletion: {line[:50]}...")
                    content = content.replace(line, "", 1)
        
        # Try word-based matching for single lines
        elif len(lines) == 1:
            words = target.split()
            if len(words) >= 3:
                # Try matching with first few words
                partial = " ".join(words[:3])
                if partial in content:
                    logger.debug(f"Found word-based match for deletion: {partial}")
                    # Find the line containing this partial match and remove it
                    content_lines = content.split('\n')
                    for i, line in enumerate(content_lines):
                        if partial in line:
                            content_lines.pop(i)
                            return '\n'.join(content_lines)
        
        logger.warning(f"Could not find content to delete: {target[:100]}...")
        return content
    
    def _apply_modification(self, content: str, modification: Dict[str, Any]) -> str:
        """Replace old content with new content."""
        old_content = modification.get("old_content", "").strip()
        new_content = modification.get("new_content", "").strip()
        
        if not old_content or not new_content:
            logger.warning("Empty old_content or new_content in modification, skipping")
            return content
        
        # Try exact replacement first
        if old_content in content:
            logger.debug("Found exact match for modification")
            return content.replace(old_content, new_content, 1)
        
        # Try fuzzy matching - find similar content
        old_words = old_content.split()
        if len(old_words) >= 3:
            # Try matching with first few words
            partial = " ".join(old_words[:3])
            if partial in content:
                logger.debug(f"Found partial match for modification: {partial}")
                return self._smart_replace(content, partial, new_content)
        
        # Try line-based matching
        old_lines = old_content.split('\n')
        if len(old_lines) > 1:
            # Multi-line modification - try to find the first line and replace the section
            first_line = old_lines[0].strip()
            if first_line and first_line in content:
                logger.debug(f"Found first line match for modification: {first_line[:50]}...")
                return self._replace_section(content, first_line, new_content, len(old_lines))
        
        logger.warning(f"Could not find content to modify: {old_content[:100]}...")
        return content
    
    def _apply_addition(self, content: str, addition: Dict[str, Any]) -> str:
        """Add new content to appropriate section."""
        section = addition.get("section", "")
        new_content = addition.get("content", "").strip()
        
        if not new_content:
            logger.warning("Empty content in addition, skipping")
            return content
        
        # If section specified, try to add to that section
        if section:
            section_pattern = rf"^#{1,6}\s+{re.escape(section)}"
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if re.match(section_pattern, line, re.IGNORECASE):
                    logger.debug(f"Found section '{section}' at line {i + 1}")
                    # Found section - add content after it
                    lines.insert(i + 1, "")
                    lines.insert(i + 2, new_content)
                    return '\n'.join(lines)
            
            # If section not found, try to find a similar section
            section_lower = section.lower()
            for i, line in enumerate(lines):
                if re.match(r'^#{1,6}\s+', line) and section_lower in line.lower():
                    logger.debug(f"Found similar section at line {i + 1}: {line}")
                    lines.insert(i + 1, "")
                    lines.insert(i + 2, new_content)
                    return '\n'.join(lines)
        
        # If no specific section or section not found, append to end
        if content.strip():
            logger.debug("Appending content to end of document")
            return content + "\n\n" + new_content
        else:
            logger.debug("Setting as initial content")
            return new_content
    
    def _smart_replace(self, content: str, search_phrase: str, replacement: str) -> str:
        """Intelligently replace content using contextual matching."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if search_phrase in line:
                # Replace the entire line with the replacement
                lines[i] = replacement
                logger.debug(f"Smart replaced line {i + 1}")
                return '\n'.join(lines)
        
        return content
    
    def _replace_section(self, content: str, start_marker: str, replacement: str, num_lines: int) -> str:
        """Replace a section of content starting from a marker."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if start_marker in line:
                # Replace the section with the new content
                # Remove the old lines
                for _ in range(min(num_lines, len(lines) - i)):
                    if i < len(lines):
                        lines.pop(i)
                
                # Insert the new content
                replacement_lines = replacement.split('\n')
                for j, repl_line in enumerate(replacement_lines):
                    lines.insert(i + j, repl_line)
                
                logger.debug(f"Replaced section starting at line {i + 1}")
                return '\n'.join(lines)
        
        return content
    
    def generate_stats(self, diff_data: Dict[str, Any]) -> Dict[str, int]:
        """Generate statistics about the diff."""
        additions = len(diff_data.get("additions", []))
        modifications = len(diff_data.get("modifications", []))
        deletions = len(diff_data.get("deletions", []))
        
        stats = {
            "additions": additions,
            "modifications": modifications,
            "deletions": deletions,
            "total_operations": additions + modifications + deletions
        }
        
        logger.info(f"Diff stats: {stats}")
        return stats
    
    def create_diff_result(self, operations: List[DiffOperation], reasoning: str, stats: Dict[str, int]) -> DiffResult:
        """Create a DiffResult object."""
        return DiffResult(
            operations=operations,
            reasoning=reasoning,
            stats=stats
        )