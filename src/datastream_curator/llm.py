"""LLM integration for DataStream Curator."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiohttp
from pydantic import BaseModel

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Response from LLM API."""
    
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: str
    finish_reason: str


class DiffRequest(BaseModel):
    """Request for diff generation."""
    
    existing_content: str
    new_data: str
    instruction: str
    context: Optional[str] = None


class LLMClient:
    """Async client for LLM API interactions."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def generate_diff(self, request: DiffRequest) -> Dict[str, Any]:
        """Generate structured diff using LLM."""
        prompt = self._build_diff_prompt(request)
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self._make_request(prompt)
                diff_data = self._parse_diff_response(response.content)
                logger.info(f"Successfully generated diff on attempt {attempt + 1}")
                return diff_data
            except Exception as e:
                logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
                if attempt == self.config.retry_attempts - 1:
                    logger.error("All LLM request attempts failed")
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    def _build_diff_prompt(self, request: DiffRequest) -> str:
        """Build structured prompt for diff generation."""
        return f"""You are a knowledge base curator. Analyze the existing knowledge base and new input data to generate structured updates.

EXISTING KNOWLEDGE BASE:
{request.existing_content or "No existing content"}

NEW INPUT DATA:
{request.new_data}

USER INSTRUCTION:
{request.instruction or "Intelligently integrate new information into the knowledge base"}

CONTEXT:
{request.context or "No additional context"}

Generate a JSON response with this exact structure:
{{
  "additions": [
    {{
      "section": "section_name",
      "content": "new content to add",
      "reasoning": "why this should be added"
    }}
  ],
  "modifications": [
    {{
      "section": "section_name", 
      "old_content": "exact text to replace",
      "new_content": "replacement text",
      "reasoning": "why this change is needed"
    }}
  ],
  "deletions": [
    {{
      "section": "section_name",
      "content": "exact text to remove",
      "reasoning": "why this should be removed"
    }}
  ],
  "reasoning": "Overall reasoning for all changes"
}}

Focus on:
1. Accuracy and factual correctness
2. Maintaining consistency and coherence  
3. Preserving important historical context
4. Following the user's specific curation goals
5. Only make necessary changes - preserve existing valuable content

Respond with ONLY the JSON structure, no additional text."""

    async def _make_request(self, prompt: str) -> LLMResponse:
        """Make request to LLM API."""
        if self.config.provider == "openrouter":
            return await self._openrouter_request(prompt)
        elif self.config.provider == "openai":
            return await self._openai_request(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    async def _openrouter_request(self, prompt: str) -> LLMResponse:
        """Make request to OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/datastream-curator",
            "X-Title": "DataStream Curator"
        }
        
        data = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        logger.debug(f"Making OpenRouter request to model: {self.config.model}")
        
        async with self.session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                usage=result.get("usage"),
                model=result["model"],
                finish_reason=result["choices"][0]["finish_reason"]
            )
    
    async def _openai_request(self, prompt: str) -> LLMResponse:
        """Make request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        logger.debug(f"Making OpenAI request to model: {self.config.model}")
        
        async with self.session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                usage=result.get("usage"),
                model=result["model"],
                finish_reason=result["choices"][0]["finish_reason"]
            )
    
    def _parse_diff_response(self, content: str) -> Dict[str, Any]:
        """Parse and validate LLM diff response."""
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            content = content.strip()
            
            # Find JSON boundaries
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = content[start:end]
            diff_data = json.loads(json_str)
            
            # Validate and ensure required structure
            required_keys = ["additions", "modifications", "deletions", "reasoning"]
            for key in required_keys:
                if key not in diff_data:
                    if key == "reasoning":
                        diff_data[key] = "No specific reasoning provided"
                    else:
                        diff_data[key] = []
            
            # Validate structure of operations
            self._validate_operations(diff_data)
            
            logger.debug(f"Successfully parsed diff with {len(diff_data['additions'])} additions, "
                        f"{len(diff_data['modifications'])} modifications, "
                        f"{len(diff_data['deletions'])} deletions")
            
            return diff_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response content: {content}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            raise ValueError(f"Failed to parse LLM response: {e}")
    
    def _validate_operations(self, diff_data: Dict[str, Any]) -> None:
        """Validate the structure of diff operations."""
        # Validate additions
        for addition in diff_data.get("additions", []):
            if not isinstance(addition, dict):
                raise ValueError("Addition must be a dictionary")
            if "content" not in addition:
                raise ValueError("Addition must have 'content' field")
        
        # Validate modifications
        for modification in diff_data.get("modifications", []):
            if not isinstance(modification, dict):
                raise ValueError("Modification must be a dictionary")
            if "old_content" not in modification or "new_content" not in modification:
                raise ValueError("Modification must have 'old_content' and 'new_content' fields")
        
        # Validate deletions
        for deletion in diff_data.get("deletions", []):
            if not isinstance(deletion, dict):
                raise ValueError("Deletion must be a dictionary")
            if "content" not in deletion:
                raise ValueError("Deletion must have 'content' field")