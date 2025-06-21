"""LLM integration for DataStream Curator."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiohttp
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import LLMConfig
from .models import SimpleDiff, DiffOperation, DiffConfig

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
    """Async client for LLM API interactions with instructor support."""
    
    def __init__(self, config: LLMConfig, retry_attempts: int = 3, retry_delay: float = 1.0):
        self.config = config
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.session: Optional[aiohttp.ClientSession] = None
        self.instructor_client: Optional[instructor.AsyncInstructor] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        
        # Setup instructor client for structured outputs
        if self.config.provider in ["openai", "openrouter"]:
            base_url = "https://api.openai.com/v1" if self.config.provider == "openai" else "https://openrouter.ai/api/v1"
            openai_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=base_url
            )
            self.instructor_client = instructor.from_openai(openai_client,
                                                            mode=instructor.Mode.JSON)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        if self.instructor_client and hasattr(self.instructor_client, 'close'):
            await self.instructor_client.close()
    
    async def generate_diff(self, request: DiffRequest) -> SimpleDiff:
        """Generate structured diff using LLM."""
        if not self.instructor_client:
            raise ValueError("Instructor client required for SimpleDiff generation")
        
        prompt = self._build_structured_diff_prompt(request)
        
        for attempt in range(self.retry_attempts):
            try:
                response = await self.instructor_client.chat.completions.create(
                    model=self.config.model,
                    response_model=SimpleDiff,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a knowledge base curator that generates precise diff operations."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                logger.info(f"Successfully generated SimpleDiff on attempt {attempt + 1}")
                return response
                
            except Exception as e:
                logger.warning(f"SimpleDiff generation attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    logger.error("All SimpleDiff generation attempts failed")
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    

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
    
    
    
    def _build_structured_diff_prompt(self, request: DiffRequest) -> str:
        """Build prompt optimized for SimpleDiff generation."""
        return f"""Analyze the existing knowledge base and new input data to generate precise diff operations.

EXISTING KNOWLEDGE BASE:
<existing_content>
{request.existing_content or "No existing content"}
</existing_content>

NEW INPUT DATA:
<new_data>
{request.new_data}
</new_data>

USER INSTRUCTION:
<instruction>
{request.instruction or "Intelligently integrate new information into the knowledge base"}
</instruction>

CONTEXT:
<context>
{request.context or "No additional context"}
</context>

You must respond with a JSON object containing diff operations. The structure must be:

{{
  "reasoning": "Overall explanation for all changes",
  "operations": [
    {{
      "reasoning": "Why this specific operation is needed",
      "search": "Text to find (or insertion point for additions)",
      "replace": "Content to replace with"
    }}
  ]
}}

OPERATION TYPES & EXAMPLES:

1. TO ADD CONTENT AT END:
{{
  "reasoning": "Add new section",
  "search": "",
  "replace": "## New Section\\nNew content here."
}}

2. TO ADD CONTENT AT SPECIFIC LOCATION:
{{
  "reasoning": "Insert after existing heading",
  "search": "## Existing Section",
  "replace": "## Existing Section\\n\\n## New Subsection\\nNew content here."
}}

3. TO REMOVE CONTENT:
{{
  "reasoning": "Remove outdated information",
  "search": "Old content to remove completely.",
  "replace": ""
}}

4. TO CHANGE EXISTING CONTENT:
{{
  "reasoning": "Update version number",
  "search": "Version 1.0",
  "replace": "Version 2.0"
}}

CRITICAL RULES:
- Each operation must have exactly three fields: "reasoning", "search", "replace"
- For removals: "replace" must be empty string ""
- For additions at end: "search" must be empty string ""
- For insertions: include the search text in the replace text to preserve it
- Use exact string matching - be precise with whitespace and formatting
- Focus on minimal, targeted changes only"""
    
