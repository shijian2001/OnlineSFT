from openai import AsyncOpenAI
from typing import Dict, Any, List, Union, Optional
from .json_parser import JSONParser
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from httpx and openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


class QAWrapper:
    """Asynchronous wrapper for LLM API client with automatic retry and error handling."""

    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1", "deepseek-r1"]

    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        base_url: str = "http://redservingapi.devops.xiaohongshu.com/v1",
        max_retries: int = 5,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        parse_json: bool = True,
    ):
        """
        Initialize an async API wrapper instance.

        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            max_retries: Maximum number of retry attempts for failed requests
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response (None for unlimited)
            parse_json: Whether to automatically parse JSON responses (default: True)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parse_json = parse_json

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.stats = {
            "calls": 0,
            "errors": 0,
            "retries": 0,
            "total_retry_time": 0.0
        }

    async def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Simple ask method for backward compatibility.
        
        Args:
            prompt: User prompt
            model: Model name (ignored, uses self.model_name)
            temperature: Temperature override
            max_tokens: Max tokens override
            system: System prompt
            **kwargs: Additional arguments
        
        Returns:
            Generated answer string
        """
        result = await self.qa(
            system=system,
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return result["answer"]

    async def qa(
        self, 
        system: Optional[str] = None,
        user_prompt: Optional[str] = None,
        rational: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a simple query to the model with system and user prompts.

        Args:
            system: System prompt (alias: system_prompt)
            user_prompt: User prompt
            rational: Whether to enable deep reasoning mode
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            system_prompt: Alias for system parameter

        Returns:
            Dict with "answer" and "rational" keys

        Raises:
            ValueError: If reasoning is requested but not supported by the model
            Exception: If all retries are exhausted
            
        Example:
            response = await qa(
                system="You are a helpful assistant.",
                user_prompt="Hello, how are you?"
            )
            print(response["answer"])
        """
        # Handle parameter aliases
        if system_prompt is not None:
            system = system_prompt
        if system is None:
            system = "You are a helpful assistant."
        
        if user_prompt is None:
            raise ValueError("user_prompt must be provided")
        start_time = time.time()
        
        if rational and self.model_name.lower() not in [m.lower() for m in self.SUPPORTED_REASONING_MODELS]:
            raise ValueError(f"Model {self.model_name} does not support reasoning")

        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Execute query
                result = await self._qa(
                    system=system,
                    user_prompt=user_prompt,
                    rational=rational,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Log successful request time
                elapsed = time.time() - start_time
                logger.info(f"✓ API call successful in {elapsed:.2f}s (attempt {attempt + 1})")
                return result

            except Exception as e:
                last_exception = e
                self.stats["errors"] += 1
                
                # Log detailed error information
                error_type = type(e).__name__
                error_msg = str(e)
                elapsed = time.time() - start_time
                
                if attempt < self.max_retries - 1:
                    self.stats["retries"] += 1
                    # Gentle exponential backoff: 0.5, 1.0, 2.0, 3.0, 5.0 seconds
                    retry_delay = min(0.5 * (1.5 ** attempt), 5.0)
                    self.stats["total_retry_time"] += retry_delay
                    
                    logger.warning(
                        f"✗ API call failed after {elapsed:.2f}s (attempt {attempt + 1}/{self.max_retries}): "
                        f"{error_type}: {error_msg} - Retrying in {retry_delay:.1f}s"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"✗ API call failed after {elapsed:.2f}s and {self.max_retries} attempts: "
                        f"{error_type}: {error_msg}"
                    )
        
        # All retries exhausted
        raise Exception(f"API call failed after {self.max_retries} retries. Last error: {str(last_exception)}")

    async def _qa(
        self,
        system: str,
        user_prompt: str,
        rational: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Internal method to execute simple query.
        
        Args:
            system: System prompt
            user_prompt: User prompt
            rational: Whether to enable reasoning
            temperature: Temperature
            max_tokens: Max tokens
            
        Returns:
            Dict with "answer" and "rational" keys
        """
        # Build message list
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ]
        
        # Add reasoning prompt if enabled
        if rational:
            messages.append({"role": "assistant", "content": "<think>\n"})

        # Prepare request parameters
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        
        # Add max_tokens if specified
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        if tokens is not None:
            request_params["max_tokens"] = tokens

        # Call API
        completion = await self.client.chat.completions.create(**request_params)

        self.stats["calls"] += 1
        
        # Extract response
        message = completion.choices[0].message
        answer = message.content
        
        # Parse JSON if enabled
        if self.parse_json and answer:
            answer = JSONParser.parse(answer)
        
        # Extract reasoning if available
        reasoning = getattr(message, "reasoning_content", "") if rational else ""
        
        result = {
            "answer": answer or "",
            "rational": reasoning
        }
        
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this API instance."""
        return self.stats.copy()