"""Utility functions for API module."""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_api_keys(config_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load API keys from config file or environment.
    
    Args:
        config_path: Path to JSON config file with format:
            [{"api_key": "sk-...", "base_url": "https://..."}]
            or just ["sk-...", "sk-..."] (list of key strings)
    
    Returns:
        List of API key configurations in format [{"api_key": "...", "base_url": "..."}]
    """
    # Try config file first
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            data = json.load(f)
            # Handle both formats: list of dicts or list of strings
            if data and isinstance(data[0], str):
                # List of key strings
                return [{"api_key": key, "base_url": "https://api.openai.com/v1"} for key in data]
            else:
                # List of config dicts
                return data
    
    # Try default locations
    default_paths = [
        Path.home() / ".config" / "openai" / "api_keys.json",
        Path("api_keys.json"),
    ]
    
    for path in default_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                if data and isinstance(data[0], str):
                    return [{"api_key": key, "base_url": "https://api.openai.com/v1"} for key in data]
                else:
                    return data
    
    # Fallback to environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if api_key:
        return [{"api_key": api_key, "base_url": base_url}]
    
    raise ValueError(
        "No API keys found. Please set OPENAI_API_KEY environment variable "
        "or provide api_keys.json"
    )


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying functions on error."""
    import asyncio
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

