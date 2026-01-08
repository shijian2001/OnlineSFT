"""Base inferencer classes."""
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop if self.stop else None,
        }


class BaseInferencer(ABC):
    """Abstract base class for model inference."""
    
    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend name (e.g., 'vllm', 'transformers')."""
        pass
    
    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """
        Load model from path.
        
        Args:
            model_path: Path to model checkpoint
            **kwargs: Additional backend-specific arguments
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """
        Generate responses for prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
        
        Returns:
            List of generated responses
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        pass

