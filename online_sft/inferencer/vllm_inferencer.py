"""vLLM-based inferencer."""
from typing import List, Optional
from .base import BaseInferencer, GenerationConfig

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None


class VLLMInferencer(BaseInferencer):
    """vLLM backend for fast inference with models like Qwen, Llama, etc."""
    
    @property
    def backend(self) -> str:
        return "vllm"
    
    def __init__(
        self,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
    ):
        """
        Initialize vLLM inferencer.
        
        Args:
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            trust_remote_code: Whether to trust remote code
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )
        
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        
        self._engine: Optional[LLM] = None
        self._model_path: Optional[str] = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._engine is not None
    
    def load(self, model_path: str, **kwargs) -> None:
        """
        Load model with vLLM.
        
        Args:
            model_path: Path to model checkpoint
            **kwargs: Additional vLLM arguments
        """
        # If same model already loaded, skip
        if self._model_path == model_path and self._engine:
            print(f"Model already loaded: {model_path}")
            return
        
        # Unload previous model if any
        self.unload()
        
        print(f"Loading model with vLLM: {model_path}")
        
        # Create vLLM engine
        self._engine = LLM(
            model=model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
            **kwargs,
        )
        
        self._model_path = model_path
        print(f"Model loaded successfully")
    
    def unload(self) -> None:
        """Unload model and free GPU memory."""
        if self._engine:
            print(f"Unloading model: {self._model_path}")
            del self._engine
            self._engine = None
            self._model_path = None
            
            # Force garbage collection
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
    
    def generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """
        Generate responses using vLLM.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
        
        Returns:
            List of generated responses
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Use default config if not provided
        cfg = config or GenerationConfig()
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            stop=cfg.stop if cfg.stop else None,
        )
        
        # Generate
        outputs = self._engine.generate(prompts, sampling_params)
        
        # Extract text from outputs
        responses = [output.outputs[0].text for output in outputs]
        
        return responses

