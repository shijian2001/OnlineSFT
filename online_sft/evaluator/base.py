"""Base evaluator classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, Type


@dataclass
class EvalResult:
    """Result of evaluating a single prediction."""
    success: bool
    prediction: Any
    ground_truth: Any
    response: str = ""
    error_type: Optional[str] = None  # 'extraction_failed', 'wrong_answer', etc.
    metadata: dict = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base class for evaluation."""
    
    @property
    @abstractmethod
    def task_type(self) -> str:
        """Task type this evaluator handles."""
        pass
    
    @abstractmethod
    def extract_answer(self, response: str) -> Any:
        """
        Extract answer from model response.
        
        Args:
            response: Model's generated response
        
        Returns:
            Extracted answer or None if extraction failed
        """
        pass
    
    @abstractmethod
    def check(self, prediction: Any, ground_truth: Any) -> bool:
        """
        Check if prediction matches ground truth.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
        
        Returns:
            True if correct
        """
        pass
    
    def evaluate(self, response: str, ground_truth: Any) -> EvalResult:
        """
        Evaluate a single response.
        
        Args:
            response: Model's generated response
            ground_truth: Ground truth answer
        
        Returns:
            Evaluation result
        """
        # Extract answer
        prediction = self.extract_answer(response)
        
        # Check if extraction failed
        if prediction is None:
            return EvalResult(
                success=False,
                prediction=None,
                ground_truth=ground_truth,
                response=response,
                error_type="extraction_failed",
            )
        
        # Check correctness
        is_correct = self.check(prediction, ground_truth)
        
        return EvalResult(
            success=is_correct,
            prediction=prediction,
            ground_truth=ground_truth,
            response=response,
            error_type=None if is_correct else "wrong_answer",
        )
    
    def evaluate_batch(
        self,
        responses: List[str],
        ground_truths: List[Any],
    ) -> List[EvalResult]:
        """
        Evaluate multiple responses.
        
        Args:
            responses: List of model responses
            ground_truths: List of ground truth answers
        
        Returns:
            List of evaluation results
        """
        return [
            self.evaluate(resp, gt)
            for resp, gt in zip(responses, ground_truths)
        ]
    
    def accuracy(self, results: List[EvalResult]) -> float:
        """
        Calculate accuracy from evaluation results.
        
        Args:
            results: List of evaluation results
        
        Returns:
            Accuracy (0.0-1.0)
        """
        if not results:
            return 0.0
        return sum(r.success for r in results) / len(results)


class EvaluatorRegistry:
    """Registry for evaluators."""
    
    _registry: Dict[str, Type[BaseEvaluator]] = {}
    
    @classmethod
    def register(cls, task_type: str):
        """
        Decorator to register an evaluator.
        
        Usage:
            @EvaluatorRegistry.register("math")
            class MathEvaluator(BaseEvaluator):
                ...
        """
        def wrapper(evaluator_class):
            cls._registry[task_type] = evaluator_class
            return evaluator_class
        return wrapper
    
    @classmethod
    def get(cls, task_type: str, **kwargs) -> BaseEvaluator:
        """
        Get evaluator by task type.
        
        Args:
            task_type: Task type
            **kwargs: Arguments to pass to evaluator constructor
        
        Returns:
            Evaluator instance
        """
        if task_type not in cls._registry:
            raise ValueError(
                f"Evaluator for '{task_type}' not found. Available: {cls.list()}"
            )
        return cls._registry[task_type](**kwargs)
    
    @classmethod
    def list(cls) -> list:
        """List all registered evaluators."""
        return list(cls._registry.keys())

