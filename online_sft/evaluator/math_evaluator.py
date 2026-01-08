"""Math evaluator for mathematical reasoning tasks."""
import re
from typing import Any, Optional
from .base import BaseEvaluator, EvaluatorRegistry

try:
    from sympy import sympify, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


@EvaluatorRegistry.register("math")
class MathEvaluator(BaseEvaluator):
    """Evaluator for mathematical reasoning tasks."""
    
    # Patterns for extracting answers
    BOXED_PATTERN = re.compile(r'\\boxed\{([^}]+)\}')
    ANSWER_PATTERN = re.compile(r'[Aa]nswer\s*(?:is|:)\s*[=]?\s*([^\n.]+)')
    NUMBER_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\b')
    
    @property
    def task_type(self) -> str:
        return "math"
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize math evaluator.
        
        Args:
            tolerance: Tolerance for numerical comparison
        """
        self.tolerance = tolerance
    
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract answer from mathematical response.
        
        Tries multiple extraction strategies:
        1. LaTeX \\boxed{...}
        2. "Answer is/: ..."
        3. Last number in response
        
        Args:
            response: Model's response
        
        Returns:
            Extracted answer string or None
        """
        # Try boxed format
        boxed_matches = self.BOXED_PATTERN.findall(response)
        if boxed_matches:
            return boxed_matches[-1].strip()
        
        # Try "Answer is/:" format
        answer_matches = self.ANSWER_PATTERN.findall(response)
        if answer_matches:
            return answer_matches[-1].strip()
        
        # Try last number
        number_matches = self.NUMBER_PATTERN.findall(response)
        if number_matches:
            return number_matches[-1].strip()
        
        return None
    
    def check(self, prediction: Any, ground_truth: Any) -> bool:
        """
        Check if prediction matches ground truth.
        
        Tries multiple comparison methods:
        1. Exact string match
        2. Numerical comparison (with tolerance)
        3. Symbolic comparison (if sympy available)
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
        
        Returns:
            True if answers match
        """
        if prediction is None:
            return False
        
        # Normalize strings
        pred_str = str(prediction).strip()
        gt_str = str(ground_truth).strip()
        
        # Exact match
        if pred_str == gt_str:
            return True
        
        # Try numerical comparison
        try:
            pred_num = float(pred_str)
            gt_num = float(gt_str)
            return abs(pred_num - gt_num) < self.tolerance
        except (ValueError, TypeError):
            pass
        
        # Try symbolic comparison (if available)
        if SYMPY_AVAILABLE:
            try:
                pred_expr = sympify(pred_str)
                gt_expr = sympify(gt_str)
                diff = simplify(pred_expr - gt_expr)
                return diff == 0
            except Exception:
                pass
        
        return False

