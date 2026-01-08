"""Base synthesizer classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from ..dataset.base import Sample
from ..evaluator.base import EvalResult


@dataclass
class SFTSample:
    """Sample for supervised fine-tuning."""
    instruction: str
    input: str
    output: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }


class BaseSynthesizer(ABC):
    """
    Abstract synthesizer for creating SFT data.
    
    Modes:
    - eval_results=None: no eval mode, use responses directly
    - eval_results provided: can filter by success/failure
    """
    
    @abstractmethod
    def synthesize(
        self,
        samples: List[Sample],
        responses: List[str],
        eval_results: Optional[List[EvalResult]] = None,
    ) -> List[SFTSample]:
        """
        Synthesize SFT samples from model responses.
        
        Args:
            samples: Original dataset samples
            responses: Model's responses
            eval_results: Optional evaluation results
        
        Returns:
            List of SFT samples
        """
        pass


class DirectSynthesizer(BaseSynthesizer):
    """Use model responses directly as training data."""
    
    def synthesize(
        self,
        samples: List[Sample],
        responses: List[str],
        eval_results: Optional[List[EvalResult]] = None,
    ) -> List[SFTSample]:
        """Use responses directly."""
        return [
            SFTSample(
                instruction=sample.instruction,
                input=sample.input,
                output=response,
            )
            for sample, response in zip(samples, responses)
        ]


class SuccessOnlySynthesizer(BaseSynthesizer):
    """Only use successful responses (requires evaluation)."""
    
    def synthesize(
        self,
        samples: List[Sample],
        responses: List[str],
        eval_results: Optional[List[EvalResult]] = None,
    ) -> List[SFTSample]:
        """Only use successful responses."""
        if not eval_results:
            raise ValueError("SuccessOnlySynthesizer requires eval_results")
        
        sft_samples = []
        for sample, response, result in zip(samples, responses, eval_results):
            if result.success:
                sft_samples.append(
                    SFTSample(
                        instruction=sample.instruction,
                        input=sample.input,
                        output=response,
                    )
                )
        
        return sft_samples


class GroundTruthSynthesizer(BaseSynthesizer):
    """Use ground truth answers as training data."""
    
    def synthesize(
        self,
        samples: List[Sample],
        responses: List[str],
        eval_results: Optional[List[EvalResult]] = None,
    ) -> List[SFTSample]:
        """Use ground truth answers."""
        return [
            SFTSample(
                instruction=sample.instruction,
                input=sample.input,
                output=sample.output,
            )
            for sample in samples
        ]


class FailureOnlySynthesizer(BaseSynthesizer):
    """Only use failed responses with ground truth correction."""
    
    def synthesize(
        self,
        samples: List[Sample],
        responses: List[str],
        eval_results: Optional[List[EvalResult]] = None,
    ) -> List[SFTSample]:
        """Only use failed responses, replacing with ground truth."""
        if not eval_results:
            raise ValueError("FailureOnlySynthesizer requires eval_results")
        
        sft_samples = []
        for sample, response, result in zip(samples, responses, eval_results):
            if not result.success:
                sft_samples.append(
                    SFTSample(
                        instruction=sample.instruction,
                        input=sample.input,
                        output=sample.output,  # Use ground truth
                    )
                )
        
        return sft_samples

