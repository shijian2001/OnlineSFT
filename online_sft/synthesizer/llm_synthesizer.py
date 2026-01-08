"""LLM-based synthesizer for generating rationales."""
import asyncio
from typing import List, Optional
from .base import BaseSynthesizer, SFTSample
from ..dataset.base import Sample
from ..evaluator.base import EvalResult
from ..api import StreamGenerator


class LLMRationalSynthesizer(BaseSynthesizer):
    """
    Generate rationales via external LLM API.
    
    Modes:
    - with eval: different prompts for success/failure
    - no eval: generate for all
    """
    
    # Default templates
    SUCCESS_TEMPLATE = """Problem: {problem}
Model Response: {response}
Correct Answer: {gt}

The response is CORRECT. Generate a clean step-by-step solution."""

    FAILURE_TEMPLATE = """Problem: {problem}
Model Response: {response}
Predicted: {pred}
Correct Answer: {gt}
Error: {error}

The response is WRONG. Generate a correct step-by-step solution."""

    NO_EVAL_TEMPLATE = """Problem: {problem}
Reference Answer: {gt}

Generate a detailed step-by-step solution."""
    
    def __init__(
        self,
        stream_generator: StreamGenerator,
        success_template: Optional[str] = None,
        failure_template: Optional[str] = None,
        no_eval_template: Optional[str] = None,
        include_success: bool = True,
        include_failure: bool = True,
        system_prompt: str = "You are a helpful assistant that generates clear, step-by-step solutions to problems.",
    ):
        """
        Initialize LLM synthesizer.
        
        Args:
            stream_generator: StreamGenerator instance for API calls
            success_template: Template for successful responses
            failure_template: Template for failed responses
            no_eval_template: Template for no-eval mode
            include_success: Whether to synthesize for successful responses
            include_failure: Whether to synthesize for failed responses
            system_prompt: System prompt for generation
        """
        self.generator = stream_generator
        self.success_template = success_template or self.SUCCESS_TEMPLATE
        self.failure_template = failure_template or self.FAILURE_TEMPLATE
        self.no_eval_template = no_eval_template or self.NO_EVAL_TEMPLATE
        self.include_success = include_success
        self.include_failure = include_failure
        self.system_prompt = system_prompt
    
    def synthesize(
        self,
        samples: List[Sample],
        responses: List[str],
        eval_results: Optional[List[EvalResult]] = None,
    ) -> List[SFTSample]:
        """Synthesize SFT samples with LLM-generated rationales."""
        return asyncio.run(
            self._synthesize_async(samples, responses, eval_results)
        )
    
    async def _synthesize_async(
        self,
        samples: List[Sample],
        responses: List[str],
        eval_results: Optional[List[EvalResult]],
    ) -> List[SFTSample]:
        """Async synthesis implementation using StreamGenerator."""
        # Build prompts with IDs for tracking
        prompts_with_ids = []
        
        for i, (sample, response) in enumerate(zip(samples, responses)):
            if eval_results:
                # With eval mode
                result = eval_results[i]
                
                # Skip based on settings
                if result.success and not self.include_success:
                    continue
                if not result.success and not self.include_failure:
                    continue
                
                # Build prompt
                prompt = self._build_prompt_with_eval(sample, response, result)
            else:
                # No eval mode
                prompt = self._build_prompt_no_eval(sample)
            
            prompts_with_ids.append({
                "id": str(i),
                "prompt": prompt
            })
        
        # Generate using StreamGenerator with unique ID mode
        sft_samples = []
        
        async for result_item in self.generator.generate_stream(
            prompts=prompts_with_ids,
            system_prompt=self.system_prompt,
        ):
            # result_item: {"id": str, "result": str}
            idx = int(result_item["id"])
            rational = result_item["result"]
            
            if rational:
                sample = samples[idx]
                sft_samples.append(
                    SFTSample(
                        instruction=sample.instruction,
                        input=sample.input,
                        output=rational,
                    )
                )
            else:
                print(f"Warning: Synthesis failed for sample {idx}")
        
        return sft_samples
    
    def _build_prompt_with_eval(
        self,
        sample: Sample,
        response: str,
        result: EvalResult,
    ) -> str:
        """Build prompt for eval mode."""
        if result.success:
            return self.success_template.format(
                problem=sample.input,
                response=response,
                gt=sample.output,
            )
        else:
            return self.failure_template.format(
                problem=sample.input,
                response=response,
                pred=result.prediction,
                gt=sample.output,
                error=result.error_type,
            )
    
    def _build_prompt_no_eval(self, sample: Sample) -> str:
        """Build prompt for no-eval mode."""
        return self.no_eval_template.format(
            problem=sample.input,
            gt=sample.output,
        )
    

