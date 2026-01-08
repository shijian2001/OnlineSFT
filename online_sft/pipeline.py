"""OnlineSFT Pipeline - Main orchestration module."""
import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from .dataset.base import BaseDataset, Sample
from .inferencer.base import BaseInferencer, GenerationConfig
from .evaluator.base import BaseEvaluator, EvalResult, EvaluatorRegistry
from .synthesizer.base import BaseSynthesizer, SFTSample, DirectSynthesizer
from .trainer.llamafactory import LLaMAFactoryTrainer, TrainConfig


@dataclass
class PipelineConfig:
    """Configuration for OnlineSFT pipeline."""
    model_path: str
    output_dir: str = "outputs"
    validate_every: int = 1
    early_stop_patience: int = 5
    min_samples: int = 1
    use_eval: bool = True  # False = no eval mode


class OnlineSFTPipeline:
    """
    Main pipeline orchestrator.
    
    Modes:
    - use_eval=True: Infer → Evaluate → Synthesize → Train
    - use_eval=False: Infer → Synthesize → Train
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        inferencer: BaseInferencer,
        evaluator: Optional[BaseEvaluator] = None,
        synthesizer: Optional[BaseSynthesizer] = None,
        trainer: Optional[LLaMAFactoryTrainer] = None,
        train_config: Optional[TrainConfig] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            inferencer: Model inferencer
            evaluator: Response evaluator (optional, auto-selected if needed)
            synthesizer: Data synthesizer (optional, defaults to DirectSynthesizer)
            trainer: Model trainer (optional, defaults to LLaMAFactoryTrainer)
            train_config: Training configuration (optional)
        """
        self.config = config
        self.inferencer = inferencer
        self.evaluator = evaluator
        self.synthesizer = synthesizer or DirectSynthesizer()
        self.trainer = trainer or LLaMAFactoryTrainer()
        self.train_config = train_config
        
        # Pipeline state
        self.current_checkpoint = config.model_path
        self.best_checkpoint = config.model_path
        self.best_score = 0.0
        self._no_improve_count = 0
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        dataset: BaseDataset,
        batch_size: int = 8,
        gen_config: Optional[GenerationConfig] = None,
    ) -> dict:
        """
        Run the OnlineSFT pipeline.
        
        Args:
            dataset: Dataset with train/val/test splits
            batch_size: Batch size for inference
            gen_config: Generation configuration
        
        Returns:
            Results dictionary with best_score, best_checkpoint, and history
        """
        print("\n" + "="*70)
        print("OnlineSFT Pipeline Starting")
        print("="*70)
        print(f"Model: {self.current_checkpoint}")
        print(f"Dataset: {dataset.name} ({dataset.task_type})")
        print(f"Mode: {'with_eval' if self.config.use_eval else 'no_eval'}")
        print(f"Output: {self.config.output_dir}")
        print("="*70 + "\n")
        
        # Auto-select evaluator if needed
        if self.config.use_eval and not self.evaluator:
            print(f"Auto-selecting evaluator for task: {dataset.task_type}")
            self.evaluator = EvaluatorRegistry.get(dataset.task_type)
        
        # Load initial model
        self.inferencer.load(self.current_checkpoint)
        
        # Training history
        history = []
        
        # Iterate over training batches
        for step, batch in enumerate(dataset.train_iter(batch_size)):
            print(f"\n{'─'*70}")
            print(f"Step {step}: Processing {len(batch)} samples")
            print(f"{'─'*70}")
            
            # 1. Inference
            print("\n[1/4] Running inference...")
            prompts = [self._format_prompt(sample) for sample in batch]
            responses = self.inferencer.generate(prompts, gen_config)
            print(f"  ✓ Generated {len(responses)} responses")
            
            # 2. Evaluation (optional)
            eval_results = None
            if self.config.use_eval:
                print("\n[2/4] Evaluating responses...")
                ground_truths = [sample.output for sample in batch]
                eval_results = self.evaluator.evaluate_batch(responses, ground_truths)
                accuracy = self.evaluator.accuracy(eval_results)
                success_count = sum(r.success for r in eval_results)
                print(f"  ✓ Accuracy: {accuracy:.2%} ({success_count}/{len(eval_results)})")
            else:
                print("\n[2/4] Skipping evaluation (no_eval mode)")
            
            # 3. Synthesis
            print("\n[3/4] Synthesizing training data...")
            sft_samples = self.synthesizer.synthesize(batch, responses, eval_results)
            print(f"  ✓ Generated {len(sft_samples)} SFT samples")
            
            # Check minimum samples
            if len(sft_samples) < self.config.min_samples:
                print(f"  ⚠ Skipping training: only {len(sft_samples)} samples "
                      f"(min: {self.config.min_samples})")
                continue
            
            # 4. Training
            print("\n[4/4] Training model...")
            step_output_dir = Path(self.config.output_dir) / f"step_{step}"
            new_checkpoint = self.trainer.train(
                base_model=self.current_checkpoint,
                sft_samples=sft_samples,
                output_dir=str(step_output_dir),
                config=self.train_config,
            )
            print(f"  ✓ Checkpoint saved: {new_checkpoint}")
            
            # 5. Validation
            if step % self.config.validate_every == 0:
                print("\n[5/5] Validating checkpoint...")
                improved = self._validate_checkpoint(
                    checkpoint=new_checkpoint,
                    val_samples=dataset.val,
                    gen_config=gen_config,
                )
                
                if improved:
                    self._no_improve_count = 0
                else:
                    self._no_improve_count += 1
            else:
                # Skip validation, just use new checkpoint
                self.current_checkpoint = new_checkpoint
                print(f"\n[5/5] Skipping validation (every {self.config.validate_every} steps)")
            
            # Record history
            step_info = {
                "step": step,
                "samples": len(sft_samples),
                "checkpoint": self.current_checkpoint,
                "best_score": self.best_score,
            }
            if eval_results:
                step_info["accuracy"] = accuracy
            history.append(step_info)
            
            # Save history
            self._save_history(history)
            
            # Check early stopping
            if self._no_improve_count >= self.config.early_stop_patience:
                print(f"\n⚠ Early stopping triggered after {step + 1} steps")
                print(f"   No improvement for {self._no_improve_count} validations")
                break
        
        # Final results
        print("\n" + "="*70)
        print("Pipeline Completed")
        print("="*70)
        print(f"Best Score: {self.best_score:.2%}")
        print(f"Best Checkpoint: {self.best_checkpoint}")
        print(f"Total Steps: {len(history)}")
        print("="*70 + "\n")
        
        return {
            "best_score": self.best_score,
            "best_checkpoint": self.best_checkpoint,
            "history": history,
        }
    
    def _format_prompt(self, sample: Sample) -> str:
        """Format sample as prompt."""
        if sample.instruction:
            return f"{sample.instruction}\n\n{sample.input}"
        return sample.input
    
    def _validate_checkpoint(
        self,
        checkpoint: str,
        val_samples: List[Sample],
        gen_config: Optional[GenerationConfig],
    ) -> bool:
        """
        Validate checkpoint on validation set.
        
        Args:
            checkpoint: Checkpoint path
            val_samples: Validation samples
            gen_config: Generation config
        
        Returns:
            True if checkpoint improved, False otherwise
        """
        # Load checkpoint
        self.inferencer.load(checkpoint)
        
        # Generate responses
        prompts = [self._format_prompt(sample) for sample in val_samples]
        responses = self.inferencer.generate(prompts, gen_config)
        
        # Evaluate
        ground_truths = [sample.output for sample in val_samples]
        
        # Use evaluator (always needed for validation)
        if not self.evaluator:
            # This shouldn't happen, but handle gracefully
            print("  ⚠ No evaluator available, using current checkpoint")
            self.current_checkpoint = checkpoint
            return False
        
        eval_results = self.evaluator.evaluate_batch(responses, ground_truths)
        score = self.evaluator.accuracy(eval_results)
        success_count = sum(r.success for r in eval_results)
        
        print(f"  Validation Score: {score:.2%} ({success_count}/{len(eval_results)})")
        
        # Check if improved
        if score > self.best_score:
            self.best_score = score
            self.best_checkpoint = checkpoint
            self.current_checkpoint = checkpoint
            print(f"  ✓ New best checkpoint! (Δ = +{score - self.best_score:.2%})")
            return True
        else:
            # Rollback to best checkpoint
            self.current_checkpoint = self.best_checkpoint
            self.inferencer.load(self.best_checkpoint)
            print(f"  ✗ No improvement. Rolling back to best checkpoint.")
            print(f"     Current: {score:.2%} | Best: {self.best_score:.2%}")
            return False
    
    def _save_history(self, history: List[dict]):
        """Save training history to JSON."""
        history_path = Path(self.config.output_dir) / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

