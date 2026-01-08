#!/usr/bin/env python
"""Main training script for OnlineSFT pipeline."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from online_sft.pipeline import OnlineSFTPipeline, PipelineConfig
from online_sft.inferencer import VLLMInferencer, GenerationConfig
from online_sft.dataset.registry import DatasetRegistry
from online_sft.synthesizer import (
    DirectSynthesizer,
    SuccessOnlySynthesizer,
    GroundTruthSynthesizer,
    LLMRationalSynthesizer,
)
from online_sft.trainer import TrainConfig
from online_sft.api import StreamGenerator, load_api_keys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OnlineSFT training pipeline"
    )
    
    # Model & Dataset
    parser.add_argument(
        "--model",
        required=True,
        help="Path to base model or HuggingFace model ID",
    )
    parser.add_argument(
        "--dataset",
        default="aime",
        help="Dataset name (default: aime)",
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to dataset file (optional)",
    )
    
    # Output
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory (default: outputs)",
    )
    
    # Pipeline config
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="Validate every N steps (default: 1)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples required for training (default: 1)",
    )
    
    # Mode
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Run without evaluation (direct synthesis mode)",
    )
    
    # Synthesizer
    parser.add_argument(
        "--synthesizer",
        choices=["direct", "success", "ground_truth", "llm"],
        default="direct",
        help="Synthesizer type (default: direct)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o",
        help="LLM model for synthesis (default: gpt-4o)",
    )
    parser.add_argument(
        "--api-keys-path",
        help="Path to API keys JSON file",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent requests per API key (default: 10)",
    )
    
    # Generation config
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    
    # Training config
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Training epochs per step (default: 1)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    
    # vLLM config
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max model length (default: 32768)",
    )
    
    return parser.parse_args()


def create_synthesizer(args):
    """Create synthesizer based on arguments."""
    if args.synthesizer == "direct":
        return DirectSynthesizer()
    elif args.synthesizer == "success":
        return SuccessOnlySynthesizer()
    elif args.synthesizer == "ground_truth":
        return GroundTruthSynthesizer()
    elif args.synthesizer == "llm":
        # Load API keys
        api_keys_config = load_api_keys(args.api_keys_path)
        
        # Extract just the API key strings
        api_keys = [cfg["api_key"] for cfg in api_keys_config]
        base_url = api_keys_config[0].get("base_url", "https://api.openai.com/v1")
        
        # Create StreamGenerator with unique_id mode enabled
        stream_generator = StreamGenerator(
            model_name=args.llm_model,
            api_keys=api_keys,
            max_concurrent_per_key=args.max_concurrent,
            with_unique_id=True,  # Enable ID tracking for maintaining order
        )
        
        print(f"   Using {len(api_keys)} API key(s) with StreamGenerator")
        print(f"   Base URL: {base_url}")
        print(f"   Max concurrent per key: {args.max_concurrent}")
        
        return LLMRationalSynthesizer(
            stream_generator=stream_generator,
        )
    else:
        raise ValueError(f"Unknown synthesizer: {args.synthesizer}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create dataset
    print(f"\n📊 Loading dataset: {args.dataset}")
    dataset_kwargs = {}
    if args.dataset_path:
        dataset_kwargs["data_path"] = args.dataset_path
    dataset = DatasetRegistry.get(args.dataset, **dataset_kwargs)
    
    print(f"   Train: {len(dataset.train)} samples")
    print(f"   Val: {len(dataset.val)} samples")
    print(f"   Test: {len(dataset.test)} samples")
    
    # Create inferencer
    print(f"\n🤖 Initializing inferencer...")
    inferencer = VLLMInferencer(
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
    )
    
    # Create synthesizer
    print(f"\n⚙️  Creating synthesizer: {args.synthesizer}")
    synthesizer = create_synthesizer(args)
    
    # Generation config
    gen_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    # Training config
    train_config = TrainConfig(
        num_epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Pipeline config
    pipeline_config = PipelineConfig(
        model_path=args.model,
        output_dir=args.output,
        validate_every=args.validate_every,
        early_stop_patience=args.early_stop_patience,
        min_samples=args.min_samples,
        use_eval=not args.no_eval,
    )
    
    # Create pipeline
    pipeline = OnlineSFTPipeline(
        config=pipeline_config,
        inferencer=inferencer,
        synthesizer=synthesizer,
        train_config=train_config,
    )
    
    # Run pipeline
    try:
        result = pipeline.run(
            dataset=dataset,
            batch_size=args.batch_size,
            gen_config=gen_config,
        )
        
        print("\n" + "="*70)
        print("✅ Pipeline Completed Successfully")
        print("="*70)
        print(f"Best Score: {result['best_score']:.2%}")
        print(f"Best Checkpoint: {result['best_checkpoint']}")
        print(f"Total Steps: {len(result['history'])}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}\n")
        raise


if __name__ == "__main__":
    main()

