#!/usr/bin/env python
"""Evaluate checkpoint on test set."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from online_sft.inferencer import VLLMInferencer, GenerationConfig
from online_sft.dataset.registry import DatasetRegistry
from online_sft.evaluator.base import EvaluatorRegistry


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint on test set"
    )
    
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint",
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
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Output file for results (default: eval_results.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
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
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("Checkpoint Evaluation")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print("="*70 + "\n")
    
    # Load dataset
    print("📊 Loading dataset...")
    dataset_kwargs = {}
    if args.dataset_path:
        dataset_kwargs["data_path"] = args.dataset_path
    dataset = DatasetRegistry.get(args.dataset, **dataset_kwargs)
    test_samples = dataset.test
    
    print(f"   Test samples: {len(test_samples)}")
    
    # Load evaluator
    print(f"\n⚙️  Loading evaluator for task: {dataset.task_type}")
    evaluator = EvaluatorRegistry.get(dataset.task_type)
    
    # Load inferencer
    print("\n🤖 Loading model...")
    inferencer = VLLMInferencer(
        tensor_parallel_size=args.tensor_parallel,
    )
    inferencer.load(args.checkpoint)
    
    # Generation config
    gen_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    # Run inference
    print("\n🔄 Running inference...")
    prompts = []
    for sample in test_samples:
        if sample.instruction:
            prompt = f"{sample.instruction}\n\n{sample.input}"
        else:
            prompt = sample.input
        prompts.append(prompt)
    
    responses = inferencer.generate(prompts, gen_config)
    print(f"   ✓ Generated {len(responses)} responses")
    
    # Evaluate
    print("\n📈 Evaluating responses...")
    ground_truths = [sample.output for sample in test_samples]
    eval_results = evaluator.evaluate_batch(responses, ground_truths)
    
    accuracy = evaluator.accuracy(eval_results)
    success_count = sum(r.success for r in eval_results)
    
    print(f"   ✓ Accuracy: {accuracy:.2%} ({success_count}/{len(eval_results)})")
    
    # Prepare output
    output_data = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "accuracy": accuracy,
        "total": len(test_samples),
        "correct": success_count,
        "details": [
            {
                "id": sample.id,
                "success": result.success,
                "prediction": result.prediction,
                "ground_truth": result.ground_truth,
                "response": result.response[:200] + "..." if len(result.response) > 200 else result.response,
                "error_type": result.error_type,
            }
            for sample, result in zip(test_samples, eval_results)
        ],
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct: {success_count} / {len(test_samples)}")
    print(f"Incorrect: {len(test_samples) - success_count}")
    
    # Show some examples
    print("\n" + "-"*70)
    print("Sample Results (first 3):")
    print("-"*70)
    for i, (sample, result) in enumerate(zip(test_samples[:3], eval_results[:3])):
        status = "✓" if result.success else "✗"
        print(f"\n{status} Sample {i+1}: {sample.id}")
        print(f"   Problem: {sample.input[:100]}...")
        print(f"   Predicted: {result.prediction}")
        print(f"   Ground Truth: {result.ground_truth}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

