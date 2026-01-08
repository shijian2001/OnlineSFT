"""LLaMA-Factory trainer wrapper."""
import json
import yaml
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from ..synthesizer.base import SFTSample


@dataclass
class TrainConfig:
    """Training configuration."""
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    cutoff_len: int = 4096
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    bf16: bool = True


class LLaMAFactoryTrainer:
    """Wrapper for LLaMA-Factory training."""
    
    def __init__(
        self,
        template_path: str = "configs/train_template.yaml",
    ):
        """
        Initialize trainer.
        
        Args:
            template_path: Path to training template YAML
        """
        self.template_path = Path(template_path)
    
    def _load_template(self) -> dict:
        """Load training template."""
        if self.template_path.exists():
            with open(self.template_path) as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def train(
        self,
        base_model: str,
        sft_samples: List[SFTSample],
        output_dir: str,
        config: Optional[TrainConfig] = None,
    ) -> str:
        """
        Train model with LLaMA-Factory.
        
        Args:
            base_model: Base model path
            sft_samples: SFT training samples
            output_dir: Output directory for checkpoint
            config: Training configuration
        
        Returns:
            Path to trained checkpoint
        """
        cfg = config or TrainConfig()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training with {len(sft_samples)} samples")
        print(f"Base model: {base_model}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        
        # Save training data
        data_path = output_path / "train_data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(
                [sample.to_dict() for sample in sft_samples],
                f,
                ensure_ascii=False,
                indent=2,
            )
        
        print(f"Training data saved to: {data_path}")
        
        # Build training config
        train_config = {
            **self._load_template(),
            "model_name_or_path": base_model,
            "dataset": str(data_path),
            "output_dir": str(output_path),
            "num_train_epochs": cfg.num_epochs,
            "per_device_train_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "lr_scheduler_type": cfg.lr_scheduler_type,
            "warmup_ratio": cfg.warmup_ratio,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "cutoff_len": cfg.cutoff_len,
            "logging_steps": cfg.logging_steps,
            "save_steps": cfg.save_steps,
            "save_total_limit": cfg.save_total_limit,
            "bf16": cfg.bf16,
        }
        
        # Save config
        config_path = output_path / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(train_config, f)
        
        print(f"Training config saved to: {config_path}")
        
        # Run training using LLaMA-Factory CLI
        try:
            print("\nStarting training...\n")
            subprocess.run(
                ["python", "-m", "llamafactory.cli", "train", str(config_path)],
                check=True,
            )
            print("\n✅ Training completed successfully\n")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed: {e}\n")
            raise
        except FileNotFoundError:
            print(
                f"\n❌ LLaMA-Factory not found. Please ensure it's installed:\n"
                f"   uv sync --active\n"
            )
            raise
        
        return str(output_path)

