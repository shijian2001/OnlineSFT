# OnlineSFT

**Online Supervised Fine-Tuning Pipeline Framework**

A framework for continuous learning and model improvement through iterative inference, evaluation, and fine-tuning.

## 🌟 Features

- **Continuous Learning**: Stream through training data, continuously improving the model
- **Automatic Validation**: Validate checkpoints and automatically rollback if performance degrades
- **Flexible Synthesis**: Multiple strategies for generating training data (direct, success-only, LLM-based rationales)
- **Two Modes**:
  - **with_eval**: Infer → Evaluate → Synthesize → Train (with accuracy feedback)
  - **no_eval**: Infer → Synthesize → Train (direct synthesis)
- **Extensible**: Easy to add new datasets, evaluators, and synthesizers

## 📋 Requirements

- Python 3.10+
- PyTorch 2.0+
- vLLM 0.6+ (for inference)
- LLaMA-Factory (for training)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo> OnlineSFT
cd OnlineSFT

# Create and activate virtual environment
./scripts/setup_env.sh ~/.virtualenvs/online_sft/dev
source scripts/activate_env.sh ~/.virtualenvs/online_sft/dev

# Install all dependencies (includes vLLM and LLaMA-Factory)
uv sync --active
```

### 2. Run Training

```bash
# Basic usage
python scripts/run_pipeline.py \
    --model Qwen/Qwen3-8B \
    --dataset aime \
    --output outputs/my_run

# Common parameters:
#   --tensor-parallel N          # Number of GPUs (default: 8)
#   --batch-size N               # Inference batch size (default: 8)
#   --synthesizer TYPE           # direct|success|ground_truth|llm (default: direct)
#   --api-keys-path PATH         # Required for --synthesizer llm
#   --llm-model MODEL            # LLM model name (default: gpt-4o)
#   --no-eval                    # Skip evaluation step
#   --validate-every N           # Validate every N steps (default: 1)
#   --early-stop-patience N      # Early stopping patience (default: 5)
```

### 3. Evaluate Checkpoint

```bash
python scripts/eval_checkpoint.py \
    --checkpoint outputs/my_run/step_5 \
    --dataset aime \
    --output eval_results.json
```

## 📁 Project Structure

```
OnlineSFT/
├── pyproject.toml              # Project configuration
├── configs/
│   ├── train_template.yaml     # LLaMA-Factory training template
│   └── pipeline.yaml           # Pipeline configuration example
├── scripts/
│   ├── setup_env.sh            # Environment setup
│   ├── activate_env.sh         # Environment activation
│   ├── run_pipeline.py         # Main training script
│   └── eval_checkpoint.py      # Checkpoint evaluation
└── online_sft/
    ├── api/                    # LLM API clients
    ├── dataset/                # Dataset implementations
    ├── inferencer/             # Model inference backends
    ├── evaluator/              # Response evaluators
    ├── synthesizer/            # Data synthesis strategies
    ├── trainer/                # Training wrappers
    ├── pipeline.py             # Main pipeline orchestration
    └── utils/                  # Utility functions
```

## 🎯 Adding New Components

### Add a New Dataset

```python
# online_sft/dataset/my_dataset.py
from .base import BaseDataset, Sample
from .registry import DatasetRegistry

@DatasetRegistry.register("my_dataset")
class MyDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "my_dataset"
    
    @property
    def task_type(self) -> str:
        return "math"  # or "code", "text", etc.
    
    def load(self) -> None:
        # Load your data into self._train, self._val, self._test
        pass
```

### Add a New Evaluator

```python
# online_sft/evaluator/my_evaluator.py
from .base import BaseEvaluator, EvaluatorRegistry

@EvaluatorRegistry.register("my_task")
class MyEvaluator(BaseEvaluator):
    @property
    def task_type(self) -> str:
        return "my_task"
    
    def extract_answer(self, response: str):
        # Extract answer from response
        pass
    
    def check(self, prediction, ground_truth) -> bool:
        # Check if prediction is correct
        pass
```

### Add a New Synthesizer

```python
# online_sft/synthesizer/my_synthesizer.py
from .base import BaseSynthesizer, SFTSample

class MySynthesizer(BaseSynthesizer):
    def synthesize(self, samples, responses, eval_results=None):
        # Generate SFT samples
        return [SFTSample(...) for ...]
```

## 📊 Configuration

Example `configs/pipeline.yaml`:

```yaml
model_path: "Qwen/Qwen3-8B"
output_dir: "outputs"
dataset: "aime"
batch_size: 8
validate_every: 1
early_stop_patience: 5
use_eval: true

generation:
  max_tokens: 4096
  temperature: 0.7
  top_p: 0.9

training:
  num_epochs: 1
  batch_size: 4
  learning_rate: 1.0e-5
  lora_rank: 8
  lora_alpha: 16

synthesizer:
  type: "llm"
  model: "gpt-4o"
  include_success: true
  include_failure: true
```

## 🔑 API Keys (for LLM Synthesis)

Create `api_keys.json` with one of these formats:

**Simple format (list of keys):**
```json
[
  "sk-key-1",
  "sk-key-2"
]
```

**Full format (with custom base URLs):**
```json
[
  {
    "api_key": "sk-key-1",
    "base_url": "https://api.openai.com/v1"
  }
]
```

Or set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## 📈 Monitoring

The pipeline saves:
- `history.json`: Training history with scores
- `step_*/`: Checkpoints for each training step
- `step_*/train_data.json`: Training data used
- `step_*/training_config.yaml`: Training configuration

