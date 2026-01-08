"""AIME (American Invitational Mathematics Examination) dataset."""
import json
from pathlib import Path
from typing import Optional, List
from .base import BaseDataset, Sample
from .registry import DatasetRegistry


@DatasetRegistry.register("aime")
class AIMEDataset(BaseDataset):
    """
    AIME math competition dataset.
    
    Expected format: JSON file with list of problems:
    [
        {
            "id": "aime_1983_1",
            "problem": "Problem text...",
            "answer": "42",
            "year": 1983,
            "problem_num": 1
        },
        ...
    ]
    """
    
    @property
    def name(self) -> str:
        return "aime"
    
    @property
    def task_type(self) -> str:
        return "math"
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        instruction: Optional[str] = None,
    ):
        """
        Initialize AIME dataset.
        
        Args:
            data_path: Path to AIME JSON file
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for test
            instruction: Custom instruction template
        """
        super().__init__()
        self.data_path = data_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.instruction = instruction or (
            "Solve the following math problem. "
            "Provide your final answer in the format: \\boxed{answer}"
        )
    
    def load(self) -> None:
        """Load AIME data from file or generate sample data."""
        if self.data_path and Path(self.data_path).exists():
            self._load_from_file(self.data_path)
        else:
            # Generate sample data for testing
            self._load_sample_data()
        
        self._loaded = True
    
    def _load_from_file(self, path: str) -> None:
        """Load data from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = Sample(
                id=item.get("id", f"aime_{len(samples)}"),
                instruction=self.instruction,
                input=item["problem"],
                output=str(item["answer"]),
                metadata={
                    "year": item.get("year"),
                    "problem_num": item.get("problem_num"),
                },
            )
            samples.append(sample)
        
        self._split_data(samples)
    
    def _load_sample_data(self) -> None:
        """Generate sample AIME-style problems for testing."""
        problems = [
            {
                "id": "aime_sample_1",
                "problem": "Find the number of positive integers less than 1000 that are divisible by 7 but not by 2 or 5.",
                "answer": "68",
            },
            {
                "id": "aime_sample_2",
                "problem": "What is the sum of all positive integers n such that n^2 + 12n - 2023 = 0?",
                "answer": "0",
            },
            {
                "id": "aime_sample_3",
                "problem": "Find the smallest positive integer n such that 2^n > 10^6.",
                "answer": "20",
            },
            {
                "id": "aime_sample_4",
                "problem": "How many ways can you arrange the letters in MATHEMATICS?",
                "answer": "4989600",
            },
            {
                "id": "aime_sample_5",
                "problem": "What is the largest prime factor of 2023?",
                "answer": "289",
            },
            {
                "id": "aime_sample_6",
                "problem": "Find the sum of the first 100 positive integers.",
                "answer": "5050",
            },
            {
                "id": "aime_sample_7",
                "problem": "What is the value of 15! / (10! * 5!)?",
                "answer": "3003",
            },
            {
                "id": "aime_sample_8",
                "problem": "How many diagonals does a regular 20-gon have?",
                "answer": "170",
            },
            {
                "id": "aime_sample_9",
                "problem": "What is the GCD of 12345 and 67890?",
                "answer": "15",
            },
            {
                "id": "aime_sample_10",
                "problem": "Find the number of perfect squares between 1000 and 2000.",
                "answer": "13",
            },
        ]
        
        samples = []
        for item in problems:
            sample = Sample(
                id=item["id"],
                instruction=self.instruction,
                input=item["problem"],
                output=item["answer"],
                metadata={"source": "sample"},
            )
            samples.append(sample)
        
        self._split_data(samples)
    
    def _split_data(self, samples: List[Sample]) -> None:
        """Split samples into train/val/test."""
        n = len(samples)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)
        
        self._train = samples[:n_train]
        self._val = samples[n_train:n_train + n_val]
        self._test = samples[n_train + n_val:]

