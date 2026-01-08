"""Base dataset classes."""
from abc import ABC, abstractmethod
from typing import List, Iterator
from dataclasses import dataclass, field


@dataclass
class Sample:
    """Universal sample format."""
    id: str
    instruction: str
    input: str
    output: str  # Ground truth
    metadata: dict = field(default_factory=dict)


class BaseDataset(ABC):
    """Abstract dataset with train/val/test splits."""
    
    def __init__(self):
        self._train: List[Sample] = []
        self._val: List[Sample] = []
        self._test: List[Sample] = []
        self._loaded = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass
    
    @property
    @abstractmethod
    def task_type(self) -> str:
        """Task type: 'math', 'code', 'text', etc."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load data into _train, _val, _test."""
        pass
    
    def _ensure_loaded(self):
        """Ensure dataset is loaded."""
        if not self._loaded:
            self.load()
            self._loaded = True
    
    @property
    def train(self) -> List[Sample]:
        """Get training samples."""
        self._ensure_loaded()
        return self._train
    
    @property
    def val(self) -> List[Sample]:
        """Get validation samples."""
        self._ensure_loaded()
        return self._val
    
    @property
    def test(self) -> List[Sample]:
        """Get test samples."""
        self._ensure_loaded()
        return self._test
    
    def train_iter(self, batch_size: int) -> Iterator[List[Sample]]:
        """
        Iterate over training samples in batches.
        
        Args:
            batch_size: Batch size
        
        Yields:
            Batches of samples
        """
        train_data = self.train
        for i in range(0, len(train_data), batch_size):
            yield train_data[i:i + batch_size]

