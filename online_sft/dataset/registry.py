"""Dataset registry."""
from typing import Dict, Type
from .base import BaseDataset


class DatasetRegistry:
    """Registry for datasets."""
    
    _registry: Dict[str, Type[BaseDataset]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a dataset.
        
        Usage:
            @DatasetRegistry.register("my_dataset")
            class MyDataset(BaseDataset):
                ...
        """
        def wrapper(dataset_class):
            cls._registry[name] = dataset_class
            return dataset_class
        return wrapper
    
    @classmethod
    def get(cls, name: str, **kwargs) -> BaseDataset:
        """
        Get dataset by name.
        
        Args:
            name: Dataset name
            **kwargs: Arguments to pass to dataset constructor
        
        Returns:
            Dataset instance
        """
        if name not in cls._registry:
            raise ValueError(
                f"Dataset '{name}' not found. Available: {cls.list()}"
            )
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list(cls) -> list:
        """List all registered datasets."""
        return list(cls._registry.keys())

