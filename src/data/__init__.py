"""
Dataset loading utilities for molecular datasets.
"""

from src.data.qm9 import QM9Dataset
from src.data.md17 import MD17Dataset
from src.data.ani1x import ANI1xDataset

__all__ = ["QM9Dataset", "MD17Dataset", "ANI1xDataset", "get_dataset"]


def get_dataset(name, split="train", **kwargs):
    """
    Get dataset by name.
    
    Args:
        name: Dataset name (qm9, md17, ani1x, or task-specific)
        split: Data split (train, val, test)
        **kwargs: Additional dataset arguments
        
    Returns:
        Dataset instance
    """
    datasets = {
        "qm9": QM9Dataset,
        "md17": MD17Dataset,
        "md17_aspirin": lambda **kw: MD17Dataset(molecule="aspirin", **kw),
        "md17_benzene": lambda **kw: MD17Dataset(molecule="benzene", **kw),
        "ani1x": ANI1xDataset,
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")
    
    return datasets[name](split=split, **kwargs)
