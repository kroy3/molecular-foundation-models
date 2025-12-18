"""QM9 dataset loader."""

import os
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.datasets import QM9 as PyGQM9


class QM9Dataset(InMemoryDataset):
    """
    QM9 dataset wrapper.
    
    134k small organic molecules with 13 quantum mechanical properties.
    """
    
    def __init__(self, root="./data/qm9", split="train", transform=None):
        """
        Args:
            root: Root directory for dataset
            split: train, val, or test
            transform: Optional transform
        """
        self.split = split
        super().__init__(root, transform)
        
        # Load split
        if split == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == "val":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:  # test
            self.data, self.slices = torch.load(self.processed_paths[2])
    
    @property
    def raw_file_names(self):
        return ['qm9.pt']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    
    def download(self):
        """Download QM9 dataset."""
        # Use PyTorch Geometric's QM9
        dataset = PyGQM9(root=self.root)
        torch.save(dataset, os.path.join(self.raw_dir, 'qm9.pt'))
    
    def process(self):
        """Process and split QM9 dataset."""
        # Load raw dataset
        dataset = torch.load(os.path.join(self.raw_dir, 'qm9.pt'))
        
        # Split: 80% train, 10% val, 10% test
        n = len(dataset)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        # Shuffle with fixed seed
        torch.manual_seed(42)
        perm = torch.randperm(n)
        
        train_data = [dataset[i] for i in perm[:n_train]]
        val_data = [dataset[i] for i in perm[n_train:n_train+n_val]]
        test_data = [dataset[i] for i in perm[n_train+n_val:]]
        
        # Save splits
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
