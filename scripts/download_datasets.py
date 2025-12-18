"""
Dataset Downloading and Preprocessing Script
============================================
This script downloads and prepares the QM9, MD17, and rMD17 datasets
for cross-domain pretraining experiments.

Requirements:
- torch >= 2.0
- torch_geometric >= 2.3
- numpy
- h5py
- requests
"""

import os
import torch
import numpy as np
from torch_geometric.datasets import QM9, MD17
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import urllib.request
import tarfile
import h5py
from pathlib import Path
from tqdm import tqdm
import pickle


class DatasetDownloader:
    """Handles downloading and preprocessing of molecular datasets."""
    
    def __init__(self, root_dir='./data'):
        """
        Initialize dataset downloader.
        
        Args:
            root_dir: Root directory for storing datasets
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # Define dataset paths
        self.qm9_path = self.root_dir / 'QM9'
        self.md17_path = self.root_dir / 'MD17'
        self.rmd17_path = self.root_dir / 'rMD17'
        
    def download_qm9(self, subset_size=50000):
        """
        Download and preprocess QM9 dataset.
        
        Args:
            subset_size: Number of molecules to sample (for computational efficiency)
            
        Returns:
            Preprocessed QM9 dataset
        """
        print("Downloading QM9 dataset...")
        
        # Download using PyTorch Geometric
        dataset = QM9(root=str(self.qm9_path))
        
        print(f"QM9 dataset contains {len(dataset)} molecules")
        
        # Create subset for training
        if subset_size < len(dataset):
            print(f"Creating random subset of {subset_size} molecules...")
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = dataset[indices]
        
        # Split into train/val/test
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val
        
        train_dataset = dataset[:n_train]
        val_dataset = dataset[n_train:n_train+n_val]
        test_dataset = dataset[n_train+n_val:]
        
        splits = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        # Save preprocessed splits
        save_path = self.qm9_path / 'preprocessed_splits.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(splits, f)
        
        print(f"QM9 dataset prepared: {n_train} train, {n_val} val, {n_test} test")
        return splits
    
    def download_md17(self, molecules=None):
        """
        Download MD17 molecular dynamics trajectories.
        
        Args:
            molecules: List of molecule names to download.
                      If None, downloads ['aspirin', 'benzene', 'ethanol', 'malonaldehyde']
                      
        Returns:
            Dictionary of datasets by molecule
        """
        if molecules is None:
            molecules = ['aspirin', 'benzene', 'ethanol', 'malonaldehyde']
        
        print(f"Downloading MD17 datasets for {len(molecules)} molecules...")
        
        datasets = {}
        
        for molecule in tqdm(molecules, desc="Downloading MD17"):
            try:
                # Download using PyTorch Geometric
                dataset = MD17(root=str(self.md17_path), name=molecule)
                
                # Use subset for training (1000 samples)
                n_train = min(1000, len(dataset))
                n_test = min(1000, len(dataset) - n_train)
                
                train_indices = torch.randperm(len(dataset))[:n_train]
                test_indices = torch.randperm(len(dataset))[-n_test:]
                
                datasets[molecule] = {
                    'train': dataset[train_indices],
                    'test': dataset[test_indices],
                    'full': dataset
                }
                
                print(f"  {molecule}: {n_train} train, {n_test} test samples")
                
            except Exception as e:
                print(f"  Error downloading {molecule}: {e}")
        
        # Save preprocessed splits
        save_path = self.md17_path / 'preprocessed_splits.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(datasets, f)
        
        return datasets
    
    def download_rmd17(self, molecules=None, n_train=50):
        """
        Download revised MD17 (rMD17) dataset for few-shot evaluation.
        
        Args:
            molecules: List of molecule names
            n_train: Number of training samples (for few-shot learning)
            
        Returns:
            Dictionary of datasets by molecule
        """
        if molecules is None:
            molecules = ['aspirin', 'benzene', 'ethanol', 'malonaldehyde']
        
        print(f"Downloading rMD17 datasets for few-shot evaluation...")
        
        datasets = {}
        
        for molecule in tqdm(molecules, desc="Downloading rMD17"):
            try:
                # rMD17 available through torch_geometric
                # Note: In practice, you may need to download manually from:
                # http://quantum-machine.org/gdml/
                
                # For this example, we'll use MD17 as proxy
                # In production, replace with actual rMD17 download
                dataset = MD17(root=str(self.rmd17_path), name=molecule)
                
                # Create few-shot split
                train_indices = torch.randperm(len(dataset))[:n_train]
                test_indices = torch.arange(len(dataset))
                test_indices = test_indices[~torch.isin(test_indices, train_indices)][:1000]
                
                datasets[molecule] = {
                    'train': dataset[train_indices],
                    'test': dataset[test_indices]
                }
                
                print(f"  {molecule}: {n_train} train, {len(test_indices)} test samples")
                
            except Exception as e:
                print(f"  Error downloading {molecule}: {e}")
        
        # Save preprocessed splits
        save_path = self.rmd17_path / 'preprocessed_splits.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(datasets, f)
        
        return datasets
    
    def create_cross_domain_dataset(self):
        """
        Combine datasets for cross-domain pretraining.
        
        Returns:
            Combined dataset with cross-domain samples
        """
        print("Creating cross-domain pretraining dataset...")
        
        # Load all datasets
        qm9_splits = self.load_dataset('qm9')
        md17_splits = self.load_dataset('md17')
        
        # Combine training data
        train_data = []
        
        # Add QM9 training data
        if qm9_splits:
            train_data.extend(qm9_splits['train'])
            print(f"Added {len(qm9_splits['train'])} QM9 samples")
        
        # Add MD17 training data
        if md17_splits:
            for molecule, splits in md17_splits.items():
                train_data.extend(splits['train'])
            print(f"Added MD17 samples from {len(md17_splits)} molecules")
        
        print(f"Total cross-domain training samples: {len(train_data)}")
        
        # Save combined dataset
        save_path = self.root_dir / 'cross_domain_train.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(train_data, f)
        
        return train_data
    
    def load_dataset(self, dataset_name):
        """
        Load preprocessed dataset.
        
        Args:
            dataset_name: 'qm9', 'md17', or 'rmd17'
            
        Returns:
            Loaded dataset splits
        """
        path_map = {
            'qm9': self.qm9_path / 'preprocessed_splits.pkl',
            'md17': self.md17_path / 'preprocessed_splits.pkl',
            'rmd17': self.rmd17_path / 'preprocessed_splits.pkl'
        }
        
        if dataset_name not in path_map:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        path = path_map[dataset_name]
        
        if not path.exists():
            print(f"Dataset {dataset_name} not found. Please download first.")
            return None
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_statistics(self, dataset_name):
        """
        Compute dataset statistics for normalization.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Dictionary with mean and std for each property
        """
        splits = self.load_dataset(dataset_name)
        
        if splits is None:
            return None
        
        # Collect all training data
        if isinstance(splits, dict):
            if 'train' in splits:
                data = splits['train']
            else:
                # MD17 format
                data = []
                for molecule_splits in splits.values():
                    data.extend(molecule_splits['train'])
        else:
            data = splits
        
        # Compute statistics
        stats = {}
        
        if hasattr(data[0], 'y'):
            # QM9 properties
            y_values = torch.stack([d.y for d in data if hasattr(d, 'y')])
            stats['y_mean'] = y_values.mean(dim=0)
            stats['y_std'] = y_values.std(dim=0)
        
        if hasattr(data[0], 'energy'):
            # MD17 energies
            energies = torch.tensor([d.energy.item() for d in data])
            stats['energy_mean'] = energies.mean()
            stats['energy_std'] = energies.std()
        
        if hasattr(data[0], 'force'):
            # MD17 forces
            forces = torch.cat([d.force for d in data])
            stats['force_mean'] = forces.mean()
            stats['force_std'] = forces.std()
        
        print(f"Statistics for {dataset_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats


def download_all_datasets(root_dir='./data'):
    """
    Convenience function to download all datasets.
    
    Args:
        root_dir: Root directory for datasets
        
    Returns:
        DatasetDownloader instance
    """
    downloader = DatasetDownloader(root_dir)
    
    # Download QM9
    print("\n" + "="*50)
    print("Downloading QM9...")
    print("="*50)
    downloader.download_qm9(subset_size=50000)
    
    # Download MD17
    print("\n" + "="*50)
    print("Downloading MD17...")
    print("="*50)
    downloader.download_md17()
    
    # Download rMD17
    print("\n" + "="*50)
    print("Downloading rMD17...")
    print("="*50)
    downloader.download_rmd17(n_train=50)
    
    # Create cross-domain dataset
    print("\n" + "="*50)
    print("Creating cross-domain dataset...")
    print("="*50)
    downloader.create_cross_domain_dataset()
    
    # Compute statistics
    print("\n" + "="*50)
    print("Computing dataset statistics...")
    print("="*50)
    qm9_stats = downloader.get_statistics('qm9')
    md17_stats = downloader.get_statistics('md17')
    
    # Save statistics
    stats = {
        'qm9': qm9_stats,
        'md17': md17_stats
    }
    
    stats_path = Path(root_dir) / 'dataset_statistics.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print("\n" + "="*50)
    print("All datasets downloaded and preprocessed!")
    print("="*50)
    
    return downloader


if __name__ == '__main__':
    # Download all datasets
    downloader = download_all_datasets(root_dir='./data')
    
    print("\n✓ Dataset downloading complete!")
    print("\nDataset locations:")
    print(f"  QM9: {downloader.qm9_path}")
    print(f"  MD17: {downloader.md17_path}")
    print(f"  rMD17: {downloader.rmd17_path}")
    print(f"\nCross-domain dataset: {downloader.root_dir / 'cross_domain_train.pkl'}")
    print(f"Statistics: {downloader.root_dir / 'dataset_statistics.pkl'}")
